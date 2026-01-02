"""
Interaction handling for ScrollWeaver simulation.
"""

from typing import Dict, Any, List, Generator, Tuple, Optional
import uuid
from modules.utils.text_utils import conceal_thoughts
from modules.utils.role_utils import name2code
from modules.utils.file_utils import remove_list_elements


class InteractionHandler:
    """Handles various types of interactions."""
    
    def __init__(self, 
                 performers: Dict,
                 orchestrator,
                 state_manager,
                 record_manager,
                 history_manager,
                 event_manager,
                 current_status: Dict,
                 role_codes: List[str],
                 logger,
                 language: str = "zh"):
        """
        Initialize InteractionHandler.
        
        Args:
            performers: Dictionary of performers
            orchestrator: Orchestrator instance
            state_manager: StateManager instance
            record_manager: RecordManager instance
            history_manager: HistoryManager instance
            event_manager: EventManager instance
            current_status: Current status dictionary
            role_codes: List of role codes
            logger: Logger instance
            language: Language code
        """
        self.performers = performers
        self.orchestrator = orchestrator
        self.state_manager = state_manager
        self.record_manager = record_manager
        self.history_manager = history_manager
        self.event_manager = event_manager
        self.current_status = current_status
        self.role_codes = role_codes
        self.logger = logger
        self.language = language
    
    def implement_next_plan(self, role_code: str, group: List[str]) -> Generator[Tuple[str, str, str, str], None, str]:
        """
        Implement the next plan for a role.
        
        Args:
            role_code: Role code
            group: List of role codes in group
            
        Yields:
            Tuple of (message_type, role_code, detail, record_id)
            
        Returns:
            str: Information text
        """
        # 验证role_code有效性
        if not role_code or not role_code.strip() or role_code not in self.performers:
            error_msg = f"[InteractionHandler] 无效的role_code: '{role_code}'，跳过此操作"
            print(error_msg)
            self.logger.error(error_msg)
            return
        
        other_roles_info = self.state_manager.get_group_members_info_dict(group)
        plan = self.performers[role_code].plan(
            other_roles_info=other_roles_info,
            available_locations=self.orchestrator.locations,
            world_description=self.orchestrator.description,
            intervention=self.event_manager.event,
        )
        
        info_text = plan["detail"]
        if plan["target_role_codes"]:
            plan["target_role_codes"] = name2code(
                plan["target_role_codes"],
                self.performers,
                self.role_codes,
                self.language
            )
        
        record_id = str(uuid.uuid4())
        self.logger.info(f"-Action-\n{self.performers[role_code].role_name}: {info_text}")
        self.record_manager.record(
            role_code=role_code,
            detail=plan["detail"],
            actor_type='role',
            act_type="plan",
            actor=role_code,
            group=plan["target_role_codes"] + [role_code],
            plan=plan,
            record_id=record_id
        )
        yield ("role", role_code, info_text, record_id)
        
        if plan["interact_type"] == "single" and len(plan["target_role_codes"]) == 1 and plan["target_role_codes"][0] in group:
            yield from self.start_single_role_interaction(plan, record_id)
        elif plan["interact_type"] == "multi" and len(plan["target_role_codes"]) > 1 and set(plan["target_role_codes"]).issubset(set(group)):
            yield from self.start_multi_role_interaction(plan, record_id)
        elif plan["interact_type"] == "enviroment":
            yield from self.start_enviroment_interaction(plan, role_code, record_id)
        elif plan["interact_type"] == "npc" and plan["target_npc_name"]:
            yield from self.start_npc_interaction(plan, role_code, target_name=plan["target_npc_name"], record_id=record_id)
        return info_text
    
    def start_enviroment_interaction(self,
                                    plan: Dict[str, Any],
                                    role_code: str,
                                    record_id: str) -> Generator[Tuple[str, str, str, str], None, str]:
        """
        Handle environment interaction.
        
        Args:
            plan: Action plan
            role_code: Role code
            record_id: Record ID
            
        Yields:
            Tuple of (message_type, role_code, detail, record_id)
            
        Returns:
            str: Interaction result
        """
        if "action" not in plan:
            plan["action"] = ""
        self.current_status['group'] = [role_code]
        location_code = self.performers[role_code].location_code
        result = self.orchestrator.enviroment_interact(
            action_maker_name=self.performers[role_code].role_name,
            action=plan["action"],
            action_detail=conceal_thoughts(self.history_manager.search_record_detail(record_id)),
            location_code=location_code
        )
        env_record_id = str(uuid.uuid4())
        self.logger.info(f"(Enviroment):{result}")
        self.record_manager.record(
            role_code=role_code,
            detail=result,
            actor_type='world',
            act_type="enviroment",
            initiator=role_code,
            actor="world",
            group=[role_code],
            record_id=env_record_id
        )
        yield ("world", "", "(Enviroment):" + result, env_record_id)
        
        return conceal_thoughts(self.history_manager.search_record_detail(record_id)) + self.history_manager.search_record_detail(env_record_id)
    
    def start_npc_interaction(self,
                             plan: Dict[str, Any],
                             role_code: str,
                             target_name: str,
                             record_id: str,
                             max_rounds: int = 3) -> Generator[Tuple[str, str, str, str], None, str]:
        """
        Handle NPC interaction.
        
        Args:
            plan: Action plan
            role_code: Role code
            target_name: Target NPC name
            record_id: Record ID
            max_rounds: Maximum number of rounds
            
        Yields:
            Tuple of (message_type, role_code, detail, record_id)
            
        Returns:
            str: Interaction result
        """
        interaction = plan
        start_idx = len(self.history_manager)
        
        self.logger.info(f"----------NPC Interaction----------\n")
        self.current_status['group'] = [role_code, target_name]
        for round_num in range(max_rounds):
            npc_interaction = self.orchestrator.npc_interact(
                action_maker_name=self.performers[role_code].role_name,
                action_detail=self.history_manager.search_record_detail(record_id),
                location_name=self.performers[role_code].location_name,
                target_name=target_name
            )
            npc_detail = npc_interaction["detail"]
            
            npc_record_id = str(uuid.uuid4())
            self.logger.info(f"{target_name}: {npc_detail}")
            self.record_manager.record(
                role_code=role_code,
                detail=npc_detail,
                actor_type='world',
                act_type="npc",
                actor="world",
                group=[role_code],
                npc_name=target_name,
                record_id=npc_record_id
            )
            yield ("world", "", f"(NPC-{target_name}):" + npc_detail, npc_record_id)
            
            if npc_interaction["if_end_interaction"]:
                break
            
            interaction = self.performers[role_code].npc_interact(
                npc_name=target_name,
                npc_response=self.history_manager.search_record_detail(npc_record_id),
                history=self.history_manager.get_subsequent_history(start_idx=start_idx),
                intervention=self.event_manager.event
            )
            detail = interaction["detail"]
            
            record_id = str(uuid.uuid4())
            self.logger.info(f"{self.performers[role_code].role_name}: {detail}")
            self.record_manager.record(
                role_code=role_code,
                detail=detail,
                actor_type='role',
                act_type="npc",
                actor=role_code,
                group=[role_code],
                npc_name=target_name,
                record_id=record_id
            )
            yield ("role", role_code, detail, record_id)
            
            if interaction["if_end_interaction"]:
                break
            if_end, epilogue = self.orchestrator.judge_if_ended("\n".join(self.history_manager.get_subsequent_history(start_idx)))
            if if_end:
                break
        
        return "\n".join(self.history_manager.get_subsequent_history(start_idx=start_idx))
    
    def start_single_role_interaction(self,
                                     plan: Dict[str, Any],
                                     record_id: str,
                                     max_rounds: int = 8) -> Generator[Tuple[str, str, str, str], None, None]:
        """
        Handle single role interaction.
        
        Args:
            plan: Action plan
            record_id: Record ID
            max_rounds: Maximum number of rounds
            
        Yields:
            Tuple of (message_type, role_code, detail, record_id)
        """
        interaction = plan
        acted_role_code = interaction["role_code"]
        acting_role_code = interaction["target_role_codes"][0]
        if acting_role_code not in self.role_codes:
            print(f"Warning: Role {acting_role_code} does not exist.")
            return
        self.current_status['group'] = [acted_role_code, acting_role_code]
        
        start_idx = len(self.history_manager)
        for round_num in range(max_rounds):
            interaction = self.performers[acting_role_code].single_role_interact(
                action_maker_code=acted_role_code,
                action_maker_name=self.performers[acted_role_code].role_name,
                action_detail=conceal_thoughts(self.history_manager.search_record_detail(record_id)),
                action_maker_profile=self.performers[acted_role_code].role_profile,
                intervention=self.event_manager.event
            )
            
            detail = interaction["detail"]
            
            record_id = str(uuid.uuid4())
            self.logger.info(f"{self.performers[acting_role_code].role_name}: {detail}")
            self.record_manager.record(
                role_code=acting_role_code,
                detail=detail,
                actor_type='role',
                act_type="single",
                group=[acted_role_code, acting_role_code],
                target_role_code=acting_role_code,
                planning_role_code=plan["role_code"],
                round=round_num,
                record_id=record_id
            )
            yield ("role", acting_role_code, detail, record_id)
            
            if interaction["if_end_interaction"]:
                return
            if interaction["extra_interact_type"] == "npc":
                print("---Extra NPC Interact---")
                result = yield from self.start_npc_interaction(
                    plan=interaction,
                    role_code=acted_role_code,
                    target_name=interaction["target_npc_name"],
                    record_id=record_id
                )
                interaction["detail"] = result
            elif interaction["extra_interact_type"] == "enviroment":
                print("---Extra Env Interact---")
                result = yield from self.start_enviroment_interaction(
                    plan=interaction,
                    role_code=acted_role_code,
                    record_id=record_id
                )
                interaction["detail"] = result
            
            if_end, epilogue = self.orchestrator.judge_if_ended("\n".join(self.history_manager.get_subsequent_history(start_idx)))
            if if_end:
                break
            acted_role_code, acting_role_code = acting_role_code, acted_role_code
        return
    
    def start_multi_role_interaction(self,
                                    plan: Dict[str, Any],
                                    record_id: str,
                                    max_rounds: int = 8) -> Generator[Tuple[str, str, str, str], None, None]:
        """
        Handle multi-role interaction.
        
        Args:
            plan: Action plan
            record_id: Record ID
            max_rounds: Maximum number of rounds
            
        Yields:
            Tuple of (message_type, role_code, detail, record_id)
        """
        interaction = plan
        acted_role_code = interaction["role_code"]
        group = interaction["target_role_codes"].copy()
        group.append(acted_role_code)
        
        for code in group:
            if code not in self.role_codes:
                print(f"Warning: Role {code} does not exist.")
                return
        self.current_status['group'] = group
        
        start_idx = len(self.history_manager)
        other_roles_info = self.state_manager.get_group_members_info_dict(group)
        
        # 记录最近发言的角色名称
        recent_speakers = []
        
        for round_num in range(max_rounds):
            acting_role_code = name2code(
                self.orchestrator.decide_next_actor(
                    history_text="\n".join(self.history_manager.get_recent_history(3)),
                    roles_info_text=self.state_manager.get_group_members_info_text(
                        remove_list_elements(group, acted_role_code),
                        status=True
                    ),
                    recent_speakers=recent_speakers[-2:] if recent_speakers else [] # 只传递最近2个发言者
                ),
                self.performers,
                self.role_codes,
                self.language
            )
            
            # 记录当前发言者
            if acting_role_code in self.performers:
                recent_speakers.append(self.performers[acting_role_code].role_name)
            
            interaction = self.performers[acting_role_code].multi_role_interact(
                action_maker_code=acted_role_code,
                action_maker_name=self.performers[acted_role_code].role_name,
                action_detail=conceal_thoughts(self.history_manager.search_record_detail(record_id)),
                action_maker_profile=self.performers[acted_role_code].role_profile,
                other_roles_info=other_roles_info,
                intervention=self.event_manager.event
            )
            
            detail = interaction["detail"]
            
            record_id = str(uuid.uuid4())
            self.logger.info(f"{self.performers[acting_role_code].role_name}: {detail}")
            self.record_manager.record(
                role_code=acting_role_code,
                detail=detail,
                actor_type='role',
                act_type="multi",
                group=group,
                actor=acting_role_code,
                planning_role_code=plan["role_code"],
                round=round_num,
                record_id=record_id
            )
            yield ("role", acting_role_code, detail, record_id)
            
            if interaction["if_end_interaction"]:
                break
            result = ""
            if interaction["extra_interact_type"] == "npc":
                print("---Extra NPC Interact---")
                result = yield from self.start_npc_interaction(
                    plan=interaction,
                    role_code=acting_role_code,
                    target_name=interaction["target_npc_name"],
                    record_id=record_id
                )
            elif interaction["extra_interact_type"] == "enviroment":
                print("---Extra Env Interact---")
                result = yield from self.start_enviroment_interaction(
                    plan=interaction,
                    role_code=acting_role_code,
                    record_id=record_id
                )
            interaction["detail"] = self.history_manager.search_record_detail(record_id) + result
            acted_role_code = acting_role_code
            if_end, epilogue = self.orchestrator.judge_if_ended("\n".join(self.history_manager.get_subsequent_history(start_idx)))
            if if_end:
                break
            
            return

