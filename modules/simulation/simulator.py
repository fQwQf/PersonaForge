"""
Simulator for ScrollWeaver simulation.
"""

from typing import Dict, Any, List, Literal, Generator, Tuple, Optional
import uuid
import random
from modules.utils.role_utils import name2code


class Simulator:
    """Core simulation loop."""
    
    def __init__(self,
                 performers: Dict,
                 orchestrator,
                 state_manager,
                 record_manager,
                 interaction_handler,
                 event_manager,
                 movement_manager,
                 scene_manager,
                 persistence,
                 history_manager,
                 current_status: Dict,
                 role_codes: List[str],
                 logger,
                 language: str = "zh",
                 user_role_code: Optional[str] = None):
        """
        Initialize Simulator.
        
        Args:
            performers: Dictionary of performers
            orchestrator: Orchestrator instance
            state_manager: StateManager instance
            record_manager: RecordManager instance
            interaction_handler: InteractionHandler instance
            event_manager: EventManager instance
            movement_manager: MovementManager instance
            scene_manager: SceneManager instance
            persistence: Persistence instance
            history_manager: HistoryManager instance
            current_status: Current status dictionary
            role_codes: List of role codes
            logger: Logger instance
            language: Language code
            user_role_code: Optional user role code
        """
        self.performers = performers
        self.orchestrator = orchestrator
        self.state_manager = state_manager
        self.record_manager = record_manager
        self.interaction_handler = interaction_handler
        self.event_manager = event_manager
        self.movement_manager = movement_manager
        self.scene_manager = scene_manager
        self.persistence = persistence
        self.history_manager = history_manager
        self.current_status = current_status
        self.role_codes = role_codes
        self.logger = logger
        self.language = language
        self.user_role_code = user_role_code
        self.cur_round: int = 0
        self.mode: str = "free"
    
    def simulate_generator(self,
                          rounds: int = 10,
                          save_dir: str = "",
                          if_save: Literal[0, 1] = 0,
                          mode: Literal["free", "script"] = "free",
                          scene_mode: Literal[0, 1] = 1,
                          meta_info: Dict[str, Any] = None) -> Generator[Tuple[str, str, str, Any], None, None]:
        """
        Main simulation generator.
        
        Args:
            rounds: Maximum number of rounds
            save_dir: Save directory
            if_save: Whether to save
            mode: Simulation mode
            scene_mode: Scene mode
            meta_info: Meta information from continue_simulation_from_file (optional)
            
        Yields:
            Tuple of (message_type, role_code, text, record_id)
        """
        self.mode = mode
        self.persistence.set_if_save(if_save)
        
        # Use provided meta_info, or create default if not provided
        # Note: continue_simulation_from_file should be called BEFORE this method
        if meta_info is None:
            meta_info = {
                "location_setted": False,
                "goal_setted": False,
                "round": 0,
                "sub_round": 0,
            }
        
        start_round: int = meta_info["round"]
        sub_start_round: int = meta_info["sub_round"] if "sub_round" in meta_info else 0
        if start_round == rounds:
            return
            
        # Soul-Transmigration Prologue (First-time user entry) - Moved to be very first
        # Check if we should generate a prologue (e.g., if there's a primary player)
        if start_round == 0 and hasattr(self, 'user_role_code') and self.user_role_code and self.user_role_code in self.performers:
            print(f"[Simulator] 正在为角色 {self.user_role_code} 生成魂穿序章...")
            player = self.performers[self.user_role_code]
            other_roles_in_loc = [p.nickname for code, p in self.performers.items() 
                                 if code != self.user_role_code and p.location_code == player.location_code]
            other_roles_text = ", ".join(other_roles_in_loc) if other_roles_in_loc else "附近没有其他人。"
            
            prologue = self.orchestrator.generate_soul_trans_prologue(
                role_name=player.nickname,
                role_profile=player.role_profile,
                motivation=player.motivation,
                location_name=player.location_name,
                location_description=self.orchestrator.locations_info.get(player.location_code, {}).get('detail', ''),
                other_roles_text=other_roles_text
            )
            
            record_id = str(uuid.uuid4())
            self.record_manager.record(
                role_code=self.user_role_code,
                detail=prologue,
                actor_type="world",
                act_type="prologue",
                actor="world",
                group=[self.user_role_code],
                record_id=record_id
            )
            yield ("world", "", prologue, record_id)
        
        # Setting Locations
        if not meta_info["location_setted"]:
            self.logger.info("========== Start Location Setting ==========")
            self._init_role_locations()
            if hasattr(self, '_server_instance'):
                self.persistence.save_current_simulation(
                    "location",
                    0,
                    0,
                    self._get_server_state(),
                    self.history_manager,
                    self.performers,
                    self.orchestrator,
                    self.role_codes
                )
        
        # Setting Goals
        if not meta_info["goal_setted"]:
            self.logger.info("========== Start Goal Setting ==========")
            
            if self.mode == "free":
                self.event_manager.get_event()
                self.logger.info(f"--------- Free Mode: Current Event ---------\n{self.event_manager.event}\n")
                yield ("system", "", f"--------- Current Event ---------\n{self.event_manager.event}\n", None)
                self.event_manager.add_event_to_history(self.event_manager.event)
            elif self.mode == "script":
                self.event_manager.get_script()
                self.logger.info(f"--------- Script Mode: Setted Script ---------\n{self.event_manager.script}\n")
                yield ("system", "", f"--------- Setted Script ---------\n{self.event_manager.script}\n", None)
                self.event_manager.add_event_to_history(self.event_manager.event)
            
            # 在当前事件之后显示"书卷加载中"提示
            loading_text = "书卷加载中..." if self.language == "zh" else "Loading scroll..."
            yield ("system", "", loading_text, None)
            
            if self.mode == "free":
                # Get all performers info for motivation setting
                all_performers_dict = self.state_manager.get_group_members_info_dict(self.role_codes)
                print(f"[Simulator] 开始为 {len(self.role_codes)} 个角色批量设置动机...")
                
                # 使用批量生成动机
                try:
                    from modules.utils.motivation_generator import MotivationGenerator
                    generator = MotivationGenerator(llm_name="gemini-2.5-flash")
                    
                    # 准备角色信息列表
                    characters_list = []
                    for role_code in self.role_codes:
                        performer = self.performers[role_code]
                        characters_list.append({
                            "role_name": performer.role_name,
                            "profile": performer.role_profile
                        })
                    
                    # 一次性批量生成所有角色的动机
                    motivations_dict = generator.generate_batch_motivations(
                        characters=characters_list,
                        world_description=self.orchestrator.description,
                        intervention=self.event_manager.event,
                        language=self.language
                    )
                    
                    # 为每个角色设置动机并记录（但不显示在对话页面）
                    for role_code in self.role_codes:
                        performer = self.performers[role_code]
                        role_name = performer.role_name
                        
                        if role_name in motivations_dict:
                            motivation = motivations_dict[role_name]
                            # 设置到 performer 对象
                            performer.motivation = motivation
                            
                            info_text = (f"{performer.nickname} 设立了动机: {motivation}"
                                       if self.language == "zh"
                                       else f"{performer.nickname} has set the motivation: {motivation}")
                            
                            record_id = str(uuid.uuid4())
                            self.logger.info(info_text)
                            # 记录到 record_manager，但不 yield 到前端（不显示在对话页面）
                            self.record_manager.record(
                                role_code=role_code,
                                detail=info_text,
                                actor=role_code,
                                group=[role_code],
                                actor_type='role',
                                act_type="goal setting",
                                record_id=record_id
                            )
                            print(f"[Simulator] 角色 {role_name} 动机设置完成（已记录但不显示）")
                            # 不 yield，这样就不会在对话页面显示
                        else:
                            print(f"[Simulator] 警告: 角色 {role_name} 的动机未在批量生成结果中找到")
                            # 使用默认动机
                            default_motivation = "追求个人目标和成长" if self.language == "zh" else "Pursue personal goals and growth"
                            performer.motivation = default_motivation
                            
                            info_text = (f"{performer.nickname} 设立了动机: {default_motivation}"
                                       if self.language == "zh"
                                       else f"{performer.nickname} has set the motivation: {default_motivation}")
                            
                            record_id = str(uuid.uuid4())
                            self.logger.info(info_text)
                            # 记录但不显示
                            self.record_manager.record(
                                role_code=role_code,
                                detail=info_text,
                                actor=role_code,
                                group=[role_code],
                                actor_type='role',
                                act_type="goal setting",
                                record_id=record_id
                            )
                            # 不 yield
                    
                    print(f"[Simulator] 批量动机设置完成，共设置 {len(motivations_dict)} 个角色的动机")
                    
                except Exception as e:
                    print(f"[Simulator] 批量生成动机失败，回退到逐个生成: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # 回退到逐个生成（备用方案）
                    for idx, role_code in enumerate(self.role_codes):
                        print(f"[Simulator] 正在为角色 {idx + 1}/{len(self.role_codes)} ({self.performers[role_code].role_name}) 设置动机...")
                        try:
                            motivation = self.performers[role_code].set_motivation(
                                world_description=self.orchestrator.description,
                                other_roles_info=all_performers_dict,
                                intervention=self.event_manager.event,
                                script=self.event_manager.script
                            )
                            info_text = (f"{self.performers[role_code].nickname} 设立了动机: {motivation}"
                                       if self.language == "zh"
                                       else f"{self.performers[role_code].nickname} has set the motivation: {motivation}")
                            
                            record_id = str(uuid.uuid4())
                            self.logger.info(info_text)
                            # 记录但不显示
                            self.record_manager.record(
                                role_code=role_code,
                                detail=info_text,
                                actor=role_code,
                                group=[role_code],
                                actor_type='role',
                                act_type="goal setting",
                                record_id=record_id
                            )
                            print(f"[Simulator] 角色 {self.performers[role_code].role_name} 动机设置完成（已记录但不显示）")
                            # 不 yield
                        except Exception as e2:
                            print(f"[Simulator] 角色 {self.performers[role_code].role_name} 动机设置失败: {e2}")
                            import traceback
                            traceback.print_exc()
                            # 继续处理下一个角色，而不是完全失败
                            nickname = self.performers[role_code].nickname if self.performers[role_code].nickname else self.performers[role_code].role_name
                            error_text = (f"{nickname} 动机设置失败: {str(e2)}"
                                        if self.language == "zh"
                                        else f"{nickname} failed to set motivation: {str(e2)}")
                            record_id = str(uuid.uuid4())
                            self.record_manager.record(
                                role_code=role_code,
                                detail=error_text,
                                actor=role_code,
                                group=[role_code],
                                actor_type='role',
                                act_type="goal setting",
                                record_id=record_id
                            )
                            # 错误信息也不显示
                print(f"[Simulator] 所有角色动机设置完成")
            
            if hasattr(self, '_server_instance'):
                self.persistence.save_current_simulation(
                    "goal",
                    0,
                    0,
                    self._get_server_state(),
                    self.history_manager,
                    self.performers,
                    self.orchestrator,
                    self.role_codes
                )
        
        yield ("system", "", "-- Simulation Started --", None)
        
        selected_role_codes = []
        
        # Simulating
        for current_round in range(start_round, rounds):
            self.cur_round = current_round
            self.record_manager.update_cur_round(current_round)
            if hasattr(self, '_server_instance'):
                self._server_instance.cur_round = current_round
            self.logger.info(f"========== Round {current_round+1} Started ==========")
            
            # 显示当前事件（每轮开始时）
            # 注意：事件在"Setting Goals"阶段已经添加到历史，这里只显示，不重复添加
            if self.event_manager.event and current_round >= 1:
                self.logger.info(f"--------- Current Event ---------\n{self.event_manager.event}\n")
                yield ("world", "", "-- Current Event --\n" + self.event_manager.event, None)
                # 不重复添加到历史，因为事件在"Setting Goals"阶段已经添加过了
                # 只有在事件更新时才需要添加到历史
            
            if len(self.movement_manager.moving_roles_info) == len(self.role_codes):
                self.movement_manager.settle_movement()
                continue
            
            # Characters in next scene
            if scene_mode:
                group = self.scene_manager.decide_scene_actors(
                    selected_role_codes,
                    self.movement_manager.moving_roles_info,
                    self.event_manager.event,
                    scene_mode=True
                )
                selected_role_codes += group
                if len(selected_role_codes) >= len(self.role_codes):
                    selected_role_codes = []
            else:
                group = self.role_codes
            
            self.current_status['group'] = group
            if group:
                self.current_status['location_code'] = self.performers[group[0]].location_code
            self.scene_manager.set_scene_characters(current_round, group)
            
            # 场景化更迭：进入新地点或新幕时的文学性描述
            if group and current_round >= start_round:
                location_code = self.performers[group[0]].location_code
                location_info_text = self.state_manager.get_location_info_text(location_code)
                history_text = "\n".join(self.history_manager.get_recent_history(5))
                
                prologue = self.orchestrator.generate_location_prologue(
                    location_code=location_code,
                    history_text=history_text,
                    event=self.event_manager.event,
                    location_info_text=location_info_text
                )
                
                record_id = str(uuid.uuid4())
                self.record_manager.record(
                    role_code="world",
                    detail=prologue,
                    actor_type="world",
                    act_type="prologue",
                    actor="world",
                    group=group,
                    record_id=record_id
                )
                yield ("world", "", prologue, record_id)
            
            start_idx = len(self.history_manager)
            
            sub_round = sub_start_round
            # 记录最近发言的角色名称
            recent_speakers = []
            
            for sub_round in range(sub_start_round, 3):
                if self.mode == "script":
                    instruction = self.event_manager.script_instruct(self.event_manager.progress)
                    for code in instruction:
                        if code == "progress":
                            progress_msg = ("剧本进度：" + instruction["progress"]
                                          if self.language == "zh"
                                          else "Current Stage:" + instruction["progress"])
                            self.logger.info(progress_msg)
                            self.event_manager.update_progress(instruction["progress"])
                        elif code in self.role_codes:
                            self.performers[code].goal = instruction[code]
                else:
                    for role_code in group:
                        self.performers[role_code].update_goal(
                            other_roles_status=self.state_manager.get_status_text(self.role_codes)
                        )
                
                for role_code in group:
                    current_role_code = role_code
                    if scene_mode:
                        next_actor = self.orchestrator.decide_next_actor(
                            "\n".join(self.history_manager.get_recent_history(3)),
                            self.state_manager.get_group_members_info_text(group, status=True),
                            self.event_manager.script,
                            recent_speakers=recent_speakers[-2:] if recent_speakers else [] # 只传递最近2个发言者
                        )
                        # 如果decide_next_actor返回None或空值，使用当前role_code
                        if next_actor and next_actor.strip():
                            current_role_code = name2code(
                                next_actor,
                                self.performers,
                                self.role_codes,
                                self.language
                            )
                            # 如果name2code转换失败或返回无效值，使用当前role_code
                            if not current_role_code or current_role_code not in self.role_codes:
                                print(f"[Simulator] decide_next_actor返回的角色代码无效: {next_actor}，使用当前角色: {role_code}")
                                current_role_code = role_code
                        else:
                            print(f"[Simulator] decide_next_actor返回None或空值，使用当前角色: {role_code}")
                            current_role_code = role_code
                    
                    # 记录当前发言者
                    if current_role_code in self.performers:
                        recent_speakers.append(self.performers[current_role_code].role_name)
                    
                    yield from self.interaction_handler.implement_next_plan(role_code=current_role_code, group=group)
                    if hasattr(self, '_server_instance'):
                        self.persistence.save_current_simulation(
                            "action",
                            current_round,
                            sub_round,
                            self._get_server_state(),
                            self.history_manager,
                            self.performers,
                            self.orchestrator,
                            self.role_codes
                        )
                
                if_end, epilogue = self.orchestrator.judge_if_ended(
                    "\n".join(self.history_manager.get_recent_history(len(self.history_manager) - start_idx))
                )
                if if_end:
                    record_id = str(uuid.uuid4())
                    self.logger.info("--Epilogue--: " + epilogue)
                    self.record_manager.record(
                        role_code="None",
                        detail=epilogue,
                        actor_type="world",
                        act_type="epilogue",
                        actor="world",
                        group=[],
                        record_id=record_id
                    )
                    yield ("world", "", epilogue, record_id)
                    
                    # 场景化更迭：将当前记录转换为故事流
                    current_scene_logs = self.history_manager.get_recent_history(len(self.history_manager) - start_idx)
                    if current_scene_logs:
                        story_segment = self.orchestrator.log2story(current_scene_logs)
                        story_record_id = str(uuid.uuid4())
                        yield ("story", "", story_segment, story_record_id)
                    break
            
            for role_code in group:
                yield from self.movement_manager.decide_whether_to_move(
                    role_code=role_code,
                    group=self.state_manager.find_group(role_code)
                )
                self.performers[role_code].update_status()
            
            self.movement_manager.settle_movement()
            self.event_manager.update_event(group)
            
            sub_start_round = 0
            if hasattr(self, '_server_instance'):
                self.persistence.save_current_simulation(
                    "action",
                    current_round + 1,
                    sub_round + 1,
                    self._get_server_state(),
                    self.history_manager,
                    self.performers,
                    self.orchestrator,
                    self.role_codes
                )
    
    def _init_role_locations(self, random_allocate: bool = True) -> None:
        """Initialize role locations."""
        init_locations_code = random.choices(self.orchestrator.locations, k=len(self.role_codes))
        for i, role_code in enumerate(self.role_codes):
            self.performers[role_code].set_location(
                init_locations_code[i],
                self.orchestrator.find_location_name(init_locations_code[i])
            )
            info_text = (f"{self.performers[role_code].nickname} 现在位于 {self.orchestrator.find_location_name(init_locations_code[i])}"
                        if self.language == "zh"
                        else f"{self.performers[role_code].nickname} is now located at {self.orchestrator.find_location_name(init_locations_code[i])}")
            self.logger.info(info_text)
    
    def _get_server_state(self) -> Dict[str, Any]:
        """Get server state for serialization."""
        if hasattr(self, '_server_instance'):
            return self._server_instance.__getstate__()
        return {}
    
    def __getstate__(self) -> Dict[str, Any]:
        """Get state for serialization."""
        states = {key: value for key, value in self.__dict__.items()
                  if isinstance(value, (str, int, list, dict, bool, type(None)))
                  and key not in ['performers', 'orchestrator', 'logger', 'state_manager',
                                  'record_manager', 'interaction_handler', 'event_manager',
                                  'movement_manager', 'scene_manager', 'persistence',
                                  'history_manager']}
        return states
    
    def __setstate__(self, states: Dict[str, Any]) -> None:
        """Set state from serialization."""
        self.__dict__.update(states)

