"""
Movement management for ScrollWeaver simulation.
"""

from typing import Dict, Any, List, Generator, Tuple
from modules.utils.text_utils import conceal_thoughts


class MovementManager:
    """Manages character movement."""
    
    def __init__(self, performers: Dict, orchestrator, state_manager, record_manager, logger, language: str = "zh"):
        """
        Initialize MovementManager.
        
        Args:
            performers: Dictionary of performers
            orchestrator: Orchestrator instance
            state_manager: StateManager instance
            record_manager: RecordManager instance
            logger: Logger instance
            language: Language code
        """
        self.performers = performers
        self.orchestrator = orchestrator
        self.state_manager = state_manager
        self.record_manager = record_manager
        self.logger = logger
        self.language = language
        self.moving_roles_info: Dict[str, Any] = {}
    
    def decide_whether_to_move(self, role_code: str, group: List[str]):
        """
        Decide whether a role should move.
        
        Args:
            role_code: Role code
            group: List of role codes in group
            
        Yields:
            Tuple of (message_type, role_code, detail, record_id)
            
        Returns:
            bool: Whether the role decided to move
        """
        if len(self.orchestrator.locations) <= 1:
            return
        
        if_move, move_detail, destination_code = self.performers[role_code].move(
            locations_info_text=self.state_manager.get_locations_info(),
            locations_info=self.orchestrator.locations_info
        )
        
        if if_move:
            self.logger.info(move_detail)
            print(f"角色选择移动。{self.performers[role_code].role_name}正在前往{self.orchestrator.find_location_name(destination_code)}" 
                  if self.language == "zh" 
                  else f"The role decides to move. {self.performers[role_code].role_name} is heading to {self.orchestrator.find_location_name(destination_code)}.")
            
            self.record_manager.record(
                role_code=role_code,
                detail=move_detail,
                actor_type='role',
                act_type="move",
                actor=role_code,
                group=[role_code],
                destinatiion_code=destination_code
            )
            yield ("role", role_code, move_detail, None)
            
            distance = self.orchestrator.get_distance(self.performers[role_code].location_code, destination_code)
            self.performers[role_code].set_location(location_code=None, location_name=None)
            self.moving_roles_info[role_code] = {
                "location_code": destination_code,
                "distance": distance
            }
    
    def settle_movement(self) -> None:
        """Settle movement for all moving roles."""
        for role_code in self.moving_roles_info.copy():
            if not self.moving_roles_info[role_code]["distance"]:
                location_code = self.moving_roles_info[role_code]["location_code"]
                self.performers[role_code].set_location(
                    location_code,
                    self.orchestrator.find_location_name(location_code)
                )
                arrival_msg = (f"{self.performers[role_code].role_name} 已到达 【{self.orchestrator.find_location_name(location_code)}】" 
                              if self.language == "zh" 
                              else f"{self.performers[role_code].role_name} has reached [{self.orchestrator.find_location_name(location_code)}]")
                self.logger.info(arrival_msg)
                del self.moving_roles_info[role_code]
            else:
                self.moving_roles_info[role_code]["distance"] -= 1

