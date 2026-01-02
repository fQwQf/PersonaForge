"""
Scene management for ScrollWeaver simulation.
"""

from typing import List, Dict, Any
from modules.utils.role_utils import name2code


class SceneManager:
    """Manages scenes and scene characters."""
    
    def __init__(self, performers: Dict, orchestrator, state_manager, history_manager, role_codes: List[str], language: str = "zh"):
        """
        Initialize SceneManager.
        
        Args:
            performers: Dictionary of performers
            orchestrator: Orchestrator instance
            state_manager: StateManager instance
            history_manager: HistoryManager instance
            role_codes: List of role codes
            language: Language code
        """
        self.performers = performers
        self.orchestrator = orchestrator
        self.state_manager = state_manager
        self.history_manager = history_manager
        self.role_codes = role_codes
        self.language = language
        self.scene_characters: Dict[str, List[str]] = {}
    
    def decide_scene_actors(self, 
                           selected_role_codes: List[str],
                           moving_roles_info: Dict[str, Any],
                           event: str,
                           scene_mode: bool = True) -> List[str]:
        """
        Decide which actors should be in the next scene.
        
        Args:
            selected_role_codes: Previously selected role codes
            moving_roles_info: Information about moving roles
            event: Current event
            scene_mode: Whether scene mode is enabled
            
        Returns:
            List of role codes for the scene
        """
        if scene_mode:
            group = name2code(
                self.orchestrator.decide_scene_actors(
                    self.state_manager.get_locations_info(False),
                    self.history_manager.get_recent_history(5),
                    event,
                    list(set(selected_role_codes + list(moving_roles_info.keys())))
                ),
                self.performers,
                self.role_codes,
                self.language
            )
        else:
            group = self.role_codes
        return group
    
    def set_scene_characters(self, round_num: int, group: List[str]) -> None:
        """Set scene characters for a round."""
        self.scene_characters[str(round_num)] = group
    
    def get_scene_characters(self, round_num: int) -> List[str]:
        """Get scene characters for a round."""
        return self.scene_characters.get(str(round_num), [])

