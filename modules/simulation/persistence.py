"""
Persistence management for ScrollWeaver simulation.
"""

import os
from typing import Dict, Any, Literal
from modules.utils.file_utils import load_json_file, save_json_file, create_dir


class Persistence:
    """Manages saving and loading simulation state."""
    
    def __init__(self, experiment_name: str, role_llm_name: str, start_time: str, config: Dict):
        """
        Initialize Persistence.
        
        Args:
            experiment_name: Experiment name
            role_llm_name: Role LLM name
            start_time: Start time string
            config: Configuration dictionary
        """
        self.experiment_name = experiment_name
        self.role_llm_name = role_llm_name
        self.start_time = start_time
        self.config = config
        self.if_save: int = 0
    
    def set_if_save(self, if_save: int) -> None:
        """Set whether to save."""
        self.if_save = if_save
    
    def save_current_simulation(self,
                                stage: Literal["location", "goal", "action"],
                                current_round: int = 0,
                                sub_round: int = 0,
                                server_state: Dict = None,
                                history_manager=None,
                                performers: Dict = None,
                                orchestrator=None,
                                role_codes: list = None) -> None:
        """
        Save the current simulation progress.
        
        Args:
            stage: Stage of simulation
            current_round: Current round number
            sub_round: Sub-round number
            server_state: Server state dictionary
            history_manager: HistoryManager instance
            performers: Dictionary of performers
            orchestrator: Orchestrator instance
            role_codes: List of role codes
        """
        if not self.if_save:
            return
        
        save_dir = f"./experiment_saves/{self.experiment_name}/{self.role_llm_name}_{self.start_time}"
        create_dir(save_dir)
        location_setted, goal_setted = False, False
        if stage in ["location", "goal", "action"]:
            location_setted = True
        if stage in ["goal", "action"]:
            goal_setted = True
        
        meta_info = {
            "location_setted": location_setted,
            "goal_setted": goal_setted,
            "round": current_round,
            "sub_round": sub_round,
        }
        
        save_json_file(os.path.join(save_dir, "meta_info.json"), meta_info)
        name = self.experiment_name.split("/")[0]
        save_json_file(os.path.join(save_dir, f"{name}.json"), self.config)
        
        if server_state:
            filename = os.path.join(save_dir, "./server_info.json")
            save_json_file(filename, server_state)
        
        if history_manager:
            history_manager.save_to_file(save_dir)
        
        if performers and role_codes:
            for role_code in role_codes:
                if role_code in performers:
                    performers[role_code].save_to_file(save_dir)
        
        if orchestrator:
            orchestrator.save_to_file(save_dir)
    
    def continue_simulation_from_file(self,
                                     save_dir: str,
                                     server_instance,
                                     performers: Dict = None,
                                     orchestrator=None,
                                     history_manager=None,
                                     role_codes: list = None) -> Dict[str, Any]:
        """
        Restore the record of the last simulation.
        
        Args:
            save_dir: Save directory path
            server_instance: Server instance to restore state to
            performers: Dictionary of performers
            orchestrator: Orchestrator instance
            history_manager: HistoryManager instance
            role_codes: List of role codes
            
        Returns:
            Dictionary with meta information
        """
        if os.path.exists(save_dir):
            meta_info = load_json_file(os.path.join(save_dir, "./meta_info.json"))
            filename = os.path.join(save_dir, "./server_info.json")
            if os.path.exists(filename):
                states = load_json_file(filename)
                server_instance.__setstate__(states)
            
            if performers and role_codes:
                for role_code in role_codes:
                    if role_code in performers:
                        performers[role_code].load_from_file(save_dir)
            
            if orchestrator:
                orchestrator.load_from_file(save_dir)
            
            if history_manager:
                history_manager.load_from_file(save_dir)
                
                if history_manager.detailed_history and performers and role_codes:
                    for record in history_manager.detailed_history:
                        for code in record.get("group", []):
                            if code in role_codes and code in performers:
                                performers[code].record(record)
            
            # Re-initialize simulation modules after loading state
            if hasattr(server_instance, '_init_simulation_modules'):
                server_instance._init_simulation_modules()
        else:
            meta_info = {
                "location_setted": False,
                "goal_setted": False,
                "round": 0,
                "sub_round": 0,
            }
        return meta_info

