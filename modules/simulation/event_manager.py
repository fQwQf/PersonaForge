"""
Event management for ScrollWeaver simulation.
"""

from typing import Dict, Any, List
from modules.utils.text_utils import conceal_thoughts


class EventManager:
    """Manages events and scripts."""
    
    def __init__(self, performers: Dict, orchestrator, state_manager, history_manager, role_codes: List[str], language: str = "zh"):
        """
        Initialize EventManager.
        
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
        self.intervention: str = ""
        self.event: str = ""
        self.script: str = ""
        self.event_history: List[str] = []
        self.progress: str = "剧本刚刚开始，还什么都没有发生" if language == 'zh' else "The story has just begun, nothing happens yet."
    
    def get_event(self) -> str:
        """Get current event."""
        if self.intervention == "" and not self.script:
            roles_info_text = self.state_manager.get_group_members_info_text(self.role_codes, profile=True)
            status_text = self.state_manager.get_status_text(self.role_codes)
            event = self.orchestrator.generate_event(
                roles_info_text=roles_info_text,
                event=self.intervention,
                history_text=status_text
            )
            # 确保事件不为空
            if not event or event.strip() == "":
                event = "故事刚刚开始，角色们正在探索他们的世界。" if self.language == 'zh' else "The story has just begun, characters are exploring their world."
            self.intervention = event
            self.event = event
        elif self.intervention == "" and self.script:
            self.intervention = self.script
            self.event = self.script
        else:
            self.event = self.intervention
        # 确保返回的事件不为空
        if not self.event or self.event.strip() == "":
            self.event = "故事刚刚开始，角色们正在探索他们的世界。" if self.language == 'zh' else "The story has just begun, characters are exploring their world."
            self.intervention = self.event
        return self.event
    
    def get_script(self) -> str:
        """Get current script."""
        if self.script == "":
            roles_info_text = self.state_manager.get_group_members_info_text(self.role_codes, profile=True)
            status = "\n".join([self.performers[role_code].status for role_code in self.role_codes])
            script = self.orchestrator.generate_script(
                roles_info_text=roles_info_text,
                event=self.intervention,
                history_text=status
            )
            self.script = script
        return self.script
    
    def update_event(self, group: List[str], top_k: int = 1) -> None:
        """Update current event."""
        if self.intervention == "":
            self.event = ""
        else:
            status_text = self.state_manager.get_status_text(group)
            self.event = self.orchestrator.update_event(
                self.event,
                self.intervention,
                status_text,
                script=self.script
            )
    
    def script_instruct(self, last_progress: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Generate script instructions for roles.
        
        Args:
            last_progress: Last progress of the script
            top_k: Number of recent history items to consider
            
        Returns:
            Dictionary of instructions for each role
        """
        roles_info_text = self.state_manager.get_group_members_info_text(self.role_codes, status=True)
        history_text = self.history_manager.get_recent_history(top_k)
        
        instruction = self.orchestrator.get_script_instruction(
            roles_info_text=roles_info_text,
            event=self.event,
            history_text=history_text,
            script=self.script,
            last_progress=last_progress
        )
        
        return instruction
    
    def set_intervention(self, intervention: str) -> None:
        """Set intervention."""
        self.intervention = intervention
        self.event = intervention
    
    def set_script(self, script: str) -> None:
        """Set script."""
        self.script = script
    
    def add_event_to_history(self, event: str) -> None:
        """Add event to history."""
        self.event_history.append(event)
    
    def update_progress(self, progress: str) -> None:
        """Update progress."""
        self.progress = progress

