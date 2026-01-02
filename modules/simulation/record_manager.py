"""
Record management for ScrollWeaver simulation.
"""

from typing import List, Dict, Any, Optional


class RecordManager:
    """Manages records of simulation events."""
    
    def __init__(self, performers: Dict, history_manager, cur_round: int):
        """
        Initialize RecordManager.
        
        Args:
            performers: Dictionary of performers
            history_manager: HistoryManager instance
            cur_round: Current round number
        """
        self.performers = performers
        self.history_manager = history_manager
        self.cur_round = cur_round
    
    def record(self,
               role_code: str,
               detail: str,
               actor_type: str,
               act_type: str,
               group: List[str] = [],
               actor: str = "",
               record_id: Optional[str] = None,
               **kwargs):
        """
        Record an event.
        
        Args:
            role_code: Role code
            detail: Event detail
            actor_type: Type of actor
            act_type: Type of action
            group: List of role codes in group
            actor: Actor identifier
            record_id: Record ID
            **kwargs: Additional arguments
        """
        if act_type == "plan" and "plan" in kwargs:
            detail = f"{self.performers[role_code].nickname}: {detail}"
            interact_type = kwargs["plan"]["interact_type"]
            target = ", ".join(kwargs["plan"]["target_role_codes"])
            other_info = f"Interact type: {interact_type}, Target: {target}"
        elif act_type == "move" and "destination_code" in kwargs:
            destination = kwargs["destination_code"]
            other_info = f"Desitination:{destination}"
        elif act_type == "single":
            detail = f"{self.performers[role_code].nickname}: {detail}"
            target, planning_role, round_num = kwargs["target_role_code"], kwargs["planning_role_code"], kwargs["round"]
            other_info = f"Target: {target}, Planning Role: {planning_role}, Round: {round_num}"
        elif act_type == "multi":
            detail = f"{self.performers[role_code].nickname}: {detail}"
            planning_role, round_num = kwargs["planning_role_code"], kwargs["round"]
            other_info = f"Group member:{group}, Planning Role: {planning_role}, Round:{round_num},"
        elif act_type == "npc":
            name = kwargs["npc_name"]
            other_info = f"Target: {name}"
        elif act_type == "enviroment":
            other_info = ""
        else:
            other_info = ""
        
        record = {
            "cur_round": self.cur_round,
            "role_code": role_code,
            "detail": detail,
            "actor": actor,
            "group": group,  # visible group
            "actor_type": actor_type,
            "act_type": act_type,
            "other_info": other_info,
            "record_id": record_id
        }
        self.history_manager.add_record(record)
        for code in group:
            if code in self.performers:
                self.performers[code].record(record)
    
    def update_cur_round(self, cur_round: int):
        """Update current round number."""
        self.cur_round = cur_round

