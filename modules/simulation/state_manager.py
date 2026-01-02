"""
State management for ScrollWeaver simulation.
"""

from typing import List, Dict, Any


class StateManager:
    """Manages game state information."""
    
    def __init__(self, performers: Dict, orchestrator, role_codes: List[str], language: str = "zh"):
        """
        Initialize StateManager.
        
        Args:
            performers: Dictionary of performers
            orchestrator: Orchestrator instance
            role_codes: List of role codes
            language: Language code
        """
        self.performers = performers
        self.orchestrator = orchestrator
        self.role_codes = role_codes
        self.language = language
    
    def get_status_text(self, group: List[str]) -> str:
        """Get status text for a group of roles."""
        return "\n".join([self.performers[role_code].status for role_code in group])
    
    def get_group_members_info_text(self, group: List[str], profile: bool = False, status: bool = False) -> str:
        """Get group members information as text."""
        roles_info_text = ""
        for i, role_code in enumerate(group):
            name = self.performers[role_code].role_name
            roles_info_text += f"{i+1}. {name}\n(role_code:{role_code})\n"
            if profile:
                profile_text = self.performers[role_code].role_profile
                roles_info_text += f"{profile_text}\n"
            if status:
                status_text = self.performers[role_code].status
                roles_info_text += f"{status_text}\n"
        return roles_info_text
    
    def get_group_members_info_dict(self, group: List[str]) -> Dict[str, Dict[str, str]]:
        """Get group members information as dictionary."""
        info = {
            role_code: {
                "nickname": self.performers[role_code].nickname,
                "profile": self.performers[role_code].role_profile
            }
            for role_code in group
        }
        return info
    
    def get_locations_info(self, detailed: bool = True) -> str:
        """Get locations information."""
        location_info_text = "---当前各角色位置---\n" if self.language == "zh" else "---Current Location of Roles---\n"
        if detailed:
            for i, location_code in enumerate(self.orchestrator.locations_info):
                location_name = self.orchestrator.find_location_name(location_code)
                description = self.orchestrator.locations_info[location_code]["description"]
                location_info_text += f"\n{i+1}. {location_name}\nlocation_code:{location_code}\n{description}\n\n"
                role_names = [f"{self.performers[code].role_name}({code})" for code in self.role_codes if self.performers[code].location_code == location_code]
                role_names = ", ".join(role_names)
                location_info_text += "目前在这里的角色有：" + role_names if self.language == "zh" else "Roles located here: " + role_names
        else:
            for i, location_code in enumerate(self.orchestrator.locations_info):
                location_name = self.orchestrator.find_location_name(location_code)
                role_names = [f"{self.performers[code].role_name}({code})" for code in self.role_codes if self.performers[code].location_code == location_code]
                if len(role_names) == 0:
                    continue
                role_names = ", ".join(role_names)
                location_info_text += f"【{location_name}】：" + role_names + "；"
        return location_info_text
    
    def get_location_info_text(self, location_code: str) -> str:
        """Get detailed information for a single location.
        
        Args:
            location_code: The code of the location
            
        Returns:
            Text describing the location with name, description, and current roles
        """
        if location_code not in self.orchestrator.locations_info:
            return ""
        
        location_name = self.orchestrator.find_location_name(location_code)
        location_data = self.orchestrator.locations_info[location_code]
        description = location_data.get("description", "")
        detail = location_data.get("detail", description)
        
        # 获取当前在这个地点的角色
        roles_at_location = [
            self.performers[code].role_name 
            for code in self.role_codes 
            if self.performers[code].location_code == location_code
        ]
        roles_text = ", ".join(roles_at_location) if roles_at_location else ("无人" if self.language == "zh" else "No one")
        
        if self.language == "zh":
            info_text = f"【{location_name}】\n{detail}\n目前在这里的角色：{roles_text}"
        else:
            info_text = f"【{location_name}】\n{detail}\nRoles currently here: {roles_text}"
        
        return info_text
    
    def find_group(self, role_code: str) -> List[str]:
        """Find group of roles at the same location."""
        return [code for code in self.role_codes if self.performers[code].location_code == self.performers[role_code].location_code]
    
    def find_roles_at_location(self, location_code: str, name: bool = False) -> List[str]:
        """Find roles at a specific location."""
        if name:
            return [self.performers[code].nickname for code in self.role_codes if self.performers[code].location_code == location_code]
        else:
            return [code for code in self.role_codes if self.performers[code].location_code == location_code]

