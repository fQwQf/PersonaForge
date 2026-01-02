"""
Role utility functions for ScrollWeaver.
"""

from typing import List, Union
from .file_utils import get_grandchild_folders


def name2code(roles: Union[str, List[str]], performers: dict, role_codes: List[str], language: str = "zh") -> Union[str, List[str]]:
    """
    Convert role name to role code.
    
    Args:
        roles: Role name(s) or code(s)
        performers: Dictionary of performers
        role_codes: List of role codes
        language: Language code
        
    Returns:
        Role code(s)
    """
    name_dic = {performers[code].role_name: code for code in role_codes}
    name_dic.update({performers[code].nickname: code for code in role_codes})
    
    if isinstance(roles, list):
        processed_roles = []
        for role in roles:
            if role in role_codes:
                processed_roles.append(role)
            elif role in name_dic:
                processed_roles.append(name_dic[role])
            elif "-" in role and role.split("-")[0] in name_dic:
                processed_roles.append(name_dic[role.split("-")[0]])
            elif role.replace("_", "·") in role_codes:
                processed_roles.append(role.replace("_", "·"))
            else:
                processed_roles.append(role)
        return processed_roles
    elif isinstance(roles, str):
        roles = roles.replace("\n", "").strip()
        # 如果输入为空字符串，返回None
        if not roles:
            return None
        if roles in role_codes:
            return roles
        elif roles in name_dic:
            return name_dic[roles]
        elif f"{roles}-{language}" in role_codes:
            return f"{roles}-{language}"
        elif "-" in roles and roles.split("-")[0] in name_dic:
            return name_dic[roles.split("-")[0]]
        elif roles.replace("_", "·") in role_codes:
            return roles.replace("_", "·")
    return roles


def check_role_code_availability(role_code: str, role_file_dir: str) -> bool:
    """Check if a role code is available in the role file directory."""
    for path in get_grandchild_folders(role_file_dir):
        if role_code in path:
            return True
    return False

