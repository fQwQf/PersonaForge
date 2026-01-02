"""
Location utility functions for ScrollWeaver.
"""

from typing import List, Dict


def find_roles_at_location(location_code: str, performers: Dict, role_codes: List[str], name: bool = False) -> List[str]:
    """
    Find roles at a specific location.
    
    Args:
        location_code: Location code
        performers: Dictionary of performers
        role_codes: List of role codes
        name: If True, return role names; if False, return role codes
        
    Returns:
        List of role codes or names
    """
    if name:
        return [performers[code].nickname for code in role_codes if performers[code].location_code == location_code]
    else:
        return [code for code in role_codes if performers[code].location_code == location_code]


def find_group(role_code: str, performers: Dict, role_codes: List[str]) -> List[str]:
    """
    Find group of roles at the same location as the given role.
    
    Args:
        role_code: Role code
        performers: Dictionary of performers
        role_codes: List of role codes
        
    Returns:
        List of role codes in the same location
    """
    return [code for code in role_codes if performers[code].location_code == performers[role_code].location_code]

