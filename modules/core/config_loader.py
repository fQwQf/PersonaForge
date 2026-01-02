"""
Configuration loader for ScrollWeaver.
"""

import os
from typing import Dict, Any
from modules.utils.file_utils import load_json_file


class ConfigLoader:
    """Loads and validates configuration."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        config = load_json_file(config_path)
        return config
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """
        Validate configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        required_keys = ['performer_codes', 'world_file_path', 'loc_file_path', 'experiment_subname']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")
        return True

