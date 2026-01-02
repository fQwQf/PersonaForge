"""
Simulation module for ScrollWeaver.

This module contains the core simulation logic, including:
- Simulator: Main simulation loop
- InteractionHandler: Handles various types of interactions
- EventManager: Manages events and scripts
- MovementManager: Manages character movement
- SceneManager: Manages scenes
- StateManager: Manages game state
- Persistence: Handles saving/loading
- RecordManager: Manages records
"""

from .simulator import Simulator
from .interaction_handler import InteractionHandler
from .event_manager import EventManager
from .movement_manager import MovementManager
from .scene_manager import SceneManager
from .state_manager import StateManager
from .persistence import Persistence
from .record_manager import RecordManager

__all__ = [
    'Simulator',
    'InteractionHandler',
    'EventManager',
    'MovementManager',
    'SceneManager',
    'StateManager',
    'Persistence',
    'RecordManager',
]

