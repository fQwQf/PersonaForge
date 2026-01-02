"""
狼人杀游戏模块

这个模块提供完整的狼人杀游戏功能，基于ScrollWeaver的多智能体框架。
采用配置驱动的设计，支持灵活的角色配置、规则定制和游戏模式。
"""

__all__ = [
    'ConfigLoader',
    'GameConfig',
    'RoleRegistry',
    'RoleDefinition',
    'WerewolfGameState',
    'RuleEngine',
    'WerewolfPerformer',
    'WerewolfOrchestrator',
]

# 延迟导入，避免循环依赖
def __getattr__(name):
    if name == 'ConfigLoader':
        from .config_loader import ConfigLoader
        return ConfigLoader
    elif name == 'GameConfig':
        from .config_loader import GameConfig
        return GameConfig
    elif name == 'RoleRegistry':
        from .role_registry import RoleRegistry
        return RoleRegistry
    elif name == 'RoleDefinition':
        from .role_registry import RoleDefinition
        return RoleDefinition
    elif name == 'WerewolfGameState':
        from .game_state import WerewolfGameState
        return WerewolfGameState
    elif name == 'RuleEngine':
        from .rule_engine import RuleEngine
        return RuleEngine
    elif name == 'WerewolfPerformer':
        from .werewolf_performer import WerewolfPerformer
        return WerewolfPerformer
    elif name == 'WerewolfOrchestrator':
        from .werewolf_orchestrator import WerewolfOrchestrator
        return WerewolfOrchestrator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

