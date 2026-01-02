"""
规则引擎模块

负责处理游戏规则的验证和执行，包括：
- 技能结算逻辑
- 阶段转换规则
- 行动合法性验证
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from .config_loader import GameConfig
from .role_registry import RoleRegistry
from .game_state import WerewolfGameState, PlayerStatus


class Rule(ABC):
    """规则基类"""
    
    @abstractmethod
    def validate(self, action: Dict[str, Any], state: WerewolfGameState) -> bool:
        """验证行动是否合法"""
        pass
    
    @abstractmethod
    def apply(self, action: Dict[str, Any], state: WerewolfGameState) -> Dict[str, Any]:
        """应用行动效果"""
        pass


class WerewolfKillRule(Rule):
    """狼人击杀规则"""
    
    def validate(self, action: Dict[str, Any], state: WerewolfGameState) -> bool:
        target = action.get('target')
        if not target or not state.is_alive(target):
            return False
        return True
    
    def apply(self, action: Dict[str, Any], state: WerewolfGameState) -> Dict[str, Any]:
        target = action['target']
        
        # 检查是否被守护
        if state.player_status.get(target) == PlayerStatus.PROTECTED:
            return {
                'success': False,
                'reason': 'protected',
                'message': f'{state.player_names[target]} 被守护，未死亡'
            }
        
        # 标记为待死亡（女巫可能救）
        # 记录到临时状态，供女巫阶段使用
        state.night_status['kill_target'] = target
        
        return {
            'success': True,
            'target': target,
            'action': 'kill',
            'message': f'{state.player_names[target]} 被狼人击杀'
        }


class SeerCheckRule(Rule):
    """预言家查验规则"""
    
    def validate(self, action: Dict[str, Any], state: WerewolfGameState) -> bool:
        player = action.get('player_id')
        target = action.get('target')
        
        # 检查是否已使用过（每晚只能查一次）
        if state.is_ability_used(player, 'check'):
            return False
        
        if not target or not state.is_alive(target):
            return False
        
        # 不能查验自己
        if target == player:
            return False
        
        return True
    
    def apply(self, action: Dict[str, Any], state: WerewolfGameState) -> Dict[str, Any]:
        player = action['player_id']
        target = action['target']
        
        # 获取目标角色
        target_role = state.get_player_role(target)
        is_werewolf = (target_role == 'werewolf')
        
        # 标记已使用
        state.mark_ability_used(player, 'check', {'target': target, 'result': is_werewolf})
        
        return {
            'success': True,
            'player_id': player,
            'target': target,
            'is_werewolf': is_werewolf,
            'message': f'{state.player_names[target]} 是{"狼人" if is_werewolf else "好人"}'
        }


class WitchAntidoteRule(Rule):
    """女巫解药规则"""
    
    def validate(self, action: Dict[str, Any], state: WerewolfGameState) -> bool:
        player = action.get('player_id')
        target = action.get('target')
        
        if not target:
            return False
        
        # 检查解药是否已使用
        if state.is_ability_used(player, 'antidote'):
            return False
        
        # 只能救被狼人杀的玩家
        kill_target = state.night_status.get('kill_target')
        if not kill_target or target != kill_target:
            return False
            
        # 标记为已救（从待死亡名单移除）
        # 注意：这里只是验证，真正的移除在apply中
        
        return True
    
    def apply(self, action: Dict[str, Any], state: WerewolfGameState) -> Dict[str, Any]:
        player = action['player_id']
        target = action['target']
        
        # 标记已使用
        state.mark_ability_used(player, 'antidote', {'target': target})
        
        # 移除死亡标记
        if state.night_status.get('kill_target') == target:
            state.night_status['kill_target'] = None
            state.night_status['saved_by_witch'] = True
        
        return {
            'success': True,
            'target': target,
            'action': 'save',
            'message': f'{state.player_names[target]} 被女巫救起'
        }


class WitchPoisonRule(Rule):
    """女巫毒药规则"""
    
    def validate(self, action: Dict[str, Any], state: WerewolfGameState) -> bool:
        player = action.get('player_id')
        target = action.get('target')
        
        # 检查毒药是否已使用
        if state.is_ability_used(player, 'poison'):
            return False
        
        # 不能毒自己
        if target == player:
            return False
        
        if not target or not state.is_alive(target):
            return False
        
        # 同一晚不能同时用两种药（如果已用解药）
        cannot_use_both = state.config.rules.get('cannot_use_both_potions_same_night', True)
        if cannot_use_both and state.is_ability_used(player, 'antidote'):
            return False
        
        return True
    
    def apply(self, action: Dict[str, Any], state: WerewolfGameState) -> Dict[str, Any]:
        player = action['player_id']
        target = action['target']
        
        # 标记已使用
        state.mark_ability_used(player, 'poison', {'target': target})
        
        return {
            'success': True,
            'target': target,
            'action': 'poison',
            'message': f'{state.player_names[target]} 被女巫毒杀'
        }


class HunterShootRule(Rule):
    """猎人开枪规则"""
    
    def validate(self, action: Dict[str, Any], state: WerewolfGameState) -> bool:
        player = action.get('player_id')
        target = action.get('target')
        death_reason = action.get('death_reason', '')
        
        # 检查猎人是否已死
        if state.is_alive(player):
            return False
        
        # 如果是被毒死，根据配置决定能否开枪
        if 'poison' in death_reason:
            can_shoot = state.config.rules.get('hunter_can_shoot_when_poisoned', False)
            if not can_shoot:
                return False
        
        if not target or not state.is_alive(target):
            return False
        
        return True
    
    def apply(self, action: Dict[str, Any], state: WerewolfGameState) -> Dict[str, Any]:
        target = action['target']
        
        # 猎人开枪击杀目标
        state.kill_player(target, reason='shot_by_hunter')
        
        return {
            'success': True,
            'target': target,
            'message': f'猎人开枪带走了 {state.player_names[target]}'
        }


class GuardProtectRule(Rule):
    """守卫守护规则"""
    
    def __init__(self):
        self.last_protected: Dict[str, Optional[str]] = {}  # player_id -> last_target
    
    def validate(self, action: Dict[str, Any], state: WerewolfGameState) -> bool:
        player = action.get('player_id')
        target = action.get('target')
        
        if not target or not state.is_alive(target):
            return False
        
        # 检查是否能连续守同一人
        cannot_protect_same = state.config.rules.get('consecutive_guard_same_player', False)
        if cannot_protect_same:
            last_target = self.last_protected.get(player)
            if last_target == target:
                return False
        
        return True
    
    def apply(self, action: Dict[str, Any], state: WerewolfGameState) -> Dict[str, Any]:
        player = action['player_id']
        target = action['target']
        
        # 记录本次守护目标
        self.last_protected[player] = target
        
        # 设置守护状态
        state.protect_player(target)
        
        return {
            'success': True,
            'target': target,
            'message': f'{state.player_names[target]} 受到守护'
        }


class RuleEngine:
    """规则引擎，负责游戏规则的验证和执行"""
    
    def __init__(self, config: GameConfig, role_registry: RoleRegistry):
        self.config = config
        self.role_registry = role_registry
        
        # 初始化规则
        self.rules: Dict[str, Rule] = {
            'werewolf_kill': WerewolfKillRule(),
            'seer_check': SeerCheckRule(),
            'witch_antidote': WitchAntidoteRule(),
            'witch_poison': WitchPoisonRule(),
            'hunter_shoot': HunterShootRule(),
            'guard_protect': GuardProtectRule(),
        }
    
    def validate_action(self, action: Dict[str, Any], state: WerewolfGameState) -> bool:
        """
        验证行动是否合法
        
        Args:
            action: 行动数据
            state: 游戏状态
            
        Returns:
            bool: 是否合法
        """
        action_type = action.get('action_type')
        rule = self.rules.get(action_type)
        
        if not rule:
            print(f"[RuleEngine] 未知行动类型: {action_type}")
            return False
        
        return rule.validate(action, state)
    
    def apply_action(self, action: Dict[str, Any], state: WerewolfGameState) -> Dict[str, Any]:
        """
        应用行动效果
        
        Args:
            action: 行动数据
            state: 游戏状态
            
        Returns:
            Dict[str, Any]: 执行结果
        """
        action_type = action.get('action_type')
        rule = self.rules.get(action_type)
        
        if not rule:
            return {'success': False, 'error': f'未知行动类型: {action_type}'}
        
        if not rule.validate(action, state):
            return {'success': False, 'error': '行动验证失败'}
        
        result = rule.apply(action, state)
        
        # 记录行动
        state.record_action({
            **action,
            'result': result,
            'timestamp': state.action_history[-1]['timestamp'] if state.action_history else None
        })
        
        return result
    
    def resolve_night_phase(self, phase_name: str, actions: List[Dict[str, Any]], 
                           state: WerewolfGameState) -> List[Dict[str, Any]]:
        """
        结算夜晚阶段的所有行动
        
        Args:
            phase_name: 阶段名称
            actions: 行动列表
            state: 游戏状态
            
        Returns:
            List[Dict[str, Any]]: 结算结果列表
        """
        results = []
        
        # 先处理守卫守护（如果有）
        for action in actions:
            if action.get('action_type') == 'guard_protect':
                result = self.apply_action(action, state)
                results.append(result)
        
        # 处理狼人击杀
        kill_target = None
        for action in actions:
            if action.get('action_type') == 'werewolf_kill':
                result = self.apply_action(action, state)
                if result.get('success'):
                    kill_target = result.get('target')
                results.append(result)
        
        # 处理预言家查验
        for action in actions:
            if action.get('action_type') == 'seer_check':
                result = self.apply_action(action, state)
                results.append(result)
        
        # 处理女巫行动
        saved_target = None
        for action in actions:
            if action.get('action_type') == 'witch_antidote':
                # 女巫看到击杀目标，决定是否救
                # 注意：此时 kill_target 可能是 None（因为分阶段调用），所以要从 state.night_status 获取
                current_kill_target = state.night_status.get('kill_target')
                if action.get('target') == current_kill_target:
                    result = self.apply_action(action, state)
                    if result.get('success'):
                        saved_target = current_kill_target
                    results.append(result)
            
            elif action.get('action_type') == 'witch_poison':
                result = self.apply_action(action, state)
                if result.get('success'):
                    poison_target = result.get('target')
                    state.kill_player(poison_target, reason='poisoned')
                results.append(result)
        
        # 最终结算死亡
        # 注意：这里不能直接结算狼人击杀，因为可能还没到女巫阶段
        # 只有在所有夜晚阶段结束后，或者在天亮时结算
        # 但为了简化，我们可以在这里结算，如果女巫救了，再复活？
        # 不，更好的方式是：狼人击杀只标记，不直接 kill。
        # 只有在天亮时（announce phase），检查 night_status['kill_target'] 是否还存在。
        
        # 但为了保持兼容性，如果当前是狼人阶段，且没有女巫（或女巫已死），可以直接结算？
        # 不，统一放到天亮结算比较安全。
        # 不过，如果这里不结算，state.is_alive(target) 还是 True。
        # 这符合逻辑：天亮前大家都是活的（或者说状态未知）。
        
        # 之前的逻辑是：
        # if kill_target and kill_target != saved_target:
        #     state.kill_player(kill_target, reason='killed_by_werewolf')
        
        # 现在改为：只在狼人阶段标记。
        # 如果是女巫阶段，且救了，则清除标记。
        
        # 那么什么时候真正 kill？
        # 我们可以在 Orchestrator 的 _handle_announce_phase 中处理。
        pass
        
        # 清除守护状态
        state.clear_protection()
        
        return results
    
    def check_win_condition(self, state: WerewolfGameState) -> Optional[str]:
        """
        检查胜利条件
        
        Args:
            state: 游戏状态
            
        Returns:
            Optional[str]: 胜利方，None表示游戏继续
        """
        return state.check_win_condition()


if __name__ == "__main__":
    # 测试规则引擎
    from .config_loader import ConfigLoader
    from .role_registry import RoleRegistry
    from .game_state import WerewolfGameState
    
    # 加载配置
    config_loader = ConfigLoader()
    role_registry = RoleRegistry()
    role_registry.load_all_roles()
    config = config_loader.load_preset("standard_12")
    
    # 创建游戏
    player_ids = [f"player_{i}" for i in range(12)]
    game_state = WerewolfGameState(
        game_id="test_game",
        config=config,
        role_registry=role_registry,
        player_ids=player_ids
    )
    game_state.assign_roles(seed=42)
    game_state.start_game()
    
    # 初始化规则引擎
    rule_engine = RuleEngine(config, role_registry)
    
    # 模拟狼人击杀
    werewolves = game_state.get_players_by_role('werewolf')
    target = [p for p in player_ids if p not in werewolves][0]
    
    kill_action = {
        'action_type': 'werewolf_kill',
        'player_id': werewolves[0],
        'target': target,
        'phase': 'night_werewolf'
    }
    
    print("\n测试狼人击杀:")
    result = rule_engine.apply_action(kill_action, game_state)
    print(f"结果: {result}")
    
    # 模拟预言家查验
    seers = game_state.get_players_by_role('seer')
    if seers:
        check_action = {
            'action_type': 'seer_check',
            'player_id': seers[0],
            'target': werewolves[0],
            'phase': 'night_seer'
        }
        
        print("\n测试预言家查验:")
        result = rule_engine.apply_action(check_action, game_state)
        print(f"结果: {result}")
    
    print(f"\n存活玩家: {len(game_state.get_alive_players())}")
    print(f"死亡玩家: {len(game_state.get_dead_players())}")
