"""
游戏状态管理器

负责维护狼人杀游戏的完整状态，包括：
- 玩家角色分配
- 存活状态
- 当前阶段
- 行动历史
- 技能使用状态
"""

import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

from .config_loader import GameConfig
from .role_registry import RoleRegistry, RoleDefinition
import random


class PlayerStatus(str, Enum):
    """玩家状态"""
    ALIVE = "alive"
    DEAD = "dead"
    PROTECTED = "protected"  # 被守护


class WerewolfGameState:
    """狼人杀游戏状态管理器"""
    
    def __init__(self, 
                 game_id: str,
                 config: GameConfig, 
                 role_registry: RoleRegistry,
                 player_ids: List[str]):
        """
        初始化游戏状态
        
        Args:
            game_id: 游戏唯一ID
            config: 游戏配置
            role_registry: 角色注册表
            player_ids: 玩家ID列表
        """
        self.game_id = game_id
        self.config = config
        self.role_registry = role_registry
        
        # 验证玩家数量
        if len(player_ids) != config.total_players:
            raise ValueError(
                f"玩家数量 ({len(player_ids)}) 与配置不符 ({config.total_players})"
            )
        
        self.player_ids = player_ids
        
        # 玩家状态
        self.player_roles: Dict[str, str] = {}  # player_id -> role_id
        self.player_status: Dict[str, PlayerStatus] = {}  # player_id -> status
        self.player_names: Dict[str, str] = {}  # player_id -> display_name
        
        # 游戏进程
        self.current_phase: str = ""
        self.current_round: int = 0
        self.phase_index: int = 0
        
        # 行动历史
        self.action_history: List[Dict[str, Any]] = []
        self.phase_actions: Dict[str, List[Dict[str, Any]]] = {}  # phase -> actions
        
        # 技能使用记录
        self.ability_used: Dict[str, Dict[str, Any]] = {}  # player_id -> {ability_id: data}
        
        # 游戏状态
        self.game_started: bool = False
        self.game_ended: bool = False
        self.winner: Optional[str] = None  # "werewolf" or "villager"
        
        # 初始化时间
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.ended_at: Optional[datetime] = None
        
        # 临时状态（用于夜晚阶段间传递信息，如被杀者）
        self.night_status: Dict[str, Any] = {}
    
    def assign_roles(self, seed: Optional[int] = None, preferred_roles: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        随机分配角色给玩家
        
        Args:
            seed: 随机数种子（用于测试）
            preferred_roles: 玩家ID到偏好角色ID的映射，例如 {'player_0': 'seer'}
            
        Returns:
            Dict[str, str]: 玩家ID到角色ID的映射
        """
        if seed is not None:
            random.seed(seed)
        
        # 构建角色列表
        all_roles = []
        for role_id, count in self.config.roles.items():
            all_roles.extend([role_id] * count)
            
        # 处理偏好角色
        assigned_roles = {}
        if preferred_roles:
            for pid, role_id in preferred_roles.items():
                if pid in self.player_ids and role_id in all_roles:
                    assigned_roles[pid] = role_id
                    all_roles.remove(role_id)  # 从可用角色池中移除
                    print(f"[GameState] 已分配偏好角色: {pid} -> {role_id}")
        
        # 随机打乱剩余角色
        random.shuffle(all_roles)
        
        # 分配剩余角色
        role_iter = iter(all_roles)
        for pid in self.player_ids:
            if pid not in assigned_roles:
                assigned_roles[pid] = next(role_iter)
        
        self.player_roles = assigned_roles
        
        # 初始化状态
        for pid in self.player_ids:
            self.player_status[pid] = PlayerStatus.ALIVE
            
            # 生成显示名称
            # 提取数字ID
            try:
                # 假设格式为 "player_X"
                idx = int(pid.split('_')[1])
                if idx == 0:
                    self.player_names[pid] = "玩家0（你）"
                else:
                    self.player_names[pid] = f"玩家{idx}"
            except:
                self.player_names[pid] = pid
            
        print(f"[GameState] 已为 {len(self.player_ids)} 名玩家分配角色")
        return self.player_roles
    
    def get_role_definition(self, role_id: str) -> Optional[RoleDefinition]:
        """获取角色定义"""
        return self.role_registry.get_role(role_id)
    
    def get_player_role(self, player_id: str) -> Optional[str]:
        """获取玩家的角色ID"""
        return self.player_roles.get(player_id)
    
    def get_players_by_role(self, role_id: str) -> List[str]:
        """获取拥有指定角色的所有玩家"""
        return [
            player_id for player_id, r_id in self.player_roles.items()
            if r_id == role_id
        ]
    
    def get_alive_players(self) -> List[str]:
        """获取所有存活的玩家"""
        return [
            player_id for player_id, status in self.player_status.items()
            if status == PlayerStatus.ALIVE
        ]
    
    def get_dead_players(self) -> List[str]:
        """获取所有死亡的玩家"""
        return [
            player_id for player_id, status in self.player_status.items()
            if status == PlayerStatus.DEAD
        ]
    
    def is_alive(self, player_id: str) -> bool:
        """检查玩家是否存活"""
        return self.player_status.get(player_id) == PlayerStatus.ALIVE
    
    def kill_player(self, player_id: str, reason: str = "killed") -> bool:
        """
        杀死玩家
        
        Args:
            player_id: 玩家ID
            reason: 死亡原因
            
        Returns:
            bool: 是否成功杀死（可能被守护）
        """
        if not self.is_alive(player_id):
            return False
        
        self.player_status[player_id] = PlayerStatus.DEAD
        
        # 记录死亡事件
        self.record_action({
            "type": "death",
            "player_id": player_id,
            "reason": reason,
            "round": self.current_round,
            "phase": self.current_phase
        })
        
        print(f"[GameState] {self.player_names[player_id]} 已死亡 (原因: {reason})")
        return True
    
    def protect_player(self, player_id: str) -> None:
        """守护玩家（仅本轮有效）"""
        if self.is_alive(player_id):
            self.player_status[player_id] = PlayerStatus.PROTECTED
            print(f"[GameState] {self.player_names[player_id]} 受到守护")
    
    def clear_protection(self) -> None:
        """清除所有守护状态"""
        for player_id in self.player_ids:
            if self.player_status[player_id] == PlayerStatus.PROTECTED:
                self.player_status[player_id] = PlayerStatus.ALIVE
    
    def update_phase(self, new_phase: str) -> None:
        """
        更新当前游戏阶段
        
        Args:
            new_phase: 新阶段名称
        """
        self.current_phase = new_phase
        
        # 如果是新的一轮（回到第一个阶段）
        if self.phase_index == 0:
            self.current_round += 1
            print(f"[GameState] 开始第 {self.current_round} 轮")
        
        print(f"[GameState] 进入阶段: {new_phase}")
    
    def next_phase(self) -> str:
        """
        切换到下一个阶段
        
        Returns:
            str: 下一个阶段名称
        """
        self.phase_index = (self.phase_index + 1) % len(self.config.phase_flow)
        next_phase = self.config.phase_flow[self.phase_index]
        self.update_phase(next_phase)
        return next_phase
    
    def record_action(self, action: Dict[str, Any]) -> None:
        """
        记录玩家行动
        
        Args:
            action: 行动数据
        """
        # 添加时间戳和ID
        action['timestamp'] = datetime.now().isoformat()
        action['action_id'] = str(uuid.uuid4())
        
        # 添加到历史
        self.action_history.append(action)
        
        # 按阶段分组
        phase = action.get('phase', self.current_phase)
        if phase not in self.phase_actions:
            self.phase_actions[phase] = []
        self.phase_actions[phase].append(action)
    
    def get_phase_actions(self, phase: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取指定阶段的所有行动
        
        Args:
            phase: 阶段名称，None表示当前阶段
            
        Returns:
            List[Dict[str, Any]]: 行动列表
        """
        if phase is None:
            phase = self.current_phase
        return self.phase_actions.get(phase, [])
    
    def mark_ability_used(self, player_id: str, ability_id: str, data: Any = None) -> None:
        """
        标记技能已使用
        
        Args:
            player_id: 玩家ID
            ability_id: 技能ID
            data: 附加数据
        """
        if player_id not in self.ability_used:
            self.ability_used[player_id] = {}
        
        self.ability_used[player_id][ability_id] = {
            'used_at_round': self.current_round,
            'used_at_phase': self.current_phase,
            'data': data
        }
    
    def is_ability_used(self, player_id: str, ability_id: str) -> bool:
        """检查技能是否已使用"""
        return (player_id in self.ability_used and 
                ability_id in self.ability_used[player_id])
    
    def get_visible_state(self, player_id: str) -> Dict[str, Any]:
        """
        获取玩家可见的游戏状态
        
        Args:
            player_id: 玩家ID
            
        Returns:
            Dict[str, Any]: 可见状态数据
        """
        # 基本信息（所有人可见）
        visible_state = {
            'game_id': self.game_id,
            'current_round': self.current_round,
            'current_phase': self.current_phase,
            'alive_players': self.get_alive_players(),
            'dead_players': self.get_dead_players(),
            'player_count': len(self.player_ids),
        }
        
        # 玩家自己的角色信息
        if player_id in self.player_roles:
            visible_state['my_role'] = self.player_roles[player_id]
            visible_state['my_status'] = self.player_status[player_id].value
            
            # 获取角色定义
            role_def = self.get_role_definition(self.player_roles[player_id])
            if role_def:
                visible_state['my_abilities'] = [
                    ab.dict() for ab in role_def.abilities
                ]
        
        # 如果是狼人，可以看到队友
        if player_id in self.player_roles:
            player_role = self.player_roles[player_id]
            if player_role == 'werewolf':
                visible_state['werewolf_team'] = self.get_players_by_role('werewolf')
        
        return visible_state
    
    def get_full_state(self) -> Dict[str, Any]:
        """获取完整游戏状态（仅用于调试和上帝视角）"""
        return {
            'game_id': self.game_id,
            'config': self.config.dict(),
            'player_roles': self.player_roles,
            'player_status': {k: v.value for k, v in self.player_status.items()},
            'current_phase': self.current_phase,
            'current_round': self.current_round,
            'action_history': self.action_history,
            'ability_used': self.ability_used,
            'game_started': self.game_started,
            'game_ended': self.game_ended,
            'winner': self.winner
        }
    
    def check_win_condition(self) -> Optional[str]:
        """
        检查游戏是否结束以及胜利方
        
        Returns:
            Optional[str]: "werewolf", "villager", 或 None（游戏继续）
        """
        alive_players = self.get_alive_players()
        
        # 统计存活的狼人和好人数量
        alive_werewolves = sum(
            1 for p in alive_players 
            if self.player_roles[p] == 'werewolf'
        )
        
        alive_villagers = sum(
            1 for p in alive_players 
            if self.player_roles[p] != 'werewolf'
        )
        
        # 胜利条件判定
        if alive_werewolves == 0:
            # 所有狼人死亡，好人胜
            self.winner = "villager"
            self.game_ended = True
            return "villager"
        
        if alive_werewolves >= alive_villagers:
            # 狼人数量 >= 好人数量，狼人胜
            self.winner = "werewolf"
            self.game_ended = True
            return "werewolf"
        
        return None
    
    def start_game(self) -> None:
        """开始游戏"""
        if self.game_started:
            raise RuntimeError("游戏已经开始")
        
        self.game_started = True
        self.started_at = datetime.now()
        self.current_round = 1
        self.phase_index = 0
        self.current_phase = self.config.phase_flow[0]
        
        print(f"[GameState] 游戏开始！")
    
    def end_game(self, winner: Optional[str] = None) -> None:
        """结束游戏"""
        self.game_ended = True
        self.ended_at = datetime.now()
        if winner:
            self.winner = winner
        
        print(f"[GameState] 游戏结束！胜利方: {self.winner or '平局'}")


if __name__ == "__main__":
    # 测试游戏状态管理器
    from .config_loader import ConfigLoader
    from .role_registry import RoleRegistry
    
    # 加载配置和角色
    config_loader = ConfigLoader()
    role_registry = RoleRegistry()
    role_registry.load_all_roles()
    
    config = config_loader.load_preset("standard_12")
    
    # 创建游戏
    player_ids = [f"player_{i}" for i in range(12)]
    game_state = WerewolfGameState(
        game_id="test_game_001",
        config=config,
        role_registry=role_registry,
        player_ids=player_ids
    )
    
    # 分配角色
    game_state.assign_roles(seed=42)
    
    # 开始游戏
    game_state.start_game()
    
    # 测试状态查询
    print("\n存活玩家:", len(game_state.get_alive_players()))
    print("狼人玩家:", game_state.get_players_by_role("werewolf"))
    
    # 测试可见状态
    werewolf_player = game_state.get_players_by_role("werewolf")[0]
    visible = game_state.get_visible_state(werewolf_player)
    print(f"\n狼人视角 ({werewolf_player}):")
    print(f"  我的角色: {visible['my_role']}")
    print(f"  狼人队友: {visible.get('werewolf_team', [])}")
    
    # 测试胜利条件
    print(f"\n当前胜利方: {game_state.check_win_condition()}")
