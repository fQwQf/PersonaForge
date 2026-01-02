"""
狼人杀命令行测试工具

提供一个简单的命令行界面来测试狼人杀游戏流程。
用于验证核心逻辑，无需前端即可测试。
"""

import sys
import os
import random
from pathlib import Path
from typing import Dict, List, Any, Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from modules.werewolf.config_loader import ConfigLoader
from modules.werewolf.role_registry import RoleRegistry
from modules.werewolf.game_state import WerewolfGameState
from modules.werewolf.rule_engine import RuleEngine


class WerewolfCLIGame:
    """命令行狼人杀游戏"""
    
    def __init__(self, preset_name: str = "standard_12"):
        """
        初始化游戏
        
        Args:
            preset_name: 预设配置名称
        """
        print("=" * 60)
        print("狼人杀游戏 - 命令行测试版")
        print("=" * 60)
        
        # 加载配置和角色
        self.config_loader = ConfigLoader()
        self.role_registry = RoleRegistry()
        self.role_registry.load_all_roles()
        
        self.config = self.config_loader.load_preset(preset_name)
        print(f"\n加载配置: {self.config.game_name}")
        print(f"玩家数: {self.config.total_players}")
        print(f"角色配置: {self.config.roles}")
        
        # 创建玩家
        self.player_ids = [f"player_{i+1}" for i in range(self.config.total_players)]
        
        # 创建游戏状态
        self.game_state = WerewolfGameState(
            game_id="cli_test_game",
            config=self.config,
            role_registry=self.role_registry,
            player_ids=self.player_ids
        )
        
        # 创建规则引擎
        self.rule_engine = RuleEngine(self.config, self.role_registry)
        
        # 给玩家起名字
        self.assign_player_names()
    
    def assign_player_names(self):
        """为玩家分配名字"""
        names = [
            "张三", "李四", "王五", "赵六", "孙七", "周八",
            "吴九", "郑十", "小明", "小红", "小华", "小李"
        ]
        for i, player_id in enumerate(self.player_ids):
            self.game_state.player_names[player_id] = names[i] if i < len(names) else f"玩家{i+1}"
    
    def start_game(self):
        """开始游戏"""
        print("\n" + "=" * 60)
        print("游戏开始！")
        print("=" * 60)
        
        # 分配角色
        self.game_state.assign_roles()
        self.game_state.start_game()
        
        # 显示角色分配（测试模式，实际游戏中不应显示）
        print("\n【上帝视角 - 角色分配】")
        for player_id in self.player_ids:
            role = self.game_state.get_player_role(player_id)
            role_def = self.role_registry.get_role(role)
            print(f"  {self.game_state.player_names[player_id]}: {role_def.role_name}")
    
    def simulate_night(self):
        """模拟夜晚阶段"""
        print("\n" + "-" * 60)
        print(f"第 {self.game_state.current_round} 晚 - 夜晚降临")
        print("-" * 60)
        
        actions = []
        
        # 1. 狼人行动
        print("\n【狼人阶段】")
        werewolves = self.game_state.get_players_by_role('werewolf')
        alive_werewolves = [w for w in werewolves if self.game_state.is_alive(w)]
        
        if alive_werewolves:
            # 随机选择一个非狼人击杀
            alive_non_werewolves = [
                p for p in self.game_state.get_alive_players()
                if p not in werewolves
            ]
            if alive_non_werewolves:
                target = random.choice(alive_non_werewolves)
                actions.append({
                    'action_type': 'werewolf_kill',
                    'player_id': alive_werewolves[0],
                    'target': target,
                    'phase': 'night_werewolf'
                })
                print(f"  狼人选择击杀: {self.game_state.player_names[target]}")
        
        # 2. 预言家行动
        print("\n【预言家阶段】")
        seers = self.game_state.get_players_by_role('seer')
        alive_seers = [s for s in seers if self.game_state.is_alive(s)]
        
        if alive_seers:
            seer = alive_seers[0]
            # 随机选择一个其他玩家查验
            other_players = [p for p in self.game_state.get_alive_players() if p != seer]
            if other_players:
                target = random.choice(other_players)
                actions.append({
                    'action_type': 'seer_check',
                    'player_id': seer,
                    'target': target,
                    'phase': 'night_seer'
                })
                print(f"  预言家查验: {self.game_state.player_names[target]}")
        
        # 3. 女巫行动
        print("\n【女巫阶段】")
        witches = self.game_state.get_players_by_role('witch')
        alive_witches = [w for w in witches if self.game_state.is_alive(w)]
        
        if alive_witches:
            witch = alive_witches[0]
            
            # 查看是否有人被杀
            kill_action = next((a for a in actions if a['action_type'] == 'werewolf_kill'), None)
            if kill_action:
                kill_target = kill_action['target']
                print(f"  女巫看到被杀的是: {self.game_state.player_names[kill_target]}")
                
                # 简单策略：首夜救人，之后随机决定
                if not self.game_state.is_ability_used(witch, 'antidote'):
                    if self.game_state.current_round == 1 or random.random() < 0.5:
                        actions.append({
                            'action_type': 'witch_antidote',
                            'player_id': witch,
                            'target': kill_target,
                            'phase': 'night_witch'
                        })
                        print(f"  女巫使用解药救: {self.game_state.player_names[kill_target]}")
        
        # 结算夜晚行动
        print("\n【夜晚结算】")
        results = self.rule_engine.resolve_night_phase('night', actions, self.game_state)
        
        for result in results:
            if result.get('message'):
                print(f"  {result['message']}")
    
    def simulate_day(self):
        """模拟白天阶段"""
        print("\n" + "-" * 60)
        print(f"第 {self.game_state.current_round} 天 - 天亮了")
        print("-" * 60)
        
        # 宣布死亡
        print("\n【昨夜死亡通告】")
        # 获取上一轮的死亡玩家
        recent_deaths = [
            action for action in self.game_state.action_history[-10:]
            if action.get('type') == 'death'
        ]
        
        if recent_deaths:
            for death in recent_deaths:
                player = death['player_id']
                reason = death.get('reason', 'unknown')
                print(f"  {self.game_state.player_names[player]} 已死亡 (原因: {reason})")
        else:
            print("  昨晚是平安夜")
        
        # 显示存活玩家
        alive = self.game_state.get_alive_players()
        print(f"\n当前存活玩家 ({len(alive)}人):")
        for player_id in alive:
            print(f"  - {self.game_state.player_names[player_id]}")
        
        # 简化投票（随机投一个人）
        print("\n【投票放逐】")
        if len(alive) > 2:
            exiled = random.choice(alive)
            print(f"  投票结果: {self.game_state.player_names[exiled]} 被放逐")
            
            # 显示被放逐者的角色
            role = self.game_state.get_player_role(exiled)
            role_def = self.role_registry.get_role(role)
            print(f"  {self.game_state.player_names[exiled]} 的身份是: {role_def.role_name}")
            
            self.game_state.kill_player(exiled, reason='voted_out')
    
    def check_game_end(self) -> bool:
        """检查游戏是否结束"""
        winner = self.rule_engine.check_win_condition(self.game_state)
        
        if winner:
            print("\n" + "=" * 60)
            print("游戏结束！")
            print("=" * 60)
            
            if winner == "werewolf":
                print("\n🐺 狼人阵营获胜！")
            else:
                print("\n👥 好人阵营获胜！")
            
            # 显示所有玩家身份
            print("\n【游戏结束 - 所有玩家身份】")
            for player_id in self.player_ids:
                role = self.game_state.get_player_role(player_id)
                role_def = self.role_registry.get_role(role)
                status = "存活" if self.game_state.is_alive(player_id) else "死亡"
                print(f"  {self.game_state.player_names[player_id]}: {role_def.role_name} ({status})")
            
            return True
        
        return False
    
    def run(self, max_rounds: int = 10):
        """
        运行游戏
        
        Args:
            max_rounds: 最大回合数
        """
        self.start_game()
        
        for round_num in range(max_rounds):
            # 模拟夜晚
            self.simulate_night()
            
            # 检查游戏是否结束
            if self.check_game_end():
                break
            
            # 模拟白天
            self.simulate_day()
            
            # 检查游戏是否结束
            if self.check_game_end():
                break
            
            # 暂停一下
            input("\n按 Enter 继续下一轮...")
        
        if not self.game_state.game_ended:
            print(f"\n达到最大回合数 ({max_rounds})，游戏结束")


def main():
    """主函数"""
    print("选择游戏配置:")
    print("1. 12人标准局 (4狼3神5民)")
    print("2. 8人简化局 (2狼2神4民)")
    
    choice = input("\n请选择 (1/2，默认1): ").strip() or "1"
    
    if choice == "2":
        preset = "simple_8"
    else:
        preset = "standard_12"
    
    # 创建并运行游戏
    game = WerewolfCLIGame(preset)
    game.run(max_rounds=10)


if __name__ == "__main__":
    main()
