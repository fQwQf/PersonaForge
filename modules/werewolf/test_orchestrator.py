"""
测试Performer和Orchestrator的交互逻辑
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from modules.werewolf.config_loader import ConfigLoader
from modules.werewolf.role_registry import RoleRegistry
from modules.werewolf.game_state import WerewolfGameState
from modules.werewolf.rule_engine import RuleEngine
from modules.werewolf.werewolf_performer import WerewolfPerformer
from modules.werewolf.werewolf_orchestrator import WerewolfOrchestrator

# Mock Performer for testing
class MockPerformer:
    def __init__(self, role_code, role_name):
        self.role_code = role_code
        self.role_name = role_name
        self.llm = self
        
    def chat(self, prompt):
        return '{"action_type": "skip", "reason": "mock_ai"}'

async def test_interaction():
    print("初始化测试环境...")
    config_loader = ConfigLoader()
    role_registry = RoleRegistry()
    role_registry.load_all_roles()
    config = config_loader.load_preset("simple_8")
    
    player_ids = [f"player_{i}" for i in range(8)]
    game_state = WerewolfGameState("test_game", config, role_registry, player_ids)
    game_state.assign_roles(seed=42)
    
    rule_engine = RuleEngine(config, role_registry)
    
    # 创建Performers
    performers = {}
    for pid in player_ids:
        role_id = game_state.get_player_role(pid)
        role_def = role_registry.get_role(role_id)
        # 使用Mock的原始Performer
        base_performer = MockPerformer(pid, f"Player {pid}")
        # 混合AI和人类
        is_human = (pid == "player_0") # 假设player_0是人类
        performers[pid] = WerewolfPerformer(base_performer, role_def, is_human=is_human)
        
    orchestrator = WerewolfOrchestrator(None, game_state, rule_engine, performers)
    
    # Mock消息回调
    async def mock_callback(player_id, message):
        print(f"\n[发送给 {player_id}] 类型: {message['type']}")
        if message['type'] == 'action_request':
            print(f"  选项: {[opt['action_type'] for opt in message['data']['options']]}")
            # 模拟人类回复
            if player_id == "player_0":
                print("  >> 模拟人类输入...")
                # 模拟延迟
                await asyncio.sleep(1)
                orchestrator.handle_player_input(player_id, {
                    "action_type": "skip",
                    "reason": "human_skip"
                })
                
    orchestrator.set_message_callback(mock_callback)
    
    print("\n开始游戏流程测试...")
    # 只运行一小段
    asyncio.create_task(orchestrator.start_game())
    
    # 运行几秒钟看看
    await asyncio.sleep(5)
    await orchestrator.stop_game()
    print("\n测试结束")

if __name__ == "__main__":
    asyncio.run(test_interaction())
