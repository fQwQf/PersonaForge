"""
狼人杀会话管理器

负责管理多个狼人杀游戏实例和WebSocket连接。
充当Server和WerewolfOrchestrator之间的桥梁。
"""

import asyncio
from typing import Dict, List, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect

from .config_loader import ConfigLoader, GameConfig
from .role_registry import RoleRegistry
from .game_state import WerewolfGameState
from .rule_engine import RuleEngine
from .werewolf_performer import WerewolfPerformer
from .werewolf_orchestrator import WerewolfOrchestrator
from .werewolf_performer import WerewolfPerformer
from .werewolf_orchestrator import WerewolfOrchestrator
from sw_utils import get_models
from .werewolf_config import WEREWOLF_LLM_NAME

# 轻量级LLM包装器，专为狼人杀设计
class SimpleLLMPerformer:
    """简化的LLM Performer，不依赖完整的ScrollWeaver Performer基础设施"""
    def __init__(self, role_code: str, role_name: str, llm_name: str = None):
        self.role_code = role_code
        self.role_name = role_name
        
        # 如果未指定，使用全局配置
        if llm_name is None:
            llm_name = WEREWOLF_LLM_NAME
        
        # 初始化LLM
        if llm_name != "none":
            try:
                self.llm = get_models(llm_name)
                print(f"[SimpleLLMPerformer] 已为 {role_code} 初始化 LLM: {llm_name}")
            except Exception as e:
                print(f"[SimpleLLMPerformer] LLM初始化失败: {e}")
                self.llm = None
        else:
            self.llm = None
        
    def chat(self, prompt: str) -> str:
        """调用LLM生成回复"""
        if self.llm is None:
            return ""
            
        try:
            # 使用LLM的chat方法
            if hasattr(self.llm, 'chat'):
                response = self.llm.chat(prompt)
            elif callable(self.llm):
                response = self.llm(prompt)
            else:
                # 降级到简单回复
                return "我需要更多时间思考这个问题。"
            
            # 确保返回字符串
            if response is None:
                return "我暂时没有明确的意见。"
            
            return response if isinstance(response, str) else str(response)
        except Exception as e:
            print(f"[SimpleLLMPerformer] LLM调用失败: {e}")
            import traceback
            traceback.print_exc()
            # 降级到简单回复
            return "我认为需要更仔细地分析当前局势。"


class WerewolfGameSession:
    """单个狼人杀游戏会话"""
    
    def __init__(self, game_id: str, config: GameConfig):
        self.game_id = game_id
        self.config = config
        
        # 初始化核心模块
        self.role_registry = RoleRegistry()
        self.role_registry.load_all_roles()
        
        # 创建玩家ID列表
        self.player_ids = [f"player_{i}" for i in range(config.total_players)]
        
        # 创建游戏状态
        self.game_state = WerewolfGameState(
            game_id=game_id,
            config=config,
            role_registry=self.role_registry,
            player_ids=self.player_ids
        )
        
        # 创建规则引擎
        self.rule_engine = RuleEngine(config, self.role_registry)
        
        # 创建Performers
        self.performers: Dict[str, WerewolfPerformer] = {}
        self.connections: Dict[str, WebSocket] = {}  # player_id -> WebSocket
        
        # 偏好角色
        self.preferred_roles: Dict[str, str] = {}
        
        # 初始化Orchestrator (稍后在add_player或start_game时完成)
        self.orchestrator: Optional[WerewolfOrchestrator] = None
        
    async def initialize(self):
        """初始化游戏组件"""
        # 预分配角色（为了确定谁是人类玩家，这里简化处理）
        # 实际逻辑：
        # 1. 创建房间
        # 2. 玩家加入（分配player_id）
        # 3. 房主开始游戏 -> 分配角色 -> 创建Performer
        
        # 这里简化：假设所有连接的WebSocket都是人类玩家，未连接的是AI
        pass

    async def connect_player(self, player_id: str, websocket: WebSocket):
        """玩家连接"""
        await websocket.accept()
        self.connections[player_id] = websocket
        print(f"[Session] 玩家 {player_id} 已连接到游戏 {self.game_id}")
        
    def disconnect_player(self, player_id: str):
        """玩家断开连接"""
        if player_id in self.connections:
            del self.connections[player_id]
            print(f"[Session] 玩家 {player_id} 断开连接")

    async def start_game(self):
        """开始游戏"""
        # 1. 分配角色
        self.game_state.assign_roles(preferred_roles=self.preferred_roles)
        
        # 2. 创建Performers
        for pid in self.player_ids:
            role_id = self.game_state.get_player_role(pid)
            role_def = self.role_registry.get_role(role_id)
            
            # 判断是否为人类（有WebSocket连接的视为人类）
            is_human = pid in self.connections
            
            # 创建基础Performer
            # AI玩家使用真实LLM，人类玩家不需要LLM
            if is_human:
                # 人类玩家不需要LLM
                base_performer = SimpleLLMPerformer(pid, f"玩家 {pid}", llm_name="none")
            else:
                # AI玩家使用真实LLM（从config.json读取）
                base_performer = SimpleLLMPerformer(pid, f"AI玩家 {pid}")
            
            self.performers[pid] = WerewolfPerformer(
                performer=base_performer,
                role_def=role_def,
                is_human=is_human
            )
            
        # 3. 创建Orchestrator
        self.orchestrator = WerewolfOrchestrator(
            orchestrator=None, # 暂时不需要基础Orchestrator
            game_state=self.game_state,
            rule_engine=self.rule_engine,
            performers=self.performers
        )
        
        # 设置消息回调
        self.orchestrator.set_message_callback(self._send_message)
        
        # 4. 启动游戏循环
        await self.orchestrator.start_game()
        print(f"[Session] 游戏 {self.game_id} 已启动")

    async def handle_message(self, player_id: str, message: dict):
        """处理来自玩家的消息"""
        if not self.orchestrator:
            return
            
        # 将消息转发给Orchestrator
        # 主要是处理行动响应
        self.orchestrator.handle_player_input(player_id, message)

    async def _send_message(self, player_id: str, message: dict):
        """发送消息给玩家"""
        if player_id in self.connections:
            try:
                await self.connections[player_id].send_json(message)
            except Exception as e:
                print(f"[Session] 发送消息失败 {player_id}: {e}")
                self.disconnect_player(player_id)


class WerewolfSessionManager:
    """全局会话管理器"""
    
    def __init__(self):
        self.sessions: Dict[str, WerewolfGameSession] = {}
        self.config_loader = ConfigLoader()
        
    def create_game(self, preset_name: str = "standard_12", preferred_role: Optional[str] = None) -> str:
        """创建新游戏"""
        import uuid
        game_id = str(uuid.uuid4())[:8]
        
        config = self.config_loader.load_preset(preset_name)
        session = WerewolfGameSession(game_id, config)
        
        # 存储偏好角色（假设给player_0）
        if preferred_role:
            session.preferred_roles = {'player_0': preferred_role}
            
        self.sessions[game_id] = session
        
        print(f"[Manager] 创建游戏 {game_id} (配置: {preset_name}, 偏好: {preferred_role})")
        return game_id
        
    def get_session(self, game_id: str) -> Optional[WerewolfGameSession]:
        return self.sessions.get(game_id)
        
    def remove_session(self, game_id: str):
        if game_id in self.sessions:
            # TODO: 清理资源，停止Orchestrator
            del self.sessions[game_id]
