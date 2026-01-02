"""
Session Management Module for ScrollWeaver Evolution
支持三种模式的会话管理：P (私密晤谈), O-P (入卷同游), A-O-P (雅集博弈)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from enum import Enum
import json
import uuid
from datetime import datetime


class SessionMode(Enum):
    """会话模式枚举"""
    CHAT = "chat"  # P Mode: 私密晤谈
    STORY = "story"  # O-P Mode: 入卷同游
    GAME = "game"  # A-O-P Mode: 雅集博弈


class BaseSession(ABC):
    """
    会话基类
    所有会话类型的基础抽象类，定义了会话的通用接口
    """
    
    def __init__(
        self,
        session_id: str,
        scroll_id: Optional[int] = None,
        user_id: Optional[int] = None,
        mode: SessionMode = SessionMode.STORY,
        room_id: Optional[str] = None
    ):
        """
        初始化会话
        
        Args:
            session_id: 会话唯一标识符
            scroll_id: 关联的书卷ID
            user_id: 用户ID
            mode: 会话模式
            room_id: 房间ID（用于多用户）
        """
        self.session_id = session_id
        self.scroll_id = scroll_id
        self.user_id = user_id
        self.mode = mode
        self.room_id = room_id or session_id  # 默认使用 session_id 作为 room_id
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.is_active = True
        self.participants: List[Dict[str, Any]] = []  # 参与者列表
        
    def add_participant(self, user_id: int, username: str, socket_id: Optional[str] = None):
        """添加参与者"""
        participant = {
            "user_id": user_id,
            "username": username,
            "socket_id": socket_id,
            "joined_at": datetime.now().isoformat()
        }
        self.participants.append(participant)
        self.updated_at = datetime.now()
        
    def remove_participant(self, user_id: int):
        """移除参与者"""
        self.participants = [p for p in self.participants if p["user_id"] != user_id]
        self.updated_at = datetime.now()
        
    def get_participant_count(self) -> int:
        """获取参与者数量"""
        return len(self.participants)
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        初始化会话
        子类必须实现此方法
        
        Returns:
            初始化结果数据
        """
        pass
    
    @abstractmethod
    async def process_message(self, message: Dict[str, Any], sender_id: int) -> Dict[str, Any]:
        """
        处理消息
        子类必须实现此方法
        
        Args:
            message: 消息内容
            sender_id: 发送者ID
            
        Returns:
            处理结果
        """
        pass
    
    @abstractmethod
    async def cleanup(self):
        """清理资源"""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "session_id": self.session_id,
            "scroll_id": self.scroll_id,
            "user_id": self.user_id,
            "mode": self.mode.value,
            "room_id": self.room_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "is_active": self.is_active,
            "participant_count": self.get_participant_count()
        }


class ChatSession(BaseSession):
    """
    P Mode: 私密晤谈会话
    1v1 聊天模式，直接与单个角色对话
    """
    
    def __init__(self, session_id: str, scroll_id: Optional[int] = None, 
                 user_id: Optional[int] = None, role_code: Optional[str] = None,
                 user_name: str = "用户", user_identity: str = "", user_profile: str = ""):
        super().__init__(session_id, scroll_id, user_id, SessionMode.CHAT)
        self.role_code = role_code  # 对话的目标角色
        self.user_name = user_name  # 用户名
        self.user_identity = user_identity  # 用户身份
        self.user_profile = user_profile  # 用户简介
        self.chat_performer = None  # ChatPerformer 实例
        self.chat_history: List[Dict[str, Any]] = []
        
    async def initialize(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """初始化聊天会话"""
        from modules.chat.chat_performer import ChatPerformer
        
        if not self.role_code:
            raise ValueError("role_code 不能为空")
        if not self.scroll_id:
            raise ValueError("scroll_id 不能为空")
        
        # 创建 ChatPerformer 实例
        llm_name = config.get("llm_name", "gemini-3-flash-preview")
        user_name = config.get("user_name", self.user_name)
        user_identity = config.get("user_identity", self.user_identity)
        user_profile = config.get("user_profile", self.user_profile)
        
        self.chat_performer = ChatPerformer(
            role_code=self.role_code,
            scroll_id=self.scroll_id,
            llm_name=llm_name,
            user_name=user_name,
            user_identity=user_identity,
            user_profile=user_profile
        )
        
        return {
            "status": "initialized",
            "role_code": self.role_code,
            "session_id": self.session_id,
            "character_name": self.chat_performer.role_name,
            "character_nickname": self.chat_performer.char_name
        }
    
    async def process_message(self, message: Dict[str, Any], sender_id: int) -> Dict[str, Any]:
        """处理聊天消息"""
        if not self.chat_performer:
            raise ValueError("会话未初始化，请先调用 initialize()")
        
        user_text = message.get("text", "")
        if not user_text:
            raise ValueError("消息内容不能为空")
        
        # 调用 ChatPerformer 生成回复
        temperature = message.get("temperature", 0.8)
        character_response = self.chat_performer.generate_response(user_text, temperature=temperature)
        
        # 更新历史
        self.chat_history = self.chat_performer.get_chat_history()
        
        return {
            "type": "chat_response",
            "role_code": self.role_code,
            "character_name": self.chat_performer.role_name,
            "character_nickname": self.chat_performer.char_name,
            "message": character_response,
            "timestamp": datetime.now().isoformat()
        }
    
    async def cleanup(self):
        """清理聊天会话"""
        self.is_active = False
        if self.chat_performer:
            self.chat_performer.clear_history()
        self.chat_history.clear()


class StorySession(BaseSession):
    """
    O-P Mode: 入卷同游会话
    故事生成模式，Orchestrator 控制剧情，Performer 扮演角色
    """
    
    def __init__(self, session_id: str, scroll_id: Optional[int] = None, 
                 user_id: Optional[int] = None, room_id: Optional[str] = None):
        super().__init__(session_id, scroll_id, user_id, SessionMode.STORY, room_id)
        self.current_act: Optional[int] = None  # 当前幕数
        self.acts: List[Dict[str, Any]] = []  # 所有幕的信息
        self.scrollweaver_instance = None  # ScrollWeaver 实例
        
    async def initialize(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """初始化故事会话"""
        # TODO: 加载 ScrollWeaver 实例和预设
        return {
            "status": "initialized",
            "session_id": self.session_id,
            "current_act": self.current_act,
            "total_acts": len(self.acts)
        }
    
    async def process_message(self, message: Dict[str, Any], sender_id: int) -> Dict[str, Any]:
        """处理故事消息"""
        # TODO: 调用 Orchestrator 和 Performer 生成故事
        return {
            "type": "story_update",
            "act": self.current_act,
            "content": "故事内容（待实现）"
        }
    
    async def cleanup(self):
        """清理故事会话"""
        self.is_active = False
        if self.scrollweaver_instance:
            # TODO: 清理 ScrollWeaver 资源
            pass


class GameSession(BaseSession):
    """
    A-O-P Mode: 雅集博弈会话
    游戏模式，Administrator 控制规则，Orchestrator 渲染，Performer 扮演玩家
    """
    
    def __init__(self, session_id: str, scroll_id: Optional[int] = None, 
                 user_id: Optional[int] = None, room_id: Optional[str] = None,
                 game_type: str = "werewolf"):
        super().__init__(session_id, scroll_id, user_id, SessionMode.GAME, room_id)
        self.game_type = game_type  # 游戏类型：werewolf, undercover, etc.
        self.administrator = None  # Administrator 实例（纯代码逻辑）
        self.game_state: Dict[str, Any] = {}  # 游戏状态
        
    async def initialize(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """初始化游戏会话"""
        # TODO: 创建 Administrator 实例
        return {
            "status": "initialized",
            "session_id": self.session_id,
            "game_type": self.game_type,
            "game_state": self.game_state
        }
    
    async def process_message(self, message: Dict[str, Any], sender_id: int) -> Dict[str, Any]:
        """处理游戏消息"""
        action_type = message.get("action_type")
        
        # TODO: 调用 Administrator 处理游戏逻辑
        # Administrator -> Orchestrator -> Performer
        
        return {
            "type": "game_update",
            "game_state": self.game_state,
            "action": action_type
        }
    
    async def cleanup(self):
        """清理游戏会话"""
        self.is_active = False
        self.game_state.clear()


class SessionManager:
    """
    会话管理器
    管理所有活跃的会话，支持 Room 机制
    """
    
    def __init__(self):
        self.sessions: Dict[str, BaseSession] = {}  # session_id -> Session
        self.rooms: Dict[str, List[str]] = {}  # room_id -> [session_id, ...]
        self.user_sessions: Dict[int, List[str]] = {}  # user_id -> [session_id, ...]
        
    def create_session(
        self,
        mode: SessionMode,
        scroll_id: Optional[int] = None,
        user_id: Optional[int] = None,
        room_id: Optional[str] = None,
        **kwargs
    ) -> BaseSession:
        """创建新会话"""
        session_id = str(uuid.uuid4())
        
        if mode == SessionMode.CHAT:
            session = ChatSession(session_id, scroll_id, user_id, **kwargs)
        elif mode == SessionMode.STORY:
            session = StorySession(session_id, scroll_id, user_id, room_id, **kwargs)
        elif mode == SessionMode.GAME:
            session = GameSession(session_id, scroll_id, user_id, room_id, **kwargs)
        else:
            raise ValueError(f"Unknown session mode: {mode}")
        
        self.sessions[session_id] = session
        
        # 添加到房间
        if room_id:
            if room_id not in self.rooms:
                self.rooms[room_id] = []
            self.rooms[room_id].append(session_id)
        
        # 添加到用户会话列表
        if user_id:
            if user_id not in self.user_sessions:
                self.user_sessions[user_id] = []
            self.user_sessions[user_id].append(session_id)
        
        return session
    
    def get_session(self, session_id: str) -> Optional[BaseSession]:
        """获取会话"""
        return self.sessions.get(session_id)
    
    def get_room_sessions(self, room_id: str) -> List[BaseSession]:
        """获取房间内所有会话"""
        session_ids = self.rooms.get(room_id, [])
        return [self.sessions[sid] for sid in session_ids if sid in self.sessions]
    
    def remove_session(self, session_id: str):
        """移除会话"""
        session = self.sessions.get(session_id)
        if session:
            # 从房间移除
            if session.room_id in self.rooms:
                self.rooms[session.room_id] = [
                    sid for sid in self.rooms[session.room_id] if sid != session_id
                ]
                if not self.rooms[session.room_id]:
                    del self.rooms[session.room_id]
            
            # 从用户会话列表移除
            if session.user_id and session.user_id in self.user_sessions:
                self.user_sessions[session.user_id] = [
                    sid for sid in self.user_sessions[session.user_id] if sid != session_id
                ]
            
            # 清理会话资源
            import asyncio
            if asyncio.iscoroutinefunction(session.cleanup):
                asyncio.create_task(session.cleanup())
            else:
                session.cleanup()
            
            del self.sessions[session_id]
    
    def get_user_sessions(self, user_id: int) -> List[BaseSession]:
        """获取用户的所有会话"""
        session_ids = self.user_sessions.get(user_id, [])
        return [self.sessions[sid] for sid in session_ids if sid in self.sessions]

