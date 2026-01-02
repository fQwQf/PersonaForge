"""
Socket.IO Manager for ScrollWeaver Evolution
支持多用户房间管理和实时通信
"""

import socketio
from typing import Dict, List, Optional, Any
from modules.core.sessions import SessionManager, SessionMode, BaseSession


class SocketIOManager:
    """
    Socket.IO 管理器
    处理 WebSocket 连接、房间管理和消息路由
    """
    
    def __init__(self):
        # 创建 Socket.IO 服务器
        # python-socketio 5.x 版本不再需要 async_mode 参数
        self.sio = socketio.AsyncServer(
            cors_allowed_origins="*"
        )
        self.app = None  # 将在 server.py 中设置
        self.session_manager = SessionManager()
        self.socket_to_user: Dict[str, int] = {}  # socket_id -> user_id
        self.socket_to_session: Dict[str, str] = {}  # socket_id -> session_id
        
    def set_app(self, app):
        """设置 FastAPI 应用"""
        self.app = app
        # 对于 FastAPI，需要将 Socket.IO 服务器挂载为 ASGI 应用
        # 注意：FastAPI 不支持直接 attach，需要手动添加路由
        # 这里先注册处理器，路由将在 server.py 中手动添加
        self._register_handlers()
    
    def _register_handlers(self):
        """注册 Socket.IO 事件处理器"""
        
        @self.sio.event
        async def connect(sid, environ, auth):
            """客户端连接"""
            print(f"[SocketIO] Client connected: {sid}")
            # 从 auth 中获取用户信息
            if auth and 'token' in auth:
                # 验证token并获取用户信息
                from database import db
                user = db.verify_token(auth['token'])
                if user:
                    self.socket_to_user[sid] = user['id']
                    print(f"[SocketIO] User authenticated: {user['username']} (ID: {user['id']})")
                else:
                    print(f"[SocketIO] Invalid token for sid: {sid}")
                    return False
            elif auth and 'user_id' in auth:
                self.socket_to_user[sid] = auth['user_id']
            else:
                print(f"[SocketIO] No authentication provided for sid: {sid}")
                return False
            return True
        
        @self.sio.event
        async def disconnect(sid):
            """客户端断开连接"""
            print(f"[SocketIO] Client disconnected: {sid}")
            # 清理会话
            if sid in self.socket_to_session:
                session_id = self.socket_to_session[sid]
                session = self.session_manager.get_session(session_id)
                if session:
                    session.remove_participant(self.socket_to_user.get(sid, 0))
                    # 如果房间为空，清理会话
                    if session.get_participant_count() == 0:
                        self.session_manager.remove_session(session_id)
                del self.socket_to_session[sid]
            
            if sid in self.socket_to_user:
                del self.socket_to_user[sid]
        
        @self.sio.event
        async def join_room(sid, data):
            """加入房间"""
            room_id = data.get('room_id')
            session_id = data.get('session_id')
            
            if not room_id or not session_id:
                return {'error': 'Missing room_id or session_id'}
            
            session = self.session_manager.get_session(session_id)
            if not session:
                return {'error': 'Session not found'}
            
            # 加入 Socket.IO 房间
            await self.sio.enter_room(sid, room_id)
            
            # 添加到会话参与者
            user_id = self.socket_to_user.get(sid, 0)
            username = data.get('username', f'user_{user_id}')
            session.add_participant(user_id, username, sid)
            self.socket_to_session[sid] = session_id
            
            # 通知房间内其他用户
            await self.sio.emit('user_joined', {
                'user_id': user_id,
                'username': username,
                'participant_count': session.get_participant_count()
            }, room=room_id, skip_sid=sid)
            
            return {
                'status': 'joined',
                'room_id': room_id,
                'session_id': session_id,
                'participant_count': session.get_participant_count()
            }
        
        @self.sio.event
        async def leave_room(sid, data):
            """离开房间"""
            room_id = data.get('room_id')
            session_id = data.get('session_id')
            
            if room_id:
                await self.sio.leave_room(sid, room_id)
            
            if session_id:
                session = self.session_manager.get_session(session_id)
                if session:
                    user_id = self.socket_to_user.get(sid, 0)
                    session.remove_participant(user_id)
                    
                    # 通知房间内其他用户
                    await self.sio.emit('user_left', {
                        'user_id': user_id,
                        'participant_count': session.get_participant_count()
                    }, room=session.room_id, skip_sid=sid)
            
            if sid in self.socket_to_session:
                del self.socket_to_session[sid]
            
            return {'status': 'left'}
        
        @self.sio.event
        async def create_session(sid, data):
            """创建新会话"""
            mode_str = data.get('mode', 'story')
            scroll_id = data.get('scroll_id')
            user_id = self.socket_to_user.get(sid, 0)
            room_id = data.get('room_id')
            
            try:
                mode = SessionMode(mode_str)
            except ValueError:
                return {'error': f'Invalid mode: {mode_str}'}
            
            # 创建会话
            session = self.session_manager.create_session(
                mode=mode,
                scroll_id=scroll_id,
                user_id=user_id,
                room_id=room_id,
                **{k: v for k, v in data.items() if k not in ['mode', 'scroll_id', 'room_id']}
            )
            
            # 初始化会话
            config = data.get('config', {})
            init_result = await session.initialize(config)
            
            return {
                'status': 'created',
                'session_id': session.session_id,
                'room_id': session.room_id,
                **init_result
            }
        
        @self.sio.event
        async def send_message(sid, data):
            """发送消息"""
            session_id = data.get('session_id')
            message = data.get('message', {})
            
            if not session_id:
                return {'error': 'Missing session_id'}
            
            session = self.session_manager.get_session(session_id)
            if not session:
                return {'error': 'Session not found'}
            
            user_id = self.socket_to_user.get(sid, 0)
            
            # 处理消息
            result = await session.process_message(message, user_id)
            
            # 广播到房间
            await self.sio.emit('message', {
                'session_id': session_id,
                'sender_id': user_id,
                'result': result
            }, room=session.room_id)
            
            return {'status': 'sent', 'result': result}
        
        @self.sio.event
        async def get_session_info(sid, data):
            """获取会话信息"""
            session_id = data.get('session_id')
            
            if not session_id:
                return {'error': 'Missing session_id'}
            
            session = self.session_manager.get_session(session_id)
            if not session:
                return {'error': 'Session not found'}
            
            return {
                'session': session.to_dict(),
                'participants': session.participants
            }
        
        @self.sio.event
        async def join_matching_room(sid, data):
            """加入匹配房间"""
            room_id = data.get('room_id')
            if not room_id:
                return {'error': 'Missing room_id'}
            
            await self.sio.enter_room(sid, f'matching_{room_id}')
            
            # 通知房间内其他用户
            await self.sio.emit('player_joined', {
                'room_id': room_id
            }, room=f'matching_{room_id}', skip_sid=sid)
            
            return {'status': 'joined'}
        
        @self.sio.event
        async def player_confirm(sid, data):
            """玩家确认"""
            room_id = data.get('room_id')
            if not room_id:
                return {'error': 'Missing room_id'}
            
            # 通知房间内其他用户
            await self.sio.emit('player_confirmed', {
                'room_id': room_id
            }, room=f'matching_{room_id}')
            
            return {'status': 'confirmed'}
        
        @self.sio.event
        async def start_game(sid, data):
            """开始游戏"""
            room_id = data.get('room_id')
            if not room_id:
                return {'error': 'Missing room_id'}
            
            # 通知房间内所有用户
            await self.sio.emit('room_started', {
                'room_id': room_id
            }, room=f'matching_{room_id}')
            
            return {'status': 'started'}
        
        @self.sio.event
        async def join_multiplayer_room(sid, data):
            """加入联机游戏房间"""
            room_id = data.get('room_id')
            if not room_id:
                return {'error': 'Missing room_id'}
            
            await self.sio.enter_room(sid, f'multiplayer_{room_id}')
            
            # 通知房间内其他用户
            await self.sio.emit('players_updated', {
                'room_id': room_id
            }, room=f'multiplayer_{room_id}', skip_sid=sid)
            
            return {'status': 'joined'}
        
        @self.sio.event
        async def role_selected(sid, data):
            """角色选择"""
            room_id = data.get('room_id')
            role_code = data.get('role_code')
            if not room_id:
                return {'error': 'Missing room_id'}
            
            user_id = self.socket_to_user.get(sid, 0)
            
            # 通知房间内其他用户
            await self.sio.emit('role_selected', {
                'room_id': room_id,
                'user_id': user_id,
                'role_code': role_code
            }, room=f'multiplayer_{room_id}', skip_sid=sid)
            
            return {'status': 'selected'}
    
    async def broadcast_to_room(self, room_id: str, event: str, data: Any):
        """向房间广播消息"""
        await self.sio.emit(event, data, room=room_id)
    
    async def send_to_user(self, socket_id: str, event: str, data: Any):
        """向特定用户发送消息"""
        await self.sio.emit(event, data, room=socket_id)

