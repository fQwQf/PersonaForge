"""
狼人杀指挥家适配器

负责管理狼人杀游戏的整体流程，充当"法官"角色。
核心功能：
1. 流程控制：推进游戏阶段（夜晚->白天->投票）
2. 行动调度：决定当前谁可以行动
3. 消息广播：向所有玩家或特定玩家发送消息
4. 交互管理：处理玩家（AI/Human）的行动请求
"""

import asyncio
import random
from typing import Dict, List, Any, Optional, Callable, Awaitable
from datetime import datetime
from collections import Counter

from ..orchestrator import Orchestrator
from .game_state import WerewolfGameState
from .rule_engine import RuleEngine
from .werewolf_performer import WerewolfPerformer
from .config_loader import GameConfig


class WerewolfOrchestrator:
    """
    狼人杀指挥家
    
    协调游戏流程，驱动Performer行动，使用RuleEngine结算。
    """
    
    def __init__(self, 
                 orchestrator: Orchestrator,
                 game_state: WerewolfGameState,
                 rule_engine: RuleEngine,
                 performers: Dict[str, WerewolfPerformer]):
        self.base_orchestrator = orchestrator
        self.game_state = game_state
        self.rule_engine = rule_engine
        self.performers = performers
        
        # 回调函数，用于发送消息到前端
        # 签名: async def send_message(player_id: str, message: dict)
        self.message_callback: Optional[Callable[[str, dict], Awaitable[None]]] = None
        
        # 游戏循环控制
        self.running = False
        self.current_task: Optional[asyncio.Task] = None
        
        # 等待玩家输入的Future
        # player_id -> Future
        self.pending_actions: Dict[str, asyncio.Future] = {}
        
        # 记录入夜时的存活玩家，用于计算昨晚死亡名单
        self.night_start_alive_players: set = set()

    def set_message_callback(self, callback: Callable[[str, dict], Awaitable[None]]):
        """设置消息发送回调"""
        self.message_callback = callback

    async def broadcast(self, message: dict, target_players: Optional[List[str]] = None):
        """广播消息"""
        if not self.message_callback:
            return
            
        if target_players is None:
            target_players = self.game_state.player_ids
            
        for pid in target_players:
            await self.message_callback(pid, message)

    async def start_game(self):
        """开始游戏循环"""
        if self.running:
            return
            
        self.running = True
        self.game_state.start_game()
        
        await self.broadcast({
            "type": "game_start",
            "data": {
                "game_id": self.game_state.game_id,
                "config": self.game_state.config.dict()
            }
        })
        
        # 向每个玩家单独发送他们的身份信息
        for pid in self.game_state.player_ids:
            role_id = self.game_state.get_player_role(pid)
            performer = self.performers.get(pid)
            if performer:
                role_def = performer.role_def
                await self.message_callback(pid, {
                    "type": "role_reveal",
                    "data": {
                        "role_id": role_id,
                        "role_name": role_def.role_name,
                        "description": role_def.description if hasattr(role_def, 'description') else '',
                        "camp": role_def.camp
                    }
                })
        
        # 发送初始玩家状态
        await self._sync_player_states()
        
        # 启动主循环
        self.current_task = asyncio.create_task(self._game_loop())

    async def stop_game(self):
        """停止游戏"""
        self.running = False
        if self.current_task:
            self.current_task.cancel()
            try:
                await self.current_task
            except asyncio.CancelledError:
                pass

    async def _game_loop(self):
        """游戏主循环"""
        try:
            while self.running and not self.game_state.game_ended:
                current_phase = self.game_state.current_phase
                
                await self.broadcast({
                    "type": "phase_change",
                    "data": {
                        "phase": current_phase,
                        "round": self.game_state.current_round
                    }
                })
                
                # 根据阶段执行逻辑
                if "night" in current_phase:
                    await self._handle_night_phase(current_phase)
                elif current_phase == "day_announce":
                    await self._handle_announce_phase()
                elif current_phase == "day_discussion":
                    await self._handle_discussion_phase()
                elif current_phase == "day_vote":
                    await self._handle_vote_phase()
                
                # 检查胜利条件
                winner = self.game_state.check_win_condition()
                if winner:
                    await self._handle_game_end(winner)
                    break
                    
                # 进入下一阶段
                next_phase = self.game_state.next_phase()
                
                # 稍微停顿，避免消息过快
                await asyncio.sleep(2)
                
        except Exception as e:
            print(f"[Orchestrator] 游戏循环异常: {e}")
            import traceback
            traceback.print_exc()

    async def _handle_night_phase(self, phase: str):
        """处理夜晚阶段（如狼人、预言家、女巫）"""
        # 如果是夜晚的第一个阶段（通常是狼人），记录当前存活状态
        if phase == "night_werewolf": # 假设这是每晚第一个阶段
             self.night_start_alive_players = set(self.game_state.get_alive_players())
             
        # 确定当前阶段的行动角色
        # 约定：阶段名格式为 night_{role_id}
        role_id = phase.replace("night_", "")
        
        # 获取需要行动的玩家
        eligible_players = [
            pid for pid in self.game_state.player_ids
            if self.game_state.get_player_role(pid) == role_id and self.game_state.is_alive(pid)
        ]
        
        if not eligible_players:
            # 无人行动，直接跳过（随机延迟以防暴露）
            await asyncio.sleep(random.uniform(3, 6))
            return
        
        # 女巫阶段需要特殊处理：提供击杀目标信息
        context_info = {}
        if phase == "night_witch":
            # 从临时状态获取击杀目标
            kill_target = self.game_state.night_status.get('kill_target')
            if kill_target and self.game_state.is_alive(kill_target):
                context_info['kill_target'] = kill_target
                context_info['kill_target_name'] = self.game_state.player_names.get(kill_target, kill_target)
        
        # 收集行动
        actions = []
        
        # 并行请求所有相关玩家行动
        # 对于狼人，如果是多人，可能需要特殊处理（如统一意见），这里简化为独立行动
        # 实际狼人杀中，狼人需要即时通讯。这里假设狼人通过"Action"来投票选择目标。
        
        action_tasks = []
        for pid in eligible_players:
            performer = self.performers[pid]
            # For witch, pass context info
            if context_info:
                action_tasks.append(self._request_action_with_context(performer, context_info))
            else:
                action_tasks.append(self._request_action(performer))
            
        # 等待所有行动完成（或超时）
        results = await asyncio.gather(*action_tasks, return_exceptions=True)
        
        # 收集行动
        actions = []
        for pid, result in zip(eligible_players, results):
            if isinstance(result, Exception):
                print(f"玩家 {pid} 行动异常: {result}")
                continue
            if result:
                # 注入必要的字段
                result['player_id'] = pid
                result['phase'] = phase
                actions.append(result)
        
        # 结算行动（注意：这里只是收集，真正的结算是所有夜晚行动结束后统一结算，还是分阶段结算？）
        # 根据RuleEngine的设计，resolve_night_phase是处理所有行动。
        # 但有些行动需要即时反馈（如预言家查验）。
        # 这里我们采用混合模式：
        # 1. 预言家查验：即时结算并反馈给预言家
        # 2. 狼人/女巫/守卫：记录行动，稍后统一结算
        
        # 实际上，RuleEngine.apply_action        # 结算行动
        # 对于狼人阶段，需要统计投票选出最终目标
        if role_id == "werewolf":
            # 收集所有狼人的击杀投票
            werewolf_votes = []
            for act in actions:
                if act.get('action_type') == 'werewolf_kill' and act.get('target'):
                    werewolf_votes.append(act.get('target'))
            
            if werewolf_votes:
                vote_counts = Counter(werewolf_votes)
                # 找出最高票数
                max_votes = max(vote_counts.values())
                # 找出所有获得最高票数的目标
                top_targets = [target for target, count in vote_counts.items() if count == max_votes]
                
                # 如果平票，随机选择一个
                final_target = random.choice(top_targets)
                
                print(f"[Orchestrator] 狼人投票: {vote_counts}, 最终击杀: {final_target}")
                
                # 构造统一的狼人击杀行动
                final_action = {
                    'player_id': 'werewolf_team',
                    'action_type': 'werewolf_kill',
                    'target': final_target,
                    'phase': phase,
                    'round': self.game_state.current_round
                }
                actions = [final_action]  # 替换为统一的击杀行动
        
        # 对每个行动调用规则引擎
        night_results = self.rule_engine.resolve_night_phase(phase, actions, self.game_state)
        
        # 处理结算结果
        for result in night_results:
            # 如果有反馈信息（如预言家查验结果），发送给玩家
            if result.get('message') and result.get('player_id'):
                await self.message_callback(result['player_id'], {
                    "type": "action_result",
                    "data": result
                })
            
            # 如果有死亡发生，记录下来以便天亮宣布
            # 注意：resolve_night_phase 可能会直接修改状态，也可能返回结果
            # 我们需要把这些信息存起来，或者直接在天亮时广播
            # 目前简化：即时结算，天亮统一看状态
            pass

    async def _handle_announce_phase(self):
        """处理天亮宣布阶段"""
        # 结算昨晚死亡
        # 检查 night_status 中的 kill_target
        kill_target = self.game_state.night_status.get('kill_target')
        if kill_target:
            # 确认死亡（如果没有被救）
            # 注意：如果被救，kill_target 应该已经被 WitchAntidoteRule 清除或标记
            # 但为了保险，再次检查 saved_by_witch
            if not self.game_state.night_status.get('saved_by_witch'):
                self.game_state.kill_player(kill_target, reason='killed_by_werewolf')
                print(f"[Orchestrator] 结算死亡: {kill_target}")
        
        # 清空临时状态
        self.game_state.night_status = {}
        
        # 重新同步状态，确保前端看到最新的存活状态
        await self._sync_player_states()
        
        # 计算昨晚死亡玩家：对比入夜前和现在的存活名单
        current_alive = set(self.game_state.get_alive_players())
        
        # 如果 night_start_alive_players 为空（可能是重启或第一晚异常），则回退到当前死者
        if not self.night_start_alive_players:
             # 这种情况下很难准确判断，为了安全，不报具体名字，只报天亮
             dead_last_night = []
        else:
             dead_last_night = list(self.night_start_alive_players - current_alive)
        
        # 清空快照，为下一晚做准备
        self.night_start_alive_players = set()

        message = f"第 {self.game_state.current_round} 天天亮了。"
        
        if not dead_last_night:
            death_msg = "昨晚是平安夜。"
        else:
            # 获取玩家名字
            dead_names = [self.game_state.player_names.get(pid, pid) for pid in dead_last_night]
            death_msg = f"昨晚死亡玩家：{', '.join(dead_names)}"
            
        await self.broadcast({
            "type": "announcement",
            "data": {
                "message": message,
                "sub_message": death_msg
            }
        })


    async def _handle_discussion_phase(self):
        """处理白天讨论阶段"""
        alive_players = self.game_state.get_alive_players()
        
        # 依次发言
        for pid in alive_players:
            performer = self.performers[pid]
            
            # 通知所有人轮到谁发言
            await self.broadcast({
                "type": "discussion_turn",
                "data": {"player_id": pid}
            })
            
            # 请求发言
            speech = await self._request_speech(performer)
            
            # **关键修复：记录发言到历史**
            self.game_state.record_action({
                "type": "speech",
                "player_id": pid,
                "content": speech,
                "round": self.game_state.current_round,
                "phase": self.game_state.current_phase
            })
            
            # 广播发言
            await self.broadcast({
                "type": "speech",
                "data": {
                    "player_id": pid,
                    "content": speech
                }
            })


    async def _handle_vote_phase(self):
        """处理投票阶段"""
        alive_players = self.game_state.get_alive_players()
        
        # 优先请求人类玩家投票，然后并行请求AI投票
        # 这样确保人类玩家从收到请求开始有完整的超时时间
        human_players = [pid for pid in alive_players if self.performers[pid].is_human]
        ai_players = [pid for pid in alive_players if not self.performers[pid].is_human]
        
        votes = {} # voter -> target
        
        # 1. 先请求人类玩家投票（顺序）
        for pid in human_players:
            performer = self.performers[pid]
            result = await self._request_action(performer)
            if isinstance(result, dict) and result.get('action_type') == 'vote':
                target = result.get('target')
                # 验证target是有效的玩家ID
                if target and target in self.game_state.player_ids:
                    votes[pid] = target
                else:
                    print(f"[Orchestrator] 玩家 {pid} 投票目标无效: {target}")
        
        # 2. 并行请求AI玩家投票
        if ai_players:
            vote_tasks = []
            for pid in ai_players:
                performer = self.performers[pid]
                vote_tasks.append(self._request_action(performer))
                
            results = await asyncio.gather(*vote_tasks, return_exceptions=True)
            
            for pid, result in zip(ai_players, results):
                if isinstance(result, dict) and result.get('action_type') == 'vote':
                    target = result.get('target')
                    if target and target in self.game_state.player_ids:
                        votes[pid] = target
        
        print(f"[Orchestrator] 有效投票: {votes}")
        
        # **关键修复：记录投票到历史**
        for voter, target in votes.items():
            self.game_state.record_action({
                "type": "vote",
                "voter": voter,
                "target": target,
                "round": self.game_state.current_round,
                "phase": self.game_state.current_phase
            })
        
        # 广播投票结果
        await self.broadcast({
            "type": "vote_result",
            "data": votes
        })
        
        # 结算投票（找票数最多的）
        if votes:
            from collections import Counter
            counts = Counter(votes.values())
            # 检查平票
            most_common = counts.most_common(2)
            if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
                # 平票，无人出局（简化规则）
                await self.broadcast({
                    "type": "announcement",
                    "data": {"message": "平票，无人出局"}
                })
            else:
                exiled = most_common[0][0]
                print(f"[Orchestrator] 放逐玩家: {exiled}")
                
                # 确保exiled是有效ID
                if exiled and exiled in self.game_state.player_ids:
                    self.game_state.kill_player(exiled, reason="voted_out")
                    
                    await self.broadcast({
                        "type": "exile",
                        "data": {"player_id": exiled}
                    })
                    
                    # 再次同步状态
                    await self._sync_player_states()
                else:
                    print(f"[Orchestrator] 错误：放逐目标无效 {exiled}")
                    await self.broadcast({
                        "type": "error",
                        "data": {"message": "投票统计错误", "severity": "error"}
                    })
        else:
            await self.broadcast({
                "type": "announcement",
                "data": {"message": "无人投票，无人出局"}
            })

    async def _handle_game_end(self, winner: str):
        """处理游戏结束"""
        # 广播游戏结束消息
        await self.broadcast({
            "type": "game_end",
            "data": {
                "winner": winner,
                "message": f"游戏结束！{'狼人' if winner == 'werewolf' else '好人'}阵营获胜！"
            }
        })
        
        # 发送复盘信息：揭露所有玩家身份
        review_data = []
        for pid in self.game_state.player_ids:
            role_id = self.game_state.player_roles.get(pid)
            role_def = self.game_state.get_role_definition(role_id)
            review_data.append({
                "player_id": pid,
                "player_name": self.game_state.player_names.get(pid, pid),
                "role_id": role_id,
                "role_name": role_def.role_name if role_def else role_id,
                "camp": role_def.camp.value if role_def else "unknown",
                "status": self.game_state.player_status.get(pid).value
            })
        
        await self.broadcast({
            "type": "game_review",
            "data": {
                "title": "游戏复盘 - 身份揭露",
                "players": review_data
            }
        })

    async def _request_action(self, performer: WerewolfPerformer, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """请求玩家行动"""
        if performer.is_human:
            # 发送请求给前端
            request_data = {
                "options": [opt.dict() for opt in performer.get_available_actions(self.game_state)],
                "timeout": 30 # 秒
            }
            # 如果有上下文，添加到请求中
            if context:
                request_data["context"] = context
                
            await self.message_callback(performer.player_id, {
                "type": "action_request",
                "data": request_data
            })
            
            # 等待用户输入
            future = asyncio.Future()
            self.pending_actions[performer.player_id] = future
            
            try:
                # 等待结果，带超时
                result = await asyncio.wait_for(future, timeout=35)
                return result
            except asyncio.TimeoutError:
                return {"action_type": "skip", "reason": "timeout"}
            finally:
                if performer.player_id in self.pending_actions:
                    del self.pending_actions[performer.player_id]
        else:
            # AI行动，带重试逻辑
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    return await performer.gen_action(self.game_state)
                except Exception as e:
                    print(f"[Orchestrator] AI行动失败 (尝试 {attempt+1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)  # 等待1秒后重试
                    else:
                        # 重试失败，广播错误并跳过
                        await self.broadcast({
                            "type": "error",
                            "data": {
                                "message": f"AI 玩家 {performer.player_id} 行动失败（网络错误），已跳过",
                                "severity": "warning"
                            }
                        })
                        return {"action_type": "skip", "reason": "network_error"}

    async def _request_action_with_context(self, performer: WerewolfPerformer, context: Dict[str, Any]) -> Dict[str, Any]:
        """请求玩家行动（带上下文）"""
        # 复用 _request_action，它现在支持 context 参数
        return await self._request_action(performer, context)

    async def _request_speech(self, performer: WerewolfPerformer) -> str:
        """请求玩家发言"""
        if performer.is_human:
            # 发送请求给前端
            await self.message_callback(performer.player_id, {
                "type": "action_request",
                "data": {
                    "options": [{
                        "action_type": "speech",
                        "description": "请发言",
                        "is_speech": True
                    }],
                    "timeout": 60
                }
            })
            
            # 等待用户输入
            future = asyncio.Future()
            self.pending_actions[performer.player_id] = future
            
            try:
                # 等待结果，带超时
                result = await asyncio.wait_for(future, timeout=65)
                # 前端返回的是 {"action_type": "speech", "content": "..."}
                return result.get("content", "（沉默）")
            except asyncio.TimeoutError:
                return "（超时未发言）"
            finally:
                if performer.player_id in self.pending_actions:
                    del self.pending_actions[performer.player_id]
        else:
            # AI发言，带重试逻辑
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    return await performer.gen_speech(self.game_state)
                except Exception as e:
                    print(f"[Orchestrator] AI发言失败 (尝试 {attempt+1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)  # 等待1秒后重试
                    else:
                        # 重试失败，广播错误
                        await self.broadcast({
                            "type": "error",
                            "data": {
                                "message": f"AI 玩家 {performer.player_id} 发言失败（网络错误），已跳过",
                                "severity": "warning"
                            }
                        })
                        return f"（{performer.player_id} 因网络问题无法发言）"

    def handle_player_input(self, player_id: str, input_data: dict):
        """处理玩家输入（来自WebSocket）"""
        if player_id in self.pending_actions:
            future = self.pending_actions[player_id]
            if not future.done():
                future.set_result(input_data)
    
    async def _sync_player_states(self):
        """同步所有玩家的状态到前端"""
        # 给每个玩家发送他们能看到的状态
        for observer_id in self.game_state.player_ids:
            # 构建玩家列表（每个玩家看到的视角不同）
            players_info = []
            observer_role = self.game_state.get_player_role(observer_id)
            
            for pid in self.game_state.player_ids:
                player_info = {
                    "id": pid,
                    "name": self.game_state.player_names.get(pid, pid),
                    "alive": self.game_state.is_alive(pid)
                }
                
                # 狼人可以看到其他狼人的身份
                if observer_role == "werewolf":
                    target_role = self.game_state.get_player_role(pid)
                    if target_role == "werewolf":
                        player_info["revealed_role"] = "狼人"
                
                players_info.append(player_info)
            
            await self.message_callback(observer_id, {
                "type": "player_states",
                "data": {
                    "players": players_info
                }
            })

import random
