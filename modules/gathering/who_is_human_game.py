"""
谁是人类游戏模块

玩法：裁判让4人（3个AI，1个玩家）描述某样物品，通过描述来辨别谁是人类
"""

import os
import uuid
import asyncio
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from fastapi import WebSocket
from google import genai

MODEL_NAME = "gemini-2.5-flash"

# 物品列表（供裁判选择）
ITEMS = [
    "苹果", "手机", "书", "椅子", "杯子", "电脑", "汽车", "花", "猫", "狗",
    "雨伞", "钥匙", "钱包", "眼镜", "手表", "笔", "纸", "灯", "门", "窗",
    "树", "山", "海", "云", "太阳", "月亮", "星星", "风", "雨", "雪",
    "茶", "咖啡", "面包", "米饭", "水", "火", "冰", "糖", "盐", "油"
]


@dataclass
class AIPlayer:
    """AI玩家"""
    player_id: str
    name: str
    client: genai.Client
    
    async def generate_description(self, item: str, round_num: int, previous_round_speeches: Dict[str, str] = None, active_players: List[str] = None) -> str:
        """生成对物品的描述
        
        Args:
            item: 要描述的物品
            round_num: 当前轮次
            previous_round_speeches: 上一轮所有玩家的发言字典 {player_id: description}
            active_players: 当前活跃玩家名称列表
        """
        # 构建上一轮发言的上下文
        history_context = ""
        if previous_round_speeches and len(previous_round_speeches) > 0:
            history_context = f"\n\n上一轮所有玩家的发言（供参考）：\n"
            for pid, prev_desc in previous_round_speeches.items():
                history_context += f"- {prev_desc}\n"
            history_context += "\n注意：你可以参考上一轮的发言，但要用自己的方式描述，不要完全重复。"
        
        active_players_text = ""
        if active_players:
            active_players_text = f"\n当前活跃玩家：{', '.join(active_players)}"
        
        prompt = f"""你是一个参与"谁是人类"游戏的AI玩家。游戏规则是：裁判会给出一个物品，你需要用一句话描述这个物品，但不能直接说出物品的名称。

当前物品：{item}
当前轮次：第{round_num}轮
{active_players_text}
{history_context}

请用一句话描述这个物品，要求：
1. 不能直接说出物品名称
2. 描述要自然、像人类会说的话
3. 可以描述物品的外观、用途、特点、使用场景等
4. 尽量简洁，一句话即可
5. 如果这是第2轮或之后，可以参考上一轮的发言，但要用自己的方式表达

只输出描述内容，不要其他解释。"""

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt
                )
            )
            description = (response.text or "").strip()
            # 清理描述，移除可能的引号或多余格式
            description = description.strip('"').strip("'").strip()
            return description if description else f"这是一个{item}。"
        except Exception as e:
            print(f"AI生成描述失败: {e}")
            return f"这是一个{item}。"


class WhoIsHumanGameSession:
    """谁是人类游戏会话"""
    
    def __init__(self, game_id: str, user_id: int, username: str):
        self.game_id = game_id
        self.user_id = user_id
        self.username = username
        
        # 初始化AI客户端
        self.client = None
        self.ai_players: List[AIPlayer] = []
        self._init_ai_client()
        
        # 游戏状态
        self.current_item: Optional[str] = None
        self.current_round = 0
        self.max_rounds = 2  # 默认2轮（正常情况）
        self.is_finished = False
        
        # 玩家信息（4人：3个AI + 1个玩家）
        self.all_players: List[Dict] = []  # 所有玩家（包括已出局的）
        self.active_players: List[Dict] = []  # 当前活跃玩家（未出局的）
        self.human_player_id: Optional[str] = None
        
        # 描述记录
        self.descriptions: Dict[str, str] = {}  # player_id -> description
        self.votes: Dict[str, str] = {}  # voter_id -> voted_player_id
        
        # 游戏历史（每轮的完整记录）
        self.history: List[Dict] = []
        
        # 出局玩家列表
        self.eliminated_players: List[str] = []  # player_id列表
        
        # WebSocket连接
        self.websocket: Optional[WebSocket] = None
    
    def _init_ai_client(self):
        """初始化AI客户端"""
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                from sw_utils import load_json_file
                try:
                    config = load_json_file("config.json")
                    api_key = config.get("GOOGLE_API_KEY") or config.get("GEMINI_API_KEY")
                except:
                    pass
            
            if api_key:
                os.environ["GOOGLE_API_KEY"] = api_key
                self.client = genai.Client()
                print(f"AI客户端初始化成功")
            else:
                print("警告: 未找到GOOGLE_API_KEY配置")
        except Exception as e:
            print(f"初始化AI客户端失败: {e}")
            import traceback
            traceback.print_exc()
    
    async def start_game(self):
        """开始游戏"""
        if not self.client:
            await self.send_error("AI服务未初始化")
            return
        
        # 在游戏开始时随机选择一个物品，整个游戏过程中都使用这个物品
        self.current_item = random.choice(ITEMS)
        
        # 创建4个玩家（3个AI + 1个玩家）
        ai_names = ["小明", "小红", "小刚"]
        self.all_players = []
        
        # 创建AI玩家
        for i, name in enumerate(ai_names):
            player_id = f"ai_{i+1}"
            ai_player = AIPlayer(player_id, name, self.client)
            self.ai_players.append(ai_player)
            self.all_players.append({
                "id": player_id,
                "name": name,
                "type": "ai"
            })
        
        # 创建人类玩家
        human_player_id = f"human_{self.user_id}"
        self.human_player_id = human_player_id
        self.all_players.append({
            "id": human_player_id,
            "name": self.username,
            "type": "human"
        })
        
        # 随机打乱玩家顺序
        random.shuffle(self.all_players)
        
        # 初始化活跃玩家（开始时所有人都在）
        self.active_players = self.all_players.copy()
        self.eliminated_players = []
        
        # 通知前端游戏开始，显示物品
        await self.send_message({
            "type": "game_start",
            "item": self.current_item,
            "players": self.all_players,
            "active_players": self.active_players,
            "max_rounds": self.max_rounds
        })
        
        # 开始第一轮
        await self.start_round()
    
    async def start_round(self):
        """开始新的一轮"""
        # 检查是否应该结束游戏（只剩1-2人时结束）
        if len(self.active_players) <= 2:
            await self.end_game()
            return
        
        self.current_round += 1
        
        # 清空当前轮的描述和投票
        self.descriptions.clear()
        self.votes.clear()
        
        # 获取上一轮的历史（用于AI参考和前端显示）
        previous_round_data = self.history[-1] if self.history else None
        
        # 通知前端开始新的一轮（物品保持不变）
        await self.send_message({
            "type": "round_start",
            "round": self.current_round,
            "max_rounds": self.max_rounds,
            "item": self.current_item,  # 使用游戏开始时选择的物品
            "active_players": self.active_players,  # 当前活跃玩家
            "eliminated_players": self.eliminated_players,  # 已出局玩家
            "previous_round": previous_round_data  # 上一轮的数据（用于显示）
        })
        
        # 让AI玩家生成描述（只让活跃的AI玩家发言）
        await self.collect_ai_descriptions()
    
    async def collect_ai_descriptions(self):
        """收集AI玩家的描述
        
        AI调度方式：
        1. 并发执行：活跃的AI同时调用LLM API生成描述，提高效率
        2. 独立实例：每个AI玩家有独立的AIPlayer实例，但共享同一个genai.Client
        3. 上下文传递：将上一轮所有玩家的发言传递给AI，让AI能够参考
        4. 错误处理：如果某个AI生成失败，使用默认描述，不影响其他AI
        """
        # 获取上一轮所有玩家的发言（用于AI参考）
        previous_round_speeches = {}
        if self.current_round > 1 and self.history:
            last_round = self.history[-1]
            if last_round.get("descriptions"):
                # 记录上一轮每个玩家的发言（包括已出局的）
                previous_round_speeches = last_round["descriptions"].copy()
        
        # 只让活跃的AI玩家生成描述（已出局的不发言）
        active_ai_players = [
            ai_player for ai_player in self.ai_players 
            if ai_player.player_id not in self.eliminated_players
        ]
        
        # 为每个活跃的AI玩家创建生成描述的任务（并发执行）
        tasks = []
        for ai_player in active_ai_players:
            # 传递上一轮所有玩家的发言，让AI能够参考
            task = ai_player.generate_description(
                self.current_item, 
                self.current_round,
                previous_round_speeches=previous_round_speeches,
                active_players=[p["name"] for p in self.active_players]
            )
            tasks.append((ai_player.player_id, task))
        
        # 并发执行所有AI任务
        for player_id, task in tasks:
            try:
                description = await task
                self.descriptions[player_id] = description
                print(f"[AI调度] {player_id} 完成描述生成")
            except Exception as e:
                print(f"[AI调度] AI玩家 {player_id} 生成描述失败: {e}")
                import traceback
                traceback.print_exc()
                # 使用默认描述，不影响游戏进行
                self.descriptions[player_id] = f"这是一个{self.current_item}。"
        
        print(f"[AI调度] 所有活跃AI描述生成完成，共{len(self.descriptions)}条")
        
        # 通知前端所有描述已收集完成（等待人类玩家输入）
        await self.send_message({
            "type": "descriptions_ready",
            "descriptions": {pid: desc for pid, desc in self.descriptions.items()},
            "waiting_for": self.human_player_id if self.human_player_id not in self.eliminated_players else None,
            "active_players": self.active_players
        })
    
    async def submit_human_description(self, description: str):
        """提交人类玩家的描述"""
        if not self.human_player_id:
            await self.send_error("游戏未初始化")
            return
        
        # 检查人类玩家是否已出局
        if self.human_player_id in self.eliminated_players:
            await self.send_error("你已被淘汰，无法发言")
            return
        
        if self.human_player_id in self.descriptions:
            await self.send_error("你已经提交过描述了")
            return
        
        # 保存人类玩家的描述
        self.descriptions[self.human_player_id] = description.strip()
        
        # 再次打乱描述顺序（隐藏谁是玩家）
        all_descriptions = list(self.descriptions.items())
        random.shuffle(all_descriptions)
        
        # 创建匿名描述列表（不显示玩家ID）
        anonymous_descriptions = []
        player_order = []
        for pid, desc in all_descriptions:
            player_info = next((p for p in self.active_players if p["id"] == pid), None)
            if player_info:
                anonymous_descriptions.append({
                    "player_id": pid,
                    "player_name": player_info["name"],
                    "description": desc
                })
                player_order.append(pid)
        
        # 通知前端所有描述已收集完成，可以开始投票
        await self.send_message({
            "type": "all_descriptions_ready",
            "descriptions": anonymous_descriptions,
            "player_order": player_order,
            "active_players": self.active_players
        })
    
    async def submit_vote(self, voter_id: str, voted_player_id: str):
        """提交投票"""
        if voter_id in self.votes:
            await self.send_error("你已经投过票了")
            return
        
        # 检查投票者是否是活跃玩家
        active_player_ids = [p["id"] for p in self.active_players]
        if voter_id not in active_player_ids:
            await self.send_error("你已被淘汰，无法投票")
            return
        
        # 检查投票目标是否有效（必须是活跃玩家，且不能是自己）
        if voted_player_id not in active_player_ids:
            await self.send_error("无效的投票目标")
            return
        
        if voted_player_id == voter_id:
            await self.send_error("不能投票给自己")
            return
        
        self.votes[voter_id] = voted_player_id
        
        # 检查是否所有活跃玩家都投票了
        if len(self.votes) >= len(active_player_ids):
            # 所有人投票完成，计算结果
            await self.calculate_results()
        elif voter_id == self.human_player_id:
            # 人类玩家投票完成，等待AI投票
            await self.collect_ai_votes()
    
    async def collect_ai_votes(self):
        """收集AI玩家的投票"""
        # AI玩家投票（不会投给自己，只能投给活跃玩家）
        active_player_ids = [p["id"] for p in self.active_players]
        
        for ai_player in self.ai_players:
            if ai_player.player_id not in self.votes and ai_player.player_id in active_player_ids:
                # 从其他活跃玩家中选择（排除自己）
                candidates = [pid for pid in active_player_ids 
                             if pid != ai_player.player_id]
                if candidates:
                    voted = random.choice(candidates)
                    self.votes[ai_player.player_id] = voted
        
        # 检查是否所有人都投票了
        if len(self.votes) >= len(active_player_ids):
            # 计算投票结果
            await self.calculate_results()
    
    async def calculate_results(self):
        """计算投票结果"""
        # 统计每个活跃玩家获得的票数
        active_player_ids = [p["id"] for p in self.active_players]
        vote_counts: Dict[str, int] = {pid: 0 for pid in active_player_ids}
        for voted_player_id in self.votes.values():
            if voted_player_id in vote_counts:
                vote_counts[voted_player_id] += 1
        
        # 找出得票最多的玩家（被最多人投票）
        max_votes = max(vote_counts.values()) if vote_counts else 0
        most_voted = [pid for pid, count in vote_counts.items() if count == max_votes]
        
        # 判断是否有平局（多人得票相同且最多）
        is_tie = len(most_voted) > 1
        
        # 记录本轮结果
        round_result = {
            "round": self.current_round,
            "item": self.current_item,
            "descriptions": self.descriptions.copy(),
            "votes": self.votes.copy(),
            "vote_counts": vote_counts,
            "most_voted": most_voted,
            "is_tie": is_tie,
            "eliminated_player": None,  # 本轮出局的玩家
            "human_player_id": self.human_player_id,
            "active_players": self.active_players.copy(),
            "eliminated_players": self.eliminated_players.copy()
        }
        
        # 如果有平局，需要加一轮（只让平局的人再次发言和投票）
        if is_tie:
            print(f"[游戏逻辑] 第{self.current_round}轮出现平局，平局玩家：{most_voted}")
            # 只保留平局的玩家作为活跃玩家
            self.active_players = [p for p in self.active_players if p["id"] in most_voted]
            round_result["tie_players"] = most_voted
            round_result["message"] = f"平局！{len(most_voted)}位玩家得票相同，将进行加时赛"
        else:
            # 没有平局，得票最多的玩家出局
            eliminated_id = most_voted[0]
            eliminated_player = next((p for p in self.active_players if p["id"] == eliminated_id), None)
            
            if eliminated_player:
                self.eliminated_players.append(eliminated_id)
                self.active_players = [p for p in self.active_players if p["id"] != eliminated_id]
                round_result["eliminated_player"] = eliminated_id
                round_result["eliminated_player_name"] = eliminated_player["name"]
                round_result["message"] = f"{eliminated_player['name']} 被投票出局"
                print(f"[游戏逻辑] 第{self.current_round}轮，{eliminated_player['name']} 出局")
                
                # 如果人类玩家出局，立即结束游戏
                if eliminated_id == self.human_player_id:
                    print(f"[游戏逻辑] 人类玩家出局，游戏立即结束")
                    self.history.append(round_result)
                    
                    # 发送结果
                    await self.send_message({
                        "type": "round_result",
                        "all_players": self.all_players,
                        **round_result
                    })
                    
                    # 等待一段时间后结束游戏
                    await asyncio.sleep(3)
                    await self.end_game()
                    return
        
        self.history.append(round_result)
        
        # 发送结果
        await self.send_message({
            "type": "round_result",
            "all_players": self.all_players,
            **round_result
        })
        
        # 等待一段时间后开始下一轮或结束游戏
        await asyncio.sleep(3)
        
        # 检查是否应该结束游戏
        if len(self.active_players) <= 2:
            await self.end_game()
        else:
            # 继续下一轮
            await self.start_round()
    
    async def end_game(self):
        """结束游戏"""
        self.is_finished = True
        
        # 判断人类玩家是否存活
        human_survived = self.human_player_id in [p["id"] for p in self.active_players]
        
        # 计算人类玩家在哪一轮被淘汰（如果被淘汰了）
        eliminated_round = None
        for i, round_data in enumerate(self.history, 1):
            if round_data.get("eliminated_player") == self.human_player_id:
                eliminated_round = i
                break
        
        await self.send_message({
            "type": "game_end",
            "total_rounds": self.current_round,
            "human_survived": human_survived,
            "eliminated_round": eliminated_round,
            "final_active_players": self.active_players,
            "eliminated_players": self.eliminated_players,
            "history": self.history
        })
    
    async def connect(self, websocket: WebSocket):
        """连接WebSocket"""
        self.websocket = websocket
        await self.send_game_state()
    
    async def send_game_state(self):
        """发送游戏状态"""
        if not self.websocket:
            return
        
        state = {
            "type": "game_state",
            "current_round": self.current_round,
            "max_rounds": self.max_rounds,
            "is_finished": self.is_finished,
            "all_players": self.all_players,
            "active_players": self.active_players,
            "eliminated_players": self.eliminated_players,
            "history": self.history
        }
        await self.websocket.send_json(state)
    
    async def send_message(self, message: dict):
        """发送消息"""
        if not self.websocket:
            return
        try:
            await self.websocket.send_json(message)
        except Exception as e:
            print(f"发送消息失败: {e}")
    
    async def send_error(self, message: str):
        """发送错误消息"""
        await self.send_message({
            "type": "error",
            "message": message
        })


class WhoIsHumanGameManager:
    """谁是人类游戏管理器"""
    
    def __init__(self):
        self.sessions: Dict[str, WhoIsHumanGameSession] = {}
    
    def create_game(self, user_id: int, username: str) -> str:
        """创建新游戏"""
        game_id = str(uuid.uuid4())
        session = WhoIsHumanGameSession(game_id, user_id, username)
        self.sessions[game_id] = session
        return game_id
    
    def get_session(self, game_id: str) -> Optional[WhoIsHumanGameSession]:
        """获取游戏会话"""
        return self.sessions.get(game_id)
    
    def remove_session(self, game_id: str):
        """移除游戏会话"""
        if game_id in self.sessions:
            del self.sessions[game_id]

