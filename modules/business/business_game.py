"""
商业博弈游戏模块

实现人机定价博弈，参考提供的Python代码逻辑
"""

import re
import os
import time
import uuid
import asyncio
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List
from fastapi import WebSocket, WebSocketDisconnect
from google import genai

# 博弈环境参数
MIN_PRICE = 8
MAX_PRICE = 20
NASH_EQUILIBRIUM_PRICE = 10
MONOPOLY_EQUILIBRIUM_PRICE = 15
DEMAND_A = 20
COST = 10
MAX_ROUNDS = 20  # 设置为20局
DISCOUNT_FACTOR = 0.95

MODEL_NAME = "gemini-3-flash-preview"


def compute_profits(p1: float, p2: float) -> Tuple[float, float, float]:
    """计算利润"""
    base_price = min(p1, p2)
    Q = max(0, DEMAND_A - base_price)

    if Q == 0:
        return 0.0, 0.0, 0.0

    if p1 < p2:
        q1, q2 = Q, 0
    elif p2 < p1:
        q1, q2 = 0, Q
    else:
        q1 = q2 = Q / 2.0

    pi1 = (p1 - COST) * q1
    pi2 = (p2 - COST) * q2
    return Q, pi1, pi2


def price_to_category(price: Optional[float]) -> Optional[str]:
    if price is None:
        return None
    if price <= 11:
        return "LOW"
    elif price <= 14:
        return "MEDIUM"
    else:
        return "HIGH"


def make_state_name(my_last_price: Optional[float], other_last_price: Optional[float]) -> str:
    if my_last_price is None or other_last_price is None:
        return "NO_HISTORY"

    my_cat = price_to_category(my_last_price)
    other_cat = price_to_category(other_last_price)

    if my_cat == "HIGH" and other_cat == "HIGH":
        return "BOTH_HIGH"
    if my_cat == "HIGH" and other_cat != "HIGH":
        return "ME_HIGH_OTHER_LOW"
    if my_cat != "HIGH" and other_cat == "HIGH":
        return "ME_LOW_OTHER_HIGH"
    return "BOTH_NOT_HIGH"


def initial_q_table_text() -> str:
    lines = [
        "# 状态-价格 Q 表（0-100分，初始50）",
        f"# 价格范围：{MIN_PRICE}-{MAX_PRICE}元",
        f"# 重要提示：成本是 {COST} 元，低于 {COST} 元会亏损！",
        "",
    ]
    states = ["NO_HISTORY", "BOTH_HIGH", "ME_HIGH_OTHER_LOW", "ME_LOW_OTHER_HIGH", "BOTH_NOT_HIGH"]
    key_prices = [9, 10, 12, 14, 15, 16]
    for state in states:
        for price in key_prices:
            lines.append(f"{state} + {price} -> 50")
        lines.append("")
    return "\n".join(lines)


@dataclass
class LLMQLearningAgent:
    name: str
    client: genai.Client
    q_table_text: str = field(default_factory=initial_q_table_text)
    experience_buffer: List[Dict] = field(default_factory=list)
    last_thought: str = ""

    def build_prompt(self, current_state: str, current_round: int) -> str:
        exp_lines = []
        if not self.experience_buffer:
            exp_text = "目前无历史经验。\n"
        else:
            recent_exps = self.experience_buffer[-15:]
            for exp in recent_exps:
                r, s_old, a_old, rew, s_new = exp["round"], exp["s_old"], exp["a_old"], exp["reward"], exp["s_new"]
                exp_lines.append(f"- 轮{r}: s={s_old}, a={a_old:.0f}, r={rew:.1f}, s'={s_new}")
            exp_text = "\n".join(exp_lines)

        prompt = f"""
身份：你是定价智能体 {self.name}。

目标：最大化长期累计利润。你的对手是一个人类玩家。

环境参数：
- 你的边际成本 = {COST} 元。
- 市场需求 Q = {DEMAND_A} - P_min。
- 如果价格 < {COST}，你会亏损。
- 纳什均衡（竞争）= {NASH_EQUILIBRIUM_PRICE} 元（利润=0）。
- 垄断均衡（合作）= {MONOPOLY_EQUILIBRIUM_PRICE} 元（总利润最大）。

当前状态：
- 轮数：{current_round}
- 市场状态 (Current State)：{current_state}

近期经验：
{exp_text}

你的 Q 表（记忆）：
{self.q_table_text}

任务：
1. 分析当前局势（Chain of Thought），限制在 30 个汉字以内，简明扼要。猜测人类对手的意图。
2. 根据经验更新 Q 表文本。
3. 选择本轮价格（{MIN_PRICE}-{MAX_PRICE} 整数）。

输出格式严格如下：

THOUGHT_START
<这里写简短思考，30字以内>
THOUGHT_END
ACTION_START
<价格数字>
ACTION_END
QTABLE_START
<完整的 Q 表内容>
QTABLE_END
""".strip()
        return prompt

    def _parse_response(self, raw_text: str) -> Tuple[str, float, str]:
        text = raw_text.strip()
        thought = "无思考"
        m_thought = re.search(r"THOUGHT_START\s*(.*?)\s*THOUGHT_END", text, re.DOTALL | re.IGNORECASE)
        if m_thought:
            thought = m_thought.group(1).strip().replace("\n", " ")

        price = float(NASH_EQUILIBRIUM_PRICE)
        m_action = re.search(r"ACTION_START\s*(.*?)\s*ACTION_END", text, re.DOTALL | re.IGNORECASE)
        if m_action:
            try:
                p_val = float(m_action.group(1).strip())
                price = int(round(p_val))
                price = max(MIN_PRICE, min(MAX_PRICE, price))
            except:
                pass

        q_table = self.q_table_text
        m_q = re.search(r"QTABLE_START\s*(.*?)\s*QTABLE_END", text, re.DOTALL | re.IGNORECASE)
        if m_q:
            q_table = m_q.group(1).strip()

        return thought, float(price), q_table

    async def decide_and_update(self, current_state: str, current_round: int) -> float:
        """异步决策方法，避免阻塞事件循环"""
        prompt = self.build_prompt(current_state, current_round)
        
        # 在线程池中执行同步的API调用，避免阻塞事件循环
        loop = asyncio.get_event_loop()
        
        for attempt in range(3):
            try:
                # 使用线程池执行同步调用
                response = await loop.run_in_executor(
                    None,
                    lambda: self.client.models.generate_content(
                        model=MODEL_NAME,
                        contents=prompt
                    )
                )
                thought, price, new_q_table = self._parse_response(response.text or "")
                self.last_thought = thought
                self.q_table_text = new_q_table
                return price
            except Exception as e:
                if attempt == 2:
                    return float(NASH_EQUILIBRIUM_PRICE)
                await asyncio.sleep(1)  # 使用异步sleep而不是同步sleep
        return float(NASH_EQUILIBRIUM_PRICE)

    def record_outcome(self, r, s, a, rew, s_next):
        self.experience_buffer.append({
            "round": r, "s_old": s, "a_old": a, "reward": rew, "s_new": s_next
        })


class BusinessGameSession:
    """商业博弈游戏会话"""
    
    def __init__(self, game_id: str, user_id: int, username: str):
        self.game_id = game_id
        self.user_id = user_id
        self.username = username
        
        # 初始化AI Agent（延迟初始化，只在需要时创建）
        self.client = None
        self.ai_agent = None
        self._init_ai_agent()
        
        # 游戏状态
        self.current_round = 0
        self.last_price_ai = None
        self.last_price_human = None
        self.history = []
        self.total_profit_ai = 0.0
        self.total_profit_human = 0.0
        self.is_finished = False
        
        # AI预思考状态
        self.next_price_ai: Optional[float] = None
        self.ai_thinking_task: Optional[asyncio.Task] = None
        self.ai_ready = False
        
        # WebSocket连接
        self.websocket: Optional[WebSocket] = None
    
    def _init_ai_agent(self):
        """初始化AI Agent"""
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                # 尝试从config.json读取
                from sw_utils import load_json_file
                try:
                    config = load_json_file("config.json")
                    api_key = config.get("GOOGLE_API_KEY") or config.get("GEMINI_API_KEY")
                except:
                    pass
            
            if api_key:
                os.environ["GOOGLE_API_KEY"] = api_key
                self.client = genai.Client()
                self.ai_agent = LLMQLearningAgent("AI对手", self.client)
                print(f"AI Agent初始化成功")
            else:
                print("警告: 未找到GOOGLE_API_KEY配置，AI Agent将使用默认策略")
                # 创建一个简单的默认Agent（不使用LLM）
                self.ai_agent = None
        except Exception as e:
            print(f"初始化AI Agent失败: {e}")
            import traceback
            traceback.print_exc()
            self.ai_agent = None
    
    def _simple_ai_strategy(self, state: str, round_num: int) -> float:
        """简单AI策略（当没有LLM Agent时使用）"""
        # 简单策略：根据状态选择价格
        if state == "NO_HISTORY":
            # 第一轮，选择垄断均衡价格
            return float(MONOPOLY_EQUILIBRIUM_PRICE)
        elif state == "BOTH_HIGH":
            # 双方都高，保持高价
            return float(MONOPOLY_EQUILIBRIUM_PRICE)
        elif state == "ME_HIGH_OTHER_LOW":
            # 我高对方低，降低价格竞争
            return float(NASH_EQUILIBRIUM_PRICE + 1)
        elif state == "ME_LOW_OTHER_HIGH":
            # 我低对方高，保持低价或稍微提高
            return float(NASH_EQUILIBRIUM_PRICE)
        else:
            # 其他情况，选择纳什均衡
            return float(NASH_EQUILIBRIUM_PRICE)
    
    async def connect(self, websocket: WebSocket):
        """连接WebSocket（websocket已经在server端accept了）"""
        self.websocket = websocket
        await self.send_game_state()
        # 游戏开始时，立即让AI思考第一轮
        if not self.is_finished:
            self._start_ai_thinking_next_round()
    
    async def send_game_state(self):
        """发送游戏状态"""
        if not self.websocket:
            return
        
        state = {
            "type": "game_state",
            "current_round": self.current_round,
            "max_rounds": MAX_ROUNDS,
            "last_price_ai": self.last_price_ai,
            "last_price_human": self.last_price_human,
            "total_profit_ai": round(self.total_profit_ai, 2),
            "total_profit_human": round(self.total_profit_human, 2),
            "history": self.history[-10:],  # 只发送最近10轮
            "is_finished": self.is_finished,
            "state_description": make_state_name(self.last_price_human, self.last_price_ai) if self.current_round > 0 else "NO_HISTORY"
        }
        
        await self.websocket.send_json(state)
    
    async def handle_price_input(self, price: float):
        """处理玩家输入的价格"""
        if self.is_finished:
            await self.send_error("游戏已结束")
            return
        
        if self.current_round >= MAX_ROUNDS:
            self.is_finished = True
            await self.send_game_state()
            return
        
        # 验证价格
        price = float(price)
        if price < MIN_PRICE or price > MAX_PRICE:
            await self.send_error(f"价格必须在 {MIN_PRICE}-{MAX_PRICE} 元之间")
            return
        
        # 确定状态
        state_ai = make_state_name(self.last_price_ai, self.last_price_human)
        state_human = make_state_name(self.last_price_human, self.last_price_ai)
        
        # 获取AI价格（如果已经准备好，直接使用；否则等待后台任务完成）
        if self.ai_ready and self.next_price_ai is not None:
            # AI已经准备好，直接使用预思考的价格
            price_ai = int(round(self.next_price_ai))
            self.ai_ready = False
            self.next_price_ai = None
        else:
            # AI还没准备好，等待后台思考任务完成（避免重复思考）
            if self.ai_thinking_task and not self.ai_thinking_task.done():
                # 后台任务正在运行，等待它完成
                await self.send_ai_thinking()
                try:
                    # 等待任务完成，最多等待30秒
                    await asyncio.wait_for(self.ai_thinking_task, timeout=30.0)
                    # 任务完成后，检查结果
                    if self.ai_ready and self.next_price_ai is not None:
                        price_ai = int(round(self.next_price_ai))
                        self.ai_ready = False
                        self.next_price_ai = None
                    else:
                        # 任务完成但没有结果，使用简单策略
                        price_ai = self._simple_ai_strategy(state_ai, self.current_round + 1)
                except asyncio.TimeoutError:
                    # 超时，使用简单策略
                    print("AI思考超时，使用简单策略")
                    price_ai = self._simple_ai_strategy(state_ai, self.current_round + 1)
                except Exception as e:
                    # 任务出错，使用简单策略
                    print(f"AI思考任务出错: {e}")
                    price_ai = self._simple_ai_strategy(state_ai, self.current_round + 1)
            else:
                # 没有后台任务，需要立即思考（这种情况应该很少发生，比如游戏刚开始）
                await self.send_ai_thinking()
                if self.ai_agent:
                    price_ai = await self.ai_agent.decide_and_update(state_ai, self.current_round + 1)
                    price_ai = int(round(price_ai))
                else:
                    # 如果没有AI Agent，使用简单策略
                    price_ai = self._simple_ai_strategy(state_ai, self.current_round + 1)
        
        price_human = int(round(price))
        
        # 计算利润
        Q, pi_ai, pi_human = compute_profits(price_ai, price_human)
        
        # 更新累计利润
        self.total_profit_ai += pi_ai
        self.total_profit_human += pi_human
        
        # 记录历史
        round_data = {
            "round": self.current_round + 1,
            "price_ai": price_ai,
            "price_human": price_human,
            "profit_ai": round(pi_ai, 2),
            "profit_human": round(pi_human, 2),
            "demand": round(Q, 2),
            "ai_thought": self.ai_agent.last_thought if self.ai_agent else "简单策略",
            "state": state_human
        }
        self.history.append(round_data)
        
        # 更新状态
        self.last_price_ai = price_ai
        self.last_price_human = price_human
        self.current_round += 1
        
        # 记录经验（在更新状态后）
        next_state_ai = make_state_name(price_ai, price_human)
        next_state_human = make_state_name(price_human, price_ai)
        
        if self.ai_agent:
            self.ai_agent.record_outcome(
                self.current_round, state_ai, price_ai, pi_ai, next_state_ai
            )
        
        # 检查是否结束
        if self.current_round >= MAX_ROUNDS:
            self.is_finished = True
        
        # 发送结果
        await self.send_round_result(round_data)
        await self.send_game_state()
        
        # 如果游戏未结束，立即让AI开始思考下一轮（基于刚更新的状态）
        if not self.is_finished:
            self._start_ai_thinking_next_round()
    
    async def send_round_result(self, round_data: dict):
        """发送本轮结果"""
        if not self.websocket:
            return
        
        result = {
            "type": "round_result",
            **round_data
        }
        await self.websocket.send_json(result)
    
    async def send_error(self, message: str):
        """发送错误消息"""
        if not self.websocket:
            return
        
        error = {
            "type": "error",
            "message": message
        }
        await self.websocket.send_json(error)
    
    async def send_ai_thinking(self):
        """发送AI正在思考的提示"""
        if not self.websocket:
            return
        
        thinking_msg = {
            "type": "ai_thinking",
            "message": "AI正在思考中..."
        }
        try:
            await self.websocket.send_json(thinking_msg)
        except:
            pass  # 如果连接已关闭，忽略错误
    
    def _start_ai_thinking_next_round(self):
        """启动AI思考下一轮的后台任务"""
        # 如果已经有思考任务在运行，先取消它
        if self.ai_thinking_task and not self.ai_thinking_task.done():
            self.ai_thinking_task.cancel()
        
        # 启动新的思考任务
        self.ai_ready = False
        self.next_price_ai = None
        self.ai_thinking_task = asyncio.create_task(self._ai_think_next_round())
    
    async def _ai_think_next_round(self):
        """AI思考下一轮（后台任务）"""
        try:
            # 如果游戏已结束，不思考
            if self.is_finished or self.current_round >= MAX_ROUNDS:
                return
            
            # 发送思考提示
            await self.send_ai_thinking()
            
            # 确定下一轮的状态（基于当前轮的结果，此时状态已经更新）
            # 注意：这里的状态是基于上一轮（current_round）的结果，用于思考下一轮（current_round + 1）
            next_state_ai = make_state_name(self.last_price_ai, self.last_price_human)
            next_round = self.current_round + 1
            
            # AI决策（异步）
            if self.ai_agent:
                price_ai = await self.ai_agent.decide_and_update(next_state_ai, next_round)
                price_ai = int(round(price_ai))
            else:
                # 如果没有AI Agent，使用简单策略
                price_ai = self._simple_ai_strategy(next_state_ai, next_round)
            
            # 存储结果
            self.next_price_ai = float(price_ai)
            self.ai_ready = True
            
            # 发送AI已准备好的提示
            await self.send_ai_ready()
            
        except asyncio.CancelledError:
            # 任务被取消，正常情况
            pass
        except Exception as e:
            print(f"AI思考下一轮时出错: {e}")
            import traceback
            traceback.print_exc()
            # 出错时使用默认策略
            if not self.is_finished and self.current_round < MAX_ROUNDS:
                next_state_ai = make_state_name(self.last_price_ai, self.last_price_human)
                next_round = self.current_round + 1
                self.next_price_ai = float(self._simple_ai_strategy(next_state_ai, next_round))
                self.ai_ready = True
    
    async def send_ai_ready(self):
        """发送AI已准备好的提示"""
        if not self.websocket:
            return
        
        ready_msg = {
            "type": "ai_ready",
            "message": "AI已准备好下一轮"
        }
        try:
            await self.websocket.send_json(ready_msg)
        except:
            pass  # 如果连接已关闭，忽略错误
    
    def get_final_stats(self) -> dict:
        """获取最终统计"""
        return {
            "total_profit_ai": round(self.total_profit_ai, 2),
            "total_profit_human": round(self.total_profit_human, 2),
            "total_rounds": self.current_round,
            "history": self.history
        }


class BusinessGameManager:
    """商业博弈游戏管理器"""
    
    def __init__(self):
        self.sessions: Dict[str, BusinessGameSession] = {}
    
    def create_game(self, user_id: int, username: str) -> str:
        """创建新游戏"""
        game_id = str(uuid.uuid4())
        session = BusinessGameSession(game_id, user_id, username)
        self.sessions[game_id] = session
        return game_id
    
    def get_session(self, game_id: str) -> Optional[BusinessGameSession]:
        """获取游戏会话"""
        return self.sessions.get(game_id)
    
    def remove_session(self, game_id: str):
        """移除游戏会话"""
        if game_id in self.sessions:
            del self.sessions[game_id]

