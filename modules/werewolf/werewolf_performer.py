"""
狼人杀表演者适配器

负责处理玩家（AI或人类）在狼人杀游戏中的行为。
核心功能：
1. AI行为生成：基于LLM生成行动、发言和思考
2. 人类交互支持：提供可用行动选项、上下文信息
3. 统一接口：Orchestrator通过统一接口调用，不区分AI/人类
"""

from typing import Dict, List, Any, Optional, Union
import json
from pydantic import BaseModel

from ..main_performer import Performer
from .game_state import WerewolfGameState, PlayerStatus
from .role_registry import RoleDefinition, TargetType
from .config_loader import GameConfig


class ActionOption(BaseModel):
    """行动选项定义，用于前端渲染交互界面"""
    action_type: str
    description: str
    targets: List[str]  # 可选目标ID列表
    max_targets: int = 1
    min_targets: int = 1
    can_skip: bool = False  # 是否可以跳过（如空刀、不救）


class WerewolfPerformer:
    """
    狼人杀表演者适配器
    
    包装原始Performer，添加狼人杀专用逻辑。
    """
    
    def __init__(self, 
                 performer: Performer, 
                 role_def: RoleDefinition,
                 is_human: bool = False):
        self.performer = performer
        self.role_def = role_def
        self.is_human = is_human
        
        # 思考链缓存
        self.last_thought: str = ""
    
    @property
    def player_id(self) -> str:
        return self.performer.role_code
        
    @property
    def name(self) -> str:
        return self.performer.role_name
    
    def get_available_actions(self, game_state: WerewolfGameState) -> List[ActionOption]:
        """
        获取当前状态下可用的行动选项
        
        用于：
        1. 前端渲染人类玩家的操作界面
        2. AI决策时的约束条件
        """
        options = []
        
        # 如果玩家已死亡，通常没有行动（除非是猎人刚死）
        if not game_state.is_alive(self.player_id):
            # 检查猎人技能
            if self.role_def.role_id == "hunter":
                # 检查是否刚死亡且未开枪
                # 注意：这里需要game_state提供更精确的"刚死亡"状态或事件上下文
                # 简化处理：检查是否在死亡结算阶段且未发动技能
                pass
            return []
            
        current_phase = game_state.current_phase
        
        # 遍历角色技能，检查是否在当前阶段可用
        for ability in self.role_def.abilities:
            if ability.phase != current_phase:
                continue
                
            # 检查技能限制（如只能用一次）
            if "use_once_per_game" in ability.restrictions:
                if game_state.is_ability_used(self.player_id, ability.ability_id):
                    continue
            
            # 确定可选目标
            targets = []
            if ability.target_type == TargetType.SINGLE_PLAYER:
                # 女巫解药特殊处理：只能针对被击杀者
                if self.role_def.role_id == "witch" and ability.ability_id == "antidote":
                    # 解药目标由context提供（从orchestrator传递的kill_target）
                    # 这里暂时为空，实际目标会在orchestrator中通过context传递
                    # 前端会自动使用context中的kill_target
                    targets = []  # 解药不需要选择目标，自动针对被击杀者
                else:
                    # 其他技能：获取所有存活玩家
                    alive_players = game_state.get_alive_players()
                    
                    for pid in alive_players:
                        # 狼人可以选择自己（自刀），其他角色检查can_target_self
                        if self.role_def.role_id == "werewolf":
                            # 狼人可以选任何存活玩家，包括自己
                            targets.append(pid)
                        else:
                            # 其他角色检查是否可以选自己
                            if pid == self.player_id and not ability.can_target_self:
                                continue
                            targets.append(pid)
            
            options.append(ActionOption(
                action_type=f"{self.role_def.role_id}_{ability.ability_id}",
                description=ability.description,
                targets=targets,
                can_skip=True # 大多数技能可以选择不发动
            ))
            
        # 白天发言阶段
        if current_phase == "day_discussion":
            options.append(ActionOption(
                action_type="speak",
                description="发表言论",
                targets=[],
                max_targets=0,
                min_targets=0,
                can_skip=False
            ))
            
        # 投票阶段
        if current_phase == "day_vote":
            # 获取存活玩家作为候选人
            candidates = game_state.get_alive_players()
            # 排除自己（通常不能投自己？或者可以？视规则而定，标准规则可以）
            
            options.append(ActionOption(
                action_type="vote",
                description="投票放逐",
                targets=candidates,
                can_skip=True # 可以弃票
            ))
            
        return options

    async def gen_action(self, game_state: WerewolfGameState, context: str = "") -> Dict[str, Any]:
        """
        生成行动（AI）
        
        Args:
            game_state: 当前游戏状态
            context: 额外的上下文信息（如"轮到你发言了"）
            
        Returns:
            Dict: 行动数据
        """
        if self.is_human:
            raise RuntimeError("Cannot generate AI action for human player")
            
        available_actions = self.get_available_actions(game_state)
        if not available_actions:
            return {"action_type": "skip", "reason": "no_available_actions"}
            
        # 构建Prompt
        prompt = self._build_action_prompt(game_state, available_actions, context)
        
        # 调用LLM
        # 这里复用Performer的LLM能力
        # 注意：需要适配Performer的chat接口
        response = await self._call_llm(prompt)
        
        # 解析响应
        return self._parse_action_response(response, available_actions)

    async def gen_speech(self, game_state: WerewolfGameState, context: str = "") -> str:
        """生成发言（AI）"""
        if self.is_human:
            raise RuntimeError("Cannot generate AI speech for human player")
            
        prompt = self._build_speech_prompt(game_state, context)
        response = await self._call_llm(prompt)
        
        # 如果返回的是JSON格式，尝试提取speech字段
        try:
            import json
            data = json.loads(response)
            if isinstance(data, dict) and 'speech' in data:
                return data['speech']
        except:
            pass
        
        # 否则直接返回文本
        return response

    def _build_action_prompt(self, game_state: WerewolfGameState, options: List[ActionOption], context: str) -> str:
        """构建行动Prompt"""
        # 获取可见状态
        visible_state = game_state.get_visible_state(self.player_id)
        
        # 格式化选项
        options_text = ""
        for opt in options:
            targets_text = ", ".join([game_state.player_names.get(t, t) for t in opt.targets])
            options_text += f"- {opt.action_type}: {opt.description}\n  可选目标: {targets_text}\n"
        
        # **关键修复：添加游戏历史信息**
        history_text = self._build_game_history_summary(game_state)
            
        prompt = f"""
你正在进行狼人杀游戏。
你的身份是：{self.role_def.role_name}
当前阶段：{game_state.current_phase}

【游戏历史】
{history_text}

【游戏状态】
存活玩家：{", ".join([game_state.player_names.get(p, p) for p in game_state.get_alive_players()])}
{self._get_role_specific_info(game_state)}

【可用行动】
{options_text}

【任务】
请根据当前局势和游戏历史，选择一个行动。
请先进行思考（分析局势、推测身份），然后给出行动决策。

**重要规范**：
1. target 必须使用玩家ID（如 "player_1", "player_2"），禁止使用中文名称
2. 请严格从上面列出的可选目标中选择
3. 你的决策必须基于【游戏历史】中的真实信息

返回格式（JSON）：
{{
    "thought": "你的思考过程...",
    "action_type": "选择的行动类型",
    "target": "player_X",
    "skip": false
}}
"""
        return prompt

    def _get_role_specific_info(self, game_state: WerewolfGameState) -> str:
        """获取角色特定的信息（如狼人队友、预言家查验历史）"""
        info = ""
        if self.role_def.role_id == "werewolf":
            teammates = game_state.get_players_by_role("werewolf")
            names = [game_state.player_names.get(p, p) for p in teammates if p != self.player_id]
            if names:
                info += f"你的狼人队友是：{', '.join(names)}\n"
                
        # TODO: 从记忆中检索查验历史等
        return info

    async def _call_llm(self, prompt: str) -> str:
        """调用LLM"""
        try:
            if self.performer.llm is None:
                raise RuntimeError("LLM未初始化")
            response = self.performer.llm.chat(prompt)
            if not response:
                raise RuntimeError("LLM返回空响应")
            return response
        except Exception as e:
            print(f"[WerewolfPerformer] LLM调用失败: {e}")
            raise  # 向上抛出异常，让Orchestrator处理

    def _parse_action_response(self, response: str, options: List[ActionOption]) -> Dict[str, Any]:
        """解析LLM响应"""
        try:
            # 尝试解析JSON
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                
                # 规范化 action_type
                action_type = data.get('action_type')
                target = data.get('target')
                
                # 查找匹配的选项
                selected_option = next((opt for opt in options if opt.action_type == action_type), None)
                
                if selected_option:
                    # 规范化 target
                    if target:
                        # 如果target是数字或字符串数字，尝试补全 player_ 前缀
                        if str(target).isdigit() or (isinstance(target, str) and target.startswith("player") is False):
                             # 尝试在选项的目标列表中找到匹配项
                             # 假设目标列表是 ['player_0', 'player_1', ...]
                             # 如果 LLM 返回 '1'，我们要映射到 'player_1'
                             
                             # 简单的数字匹配
                             target_str = str(target)
                             for valid_target in selected_option.targets:
                                 if valid_target.endswith(f"_{target_str}") or valid_target == target_str:
                                     data['target'] = valid_target
                                     break
                                     
                             # 如果还是没找到，且target是纯数字，强制加上前缀试试
                             if data['target'] == target and str(target).isdigit():
                                 candidate = f"player_{target}"
                                 if candidate in selected_option.targets:
                                     data['target'] = candidate
                    
                    return data
                elif data.get('action_type') == 'skip':
                    return data
                    
        except Exception as e:
            print(f"[WerewolfPerformer] 解析响应失败: {e}")
            
        # 默认返回跳过
        return {"action_type": "skip", "reason": "parse_error"}

    def _build_speech_prompt(self, game_state: WerewolfGameState, context: str) -> str:
        """构建发言Prompt"""
        # 获取存活和死亡玩家信息
        alive_players = [game_state.player_names.get(p, p) for p in game_state.get_alive_players()]
        dead_players = [game_state.player_names.get(p, p) for p in game_state.get_dead_players()]
        
        # 构建游戏状态描述
        game_context = f"当前天数：第 {game_state.current_round} 天\n"
        game_context += f"存活玩家：{', '.join(alive_players)}\n"
        if dead_players:
            game_context += f"已死亡玩家：{', '.join(dead_players)}\n"
        else:
            game_context += "目前无人死亡\n"
        
        # **关键修复：添加游戏历史信息**
        history_text = self._build_game_history_summary(game_state)
        
        return f"""
你正在玩狼人杀游戏。
你的身份是：{self.role_def.role_name}
当前轮到你发言。

【游戏状态】
{game_context}

【游戏历史回顾】
{history_text}

【任务】
请以1-2句话发表你的观点。你可以：
- 分析局势（基于上面的游戏历史）
- 怀疑某位玩家
- 为自己辩解
- 提供线索

**重要规则**：
1. **必须基于【游戏历史回顾】中的真实信息发言**
2. **不要编造未发生的事件**（如说"玩家X死了"但实际没死）
3. 不要暴露你的真实身份（除非你是预言家想跳身份）
4. 保持简短，不要超过50字
5. 直接说话内容，不要加"我说："等前缀

发言：
"""

    def _build_game_history_summary(self, game_state: WerewolfGameState) -> str:
        """构建游戏历史摘要（给AI看的重要上下文）"""
        if not game_state.action_history:
            return "游戏刚开始，还没有发生任何事件。"
        
        # 构建结构化的历史摘要
        summary_lines = []
        
        # **新增：我的私密行动记录（如果我是特殊角色）**
        my_private_actions = []
        my_role = game_state.get_player_role(self.player_id)
        
        # 1. 按轮次组织事件
        events_by_round = {}
        for action in game_state.action_history:
            round_num = action.get('round', 0)
            if round_num not in events_by_round:
                events_by_round[round_num] = []
            events_by_round[round_num].append(action)
            
            # **收集我的私密行动**
            action_player = action.get('player_id')
            if action_player == self.player_id:
                action_type = action.get('type', action.get('action_type', ''))
                
                # 预言家查验记录
                if action_type == 'seer_check' and my_role == 'seer':
                    result = action.get('result', {})
                    target = action.get('target')
                    target_name = game_state.player_names.get(target, target)
                    is_werewolf = result.get('is_werewolf', False)
                    my_private_actions.append(
                        f"第{round_num}晚：查验{target_name} - {'是狼人' if is_werewolf else '是好人'}"
                    )
                
                # 女巫用药记录
                elif action_type == 'witch_antidote' and my_role == 'witch':
                    target = action.get('target')
                    target_name = game_state.player_names.get(target, target)
                    my_private_actions.append(
                        f"第{round_num}晚：使用解药救了{target_name}"
                    )
                
                elif action_type == 'witch_poison' and my_role == 'witch':
                    target = action.get('target')
                    target_name = game_state.player_names.get(target, target)
                    my_private_actions.append(
                        f"第{round_num}晚：使用毒药毒杀了{target_name}"
                    )
        
        # 2. 为每一轮生成摘要（只显示公开信息）
        for round_num in sorted(events_by_round.keys()):
            actions = events_by_round[round_num]
            round_summary = []
            
            # 收集这一轮的死亡事件
            deaths_this_round = []
            exiles_this_round = []
            speeches_this_round = []
            votes_this_round = []
            
            for action in actions:
                action_type = action.get('type', action.get('action_type', ''))
                
                if action_type == 'death':
                    player_id = action.get('player_id')
                    reason = action.get('reason', '')
                    player_name = game_state.player_names.get(player_id, player_id)
                    
                    if 'werewolf' in reason or 'killed' in reason:
                        deaths_this_round.append(f"{player_name}被狼人击杀")
                    elif 'voted' in reason or 'exile' in reason:
                        exiles_this_round.append(f"{player_name}被投票放逐")
                    elif 'poison' in reason:
                        deaths_this_round.append(f"{player_name}被毒杀")
                    else:
                        deaths_this_round.append(f"{player_name}死亡({reason})")
                
                # 收集发言
                elif action_type == 'speech':
                    player_id = action.get('player_id')
                    content = action.get('content', '')
                    player_name = game_state.player_names.get(player_id, player_id)
                    # 只保留最近一轮的发言
                    if round_num == game_state.current_round:
                        speeches_this_round.append(f"{player_name}: {content}")
                
                # 收集投票
                elif action_type == 'vote':
                    voter = action.get('voter')
                    target = action.get('target')
                    voter_name = game_state.player_names.get(voter, voter)
                    target_name = game_state.player_names.get(target, target)
                    # 只显示当前轮的投票
                    if round_num == game_state.current_round:
                        votes_this_round.append(f"{voter_name}→{target_name}")
            
            # 构建这一轮的摘要
            if deaths_this_round or exiles_this_round or speeches_this_round or votes_this_round:
                summary_lines.append(f"第{round_num}轮：")
                if deaths_this_round:
                    summary_lines.append(f"  夜晚：{'; '.join(deaths_this_round)}")
                if speeches_this_round:
                    summary_lines.append(f"  讨论发言：")
                    for speech in speeches_this_round[-5:]:  # 只显示最近5条发言
                        summary_lines.append(f"    - {speech}")
                if votes_this_round:
                    summary_lines.append(f"  投票情况：{'; '.join(votes_this_round)}")
                if exiles_this_round:
                    summary_lines.append(f"  放逐结果：{'; '.join(exiles_this_round)}")
        
        if not summary_lines:
            summary_lines.append(f"这是第{game_state.current_round}天的第一次讨论，昨晚{'是平安夜' if game_state.current_round > 1 else '游戏刚开始'}。")
        
        # **添加我的私密行动记录**
        if my_private_actions:
            summary_lines.append("\n【我的秘密行动记录】")
            for action_desc in my_private_actions:
                summary_lines.append(f"  - {action_desc}")
        
        return "\n".join(summary_lines)
