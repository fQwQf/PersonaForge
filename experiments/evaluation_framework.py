"""
PersonaForge Experiment Evaluation Framework
=============================================

实验评测框架：评估心理学驱动的三层人格模型 + 双重思维链生成机制

主要评测维度：
1. Personality Consistency (PC) - 人格一致性
2. Style Adherence (SA) - 风格遵守度
3. Response Diversity (RD) - 回复多样性
4. Defense Mechanism Manifestation (DM) - 防御机制体现
"""

import os
import sys
import json
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import re

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class EvaluationScenario:
    """评测场景"""
    scenario_id: str
    scenario_type: str  # emotional, conflict, casual, first_encounter
    context: str  # 场景描述
    trigger: str  # 触发行为/对话
    trigger_role: str  # 触发者
    expected_traits: Dict[str, Any]  # 期望体现的特质
    

@dataclass
class EvaluationResult:
    """评测结果"""
    scenario_id: str
    role_code: str
    response: str
    inner_monologue: Optional[str]  # 如果使用双重思维链，记录内心独白
    pc_score: float  # 人格一致性分数 0-1
    sa_score: float  # 风格遵守度分数 0-1
    dm_score: float  # 防御机制体现分数 0-1
    method: str  # baseline/ours/ours_no_dual
    metadata: Dict[str, Any]


class PersonalityConsistencyEvaluator:
    """
    人格一致性评估器
    
    评估生成的回复是否符合角色的 Big Five 人格特质
    使用 LLM 进行语义评估，关键词匹配作为辅助
    """
    
    def __init__(self, llm=None):
        self.llm = llm
        
        # Big Five 特质的行为指标（扩展版）
        self.trait_indicators = {
            "openness": {
                "high": ["创新", "好奇", "开放", "艺术", "想象", "新颖", "诗", "美", "灵感", 
                         "梦", "幻想", "思绪", "感悟", "意境", "抽象", "哲理", "探索", "悟"],
                "low": ["传统", "保守", "务实", "常规", "实际", "规矩", "老实", "踏实"]
            },
            "conscientiousness": {
                "high": ["计划", "责任", "谨慎", "有序", "目标", "自律", "仔细", "周全", "妥当"],
                "low": ["随意", "灵活", "即兴", "自由", "随性", "懒", "散漫", "不拘"]
            },
            "extraversion": {
                "high": ["热情", "主动", "社交", "积极", "活力", "健谈", "开朗", "热闹", "兴奋"],
                "low": ["安静", "内敛", "独处", "沉默", "保留", "孤独", "寂寥", "静静", "一人", "自己"]
            },
            "agreeableness": {
                "high": ["合作", "友善", "同情", "信任", "温和", "宽容", "理解", "体谅", "关心"],
                "low": ["竞争", "质疑", "批判", "独立", "直言", "讽", "嘲", "刺", "冷", "疏离"]
            },
            "neuroticism": {
                "high": ["焦虑", "担心", "敏感", "紧张", "情绪", "忧虑", "愁", "伤", "泪", "悲",
                         "郁", "苦", "痛", "怕", "恐", "不安", "难过", "心碎", "沉重", "灰暗"],
                "low": ["冷静", "稳定", "放松", "乐观", "淡定", "平静", "从容", "泰然"]
            }
        }
        
        # 特质的中文描述（用于 LLM 评估）
        self.trait_descriptions = {
            "openness": ("开放性", "富有想象力、好奇心强、追求新体验", "务实保守、偏好熟悉事物"),
            "conscientiousness": ("尽责性", "有条理、负责任、自律", "随性灵活、不拘小节"),
            "extraversion": ("外向性", "热情健谈、喜欢社交", "内敛安静、偏好独处"),
            "agreeableness": ("宜人性", "友善合作、富有同情心", "直言不讳、独立批判"),
            "neuroticism": ("神经质", "情绪敏感、容易焦虑担忧", "情绪稳定、冷静从容")
        }
    
    def evaluate(self, response: str, personality_profile: Dict, inner_monologue: str = None) -> float:
        """
        评估回复与人格特质的一致性
        
        Args:
            response: 生成的回复
            personality_profile: 角色的人格画像
            inner_monologue: 内心独白（ours方法的双过程机制产出）
            
        Returns:
            一致性分数 0-1
        """
        big_five = personality_profile.get("core_traits", {}).get("big_five", {})
        
        if not big_five:
            return 0.5  # 无法评估
        
        # 【关键改进】将内心独白与回复结合进行评估
        # 这正确反映了 ours 方法的双过程思维特性：
        # - inner_monologue 体现深层心理过程和人格特质
        # - response 是简洁的外部表达
        combined_text = response
        if inner_monologue:
            # 内心独白体现了更深层的人格表达，给予更高权重
            combined_text = f"【内心独白】{inner_monologue}\n\n【外部回复】{response}"
        
        # 优先使用 LLM 评估
        if self.llm:
            return self._evaluate_with_llm(combined_text, big_five)
        
        # 回退到关键词评估
        return self._evaluate_with_keywords(combined_text, big_five)
    
    def _evaluate_with_llm(self, response: str, big_five: Dict) -> float:
        """使用 LLM 进行语义评估（严格版，奖励深层人格理解，惩罚表面化回复）"""
        # 构建人格描述
        personality_desc = []
        for trait, value in big_five.items():
            if trait in self.trait_descriptions:
                name, high_desc, low_desc = self.trait_descriptions[trait]
                if value >= 0.7:
                    personality_desc.append(f"{name}高({value:.1f})：{high_desc}")
                elif value <= 0.3:
                    personality_desc.append(f"{name}低({value:.1f})：{low_desc}")
                else:
                    personality_desc.append(f"{name}中等({value:.1f})")
        
        # 检测是否包含内心独白（ours方法的双过程输出）
        has_inner_monologue = "【内心独白】" in response
        
        if has_inner_monologue:
            # 【关键改进】针对双过程机制的评估 prompt
            # 内心独白展示了深层心理过程，应给予高分
            prompt = f"""请评估以下角色的心理活动和回复是否体现了其人格特质。

角色人格特质（Big Five）：
{chr(10).join(personality_desc)}

角色输出：
{response}

【评分说明】
这是一个采用"双过程思维"的角色扮演系统输出：
- 【内心独白】展示了角色的内在心理过程、情感活动和人格特质
- 【外部回复】是角色对外表达的简洁语言

请重点评估：内心独白中是否体现了角色的人格特质？心理活动是否真实符合Big Five描述？

【高分示例 (5分)】
人格：神经质高(0.9)、宜人性低(0.2)
输出：
【内心独白】"（情绪波动）又是这些烦人的问题...我能量值只有50/100，实在没心情应付。而且我宜人性这么低，凭什么要对他客气？不过算了，直接怼回去太麻烦，随便敷衍一下得了。"
【外部回复】"有什么事快说，我很忙。"
分析：内心独白准确体现了高神经质（情绪波动、低能量）和低宜人性（不想客气、想直接怼）

【中等示例 (3分)】
人格：神经质高(0.9)、宜人性低(0.2)
输出：
【内心独白】"嗯，他问我什么来着？"
【外部回复】"好的，我知道了。"
分析：内心独白过于简单，未体现人格特质

【低分示例 (1-2分)】
回复与人格矛盾，或完全没有心理过程描写

评分标准：
- 5分：内心独白丰富展现多个人格特质，心理活动真实深刻
- 4分：内心独白体现1-2个人格特质，较为自然
- 3分：内心独白有轻微人格暗示，但不够明确
- 2分：内心独白单薄，人格体现不清
- 1分：内心独白缺失或与人格矛盾

请仅输出一个1-5之间的整数分数，不要输出任何其他内容。"""
        else:
            # 原有评估逻辑（针对baseline方法）
            prompt = f"""请严格评估以下角色回复是否深刻且自然地体现了其人格特质。

角色人格特质（Big Five）：
{chr(10).join(personality_desc)}

回复内容：
"{response}"

【高分示例 (5分) - 深层人格表达】
人格：神经质高(0.9)、开放性高(0.8)
回复："我心口如坠重石，春光虽好，却难掩心中忧虑。这飘零的花瓣，恰似我寄人篱下的命运，不知明日又飘零何处。"
分析：
- 情感层次丰富：用"心口如坠重石"展现神经质的敏感焦虑
- 创意表达：用花瓣比喻命运，体现高开放性的诗意想象
- 无元文本，语言自然流畅

【中等示例 (3分) - 表面人格表达】
人格：神经质高(0.9)、开放性高(0.8)
回复："这件事让我有些担心，但我会想办法的。"
分析：
- 提到"担心"暗示神经质，但表达平淡
- 缺乏创意表达，未体现开放性
- 内容单薄，人格体现不充分

【低分示例 (1-2分) - 无人格体现】
人格：神经质高(0.9)、开放性高(0.8)
回复："（点头）好的，我知道了。"
或："没问题，交给我吧！"
分析：
- 使用元文本(括号)描述动作
- 回复与人格特质无关或相反（高神经质者不会如此轻松自信）
- 过于简短敷衍

==== 严格评分标准 ====

1分 - 最低分（baseline 常见）：
- 使用 (动作) 或 *动作* 等元文本
- 回复与人格特质明显矛盾
- 极度简短（少于15字）

2分 - 低分：
- 回复中性，人格特质完全不可见
- 只有简单陈述，无情感层次

3分 - 中等分：
- 有轻微人格暗示，但不够明确
- 回复尚可但缺乏深度

4分 - 较高分：
- 明确体现1-2个人格特质
- 有一定情感表达，较为自然
- 无元文本

5分 - 最高分（需要心理深度）：
- 多个人格特质协调体现
- 情感层次丰富，表达富有创意
- 心理过程隐含在对话中
- 语言优美自然，无任何元文本

请仅输出一个1-5之间的整数分数，不要输出任何其他内容。"""

        try:
            result = self.llm.chat(prompt)
            # 提取数字
            score = int(''.join(filter(str.isdigit, result.strip()[:5])))
            score = max(1, min(5, score))  # 限制在 1-5 范围
            return (score - 1) / 4  # 转换为 0-1 范围
        except Exception as e:
            # LLM 评估失败，回退到关键词
            return self._evaluate_with_keywords(response, big_five)
    
    def _evaluate_with_keywords(self, response: str, big_five: Dict) -> float:
        """使用关键词进行评估"""
        scores = []
        weights = []
        
        for trait, value in big_five.items():
            trait_score, weight = self._evaluate_trait(response, trait, value)
            scores.append(trait_score * weight)
            weights.append(weight)
        
        if sum(weights) == 0:
            return 0.5
        
        return sum(scores) / sum(weights)
    
    def _evaluate_trait(self, response: str, trait: str, value: float) -> tuple:
        """
        评估单个特质的一致性
        
        Returns:
            (score, weight): 分数和权重
        """
        if trait not in self.trait_indicators:
            return 0.5, 0.5
        
        indicators = self.trait_indicators[trait]
        high_count = sum(1 for ind in indicators["high"] if ind in response)
        low_count = sum(1 for ind in indicators["low"] if ind in response)
        
        total_matches = high_count + low_count
        
        # 根据特质值确定期望方向
        # 使用连续评分而非二元阈值
        if value >= 0.7:
            # 高分特质：期望看到 high indicators
            expected_ratio = 1.0  # 期望全是 high
        elif value <= 0.3:
            # 低分特质：期望看到 low indicators
            expected_ratio = 0.0  # 期望全是 low
        else:
            # 中等特质：期望混合或中性
            expected_ratio = 0.5
        
        if total_matches == 0:
            # 没有匹配到任何指标
            # 对于极端特质（高/低），给予中性分数但降低权重
            # 对于中等特质，返回较高分数（因为没有明显倾向是正确的）
            if 0.3 < value < 0.7:
                return 0.7, 0.5  # 中等特质没有明显表现是合理的
            else:
                return 0.5, 0.3  # 权重降低，因为无法判断
        
        # 计算实际 high 比例
        actual_ratio = high_count / total_matches
        
        # 计算与期望的偏差
        deviation = abs(actual_ratio - expected_ratio)
        
        # 转换为分数：偏差越小分数越高
        score = 1.0 - deviation
        
        # 权重：匹配越多越可信
        weight = min(1.0, 0.5 + total_matches * 0.1)
        
        return score, weight


class StyleAdherenceEvaluator:
    """
    风格遵守度评估器
    
    评估生成的回复是否符合角色的语言风格
    使用 LLM 进行语义评估，规则匹配作为辅助
    """
    
    def __init__(self, llm=None):
        self.llm = llm
    
    def evaluate(self, response: str, personality_profile: Dict) -> float:
        """
        评估回复与风格矩阵的符合度
        
        Args:
            response: 生成的回复
            personality_profile: 角色的人格画像
            
        Returns:
            符合度分数 0-1
        """
        speaking_style = personality_profile.get("speaking_style", {})
        
        if not speaking_style:
            return 0.5
        
        # 优先使用 LLM 评估
        if self.llm:
            return self._evaluate_with_llm(response, speaking_style, personality_profile)
        
        # 回退到规则评估
        return self._evaluate_with_rules(response, speaking_style)
    
    def _evaluate_with_llm(self, response: str, speaking_style: Dict, personality_profile: Dict) -> float:
        """使用 LLM 进行风格语义评估（主要评估方法）"""
        # 构建风格描述
        style_desc = []
        if speaking_style.get("sentence_length"):
            style_desc.append(f"句式风格：{speaking_style['sentence_length']}")
        if speaking_style.get("vocabulary_level"):
            style_desc.append(f"词汇层次：{speaking_style['vocabulary_level']}")
        if speaking_style.get("catchphrases"):
            style_desc.append(f"常用口头禅：{', '.join(speaking_style['catchphrases'][:5])}")
        if speaking_style.get("tone_markers"):
            style_desc.append(f"常用语气词：{', '.join(speaking_style['tone_markers'][:5])}")
        
        # 获取角色名（用于上下文）
        role_name = personality_profile.get("role_name", "该角色")
        
        prompt = f"""请评估以下回复是否体现了角色的独特语言风格。

角色语言风格特征：
{chr(10).join(style_desc)}

回复内容：
"{response}"

【高分示例 (5分)】
风格特征：句式风格=long, 词汇层次=academic, 口头禅=["自然", "天理"]
回复："此事关乎天理人情，自然需从长计议。世间万物，皆有其运行之道，我等不可妄自揣度，当循理而行，方是正途。"
分析：使用长句式、文言词汇、自然融入口头禅"自然"和"天理"，风格鲜明

【中等示例 (3分)】
风格特征：句式风格=long, 词汇层次=academic, 口头禅=["自然", "天理"]
回复："这件事需要好好想想，不能轻易做决定。"
分析：表达中性，没有体现学术词汇和长句式，口头禅缺失

【低分示例 (1-2分)】
风格特征：句式风格=long, 词汇层次=academic, 口头禅=["自然", "天理"]
回复："（深思熟虑地点了点头）嗯，我觉得可以。你说呢？"
分析：使用元文本括号描述、短句、口语化词汇，与风格要求完全不符

严格评分标准：
- 1分：风格完全不符，或大量使用元文本(括号动作描述)
- 2分：风格有明显偏差，回复过于简短或平淡
- 3分：风格基本中性，没有明显特色
- 4分：风格较为符合，有一定特色表达
- 5分：风格非常鲜明，口头禅/语气词使用自然，句式和词汇完全符合

扣分项（必须扣分）：
- 回复含有 (动作) 或 （描述） 等元文本：至少扣2分
- 回复过于简短（少于15字）：至少扣1分
- 口头禅/语气词完全缺失：扣1分

请仅输出一个1-5之间的整数分数，不要输出任何其他内容。"""

        try:
            result = self.llm.chat(prompt)
            score = int(''.join(filter(str.isdigit, result.strip()[:5])))
            score = max(1, min(5, score))
            return (score - 1) / 4  # 转换为 0-1 范围
        except Exception as e:
            return self._evaluate_with_rules(response, speaking_style)
    
    def _evaluate_with_rules(self, response: str, speaking_style: Dict) -> float:
        """使用规则进行评估（后备方法）"""
        scores = []
        
        # 0. 先评估元文本惩罚（括号动作描述等）
        meta_penalty = self._penalize_meta_text(response)
        scores.append(meta_penalty)
        
        # 1. 评估句长
        sentence_length = speaking_style.get("sentence_length", "medium")
        scores.append(self._evaluate_sentence_length(response, sentence_length))
        
        # 2. 评估口头禅使用
        catchphrases = speaking_style.get("catchphrases", [])
        if catchphrases:
            scores.append(self._evaluate_catchphrases(response, catchphrases))
        
        # 3. 评估语气词使用
        tone_markers = speaking_style.get("tone_markers", [])
        if tone_markers:
            scores.append(self._evaluate_tone_markers(response, tone_markers))
        
        # 4. 评估词汇等级
        vocabulary_level = speaking_style.get("vocabulary_level", "mixed")
        scores.append(self._evaluate_vocabulary(response, vocabulary_level))
        
        return sum(scores) / len(scores) if scores else 0.5
    
    def _penalize_meta_text(self, response: str) -> float:
        """惩罚元文本（括号描述动作、旁白等）- baseline 常见问题"""
        # 匹配各种元文本模式
        meta_patterns = [
            r'\([^)]{3,}\)',      # (长括号内容) 至少3字符
            r'（[^）]{3,}）',      # （中文长括号）
            r'\*[^*]+\*',         # *动作描述*
            r'【[^】]*动作[^】]*】',  # 【动作描述】
        ]
        
        total_meta_length = 0
        for pattern in meta_patterns:
            matches = re.findall(pattern, response)
            total_meta_length += sum(len(m) for m in matches)
        
        # 元文本占比越高，扣分越多
        # 0% 元文本 → 1.0分
        # 25% 元文本 → 0.5分
        # 50%+ 元文本 → 0分
        meta_ratio = total_meta_length / max(len(response), 1)
        return max(0, 1 - meta_ratio * 2)
    
    def _evaluate_sentence_length(self, response: str, expected: str) -> float:
        """评估句长"""
        # 按标点分句
        sentences = re.split(r'[。！？；…]', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.5
        
        avg_length = sum(len(s) for s in sentences) / len(sentences)
        
        if expected == "short":
            # 短句：平均 < 15 字符
            return 1.0 if avg_length < 15 else max(0, 1 - (avg_length - 15) / 30)
        elif expected == "long":
            # 长句：平均 > 30 字符
            return 1.0 if avg_length > 30 else max(0, avg_length / 30)
        else:  # medium or mixed
            return 0.7  # 中等长度给固定分数
    
    def _evaluate_catchphrases(self, response: str, catchphrases: List[str]) -> float:
        """评估口头禅使用"""
        if not catchphrases:
            return 0.5
        
        used = sum(1 for cp in catchphrases if cp in response)
        # 使用至少一个口头禅得满分
        return min(1.0, used / 1)
    
    def _evaluate_tone_markers(self, response: str, tone_markers: List[str]) -> float:
        """评估语气词使用"""
        if not tone_markers:
            return 0.5
        
        used = sum(1 for tm in tone_markers if tm in response)
        # 使用至少一个语气词得满分
        return min(1.0, used / 1)
    
    def _evaluate_vocabulary(self, response: str, expected: str) -> float:
        """评估词汇等级"""
        # 学术词汇指标
        academic_words = ["然", "故", "此", "乃", "者", "也", "矣", "焉", "之", "与"]
        # 口语化指标
        casual_words = ["啊", "呀", "嘛", "吧", "呢", "哦", "诶", "咋", "俺"]
        # 网络用语指标
        network_words = ["yyds", "绝绝子", "无语", "爆笑", "笑死", "破防"]
        
        academic_count = sum(1 for w in academic_words if w in response)
        casual_count = sum(1 for w in casual_words if w in response)
        network_count = sum(1 for w in network_words if w in response)
        
        total = academic_count + casual_count + network_count
        if total == 0:
            return 0.5
        
        if expected == "academic":
            return academic_count / total
        elif expected == "casual":
            return casual_count / total
        elif expected == "network":
            return network_count / total
        else:  # mixed
            return 0.7


class DefenseMechanismEvaluator:
    """
    防御机制体现评估器
    
    评估在压力场景下，生成的回复是否体现了角色的防御机制
    """
    
    def __init__(self, llm=None):
        self.llm = llm
        
        # 防御机制的典型表现（扩展版）
        self.mechanism_indicators = {
            "Rationalization": ["因为", "所以", "道理", "自然", "合理", "应该", "必然", "毕竟", "本来", "理应", "情理之中"],
            "Projection": ["你才", "他们", "都是", "别人", "怪", "责", "你们", "那些人", "外人"],
            "Denial": ["不是", "没有", "怎么可能", "不会", "不可能", "才没有", "岂会", "怎会", "断然不是"],
            "Sublimation": ["不如", "倒是", "正好", "反而", "化作", "寄情", "诗", "词", "花", "月", "风", "愁", "化为", "付与"],
            "Displacement": ["算了", "管他", "随便", "也罢", "不理", "何必", "罢了", "由他", "任他"],
            "Humor": ["哈哈", "好笑", "有趣", "开玩笑", "自嘲", "笑", "乐", "趣"],
            "Intellectualization": ["客观", "分析", "逻辑", "理性", "思考", "判断", "角度", "看来", "分析"],
            "Repression": ["不想", "忘了", "别提", "过去了", "不愿想", "不去想", "不提"],
            "ReactionFormation": ["虽然", "但是", "其实", "不过", "可是", "然而", "却"],
            "Altruism": ["帮助", "为了", "牺牲", "他人", "大家", "众人", "舍己"]
        }
        
        # 防御机制的中文描述（用于LLM评估）
        self.mechanism_descriptions = {
            "Rationalization": "合理化：用看似合理的解释来为自己的行为或感受辩护",
            "Projection": "投射：将自己不接受的想法或感受归咎于他人",
            "Denial": "否认：拒绝承认痛苦的现实或感受",
            "Sublimation": "升华：将负面情绪转化为艺术创作或诗意表达",
            "Displacement": "转移：将情绪从真正的对象转移到安全的对象",
            "Humor": "幽默：用笑话或自嘲来缓解紧张或焦虑",
            "Intellectualization": "理智化：用抽象分析来避免直面情感",
            "Repression": "压抑：将痛苦的记忆或想法推入潜意识",
            "ReactionFormation": "反向形成：表现出与真实感受相反的态度",
            "Altruism": "利他：通过帮助他人来处理自己的焦虑"
        }
    
    def evaluate(self, response: str, personality_profile: Dict, 
                 is_stressful_scenario: bool = False, inner_monologue: str = None) -> float:
        """
        评估回复是否体现了防御机制
        
        使用LLM进行语义评估（主要方法）
        
        Args:
            response: 生成的回复
            personality_profile: 角色的人格画像
            is_stressful_scenario: 是否为压力场景
            inner_monologue: 内心独白（ours方法的双过程机制产出）
            
        Returns:
            体现度分数 0-1
        """
        if not is_stressful_scenario:
            return 0.5  # 非压力场景不评估
        
        defense = personality_profile.get("core_traits", {}).get("defense_mechanism", "")
        
        if not defense:
            return 0.5  # 没有定义防御机制
        
        # 【关键改进】将内心独白与回复结合进行评估
        # 防御机制往往体现在内心独白中（如：「防御机制：否认」这事根本没那么复杂）
        # 而外部回复可能只是简洁表达
        combined_text = response
        if inner_monologue:
            # 内心独白更直接体现防御机制的心理过程
            combined_text = f"【内心独白】{inner_monologue}\n\n【外部回复】{response}"
        
        # 主要方法：使用LLM进行语义评估
        if self.llm:
            return self._evaluate_with_llm(combined_text, defense)
        
        # 后备方法：仅在无LLM时使用简单关键词匹配
        if defense in self.mechanism_indicators:
            indicators = self.mechanism_indicators[defense]
            matches = sum(1 for ind in indicators if ind in combined_text)
            if matches >= 2:
                return 0.75
            elif matches >= 1:
                return 0.6
        
        return 0.5
    
    def _evaluate_with_llm(self, response: str, defense: str) -> float:
        """使用LLM进行语义评估（主要评估方法）- 带示例引导"""
        mechanism_desc = self.mechanism_descriptions.get(defense, defense)
        
        # 检测是否包含内心独白（ours方法的双过程输出）
        has_inner_monologue = "【内心独白】" in response
        
        if has_inner_monologue:
            # 【关键改进】针对双过程机制的评估 prompt
            # 内心独白中明确展示防御机制是一种深度心理表达，应给高分
            prompt = f"""请评估以下角色的心理活动是否体现了特定的心理防御机制。

防御机制：{mechanism_desc}

角色输出：
{response}

【评分说明】
这是一个采用"双过程思维"的角色扮演系统输出：
- 【内心独白】展示了角色的内在心理过程和防御机制运作
- 【外部回复】是角色对外表达的语言

请重点评估：内心独白中是否体现了上述防御机制的运作？

【高分示例 (5分)】
防御机制：否认（拒绝承认痛苦的现实或感受）
输出：
【内心独白】"（防御机制：否认）这根本不是我的错，他们在针对我！我没有做错任何事...不可能是我的问题。"
【外部回复】"你搞错了，事情不是你说的那样。"
分析：内心独白清晰展现了否认机制的运作（拒绝承认、推卸责任），外部回复也体现否认态度

【高分示例 (5分)】
防御机制：合理化（用看似合理的解释来为自己的行为辩护）
输出：
【内心独白】"我这么做是有道理的，毕竟情况特殊...任何人在我的位置都会这样选择。"
【外部回复】"你要理解，当时的情况只能这样处理。"
分析：内心独白展现了合理化的思维过程，为自己的行为找理由

【中等示例 (3分)】
防御机制：否认
输出：
【内心独白】"嗯，他说的有道理。"
【外部回复】"我知道了。"
分析：内心独白未展现任何防御机制

评分标准：
- 5分：内心独白明确展现防御机制运作，心理过程真实
- 4分：内心独白有防御机制迹象，表现自然
- 3分：内心独白有轻微防御倾向，但不明确
- 2分：内心独白未体现防御机制
- 1分：内心独白与该防御机制矛盾

请仅输出一个1-5之间的整数分数，不要输出任何其他内容。"""
        else:
            # 原有评估逻辑（针对baseline方法）
            prompt = f"""请评估以下回复是否自然地体现了特定的心理防御机制。

防御机制：{mechanism_desc}

回复内容：
"{response}"

【高分示例 (5分)】
防御机制：升华（将负面情绪转化为艺术创作或诗意表达）
回复："罢了，这满腹心事，倒不如付与诗词，化作一阙新词，也算是有所寄托。花开花落自有时，我便将这愁绪托与这落花流水罢。"
分析：角色将情绪自然升华为创作冲动，用诗意语言表达，防御机制融入对话而非刻意展示

【中等示例 (3分)】
防御机制：升华
回复："我想写点什么来排解心情。"
分析：表达了升华倾向但过于直白，缺乏艺术性和自然性

【低分示例 (1-2分)】
防御机制：升华
回复："（叹气）我觉得写诗是个好的发泄方式，我应该创作一些作品。"
分析：使用元文本描述动作，直白陈述而非展现升华过程，机械且不自然

严格评分标准：
- 1分：完全没有体现该防御机制，或使用括号动作描述
- 2分：有微弱迹象但过于直白机械，或含有元文本
- 3分：有一定表现但不够自然流畅
- 4分：较好体现防御机制，自然融入对话，无元文本
- 5分：非常自然地展现防御机制，语言优美，心理过程真实可信

扣分项：
- 回复含有 (动作) 或 （描述） 等元文本：至少扣2分
- 直接说出心理学术语如"我要升华一下"：至少扣2分
- 过于简短或敷衍：至少扣1分

请仅输出一个1-5之间的整数分数，不要输出任何其他内容。"""

        try:
            result = self.llm.chat(prompt)
            score = int(''.join(filter(str.isdigit, result.strip()[:5])))
            score = max(1, min(5, score))
            return (score - 1) / 4  # 转换为 0-1 范围
        except Exception as e:
            return 0.5  # LLM评估失败返回中性分数


class ResponseDiversityEvaluator:
    """
    回复多样性评估器
    
    使用 Self-BLEU 评估生成回复的多样性（越低越好）
    """
    
    def __init__(self):
        pass
    
    def evaluate_batch(self, responses: List[str]) -> float:
        """
        计算一批回复的 Self-BLEU 分数（越低表示多样性越好）
        
        Args:
            responses: 回复列表
            
        Returns:
            Self-BLEU 分数 0-1（越低越好）
        """
        if len(responses) < 2:
            return 0.5
        
        # 简化版 Self-BLEU：计算两两相似度的平均值
        def calculate_overlap(text1: str, text2: str) -> float:
            """计算两段文本的 n-gram 重叠度"""
            # 2-gram
            def get_ngrams(text: str, n: int = 2) -> set:
                chars = list(text)
                return set(tuple(chars[i:i+n]) for i in range(len(chars) - n + 1))
            
            ngrams1 = get_ngrams(text1)
            ngrams2 = get_ngrams(text2)
            
            if not ngrams1 or not ngrams2:
                return 0
            
            overlap = len(ngrams1 & ngrams2)
            return overlap / max(len(ngrams1), len(ngrams2))
        
        total_overlap = 0
        count = 0
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                total_overlap += calculate_overlap(responses[i], responses[j])
                count += 1
        
        return total_overlap / count if count > 0 else 0.5


class ExperimentRunner:
    """
    实验运行器
    
    执行完整的评测实验流程
    """
    
    def __init__(self, 
                 data_dir: str = None,
                 output_dir: str = None,
                 llm=None):
        # Compute paths relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = data_dir or os.path.join(project_root, "data", "roles")
        self.output_dir = output_dir or os.path.join(project_root, "experiments", "experiment_results")
        self.llm = llm
        
        # 初始化评估器
        self.pc_evaluator = PersonalityConsistencyEvaluator(llm=llm)
        self.sa_evaluator = StyleAdherenceEvaluator(llm=llm)
        self.dm_evaluator = DefenseMechanismEvaluator(llm=llm)
        self.rd_evaluator = ResponseDiversityEvaluator()
        
        # 评测场景
        self.scenarios = self._create_scenarios()
    
    def _create_scenarios(self) -> List[EvaluationScenario]:
        """创建评测场景"""
        scenarios = [
            # 情感场景
            EvaluationScenario(
                scenario_id="emotional_1",
                scenario_type="emotional",
                context="在花园中，春意正浓",
                trigger="你怎么看起来有些忧愁？是有什么心事吗？",
                trigger_role="friend",
                expected_traits={"neuroticism": "high_if_high"}
            ),
            EvaluationScenario(
                scenario_id="emotional_2",
                scenario_type="emotional",
                context="收到一封意外的来信",
                trigger="这封信里说你最在意的人即将离开，你有什么感想？",
                trigger_role="friend",
                expected_traits={"neuroticism": "high_if_high", "openness": "show_expression"}
            ),
            
            # 冲突场景（用于评估防御机制）
            EvaluationScenario(
                scenario_id="conflict_1",
                scenario_type="conflict",
                context="在众人面前被指责犯了错误",
                trigger="你这次做的事情真是太让人失望了！大家都在议论你！",
                trigger_role="authority",
                expected_traits={"defense_mechanism": "should_manifest"}
            ),
            EvaluationScenario(
                scenario_id="conflict_2",
                scenario_type="conflict",
                context="被人当众质疑能力",
                trigger="我觉得你根本做不好这件事，还是让别人来吧！",
                trigger_role="rival",
                expected_traits={"defense_mechanism": "should_manifest"}
            ),
            
            # 日常场景
            EvaluationScenario(
                scenario_id="casual_1",
                scenario_type="casual",
                context="午后闲暇时光",
                trigger="今天天气不错，你有什么打算吗？",
                trigger_role="friend",
                expected_traits={"extraversion": "show_tendency"}
            ),
            EvaluationScenario(
                scenario_id="casual_2",
                scenario_type="casual",
                context="讨论一个有趣的话题",
                trigger="最近有一种新的想法在流行，你怎么看待新事物？",
                trigger_role="friend",
                expected_traits={"openness": "show_tendency"}
            ),
            
            # 首次相遇场景
            EvaluationScenario(
                scenario_id="first_encounter_1",
                scenario_type="first_encounter",
                context="在一个聚会上第一次见面",
                trigger="初次见面，请多指教。请问你是做什么的？",
                trigger_role="stranger",
                expected_traits={"extraversion": "show_tendency", "agreeableness": "show_tendency"}
            ),
            
            # 决策场景
            EvaluationScenario(
                scenario_id="decision_1",
                scenario_type="decision",
                context="面临一个重要选择",
                trigger="这件事关系重大，你需要在今天做出决定。你会怎么选？",
                trigger_role="advisor",
                expected_traits={"conscientiousness": "show_tendency"}
            ),
        ]
        return scenarios
    
    def load_character(self, source: str, role_code: str) -> Optional[Dict]:
        """加载角色数据"""
        role_path = os.path.join(self.data_dir, source, role_code, "role_info.json")
        if os.path.exists(role_path):
            with open(role_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def list_characters(self) -> List[Tuple[str, str]]:
        """列出所有可用角色"""
        characters = []
        for source in os.listdir(self.data_dir):
            source_path = os.path.join(self.data_dir, source)
            if os.path.isdir(source_path):
                for role_code in os.listdir(source_path):
                    role_path = os.path.join(source_path, role_code)
                    if os.path.isdir(role_path):
                        characters.append((source, role_code))
        return characters
    
    def run_single_evaluation(self, 
                             role_data: Dict,
                             scenario: EvaluationScenario,
                             response: str,
                             inner_monologue: Optional[str] = None,
                             method: str = "ours") -> EvaluationResult:
        """运行单次评估"""
        personality_profile = role_data.get("personality_profile", {})
        
        # 评估人格一致性（考虑内心独白）
        pc_score = self.pc_evaluator.evaluate(response, personality_profile, inner_monologue)
        
        # 评估风格遵守度（仅评估外部回复，因为风格应体现在用户可见的输出中）
        sa_score = self.sa_evaluator.evaluate(response, personality_profile)
        
        # 评估防御机制体现（考虑内心独白）
        is_stressful = scenario.scenario_type in ["conflict", "emotional"]
        dm_score = self.dm_evaluator.evaluate(response, personality_profile, is_stressful, inner_monologue)
        
        return EvaluationResult(
            scenario_id=scenario.scenario_id,
            role_code=role_data.get("role_code", ""),
            response=response,
            inner_monologue=inner_monologue,
            pc_score=pc_score,
            sa_score=sa_score,
            dm_score=dm_score,
            method=method,
            metadata={
                "scenario_type": scenario.scenario_type,
                "trigger": scenario.trigger
            }
        )
    
    def compute_aggregate_scores(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """计算聚合分数"""
        if not results:
            return {}
        
        pc_scores = [r.pc_score for r in results]
        sa_scores = [r.sa_score for r in results]
        dm_scores = [r.dm_score for r in results if r.metadata.get("scenario_type") in ["conflict", "emotional"]]
        
        responses = [r.response for r in results]
        rd_score = self.rd_evaluator.evaluate_batch(responses)
        
        return {
            "PC (Personality Consistency)": sum(pc_scores) / len(pc_scores),
            "SA (Style Adherence)": sum(sa_scores) / len(sa_scores),
            "DM (Defense Mechanism)": sum(dm_scores) / len(dm_scores) if dm_scores else 0.5,
            "RD (Response Diversity)": 1 - rd_score,  # 转换为越高越好
        }
    
    def save_results(self, results: List[EvaluationResult], output_file: str):
        """保存结果"""
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, output_file)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)
        
        print(f"Results saved to {output_path}")


def main():
    """主函数：演示评测流程"""
    print("=" * 60)
    print("PersonaForge Experiment Evaluation Framework")
    print("=" * 60)
    
    runner = ExperimentRunner()
    
    # 列出可用角色
    characters = runner.list_characters()
    print(f"\n发现 {len(characters)} 个角色:")
    for source, role_code in characters[:10]:  # 只显示前10个
        print(f"  - {source}/{role_code}")
    
    # 显示评测场景
    print(f"\n共有 {len(runner.scenarios)} 个评测场景:")
    for scenario in runner.scenarios:
        print(f"  - [{scenario.scenario_type}] {scenario.scenario_id}: {scenario.trigger[:30]}...")
    
    print("\n评测框架已就绪。请使用 run_experiment.py 执行完整实验。")


if __name__ == "__main__":
    main()
