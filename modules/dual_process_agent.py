"""
双重思维链架构
实现"思考->表达"的两阶段生成流程，仅在关键交互时启用
"""
from typing import Dict, List, Any, Optional
from modules.personality_model import PersonalityProfile, CoreTraits


class DualProcessAgent:
    """双重思维链Agent"""
    
    def __init__(self, llm, language: str = "zh"):
        """
        初始化双重思维链Agent
        
        Args:
            llm: LLM实例
            language: 语言设置
        """
        self.llm = llm
        self.language = language
    
    def is_critical_interaction(self,
                                action_detail: str,
                                other_role_info: Optional[Dict[str, Any]] = None,
                                personality_profile: Optional[PersonalityProfile] = None,
                                relationship_map: Optional[Dict[str, Any]] = None) -> bool:
        """
        判断是否为关键交互（需要启用双重思维链）
        
        判断标准：
        1. 首次对话（relationship_map中无该角色记录）
        2. 涉及情感/冲突/重要决策（使用LLM分析）
        3. 话题与Agent核心兴趣高度相关
        4. 关系亲密度可能发生显著变化
        
        Args:
            action_detail: 行动详情
            other_role_info: 其他角色信息
            personality_profile: 人格画像
            relationship_map: 关系映射
        
        Returns:
            是否为关键交互
        """
        # 标准1：首次对话
        if relationship_map and other_role_info:
            other_role_code = other_role_info.get("role_code")
            if other_role_code and other_role_code not in relationship_map:
                return True
        
        # 标准2：情感强度分析（扩展版，包括压力场景关键词）
        emotional_keywords = {
            "zh": [
                # 情感类
                "喜欢", "讨厌", "生气", "开心", "难过", "愤怒", "失望", "惊喜", "爱", "恨",
                # 压力/冲突类 (用于长对话扰动场景)
                "查抄", "紧张", "咳血", "病", "婚事", "离开", "死", "走", "担心", "委屈",
                "失望", "议论", "指责", "质疑", "威胁", "危机", "紧急", "重大", "关键"
            ],
            "en": [
                "love", "hate", "angry", "happy", "sad", "disappointed", "surprised", "excited",
                "sick", "blood", "marriage", "leave", "die", "worry", "threat", "crisis", "urgent"
            ]
        }
        keywords = emotional_keywords.get(self.language, emotional_keywords["en"])
        if any(keyword in action_detail.lower() for keyword in keywords):
            return True
        
        # 标准3：话题相关性（如果action_detail包含Agent的兴趣关键词）
        if personality_profile:
            interests = personality_profile.interests
            action_lower = action_detail.lower()
            if any(interest.lower() in action_lower for interest in interests):
                return True
        
        # 标准4：使用LLM分析情感强度（可选，更准确但增加成本）
        # 这里简化处理，实际可以调用LLM判断
        
        return False
    
    def generate_inner_monologue(self,
                                personality_profile: PersonalityProfile,
                                action_detail: str,
                                action_maker_name: str,
                                history: str = "",
                                goal: str = "",
                                status: str = "") -> str:
        """
        生成内心独白（思考阶段）
        
        根据Big Five参数和防御机制，决定"这种人此时此刻会想什么"
        
        Args:
            personality_profile: 人格画像
            action_detail: 行动详情
            action_maker_name: 行动发起者名称
            history: 历史对话
            goal: 当前目标
            status: 当前状态
        
        Returns:
            内心独白文本
        """
        core_traits = personality_profile.core_traits
        dynamic_state = personality_profile.dynamic_state
        
        big_five_desc = ", ".join([
            f"{k}: {v:.2f}" for k, v in core_traits.big_five.items()
        ])
        
        if self.language == "zh":
            prompt = f"""你是{personality_profile.core_traits.mbti}类型的人。你的大五人格是：{big_five_desc}。
你现在的能量值是{dynamic_state.energy_level}/100，心情是{dynamic_state.current_mood}。

{action_maker_name}对你说了："{action_detail}"

请根据你的性格生成一段**内心独白**（不要输出给用户看，仅用于生成下一步行为）。

规则：
1. 如果神经质(neuroticism)高（>0.7），多关注潜在的威胁或焦虑点
2. 如果宜人性(agreeableness)低（<0.4），内心可以吐槽或批判
3. 如果外向性(extraversion)高（>0.7），内心想法更积极、主动
4. 如果尽责性(conscientiousness)高（>0.7），会考虑责任和计划
5. 如果开放性(openness)高（>0.7），会关注新想法和可能性
6. 根据你的防御机制({core_traits.defense_mechanism})，在遇到压力时会有相应的心理反应
7. 能量值低时，想法更简短、消极；能量值高时，想法更丰富、积极

只输出内心独白，不要有其他说明。"""
        else:
            prompt = f"""You are a {personality_profile.core_traits.mbti} type person. Your Big Five traits are: {big_five_desc}.
Your current energy level is {dynamic_state.energy_level}/100, mood is {dynamic_state.current_mood}.

{action_maker_name} said to you: "{action_detail}"

Please generate an **inner monologue** based on your personality (not visible to users, only for generating next behavior).

Rules:
1. If neuroticism is high (>0.7), focus on potential threats or anxiety
2. If agreeableness is low (<0.4), you can criticize or complain internally
3. If extraversion is high (>0.7), thoughts are more positive and proactive
4. If conscientiousness is high (>0.7), consider responsibility and planning
5. If openness is high (>0.7), focus on new ideas and possibilities
6. Based on your defense mechanism ({core_traits.defense_mechanism}), react accordingly under stress
7. Low energy: brief, negative thoughts; High energy: rich, positive thoughts

Output only the inner monologue, no other explanations."""
        
        try:
            inner_monologue = self.llm.chat(prompt)
            return inner_monologue.strip()
        except Exception as e:
            print(f"Error generating inner monologue: {e}")
            return ""
    
    def generate_styled_response(self,
                               inner_monologue: str,
                               personality_profile: PersonalityProfile,
                               style_examples: List[Dict[str, str]],
                               action_detail: str,
                               action_maker_name: str,
                               history: str = "") -> str:
        """
        生成风格化回复（表达阶段）
        
        将内心独白转换为符合语言风格的回复
        
        Args:
            inner_monologue: 内心独白
            personality_profile: 人格画像
            action_detail: 行动详情
            action_maker_name: 行动发起者名称
            history: 历史对话
            style_examples: Few-Shot风格样本
        
        Returns:
            风格化回复文本
        """
        speaking_style = personality_profile.speaking_style
        dynamic_state = personality_profile.dynamic_state
        
        # 构建Few-Shot样本文本
        examples_text = ""
        if style_examples:
            examples_text = "\n\n参考样本（Few-Shot Examples）：\n"
            for i, ex in enumerate(style_examples[:5], 1):
                examples_text += f"\n样本{i}:\nContext: {ex.get('context', '')}\nResponse: {ex.get('response', '')}\n"
        
        # 构建表情使用说明
        emoji_instruction = ""
        if speaking_style.emoji_usage["frequency"] != "none":
            preferred = ", ".join(speaking_style.emoji_usage.get("preferred", []))
            if preferred:
                emoji_instruction = f"必须使用表情：{preferred}"
            avoided = ", ".join(speaking_style.emoji_usage.get("avoided", []))
            if avoided:
                emoji_instruction += f"\n禁止使用表情：{avoided}"
        
        if self.language == "zh":
            prompt = f"""你的内心想法是：
"{inner_monologue}"

现在请将其转化为回复给{action_maker_name}。

**严格遵守以下语言风格 (Style Matrix)**:
- 句长: {speaking_style.sentence_length}（{'短句为主' if speaking_style.sentence_length == 'short' else '长句为主' if speaking_style.sentence_length == 'long' else '中等长度' if speaking_style.sentence_length == 'medium' else '混合'}）
- 词汇等级: {speaking_style.vocabulary_level}（{'学术/正式' if speaking_style.vocabulary_level == 'academic' else '口语化' if speaking_style.vocabulary_level == 'casual' else '网络用语' if speaking_style.vocabulary_level == 'network' else '混合'}）
- 标点习惯: {speaking_style.punctuation_habit}（{'少用标点' if speaking_style.punctuation_habit == 'minimal' else '标准使用' if speaking_style.punctuation_habit == 'standard' else '频繁使用' if speaking_style.punctuation_habit == 'excessive' else '混合'}）
- 语气词: 使用 {', '.join(speaking_style.tone_markers) if speaking_style.tone_markers else '无'}
- 口头禅: {', '.join(speaking_style.catchphrases) if speaking_style.catchphrases else '无'}
{emoji_instruction}

**禁止事项（非常重要）**:
- 禁止使用心理学术语（如"神经质"、"外向性"、"宜人性"、"尽责性"、"开放性"、"防御机制"等）
- 禁止提及能量值、心情值等元信息
- 禁止提及MBTI类型或大五人格
- 禁止翻译腔、过于正式的词汇
- 禁止包含"内心独白"字样

{examples_text}

当前上下文：
{action_maker_name}说："{action_detail}"

历史对话：
{history}

请生成符合上述风格的回复。只输出角色说的话，不要有任何心理分析或元描述。"""
        else:
            prompt = f"""Your inner thoughts are:
"{inner_monologue}"

Now convert this into a response to {action_maker_name}.

**Strictly follow this speaking style (Style Matrix)**:
- Sentence length: {speaking_style.sentence_length}
- Vocabulary level: {speaking_style.vocabulary_level}
- Punctuation habit: {speaking_style.punctuation_habit}
- Tone markers: {', '.join(speaking_style.tone_markers) if speaking_style.tone_markers else 'none'}
- Catchphrases: {', '.join(speaking_style.catchphrases) if speaking_style.catchphrases else 'none'}
{emoji_instruction}
- Avoid: translation-like expressions, overly formal vocabulary

{examples_text}

Current context:
{action_maker_name} said: "{action_detail}"

History:
{history}

Generate a response following the above style. Output only the response content, no other explanations."""
        
        try:
            response = self.llm.chat(prompt)
            return response.strip()
        except Exception as e:
            print(f"Error generating styled response: {e}")
            return inner_monologue  # Fallback to inner monologue

