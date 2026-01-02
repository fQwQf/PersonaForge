"""
动态状态管理器
负责更新Agent的心情、能量值和关系信息
"""
from typing import Dict, List, Any, Optional
from modules.personality_model import PersonalityProfile, DynamicState, RelationshipInfo


class DynamicStateManager:
    """动态状态管理器"""
    
    def __init__(self, llm, language: str = "zh"):
        """
        初始化动态状态管理器
        
        Args:
            llm: LLM实例
            language: 语言设置
        """
        self.llm = llm
        self.language = language
    
    def update_state_after_interaction(self,
                                     personality_profile: PersonalityProfile,
                                     interaction_detail: str,
                                     other_role_code: Optional[str] = None,
                                     other_role_name: Optional[str] = None) -> Dict[str, Any]:
        """
        交互后更新动态状态
        
        Args:
            personality_profile: 人格画像
            interaction_detail: 交互详情
            other_role_code: 其他角色代码
            other_role_name: 其他角色名称
        
        Returns:
            更新后的状态信息
        """
        # 分析情感变化
        mood_change = self._analyze_mood_change(interaction_detail, personality_profile)
        
        # 分析能量变化
        energy_delta = self._analyze_energy_change(interaction_detail, personality_profile)
        
        # 更新心情
        new_mood = self._determine_new_mood(personality_profile.dynamic_state.current_mood, mood_change)
        personality_profile.dynamic_state.update_mood(new_mood)
        
        # 更新能量值
        personality_profile.dynamic_state.update_energy(energy_delta)
        
        # 更新关系信息（如果有其他角色）
        relationship_update = None
        if other_role_code:
            relationship_update = self._update_relationship(
                personality_profile,
                other_role_code,
                other_role_name,
                interaction_detail
            )
        
        return {
            "mood": new_mood,
            "energy_level": personality_profile.dynamic_state.energy_level,
            "mood_change": mood_change,
            "energy_delta": energy_delta,
            "relationship_update": relationship_update
        }
    
    def _analyze_mood_change(self, interaction_detail: str, personality_profile: PersonalityProfile) -> str:
        """
        分析心情变化
        
        Args:
            interaction_detail: 交互详情
            personality_profile: 人格画像
        
        Returns:
            心情变化描述（positive/negative/neutral）
        """
        # 简化版：基于关键词判断
        # 实际可以使用LLM进行更准确的分析
        
        positive_keywords = {
            "zh": ["开心", "高兴", "喜欢", "感谢", "赞美", "夸奖", "同意", "支持"],
            "en": ["happy", "glad", "like", "thanks", "praise", "agree", "support"]
        }
        negative_keywords = {
            "zh": ["生气", "讨厌", "拒绝", "批评", "反对", "失望", "难过"],
            "en": ["angry", "hate", "reject", "criticize", "oppose", "disappointed", "sad"]
        }
        
        keywords_pos = positive_keywords.get(self.language, positive_keywords["en"])
        keywords_neg = negative_keywords.get(self.language, negative_keywords["en"])
        
        detail_lower = interaction_detail.lower()
        
        pos_count = sum(1 for kw in keywords_pos if kw in detail_lower)
        neg_count = sum(1 for kw in keywords_neg if kw in detail_lower)
        
        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        else:
            return "neutral"
    
    def _analyze_energy_change(self, interaction_detail: str, personality_profile: PersonalityProfile) -> int:
        """
        分析能量变化
        
        Args:
            interaction_detail: 交互详情
            personality_profile: 人格画像
        
        Returns:
            能量变化值（-20到+20之间）
        """
        mood_change = self._analyze_mood_change(interaction_detail, personality_profile)
        
        # 根据心情变化调整能量
        if mood_change == "positive":
            return 10  # 积极交互增加能量
        elif mood_change == "negative":
            return -15  # 消极交互减少能量
        else:
            return -2  # 中性交互略微减少能量（消耗）
    
    def _determine_new_mood(self, current_mood: str, mood_change: str) -> str:
        """
        确定新心情
        
        Args:
            current_mood: 当前心情
            mood_change: 心情变化
        
        Returns:
            新心情
        """
        mood_map = {
            "neutral": {
                "positive": "cheerful",
                "negative": "melancholy",
                "neutral": "neutral"
            },
            "cheerful": {
                "positive": "cheerful",
                "negative": "neutral",
                "neutral": "cheerful"
            },
            "melancholy": {
                "positive": "neutral",
                "negative": "melancholy",
                "neutral": "melancholy"
            }
        }
        
        return mood_map.get(current_mood, {}).get(mood_change, "neutral")
    
    def _update_relationship(self,
                           personality_profile: PersonalityProfile,
                           other_role_code: str,
                           other_role_name: Optional[str],
                           interaction_detail: str) -> Dict[str, Any]:
        """
        更新关系信息
        
        Args:
            personality_profile: 人格画像
            other_role_code: 其他角色代码
            other_role_name: 其他角色名称
            interaction_detail: 交互详情
        
        Returns:
            关系更新信息
        """
        mood_change = self._analyze_mood_change(interaction_detail, personality_profile)
        
        # 获取当前关系
        if other_role_code not in personality_profile.dynamic_state.relationship_map:
            current_intimacy = 0
            current_history = ""
        else:
            rel_info = personality_profile.dynamic_state.relationship_map[other_role_code]
            current_intimacy = rel_info.intimacy
            current_history = rel_info.history_summary
        
        # 根据心情变化调整亲密度
        intimacy_delta = 0
        if mood_change == "positive":
            intimacy_delta = 5  # 积极交互增加亲密度
        elif mood_change == "negative":
            intimacy_delta = -3  # 消极交互减少亲密度
        
        new_intimacy = max(0, min(100, current_intimacy + intimacy_delta))
        
        # 更新历史摘要（简化版，实际可以用LLM生成更详细的摘要）
        new_history = self._update_history_summary(current_history, interaction_detail, other_role_name)
        
        # 更新关系映射
        personality_profile.dynamic_state.update_relationship(
            other_role_code,
            intimacy=new_intimacy,
            history_summary=new_history
        )
        
        return {
            "role_code": other_role_code,
            "role_name": other_role_name,
            "intimacy": new_intimacy,
            "intimacy_delta": intimacy_delta,
            "history_summary": new_history
        }
    
    def _update_history_summary(self, current_history: str, interaction_detail: str, other_role_name: Optional[str]) -> str:
        """
        更新历史摘要
        
        Args:
            current_history: 当前历史摘要
            interaction_detail: 交互详情
            other_role_name: 其他角色名称
        
        Returns:
            新的历史摘要
        """
        # 简化版：如果历史摘要为空，生成一个简单的摘要
        if not current_history:
            if self.language == "zh":
                return f"与{other_role_name or '对方'}进行了交流"
            else:
                return f"Had a conversation with {other_role_name or 'the other person'}"
        
        # 如果历史摘要已存在，可以追加或使用LLM生成更详细的摘要
        # 这里简化处理，保持原有摘要
        return current_history
    
    def get_state_summary(self, personality_profile: PersonalityProfile) -> Dict[str, Any]:
        """
        获取状态摘要
        
        Args:
            personality_profile: 人格画像
        
        Returns:
            状态摘要字典
        """
        dynamic_state = personality_profile.dynamic_state
        
        return {
            "mood": dynamic_state.current_mood,
            "energy_level": dynamic_state.energy_level,
            "relationship_count": len(dynamic_state.relationship_map),
            "relationships": {
                k: {
                    "intimacy": v.intimacy,
                    "history_summary": v.history_summary
                } for k, v in dynamic_state.relationship_map.items()
            }
        }

