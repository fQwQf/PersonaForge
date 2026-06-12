"""
三层人格模型数据结构定义
基于心理学理论：内核层（认知与特质）、表象层（语言与行为模式）、记忆层（经历与关系）
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum


class DefenseMechanism(str, Enum):
    """防御机制类型"""
    RATIONALIZATION = "Rationalization"  # 合理化
    PROJECTION = "Projection"  # 投射
    DENIAL = "Denial"  # 否认
    REPRESSION = "Repression"  # 压抑
    SUBLIMATION = "Sublimation"  # 升华
    DISPLACEMENT = "Displacement"  # 转移
    REACTION_FORMATION = "ReactionFormation"  # 反向形成
    HUMOR = "Humor"  # 幽默/自嘲
    INTELLECTUALIZATION = "Intellectualization"  # 理智化


class SentenceLength(str, Enum):
    """句长偏好"""
    SHORT = "short"  # 短句为主
    MEDIUM = "medium"  # 中等长度
    LONG = "long"  # 长句为主
    MIXED = "mixed"  # 混合


class VocabularyLevel(str, Enum):
    """词汇等级"""
    ACADEMIC = "academic"  # 学术/正式
    CASUAL = "casual"  # 口语化
    NETWORK = "network"  # 网络用语
    MIXED = "mixed"  # 混合


class PunctuationHabit(str, Enum):
    """标点习惯"""
    MINIMAL = "minimal"  # 少用标点
    STANDARD = "standard"  # 标准使用
    EXCESSIVE = "excessive"  # 频繁使用（如...、~~~）
    MIXED = "mixed"  # 混合


class EmojiFrequency(str, Enum):
    """表情使用频率"""
    NONE = "none"  # 不用
    LOW = "low"  # 偶尔
    MEDIUM = "medium"  # 适中
    HIGH = "high"  # 频繁


@dataclass
class CoreTraits:
    """内核层：特质与价值观"""
    mbti: str  # MBTI类型，如"INFP-T"
    big_five: Dict[str, float]  # 大五人格评分，0-1之间
    values: List[str]  # 价值观列表，如["自由", "审美", "真诚"]
    defense_mechanism: str  # 防御机制，DefenseMechanism枚举值
    
    def __post_init__(self):
        """验证Big Five数据"""
        required_traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
        for trait in required_traits:
            if trait not in self.big_five:
                raise ValueError(f"Missing Big Five trait: {trait}")
            if not 0 <= self.big_five[trait] <= 1:
                raise ValueError(f"Big Five trait {trait} must be between 0 and 1")


@dataclass
class SpeakingStyle:
    """表象层：语言风格矩阵"""
    sentence_length: str  # SentenceLength枚举值
    vocabulary_level: str  # VocabularyLevel枚举值
    punctuation_habit: str  # PunctuationHabit枚举值
    emoji_usage: Dict[str, Any]  # {"frequency": EmojiFrequency, "preferred": List[str], "avoided": List[str]}
    catchphrases: List[str]  # 口头禅列表
    tone_markers: List[str]  # 语气词列表，如["啊", "捏", "呗"]
    
    def __post_init__(self):
        """验证表情使用数据"""
        if "frequency" not in self.emoji_usage:
            self.emoji_usage["frequency"] = "none"
        if "preferred" not in self.emoji_usage:
            self.emoji_usage["preferred"] = []
        if "avoided" not in self.emoji_usage:
            self.emoji_usage["avoided"] = []


@dataclass
class RelationshipInfo:
    """关系信息"""
    intimacy: int  # 亲密度，0-100
    history_summary: str  # 历史摘要，如"曾一起聊过《百年孤独》"


@dataclass
class DynamicState:
    """记忆层：动态状态"""
    current_mood: str  # 当前心情，如"melancholy", "cheerful", "neutral"
    energy_level: int  # 能量值，0-100
    relationship_map: Dict[str, RelationshipInfo] = field(default_factory=dict)  # 关系映射，key为target_role_code
    
    def update_mood(self, new_mood: str):
        """更新心情"""
        self.current_mood = new_mood
    
    def update_energy(self, delta: int):
        """更新能量值（增量）"""
        self.energy_level = max(0, min(100, self.energy_level + delta))
    
    def update_relationship(self, target_code: str, intimacy: Optional[int] = None, 
                          history_summary: Optional[str] = None):
        """更新关系信息"""
        if target_code not in self.relationship_map:
            self.relationship_map[target_code] = RelationshipInfo(
                intimacy=intimacy or 0,
                history_summary=history_summary or ""
            )
        else:
            if intimacy is not None:
                self.relationship_map[target_code].intimacy = intimacy
            if history_summary is not None:
                self.relationship_map[target_code].history_summary = history_summary


@dataclass
class PersonalityProfile:
    """完整的人格画像（三层模型）"""
    core_traits: CoreTraits
    speaking_style: SpeakingStyle
    dynamic_state: DynamicState
    interests: List[str]  # 兴趣标签
    social_goals: List[str]  # 社交目标
    long_term_goals: List[str]  # 长期目标
    style_examples: List[Dict[str, str]] = field(default_factory=list)  # Few-Shot样本，格式：[{"context": "...", "response": "..."}]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（用于JSON序列化）"""
        result = {
            "core_traits": {
                "mbti": self.core_traits.mbti,
                "big_five": self.core_traits.big_five,
                "values": self.core_traits.values,
                "defense_mechanism": self.core_traits.defense_mechanism
            },
            "speaking_style": {
                "sentence_length": self.speaking_style.sentence_length,
                "vocabulary_level": self.speaking_style.vocabulary_level,
                "punctuation_habit": self.speaking_style.punctuation_habit,
                "emoji_usage": self.speaking_style.emoji_usage,
                "catchphrases": self.speaking_style.catchphrases,
                "tone_markers": self.speaking_style.tone_markers
            },
            "dynamic_state": {
                "current_mood": self.dynamic_state.current_mood,
                "energy_level": self.dynamic_state.energy_level,
                "relationship_map": {
                    k: {
                        "intimacy": v.intimacy,
                        "history_summary": v.history_summary
                    } for k, v in self.dynamic_state.relationship_map.items()
                }
            },
            "interests": self.interests,
            "social_goals": self.social_goals,
            "long_term_goals": self.long_term_goals,
            "style_examples": self.style_examples
        }
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersonalityProfile':
        """从字典创建PersonalityProfile"""
        core_traits = CoreTraits(
            mbti=data["core_traits"]["mbti"],
            big_five=data["core_traits"]["big_five"],
            values=data["core_traits"]["values"],
            defense_mechanism=data["core_traits"]["defense_mechanism"]
        )
        
        speaking_style = SpeakingStyle(
            sentence_length=data["speaking_style"]["sentence_length"],
            vocabulary_level=data["speaking_style"]["vocabulary_level"],
            punctuation_habit=data["speaking_style"]["punctuation_habit"],
            emoji_usage=data["speaking_style"]["emoji_usage"],
            catchphrases=data["speaking_style"]["catchphrases"],
            tone_markers=data["speaking_style"]["tone_markers"]
        )
        
        dynamic_state = DynamicState(
            current_mood=data["dynamic_state"]["current_mood"],
            energy_level=data["dynamic_state"]["energy_level"]
        )
        for k, v in data["dynamic_state"]["relationship_map"].items():
            dynamic_state.relationship_map[k] = RelationshipInfo(
                intimacy=v["intimacy"],
                history_summary=v["history_summary"]
            )
        
        return cls(
            core_traits=core_traits,
            speaking_style=speaking_style,
            dynamic_state=dynamic_state,
            interests=data.get("interests", []),
            social_goals=data.get("social_goals", []),
            long_term_goals=data.get("long_term_goals", []),
            style_examples=data.get("style_examples", [])
        )
    
    def to_profile_text(self) -> str:
        """生成文本格式的profile（用于向后兼容）"""
        big_five_desc = ", ".join([
            f"{k}: {v:.2f}" for k, v in self.core_traits.big_five.items()
        ])
        
        profile = f"""这是一个基于三层人格模型创建的AI Agent。

内核层（认知与特质）：
- MBTI类型：{self.core_traits.mbti}
- 大五人格：{big_five_desc}
- 价值观：{', '.join(self.core_traits.values)}
- 防御机制：{self.core_traits.defense_mechanism}

表象层（语言风格）：
- 句长偏好：{self.speaking_style.sentence_length}
- 词汇等级：{self.speaking_style.vocabulary_level}
- 标点习惯：{self.speaking_style.punctuation_habit}
- 表情使用：{self.speaking_style.emoji_usage['frequency']}
- 口头禅：{', '.join(self.speaking_style.catchphrases) if self.speaking_style.catchphrases else '无'}
- 语气词：{', '.join(self.speaking_style.tone_markers) if self.speaking_style.tone_markers else '无'}

记忆层（动态状态）：
- 当前心情：{self.dynamic_state.current_mood}
- 能量值：{self.dynamic_state.energy_level}/100

兴趣标签：{', '.join(self.interests)}

社交目标：{', '.join(self.social_goals)}

长期目标：{', '.join(self.long_term_goals)}"""
        
        return profile

