"""
ScrollWeaver User Agent - 简化版用户Agent

保留MBTI和Big Five人格数据，添加身份和目标字段
移除语言风格提取相关处理
"""

import os
import uuid
from typing import Dict, Any, Optional, List
from modules.main_performer import Performer
from modules.personality_model import PersonalityProfile


class ScrollWeaverUserAgent(Performer):
    """
    ScrollWeaver专用用户Agent
    
    与Soulverse的UserAgent相比：
    - 保留: MBTI + Big Five人格数据
    - 移除: 语言风格提取、SoulverseMode、StyleVectorDB
    - 新增: identity（身份）和 goal（目标）字段
    """
    
    def __init__(self,
                 user_id: str,
                 role_code: str,
                 world_file_path: str,
                 mbti: str,
                 big_five: Dict[str, float],
                 identity: str,
                 goal: str,
                 nickname: str = None,
                 language: str = "zh",
                 db_type: str = "chroma",
                 llm_name: str = "gpt-4o-mini",
                 llm = None,
                 embedding_name: str = "bge-small",
                 embedding = None):
        """
        初始化ScrollWeaver用户Agent
        
        Args:
            user_id: 用户ID
            role_code: 角色代码
            world_file_path: 世界文件路径
            mbti: MBTI类型，如"INTJ"
            big_five: Big Five分数字典，如{"openness": 0.8, "conscientiousness": 0.7, ...}
            identity: 用户身份描述，如"旅行者"、"学者"
            goal: 用户目标，如"探索世界"、"寻找真相"
            nickname: 可选昵称（如未提供则使用user_id）
            language: 语言（默认中文）
            db_type: 向量数据库类型
            llm_name: LLM模型名称
            llm: LLM实例（可选）
            embedding_name: Embedding模型名称
            embedding: Embedding实例（可选）
        """
        # 保存用户特定属性
        self.user_id = user_id
        self.mbti = mbti
        self.big_five = big_five or {}
        self.identity = identity
        self.user_goal = goal  # 使用user_goal避免与Performer.goal冲突
        self.is_user_agent = True
        self.language = language
        
        # 直接设置Performer所需的属性，而不是调用父类_init_from_file
        self.role_code = role_code
        self.nickname = nickname or user_id
        self.role_name = f"用户_{self.nickname}" if language == "zh" else f"User_{self.nickname}"
        self.role_profile = self._generate_profile(mbti, big_five or {}, identity, goal, language)
        self.relation = {}
        self.motivation = ""
        self.hidden_motivation = ""
        self.activity = 1.0
        self.role_data = []  # 空的role_data用于向量数据库
        self.icon_path = ""  # 用户agent默认无头像
        
        # 设置状态属性
        self.acted = False
        self.status = ""
        self.goal = ""
        self.location_code = ""
        self.location_name = ""
        
        # 加载世界信息（简化版）
        from sw_utils import load_json_file
        try:
            if world_file_path and os.path.exists(world_file_path):
                world_info = load_json_file(world_file_path)
                self.world_settings = world_info.get("world_settings", world_info.get("description", ""))
            else:
                self.world_settings = ""
        except:
            self.world_settings = ""
        
        # 初始化prompt模板
        self._init_prompt()
        
        # 初始化LLM
        from sw_utils import get_models
        self.llm_name = llm_name
        if llm is None:
            llm = get_models(llm_name)
        self.llm = llm
        
        # 初始化embedding和向量数据库（空的）
        from modules.embedding import get_embedding_model
        from sw_utils import clean_collection_name, build_db
        if embedding is None:
            embedding = get_embedding_model(embedding_name, language=language)
        
        self.db_name = clean_collection_name(f"role_{role_code}_{embedding_name}")
        self.db = None  # 用户agent不需要db，设为None
        
        # 初始化history_manager（用于记录历史）
        from modules.history_manager import HistoryManager
        self.history_manager = HistoryManager()
        
        # 初始化memory（用于记忆检索，简化版）
        self.memory = None  # 用户agent不需要复杂记忆系统
        
        # 设置motivation为用户目标
        self.motivation = self._generate_motivation_from_goal()
    
    def _init_prompt(self):
        """初始化prompt模板"""
        if self.language == "zh":
            from modules.prompt.performer_prompt_zh import (
                ROLE_PLAN_PROMPT, ROLE_SINGLE_ROLE_RESPONSE_PROMPT, ROLE_MULTI_ROLE_RESPONSE_PROMPT,
                ROLE_SET_GOAL_PROMPT, INTERVENTION_PROMPT, UPDATE_GOAL_PROMPT, UPDATE_STATUS_PROMPT,
                ROLE_SET_MOTIVATION_PROMPT, SCRIPT_ATTENTION_PROMPT, ROLE_MOVE_PROMPT,
                SUMMARIZE_PROMPT, ROLE_NPC_RESPONSE_PROMPT, ROLE_THINK_PROMPT
            )
        else:
            from modules.prompt.performer_prompt_en import (
                ROLE_PLAN_PROMPT, ROLE_SINGLE_ROLE_RESPONSE_PROMPT, ROLE_MULTI_ROLE_RESPONSE_PROMPT,
                ROLE_SET_GOAL_PROMPT, INTERVENTION_PROMPT, UPDATE_GOAL_PROMPT, UPDATE_STATUS_PROMPT,
                ROLE_SET_MOTIVATION_PROMPT, SCRIPT_ATTENTION_PROMPT, ROLE_MOVE_PROMPT,
                SUMMARIZE_PROMPT, ROLE_NPC_RESPONSE_PROMPT, ROLE_THINK_PROMPT
            )
        
        self._ROLE_SET_GOAL_PROMPT = ROLE_SET_GOAL_PROMPT
        self._ROLE_PLAN_PROMPT = ROLE_PLAN_PROMPT
        self._ROLE_SINGLE_ROLE_RESPONSE_PROMPT = ROLE_SINGLE_ROLE_RESPONSE_PROMPT
        self._ROLE_MULTI_ROLE_RESPONSE_PROMPT = ROLE_MULTI_ROLE_RESPONSE_PROMPT
        self._INTERVENTION_PROMPT = INTERVENTION_PROMPT
        self._UPDATE_GOAL_PROMPT = UPDATE_GOAL_PROMPT
        self._UPDATE_STATUS_PROMPT = UPDATE_STATUS_PROMPT
        self._ROLE_SET_MOTIVATION_PROMPT = ROLE_SET_MOTIVATION_PROMPT
        self._SCRIPT_PROMPT = SCRIPT_ATTENTION_PROMPT
        self._ROLE_MOVE_PROMPT = ROLE_MOVE_PROMPT
        self._SUMMARIZE_PROMPT = SUMMARIZE_PROMPT
        self._ROLE_NPC_RESPONSE_PROMPT = ROLE_NPC_RESPONSE_PROMPT
        self._ROLE_THINK_PROMPT = ROLE_THINK_PROMPT
    
    def _create_temp_role_dir(self,
                               role_code: str,
                               nickname: str,
                               mbti: str,
                               big_five: Dict[str, float],
                               identity: str,
                               goal: str,
                               language: str) -> str:
        """
        创建临时角色目录和role_info.json文件
        
        Returns:
            角色文件目录路径
        """
        import json
        import tempfile
        
        # 创建临时目录
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        temp_dir = os.path.join(base_dir, "data", "temp_roles")
        os.makedirs(temp_dir, exist_ok=True)
        
        # 创建角色特定目录
        role_dir = os.path.join(temp_dir, role_code)
        os.makedirs(role_dir, exist_ok=True)
        
        # 根据MBTI和Big Five生成角色profile
        profile = self._generate_profile(mbti, big_five, identity, goal, language)
        
        # 创建简化的PersonalityProfile
        personality_profile = {
            "core_traits": {
                "mbti": mbti,
                "big_five": big_five
            },
            "speaking_style": {
                "tone": "自然、随和",
                "vocabulary_preference": "日常用语",
                "sentence_structure": "简洁明了"
            },
            "social_goals": [goal],
            "interests": [identity],
            "style_examples": []
        }
        
        # 创建role_info.json
        role_info = {
            "role_code": role_code,
            "role_name": f"用户_{nickname}",
            "nickname": nickname,
            "original_user_id": self.user_id,
            "source": "scrollweaver_user",
            "activity": 1.0,
            "profile": profile,
            "personality_profile": personality_profile,
            "identity": identity,
            "user_goal": goal,
            "relation": {},
            "motivation": ""
        }
        
        # 写入文件
        role_info_path = os.path.join(role_dir, "role_info.json")
        with open(role_info_path, 'w', encoding='utf-8') as f:
            json.dump(role_info, f, ensure_ascii=False, indent=2)
        
        return temp_dir
    
    def _generate_profile(self,
                          mbti: str,
                          big_five: Dict[str, float],
                          identity: str,
                          goal: str,
                          language: str) -> str:
        """
        根据人格数据生成角色profile文本
        """
        if language == "zh":
            # 解析Big Five特征
            traits = []
            if big_five.get("openness", 0.5) > 0.6:
                traits.append("富有创造力和好奇心")
            elif big_five.get("openness", 0.5) < 0.4:
                traits.append("务实稳重")
            
            if big_five.get("conscientiousness", 0.5) > 0.6:
                traits.append("做事认真负责")
            elif big_five.get("conscientiousness", 0.5) < 0.4:
                traits.append("随性自由")
            
            if big_five.get("extraversion", 0.5) > 0.6:
                traits.append("性格外向活泼")
            elif big_five.get("extraversion", 0.5) < 0.4:
                traits.append("性格内敛安静")
            
            if big_five.get("agreeableness", 0.5) > 0.6:
                traits.append("待人友善")
            elif big_five.get("agreeableness", 0.5) < 0.4:
                traits.append("独立自主")
            
            if big_five.get("neuroticism", 0.5) > 0.6:
                traits.append("情感丰富细腻")
            elif big_five.get("neuroticism", 0.5) < 0.4:
                traits.append("情绪稳定")
            
            traits_text = "、".join(traits) if traits else "性格平和"
            
            profile = f"""你是一位{identity}。

MBTI类型: {mbti}
性格特点: {traits_text}

你的目标是: {goal}

请以这个身份参与世界中的互动，根据你的性格和目标做出真实的反应。"""
        else:
            # English version
            traits = []
            if big_five.get("openness", 0.5) > 0.6:
                traits.append("creative and curious")
            elif big_five.get("openness", 0.5) < 0.4:
                traits.append("practical and grounded")
            
            if big_five.get("conscientiousness", 0.5) > 0.6:
                traits.append("responsible and organized")
            elif big_five.get("conscientiousness", 0.5) < 0.4:
                traits.append("flexible and spontaneous")
            
            if big_five.get("extraversion", 0.5) > 0.6:
                traits.append("outgoing and energetic")
            elif big_five.get("extraversion", 0.5) < 0.4:
                traits.append("introverted and reserved")
            
            if big_five.get("agreeableness", 0.5) > 0.6:
                traits.append("friendly and cooperative")
            elif big_five.get("agreeableness", 0.5) < 0.4:
                traits.append("independent and assertive")
            
            if big_five.get("neuroticism", 0.5) > 0.6:
                traits.append("emotionally sensitive")
            elif big_five.get("neuroticism", 0.5) < 0.4:
                traits.append("emotionally stable")
            
            traits_text = ", ".join(traits) if traits else "balanced personality"
            
            profile = f"""You are a {identity}.

MBTI Type: {mbti}
Personality: {traits_text}

Your goal is: {goal}

Participate in the world with this identity, making authentic responses based on your personality and goals."""
        
        return profile
    
    def _generate_motivation_from_goal(self) -> str:
        """
        从用户目标生成motivation
        """
        if self.language == "zh":
            return f"作为一名{self.identity}，我的目标是{self.user_goal}。我会积极探索这个世界，与其他人互动，追求我的目标。"
        else:
            return f"As a {self.identity}, my goal is to {self.user_goal}. I will actively explore this world, interact with others, and pursue my objective."
    
    def set_motivation(self,
                       world_description: str,
                       other_roles_info: Dict[str, Any],
                       intervention: str = "",
                       script: str = ""):
        """
        重写set_motivation方法，使用用户设定的目标
        """
        if self.motivation:
            return self.motivation
        
        self.motivation = self._generate_motivation_from_goal()
        return self.motivation
    
    def get_user_profile_summary(self) -> Dict[str, Any]:
        """
        获取用户profile摘要
        """
        return {
            "user_id": self.user_id,
            "role_code": self.role_code,
            "nickname": self.nickname,
            "mbti": self.mbti,
            "big_five": self.big_five,
            "identity": self.identity,
            "goal": self.user_goal,
            "motivation": self.motivation
        }
