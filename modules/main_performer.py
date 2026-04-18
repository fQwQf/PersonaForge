# Copyright 2025 The BookWorld Authors.
# Modifications Copyright 2026 fQwQf.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified from the original BookWorld codebase.
# For original implementation: https://github.com/alienet1109/BookWorld

import sys
from collections import defaultdict
sys.path.append("../")
import os
from typing import Any, Dict, List, Optional, Literal
from modules.embedding import get_embedding_model
from modules.memory import build_performer_memory
from modules.history_manager import HistoryManager
from modules.models import RolePlan, SingleRoleResponse, MultiRoleResponse, NPCRoleResponse, UpdateGoal, UpdateStatus, MoveResponse
from modules.personality_model import PersonalityProfile
from modules.dual_process_agent import DualProcessAgent
from modules.dynamic_state_manager import DynamicStateManager
from modules.style_vector_db import StyleVectorDB
from sw_utils import *
import random
import warnings
warnings.filterwarnings("ignore")



class Performer:
    def __init__(self, 
                 role_code: str,
                 role_file_dir: str,
                 world_file_path: str,
                 source: str = "",
                 language: str = "en",
                 db_type: str = "chroma",
                 llm_name: str = "gpt-4o-mini",
                 llm = None,
                 embedding_name: str = "bge-small",
                 embedding = None
                 ):
        super(Performer, self).__init__()
        self.language: str  = language
        self.role_code: str = role_code
        
        self.history_manager = HistoryManager()
        self.prompts: List[Dict] = []
        self.acted: bool = False
        self.status: str = ""
        self.goal: str = ""
        self.location_code: str = ""
        self.location_name: str = ""
        self.motivation: str = ""
        
        self._init_from_file(role_code, role_file_dir, world_file_path, source)
        self._init_prompt()
        
        self.llm_name = llm_name
        if llm == None:
            llm = get_models(llm_name)
        self.llm = llm
        
        if embedding is None:
            embedding = get_embedding_model(embedding_name, language=self.language)
        
        self.db_name = clean_collection_name(f"role_{role_code}_{embedding_name}")
        self.db = build_db(data = self.role_data,
                           db_name = self.db_name,
                           db_type = db_type,
                           embedding = embedding)
        self.world_db = None
        self.world_db_name = ""
        self.memory = build_performer_memory(llm_name=llm_name,
                                              embedding_name = embedding_name,
                                              embedding = embedding,
                                              db_name = self.db_name.replace("role","memory"),
                                              language = self.language,
                                              type="naive"
                                              )
        
        # 初始化三层人格模型相关组件（如果存在personality_profile）
        self.dual_process_agent: Optional[DualProcessAgent] = None
        self.dynamic_state_manager: Optional[DynamicStateManager] = None
        self.style_vector_db: Optional[StyleVectorDB] = None
        
        if self.personality_profile:
            self.dual_process_agent = DualProcessAgent(llm=self.llm, language=self.language)
            self.dynamic_state_manager = DynamicStateManager(llm=self.llm, language=self.language)
            
            # 初始化风格向量数据库（如果存在）
            if self.style_vector_db_name and embedding:
                self.style_vector_db = StyleVectorDB(
                    db_name=self.style_vector_db_name,
                    embedding_name=embedding_name,
                    db_type=db_type,
                    language=self.language
                )
        
    def _init_prompt(self):
        if self.language == 'zh':
            from modules.prompt.performer_prompt_zh \
                import ROLE_PLAN_PROMPT,ROLE_SINGLE_ROLE_RESPONSE_PROMPT,ROLE_MULTI_ROLE_RESPONSE_PROMPT,ROLE_SET_GOAL_PROMPT,INTERVENTION_PROMPT,UPDATE_GOAL_PROMPT,UPDATE_STATUS_PROMPT,ROLE_SET_MOTIVATION_PROMPT,SCRIPT_ATTENTION_PROMPT,ROLE_MOVE_PROMPT,SUMMARIZE_PROMPT,ROLE_NPC_RESPONSE_PROMPT,ROLE_THINK_PROMPT
            
        else:
            from modules.prompt.performer_prompt_en \
                import ROLE_PLAN_PROMPT,ROLE_SINGLE_ROLE_RESPONSE_PROMPT,ROLE_MULTI_ROLE_RESPONSE_PROMPT,ROLE_SET_GOAL_PROMPT,INTERVENTION_PROMPT,UPDATE_GOAL_PROMPT,UPDATE_STATUS_PROMPT,ROLE_SET_MOTIVATION_PROMPT,SCRIPT_ATTENTION_PROMPT,ROLE_MOVE_PROMPT,SUMMARIZE_PROMPT,ROLE_NPC_RESPONSE_PROMPT,ROLE_THINK_PROMPT
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
            
    def _init_from_file(self, 
                        role_code: str, 
                        role_file_dir: str, 
                        world_file_path: str,
                        source:str):
        if source and os.path.exists(os.path.join(role_file_dir, source)):
            for path in get_child_folders(os.path.join(role_file_dir, source)):
                if role_code in path:
                    role_path = path
                    break
        else:
            for path in get_grandchild_folders(role_file_dir):
                if role_code in path:
                    role_path = path
                    break
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        role_profile_path = os.path.join(base_dir, role_path,"role_info.json")
        
        role_info = load_json_file(role_profile_path)
        # self.role_info = role_info
        self.role_profile: str = role_info['profile']
        # 确保nickname不为None或null，如果为空则使用role_name
        nickname_value = role_info.get("nickname")
        if not nickname_value or nickname_value == "null" or nickname_value is None:
            nickname_value = role_info["role_name"]
        self.nickname: str = nickname_value
        self.role_name: str = role_info["role_name"]
        self.relation: str = role_info["relation"]
        self.motivation: str = role_info["motivation"] if "motivation" in role_info else ""
        self.hidden_motivation: str = role_info.get("hidden_motivation", "")
        
        self.activity: float = float(role_info["activity"]) if "activity" in role_info else 1.0
        
        # 加载三层人格模型（如果存在）
        self.personality_profile: Optional[PersonalityProfile] = None
        if "personality_profile" in role_info:
            try:
                self.personality_profile = PersonalityProfile.from_dict(role_info["personality_profile"])
            except Exception as e:
                print(f"Warning: Failed to load personality_profile for {role_code}: {e}")
        
        # 加载风格样本（如果存在）
        self.style_examples: List[Dict[str, str]] = role_info.get("style_examples", [])
        
        # 风格向量数据库名称（如果存在）
        self.style_vector_db_name: Optional[str] = role_info.get("style_vector_db_name")
        
        self.icon_path: str = os.path.join(base_dir, role_path,"icon.png")
        self.avatar_path: str = os.path.join(base_dir, role_path,"avatar.png")
        for image_type in ['jpg', 'png', 'bmp', 'webp']:
            if os.path.exists(os.path.join(base_dir, role_path,f"./avatar.{image_type}")):
                self.avatar_path: str = os.path.join(base_dir, role_path,f"avatar.{image_type}")
            if os.path.exists(os.path.join(base_dir, role_path,f"./icon.{image_type}")):
                self.icon_path: str = os.path.join(base_dir, role_path,f"icon.{image_type}")

        self.role_data: List[str] = build_performer_data(os.path.join(base_dir, role_path))
        
    # Agent
    def set_motivation(self, 
                       world_description: str, 
                       other_roles_info: Dict[str, Any], 
                       intervention: str = "", 
                       script: str = ""):
        if self.motivation:
            return self.motivation
        other_roles_info_text = self.get_other_roles_info_text(other_roles_info)
        prompt = self._ROLE_SET_MOTIVATION_PROMPT.format(**
            {
                "role_name": self.role_name,
                "profile":self.role_profile,
                "world_description": world_description,
                "other_roles_description": other_roles_info_text,
                "location": self.location_name
            })  
        if script:
            script = self._SCRIPT_PROMPT.format(**
                {"script": script}
            )  
            prompt = prompt + script
        elif intervention:
            intervention = self._INTERVENTION_PROMPT.format(**
                {"intervention": intervention}
            )
            prompt = intervention + prompt + "\n**注意: 在你的动机中考虑全局事件的影响**" if self.language == "zh" else intervention + prompt + "\n**Notice that: You should take the global event into consideration.**"
        
        from .models import MotivationText
        print(f"[Performer] 开始为角色 {self.role_name} 设置动机...")
        try:
            # 使用结构化输出
            response_model = self.llm.chat(prompt, response_model=MotivationText)
            motivation = response_model.motivation
            print(f"[Performer] 角色 {self.role_name} 动机设置成功（结构化输出）: {motivation[:100] if motivation else 'None'}...")
        except Exception as e:
            print(f"[Performer] 角色 {self.role_name} 动机设置结构化输出失败: {e}")
            import traceback
            traceback.print_exc()
            # 回退到文本输出
            try:
                motivation = self.llm.chat(prompt)
                if not isinstance(motivation, str):
                    motivation = str(motivation)
                print(f"[Performer] 角色 {self.role_name} 动机设置成功（文本输出）: {motivation[:100] if motivation else 'None'}...")
            except Exception as e2:
                print(f"[Performer] 角色 {self.role_name} 文本输出也失败: {e2}")
                import traceback
                traceback.print_exc()
                motivation = "追求个人目标和成长" if self.language == "zh" else "Pursue personal goals and growth"
                print(f"[Performer] 角色 {self.role_name} 使用默认动机: {motivation}")
        
        # 确保motivation是字符串且不为空
        if not isinstance(motivation, str):
            motivation = str(motivation)
        if not motivation or not motivation.strip():
            motivation = "追求个人目标和成长" if self.language == "zh" else "Pursue personal goals and growth"
            print(f"[Performer] 警告: 角色 {self.role_name} 动机为空，使用默认值")
        
        # 安全地截取前50个字符用于日志
        motivation_preview = motivation[:50] if len(motivation) > 50 else motivation
        print(f"[Performer] 角色 {self.role_name} 动机设置成功: {motivation_preview}...")
        self.save_prompt(prompt = prompt, detail = motivation)
        self.motivation = motivation
        return motivation
    
    def think(self,
              context: str,
              other_roles_info: Dict[str, Any],
              world_description: str = "") -> Dict[str, Any]:
        """
        思考链方法 - 内部思考步骤
        
        Args:
            context: 当前情况的描述（如"轮到你发言了"）
            other_roles_info: 其他角色信息
            world_description: 世界观描述
            
        Returns:
            包含 analysis, plan, memory_to_save 的字典
        """
        from modules.models import ThoughtChain
        
        action_history_text = self.retrieve_history(query="", retrieve=False)
        
        if len(other_roles_info) == 1:
            other_roles_info_text = "没有人在这里。你不能进行涉及角色的互动。" if self.language == "zh" else "No one else is here. You can not interact with roles."
        else:
            other_roles_info_text = self.get_other_roles_info_text(other_roles_info, if_profile=False)
        
        # 构建思考链 prompt
        prompt = self._ROLE_THINK_PROMPT.format(
            role_name=self.role_name,
            nickname=self.nickname,
            context=context,
            profile=self.role_profile,
            goal=self.goal,
            status=self.status,
            hidden_motivation=self.hidden_motivation if self.hidden_motivation else (
                "追求个人目标和成长" if self.language == "zh" else "Pursue personal goals and growth"
            ),
            history=action_history_text,
            other_roles_info=other_roles_info_text
        )
        
        max_tries = 3
        thought_result = {
            "analysis": "",
            "plan": "",
            "memory_to_save": None
        }
        
        for i in range(max_tries):
            try:
                # 使用结构化输出
                response_model = self.llm.chat(prompt, response_model=ThoughtChain, temperature=0.7)
                thought_result.update(response_model.model_dump())
                break
            except Exception as e:
                print(f"[{self.role_name}] 思考链结构化输出失败! 第{i+1}次尝试. 错误: {e}")
                # 如果结构化输出失败，回退到文本解析
                if i < max_tries - 1:
                    try:
                        response_text = self.llm.chat(prompt, temperature=0.7)
                        parsed = json_parser(response_text)
                        thought_result.update({
                            "analysis": parsed.get("analysis", ""),
                            "plan": parsed.get("plan", ""),
                            "memory_to_save": parsed.get("memory_to_save")
                        })
                        break
                    except Exception as e2:
                        print(f"[{self.role_name}] 思考链文本解析也失败: {e2}")
                        continue
                else:
                    # 最后一次尝试，使用默认值
                    print(f"[{self.role_name}] 所有思考链尝试都失败，使用默认值")
                    thought_result["analysis"] = f"{self.role_name}正在思考当前情况。" if self.language == "zh" else f"{self.role_name} is thinking about the current situation."
                    thought_result["plan"] = f"{self.role_name}将根据情况采取行动。" if self.language == "zh" else f"{self.role_name} will act according to the situation."
        
        # 保存思考结果到短期记忆向量库
        if thought_result.get("plan"):
            memory_text = f"计划: {thought_result['plan']}"
            if thought_result.get("memory_to_save"):
                memory_text += f" | 记忆要点: {thought_result['memory_to_save']}"
            try:
                self.memory.add_record(memory_text)
                print(f"[{self.role_name}] 思考结果已存入短期记忆")
            except Exception as e:
                print(f"[{self.role_name}] 保存思考结果到记忆失败: {e}")
        
        self.save_prompt(prompt=prompt, detail=str(thought_result))
        return thought_result
    
    def plan(self, 
             other_roles_info: Dict[str, Any], 
             available_locations: List[str], 
             world_description: str, 
             intervention: str = "",
             use_thought_chain: bool = True):
        """
        计划方法 - 支持思考链的两阶段流程
        
        Args:
            other_roles_info: 其他角色信息
            available_locations: 可用地点列表
            world_description: 世界观描述
            intervention: 干预信息
            use_thought_chain: 是否使用思考链（默认 True）
        """
        action_history_text = self.retrieve_history(query = "", retrieve=False)
        references = self.retrieve_references(query = action_history_text)
        knowledges = self.retrieve_knowledges(query = action_history_text)
        
        if len(other_roles_info) == 1:
            other_roles_info_text = "没有人在这里。你不能进行涉及角色的互动。" if self.language == "zh" else "No one else is here. You can not interact with roles."
        else:
            other_roles_info_text = self.get_other_roles_info_text(other_roles_info, if_profile = False)
        
        # 第一阶段：思考链（如果启用）
        thought_result = None
        if use_thought_chain:
            # 在context中加入事件信息，确保思考链也考虑当前事件
            event_context = f"\n当前全局事件: {intervention}" if intervention else ""
            context = f"你需要基于你的目标、状态和提供的其它信息实行下一步行动。当前地点: {self.location_name}{event_context}"
            thought_result = self.think(
                context=context,
                other_roles_info=other_roles_info,
                world_description=world_description
            )
            print(f"[{self.role_name}] 思考链完成 - 分析: {thought_result.get('analysis', '')[:50]}...")
        
        if intervention:
            intervention = self._INTERVENTION_PROMPT.format(**
                {"intervention": intervention}
            )
        
        # 构建计划 prompt，如果使用了思考链，将思考结果加入 prompt
        plan_prompt_dict = {
            "role_name": self.role_name,
            "nickname": self.nickname,
            "profile": self.role_profile,
            "goal": self.goal,
            "status": self.status,
            "history": action_history_text,
            "other_roles_info": other_roles_info_text,
            "world_description": world_description,
            "location": self.location_name,
            "references": references,
            "knowledges": knowledges,
        }
        
        # 如果有思考链结果，添加到 prompt
        if thought_result:
            thought_hint = f"\n\n## 你的思考结果\n分析: {thought_result.get('analysis', '')}\n计划: {thought_result.get('plan', '')}"
            plan_prompt_dict["knowledges"] = knowledges + thought_hint
        
        prompt = self._ROLE_PLAN_PROMPT.format(**plan_prompt_dict)
        prompt = intervention + prompt
        
        max_tries = 3
        plan = {"action": "待机" if self.language == "zh" else "Stay", 
                "destination": None,
                "interact_type":'no',
                "target_role_codes": [],
                "target_npc_name":None,
                "detail": f"{self.role_name}原地不动，观察情况。" if self.language == "zh" else f"{self.role_name} stays put."
                }
        
        for i in range(max_tries):
            try:
                # 使用结构化输出
                response_model = self.llm.chat(prompt, response_model=RolePlan)
                # 转换为字典格式
                plan.update(response_model.model_dump())
                break
            except Exception as e:
                print(f"[{self.role_name}] 结构化输出失败! 第{i+1}次尝试. 错误: {e}")
                # 如果结构化输出失败，回退到文本解析
                if i < max_tries - 1:
                    try:
                        response_text = self.llm.chat(prompt)
                        plan.update(json_parser(response_text))
                        break
                    except Exception as e2:
                        print(f"[{self.role_name}] 文本解析也失败: {e2}")
                        continue
                else:
                    # 最后一次尝试，使用默认值
                    print(f"[{self.role_name}] 所有尝试都失败，使用默认值")
        
        plan["role_code"] = self.role_code
        self.save_prompt(detail=plan["detail"], prompt=prompt)
        return plan
    
    def plan_with_style(self, 
             other_roles_info: Dict[str, Any], 
             available_locations: List[str], 
             world_description: str, 
             intervention: str = "",
             style_hint: str = "",
             temperature: float = 0.8):
        """带风格提示和温度参数的plan方法"""
        action_history_text = self.retrieve_history(query = "", retrieve=False)
        references = self.retrieve_references(query = action_history_text)
        knowledges = self.retrieve_knowledges(query = action_history_text)
        
        if len(other_roles_info) == 1:
            other_roles_info_text = "没有人在这里。你不能进行涉及角色的互动。" if self.language == "zh" else "No one else is here. You can not interact with roles."
        else:
            other_roles_info_text = self.get_other_roles_info_text(other_roles_info, if_profile = False)
        
        if intervention:
            intervention = self._INTERVENTION_PROMPT.format(**{"intervention": intervention})
        
        # 构建基础prompt
        prompt = self._ROLE_PLAN_PROMPT.format(**{
            "role_name": self.role_name,
            "nickname": self.nickname,
            "profile": self.role_profile,
            "goal": self.goal,
            "status": self.status,
            "history": action_history_text,
            "other_roles_info": other_roles_info_text,
            "world_description": world_description,
            "location": self.location_name,
            "references": references,
            "knowledges": knowledges,
        })
        
        # 添加风格提示
        if style_hint:
            style_prompt = f"\n\n## 行动风格要求\n{style_hint}\n" if self.language == "zh" else f"\n\n## Action Style Requirement\n{style_hint}\n"
            prompt = prompt + style_prompt
        
        prompt = intervention + prompt
        max_tries = 3
        plan = {"action": "待机" if self.language == "zh" else "Stay", 
                "destination": None,
                "interact_type":'no',
                "target_role_codes": [],
                "target_npc_name":None,
                "detail": f"{self.role_name}原地不动，观察情况。" if self.language == "zh" else f"{self.role_name} stays put."
                }
        
        for i in range(max_tries):
            try:
                # 使用结构化输出和指定的温度参数
                response_model = self.llm.chat(prompt, temperature=temperature, response_model=RolePlan)
                # 转换为字典格式
                plan.update(response_model.model_dump())
                break
            except Exception as e:
                print(f"[{self.role_name}] 结构化输出失败! 第{i+1}次尝试. 错误: {e}")
                # 如果结构化输出失败，回退到文本解析
                if i < max_tries - 1:
                    try:
                        response_text = self.llm.chat(prompt, temperature=temperature)
                        plan.update(json_parser(response_text))
                        break
                    except Exception as e2:
                        print(f"[{self.role_name}] 文本解析也失败: {e2}")
                        continue
                else:
                    # 最后一次尝试，使用默认值
                    print(f"[{self.role_name}] 所有尝试都失败，使用默认值")
        plan["role_code"] = self.role_code
        self.save_prompt(detail=plan["detail"], prompt=prompt)
        return plan
    
    def npc_interact(self,
                     npc_name:str,
                     npc_response:str,
                     history:str,
                     intervention:str = ""
                     ):
        references = self.retrieve_references(npc_response)
        knowledges = self.retrieve_knowledges(query = npc_response)
        
        if intervention:
            intervention = self._INTERVENTION_PROMPT.format(**
                {"intervention": intervention}
            )
        prompt = self._ROLE_NPC_RESPONSE_PROMPT.format(**
            {
                "role_name": self.role_name,
                "nickname": self.nickname,
                "profile": self.role_profile,
                "goal": self.goal,
                "npc_name":npc_name,
                "npc_response":npc_response,
                "references": references,
                "knowledges":knowledges,
                "dialogue_history": history
            }
            )
        prompt = intervention + prompt
        interaction = {
                    "if_end_interaction": True,
                    "detail": "",
                    }
        try:
            # 使用结构化输出
            response_model = self.llm.chat(prompt, response_model=NPCRoleResponse)
            interaction.update(response_model.model_dump())
        except Exception as e:
            print(f"[{self.role_name}] 结构化输出失败! 错误: {e}")
            # 回退到文本解析
            try:
                response_text = self.llm.chat(prompt)
                interaction.update(json_parser(response_text))
            except Exception as e2:
                print(f"[{self.role_name}] 文本解析也失败: {e2}")
        print(f"[{self.role_name}] 所有尝试都失败，使用默认值")
        
        self.save_prompt(detail = interaction["detail"], 
                      prompt = prompt)
        return interaction
    
    def single_role_interact(self, 
                             action_maker_code: str, 
                             action_maker_name: str,
                             action_detail: str, 
                             action_maker_profile: str, 
                             intervention: str = ""):
        references = self.retrieve_references(action_detail)
        history = self.retrieve_history(query = action_detail)
        knowledges = self.retrieve_knowledges(query = action_detail)
        
        relation = f"role_code:{action_maker_code}\n" + self.search_relation(action_maker_code)
        
        # 检查是否应该使用双重思维链
        use_dual_process = False
        if self.personality_profile and self.dual_process_agent:
            other_role_info = {"role_code": action_maker_code, "role_name": action_maker_name}
            relationship_map = self.personality_profile.dynamic_state.relationship_map
            use_dual_process = self.dual_process_agent.is_critical_interaction(
                action_detail=action_detail,
                other_role_info=other_role_info,
                personality_profile=self.personality_profile,
                relationship_map=relationship_map
            )
        
        if use_dual_process and self.personality_profile:
            # 使用双重思维链
            # Step 1: 生成内心独白
            inner_monologue = self.dual_process_agent.generate_inner_monologue(
                personality_profile=self.personality_profile,
                action_detail=action_detail,
                action_maker_name=action_maker_name,
                history=history,
                goal=self.goal,
                status=self.status
            )
            
            # 检索风格样本
            style_examples = self.style_examples.copy()
            if self.style_vector_db:
                similar_examples = self.style_vector_db.search_similar_style(action_detail, top_k=3)
                style_examples.extend([
                    {"context": ex.get("context", ""), "response": ex["text"]}
                    for ex in similar_examples
                ])
            
            # Step 2: 生成风格化回复
            styled_response = self.dual_process_agent.generate_styled_response(
                inner_monologue=inner_monologue,
                personality_profile=self.personality_profile,
                style_examples=style_examples,
                action_detail=action_detail,
                action_maker_name=action_maker_name,
                history=history
            )
            
            # 构建回复（使用标准格式）
            interaction = {
                "if_end_interaction": False,
                "extra_interact_type": "no",
                "target_npc_name": "",
                "detail": styled_response
            }
            
            # 更新动态状态
            if self.dynamic_state_manager:
                self.dynamic_state_manager.update_state_after_interaction(
                    personality_profile=self.personality_profile,
                    interaction_detail=styled_response,
                    other_role_code=action_maker_code,
                    other_role_name=action_maker_name
                )
                # 保存更新后的状态到role_info.json
                self._save_personality_profile()
            
            self.save_prompt(detail=interaction["detail"], prompt="")
        else:
            # 使用传统方法
            if intervention:
                intervention = self._INTERVENTION_PROMPT.format(**
                    {"intervention": intervention}
                )
            
            # 构建prompt，如果有人格画像则添加人格信息
            prompt_dict = {
                "role_name": self.role_name,
                "nickname": self.nickname,
                "action_maker_name": action_maker_name,
                "action_detail": action_detail, 
                "profile": self.role_profile,
                "action_maker_profile": action_maker_profile,
                "relation": relation,
                "goal": self.goal,
                "status": self.status,
                "references": references,
                "knowledges": knowledges,
                "history": history
            }
            
            prompt = self._ROLE_SINGLE_ROLE_RESPONSE_PROMPT.format(**prompt_dict)
            
            # 如果有人格画像，附加人格信息到prompt
            if self.personality_profile:
                personality_info = self._format_big_five_info() + self._format_speaking_style_info() + self._format_style_examples()
                prompt = prompt + personality_info
            
            prompt = intervention + prompt
            
            max_tries = 3
            interaction = {
                        "if_end_interaction": True,
                        "extra_interact_type":"no",
                        "target_npc_name":"",
                        "detail": "",
                        }
            
            for i in range(max_tries):
                try:
                    # 使用结构化输出
                    response_model = self.llm.chat(prompt, response_model=SingleRoleResponse)
                    interaction.update(response_model.model_dump())
                    break
                except Exception as e:
                    print(f"[{self.role_name}] 结构化输出失败! 第{i+1}次尝试. 错误: {e}")
                    # 如果结构化输出失败，回退到文本解析
                    if i < max_tries - 1:
                        try:
                            response_text = self.llm.chat(prompt)
                            interaction.update(json_parser(response_text))
                            break
                        except Exception as e2:
                            print(f"[{self.role_name}] 文本解析也失败: {e2}")
                            continue
                    else:
                        # 最后一次尝试，使用默认值
                        print(f"[{self.role_name}] 所有尝试都失败，使用默认值")
            
            # 更新动态状态（即使不使用双重思维链）
            if self.personality_profile and self.dynamic_state_manager:
                self.dynamic_state_manager.update_state_after_interaction(
                    personality_profile=self.personality_profile,
                    interaction_detail=interaction.get("detail", ""),
                    other_role_code=action_maker_code,
                    other_role_name=action_maker_name
                )
                self._save_personality_profile()
            
            self.save_prompt(detail=interaction["detail"], prompt=prompt)
        
        return interaction
    
    def multi_role_interact(self, 
                            action_maker_code: str, 
                            action_maker_name: str, 
                            action_detail: str, 
                            action_maker_profile: str, 
                            other_roles_info: Dict[str, Any], 
                            intervention: str = ""):
        references = self.retrieve_references(query = action_detail)
        history = self.retrieve_history(query = action_detail)
        knowledges = self.retrieve_knowledges(query = action_detail)
        
        other_roles_info_text = self.get_other_roles_info_text(other_roles_info, if_profile = False)

        if intervention:
            intervention = self._INTERVENTION_PROMPT.format(**
                {"intervention": intervention}
            )
        prompt = self._ROLE_MULTI_ROLE_RESPONSE_PROMPT.format(**
            {
                "role_name": self.role_name,
                "nickname": self.nickname,
                "action_maker_name": action_maker_name,
                "action_detail": action_detail, 
                "profile": self.role_profile,
                "action_maker_profile": action_maker_profile,
                "other_roles_info":other_roles_info_text,
                "goal":self.goal,
                "status": self.status,
                "references": references,
                "knowledges":knowledges,
                "history": history
            }
            )
        
        # 如果有人格画像，附加人格信息到prompt
        if self.personality_profile:
            personality_info = self._format_big_five_info() + self._format_speaking_style_info() + self._format_style_examples()
            prompt = prompt + personality_info
        
        prompt = intervention + prompt
        max_tries = 3
        interaction = {
                    "if_end_interaction": True,
                    "extra_interact_type":"no",
                    "target_role_code":"",
                    "target_npc_name":"",
                    "visible_role_codes":[],
                    "detail": "",
                    }
        
        for i in range(max_tries):
            try:
                # 使用结构化输出
                response_model = self.llm.chat(prompt, response_model=MultiRoleResponse)
                interaction.update(response_model.model_dump())
                break
            except Exception as e:
                print(f"[{self.role_name}] 结构化输出失败! 第{i+1}次尝试. 错误: {e}")
                # 如果结构化输出失败，回退到文本解析
                if i < max_tries - 1:
                    try:
                        response_text = self.llm.chat(prompt)
                        interaction.update(json_parser(response_text))
                        break
                    except Exception as e2:
                        print(f"[{self.role_name}] 文本解析也失败: {e2}")
                        continue
                else:
                    # 最后一次尝试，使用默认值
                    print(f"[{self.role_name}] 所有尝试都失败，使用默认值")
        self.save_prompt(detail = interaction["detail"], prompt=prompt)
        return interaction
    
    def update_status(self,):
        prompt = self._UPDATE_STATUS_PROMPT.format(**{
            "role_name":self.role_name,
            "status":self.status,
            "history_text":self.retrieve_history(query=""),
            "activity":self.activity
        })
        max_tries = 3
        for i in range(max_tries):
            response = self.llm.chat(prompt) 
            try:
                status = json_parser(response)
                self.status = status["updated_status"]
                self.activity = float(status["activity"])
                break
            except Exception as e:
                print(f"Parsing failure! {i}th tries. Error:", e)    
                print(response)
        
        return
    
    def update_goal(self,other_roles_status: str,instruction: str = ""):
        motivation = self.motivation
        if instruction:
            motivation = instruction
        history = self.retrieve_history(self.motivation)
        if len(history) == 0:
            self.goal = motivation
            return motivation
        
        prompt = self._UPDATE_GOAL_PROMPT.format(**{
            "history":history,
            "motivation":motivation,
            "goal":self.goal,
            "other_roles_status":other_roles_status,
            "location":self.location_name
        })
        try:
            # 使用结构化输出
            response_model = self.llm.chat(prompt, response_model=UpdateGoal)
            if response_model.if_change_goal and response_model.updated_goal:
                goal = response_model.updated_goal
                self.save_prompt(prompt, response_model.model_dump_json())
                self.goal = goal
                return goal
        except Exception as e:
            print(f"[{self.role_name}] 结构化输出失败! 错误: {e}")
            # 回退到文本解析
            try:
                response_text = self.llm.chat(prompt)
                new_plan = json_parser(response_text)
                if new_plan.get("if_change_goal") and new_plan.get("updated_goal"):
                    goal = new_plan["updated_goal"]
                    self.save_prompt(prompt, response_text)
                    self.goal = goal
                    return goal
            except Exception as e2:
                print(f"[{self.role_name}] 文本解析也失败: {e2}")
        return ""
    
    def move(self, 
             locations_info_text: str, 
             locations_info: Dict[str, Any]):
        history_text = self.retrieve_history(query="")
        prompt = self._ROLE_MOVE_PROMPT.format(**{
            "role_name":self.role_name,
            "profile": self.role_profile,
            "goal":self.goal,
            "status":self.status,
            "history":history_text,
            "location":self.location_name,
            "locations_info_text":locations_info_text
            
        })
        try:
            # 使用结构化输出
            response_model = self.llm.chat(prompt, response_model=MoveResponse)
            if response_model.if_move and response_model.destination_code:
                destination_code = response_model.destination_code
                if destination_code in locations_info and destination_code != self.location_code:
                    self.save_prompt(detail=response_model.detail, prompt=prompt)
                    return True, response_model.detail, destination_code
        except Exception as e:
            print(f"[{self.role_name}] 结构化输出失败! 错误: {e}")
            # 回退到文本解析
            try:
                response_text = self.llm.chat(prompt)
                result = json_parser(response_text)
                if result.get("if_move") and result.get("destination_code"):
                    destination_code = result["destination_code"]
                    if destination_code in locations_info and destination_code != self.location_code:
                        self.save_prompt(detail=result["detail"], prompt=prompt)
                        return True, result["detail"], destination_code
            except Exception as e2:
                print(f"[{self.role_name}] 文本解析也失败: {e2}")
        return False, "", self.location_code
    
    def record(self, 
                record):
        self.history_manager.add_record(record)
        
    def save_prompt(self,prompt,detail):
        if prompt:
            self.prompts.append({"prompt":prompt,
                                 "response":detail})
    # Other
    def action_check(self,):
        if self.acted == False:
            self.acted = True
            return True
        dice = random.uniform(0,1)
        if dice > self.activity:
            self.acted = False
            return False
        return True
    
    def retrieve_knowledges(self, query:str, top_k:int=1, max_words = 100):
        if self.world_db is None:
            return ""
        knowledges = "\n".join(self.world_db.search(query, top_k,self.world_db_name))
        knowledges = knowledges[:max_words]
        return knowledges
    
    def retrieve_references(self, query: str, top_k: int = 1):
        if self.db is None:
            return ""
        references = "\n".join(self.db.search(query, top_k,self.db_name))
        return references
    
    def retrieve_history(self, query: str, top_k: int = 5, retrieve: bool = False):
        if len(self.history_manager) == 0: return ""
        if len(self.history_manager) >= top_k and retrieve:
            history = "\n" + "\n".join(self.memory.search(query, top_k)) + "\n"
        else:
            history = "\n" + "\n".join(self.history_manager.get_recent_history(top_k))
        return history
        
    def get_other_roles_info_text(self, other_roles: List[str], if_relation: bool = True, if_profile: bool = True):
        roles_info_text = ""
        for i, role_code in enumerate(other_roles):
            if role_code == self.role_code :continue
            name = other_roles[role_code]["nickname"]
            profile = other_roles[role_code]["profile"]  if if_profile else ""
            relation = self.search_relation(role_code) if if_relation else ""
            roles_info_text += f"\n{i+1}. {name}\nrole_code:{role_code}\n{relation}\n{profile}\n\n"

        return roles_info_text
    
    def search_relation(self, other_role_code: str):
        if self.language == 'en':
            if other_role_code in self.relation:
                relation_text = ",".join(self.relation[other_role_code]["relation"])
                detail_text = self.relation[other_role_code]["detail"]
                return f"This is your {relation_text}. {detail_text}\n"
            else:
                return ""
        elif self.language == 'zh':
            if other_role_code in self.relation:
                relation_text = ",".join(self.relation[other_role_code]["relation"])
                detail_text = self.relation[other_role_code]["detail"]
                return f"这是你的{relation_text}. {detail_text}\n"
            else:
                return ""
    def set_location(self, location_code, location_name):
        self.location_code: Optional[str] = location_code
        self.location_name: Optional[str] = location_name
    
    # ========== 三层人格模型相关方法 ==========
    
    def _format_big_five_info(self) -> str:
        """格式化Big Five信息用于prompt"""
        if not self.personality_profile:
            return ""
        
        big_five = self.personality_profile.core_traits.big_five
        if self.language == "zh":
            return f"""
## 大五人格特征（影响你的思维和决策）：
- 开放性(Openness): {big_five['openness']:.2f} - {'高' if big_five['openness'] > 0.7 else '低' if big_five['openness'] < 0.4 else '中'} - 影响你对新想法和可能性的接受程度
- 尽责性(Conscientiousness): {big_five['conscientiousness']:.2f} - {'高' if big_five['conscientiousness'] > 0.7 else '低' if big_five['conscientiousness'] < 0.4 else '中'} - 影响你的组织性和责任感
- 外向性(Extraversion): {big_five['extraversion']:.2f} - {'高' if big_five['extraversion'] > 0.7 else '低' if big_five['extraversion'] < 0.4 else '中'} - 影响你的社交性和积极性
- 宜人性(Agreeableness): {big_five['agreeableness']:.2f} - {'高' if big_five['agreeableness'] > 0.7 else '低' if big_five['agreeableness'] < 0.4 else '中'} - 影响你的合作性和同理心
- 神经质(Neuroticism): {big_five['neuroticism']:.2f} - {'高' if big_five['neuroticism'] > 0.7 else '低' if big_five['neuroticism'] < 0.4 else '中'} - 影响你的情绪稳定性
"""
        else:
            return f"""
## Big Five Personality Traits (affecting your thinking and decisions):
- Openness: {big_five['openness']:.2f} - {'High' if big_five['openness'] > 0.7 else 'Low' if big_five['openness'] < 0.4 else 'Medium'}
- Conscientiousness: {big_five['conscientiousness']:.2f} - {'High' if big_five['conscientiousness'] > 0.7 else 'Low' if big_five['conscientiousness'] < 0.4 else 'Medium'}
- Extraversion: {big_five['extraversion']:.2f} - {'High' if big_five['extraversion'] > 0.7 else 'Low' if big_five['extraversion'] < 0.4 else 'Medium'}
- Agreeableness: {big_five['agreeableness']:.2f} - {'High' if big_five['agreeableness'] > 0.7 else 'Low' if big_five['agreeableness'] < 0.4 else 'Medium'}
- Neuroticism: {big_five['neuroticism']:.2f} - {'High' if big_five['neuroticism'] > 0.7 else 'Low' if big_five['neuroticism'] < 0.4 else 'Medium'}
"""
    
    def _format_speaking_style_info(self) -> str:
        """格式化语言风格信息用于prompt"""
        if not self.personality_profile:
            return ""
        
        style = self.personality_profile.speaking_style
        if self.language == "zh":
            emoji_info = ""
            if style.emoji_usage["frequency"] != "none":
                preferred = ", ".join(style.emoji_usage.get("preferred", []))
                if preferred:
                    emoji_info = f"\n- 表情使用：{style.emoji_usage['frequency']}，常用表情：{preferred}"
            
            return f"""
## 语言风格要求（严格遵守）：
- 句长偏好：{style.sentence_length}（{'短句为主' if style.sentence_length == 'short' else '长句为主' if style.sentence_length == 'long' else '中等长度' if style.sentence_length == 'medium' else '混合'}）
- 词汇等级：{style.vocabulary_level}（{'学术/正式' if style.vocabulary_level == 'academic' else '口语化' if style.vocabulary_level == 'casual' else '网络用语' if style.vocabulary_level == 'network' else '混合'}）
- 标点习惯：{style.punctuation_habit}（{'少用标点' if style.punctuation_habit == 'minimal' else '标准使用' if style.punctuation_habit == 'standard' else '频繁使用' if style.punctuation_habit == 'excessive' else '混合'}）
- 语气词：{', '.join(style.tone_markers) if style.tone_markers else '无'}
- 口头禅：{', '.join(style.catchphrases) if style.catchphrases else '无'}{emoji_info}
"""
        else:
            emoji_info = ""
            if style.emoji_usage["frequency"] != "none":
                preferred = ", ".join(style.emoji_usage.get("preferred", []))
                if preferred:
                    emoji_info = f"\n- Emoji usage: {style.emoji_usage['frequency']}, preferred: {preferred}"
            
            return f"""
## Speaking Style Requirements (strictly follow):
- Sentence length: {style.sentence_length}
- Vocabulary level: {style.vocabulary_level}
- Punctuation habit: {style.punctuation_habit}
- Tone markers: {', '.join(style.tone_markers) if style.tone_markers else 'none'}
- Catchphrases: {', '.join(style.catchphrases) if style.catchphrases else 'none'}{emoji_info}
"""
    
    def _format_style_examples(self) -> str:
        """格式化Few-Shot样本用于prompt"""
        if not self.style_examples:
            return ""
        
        if self.language == "zh":
            examples_text = "\n## 参考样本（Few-Shot Examples，模仿这些风格）：\n"
            for i, ex in enumerate(self.style_examples[:5], 1):
                examples_text += f"\n样本{i}:\nContext: {ex.get('context', '')}\nResponse: {ex.get('response', '')}\n"
            return examples_text
        else:
            examples_text = "\n## Reference Examples (Few-Shot, imitate this style):\n"
            for i, ex in enumerate(self.style_examples[:5], 1):
                examples_text += f"\nExample {i}:\nContext: {ex.get('context', '')}\nResponse: {ex.get('response', '')}\n"
            return examples_text
    
    def _save_personality_profile(self):
        """保存更新后的personality_profile到role_info.json"""
        if not self.personality_profile:
            return
        
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            role_info_path = None
            
            # 查找role_info.json路径
            for path in get_grandchild_folders(os.path.join(base_dir, "data", "roles")):
                if self.role_code in path:
                    role_info_path = os.path.join(base_dir, path, "role_info.json")
                    break
            
            if role_info_path and os.path.exists(role_info_path):
                role_info = load_json_file(role_info_path)
                role_info["personality_profile"] = self.personality_profile.to_dict()
                save_json_file(role_info_path, role_info)
        except Exception as e:
            print(f"Warning: Failed to save personality_profile: {e}")

            
    def __getstate__(self):
        states = {key: value for key, value in self.__dict__.items() \
            if isinstance(value, (str, int, list, dict, bool, type(None))) \
                and key not in ['role_info','role_data','llm','embedding','db',"memory"]
                and "PROMPT" not in key}
        return states

    def __setstate__(self, states):
        self.__dict__.update(states)
        self._init_prompt()

    def save_to_file(self, root_dir):
        filename = os.path.join(root_dir, f"./roles/{self.role_code}.json")
        save_json_file(filename, self.__getstate__() )

    def load_from_file(self, root_dir):
        filename = os.path.join(root_dir, f"./roles/{self.role_code}.json")
        states = load_json_file(filename)
        self.__setstate__(states)     
        self.memory.init_from_data(self.history_manager.get_complete_history())

def build_performer_data(role_dir: str):
    role_data: List[str] = []
    for path in get_child_paths(role_dir):
        if os.path.splitext(path)[-1] == ".txt":
            text = load_text_file(path)
            role_data += split_text_by_max_words(text)
        elif os.path.splitext(path)[-1] == ".jsonl":
            role_data += [line["text"] for line in load_jsonl_file(path)]
    return role_data      


if __name__ == "__main__":
    agent = Performer(role_code='Harry-en')
    agent.single_role_interact("Hi,Harry, Who is Ron?")


    