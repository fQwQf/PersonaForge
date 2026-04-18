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
sys.path.append("../")
import csv
from typing import Any, Dict, List, Optional, Literal
from sw_utils import *
from modules.embedding import get_embedding_model
from modules.utils.map_manager import MapDataManager

class Orchestrator:
    # Init
    def __init__(self, 
                 world_file_path: str,
                 location_file_path: str,
                 map_file_path: Optional[str] = "",
                 world_description: str = "",
                 llm_name: str = "gpt-4o-mini",
                 llm = None,
                 embedding_name: str = "bge-small",
                 embedding = None,
                 db_type: str = "chroma",
                 language: str = "zh",
                 ):
        if llm is None:
            llm = get_models(llm_name)
        if embedding is None:
            embedding = get_embedding_model(embedding_name, language=language)
        self.llm = llm
        self.world_info: Dict[str, Any] = load_json_file(world_file_path)
        self.world_name: str = self.world_info["world_name"]
        self.language: str = language
        self.description:str = self.world_info["description"] if world_description == "" else world_description
        source = self.world_info["source"]
        
        self.locations_info: Dict[str, Any] = {}  
        self.locations: List[str] = []
        self.history: List[str] = []
        self.edges: Dict[tuple, int] = {}  # 地点间距离
        self.prompts: List[Dict] = []
        
        self.init_from_file(map_file_path = map_file_path,
                            location_file_path = location_file_path)
        self.init_prompt()

            
        self.world_data,self.world_settings = build_orchestrator_data(world_file_path = world_file_path, max_words = 50)
        self.db_name = clean_collection_name(f"settings_{source}_{embedding_name}")
        self.db = build_db(data = [row for row in self.world_data], 
                           db_name = self.db_name, 
                           db_type = db_type, 
                           embedding = embedding)
        
    def init_from_file(self, map_file_path: str, location_file_path: str, default_distance: int = 1):
        source = self.world_info.get("source")
        if source:
            # 尝试通过 MapDataManager 加载新格式
            map_data = MapDataManager.load_map_data(source)
            if map_data and map_data.get("locations"):
                print(f"[Orchestrator] Loading map from new format for source: {source}")
                for loc_data in map_data["locations"]:
                    code = loc_data["code"]
                    self.locations_info[code] = {
                        "location_code": code,
                        "location_name": loc_data.get("name", code),
                        "description": loc_data.get("description", ""),
                        "detail": loc_data.get("detail", "")
                    }
                    self.locations.append(code)
                    for adj in loc_data.get("adjacencies", []):
                        self._add_edge(code, adj["to"], adj["distance"])
                return

        # 降级到原有的 CSV/JSON 加载逻辑
        if map_file_path and os.path.exists(map_file_path):
            valid_locations = load_json_file(location_file_path) if "locations" not in load_json_file(location_file_path) else load_json_file(location_file_path)["locations"]
            with open(map_file_path, mode='r',encoding="utf-8") as file:
                csv_reader = csv.reader(file)
                locations = next(csv_reader)[1:]  
                for row in csv_reader:
                    loc1 = row[0]
                    if loc1 not in valid_locations:
                        print(f"Warning: The location {loc1} does not exist")
                        continue
                    self.locations_info[loc1] = valid_locations[loc1]
                    self.locations.append(loc1)
                    distances = row[1:]
                    for i, distance in enumerate(distances):
                        loc2 = locations[i]
                        if loc2 not in valid_locations:
                            print(f"Warning: The location {loc2} does not exist")
                            continue
                        if distance != '0':  # Skip self-loops
                            self._add_edge(loc1, loc2, int(distance))
        else:
            valid_locations = load_json_file(location_file_path) if "locations" not in load_json_file(location_file_path) else load_json_file(location_file_path)["locations"]
            for loc1 in valid_locations:
                self.locations_info[loc1] = valid_locations[loc1]
                self.locations.append(loc1)
                for loc2 in valid_locations:
                    if loc2 != loc1:
                        self._add_edge(loc1, loc2, default_distance)
                        
    def init_prompt(self,):
        if self.language == "zh":
            from modules.prompt.orchestrator_prompt_zh import ENVIROMENT_INTERACTION_PROMPT,NPC_INTERACTION_PROMPT,SCRIPT_INSTRUCTION_PROMPT,SCRIPT_ATTENTION_PROMPT,DECIDE_NEXT_ACTOR_PROMPT,GENERATE_INTERVENTION_PROMPT,UPDATE_EVENT_PROMPT,LOCATION_PROLOGUE_PROMPT,SELECT_SCREEN_ACTORS_PROMPT,JUDGE_IF_ENDED_PROMPT,LOG2STORY_PROMPT,SOUL_TRANS_PROLOGUE_PROMPT
        else:
            from modules.prompt.orchestrator_prompt_en import ENVIROMENT_INTERACTION_PROMPT,NPC_INTERACTION_PROMPT,SCRIPT_INSTRUCTION_PROMPT,SCRIPT_ATTENTION_PROMPT,DECIDE_NEXT_ACTOR_PROMPT,GENERATE_INTERVENTION_PROMPT,UPDATE_EVENT_PROMPT,LOCATION_PROLOGUE_PROMPT,SELECT_SCREEN_ACTORS_PROMPT,JUDGE_IF_ENDED_PROMPT,LOG2STORY_PROMPT
            self._SOUL_TRANS_PROLOGUE_PROMPT = "" # English version not yet implemented
            
        self._ENVIROMENT_INTERACTION_PROMPT = ENVIROMENT_INTERACTION_PROMPT
        self._NPC_INTERACTION_PROMPT = NPC_INTERACTION_PROMPT
        self._SCRIPT_INSTRUCTION_PROMPT = SCRIPT_INSTRUCTION_PROMPT
        self._SCRIPT_ATTENTION = SCRIPT_ATTENTION_PROMPT
        self._DECIDE_NEXT_ACTOR_PROMPT= DECIDE_NEXT_ACTOR_PROMPT
        self._LOCATION_PROLOGUE_PROMPT = LOCATION_PROLOGUE_PROMPT
        self._GENERATE_INTERVENTION_PROMPT = GENERATE_INTERVENTION_PROMPT
        self._UPDATE_EVENT_PROMPT = UPDATE_EVENT_PROMPT
        self._SELECT_SCREEN_ACTORS_PROMPT = SELECT_SCREEN_ACTORS_PROMPT
        self._JUDGE_IF_ENDED_PROMPT = JUDGE_IF_ENDED_PROMPT
        self._LOG2STORY_PROMPT = LOG2STORY_PROMPT
        self._SOUL_TRANS_PROLOGUE_PROMPT = SOUL_TRANS_PROLOGUE_PROMPT
        
    # Agent
    def update_event(self, 
                     cur_event: str, 
                     intervention:str,
                     history_text: str, 
                     script: str = ""):
        from .models import EventText
        prompt = self._UPDATE_EVENT_PROMPT.format(**{
            "event":cur_event,
            "intervention":intervention,
            "history":history_text
        })
        if script:
            prompt = self._SCRIPT_ATTENTION.format(script = script) + prompt
        try:
            # 使用结构化输出
            response_model = self.llm.chat(prompt, response_model=EventText)
            new_event = response_model.event
        except Exception as e:
            print(f"[Orchestrator] 事件更新结构化输出失败: {e}")
            # 回退到文本输出
            try:
                new_event = self.llm.chat(prompt)
                if not isinstance(new_event, str):
                    new_event = str(new_event)
            except Exception as e2:
                print(f"[Orchestrator] 文本输出也失败: {e2}")
                new_event = "故事正在继续发展。" if self.language == "zh" else "The story continues to develop."
        # 确保new_event不为空
        if not new_event or not new_event.strip():
            new_event = "故事正在继续发展。" if self.language == "zh" else "The story continues to develop."
        self.record(new_event, prompt)
        return new_event
    
    def decide_next_actor(self, 
                          history_text: str, 
                          roles_info_text: str,
                          script: str = "",
                          event:str = "",
                          recent_speakers: List[str] = None):
        prompt = self._DECIDE_NEXT_ACTOR_PROMPT.format(**{
            "roles_info":roles_info_text,
            "history_text":history_text,
        })
        
        # 如果有最近发言的角色，添加到提示中
        if recent_speakers and len(recent_speakers) > 0:
            avoid_instruction = (
                f"\n请注意，以下角色最近刚刚发言过：{', '.join(recent_speakers)}。"
                "为了保持对话流畅，请尽量选择其他角色发言，除非当前情境必须由该角色回应。"
                if self.language == "zh" else
                f"\nNote that the following characters have just spoken: {', '.join(recent_speakers)}."
                "To keep the conversation flowing, please try to choose other characters to speak, unless the current situation requires a response from that character."
            )
            prompt += avoid_instruction
        
        max_tries = 3
        response = None  # 初始化response
        for i in range(max_tries):
            try:
                response = self.llm.chat(prompt)
                # 确保response不为空字符串
                if response and response.strip():
                    break
                else:
                    print(f"第{i+1}次尝试返回空响应，继续重试...")
            except Exception as e:
                print(f"Parsing failure! 第{i+1}次尝试失败. Error:", e)
                if response is not None:
                    print(f"Previous response: {response}")
        
        # 如果所有尝试都失败或返回空值，返回None而不是空字符串
        if not response or not response.strip():
            print(f"[Orchestrator] decide_next_actor 所有尝试都失败或返回空值，返回None")
            return None
        
        role_code = response.strip()
        self.prompts.append({"prompt":prompt,
                            "response":f"{role_code}"})

        return role_code
    
    def judge_if_ended(self,history_text):
        from .models import JudgeIfEnded
        prompt = self._JUDGE_IF_ENDED_PROMPT.format(**{
            "history":history_text
        })
        max_tries = 3
        response = {"if_end":True, "detail":""}
        for i in range(max_tries):
            try:
                # 使用结构化输出
                response_model = self.llm.chat(prompt, response_model=JudgeIfEnded)
                response.update(response_model.model_dump())
                break
            except Exception as e:
                print(f"结构化输出失败! 第{i+1}次尝试. 错误: {e}")
                # 回退到文本解析
                if i < max_tries - 1:
                    try:
                        response_text = self.llm.chat(prompt)
                        response.update(json_parser(response_text))
                        break
                    except Exception as e2:
                        print(f"文本解析也失败: {e2}")
                        continue
                else:
                    print(f"所有尝试都失败，使用默认值")
        
        return response["if_end"],response["detail"]
        
    def decide_scene_actors(self,roles_info_text, history_text, event, previous_role_codes):
        from .models import SceneActors
        prompt = self._SELECT_SCREEN_ACTORS_PROMPT.format(**{
            "roles_info":roles_info_text,
            "history_text":history_text,
            "event":event,
            "previous_role_codes":previous_role_codes
            
        })
        try:
            # 使用结构化输出
            response_model = self.llm.chat(prompt, response_model=SceneActors)
            return response_model.role_codes
        except Exception as e:
            print(f"[Orchestrator] 结构化输出失败: {e}")
            # 回退到文本解析
            try:
                response_text = self.llm.chat(prompt)
                if not isinstance(response_text, str):
                    response_text = str(response_text)
                import json
                role_codes = json.loads(response_text)
                if isinstance(role_codes, list):
                    return role_codes
                elif isinstance(role_codes, dict) and "role_codes" in role_codes:
                    return role_codes["role_codes"]
                else:
                    return []
            except Exception as e2:
                print(f"[Orchestrator] 文本解析也失败: {e2}")
                return []
    
    def generate_location_prologue(self,
                                   location_code,
                                   history_text,
                                   event,
                                   location_info_text):
        prompt = self._LOCATION_PROLOGUE_PROMPT.format(**{
            "location_name":self.locations_info[location_code]["location_name"],
            "location_description":self.locations_info[location_code]["location_name"],
            "location_info":location_info_text,
            "history_text":history_text,
            "event":event,
            "world_description":self.description
        })
        try:
            response = self.llm.chat(prompt)
            self.record(detail = response,prompt = prompt)
            return "\n"+response
        except Exception as e:
            print(f"[Orchestrator] generate_location_prologue 失败: {e}")
            location_name = self.locations_info[location_code]["location_name"]
            default_response = f"场景转至{location_name}..." if self.language == "zh" else f"The scene shifts to {location_name}..."
            self.record(detail=default_response, prompt=prompt)
            return "\n"+default_response
    
    def generate_soul_trans_prologue(self, 
                                     role_name, 
                                     role_profile, 
                                     motivation, 
                                     location_name, 
                                     location_description, 
                                     other_roles_text):
        if not self._SOUL_TRANS_PROLOGUE_PROMPT:
            return "意识正在苏醒..." if self.language == "zh" else "Consciousness awakening..."
            
        prompt = self._SOUL_TRANS_PROLOGUE_PROMPT.format(**{
            "world_description": self.description,
            "location_name": location_name,
            "location_description": location_description,
            "role_name": role_name,
            "role_profile": role_profile,
            "motivation": motivation,
            "other_roles": other_roles_text
        })
        
        try:
            response = self.llm.chat(prompt)
            self.record(detail=response, prompt=prompt)
            return response
        except Exception as e:
            print(f"[Orchestrator] generate_soul_trans_prologue 失败: {e}")
            # 返回默认的觉醒描述
            default_response = (
                f"意识从虚无中苏醒，你发现自己置身于{location_name}。"
                f"作为{role_name}，你感到一种陌生又熟悉的感觉涌上心头。"
                f"周围的一切开始变得清晰..."
            ) if self.language == "zh" else (
                f"Consciousness awakens from the void. You find yourself in {location_name}. "
                f"As {role_name}, a strange yet familiar feeling washes over you. "
                f"Everything around you begins to come into focus..."
            )
            self.record(detail=default_response, prompt=prompt)
            return default_response
    
    def enviroment_interact(self, 
                            action_maker_name: str, 
                            action: str,
                            action_detail: str, 
                            location_code: str):
        references = self.retrieve_references(query = action_detail)
        prompt = self._ENVIROMENT_INTERACTION_PROMPT.format(**
            {
                "role_name":action_maker_name,
                "action":action,
                "action_detail":action_detail,
                "world_description":self.description,
                "location":location_code,
                "location_description":self.locations_info[location_code]["detail"],
                "references":references,
            }
            )
        response = "无事发生。" if self.language == "zh" else "Nothing happens."
        for i in range(3):
            try:
                response = self.llm.chat(prompt) 
                if response:
                    break
            except Exception as e:
                print("Enviroment Interaction failed! {i}th tries. Error:", e)
        self.record(response, prompt)
        return response
    
    
    def npc_interact(self, 
                     action_maker_name: str, 
                     action_detail: str, 
                     location_name: str,
                     target_name: str):
        references = self.retrieve_references(query = action_detail)
        prompt = self._NPC_INTERACTION_PROMPT.format(**
            {
                "role_name":action_maker_name,
                "action_detail":action_detail,
                "world_description":self.description,
                "target":target_name,
                "references":references,
                "location":location_name
            }
            )
        
        from .models import NPCRoleResponse
        default_detail = "无事发生。" if self.language == "zh" else "Nothing happens"
        npc_interaction = {"if_end_interaction": True, "detail": default_detail}
        try:
            # 使用结构化输出
            response_model = self.llm.chat(prompt, response_model=NPCRoleResponse)
            npc_interaction = response_model.model_dump()
            response = npc_interaction["detail"]
            self.record(response, prompt)
        except Exception as e:
            print(f"[Orchestrator] NPC交互结构化输出失败: {e}")
            # 回退到文本解析
            try:
                response_text = self.llm.chat(prompt)
                npc_interaction = json_parser(response_text)
                response = npc_interaction.get("detail", default_detail)
                self.record(response, prompt)
            except Exception as e2:
                print(f"[Orchestrator] NPC交互文本解析也失败: {e2}")
        
        return npc_interaction
    
    
    def get_script_instruction(self, 
                               roles_info_text: str, 
                               event: str, 
                               history_text: str, 
                               script: str, 
                               last_progress: str):
        prompt = self._SCRIPT_INSTRUCTION_PROMPT.format(**{
            "roles_info":roles_info_text,
            "event":event,
            "history_text":history_text,
            "script":script,
            "last_progress":last_progress
        })
        from .models import ScriptInstruction
        max_tries = 3
        instruction = {}
        for i in range(max_tries):
            try:
                # 使用结构化输出
                response_model = self.llm.chat(prompt, response_model=ScriptInstruction)
                instruction = response_model.model_dump()
                break
            except Exception as e:
                print(f"[Orchestrator] 结构化输出失败! 第{i+1}次尝试. 错误: {e}")
                # 回退到文本解析
                if i < max_tries - 1:
                    try:
                        response_text = self.llm.chat(prompt)
                        instruction = json_parser(response_text)
                        break
                    except Exception as e2:
                        print(f"[Orchestrator] 文本解析也失败: {e2}")
                        continue
                else:
                    print(f"[Orchestrator] 所有尝试都失败，使用默认值")
                    instruction = {"progress": "故事正在继续发展。" if self.language == "zh" else "The story continues to develop."}
        # 记录响应（使用最后一次尝试的响应）
        if 'response' in locals():
            self.record(response if isinstance(response, str) else str(response), prompt)
        return instruction
    
    def generate_event(self,roles_info_text: str, event: str, history_text: str):
        from .models import EventText
        prompt = self._GENERATE_INTERVENTION_PROMPT.format(**{
            "world_description":self.description,
            "roles_info":roles_info_text,
            "history_text":history_text
        })
        print(f"[Orchestrator] 开始生成事件，角色数量: {len(roles_info_text.split('角色')) if roles_info_text else 0}")
        try:
            # 使用结构化输出
            response_model = self.llm.chat(prompt, response_model=EventText)
            response = response_model.event
            print(f"[Orchestrator] 事件生成成功（结构化输出）: {response[:100] if response else 'None'}...")
        except Exception as e:
            print(f"[Orchestrator] 事件生成结构化输出失败: {e}")
            import traceback
            traceback.print_exc()
            # 回退到文本输出
            try:
                response = self.llm.chat(prompt)
                if not isinstance(response, str):
                    response = str(response)
                print(f"[Orchestrator] 事件生成成功（文本输出）: {response[:100] if response else 'None'}...")
            except Exception as e2:
                print(f"[Orchestrator] 文本输出也失败: {e2}")
                import traceback
                traceback.print_exc()
                response = "故事正在继续发展。" if self.language == "zh" else "The story continues to develop."
                print(f"[Orchestrator] 使用默认事件: {response}")
        # 确保response不为空
        if not response or not response.strip():
            print(f"[Orchestrator] 警告: 事件为空，使用默认值")
            response = "故事正在继续发展。" if self.language == "zh" else "The story continues to develop."
        self.record(response, prompt)
        return response
        
    def generate_script(self, roles_info_text: str, event: str, history_text: str):
        from .models import ScriptText
        prompt = self._GENERATE_INTERVENTION_PROMPT.format(**{
            "world_description":self.description,
            "roles_info":roles_info_text,
            "history_text":history_text
        })
        try:
            # 使用结构化输出
            response_model = self.llm.chat(prompt, response_model=ScriptText)
            response = response_model.script
        except Exception as e:
            print(f"[Orchestrator] 脚本生成结构化输出失败: {e}")
            # 回退到文本输出
            try:
                response = self.llm.chat(prompt)
                if not isinstance(response, str):
                    response = str(response)
            except Exception as e2:
                print(f"[Orchestrator] 文本输出也失败: {e2}")
                response = "场景描述。" if self.language == "zh" else "Scene description."
        self.record(response, prompt)
        return response
    
    def log2story(self,logs):
        import re
        # 将日志列表转换为字符串，每个记录用换行符分隔
        if isinstance(logs, list):
            logs_text = "\n".join(str(log) for log in logs)
        else:
            logs_text = str(logs)
        prompt = self._LOG2STORY_PROMPT.format(**{
            "logs":logs_text
        })
        try:
            # 直接使用文本输出，不使用结构化输出
            response = self.llm.chat(prompt)
            if not isinstance(response, str):
                response = str(response)
        except Exception as e:
            print(f"[Orchestrator] 故事生成文本输出失败: {e}")
            import traceback
            traceback.print_exc()
            response = "故事正在继续发展。" if self.language == "zh" else "The story continues to develop."
        
        # 注意：后处理逻辑已移至server.py的故事输出部分
        # 这里只返回原始响应，保持log2story函数的通用性
        return response
    
    # Other
    def record(self, detail: str, prompt: str = ""):
        if prompt:
            self.prompts.append({"prompt":prompt,
                                 "response":detail})
        self.history.append(detail)
    
    def add_location_during_simulation(self, location: str, detail: str):
        self.locations.append(location)
        self.locations_info[location] = {
            'location_code': location,
            "location_name": location,
            'description': '',
            'detail':detail
        }
        for loc in self.locations:
            if loc != location:
                self._add_edge(loc, location, 1)
                self._add_edge(location,loc, 1)
        return
    
    def retrieve_references(self, query: str, top_k = 3, max_words = 100):
        if self.db is None:
            return ""
        references = "\n".join(self.db.search(query, top_k,self.db_name))
        references = references[:max_words]
        return references

    def find_location_name(self, code: str):
        """查找地点名称，如果代码不存在则返回代码本身"""
        if code in self.locations_info:
            return self.locations_info[code]["location_name"]
        else:
            # 如果地点代码不存在，返回代码本身作为后备方案
            print(f"[Orchestrator] 警告: 地点代码 '{code}' 不存在于 locations_info 中，使用代码本身作为名称")
            return code
              
    def _add_location(self, code: str, location_info: Dict[str, Any]):
        self.locations_info[code] = location_info
        
    def _add_edge(self, code1: str, code2: str, distance: int):
        self.edges[(code1,code2)] = distance
        self.edges[(code2,code1)] = distance  
        
    def get_distance(self, code1: str, code2: str):
        if (code1,code2) in self.edges:
            return self.edges[(code1,code2)]
        else:
            return None
        
    def __getstate__(self):
        state = {key: value for key, value in self.__dict__.items() 
                 if isinstance(value, (str, int, list, dict, float, bool, type(None)))
                 and (key not in ['llm','embedding','db','locations_info','edges','world_data','world_settings']
                 and "PROMPT" not in key)
                 }
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def save_to_file(self, root_dir):
        filename = os.path.join(root_dir, f"./orchestrator.json")
        save_json_file(filename, self.__getstate__() )

    def load_from_file(self, root_dir):
        filename = os.path.join(root_dir, f"./orchestrator.json")
        state = load_json_file(filename)
        self.__setstate__(state)  

