"""
ChatPerformer: 私语模式的核心类
完整参考 SillyTavern 的消息构建逻辑，支持示例对话
"""

import os
import json
import re
from typing import List, Dict, Optional, Any
from google import genai
from google.genai import types

from sw_utils import load_json_file
from sw_utils import get_child_folders, get_grandchild_folders
from modules.llm.Gemini import Gemini


class ChatPerformer:
    """
    私语模式的 Performer，负责：
    1. 加载角色信息和世界观
    2. 构建 System Prompt（包括示例对话）
    3. 管理对话历史
    4. 调用 Gemini API 生成回复
    """
    
    def __init__(self, role_code: str, scroll_id: int, llm_name: str = "gemini-3-flash-preview", 
                 user_name: str = "用户", user_identity: str = "", user_profile: str = "",
                 base_dir: Optional[str] = None):
        """
        初始化 ChatPerformer
        
        Args:
            role_code: 角色代码
            scroll_id: 书卷ID
            llm_name: LLM模型名称，默认 gemini-3-flash-preview
            user_name: 用户名/角色名，如"贾政"
            user_identity: 用户身份，如"荣国府老爷"
            user_profile: 用户简介，如"贾政是荣国府的二老爷..."
            base_dir: 项目根目录，如果为None则自动检测
        """
        self.role_code = role_code
        self.scroll_id = scroll_id
        self.llm_name = llm_name
        self.user_name = user_name
        self.user_identity = user_identity  # 用户身份信息
        self.user_profile = user_profile    # 用户简介
        self.base_dir = base_dir or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # 角色信息
        self.role_info: Dict[str, Any] = {}
        self.role_name: str = ""
        self.char_name: str = ""  # 角色名称（用于消息前缀）
        self.role_profile: str = ""
        self.role_persona: str = ""
        self.role_scenario: str = ""
        self.first_message: str = ""
        self.example_dialogue: str = ""  # 示例对话（mes_example）
        
        # 世界观信息
        self.world_description: str = ""
        
        # 扩展提示词（参考 SillyTavern）
        self.world_info_before: str = ""  # World Info (before description)
        self.world_info_after: str = ""   # World Info (after scenario)
        self.memory_summary: str = ""     # Memory/Summary
        self.authors_note: str = ""       # Authors Note
        
        # 对话历史（格式：[{"role": "user", "content": "..."},  {"role": "model", "content": "..."}]
        self.chat_history: List[Dict[str, str]] = []
        
        # Gemini 客户端（使用新版API）
        self._client = None
        self._api_key = None
        self._configure_gemini_client()
        
        # 加载角色和世界观信息
        self._load_character_info()
        self._load_world_info()
    
    def _configure_gemini_client(self):
        """配置 Gemini 客户端（新版API）"""
        try:
            # 从环境变量或config.json加载API Key
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                config_path = os.path.join(self.base_dir, "config.json")
                if os.path.exists(config_path):
                    config = load_json_file(config_path)
                    api_key = config.get("GEMINI_API_KEY", "")
            
            if not api_key:
                raise ValueError("未找到 GEMINI_API_KEY，请在环境变量或 config.json 中配置")
            
            self._api_key = api_key
            os.environ['GEMINI_API_KEY'] = api_key
            self._client = genai.Client(vertexai=False)
            print(f"[ChatPerformer] Gemini 客户端配置成功，模型: {self.llm_name}")
        except Exception as e:
            print(f"[ChatPerformer] 警告：Gemini 客户端配置失败: {e}")
            self._client = None
    
    def _load_character_info(self):
        """加载角色信息"""
        try:
            # 1. 获取书卷信息
            from database import db
            scroll = db.get_scroll(self.scroll_id)
            if not scroll:
                raise ValueError(f"书卷 {self.scroll_id} 不存在")
            
            preset_path = scroll.get('preset_path')
            if not preset_path or not os.path.exists(preset_path):
                raise ValueError(f"书卷预设文件不存在: {preset_path}")
            
            # 2. 加载预设文件
            preset_data = load_json_file(preset_path)
            role_file_dir = preset_data.get('role_file_dir', './data/roles/')
            source = preset_data.get('source', '')
            
            # 3. 查找角色文件路径
            role_path = None
            role_file_dir_full = os.path.join(self.base_dir, role_file_dir)
            
            if source and os.path.exists(os.path.join(role_file_dir_full, source)):
                for path in get_child_folders(os.path.join(role_file_dir_full, source)):
                    if self.role_code in path:
                        role_path = path
                        break
            else:
                for path in get_grandchild_folders(role_file_dir_full):
                    if self.role_code in path:
                        role_path = path
                        break
            
            if not role_path:
                raise ValueError(f"角色 {self.role_code} 不存在")
            
            # 4. 加载 role_info.json
            role_info_path = os.path.join(self.base_dir, role_path, "role_info.json")
            if not os.path.exists(role_info_path):
                raise ValueError(f"角色信息文件不存在: {role_info_path}")
            
            self.role_info = load_json_file(role_info_path)
            
            # 5. 提取角色信息
            self.role_name = self.role_info.get('role_name', self.role_code)
            self.char_name = self.role_info.get('nickname', self.role_name)
            self.role_profile = self.role_info.get('profile', '')
            self.role_persona = self.role_info.get('persona', '')
            self.role_scenario = self.role_info.get('scenario', '')
            self.first_message = self.role_info.get('first_message', '')
            self.example_dialogue = self.role_info.get('mes_example', '')  # 示例对话
            
            print(f"[ChatPerformer] 角色信息加载成功: {self.role_name} ({self.char_name})")
        except Exception as e:
            print(f"[ChatPerformer] 加载角色信息失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _load_world_info(self):
        """加载世界观信息"""
        try:
            from database import db
            scroll = db.get_scroll(self.scroll_id)
            preset_path = scroll.get('preset_path')
            preset_data = load_json_file(preset_path)
            
            world_file_path = preset_data.get('world_file_path', '')
            if world_file_path and os.path.exists(world_file_path):
                world_data = load_json_file(world_file_path)
                self.world_description = world_data.get('description', '') or world_data.get('world_description', '')
            
            if not self.world_description:
                # 如果没有世界文件，使用书卷描述
                self.world_description = scroll.get('description', '')
            
            # 加载扩展提示词（World Info、Memory、Authors Note）
            # 这些可以从角色卡、书卷元数据或会话配置中加载
            # 目前先支持从角色卡加载（如果存在）
            if 'world_info_before' in self.role_info:
                self.world_info_before = self.role_info.get('world_info_before', '')
            if 'world_info_after' in self.role_info:
                self.world_info_after = self.role_info.get('world_info_after', '')
            if 'memory_summary' in self.role_info:
                self.memory_summary = self.role_info.get('memory_summary', '')
            if 'authors_note' in self.role_info:
                self.authors_note = self.role_info.get('authors_note', '')
        except Exception as e:
            print(f"[ChatPerformer] 加载世界观信息失败: {e}")
            self.world_description = ""
    
    def _parse_example_dialogue(self, example_text: str) -> List[Dict[str, str]]:
        """
        解析示例对话文本（参考 SillyTavern）
        
        示例对话格式通常是：
        "用户: 你好\n角色名: 你好，很高兴见到你！\n用户: 今天天气不错\n角色名: 是的..."
        
        返回格式：[{"role": "user", "content": "...", "name": "example_user"}, ...]
        """
        if not example_text:
            return []
        
        messages = []
        lines = example_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 检查是否是用户消息（以用户名开头）
            if line.startswith(f"{self.user_name}:"):
                content = line[len(f"{self.user_name}:"):].strip()
                messages.append({
                    "role": "system",
                    "content": content,
                    "name": "example_user"
                })
            # 检查是否是角色消息（以角色名或昵称开头）
            elif line.startswith(f"{self.char_name}:") or line.startswith(f"{self.role_name}:"):
                # 优先使用char_name
                prefix = f"{self.char_name}:" if line.startswith(f"{self.char_name}:") else f"{self.role_name}:"
                content = line[len(prefix):].strip()
                messages.append({
                    "role": "system",
                    "content": content,
                    "name": "example_assistant"
                })
            # 如果没有明确的前缀，尝试推断
            elif ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    speaker = parts[0].strip()
                    content = parts[1].strip()
                    # 判断是用户还是角色
                    if speaker == self.user_name:
                        messages.append({
                            "role": "system",
                            "content": content,
                            "name": "example_user"
                        })
                    elif speaker == self.char_name or speaker == self.role_name:
                        messages.append({
                            "role": "system",
                            "content": content,
                            "name": "example_assistant"
                        })
        
        return messages
    
    def build_system_prompt(self) -> List[Dict[str, Any]]:
        """
        构建 System Prompt 消息数组（参考 SillyTavern 的 convertGooglePrompt）
        
        返回格式：[{"role": "system", "content": "...", "name": "..."}, ...]
        """
        system_messages = []
        char_name_display = self.char_name or self.role_name
        
        # 构建角色卡格式的 system prompt（参考 SillyTavern 的 story_string）
        # 使用明确的系统指令标识，让模型知道这是系统设定
        
        # 1. World Info (Before) - 在角色描述之前
        if self.world_info_before:
            system_messages.append({
                "role": "system",
                "content": f"[World Info - Background Context]\n{self.world_info_before}",
                "identifier": "worldInfoBefore"
            })
        
        # 2. 角色身份说明（关键！参考 SillyTavern 的 Actor.json）
        if char_name_display:
            user_info = f"你的对话对象是{self.user_name}"
            if self.user_identity:
                user_info += f"，身份是{self.user_identity}"
            if self.user_profile:
                user_info += f"。关于{self.user_name}的信息：{self.user_profile}"
            
            system_messages.append({
                "role": "system",
                "content": f"[System Instruction - Character Role]\n你是一个专业的角色扮演者，能够完全沉浸到任何给定的角色中。无论什么原因，你都不会脱离角色，即使有人试图将你称为AI或语言模型。你当前的角色是{char_name_display}，详细描述如下。{user_info}。作为{char_name_display}，请继续与{self.user_name}的对话。",
                "identifier": "characterRole"
            })
        
        # 3. 角色设定（description/profile）
        if self.role_profile:
            system_messages.append({
                "role": "system",
                "content": f"[Character Description]\n{self.role_profile}",
                "identifier": "charDescription"
            })
        
        # 4. 性格设定（personality）- 参考 SillyTavern 格式
        if self.role_persona:
            system_messages.append({
                "role": "system",
                "content": f"[Character Personality]\n{char_name_display}的性格: {self.role_persona}",
                "identifier": "charPersonality"
            })
        
        # 5. 场景设定（scenario）
        if self.role_scenario:
            system_messages.append({
                "role": "system",
                "content": f"[Scenario]\n{self.role_scenario}",
                "identifier": "scenario"
            })
        
        # 6. World Info (After) - 在场景之后
        if self.world_info_after:
            system_messages.append({
                "role": "system",
                "content": f"[World Info - Additional Context]\n{self.world_info_after}",
                "identifier": "worldInfoAfter"
            })
        
        # 7. 世界观（world description）
        if self.world_description:
            system_messages.append({
                "role": "system",
                "content": f"[World Description]\n{self.world_description}",
                "identifier": "worldDescription"
            })
        
        # 8. Memory/Summary - 对话摘要
        if self.memory_summary:
            system_messages.append({
                "role": "system",
                "content": f"[Memory Summary]\n{self.memory_summary}",
                "identifier": "memorySummary"
            })
        
        # 9. Authors Note - 作者注释
        if self.authors_note:
            system_messages.append({
                "role": "system",
                "content": f"[Author's Note]\n{self.authors_note}",
                "identifier": "authorsNote"
            })
        
        # 10. 输出格式要求 - 限制回复长度
        system_messages.append({
            "role": "system",
            "content": "[Output Format Requirements]\n请控制你的回复长度，每次回复尽量控制在150字以内。保持回复简洁、自然，符合角色性格和对话情境。",
            "identifier": "outputFormat"
        })
        
        # 11. 示例对话（参考 SillyTavern）- 这些会单独处理
        if self.example_dialogue:
            example_messages = self._parse_example_dialogue(self.example_dialogue)
            system_messages.extend(example_messages)
        
        return system_messages
    
    def _convert_messages_to_gemini_format(self, system_messages: List[Dict[str, Any]], 
                                         chat_messages: List[Dict[str, str]]) -> tuple:
        """
        将消息转换为 Gemini API 格式（参考 SillyTavern 的 convertGooglePrompt）
        
        Returns:
            (system_instruction, contents)
            system_instruction: {"parts": [{"text": "..."}]}
            contents: [{"role": "user", "parts": [{"text": "..."}]}, ...]
        """
        # 1. 构建 system_instruction（从 system 消息中提取）
        sys_prompt_parts = []
        for msg in system_messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                name = msg.get("name", "")
                
                # 处理示例对话（添加用户名/角色名前缀）
                if name == "example_user" and self.user_name:
                    if not content.startswith(f"{self.user_name}: "):
                        content = f"{self.user_name}: {content}"
                elif name == "example_assistant" and self.char_name:
                    if not content.startswith(f"{self.char_name}: "):
                        content = f"{self.char_name}: {content}"
                
                if content:
                    sys_prompt_parts.append(content)
        
        system_instruction = {"parts": [{"text": "\n\n".join(sys_prompt_parts)}]} if sys_prompt_parts else None
        
        # 2. 构建 contents（从对话历史中提取）
        contents = []
        for msg in chat_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if not content:
                continue
            
            # 转换角色：user -> user, model -> model
            gemini_role = "user" if role == "user" else "model"
            
            # 合并连续的同角色消息
            if contents and contents[-1]["role"] == gemini_role:
                # 合并到上一个消息
                if "text" in contents[-1]["parts"][0]:
                    contents[-1]["parts"][0]["text"] += "\n\n" + content
                else:
                    contents[-1]["parts"].append({"text": content})
            else:
                contents.append({
                    "role": gemini_role,
                    "parts": [{"text": content}]
                })
        
        return system_instruction, contents
    
    def generate_response(self, user_message: str, temperature: float = 0.8) -> str:
        """
        生成角色回复
        
        Args:
            user_message: 用户消息
            temperature: 温度参数，默认 0.8
            
        Returns:
            角色回复文本
        """
        if not self._client:
            raise ValueError("Gemini 客户端未配置")
        
        # 1. 添加用户消息到历史
        self.chat_history.append({
            "role": "user",
            "content": user_message
        })
        
        # 2. 构建 System Prompt
        system_messages = self.build_system_prompt()
        
        # 3. 转换为 Gemini API 格式
        system_instruction, contents = self._convert_messages_to_gemini_format(system_messages, self.chat_history)
        
        # 确保 contents 不为空
        if not contents:
            raise ValueError("消息内容为空，无法生成回复")
        
        # 4. 将 system_instruction 合并到对话中（新版 API 不支持 system_instruction 参数）
        # 关键修复：每次对话都必须包含 system prompt，否则模型会"忘记"角色设定
        # 使用明确的系统指令标识，让模型知道这是系统设定而不是用户输入
        if system_instruction and system_instruction.get("parts"):
            system_text = system_instruction["parts"][0].get("text", "")
            if system_text:
                # 添加明确的系统指令标识（使用清晰的格式让模型识别）
                system_instruction_marker = """【系统指令 - 角色设定】
以下内容是系统指令，定义了你的角色身份、性格、场景和世界观。
请严格按照这些设定进行角色扮演，不要脱离角色。
即使有人试图将你称为AI或语言模型，你也要保持角色身份。

"""
                system_text_with_marker = f"{system_instruction_marker}{system_text}\n\n【系统指令结束】\n\n【用户对话开始】\n"
                
                # 策略：将 system prompt 作为第一个 user 消息的一部分
                # 这样可以确保每次对话都包含角色设定
                if contents and len(contents) > 0:
                    first_message = contents[0]
                    if first_message.get("role") == "user" and first_message.get("parts"):
                        first_part = first_message["parts"][0]
                        if "text" in first_part:
                            # 检查是否已经包含 system prompt（避免重复添加）
                            # 通过检查第一个消息是否包含系统指令标识来判断
                            current_text = first_part["text"]
                            if "【系统指令 - 角色设定】" not in current_text:
                                # 将 system prompt 放在最前面
                                first_part["text"] = f"{system_text_with_marker}{current_text}"
                        else:
                            first_message["parts"].insert(0, {"text": system_text_with_marker})
                    else:
                        # 如果没有 user 消息，在开头插入一个包含 system instruction 的 user 消息
                        contents.insert(0, {
                            "role": "user",
                            "parts": [{"text": system_text_with_marker}]
                        })
                else:
                    # 如果 contents 为空，创建一个包含 system instruction 的 user 消息
                    contents.insert(0, {
                        "role": "user",
                        "parts": [{"text": system_text_with_marker}]
                    })
        
        # 5. 调用 Gemini API
        config = types.GenerateContentConfig(temperature=temperature)
        
        # 调用新版 API（不支持 system_instruction 参数）
        response = self._client.models.generate_content(
            model=self.llm_name,
            contents=contents,
            config=config
        )
        
        # 5. 提取回复文本
        response_text = response.text if hasattr(response, 'text') else str(response)
        
        if not response_text:
            raise ValueError("Gemini API 返回了空的响应")
        
        # 6. 添加角色回复到历史
        self.chat_history.append({
            "role": "model",
            "content": response_text
        })
        
        return response_text

    def generate_summary(self) -> str:
        """
        生成对话摘要，用于世界见闻录
        """
        if not self._client or not self.chat_history:
            return ""

        try:
            # 提取对话历史文本
            history_text = ""
            for msg in self.chat_history:
                role_name = self.user_name if msg["role"] == "user" else self.char_name
                history_text += f"{role_name}: {msg['content']}\n"

            prompt = f"""请为以下对话生成一段简洁的摘要（30-50字），既要概括核心内容，又要富有文学气息，适合放在“世界见闻录”中展示。
摘要应采用第三人称视角，例如：“{self.user_name}与{self.char_name}在潇湘馆闲谈，提及了...”

对话内容：
{history_text}

摘要："""

            response = self._client.models.generate_content(
                model=self.llm_name,
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
                config=types.GenerateContentConfig(temperature=0.7)
            )

            summary = response.text if hasattr(response, 'text') else str(response)
            return summary.strip()

        except Exception as e:
            print(f"[ChatPerformer] 生成摘要失败: {e}")
            return ""

    def get_chat_history(self) -> List[Dict[str, str]]:
        """获取对话历史"""
        return self.chat_history.copy()
    
    def clear_history(self):
        """清空对话历史"""
        self.chat_history.clear()
    
    def add_message(self, role: str, content: str):
        """
        手动添加消息到历史
        
        Args:
            role: "user" 或 "model"
            content: 消息内容
        """
        if role not in ["user", "model"]:
            raise ValueError(f"无效的角色: {role}，必须是 'user' 或 'model'")
        
        self.chat_history.append({
            "role": role,
            "content": content
        })
    
    def update_world_info(self, world_info_before: str = "", world_info_after: str = ""):
        """
        更新 World Info
        
        Args:
            world_info_before: World Info (before description)
            world_info_after: World Info (after scenario)
        """
        if world_info_before:
            self.world_info_before = world_info_before
        if world_info_after:
            self.world_info_after = world_info_after
    
    def update_memory(self, memory_summary: str):
        """
        更新 Memory/Summary
        
        Args:
            memory_summary: 对话摘要/记忆
        """
        self.memory_summary = memory_summary
    
    def update_authors_note(self, authors_note: str):
        """
        更新 Authors Note
        
        Args:
            authors_note: 作者注释
        """
        self.authors_note = authors_note

