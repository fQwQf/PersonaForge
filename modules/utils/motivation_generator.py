"""
动机生成系统 - 为角色生成隐藏动机

根据文档要求，使用 Gemini-2.5-Flash 分析角色在世界观中的深层欲望、恐惧和潜在目标，
生成一段 100 字以内的隐藏动机，并保存到角色 JSON 文件的 hidden_motivation 字段。
支持一次性批量生成多个角色的动机。
"""
import os
import json
from typing import Dict, List, Optional, Any
from sw_utils import load_json_file, save_json_file, get_models
from modules.models import BatchMotivations, CharacterMotivation
from google import genai


class MotivationGenerator:
    """动机生成器类"""
    
    def __init__(self, llm_name: str = "gemini-2.5-flash", llm=None):
        """
        初始化动机生成器
        
        Args:
            llm_name: LLM 模型名称，默认使用 gemini-2.5-flash
            llm: 可选的 LLM 实例，如果提供则直接使用
        """
        self.llm_name = llm_name
        if llm is None:
            self.llm = get_models(llm_name)
        else:
            self.llm = llm
        
        # 初始化 Google genai 客户端（用于结构化输出）
        try:
            # 尝试从环境变量获取 API Key
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEYS", "")
            if not api_key:
                # 尝试从 config.json 读取
                try:
                    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config.json")
                    if os.path.exists(config_path):
                        config_data = load_json_file(config_path)
                        api_key = config_data.get("GEMINI_API_KEY", "")
                except:
                    pass
            
            # 处理多个 key 的情况
            if api_key:
                if "," in api_key:
                    api_key = api_key.split(",")[0].strip()
                api_key = api_key.strip()
            
            if api_key:
                os.environ['GEMINI_API_KEY'] = api_key
                self.genai_client = genai.Client(vertexai=False)
                print("[MotivationGenerator] Google genai 客户端初始化成功")
            else:
                self.genai_client = None
                print("[MotivationGenerator] 警告: 未找到 GEMINI_API_KEY，将使用 LLM 接口")
        except Exception as e:
            print(f"[MotivationGenerator] 初始化 genai 客户端失败: {e}")
            self.genai_client = None
    
    def generate_motivation(self, 
                           role_name: str,
                           role_profile: str,
                           world_description: str,
                           language: str = "zh") -> str:
        """
        为单个角色生成隐藏动机
        
        Args:
            role_name: 角色名称
            role_profile: 角色简介
            world_description: 世界观描述
            language: 语言，默认中文
            
            Returns:
            生成的隐藏动机文本（约 100 字以内）
        """
        if language == "zh":
            prompt = f"""请分析角色【{role_name}】在以下世界观中的深层欲望、恐惧和潜在目标，生成一段约 100 字的隐藏动机（必须控制在 100 字以内）。

角色简介：
{role_profile}

世界观描述：
{world_description}

请从以下角度分析：
1. 角色的核心欲望和追求
2. 角色最深层的恐惧
3. 角色可能隐藏的目标和计划
4. 角色与世界观的互动关系

生成一段连贯的、富有深度的隐藏动机描述，必须控制在 100 字以内。要求：
- 深入挖掘角色的心理层面
- 体现角色与世界观的内在联系
- 语言流畅自然，符合角色性格
- 严格控制字数在 100 字以内
- 不要使用 Markdown 格式

隐藏动机："""
        else:
            prompt = f"""Please analyze the character 【{role_name}】's deep desires, fears, and potential goals in the following world setting, and generate a hidden motivation of approximately 100 words (within 100 characters in Chinese).

Character Profile:
{role_profile}

World Description:
{world_description}

Please analyze from the following perspectives:
1. The character's core desires and pursuits
2. The character's deepest fears
3. The character's possible hidden goals and plans
4. The character's interaction with the world setting

Generate a coherent and profound hidden motivation description of approximately 100 words. Requirements:
- Deeply explore the character's psychological level
- Reflect the intrinsic connection between the character and the world setting
- Natural and fluent language that matches the character's personality
- Keep within 100 words
- Do not use Markdown formatting

Hidden Motivation:"""
        
        try:
            print(f"[MotivationGenerator] 开始为角色 {role_name} 生成隐藏动机...")
            motivation = self.llm.chat(prompt, temperature=0.8)
            
            # 确保返回的是字符串
            if not isinstance(motivation, str):
                motivation = str(motivation)
            
            # 清理可能的格式标记
            motivation = motivation.strip()
            
            # 如果响应包含"隐藏动机："等前缀，移除它们
            prefixes = ["隐藏动机：", "Hidden Motivation:", "动机：", "Motivation:"]
            for prefix in prefixes:
                if motivation.startswith(prefix):
                    motivation = motivation[len(prefix):].strip()
            
            print(f"[MotivationGenerator] 角色 {role_name} 的隐藏动机生成成功（长度: {len(motivation)} 字符）")
            return motivation
            
        except Exception as e:
            print(f"[MotivationGenerator] 生成隐藏动机失败: {e}")
            import traceback
            traceback.print_exc()
            # 返回默认动机
            default_motivation = (
                f"{role_name}追求个人目标和成长，在复杂的世界中寻找属于自己的位置。"
                if language == "zh" 
                else f"{role_name} pursues personal goals and growth, seeking their place in a complex world."
            )
            return default_motivation
    
    def generate_for_role_file(self,
                              role_info_path: str,
                              world_file_path: str,
                              language: str = "zh") -> bool:
        """
        为角色文件生成并保存隐藏动机
        
        Args:
            role_info_path: 角色信息文件路径（role_info.json）
            world_file_path: 世界观文件路径
            language: 语言，默认中文
            
        Returns:
            是否成功生成并保存
        """
        try:
            # 加载角色信息
            if not os.path.exists(role_info_path):
                print(f"[MotivationGenerator] 角色文件不存在: {role_info_path}")
                return False
            
            role_info = load_json_file(role_info_path)
            role_name = role_info.get("role_name", "")
            role_profile = role_info.get("profile", "")
            
            if not role_name or not role_profile:
                print(f"[MotivationGenerator] 角色信息不完整: {role_info_path}")
                return False
            
            # 检查是否已有隐藏动机
            if "hidden_motivation" in role_info and role_info["hidden_motivation"]:
                print(f"[MotivationGenerator] 角色 {role_name} 已有隐藏动机，跳过生成")
                return True
            
            # 加载世界观描述
            world_description = ""
            if os.path.exists(world_file_path):
                world_data = load_json_file(world_file_path)
                world_description = world_data.get("description", "") or world_data.get("world_description", "")
            
            if not world_description:
                print(f"[MotivationGenerator] 警告: 无法加载世界观描述，使用空描述")
            
            # 生成隐藏动机
            hidden_motivation = self.generate_motivation(
                role_name=role_name,
                role_profile=role_profile,
                world_description=world_description,
                language=language
            )
            
            # 保存到角色文件
            role_info["hidden_motivation"] = hidden_motivation
            save_json_file(role_info_path, role_info)
            
            print(f"[MotivationGenerator] 成功为角色 {role_name} 生成并保存隐藏动机")
            return True
            
        except Exception as e:
            print(f"[MotivationGenerator] 处理角色文件失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_batch(self,
                      role_file_dir: str,
                      world_file_path: str,
                      source: Optional[str] = None,
                      language: str = "zh") -> Dict[str, bool]:
        """
        批量生成多个角色的隐藏动机
        
        Args:
            role_file_dir: 角色文件目录
            world_file_path: 世界观文件路径
            source: 可选的源目录名称（用于过滤特定源的角色）
            language: 语言，默认中文
            
        Returns:
            字典，键为角色代码，值为是否成功
        """
        from sw_utils import get_grandchild_folders, get_child_folders
        
        results = {}
        
        # 确定搜索路径
        if source and os.path.exists(os.path.join(role_file_dir, source)):
            search_path = os.path.join(role_file_dir, source)
            folders = get_child_folders(search_path)
        else:
            search_path = role_file_dir
            folders = get_grandchild_folders(search_path)
        
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        for folder_path in folders:
            role_info_path = os.path.join(base_dir, folder_path, "role_info.json")
            
            if not os.path.exists(role_info_path):
                continue
            
            try:
                role_info = load_json_file(role_info_path)
                role_code = role_info.get("role_code", "")
                
                if not role_code:
                    continue
                
                success = self.generate_for_role_file(
                    role_info_path=role_info_path,
                    world_file_path=world_file_path,
                    language=language
                )
                
                results[role_code] = success
                
            except Exception as e:
                print(f"[MotivationGenerator] 处理角色失败: {e}")
                role_code = os.path.basename(folder_path)
                results[role_code] = False
        
        print(f"[MotivationGenerator] 批量生成完成，成功: {sum(results.values())}/{len(results)}")
        return results
    
    def generate_batch_motivations(self,
                                   characters: List[Dict[str, str]],
                                   world_description: str,
                                   intervention: str = "",
                                   language: str = "zh") -> Dict[str, str]:
        """
        一次性批量生成多个角色的隐藏动机
        
        Args:
            characters: 角色列表，每个元素包含 role_name 和 profile
            world_description: 世界观描述
            intervention: 全局事件描述（可选）
            language: 语言，默认中文
            
        Returns:
            字典，键为角色名称，值为生成的动机文本（100字以内）
        """
        if not characters:
            return {}
        
        # 构建批量生成的 prompt
        if language == "zh":
            characters_text = "\n\n".join([
                f"角色名称：{char['role_name']}\n角色简介：{char.get('profile', '')}"
                for char in characters
            ])
            prompt = f"""请为以下所有角色一次性生成隐藏动机。每个角色的动机必须严格控制在 100 字以内（100 个中文字符），深入挖掘角色的心理层面，体现角色与世界观的内在联系。

## 世界观描述
{world_description}
"""
            if intervention:
                prompt += f"""
## 当前全局事件
{intervention}
注意：在生成动机时，请考虑全局事件对角色心理的影响。
"""
            prompt += f"""
## 角色列表
{characters_text}

请为每个角色生成一段连贯的、富有深度的隐藏动机描述。要求：
- 深入挖掘角色的心理层面
- 体现角色与世界观的内在联系
- 语言流畅自然，符合角色性格
- **必须严格控制在 100 字以内，绝对不能超过 100 字**
- 不要使用 Markdown 格式

**重要：每个角色的动机必须精确控制在 100 字以内，如果超过 100 字，请精简内容。**

请一次性为所有角色生成动机，确保每个角色都有对应的动机，且每个动机都在 100 字以内。"""
        else:
            characters_text = "\n\n".join([
                f"Character Name: {char['role_name']}\nCharacter Profile: {char.get('profile', '')}"
                for char in characters
            ])
            prompt = f"""Please generate hidden motivations for all the following characters at once. Each character's motivation must be strictly within 100 words (within 100 characters in Chinese), deeply exploring the character's psychological level and reflecting the intrinsic connection between the character and the world setting.

## World Description
{world_description}
"""
            if intervention:
                prompt += f"""
## Current Global Event
{intervention}
Note: Please consider the impact of the global event on the character's psychology when generating motivations.
"""
            prompt += f"""
## Character List
{characters_text}

Please generate a coherent and profound hidden motivation description for each character. Requirements:
- Deeply explore the character's psychological level
- Reflect the intrinsic connection between the character and the world setting
- Natural and fluent language that matches the character's personality
- **Must be strictly within 100 words**
- Do not use Markdown formatting

**Important: Each character's motivation must be precisely within 100 words. If it exceeds 100 words, please condense the content.**

Please generate motivations for all characters at once, ensuring each character has a corresponding motivation, and each motivation is within 100 words."""
        
        try:
            print(f"[MotivationGenerator] 开始一次性生成 {len(characters)} 个角色的隐藏动机...")
            
            # 使用 Google genai 的结构化输出
            if self.genai_client:
                response = self.genai_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config={
                        "temperature": 0.8,
                        "response_mime_type": "application/json",
                        "response_json_schema": BatchMotivations.model_json_schema(),
                    },
                )
                
                batch_result = BatchMotivations.model_validate_json(response.text)
                
                # 转换为字典格式（不截断）
                results = {}
                for char_motivation in batch_result.motivations:
                    motivation = char_motivation.motivation.strip()
                    results[char_motivation.role_name] = motivation
                
                print(f"[MotivationGenerator] 批量生成成功，共生成 {len(results)} 个角色的动机")
                return results
            else:
                # 使用现有的 LLM 接口（如果 genai_client 不可用）
                print("[MotivationGenerator] 使用 LLM 接口生成动机...")
                response = self.llm.chat(prompt, temperature=0.8, response_model=BatchMotivations)
                
                if isinstance(response, BatchMotivations):
                    results = {}
                    for char_motivation in response.motivations:
                        motivation = char_motivation.motivation.strip()
                        results[char_motivation.role_name] = motivation
                    print(f"[MotivationGenerator] 批量生成成功，共生成 {len(results)} 个角色的动机")
                    return results
                else:
                    raise ValueError("批量生成返回了无效的响应格式")
                    
        except Exception as e:
            print(f"[MotivationGenerator] 批量生成失败: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"批量生成动机失败: {e}")


if __name__ == "__main__":
    # 示例用法
    generator = MotivationGenerator(llm_name="gemini-2.5-pro")
    
    # 单个角色生成示例
    # generator.generate_for_role_file(
    #     role_info_path="./data/roles/Romance_of_the_Three_Kingdoms/caocao-zh/role_info.json",
    #     world_file_path="./data/worlds/Romance_of_the_Three_Kingdoms.json",
    #     language="zh"
    # )
    
    # 批量生成示例
    # generator.generate_batch(
    #     role_file_dir="./data/roles/",
    #     world_file_path="./data/worlds/Romance_of_the_Three_Kingdoms.json",
    #     source="Romance_of_the_Three_Kingdoms",
    #     language="zh"
    # )

