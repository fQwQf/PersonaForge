"""
快速制作书卷功能 - 从用户描述生成完整的书卷配置

根据文档要求，使用 Gemini-3-Pro 从用户的一句话描述（如"赛博朋克版红楼梦"）
自动生成完整的 World + Roles 配置。
"""
import os
import json
import re
from typing import Dict, List, Optional, Any
from sw_utils import load_json_file, save_json_file, get_models
from modules.models import ScrollConfig
from google import genai


class FastScrollGenerator:
    """快速书卷生成器类"""
    
    def __init__(self, llm_name: str = "gemini-2.5-flash", llm=None):
        """
        初始化快速书卷生成器
        
        Args:
            llm_name: LLM 模型名称，默认使用 gemini-2.5-flash（支持结构化输出）
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
                print("[FastScrollGenerator] Google genai 客户端初始化成功")
            else:
                self.genai_client = None
                print("[FastScrollGenerator] 警告: 未找到 GEMINI_API_KEY，将使用 LLM 接口")
        except Exception as e:
            print(f"[FastScrollGenerator] 初始化 genai 客户端失败: {e}")
            self.genai_client = None
    
    def generate_scroll_config(self,
                              user_description: str,
                              language: str = "zh",
                              num_characters: int = 5,
                              num_locations: int = 5) -> Dict[str, Any]:
        """
        从用户描述生成完整的书卷配置
        
        Args:
            user_description: 用户的一句话描述（如"赛博朋克版红楼梦"）
            language: 语言，默认中文
            num_characters: 生成的角色数量
            num_locations: 生成的地点数量
            
        Returns:
            完整的书卷配置字典，包含：
            - world: 世界观配置
            - characters: 角色列表
            - locations: 地点列表
            - event_chain: 事件链（可选）
        """
        if language == "zh":
            prompt = self._build_chinese_prompt(user_description, num_characters, num_locations)
        else:
            prompt = self._build_english_prompt(user_description, num_characters, num_locations)
        
        try:
            print(f"[FastScrollGenerator] 开始生成书卷配置: {user_description}")
            print(f"[FastScrollGenerator] 需要生成 {num_characters} 个角色，{num_locations} 个地点")
            
            # 优先使用结构化输出（如果 genai_client 可用）
            if self.genai_client:
                try:
                    print("[FastScrollGenerator] 使用结构化输出生成配置...")
                    
                    # 获取 Pydantic 模型的 JSON Schema
                    schema = ScrollConfig.model_json_schema()
                    
                    # 确保 schema 中明确要求生成指定数量的角色和地点
                    # 通过修改 schema 的 minItems 来强制要求
                    if "properties" in schema and "characters" in schema["properties"]:
                        schema["properties"]["characters"]["minItems"] = num_characters
                        schema["properties"]["characters"]["maxItems"] = num_characters
                    if "properties" in schema and "locations" in schema["properties"]:
                        schema["properties"]["locations"]["minItems"] = num_locations
                        schema["properties"]["locations"]["maxItems"] = num_locations
                    
                    response = self.genai_client.models.generate_content(
                        model=self.llm_name,
                        contents=prompt,
                        config={
                            "temperature": 0.8,
                            "response_mime_type": "application/json",
                            "response_json_schema": schema,
                        },
                    )
                    
                    # 解析结构化响应
                    scroll_config = ScrollConfig.model_validate_json(response.text)
                    
                    # 验证角色数量
                    if len(scroll_config.characters) < num_characters:
                        print(f"[FastScrollGenerator] 警告: 只生成了 {len(scroll_config.characters)} 个角色，需要 {num_characters} 个")
                    
                    # 验证地点数量
                    if len(scroll_config.locations) < num_locations:
                        print(f"[FastScrollGenerator] 警告: 只生成了 {len(scroll_config.locations)} 个地点，需要 {num_locations} 个")
                    
                    # 转换为字典格式
                    config_data = {
                        "world": {
                            "world_name": scroll_config.world.world_name,
                            "description": scroll_config.world.description,
                            "language": scroll_config.world.language
                        },
                        "characters": [
                            {
                                "role_name": char.role_name,
                                "nickname": char.nickname,
                                "profile": char.profile,
                                "gender": char.gender,
                                "identity": char.identity,
                                "motivation": char.motivation
                            }
                            for char in scroll_config.characters
                        ],
                        "locations": [
                            {
                                "location_name": loc.location_name,
                                "description": loc.description,
                                "detail": loc.detail
                            }
                            for loc in scroll_config.locations
                        ]
                    }
                    
                    print(f"[FastScrollGenerator] 结构化输出成功，生成了 {len(config_data['characters'])} 个角色，{len(config_data['locations'])} 个地点")
                    
                except Exception as e:
                    print(f"[FastScrollGenerator] 结构化输出失败: {e}，回退到普通输出")
                    import traceback
                    traceback.print_exc()
                    # 回退到普通输出
                    response = self.llm.chat(prompt, temperature=0.8)
                    
                    # 解析响应
                    if isinstance(response, str):
                        # 尝试解析 JSON
                        try:
                            # 移除可能的 markdown 代码块标记
                            response = response.strip()
                            if response.startswith("```json"):
                                response = response[7:]
                            if response.startswith("```"):
                                response = response[3:]
                            if response.endswith("```"):
                                response = response[:-3]
                            response = response.strip()
                            
                            config_data = json.loads(response)
                        except json.JSONDecodeError as e:
                            print(f"[FastScrollGenerator] JSON 解析失败: {e}")
                            print(f"[FastScrollGenerator] 原始响应: {response[:500]}")
                            # 返回默认配置
                            return self._generate_default_config(user_description, language, num_characters, num_locations)
                    else:
                        config_data = response
            else:
                # 使用普通 LLM 接口
                response = self.llm.chat(prompt, temperature=0.8)
                
                # 解析响应
                if isinstance(response, str):
                    # 尝试解析 JSON
                    try:
                        # 移除可能的 markdown 代码块标记
                        response = response.strip()
                        if response.startswith("```json"):
                            response = response[7:]
                        if response.startswith("```"):
                            response = response[3:]
                        if response.endswith("```"):
                            response = response[:-3]
                        response = response.strip()
                        
                        config_data = json.loads(response)
                    except json.JSONDecodeError as e:
                        print(f"[FastScrollGenerator] JSON 解析失败: {e}")
                        print(f"[FastScrollGenerator] 原始响应: {response[:500]}")
                        # 返回默认配置
                        return self._generate_default_config(user_description, language, num_characters, num_locations)
                else:
                    config_data = response
            
            # 验证和规范化配置
            config = self._validate_and_normalize_config(config_data, user_description, language, num_characters, num_locations)
            
            print(f"[FastScrollGenerator] 书卷配置生成成功，最终包含 {len(config['characters'])} 个角色，{len(config['locations'])} 个地点")
            return config
            
        except Exception as e:
            print(f"[FastScrollGenerator] 生成书卷配置失败: {e}")
            import traceback
            traceback.print_exc()
            # 返回默认配置
            return self._generate_default_config(user_description, language, num_characters, num_locations)
    
    def _build_chinese_prompt(self, user_description: str, num_characters: int, num_locations: int) -> str:
        """构建中文 prompt"""
        prompt = f"""请根据以下描述，生成一个完整的故事书卷配置。

## 用户描述
{user_description}

## 要求

请生成以下内容：

1. **世界观 (world)**: 
   - world_name: 世界观名称
   - description: 详细的世界观描述（200-300字）
   - language: 语言代码（"zh" 或 "en"）

2. **角色列表 (characters)**: 生成 {num_characters} 个主要角色
   每个角色包含：
   - role_name: 角色名称（必须是具体的人名，如"张三"、"李四"等，禁止使用"角色1"、"角色2"等占位符）
   - nickname: 昵称（可与角色名相同）
   - profile: 角色简介（100-150字，必须是对该角色的具体介绍，包括性格、背景、特点等，禁止使用"基于xxx的角色"这样的占位符）
   - gender: 性别
   - identity: 身份列表（如 ["学生", "主角"]）
   - motivation: 角色的动机（50字以内）

3. **地点列表 (locations)**: 生成 {num_locations} 个地点
   每个地点包含：
   - location_name: 地点名称
   - description: 地点简介（50字以内）
   - detail: 地点详细描述（100-150字）

## 输出格式

请以 JSON 格式输出，格式如下：

{{
  "world": {{
    "world_name": "世界观名称",
    "description": "世界观详细描述",
    "language": "zh"
  }},
  "characters": [
    {{
      "role_name": "角色名",
      "nickname": "昵称",
      "profile": "角色简介",
      "gender": "男/女",
      "identity": ["身份1", "身份2"],
      "motivation": "角色动机"
    }},
    ...
  ],
  "locations": [
    {{
      "location_name": "地点名",
      "description": "地点简介",
      "detail": "地点详细描述"
    }},
    ...
  ]
}}

要求：
- 世界观描述应该丰富详细，体现设定的特色
- **必须生成恰好 {num_characters} 个角色**，不能多也不能少。这是硬性要求，如果角色数量不足，系统将无法正常工作。
- 角色必须有具体的人名（如"张三"、"李四"、"王五"等），禁止使用"角色1"、"角色2"等占位符
- 角色的profile必须是对该角色的具体介绍，包括性格、背景、特点等，禁止使用"基于xxx的角色"这样的占位符
- 角色应该有鲜明的个性和背景
- **必须生成恰好 {num_locations} 个地点**，不能多也不能少。这是硬性要求，如果地点数量不足，系统将无法正常工作。
- 地点应该与世界观相符
- 输出必须是有效的 JSON 格式，不要包含 Markdown 代码块标记

**重要提醒：请确保 characters 数组包含恰好 {num_characters} 个元素，locations 数组包含恰好 {num_locations} 个元素。**
"""
        return prompt
    
    def _build_english_prompt(self, user_description: str, num_characters: int, num_locations: int) -> str:
        """构建英文 prompt"""
        prompt = f"""Please generate a complete story scroll configuration based on the following description.

## User Description
{user_description}

## Requirements

Please generate the following:

1. **World (world)**: 
   - world_name: World name
   - description: Detailed world description (200-300 words)
   - language: Language code ("zh" or "en")

2. **Character List (characters)**: Generate {num_characters} main characters
   Each character includes:
   - role_name: Character name (must be a specific name like "John", "Mary", etc., DO NOT use placeholders like "Character1", "Character2")
   - nickname: Nickname (can be same as role_name)
   - profile: Character profile (100-150 words, must be a specific introduction of the character including personality, background, traits, etc., DO NOT use placeholders like "A character based on xxx")
   - gender: Gender
   - identity: List of identities (e.g., ["student", "protagonist"])
   - motivation: Character motivation (within 50 words)

3. **Location List (locations)**: Generate {num_locations} locations
   Each location includes:
   - location_name: Location name
   - description: Location brief description (within 50 words)
   - detail: Location detailed description (100-150 words)

## Output Format

Please output in JSON format as follows:

{{
  "world": {{
    "world_name": "World Name",
    "description": "Detailed world description",
    "language": "en"
  }},
  "characters": [
    {{
      "role_name": "Character Name",
      "nickname": "Nickname",
      "profile": "Character profile",
      "gender": "Male/Female",
      "identity": ["Identity1", "Identity2"],
      "motivation": "Character motivation"
    }},
    ...
  ],
  "locations": [
    {{
      "location_name": "Location Name",
      "description": "Location brief description",
      "detail": "Location detailed description"
    }},
    ...
  ]
}}

Requirements:
- World description should be rich and detailed, reflecting the setting's characteristics
- **You MUST generate exactly {num_characters} characters**, no more, no less. This is a hard requirement. If the number of characters is insufficient, the system will not work properly.
- Characters must have specific names (like "John", "Mary", "Tom", etc.), DO NOT use placeholders like "Character1", "Character2"
- Character profiles must be specific introductions of each character including personality, background, traits, etc., DO NOT use placeholders like "A character based on xxx"
- Characters should have distinct personalities and backgrounds
- **You MUST generate exactly {num_locations} locations**, no more, no less. This is a hard requirement. If the number of locations is insufficient, the system will not work properly.
- Locations should match the world setting
- Output must be valid JSON format, do not include Markdown code block markers

**IMPORTANT REMINDER: Please ensure the characters array contains exactly {num_characters} elements and the locations array contains exactly {num_locations} elements.**
"""
        return prompt
    
    def _validate_and_normalize_config(self,
                                       config_data: Dict[str, Any],
                                       user_description: str,
                                       language: str,
                                       num_characters: int,
                                       num_locations: int) -> Dict[str, Any]:
        """验证和规范化配置数据"""
        # 验证世界配置
        world = config_data.get("world", {})
        if not isinstance(world, dict):
            world = {}
        
        normalized_world = {
            "world_name": world.get("world_name", user_description),
            "description": world.get("description", f"基于'{user_description}'的世界观"),
            "language": world.get("language", language),
            "source": ""  # 将在保存时设置
        }
        
        # 验证角色列表
        characters = config_data.get("characters", [])
        if not isinstance(characters, list):
            characters = []
        
        normalized_characters = []
        for i, char in enumerate(characters[:num_characters]):
            if not isinstance(char, dict):
                continue
            
            role_name = char.get("role_name", "").strip()
            profile = char.get("profile", "").strip()
            
            # 检查角色名称是否是占位符（更严格的检查）
            is_placeholder_name = False
            if not role_name:
                is_placeholder_name = True
            elif role_name.startswith("角色") and len(role_name) > 2:
                # 检查是否是"角色1"、"角色2"等格式
                remaining = role_name[2:].strip()
                if remaining.isdigit() or remaining == "":
                    is_placeholder_name = True
            
            # 如果角色名是占位符，使用LLM生成具体名称
            if is_placeholder_name:
                try:
                    # 尝试使用LLM生成角色名称
                    name_prompt = f"""根据以下描述，生成一个具体的角色名称（必须是真实的人名，如"张三"、"李四"、"王五"等，不要使用"角色1"、"角色2"等占位符）。

用户描述：{user_description}
角色序号：第{len(normalized_characters)+1}个角色

请只返回角色名称，不要返回其他内容。"""
                    name_response = self.llm.chat(name_prompt, temperature=0.9)
                    if isinstance(name_response, str):
                        generated_name = name_response.strip()
                        # 清理可能的引号或多余字符
                        generated_name = generated_name.strip('"\'`').strip()
                        if generated_name and len(generated_name) <= 10 and not generated_name.startswith("角色"):
                            role_name = generated_name
                        else:
                            # 如果生成失败，使用常见的中文姓名
                            common_names = ["张三", "李四", "王五", "赵六", "钱七", "孙八", "周九", "吴十"]
                            role_name = common_names[len(normalized_characters) % len(common_names)]
                    else:
                        common_names = ["张三", "李四", "王五", "赵六", "钱七", "孙八", "周九", "吴十"]
                        role_name = common_names[len(normalized_characters) % len(common_names)]
                except Exception as e:
                    print(f"[FastScrollGenerator] 生成角色名称失败: {e}")
                    common_names = ["张三", "李四", "王五", "赵六", "钱七", "孙八", "周九", "吴十"]
                    role_name = common_names[len(normalized_characters) % len(common_names)]
            
            # 检查profile是否是占位符（更严格的检查）
            is_placeholder_profile = False
            if not profile:
                is_placeholder_profile = True
            elif "基于" in profile and "的角色" in profile:
                is_placeholder_profile = True
            elif profile.startswith("基于") or profile.endswith("的角色"):
                is_placeholder_profile = True
            
            # 如果profile是占位符，使用LLM生成具体介绍
            if is_placeholder_profile:
                try:
                    # 使用LLM生成角色介绍
                    profile_prompt = f"""根据以下信息，生成一个角色的具体介绍（100-150字），包括性格、背景、特点等。

用户描述：{user_description}
角色名称：{role_name}
角色序号：第{len(normalized_characters)+1}个角色

要求：
1. 必须是对该角色的具体介绍，包括性格、背景、特点等
2. 禁止使用"基于xxx的角色"这样的占位符
3. 介绍应该生动具体，符合{user_description}的世界观

请只返回角色介绍，不要返回其他内容。"""
                    profile_response = self.llm.chat(profile_prompt, temperature=0.8)
                    if isinstance(profile_response, str):
                        generated_profile = profile_response.strip()
                        # 清理可能的引号或多余字符
                        generated_profile = generated_profile.strip('"\'`').strip()
                        if generated_profile and len(generated_profile) > 20 and not ("基于" in generated_profile and "的角色" in generated_profile):
                            profile = generated_profile
                        else:
                            profile = f"{role_name}是一个在{user_description}世界中的角色，具有独特的性格和背景。"
                    else:
                        profile = f"{role_name}是一个在{user_description}世界中的角色，具有独特的性格和背景。"
                except Exception as e:
                    print(f"[FastScrollGenerator] 生成角色介绍失败: {e}")
                    profile = f"{role_name}是一个在{user_description}世界中的角色，具有独特的性格和背景。"
            
            normalized_char = {
                "role_name": role_name,
                "nickname": char.get("nickname", role_name).strip() or role_name,
                "profile": profile,
                "gender": char.get("gender", "未知"),
                "identity": char.get("identity", []) if isinstance(char.get("identity"), list) else [],
                "motivation": char.get("motivation", "").strip()
            }
            normalized_characters.append(normalized_char)
        
        # 如果角色数量不够，补充默认角色（使用更合理的默认值）
        while len(normalized_characters) < num_characters:
            char_num = len(normalized_characters) + 1
            normalized_characters.append({
                "role_name": f"角色{char_num}",
                "nickname": f"角色{char_num}",
                "profile": f"一个在{user_description}世界中的角色，具有独特的性格和背景。",
                "gender": "未知",
                "identity": [],
                "motivation": ""
            })
        
        # 验证地点列表
        locations = config_data.get("locations", [])
        if not isinstance(locations, list):
            locations = []
        
        normalized_locations = []
        for i, loc in enumerate(locations[:num_locations]):
            if not isinstance(loc, dict):
                continue
            normalized_loc = {
                "location_name": loc.get("location_name", f"地点{i+1}"),
                "description": loc.get("description", ""),
                "detail": loc.get("detail", "")
            }
            normalized_locations.append(normalized_loc)
        
        # 如果地点数量不够，补充默认地点
        while len(normalized_locations) < num_locations:
            normalized_locations.append({
                "location_name": f"地点{len(normalized_locations)+1}",
                "description": f"基于'{user_description}'的地点",
                "detail": ""
            })
        
        return {
            "world": normalized_world,
            "characters": normalized_characters,
            "locations": normalized_locations
        }
    
    def _generate_default_config(self,
                                 user_description: str,
                                 language: str,
                                 num_characters: int,
                                 num_locations: int) -> Dict[str, Any]:
        """生成默认配置"""
        world = {
            "world_name": user_description,
            "description": f"基于'{user_description}'的世界观设定",
            "language": language,
            "source": ""
        }
        
        characters = []
        for i in range(num_characters):
            characters.append({
                "role_name": f"角色{i+1}",
                "nickname": f"角色{i+1}",
                "profile": f"基于'{user_description}'的角色",
                "gender": "未知",
                "identity": [],
                "motivation": ""
            })
        
        locations = []
        for i in range(num_locations):
            locations.append({
                "location_name": f"地点{i+1}",
                "description": f"基于'{user_description}'的地点",
                "detail": ""
            })
        
        return {
            "world": world,
            "characters": characters,
            "locations": locations
        }
    
    def save_scroll_config(self,
                          config: Dict[str, Any],
                          source_name: str,
                          base_dir: str = "./data") -> Dict[str, str]:
        """
        保存书卷配置到文件系统
        
        Args:
            config: 书卷配置
            source_name: 源名称（用于文件路径）
            base_dir: 基础目录
            
        Returns:
            包含保存路径的字典
        """
        paths = {}
        
        try:
            # 创建目录结构
            world_dir = f"{base_dir}/worlds/{source_name}"
            roles_dir = f"{base_dir}/roles/{source_name}"
            locations_file = f"{base_dir}/locations/{source_name}.json"
            
            os.makedirs(world_dir, exist_ok=True)
            os.makedirs(roles_dir, exist_ok=True)
            os.makedirs(os.path.dirname(locations_file), exist_ok=True)
            
            # 保存世界观
            world = config.get("world", {})
            world["source"] = source_name
            world_file = f"{world_dir}/general.json"
            save_json_file(world_file, world)
            paths["world_file"] = world_file
            
            # 保存角色
            characters = config.get("characters", [])
            world_language = world.get('language', 'zh')
            
            # 首先生成所有角色的代码映射
            char_code_map = {}  # 存储角色名到完整代码的映射
            for char in characters:
                role_name = char.get("role_name", "")
                if not role_name:
                    continue
                base_code = self._name_to_code(role_name)
                full_code = f"{base_code}-{world_language}"
                char_code_map[role_name] = full_code
            
            # 然后保存每个角色，并设置关系
            for char in characters:
                role_name = char.get("role_name", "").strip()
                if not role_name:
                    continue
                
                # 最终检查：确保role_name不是占位符
                final_role_name = role_name
                if final_role_name.startswith("角色") and len(final_role_name) > 2:
                    remaining = final_role_name[2:].strip()
                    if remaining.isdigit() or remaining == "":
                        # 使用常见的中文姓名替换占位符
                        common_names = ["张三", "李四", "王五", "赵六", "钱七", "孙八", "周九", "吴十", "郑一", "冯二"]
                        char_index = len([c for c in characters if c.get("role_name", "").startswith("角色")])
                        final_role_name = common_names[char_index % len(common_names)]
                        # 更新char_code_map
                        base_code = self._name_to_code(final_role_name)
                        full_code = f"{base_code}-{world_language}"
                        char_code_map[final_role_name] = full_code
                    
                role_code = char_code_map.get(final_role_name)
                if not role_code:
                    # 如果找不到，重新生成code
                    base_code = self._name_to_code(final_role_name)
                    role_code = f"{base_code}-{world_language}"
                    char_code_map[final_role_name] = role_code
                    
                role_dir = f"{roles_dir}/{role_code}"
                os.makedirs(role_dir, exist_ok=True)
                
                # 构建角色关系（与其他所有角色的关系）
                relations = {}
                for other_char in characters:
                    other_name = other_char.get("role_name", "").strip()
                    if other_name and other_name != final_role_name:
                        # 检查other_name是否也是占位符
                        if other_name.startswith("角色") and len(other_name) > 2:
                            remaining = other_name[2:].strip()
                            if remaining.isdigit() or remaining == "":
                                continue  # 跳过占位符角色
                        other_code = char_code_map.get(other_name)
                        if other_code:
                            relations[other_code] = {
                                "relation": [],
                                "detail": ""
                            }
                
                # 最终检查：确保profile不是占位符
                final_profile = char.get("profile", "").strip()
                if not final_profile or ("基于" in final_profile and "的角色" in final_profile) or final_profile.startswith("基于") or final_profile.endswith("的角色"):
                    world_name = world.get('world_name', '')
                    if world_name:
                        final_profile = f"{final_role_name}是一个在{world_name}世界中的角色，具有独特的性格和背景。"
                    else:
                        final_profile = f"{final_role_name}是一个具有独特性格和背景的角色。"
                
                role_info = {
                    "role_code": role_code,
                    "role_name": final_role_name,
                    "nickname": char.get("nickname", final_role_name).strip() or final_role_name,
                    "profile": final_profile,
                    "gender": char.get("gender", "未知"),
                    "identity": char.get("identity", []),
                    "motivation": char.get("motivation", "").strip(),
                    "activity": 1.0,
                    "relation": relations,
                    "source": source_name
                }
                
                role_info_file = f"{role_dir}/role_info.json"
                save_json_file(role_info_file, role_info)
            
            paths["roles_dir"] = roles_dir
            
            # 保存地点
            locations = config.get("locations", [])
            locations_dict = {}
            for loc in locations:
                location_code = self._name_to_code(loc.get("location_name", ""))
                locations_dict[location_code] = {
                    "location_code": location_code,
                    "location_name": loc.get("location_name", ""),
                    "description": loc.get("description", ""),
                    "detail": loc.get("detail", ""),
                    "source": source_name
                }
            
            save_json_file(locations_file, locations_dict)
            paths["locations_file"] = locations_file
            
            print(f"[FastScrollGenerator] 书卷配置已保存到: {source_name}")
            return paths
            
        except Exception as e:
            print(f"[FastScrollGenerator] 保存书卷配置失败: {e}")
            import traceback
            traceback.print_exc()
            return paths
    
    def _name_to_code(self, name: str) -> str:
        """将名称转换为代码（用于文件名）"""
        # 移除特殊字符，转换为小写，用下划线连接
        code = re.sub(r'[^\w\s-]', '', name)
        code = re.sub(r'[-\s]+', '_', code).lower()
        return code


if __name__ == "__main__":
    # 示例用法
    generator = FastScrollGenerator(llm_name="gemini-2.5-pro")
    
    # 生成书卷配置示例
    # config = generator.generate_scroll_config(
    #     user_description="赛博朋克版红楼梦",
    #     language="zh",
    #     num_characters=5,
    #     num_locations=5
    # )
    # 
    # generator.save_scroll_config(config, "cyberpunk_dream", "./data")

