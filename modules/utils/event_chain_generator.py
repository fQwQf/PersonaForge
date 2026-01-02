"""
事件链生成引擎 - 为故事生成预设的事件链

根据文档要求，使用 Gemini-3-Pro 生成完整的故事大纲 (Event Chain)，
包含每一幕的标题、明线、暗线。支持 3/5/8/10 幕的生成。
"""
import os
import json
from typing import Dict, List, Optional, Any, Literal
from sw_utils import load_json_file, save_json_file, get_models


class EventChainGenerator:
    """事件链生成器类"""
    
    def __init__(self, llm_name: str = "gemini-2.5-flash", llm=None):
        """
        初始化事件链生成器
        
        Args:
            llm_name: LLM 模型名称，默认使用 gemini-2.5-flash（更快，避免超时）
            llm: 可选的 LLM 实例，如果提供则直接使用
        """
        self.llm_name = llm_name
        if llm is None:
            # 如果使用 Gemini 模型，增加超时时间到60秒，并增加重试次数
            from modules.llm.Gemini import Gemini
            if "gemini" in llm_name.lower():
                self.llm = Gemini(model=llm_name, timeout=60)
                # 增加重试次数以应对 503 错误
                self.llm.max_retries = 3
            else:
                self.llm = get_models(llm_name)
        else:
            self.llm = llm
    
    def generate_event_chain(self,
                             world_description: str,
                             character_names: List[str],
                             total_acts: Literal[1, 3, 5, 8, 10] = 5,
                             language: str = "zh",
                             user_description: Optional[str] = None) -> Dict[str, Any]:
        """
        生成事件链
        
        Args:
            world_description: 世界观描述
            character_names: 角色名称列表
            total_acts: 总幕数（1/3/5/8/10）
            language: 语言，默认中文
            user_description: 用户描述（可选，用于快速制作）
            
        Returns:
            事件链字典，包含 acts 数组和 event_chain 对象
        """
        if language == "zh":
            prompt = self._build_chinese_prompt(world_description, character_names, total_acts, user_description)
        else:
            prompt = self._build_english_prompt(world_description, character_names, total_acts, user_description)
        
        # 添加重试机制
        max_attempts = 3
        last_exception = None
        
        for attempt in range(max_attempts):
            try:
                if attempt > 0:
                    wait_time = attempt * 3  # 3秒、6秒
                    print(f"[EventChainGenerator] 第 {attempt + 1} 次尝试生成事件链（{total_acts} 幕），等待 {wait_time} 秒后重试...")
                    import time
                    time.sleep(wait_time)
                
                print(f"[EventChainGenerator] 开始生成事件链（{total_acts} 幕），尝试 {attempt + 1}/{max_attempts}...")
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
                        
                        event_chain_data = json.loads(response)
                        print(f"[EventChainGenerator] JSON 解析成功")
                    except json.JSONDecodeError as e:
                        print(f"[EventChainGenerator] JSON 解析失败，尝试文本解析: {e}")
                        print(f"[EventChainGenerator] 原始响应前500字符: {response[:500]}")
                        event_chain_data = self._parse_text_response(response, total_acts, language)
                else:
                    event_chain_data = response
                
                # 调试：打印解析后的数据
                print(f"[EventChainGenerator] 解析后的数据: {json.dumps(event_chain_data, ensure_ascii=False, indent=2)[:1000]}")
                
                # 验证和规范化数据
                event_chain = self._validate_and_normalize(event_chain_data, total_acts, language)
                
                # 调试：打印规范化后的数据
                print(f"[EventChainGenerator] 规范化后的数据: {json.dumps(event_chain, ensure_ascii=False, indent=2)[:1000]}")
                
                print(f"[EventChainGenerator] 事件链生成成功（{len(event_chain.get('acts', []))} 幕）")
                return event_chain
                
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                print(f"[EventChainGenerator] 生成事件链失败（尝试 {attempt + 1}/{max_attempts}）: {e}")
                import traceback
                traceback.print_exc()
                
                # 检查是否是503错误或其他可重试的错误
                is_retryable = (
                    "503" in str(e) or 
                    "unavailable" in error_str or 
                    "overloaded" in error_str or
                    "超时" in str(e) or 
                    "timeout" in error_str or
                    "rate limit" in error_str or
                    "429" in str(e)
                )
                
                # 如果是可重试的错误且还有重试机会，继续循环
                if is_retryable and attempt < max_attempts - 1:
                    print(f"[EventChainGenerator] 检测到可重试错误，将在下次尝试时重试...")
                    continue
                else:
                    # 如果不可重试或已达到最大重试次数，抛出异常
                    if "超时" in str(e) or "timeout" in error_str:
                        raise Exception(f"事件链生成超时，请稍后重试或尝试减少幕数。错误: {e}")
                    elif "503" in str(e) or "unavailable" in error_str or "overloaded" in error_str:
                        raise Exception(f"Gemini API 当前过载，请稍后重试。如果问题持续，请尝试减少幕数或稍后再试。错误: {e}")
                    else:
                        raise Exception(f"事件链生成失败: {e}")
        
        # 如果所有重试都失败
        if last_exception:
            error_str = str(last_exception).lower()
            if "503" in str(last_exception) or "unavailable" in error_str or "overloaded" in error_str:
                raise Exception(f"Gemini API 当前过载，已重试 {max_attempts} 次仍失败。请稍后重试或尝试减少幕数。")
            else:
                raise Exception(f"事件链生成失败（已重试 {max_attempts} 次）: {last_exception}")
    
    def _build_chinese_prompt(self,
                              world_description: str,
                              character_names: List[str],
                              total_acts: int,
                              user_description: Optional[str] = None) -> str:
        """构建中文 prompt"""
        characters_text = "\n".join([f"- {name}" for name in character_names])
        
        prompt = f"""请为以下故事生成一个完整的事件链大纲，包含 {total_acts} 幕（Act）。

## 世界观描述
{world_description}

## 角色列表
{characters_text}
"""
        
        if user_description:
            prompt += f"""
## 用户描述
{user_description}
"""
        
        prompt += f"""
## 要求

请生成一个包含 {total_acts} 幕的完整故事大纲。每一幕应该包含：

1. **标题 (title)**: 该幕的标题
2. **明线 (main_plot)**: 该幕的主要剧情线，描述公开发生的事件
3. **暗线 (sub_plot)**: 该幕的隐藏剧情线，描述暗中进行的事件或角色的内心活动
4. **关键事件 (key_events)**: 该幕的关键事件列表（3-5个）
5. **角色关系变化 (relationship_changes)**: 该幕中角色关系的变化

## 输出格式

请以 JSON 格式输出，格式如下：

{{
  "acts": [
    {{
      "act_number": 1,
      "title": "第一幕标题",
      "main_plot": "明线描述",
      "sub_plot": "暗线描述",
      "key_events": ["事件1", "事件2", "事件3"],
      "relationship_changes": "角色关系变化描述"
    }},
    ...
  ],
  "overall_theme": "整体主题",
  "climax_act": 3,
  "resolution_act": {total_acts}
}}

要求：
- 故事应该有起承转合，有冲突和高潮
- 每一幕应该推进剧情，角色关系应该有所发展
- 明线和暗线应该相互呼应
- **重要**：明线和暗线必须填写具体内容，不能为空字符串，不能使用"剧情发展"、"隐藏线索"等占位符
- 每一幕的明线应该描述该幕中发生的具体事件和情节
- 每一幕的暗线应该描述该幕中隐藏的线索、角色的内心活动或未公开的秘密
- 关键事件应该具体描述，不能是空列表
- 输出必须是有效的 JSON 格式，不要包含 Markdown 代码块标记
"""
        return prompt
    
    def _build_english_prompt(self,
                              world_description: str,
                              character_names: List[str],
                              total_acts: int,
                              user_description: Optional[str] = None) -> str:
        """构建英文 prompt"""
        characters_text = "\n".join([f"- {name}" for name in character_names])
        
        prompt = f"""Please generate a complete event chain outline for the following story, containing {total_acts} acts.

## World Description
{world_description}

## Character List
{characters_text}
"""
        
        if user_description:
            prompt += f"""
## User Description
{user_description}
"""
        
        prompt += f"""
## Requirements

Please generate a complete story outline containing {total_acts} acts. Each act should include:

1. **Title**: The title of the act
2. **Main Plot**: The main plotline of the act, describing publicly occurring events
3. **Sub Plot**: The hidden plotline of the act, describing events happening in secret or characters' inner activities
4. **Key Events**: List of key events in the act (3-5 items)
5. **Relationship Changes**: Changes in character relationships in this act

## Output Format

Please output in JSON format as follows:

{{
  "acts": [
    {{
      "act_number": 1,
      "title": "Act 1 Title",
      "main_plot": "Main plot description",
      "sub_plot": "Sub plot description",
      "key_events": ["Event 1", "Event 2", "Event 3"],
      "relationship_changes": "Relationship changes description"
    }},
    ...
  ],
  "overall_theme": "Overall theme",
  "climax_act": 3,
  "resolution_act": {total_acts}
}}

Requirements:
- The story should have a beginning, development, climax, and resolution
- Each act should advance the plot, and character relationships should develop
- Main plot and sub plot should echo each other
- Output must be valid JSON format, do not include Markdown code block markers
"""
        return prompt
    
    def _parse_text_response(self, response: str, total_acts: int, language: str) -> Dict[str, Any]:
        """解析文本响应（当 JSON 解析失败时使用）"""
        acts = []
        lines = response.split('\n')
        
        current_act = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 尝试识别幕的开始
            if 'act' in line.lower() or '幕' in line or '第' in line:
                if current_act:
                    acts.append(current_act)
                current_act = {
                    "act_number": len(acts) + 1,
                    "title": line,
                    "main_plot": "",
                    "sub_plot": "",
                    "key_events": [],
                    "relationship_changes": ""
                }
            elif current_act:
                if '明线' in line or 'main' in line.lower():
                    current_act["main_plot"] = line
                elif '暗线' in line or 'sub' in line.lower():
                    current_act["sub_plot"] = line
                elif '关键' in line or 'key' in line.lower():
                    current_act["key_events"].append(line)
                elif '关系' in line or 'relationship' in line.lower():
                    current_act["relationship_changes"] = line
        
        if current_act:
            acts.append(current_act)
        
        # 如果解析的幕数不够，补充默认幕
        while len(acts) < total_acts:
            acts.append({
                "act_number": len(acts) + 1,
                "title": f"{'第' if language == 'zh' else 'Act '}{len(acts) + 1}{'幕' if language == 'zh' else ''}",
                "main_plot": f"{'剧情发展' if language == 'zh' else 'Plot development'}",
                "sub_plot": f"{'隐藏线索' if language == 'zh' else 'Hidden clues'}",
                "key_events": [],
                "relationship_changes": ""
            })
        
        return {
            "acts": acts[:total_acts],
            "overall_theme": "故事主题" if language == "zh" else "Story theme",
            "climax_act": max(1, total_acts) if total_acts == 1 else max(1, total_acts // 2),
            "resolution_act": total_acts
        }
    
    def _validate_and_normalize(self, data: Dict[str, Any], total_acts: int, language: str) -> Dict[str, Any]:
        """验证和规范化事件链数据"""
        if not isinstance(data, dict):
            return self._generate_default_event_chain(total_acts, language)
        
        acts = data.get("acts", [])
        if not isinstance(acts, list):
            acts = []
        
        # 确保有足够的幕
        while len(acts) < total_acts:
            acts.append({
                "act_number": len(acts) + 1,
                "title": f"{'第' if language == 'zh' else 'Act '}{len(acts) + 1}{'幕' if language == 'zh' else ''}",
                "main_plot": f"{'剧情发展' if language == 'zh' else 'Plot development'}",
                "sub_plot": f"{'隐藏线索' if language == 'zh' else 'Hidden clues'}",
                "key_events": [],
                "relationship_changes": ""
            })
        
        # 规范化每一幕的数据
        normalized_acts = []
        for i, act in enumerate(acts[:total_acts]):
            # 获取原始值，不要使用默认值
            main_plot = act.get("main_plot")
            sub_plot = act.get("sub_plot")
            
            # 如果值为空字符串或None，保持为空字符串（不要用默认值替换）
            if main_plot is None:
                main_plot = ""
            if sub_plot is None:
                sub_plot = ""
            
            normalized_act = {
                "act_number": i + 1,
                "title": act.get("title", f"{'第' if language == 'zh' else 'Act '}{i + 1}{'幕' if language == 'zh' else ''}"),
                "main_plot": main_plot,
                "sub_plot": sub_plot,
                "key_events": act.get("key_events", []) if isinstance(act.get("key_events"), list) else [],
                "relationship_changes": act.get("relationship_changes", "")
            }
            normalized_acts.append(normalized_act)
            print(f"[EventChainGenerator] 规范化第 {i+1} 幕: title={normalized_act['title']}, main_plot长度={len(main_plot)}, sub_plot长度={len(sub_plot)}")
        
        return {
            "acts": normalized_acts,
            "overall_theme": data.get("overall_theme", "故事主题" if language == "zh" else "Story theme"),
            "climax_act": data.get("climax_act", max(1, total_acts) if total_acts == 1 else max(1, total_acts // 2)),
            "resolution_act": data.get("resolution_act", total_acts),
            "total_acts": total_acts
        }
    
    def _generate_default_event_chain(self, total_acts: int, language: str) -> Dict[str, Any]:
        """生成默认事件链"""
        acts = []
        for i in range(total_acts):
            acts.append({
                "act_number": i + 1,
                "title": f"{'第' if language == 'zh' else 'Act '}{i + 1}{'幕' if language == 'zh' else ''}",
                "main_plot": f"{'剧情发展' if language == 'zh' else 'Plot development'}",
                "sub_plot": f"{'隐藏线索' if language == 'zh' else 'Hidden clues'}",
                "key_events": [],
                "relationship_changes": ""
            })
        
        return {
            "acts": acts,
            "overall_theme": "故事主题" if language == "zh" else "Story theme",
            "climax_act": max(1, total_acts) if total_acts == 1 else max(1, total_acts // 2),
            "resolution_act": total_acts,
            "total_acts": total_acts
        }
    
    def save_event_chain(self, event_chain: Dict[str, Any], save_path: str) -> bool:
        """
        保存事件链到文件
        
        Args:
            event_chain: 事件链数据
            save_path: 保存路径
            
        Returns:
            是否成功保存
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_json_file(save_path, event_chain)
            print(f"[EventChainGenerator] 事件链已保存到: {save_path}")
            return True
        except Exception as e:
            print(f"[EventChainGenerator] 保存事件链失败: {e}")
            return False
    
    def load_event_chain(self, load_path: str) -> Optional[Dict[str, Any]]:
        """
        从文件加载事件链
        
        Args:
            load_path: 文件路径
            
        Returns:
            事件链数据，如果加载失败则返回 None
        """
        try:
            if os.path.exists(load_path):
                return load_json_file(load_path)
            return None
        except Exception as e:
            print(f"[EventChainGenerator] 加载事件链失败: {e}")
            return None


if __name__ == "__main__":
    # 示例用法
    generator = EventChainGenerator(llm_name="gemini-2.5-pro")
    
    # 生成事件链示例
    # event_chain = generator.generate_event_chain(
    #     world_description="一个现代都市背景的故事",
    #     character_names=["张三", "李四", "王五"],
    #     total_acts=5,
    #     language="zh"
    # )
    # 
    # generator.save_event_chain(event_chain, "./data/event_chains/example.json")

