from .BaseLLM import BaseLLM
from google import genai
from google.genai import types
import os
import time
import threading
import json
from typing import Optional, List, Type, TypeVar
from pydantic import BaseModel
from openai import OpenAI

T = TypeVar('T', bound=BaseModel)

class Gemini(BaseLLM):
    """
    Gemini API 封装类，用于调用 Google 的 Gemini 模型（使用新版API）。
    
    需要设置环境变量 GEMINI_API_KEY 或在 config.json 中配置 GEMINI_API_KEY。
    支持的模型包括：
    - gemini-2.0-flash
    - gemini-1.5-flash
    - gemini-1.5-pro
    - gemini-2.5-flash
    - gemini-2.5-flash-lite
    - gemini-2.5-pro
    
    注意：本类使用新版API (`from google import genai`)，而不是旧版API (`google.generativeai`)。
    """
    
    def __init__(self, model="gemini-2.5-flash-lite", timeout: Optional[int] = 20, display_name: Optional[str] = None):
        """
        初始化 Gemini 客户端。
        
        Args:
            model: 模型名称，默认为 gemini-2.5-flash-lite
            timeout: API 调用超时时间（秒），默认 20 秒
            display_name: 用于日志输出的模型名称
        """
        super(Gemini, self).__init__()
        self.model_name = model
        self.display_model_name = display_name or model
        self.messages = []
        self.system_instruction = None
        self.timeout = timeout
        self.max_retries = 3  # 最大重试次数（对于网络错误，最多重试3次）

        # 配置 API Key（支持多个 key 轮换）
        self.api_keys: List[str] = self._load_api_keys()
        self._api_key_lock = threading.Lock()
        self._api_key_index = 0
        self._current_api_key = None
        self._client = None
        
        # 如果有 Google API Key，先配置第一个 key
        if self.api_keys:
            initial_key = self.api_keys[0]
            self._configure_client(initial_key)
        # 客户端将在需要时创建
        
        # 检查是否有备用中转 API（OpenAI 兼容）
        self.fallback_api_base = os.getenv("OPENAI_API_BASE", "")
        self.fallback_api_key = os.getenv("OPENAI_API_KEY", "")
        self.fallback_client = None
        if self.fallback_api_base and self.fallback_api_key:
            try:
                self.fallback_client = OpenAI(api_key=self.fallback_api_key, base_url=self.fallback_api_base)
                print(f"[Gemini] 已配置备用中转 API: {self.fallback_api_base}")
            except Exception as e:
                print(f"[Gemini] 警告：备用中转 API 配置失败: {e}")
                self.fallback_client = None

    def _load_api_keys(self) -> List[str]:
        """
        从环境变量中加载 API Key。支持以下格式：
        - GEMINI_API_KEYS: JSON 数组或逗号/分号分隔的字符串
        - GEMINI_API_KEY: 单个 key 或逗号/分号分隔的多个 key
        
        注意：如果 GEMINI_API_KEY 为空，但配置了备用中转 API（OPENAI_API_BASE），
        则允许使用备用 API 作为唯一方案。
        """
        raw_value = os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY")
        if not raw_value:
            # 检查是否有备用 API
            fallback_base = os.getenv("OPENAI_API_BASE", "")
            fallback_key = os.getenv("OPENAI_API_KEY", "")
            if fallback_base and fallback_key:
                # 允许只使用备用 API，返回空列表
                return []
            else:
                raise ValueError(
                    "未检测到 GEMINI_API_KEY 或 GEMINI_API_KEYS，且未配置备用中转 API。"
                    "请在 config.json 中配置 GEMINI_API_KEY 或 OPENAI_API_BASE + OPENAI_API_KEY。"
                )

        keys: List[str] = []
        raw_value = raw_value.strip()
        if raw_value.startswith("["):
            try:
                parsed = json.loads(raw_value)
                if isinstance(parsed, list):
                    keys = [str(item).strip() for item in parsed if str(item).strip()]
            except json.JSONDecodeError:
                pass

        if not keys:
            separators = [",", ";"]
            temp_value = raw_value
            for sep in separators:
                temp_value = temp_value.replace(sep, ",")
            keys = [item.strip() for item in temp_value.split(",") if item.strip()]

        if not keys:
            raise ValueError("未找到有效的 Gemini API Key，请检查配置。")

        return keys

    def _get_next_api_key(self) -> str:
        """轮询获取下一个 API Key。"""
        if not self.api_keys:
            raise ValueError("没有可用的 Google API Key")
        with self._api_key_lock:
            key = self.api_keys[self._api_key_index]
            self._api_key_index = (self._api_key_index + 1) % len(self.api_keys)
            return key

    def _configure_client(self, api_key: str):
        """使用指定 key 配置 Gemini 客户端（新版API）。"""
        if api_key != self._current_api_key:
            # 新版API：设置环境变量
            os.environ['GEMINI_API_KEY'] = api_key
            self._current_api_key = api_key
            # 创建客户端（使用Gemini Developer API，不是Vertex AI）
            self._client = genai.Client(vertexai=False)
            masked_key = api_key[:4] + "..." if len(api_key) > 8 else "****"
            print(f"[Gemini] 切换 API Key: {masked_key}，使用新版API")

    def initialize_message(self):
        """初始化消息列表。"""
        self.messages = []
        self.system_instruction = None

    def ai_message(self, payload):
        """
        添加 AI 回复消息。
        
        Args:
            payload: AI 消息内容（字符串）
        """
        self.messages.append({"role": "model", "content": payload})

    def system_message(self, payload):
        """
        添加系统提示消息。
        
        Args:
            payload: 系统提示内容
        """
        self.system_instruction = payload

    def user_message(self, payload):
        """
        添加用户消息。
        
        Args:
            payload: 用户消息内容
        """
        self.messages.append({"role": "user", "content": payload})

    def get_response(self, temperature=0.8, response_model: Optional[Type[T]] = None):
        """
        获取模型响应，带超时和重试机制。
        
        Args:
            temperature: 温度参数，控制输出的随机性，默认 0.8
            response_model: 可选的Pydantic模型，用于结构化输出
            
        Returns:
            如果提供了response_model，返回解析后的Pydantic模型实例
            否则返回模型生成的文本响应
        """
        last_exception = None
        
        # 如果没有 Google API Key，直接使用备用 API
        if not self.api_keys:
            if self.fallback_client:
                print(f"[Gemini] 未配置 Google API Key，直接使用备用中转 API")
                return self._get_response_fallback(temperature)
            else:
                raise ValueError("未配置 Google API Key 且未配置备用中转 API")
        
        # 重试机制
        for attempt in range(self.max_retries):
            try:
                api_key = self._get_next_api_key()
                self._configure_client(api_key)

                print(f"[Gemini] 开始 API 调用（尝试 {attempt + 1}/{self.max_retries}），模型: {self.display_model_name}，超时: {self.timeout}秒，使用新版API")
                
                # 确保客户端已创建
                if not self._client:
                    api_key = self._get_next_api_key()
                    self._configure_client(api_key)
                
                # 使用 threading 实现跨平台超时
                response_result = [None]
                exception_result = [None]
                
                def api_call():
                    """执行 API 调用的函数（新版API）"""
                    try:
                        # 构建contents（新版API格式）
                        # 对于单次对话，直接使用字符串
                        # 对于聊天历史，将所有消息合并成一个字符串
                        if len(self.messages) == 0:
                            prompt_text = ""
                        elif len(self.messages) == 1:
                            # 单次对话
                            prompt_text = self.messages[0]["content"]
                        else:
                            # 多条消息：合并成对话格式
                            # 格式：用户消息1\n模型回复1\n用户消息2\n...
                            conversation_parts = []
                            for msg in self.messages:
                                role_prefix = "用户" if msg["role"] == "user" else "助手"
                                conversation_parts.append(f"{role_prefix}: {msg['content']}")
                            prompt_text = "\n".join(conversation_parts)
                        
                        # 如果有system_instruction，添加到prompt开头
                        if self.system_instruction:
                            prompt_text = f"系统指令: {self.system_instruction}\n\n{prompt_text}"
                        
                        # 构建config参数（温度设置和结构化输出）
                        config_dict = {"temperature": temperature}
                        
                        # 如果提供了response_model，添加结构化输出配置
                        if response_model:
                            config_dict["response_mime_type"] = "application/json"
                            config_dict["response_json_schema"] = response_model.model_json_schema()
                        
                        config = types.GenerateContentConfig(**config_dict)
                        
                        # 调用新版API（contents可以是字符串或列表）
                        response_result[0] = self._client.models.generate_content(
                            model=self.model_name,
                            contents=prompt_text,
                            config=config
                        )
                    except Exception as e:
                        exception_result[0] = e
                
                # 在单独线程中执行 API 调用
                api_thread = threading.Thread(target=api_call)
                api_thread.daemon = True
                api_thread.start()
                api_thread.join(timeout=self.timeout)
                
                # 检查是否超时
                if api_thread.is_alive():
                    raise TimeoutError(f"Gemini API 调用超时（{self.timeout}秒）")
                
                # 检查是否有异常
                if exception_result[0]:
                    raise exception_result[0]
                
                # 获取响应
                response = response_result[0]
                
                # 检查响应是否有效（新版API）
                if not response:
                    raise ValueError("Gemini API 返回了无效的响应")
                
                # 新版API的响应格式
                response_text = response.text if hasattr(response, 'text') else str(response)
                
                if not response_text:
                    raise ValueError("Gemini API 返回了空的响应")
                
                # 如果提供了response_model，解析为Pydantic模型
                if response_model:
                    try:
                        parsed_model = response_model.model_validate_json(response_text)
                        print(f"[Gemini] API 调用成功，结构化输出解析成功")
                        return parsed_model
                    except Exception as e:
                        print(f"[Gemini] 结构化输出解析失败: {e}")
                        print(f"[Gemini] 原始响应: {response_text[:500]}")
                        raise ValueError(f"无法解析结构化输出: {e}")
                
                print(f"[Gemini] API 调用成功，响应长度: {len(response_text)} 字符")
                return response_text
                
            except TimeoutError as e:
                last_exception = e
                print(f"Gemini API 调用超时（尝试 {attempt + 1}/{self.max_retries}）: {e}")
                if attempt < self.max_retries - 1:
                    wait_time = (attempt + 1) * 2  # 递增等待时间：2秒、4秒、6秒
                    print(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    raise
            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()
                error_type = type(e).__name__.lower()
                
                # 检查是否是网络错误或可重试的错误
                # 包括：连接重置、超时、网络错误、服务器错误等
                retryable_errors = [
                    'timeout', 'connection', 'network', 'rate limit', 
                    '429', '503', '502', '500', '504',
                    'reset', 'peer', 'refused', 'unreachable',
                    'connecterror', 'httpxerror', 'httperror'
                ]
                
                # 检查错误消息和错误类型
                is_retryable = (
                    any(keyword in error_msg for keyword in retryable_errors) or
                    any(keyword in error_type for keyword in retryable_errors) or
                    'errno 54' in error_msg or  # Connection reset by peer
                    'errno 61' in error_msg or  # Connection refused
                    'errno 51' in error_msg     # Network unreachable
                )
                
                if is_retryable and attempt < self.max_retries - 1:
                    print(f"[Gemini] API 调用错误（尝试 {attempt + 1}/{self.max_retries}）: {e}")
                    print(f"[Gemini] 错误类型: {error_type}, 错误消息: {error_msg[:200]}")
                    wait_time = (attempt + 1) * 2  # 递增等待时间：2秒、4秒、6秒
                    print(f"[Gemini] 等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    print(f"[Gemini] API 调用错误: {e}")
                    print(f"[Gemini] 错误类型: {error_type}")
                    if not is_retryable:
                        print(f"[Gemini] 此错误不可重试，直接抛出")
                    import traceback
                    traceback.print_exc()
                    raise
        
        # 如果所有重试都失败，尝试使用备用中转 API
        if last_exception and self.fallback_client:
            print(f"[Gemini] Google API 调用失败，切换到备用中转 API")
            try:
                return self._get_response_fallback(temperature)
            except Exception as fallback_error:
                print(f"[Gemini] 备用中转 API 也失败: {fallback_error}")
                # 如果备用 API 也失败，抛出原始错误
                raise last_exception
        
        # 如果没有备用 API 或备用 API 未配置，抛出原始错误
        if last_exception:
            raise last_exception
    
    def chat(self, text, temperature=0.8, response_model: Optional[Type[T]] = None):
        """
        简单的聊天接口。
        
        Args:
            text: 用户输入的文本
            temperature: 温度参数，默认 0.8
            response_model: 可选的Pydantic模型，用于结构化输出
            
        Returns:
            如果提供了response_model，返回解析后的Pydantic模型实例
            否则返回模型生成的文本响应
        """
        self.initialize_message()
        self.user_message(text)
        response = self.get_response(temperature=temperature, response_model=response_model)
        return response
    
    def _get_response_fallback(self, temperature=0.8):
        """
        使用备用中转 API（OpenAI 兼容）获取响应。
        
        Args:
            temperature: 温度参数，控制输出的随机性，默认 0.8
            
        Returns:
            模型生成的文本响应
        """
        if not self.fallback_client:
            raise ValueError("备用中转 API 未配置")
        
        # 转换消息格式为 OpenAI 格式
        openai_messages = []
        
        # 如果有 system_instruction，添加为 system 消息
        if self.system_instruction:
            openai_messages.append({
                "role": "system",
                "content": self.system_instruction
            })
        
        # 转换历史消息
        for msg in self.messages:
            role = msg["role"]
            # Gemini 使用 "model"，OpenAI 使用 "assistant"
            if role == "model":
                role = "assistant"
            elif role == "user":
                role = "user"
            else:
                # 其他角色保持原样或映射为 user
                role = "user"
            
            openai_messages.append({
                "role": role,
                "content": msg["content"]
            })
        
        print(f"[Gemini] 使用备用中转 API 调用，模型: {self.display_model_name}")
        
        try:
            completion = self.fallback_client.chat.completions.create(
                model=self.model_name,
                messages=openai_messages,
                temperature=temperature,
                top_p=0.8
            )
            response_text = completion.choices[0].message.content
            print(f"[Gemini] 备用中转 API 调用成功，响应长度: {len(response_text) if response_text else 0} 字符")
            return response_text
        except Exception as e:
            print(f"[Gemini] 备用中转 API 调用失败: {e}")
            raise
    
    def print_prompt(self):
        """打印当前的消息历史（用于调试）。"""
        if self.system_instruction:
            print(f"System: {self.system_instruction}")
        for message in self.messages:
            print(f"{message['role']}: {message['content']}")