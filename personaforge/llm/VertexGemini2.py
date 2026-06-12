from .BaseLLM2 import BaseLLM
import os

class VertexGemini(BaseLLM):
    def __init__(self, model="gemini-1.5-pro-002", project_id=None, location="us-central1"):
        super(VertexGemini, self).__init__()
        self.model_name = model
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = location
        self.messages = []
        
        # Initialize Vertex AI client lazily; genai may not be installed in the environment.
        try:
            from google import genai
            self.client = genai.Client(
                vertexai=True,
                project=self.project_id,
                location=self.location
            )
        except Exception:
            # If genai (or google) is not available, set client to None.
            # get_response will raise a clear error when called.
            self.client = None

    def initialize_message(self):
        self.messages = []

    def ai_message(self, payload):
        self.messages.append({"role": "model", "parts": [{"text": payload}]})

    def system_message(self, payload):
        # Vertex AI Gemini may treat system messages as user-scoped instructions
        self.messages.append({"role": "user", "parts": [{"text": f"System: {payload}"}]})

    def user_message(self, payload):
        self.messages.append({"role": "user", "parts": [{"text": payload}]})

    def get_response(self, temperature=0.8, max_output_tokens=1024):
        # Build contents for the Vertex generate_content API
        contents = []
        for message in self.messages:
            contents.append({
                "role": message["role"],
                "parts": message["parts"]
            })
        
        # 检查client是否已初始化
        if self.client is None:
            raise RuntimeError("Vertex AI client is not initialized. Please check your Google Cloud credentials and project settings.")
        
        # 根据Google Gemini API的最新版本，generation_config参数已更改
        # 尝试多种方式以确保兼容性
        response = None
        last_exception = None
        
        # 方式1：尝试使用config参数（新版本API）
        try:
            from google.genai import types
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens
            )
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config
            )
        except (AttributeError, TypeError, ImportError) as e:
            last_exception = e
            # 方式2：尝试直接传递config字典
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config={
                        "temperature": temperature,
                        "max_output_tokens": max_output_tokens
                    }
                )
            except (TypeError, AttributeError) as e:
                last_exception = e
                # 方式3：尝试直接传递temperature和max_output_tokens作为关键字参数
                try:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=contents,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens
                    )
                except (TypeError, AttributeError) as e:
                    last_exception = e
                    # 方式4：只传递基本参数，不使用generation_config
                    try:
                        response = self.client.models.generate_content(
                            model=self.model_name,
                            contents=contents
                        )
                    except Exception as e:
                        last_exception = e
        
        # 如果所有方式都失败，抛出清晰的错误信息
        if response is None:
            error_msg = f"Failed to call generate_content API. Last error: {last_exception}. Please check your Google Gemini API version and parameters."
            raise RuntimeError(error_msg)
        
        # response.text contains the generated text in genai SDK
        return getattr(response, 'text', str(response))
    
    def chat(self, text, temperature=0.8, max_output_tokens=1024):
        self.initialize_message()
        self.user_message(text)
        response = self.get_response(temperature=temperature, max_output_tokens=max_output_tokens)
        return response
    
    def print_prompt(self):
        for message in self.messages:
            print(f"{message['role']}: {message['parts'][0]['text']}")
