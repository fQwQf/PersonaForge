from .BaseLLM2 import BaseLLM
from openai import OpenAI
import os

class LangChainGPT(BaseLLM):

    def __init__(self, model="gpt-4o-mini"):
        super(LangChainGPT, self).__init__()
        # Support custom API base (mirror) via OPENAI_API_BASE environment variable
        api_base = os.getenv("OPENAI_API_BASE", "")
        if api_base:
            # Use base_url param (used elsewhere in code)
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=api_base)
        else:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        self.model_name = model
        self.messages = []

    def initialize_message(self):
        self.messages = []

    def ai_message(self, payload):
        self.messages.append({"role": "ai", "content": payload})

    def system_message(self, payload):
        self.messages.append({"role": "system", "content": payload})

    def user_message(self, payload):
        self.messages.append({"role": "user", "content": payload})

    def get_response(self, temperature=0.8):
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messages,
            temperature=temperature,
            top_p=0.8
        )
        return completion.choices[0].message.content

    def chat(self, text, temperature=0.8):
        self.initialize_message()
        self.user_message(text)
        response = self.get_response(temperature=temperature)
        return response

    def print_prompt(self):
        for message in self.messages:
            print(message)
