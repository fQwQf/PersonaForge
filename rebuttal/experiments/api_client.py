"""
Secure API Client for PersonaForge Experiments - Updated for google-genai
Uses environment variables for API keys (never hardcoded)
Supports xray proxy for China network access
"""

import os
import time
from typing import Optional, Dict, Any
from google import genai
from google.genai import types

class SecureAPIClient:
    """Secure API client that reads keys from environment variables."""
    
    def __init__(self, api_key: Optional[str] = None, proxy_url: str = "http://127.0.0.1:10808"):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("API key not found. Set GEMINI_API_KEY environment variable.")
        
        self.proxy_url = proxy_url
        self._setup_proxy()
        self._setup_gemini()
        self.last_call_time = 0
        self.min_delay = 1.0
        
    def _setup_proxy(self):
        """Configure proxy for requests"""
        os.environ['HTTP_PROXY'] = self.proxy_url
        os.environ['HTTPS_PROXY'] = self.proxy_url
        
    def _setup_gemini(self):
        """Configure Gemini API with new client"""
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = "gemini-2.0-flash"
        
    def _rate_limit(self):
        """Ensure minimum delay between API calls"""
        elapsed = time.time() - self.last_call_time
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)
        self.last_call_time = time.time()
        
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1024) -> Dict[str, Any]:
        """Generate text using Gemini API."""
        self._rate_limit()
        
        start_time = time.time()
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )
            latency_ms = (time.time() - start_time) * 1000
            
            return {
                'text': response.text,
                'latency_ms': latency_ms,
                'error': None
            }
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return {
                'text': '',
                'latency_ms': latency_ms,
                'error': str(e)
            }

    def evaluate_personality_consistency(self, character: str, response: str, trait: str) -> float:
        """Evaluate if a response is consistent with a character's personality."""
        prompt = f"""Evaluate if the following response is consistent with {character}'s personality trait: {trait}

Response: {response}

Rate consistency from 0.0 (completely inconsistent) to 1.0 (perfectly consistent).
Return ONLY a number between 0.0 and 1.0."""

        result = self.generate(prompt, temperature=0.1, max_tokens=10)
        
        if result['error']:
            return 0.5
            
        try:
            import re
            match = re.search(r'0?\.\d+', result['text'].strip())
            if match:
                return float(match.group())
            return 0.5
        except:
            return 0.5

def get_api_client():
    """Factory function to get configured API client"""
    return SecureAPIClient()
