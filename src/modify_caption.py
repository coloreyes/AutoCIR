import time
import json
import requests
import torch
from typing import Optional, List


class ModelFactory:
    MODEL_REGISTRY = {
        "gpt-4o": {
            "url": "https://api.openai.com/v1/chat/completions",
            "model": "gpt-4o"
        },
        "gpt-4o-mini": {
            "url": "https://api.openai.com/v1/chat/completions",
            "model": "gpt-4o-mini"
        },
        "gpt-3.5-turbo": {
            "url": "https://api.openai.com/v1/chat/completions",
            "model": "gpt-3.5-turbo"
        },
        "qwen-turbo": {
            "url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            "model": "qwen-plus"
        }
    }

    def __init__(self, model_name: str, device: torch.device) -> None:
        if model_name not in self.MODEL_REGISTRY:
            raise ValueError(f"Unsupported model: {model_name}")
        
        self.device = device
        self.api_url = self.MODEL_REGISTRY[model_name]["url"]
        self.model_type = self.MODEL_REGISTRY[model_name]["model"]
        self.model_name = model_name
        self.modify_function = self.chat_with_api

    def build_messages(self, prompt: str) -> List[dict]:
        """Construct the message format."""
        return [{'role': 'user', 'content': prompt}]

    def chat_with_api(self, prompt: str, openai_key: str, max_length: int = 800) -> str:
        """Generic method for chatting with GPT or Qwen API."""
        payload = {
            "model": self.model_type,
            "messages": self.build_messages(prompt),
            "max_tokens": max_length,
            "temperature": 1.0
        }
        return self._send_request(payload, openai_key) or "Error during API call."

    def _send_request(self, payload: dict, openai_key: str, retries: int = 5, sleep_time: float = 0.25) -> Optional[str]:
        """Send the request to the API and handle retries."""
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {openai_key}'
        }

        for attempt in range(retries):
            try:
                response = requests.post(self.api_url, headers=headers, json=payload)
                if response.status_code != 200:
                    print(f"[{attempt+1}] HTTP Error: {response.status_code} - {response.text}")
                    time.sleep(sleep_time)
                    sleep_time *= 2
                    continue

                try:
                    data = response.json()
                    if 'error' in data:
                        print(f"[{attempt+1}] API Error: {data['error'].get('message', 'Unknown')}")
                        time.sleep(sleep_time)
                        sleep_time *= 2
                        continue

                    return data['choices'][0]['message']['content']
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"[{attempt+1}] Response parsing error: {e}")

            except requests.RequestException as e:
                print(f"[{attempt+1}] Request failed: {e}")

            time.sleep(sleep_time)
            sleep_time *= 2

        return None
