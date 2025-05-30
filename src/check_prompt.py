import json
import requests
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from enum import Enum, auto

class CheckModel(Enum):
    gpt_4o = auto()
    gpt_4o_mini = auto()
    gpt_3_5_turbo = auto()
    qwen_turbo = auto()

    @staticmethod
    def from_string(s: str):
        try:
            return CheckModel[s.replace("-", "_").lower()]
        except KeyError:
            raise ValueError(f"Unknown model type: {s}")
        
class ModelHandler:
    def __init__(self, model_type: str, device: torch.device, openai_key: str = None):
        self.model_type = CheckModel.from_string(model_type.replace("-", "_"))
        self.device = device
        self.openai_key = openai_key

        # Load model depending on whether it's local or API-based
        if self.model_type in [CheckModel.gpt_4o_mini, CheckModel.qwen_turbo]:
            self.model = None
            self.tokenizer = None
            self.chat_function = self._chat_api
        else:
            self.model, self.tokenizer = self._load_model_and_process(device)
            self.chat_function = self._chat_local
    
    def _load_model_and_process(self, device: torch.device):
        tokenizer = AutoTokenizer.from_pretrained(self.model_type.value)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_type.value, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map=device, attn_implementation="flash_attention_2"
        )
        return model, tokenizer

    def _send_request(self, url, headers, payload, max_retries=5, backoff=0.25):
        """Generic function for sending requests with retry logic."""
        for _ in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload)
                if response.status_code != 200:
                    print(f"Error: {response.status_code} - {response.text}")
                    continue  # Proceed to next retry

                try:
                    response_data = response.json()
                    if 'error' in response_data:
                        print(f"API Error: {response_data['error']['message']}")
                        continue  # Proceed to next retry
                    return response_data['choices'][0]['message']['content']
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error decoding response: {e}")
                    continue  # Proceed to next retry

            except requests.RequestException as e:
                print(f"Request failed: {e}")

            # Wait before retrying
            time.sleep(backoff)
            backoff *= 2  # Exponential backoff

        return "Request failed after multiple attempts"
   
    def _chat_local(self, top_captions, image_caption, relative_caption, max_length=800):
        prompt = self._generate_check_prompt(image_caption, relative_caption, top_captions, len(top_captions))
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        generated_ids = self.model.generate(
            inputs,
            max_length=max_length,
            temperature=1.0,
            num_beams=5,  # Using beam search for better results
            early_stopping=True
        )
        
        output_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        return output_text

    def _chat_api(self, top_captions, image_caption, relative_caption, max_length=800):
        model_map = {
            CheckModel.gpt_4o: {
                "url": "https://api.openai.com/v1/chat/completions",
                "model_name": "gpt-4o"
            },
            CheckModel.gpt_4o_mini: {
                "url": "https://api.openai.com/v1/chat/completions",
                "model_name": "gpt-4o-mini"
            },
            CheckModel.gpt_3_5_turbo: {
                "url": "https://api.openai.com/v1/chat/completions",
                "model_name": "gpt-3.5-turbo"
            },
            CheckModel.qwen_turbo: {
                "url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                "model_name": "qwen-plus"
            }
        }

        if self.model_type not in model_map:
            raise ValueError("Unsupported API model.")

        model_info = model_map[self.model_type]
        prompt = self._generate_check_prompt(image_caption, relative_caption, top_captions, len(top_captions))

        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'Authorization': f'Bearer {self.openai_key}'
        }

        payload = {
            "model": model_info["model_name"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_length,
            "temperature": 1.0
        }

        return self._send_request(model_info["url"], headers, payload)

    def _generate_check_prompt(self, image_caption, relative_caption, top_captions, num_top_captions):
        return f'''
        You are a skilled image retrieval expert. Your job is to help improve compositional image descriptions based on a reference image and a user's modification request.
        The LLM has already generated a modified caption based on the reference image and user's instruction. Now, a retrieval has been performed using that caption, and you are given the top-{len(top_captions)} retrieved image captions.
        Please follow the 3-step reasoning process below in your thinking, but only output the final suggestion result.

        Step 1: Identify Modifications
        Analyze the following:
        - Reference caption: "{image_caption}"
        - User modification instruction: "{relative_caption}"

        Extract and list:
        - Changing characteristics (e.g. “long sleeves” → “short sleeves”)
        - Adding or deleting entities (e.g. “add hat”, “remove belt”)

        Step 2: Compare with Retrieved Results
        Compare the modification phrases above with the retrieved captions:
        {top_captions}
        Mark which modifications are satisfied, and which are unmet. Focus only on unmet ones.

        Step 3: Suggest Revisions
        For each unmet modification, suggest how the previously generated caption can be improved.

        Important: Only output the following section
        Improvement Suggestions:  
        - "modification phrase 1": ...  
        - "modification phrase 2": ...  
        '''