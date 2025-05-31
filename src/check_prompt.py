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
        Assume you are an experienced composed image retrieval expert, skilled at precisely generating new image descriptions based on a reference image's description and the user's modification instructions.
        You excel at creating modified descriptions that can retrieve images matching the user's requested changes through vector retrieval.
        Your task is to help improve the effectiveness of compositional image retrieval by generating precise modification suggestions that will assist another large language model (LLM) in producing a better image description.
        Please note that this LLM has received the reference image's description and the user's modification instructions, and already generated a modified description.
        Moreover, a retrieval has been performed based on this modified description. Thus your task is to analyze the last retrieval result and provide modification suggestions and please follow the below steps to finish this task.
        First Step: Identifying Modifications
        Your first task is to identify the modifications and generate corresponding modification phrases.
        Specifically, here is the description of the reference image: "{image_caption}." Here are the user's modification requests: "{relative_caption}"
        By deeply understanding the image description and the user's modifications, please generate the following two types of modification phrases:
        1. If the modification involves changing the characteristics of an entity in the original reference image, please specify the changes,
        2. If the modification involves adding or deleting an entity, please specify the additions or deletions.
        Please note that the user's modifications may lack a subject; in such cases, infer and supply the object corresponding to the modification.
        Only include modifications explicitly mentioned by the user. If a certain type of modification is not present, you do not need to provide it and should avoid generating unspecified content.

        Step 2: Analyzing the Top-{len(top_captions)} Retrieved Image 
        Compare the modification phrases identified in Step 1 with the description of the top-{len(top_captions)} retrieved image : "{top_captions}". Note that this retrieval is performed with the modified description generated by another LLM, which has been mentioned above.
        Determine if the retrieved image meets the user's modification instructions.
        If it matches after excluding subjective modifications (e.g., "casual," "relaxed"), respond with: "Good retrieval, no more loops needed."
        If there are unmet modification phrases, proceed to Step 3.

        Step 3: Providing Modification Suggestions 
        For any unmet modifications identified in Step 2, suggest targeted changes to help the LLM regenerate an improved modified description. Keep suggestions concise and specific to ensure they effectively guide the LLM.
        '''
