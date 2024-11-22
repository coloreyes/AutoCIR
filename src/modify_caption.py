from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import openai
import json
import time
import requests
Qwen2_7B_model_file = '/home/uestc_zhou/myh/model/qwen/Qwen2-7B-Instruct'

class modify_factory:
    def __init__(self,model_name,device:torch.device) -> None:
        self.device = device
        if model_name == 'qwen_turbo':
            self.modify_function = self.chat_qwen_trubo
        elif model_name == 'gpt-4o-mini':
            self.modify_function = self.chat_gpt_4o_mini
    def load_model_and_tokenizer(self):
        model = AutoModelForCausalLM.from_pretrained(
        Qwen2_7B_model_file,
        torch_dtype=torch.bfloat16,
        device_map = self.device
        )
        tokenizer = AutoTokenizer.from_pretrained(Qwen2_7B_model_file)
        return model,tokenizer

    def generate(self,prompt:str,max_length:int = 100)->str:
        messages = self.build_messages(prompt)
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors='pt').to(self.device)
        attention_mask = torch.ones(
            model_inputs.input_ids.shape, dtype=torch.long, device=self.device)
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=max_length,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.eos_token_id
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def build_messages(self,prompt:str)->list[dict[str, str]]:
        messages = [
        {'role': 'user', 'content': f'{prompt}'}]
        return messages

    def chat_gpt_4o_mini(self, prompt: str, openai_key: str, max_length=800) -> str:
        final_result = ''
        # url = 'https://api.openai.com/v1/chat/completions'
        url = 'https://forward.ronpay.org/v1/chat/completions'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {openai_key}'
        }
        payload = {
            # "model": "gpt-3.5-turbo",  
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_length,
            "temperature": 1.0
        }

        sleep_time = 0.25
        for _ in range(5):
            try:
                # 发送请求
                response = requests.post(url, headers=headers, json=payload)
                
                # 如果响应状态码不为 200，输出错误信息
                if response.status_code != 200:
                    print(f"Error executing request: {response.status_code} - {response.text}")
                    final_result = "Error executing request"
                
                # 尝试解析 JSON 响应
                try:
                    response_data = response.json()
                    if 'error' in response_data:
                        error_message = response_data['error'].get('message', 'Unknown error occurred')
                        print(f"API Error: {error_message}")
                        final_result = f"API Error: {error_message}"
                    return response_data['choices'][0]['message']['content']
                
                except json.JSONDecodeError:
                    print("Failed to decode JSON response.")
                    final_result = "Error: Failed to decode JSON response."
                except KeyError:
                    print("Unexpected response format.")
                    final_result = "Error: Unexpected response format."
            
            except requests.RequestException as e:
                # 捕获请求异常并输出错误
                print(f"Request failed: {e}")
                final_result = "Request error!"
            
            # 如果请求失败或响应错误，等待 1 秒再重试
            time.sleep(1)
            sleep_time = sleep_time * 2

        return final_result
    
    def chat_qwen_trubo(self, prompt: str, openai_key: str, max_length=800) -> str:
        final_result = ''
        url = 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {openai_key}'
        }
        payload = {
            "model": "qwen-plus",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_length,
            "temperature": 1.0
        }

        sleep_time = 0.25
        for _ in range(5):
            try:
                # 发送请求
                response = requests.post(url, headers=headers, json=payload)
                
                # 如果响应状态码不为 200，输出错误信息
                if response.status_code != 200:
                    print(f"Error executing request: {response.status_code} - {response.text}")
                    final_result = "Error executing request"
                
                # 尝试解析 JSON 响应
                try:
                    response_data = response.json()
                    if 'error' in response_data:
                        error_message = response_data['error'].get('message', 'Unknown error occurred')
                        print(f"API Error: {error_message}")
                        final_result = f"API Error: {error_message}"
                    return response_data['choices'][0]['message']['content']
                
                except json.JSONDecodeError:
                    print("Failed to decode JSON response.")
                    final_result = "Error: Failed to decode JSON response."
                except KeyError:
                    print("Unexpected response format.")
                    final_result = "Error: Unexpected response format."
            
            except requests.RequestException as e:
                # 捕获请求异常并输出错误
                print(f"Request failed: {e}")
                final_result = "Request error!"
            
            # 如果请求失败或响应错误，等待 1 秒再重试
            time.sleep(sleep_time)
            sleep_time = sleep_time * 2

        return final_result
