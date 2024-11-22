from classes import Captioner
import argparse
import prompts
from argparse import Namespace
from pathlib import Path
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
import os
import tqdm
import pandas as pd

class GenerateCaptions:
    def __init__(self,args:Namespace) -> None:
        super().__init__()
        # Get Arguments
        for arg in vars(args):
            value_arg = getattr(args,arg)
            self.__setattr__(arg,value_arg)

    def run(self):
        self.current_directory = Path(__file__).resolve().parent
        captioner_model,captioner_processor = self.load_captioner_model()
        if self.captioner is Captioner.Qwen2_VL_7B_Instruct:
            self.generater = self.qwen_caption
        elif self.captioner is Captioner.blip2_opt_6_7B:
            self.generater = self.BLIP_caption
        elif self.captioner is Captioner.CoCa:
            self.generater = self.coca_caption
        elif self.captioner is Captioner.llava_onevision_qwen2_7b_ov_hf:
            self.generater = self.llava_caption
        img_file_list = os.listdir(self.image_path)
        image_id_list = []
        generated_text_list = []
        for img_file in tqdm(img_file_list):
            url = self.image_path + img_file
            image = Image.open(url)
            inputs = captioner_processor(images=image,  return_tensors="pt").to(self.device)
            generated_ids = captioner_model.generate(**inputs)
            generated_text = captioner_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            image_id_list.append(img_file[:-4])
            generated_text_list.append(generated_text)
        df = pd.DataFrame({"image_id": image_id_list, "generated_text": generated_text_list})
        df.to_csv(self.output_path, index=False)        
        print("Finish.")
        return
    
    def load_captioner_model(self):
        self.captioner = Captioner.from_string(s=self.captioner)
        print('==================Loading Captioner Model==================')
        captioner_model,captioner_processor = self.captioner.load_model_and_preprocess(device=self.device,path=self.current_directory)
        print('Done.')
        return captioner_model,captioner_processor

    def qwen_caption(model, processor, image_path, device):
        text_prompt = f"""Describe the image concisely."""
        messages = [
            {
            "role": "user",
            "content": [
                {'type': 'image', 'image': image_path},
                {"type": "text", "text": text_prompt},
                ],
            },
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        generated_ids = model.generate(**inputs, max_new_tokens=512,temperature=0.5)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text

    def BLIP_caption(model, processor, image_path, device):
        raw_image = Image.open(image_path).convert('RGB')
        text = "a photography of"
        inputs = processor(raw_image, text, return_tensors="pt").to(device)
        out = model.generate(**inputs)
        return processor.decode(out[0], skip_special_tokens=True).replace("a photography of", "")

    def llava_caption(model, processor, image_path, device):
        text_prompt = f"""Describe the image concisely."""
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": text_prompt},
                    ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(conversation)
        inputs = processor(images=image_inputs, text=prompt, return_tensors="pt").to(device, torch.float16)

        out = model.generate(**inputs, max_new_tokens=512, pad_token_id=processor.tokenizer.pad_token_id)
        output_text = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return output_text[-1].split("assistant")[-1]

    def coca_caption(model, processor, tokenizer, image_path):
        image = Image.open(image_path)
        image_input = processor(image).unsqueeze(0)  
        with torch.no_grad():
            generated_text = model.generate(image_input,seq_len=512)  
        generated_text = tokenizer.decode(generated_text[0].cpu().numpy())
        return generated_text.replace("<start_of_text>", "").replace("<end_of_text>", "")

def parse_arguments():
    parser = argparse.ArgumentParser()
    # Base Arguments
    parser.add_argument("--device", type=int, default=0, 
                        help="GPU ID to use.")
    # Dataset Arguments ['dress', 'toptee', 'shirt']
    parser.add_argument("--image_path", default="../datasets/FASHIONIQ/images/", type=str, required=False, 
                        choices=['cirr', 'circo',
                                 'fashioniq_dress', 'fashioniq_toptee', 'fashioniq_shirt'],
                        help="Dataset to use")
    # Base Model Arguments
    parser.add_argument("--captioner", type=str, default='blip2-opt-6.7B', choices=['blip2-opt-6.7B','Qwen2-VL-7B-Instruct','llava-onevision-qwen2-7b-ov-hf','CoCa'],
                        help="Which model to use for generating image captions.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    GenerateCaptions(args).run()