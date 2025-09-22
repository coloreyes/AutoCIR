import os
import argparse
import warnings
import logging
import torch
import pandas as pd
from tqdm import tqdm

from transformers import (
    Blip2Processor, Blip2ForConditionalGeneration,
    Qwen2VLProcessor, Qwen2VLForConditionalGeneration,
    LlavaNextProcessor, LlavaNextForConditionalGeneration,
    AutoProcessor, AutoModelForCausalLM
)
from PIL import Image
from qwen_vl_utils import process_vision_info

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)
import open_clip
def load_model_and_processor(captioner, device):
    """
    根据模型类型加载对应的模型和处理器
    """
    if captioner == "blip2":
        processor = Blip2Processor.from_pretrained("../model/blip/blip2-opt-6.7b")
        model = Blip2ForConditionalGeneration.from_pretrained("../model/blip/blip2-opt-6.7b", device_map=device)
        tokenizer = None

    elif captioner == "qwen2_vl":
        processor = Qwen2VLProcessor.from_pretrained("../model/Qwen/Qwen2-VL-7B-Instruct")
        model = Qwen2VLForConditionalGeneration.from_pretrained("../model/Qwen/Qwen2-VL-7B-Instruct", device_map=device)
        tokenizer = None

    elif captioner == "llava_ov":
        processor = LlavaNextProcessor.from_pretrained("../model/llava-hf/llava-onevision-qwen2-7b-ov-hf")
        model = LlavaNextForConditionalGeneration.from_pretrained("../model/llava-hf/llava-onevision-qwen2-7b-ov-hf", device_map=device)
        tokenizer = None

    elif captioner == "coca":
        model, _, processor = open_clip.create_model_and_transforms(
            "coca_ViT-L-14", pretrained="laion2B-s13B-b90k"
        )
        tokenizer = open_clip.get_tokenizer("coca_ViT-L-14")
    else:
        raise ValueError(f"Unsupported model type: {captioner}")

    return processor, model, tokenizer


def generate_caption(model_type, model, processor, tokenizer, img_path, device):
    prompt = "Describe the image in complete detail. You must especially focus on all the objects in the image."
    if model_type == "blip2":
        raw_image = Image.open(img_path).convert('RGB')
        inputs = processor(raw_image, prompt, return_tensors="pt").to(device)
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        return processor.decode(generated_ids[0], skip_special_tokens=True)

    elif model_type == "qwen2_vl":
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img_path, "max_pixels": 360 * 420},
                {"type": "text", "text": prompt},
            ]
        }]
        image_inputs, _ = process_vision_info(messages)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(device)
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    elif model_type == "llava_ov":
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": prompt},
                    ],
            },
        ]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(conversation)
        inputs = processor(images=image_inputs, text=text, return_tensors="pt").to(device, torch.float16)

        out = model.generate(**inputs, max_new_tokens=512, pad_token_id=processor.tokenizer.pad_token_id)
        output_text = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return output_text[-1].split("assistant")[-1]

    elif model_type == "coca":
        image = Image.open(img_path)
        image_input = processor(image).unsqueeze(0)
        with torch.no_grad():
            generated_text = model.generate(image_input,seq_len=512)
        generated_text = tokenizer.decode(generated_text[0].cpu().numpy())
        return generated_text.replace("<start_of_text>", "").replace("<end_of_text>", "")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def generate_captions(img_dir, output_csv, captioner, batch_size=1, device=None):
    """
    遍历目录生成 captions
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor, model, tokenizer = load_model_and_processor(captioner, device)
    img_file_list = os.listdir(img_dir)
    image_id_list, generated_text_list = [], []

    for i in tqdm(range(0, len(img_file_list), batch_size)):
        batch_files = img_file_list[i:i + batch_size]

        for img_file in batch_files:
            img_path = os.path.join(img_dir, img_file)
            try:
                caption = generate_caption(captioner, model, processor, tokenizer, img_path, device)
            except Exception as e:
                caption = f"Error: {e}"

            image_id_list.append(os.path.splitext(img_file)[0])
            generated_text_list.append(caption)

        torch.cuda.empty_cache()

    df = pd.DataFrame({"image_id": image_id_list, "generated_text": generated_text_list})
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Finished, save in: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Image Caption Generator")
    parser.add_argument("--dataset_name", type=str, default="FASHIONIQ", choices=["FASHIONIQ", "circo", "CIRR"], help="Dataset name")
    parser.add_argument("--captioner", type=str, required=True, choices=["blip2", "qwen2_vl", "llava_ov", "coca"], help="Model type")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device: cuda or cpu")
    args = parser.parse_args()

    DATASET_IMAGE_PATH = {
        "FASHIONIQ": "datasets/FASHIONIQ/images",
        "circo": "datasets/circo/COCO2017_unlabeled/unlabeled2017",
        "CIRR": "datasets/CIRR/dev"
    }

    output_csv = os.path.join("../datasets", args.dataset_name, "preload/image_captions", args.output_csv)

    generate_captions(DATASET_IMAGE_PATH[args.dataset_name], output_csv, args.captioner, args.batch_size, args.device)

if __name__ == "__main__":
    main()
