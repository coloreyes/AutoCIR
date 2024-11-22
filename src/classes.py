from enum import Enum, auto
import torch
import open_clip
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import AutoProcessor,Qwen2VLForConditionalGeneration,LlavaOnevisionForConditionalGeneration
from torchvision import transforms
from pathlib import Path
class Captioner(Enum):
    blip_image_captioning_base = auto()
    blip2_opt_2_7B = auto()
    blip2_opt_6_7B = auto()
    CoCa = auto()
    Qwen2_VL_7B_Instruct = auto()
    llava_onevision_qwen2_7b_ov_hf = auto()

    @staticmethod
    def from_string(s: str):
        try:
            model_names = {
                "blip2-opt-6.7B":"blip2_opt_6_7B",
                "CoCa":"CoCa",
                "Qwen2-VL-7B-Instruct":"Qwen2_VL_7B_Instruct",
                "llava-onevision-qwen2-7b-ov-hf":"llava_onevision_qwen2_7b_ov_hf"
            }
            return Captioner[model_names[s]]
        except KeyError:
            raise ValueError()
        
    def load_model_and_preprocess(self,device:torch.device,path:Path):
        if self in [Captioner.blip2_opt_6_7B]:
            model_folders = {
                Captioner.blip2_opt_6_7B:"blip2-opt-6.7B",
            }
            model_file = path.parent / "model" / "Salesforce" / model_folders[self]
            captioner_processor = Blip2Processor.from_pretrained(model_file)
            captioner_model = Blip2ForConditionalGeneration.from_pretrained(model_file).to(device)       
        elif self is Captioner.Qwen2_VL_7B_Instruct:
            model_file = path.parent / "model" / "Qwen" / "Qwen2-VL-7B-Instruct"
            captioner_processor = AutoProcessor.from_pretrained(model_file)
            captioner_model = Qwen2VLForConditionalGeneration.from_pretrained(model_file).to(device)    
        elif self is Captioner.llava_onevision_qwen2_7b_ov_hf:
            model_file = path.parent / "model" / "lmms-lab" / "llava-onevision-qwen2-7b-ov-hf"
            captioner_processor = AutoProcessor.from_pretrained(model_file)
            captioner_model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_file).to(device) 
        elif self is Captioner.CoCa:
            model_file = "../model/open_clip/"
            captioner_model, _, captioner_processor = open_clip.create_model_and_transforms(
                "coca_ViT-L-14", pretrained="laion2B-s13B-b90k",cache_dir=model_file)
        return captioner_model,captioner_processor
        
def _convert_image_to_rgb(image):
    return image.convert("RGB")
    
class Arch(Enum):
    ViT_B_32_openai = auto() # openai ViT-B/32
    ViT_B_32_openclip = auto() # openclip Vit-B-32
    ViT_L_14_openclip = auto()
    ViT_g_14_openclip = auto()
    ViT_bigG_14_openclip = auto()

    @staticmethod
    def from_string(s: str):
        try:
            model_names = {
                "ViT-B/32" : "ViT_B_32_openai",
                "ViT-B-32" : "ViT_B_32_openclip",
                "ViT-L-14" : "ViT_L_14_openclip",
                "ViT-g-14" : "ViT_g_14_openclip",
                "ViT-G-14": "ViT_bigG_14_openclip"
            }
            return Arch[model_names[s]]
        except KeyError:
            raise ValueError()

    def load_model_and_preprocess(self,device:torch.device,jit:bool=False):
        if self in [Arch.ViT_B_32_openai]:
            model_file = "../model/openai/CLIP-ViT-B-32/ViT-B-32.pt"
            clip_model = torch.jit.load(model_file).to(device)
            clip_preprocess = transforms.Compose([
                transforms.Resize((224, 224)), 
                transforms.CenterCrop(224),
                _convert_image_to_rgb,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])  # Using ViT standard arguments.
            ])
            return clip_model,clip_preprocess
        elif self in [Arch.ViT_B_32_openclip,Arch.ViT_L_14_openclip,Arch.ViT_g_14_openclip,Arch.ViT_bigG_14_openclip]:
            model_file = "../model/open_clip/"
            pretraining = {
            Arch.ViT_B_32_openclip:'laion2b_s34b_b79k',
            Arch.ViT_L_14_openclip:'laion2b_s32b_b82k',
            Arch.ViT_g_14_openclip:'laion2b_s34b_b88k',
            Arch.ViT_bigG_14_openclip:'laion2b_s39b_b160k'
            }
            clip_name = self.name.replace("_openclip", "").replace("_", "-")
            clip_model,_,clip_processor = open_clip.create_model_and_transforms(clip_name,pretraining[self],cache_dir=model_file)
            clip_model = clip_model.eval().requires_grad_(False).to(device)
            return clip_model,clip_processor

