from enum import Enum, auto
import torch
import open_clip
from transformers import CLIPProcessor, CLIPModel
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from torchvision import transforms
class Captioner(Enum):
    blip_image_captioning_base = auto()
    blip2_opt_2_7B = auto()
    blip2_opt_6_7B = auto()
    qwen_vl_7B = auto()
    coca = auto()
    llava_ov = auto()

    @staticmethod
    def from_string(s: str):
        try:
            return Captioner[s]
        except KeyError:
            raise ValueError()
        
    def load_model_and_preprocess(self, device: torch.device):
        config = CAPTIONER_REGISTRY.get(self)
        if config is None:
            raise ValueError(f"No configuration found for captioner: {self.name}")

        model_path = config["model_path"]
        processor_cls = config["processor"]
        model_cls = config["model"]

        processor = processor_cls.from_pretrained(model_path)
        model = model_cls.from_pretrained(model_path).to(device)
        return model, processor

CAPTIONER_REGISTRY = {
    Captioner.blip_image_captioning_base: {
        "model_path": "../model/blip/blip-image-captioning-base",
        "processor": BlipProcessor,
        "model": BlipForConditionalGeneration,
    },
    Captioner.blip2_opt_2_7B: {
        "model_path": "../model/blip/blip2-opt-2.7b",
        "processor": Blip2Processor,
        "model": Blip2ForConditionalGeneration,
    },
    Captioner.blip2_opt_6_7B: {
        "model_path": "../model/blip/blip2-opt-6.7b",
        "processor": Blip2Processor,
        "model": Blip2ForConditionalGeneration,
    },
    Captioner.qwen_vl_7B: {
        "model_path": "../model/qwen/qwen-vl-7b",
        "processor": QwenVLProcessor, 
        "model": QwenVLForConditionalGeneration,
    },
    Captioner.coca: {
        "model_path": "../model/coca/coca-vit-b-32",
        "processor": CocaProcessor,
        "model": CocaForConditionalGeneration,
    },
    Captioner.llava_ov: {
        "model_path": "../model/llava/llava-7b",
        "processor": LlavaProcessor,
        "model": LlavaForConditionalGeneration,
    },
}

def _convert_image_to_rgb(image):
    return image.convert("RGB")
    
class Arch(Enum):
    ViT_B_32_openai = auto()
    ViT_B_16_openai = auto()
    ViT_L_14_openai = auto()
    ViT_B_32_openclip = auto()
    ViT_L_14_openclip = auto()
    ViT_g_14_openclip = auto()
    ViT_bigG_14_openclip = auto()

    @staticmethod
    def from_string(s: str):
        try:
            model_names = {
                "ViT-B/32" : "ViT_B_32_openai",
                "ViT-B/16" : "ViT_B_16_openai",
                "ViT-L/14" : "ViT_L_14_openai",
                "ViT-B-32" : "ViT_B_32_openclip",
                "ViT-L-14" : "ViT_L_14_openclip",
                "ViT-g-14" : "ViT_g_14_openclip",
                "ViT-G-14": "ViT_bigG_14_openclip"
            }
            return Arch[model_names[s]]
        except KeyError:
            raise ValueError()

    def load_model_and_preprocess(self,device:torch.device,jit:bool=False):

        if self in [Arch.ViT_B_32_openai, Arch.ViT_B_16_openai, Arch.ViT_L_14_openai]:
            openai_model_paths = {
                Arch.ViT_B_32_openai: "../model/openai/CLIP-ViT-B-32/ViT-B-32.pt",
                Arch.ViT_B_16_openai: "../model/openai/CLIP-ViT-B-16/ViT-B-16.pt",
                Arch.ViT_L_14_openai: "../model/openai/CLIP-ViT-L-14/ViT-L-14.pt",
            }
            model_file = openai_model_paths[self]
            clip_model = torch.jit.load(model_file).to(device)
            clip_preprocess = transforms.Compose([
                transforms.Resize((224, 224)), 
                transforms.CenterCrop(224),
                _convert_image_to_rgb,
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ])
            return clip_model, clip_preprocess
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

