import os
from .clip_encoder import CLIPVisionTower
from .egogpt_imagebind import ImageBindWrapper,EgoGPTWrapper
from .open_clip_encoder import OpenCLIPVisionTower
from .eva_vit import EvaViTWrapper
from .hf_vision import HFVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    image_processor = getattr(vision_tower_cfg, 'image_processor', getattr(vision_tower_cfg, 'image_processor', "./model_zoo/OpenAI/clip-vit-large-patch14"))
    # import pdb; pdb.set_trace()
    is_absolute_path_exists = os.path.exists(vision_tower)
    if vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif "egogpt" in vision_tower.lower():
        return EgoGPTWrapper(args=vision_tower_cfg, **kwargs)
    elif vision_tower in ["imagebind_huge"]:
        return ImageBindWrapper(vision_tower, args=vision_tower_cfg, **kwargs)
    elif "lavis" in vision_tower.lower() or "eva" in vision_tower.lower():
        # import pdb; pdb.set_trace()
        return EvaViTWrapper(vision_tower, image_processor, args=vision_tower_cfg, **kwargs)
    elif vision_tower.startswith("hf:"):
        return HFVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif vision_tower.startswith("hf-hub"):
        return OpenCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
