import torch
import torch.nn as nn
import json
from decord import VideoReader, cpu
import numpy as np
import torch.nn.functional as F
from transformers import CLIPImageProcessor

try:
    from llava.model.multimodal_encoder.egogpt_imagebind.models import imagebind_model
    from llava.model.multimodal_encoder.egogpt_imagebind.models.imagebind_model import ImageBindModel
    from llava.model.multimodal_encoder.egogpt_imagebind.models.imagebind_model import ModalityType
    from llava.model.multimodal_encoder.egogpt_imagebind.data import load_and_transform_audio_data
except ImportError:
    pass

class EgoGPTWrapper(ImageBindModel,nn.Module):
    def __init__(self, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.config=args
        # self.vision_tower_name = vision_tower
        # self.select_layer = select_layer
        # self.select_feature = select_feature
        if not delay_load:
            self.load_model()

    def load_model(self):
        self.vision_tower = imagebind_model.imagebind_huge(pretrained=True)
        for p in self.vision_tower.parameters():
            p.requires_grad = False
        self.vision_tower.eval()
        self.is_loaded = True

    def train(self, mode = True):
        self.training = mode

        if self.is_loaded:
            self.vision_tower.eval()

    @torch.no_grad()
    def forward(self, inputs):
        outputs = {}
        for modality_key, modality_value in inputs.items():
            reduce_list = (
                modality_value.ndim >= 5
            )  # Audio and Video inputs consist of multiple clips
            if reduce_list:
                B, S = modality_value.shape[:2]
                modality_value = modality_value.reshape(
                    B * S, *modality_value.shape[2:]
                )

            if modality_value is not None:
                # modality_value = self.modality_preprocessors[modality_key](
                #     **{modality_key: modality_value}
                # )
                modality_value = self.modality_preprocessors[modality_key](modality_value)
                trunk_inputs = modality_value["trunk"]
                head_inputs = modality_value["head"]
                modality_value = self.modality_trunks[modality_key](**trunk_inputs)
                modality_value = self.modality_heads[modality_key](
                    modality_value, **head_inputs
                )
                modality_value = self.modality_postprocessors[modality_key](
                    modality_value
                )

                if reduce_list:
                    modality_value = modality_value.reshape(B, S, -1)
                    modality_value = modality_value.mean(dim=1)

                # outputs[modality_key] = modality_value
                if self.config.torch_dtype==torch.bfloat16: #convert input feature to bfloat16
                    outputs[modality_key] = modality_value.to(torch.bfloat16)

        return outputs

    @property
    def dummy_feature(self):
        return torch.zeros(1, 1024, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.modality_preprocessors.vision.cls_token.dtype

    @property
    def device(self):
        return self.vision_tower.modality_preprocessors.vision.cls_token.device

    @property
    def hidden_size(self):
        return 1024

class ImageBindWrapper(nn.Module):
    def __init__(self, vision_tower, select_layer, select_feature='patch', delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = select_layer
        self.select_feature = select_feature

        if not delay_load:
            self.load_model()

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.vision_tower = imagebind_model.imagebind_huge(pretrained=True)
        for p in self.vision_tower.parameters():
            p.requires_grad = False
        self.vision_tower.eval()
        self.is_loaded = True

    def train(self, mode = True):
        self.training = mode

        if self.is_loaded:
            self.vision_tower.eval()

    @torch.no_grad()
    def forward(self, x):
        if type(x) == dict:
            if x['audios'] is not None:
                inputs = {ModalityType.AUDIO: load_and_transform_audio_data(x['audios'], device=self.device).half()}
                embeddings = self.vision_tower(inputs)
                audio_embedding = embeddings[ModalityType.AUDIO]
                return audio_embedding.unsqueeze(1)
        else:
            inputs = {ModalityType.VISION: x.to(dtype=self.dtype)}
            embeddings = self.vision_tower(inputs)
            vision_embedding = embeddings[ModalityType.VISION]
            if vision_embedding.ndim == 2:
                return vision_embedding.unsqueeze(1)
            if vision_embedding.shape[1] == 257:
                return vision_embedding[:, 1:]
            raise ValueError(f'Unexpected shape: {vision_embedding.shape}')

    @property
    def dummy_feature(self):
        return torch.zeros(1, 1024, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.modality_preprocessors.vision.cls_token.dtype

    @property
    def device(self):
        return self.vision_tower.modality_preprocessors.vision.cls_token.device

    @property
    def hidden_size(self):
        return 1024
