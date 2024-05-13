# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import ast

import torch
import time
import random

import transformers
import tokenizers

from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.constants import *
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer
from llava.train.EgoGPT_trainer import EgoGPT_trainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import process_highres_image, process_anyres_image, process_highres_image_crop_split, tokenizer_image_token

from PIL import Image
from decord import VideoReader, cpu

import cv2

from packaging import version

import numpy as np

from transformers import AutoConfig

import math

from llava.model.multimodal_encoder.egogpt_imagebind.models.imagebind_model import ModalityType
from llava.model.multimodal_encoder.egogpt_imagebind import data as egogpt_processor

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')



@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_mm_vision_resampler: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    image_processor: Optional[str] = field(default=None)
    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_resampler_type: Optional[str] = field(default=None)
    mm_mask_drop_mode: str = field(default="fixed")
    mm_mask_drop_skip_percentage: float = field(default=0.)
    mm_mask_drop_ratio: float = field(default=0.25)
    mm_mask_drop_ratio_upper: Optional[float] = field(default=None)
    mm_mask_drop_ratio_lower: Optional[float] = field(default=None)
    mm_spatial_pool_stride: Optional[int] = field(default=None)
    mm_spatial_pool_mode: str = field(default="average")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None)
    patchify_video_feature: bool = field(default=False)
    mm_perceiver_depth: Optional[int] = field(default=3)
    mm_perceiver_latents: Optional[int] = field(default=32)
    mm_perceiver_ff_mult: Optional[float] = field(default=4)
    mm_perceiver_pretrained: Optional[str] = field(default=None)
    mm_qformer_depth: Optional[int] = field(default=3)
    mm_qformer_latents: Optional[int] = field(default=32)
    mm_qformer_pretrained: Optional[str] = field(default=None)
    mm_vlmattention_pretrained: Optional[str] = field(default=None)
    mm_vlmattention_bert_type: Optional[str] = field(default="qformer_pretrain")
    mm_vlmattention_num_query: Optional[int] = field(default=32)
    mm_vlmattention_compress_type: Optional[str] = field(default=None)

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    video_folder: Optional[str] = field(default=None)
    video_fps: Optional[int] = field(default=1)
    frames_upbound: Optional[int] = field(default=0)
    video_token: Optional[int] = field(default=2)
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    image_crop_resolution: int = 224
    image_split_resolution: int = 224
    input_prompt: Optional[str] = field(default=None)
    refine_prompt: Optional[bool] = field(default=False)



@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    freeze_mm_vision_resampler: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    mm_vision_tower_lr: Optional[float] = None
    group_by_varlen: bool = field(default=False)
    group_by_modality_length: bool = field(default=False)
    group_by_modality_length_auto: bool = field(default=False)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector', 'vision_resampler']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value'] and not sentence['value'].startswith(DEFAULT_IMAGE_TOKEN):
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources

def preprocess_multimodal_movie(
    sources: Sequence[str],
    data_args: DataArguments,
    video_inputs: str
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                prompt = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
            replace_token = video_inputs
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources, prompt


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_imgsp_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    img_token: str = '<image>',
    refine_prompt: bool = False,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    guided_prompt = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        img_in_text = False
        # import pdb; pdb.set_trace()
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            
            # add guided prompt
            if role==conv.roles[0]:
                guided_sent = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, '').replace('\n', '')
                if refine_prompt:
                    # only keep the useful part of the prompt
                    if '\n' in guided_sent:
                        for _sent in guided_sent.split('\n'):
                            if '?' in _sent:
                                guided_sent = _sent
                                break
                guided_prompt.append(guided_sent)
            # check if image token in text
            if img_token in sentence["value"]:
                img_in_text = True
            # add image token to all sentence if multimoal input
            if role==conv.roles[0] and img_in_text and img_token not in sentence["value"]:
                # randomly add image token to the beginning or end of the sentence
                if random.randint(0,1)==0:
                    img_conv = img_token + '\n' + sentence["value"]
                else:
                    img_conv = sentence["value"] + '\n' + img_token
                
                conv.append_message(role, img_conv)
            else:
                conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    # import pdb; pdb.set_trace()
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    # import pdb; pdb.set_trace()
    return dict(
        input_ids=input_ids,
        labels=targets,
        prompt=guided_prompt,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f"(#turns={len(re_rounds)} ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess_plain_guided(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    prompt: str = None,
) -> Dict:
    # add end signal and concatenate together
    guided_prompt = []
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        guided_prompt.append(source[0]['value'].replace(DEFAULT_IMAGE_TOKEN, '').replace('\n', ''))
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    prompt: str = None,
    refine_prompt: bool = False,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.version.startswith("plain_guided"):
        return preprocess_plain_guided(sources, tokenizer, prompt=prompt)
    elif conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    elif conversation_lib.default_conversation.version.startswith("imgsp"):
        return preprocess_imgsp_v1(sources, tokenizer, has_image=has_image, refine_prompt=refine_prompt)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            # cur_len = cur_len if 'image' in sample else -cur_len
            cur_len = cur_len if ('image' in sample) or ('video' in sample) else -cur_len
            length_list.append(cur_len)
        return length_list

    def process_image(self, image_file):
        image_folder = self.data_args.image_folder
        processor = self.data_args.image_processor
        if not isinstance(image_file, np.ndarray):
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
        else:
            image = Image.fromarray(image_file).convert('RGB')
        image_size = image.size
        if self.data_args.image_aspect_ratio == 'highres':
            image = process_highres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
        elif self.data_args.image_aspect_ratio == 'anyres':
            image = process_anyres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
        elif self.data_args.image_aspect_ratio == 'crop_split':
            image = process_highres_image_crop_split(image, self.data_args)
        elif self.data_args.image_aspect_ratio == 'pad':
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result
            image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        else:
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        return image, image_size,"image"

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # TODO: define number of retries somewhere else
        num_base_retries = 3
        num_final_retries = 300

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f'[try #{attempt_idx}] Failed to fetch sample {i}. Exception:', e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                sample_idx = random.choice(range(len(self)))
                sample = self._get_item(sample_idx)
                return sample
            except Exception as e:
                # no need to sleep
                print(f'[try other #{attempt_idx}] Failed to fetch sample {sample_idx}. Exception:', e)
                pass

        # still fail, most likely to be path issue or cloud disk issue, retry the same sample for longer
        for attempt_idx in range(num_final_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f'[final try #{attempt_idx}] Failed to fetch sample {i}. Exception:', e)
                time.sleep(1)

        # Finally raise exception on failing.
        assert False, "Failed to fetch sample."

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        suffix = None
        # import pdb;pdb.set_trace()
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            if type(image_file) is list:
                image = [self.process_image(f) for f in image_file]
            else:
                image = [self.process_image(image_file)]
            # import pdb;pdb.set_trace()
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        elif 'video' in sources[0]:
            # import pdb;pdb.set_trace()
            video_file = self.list_data_dict[i]['video']
            video_folder = self.data_args.video_folder
            video_file = os.path.join(video_folder, video_file)
            suffix = video_file.split('.')[-1]
            if not os.path.exists(video_file):
                print('File {} not exist!'.format(video_file))
            
            if suffix == 'pkl':
                video_info = pickle.load(open(video_file, 'rb'))
                image = torch.from_numpy(video_info['feats'][:, 1:])
                input_prompt = video_info['inputs'].replace('...', '')
                # replace the default image token with multiple tokens
                input_prompt = input_prompt.replace(DEFAULT_IMAGE_TOKEN, 
                                                    DEFAULT_IMAGE_TOKEN * self.data_args.video_token)
                sources, query_prompt = preprocess_multimodal_movie(
                    copy.deepcopy([e["conversations"] for e in sources]),
                    self.data_args, input_prompt)
            else:
                # import pdb;pdb.set_trace()
                try:
                    # using videoreader
                    if "liangkeg" not in video_file:
                        vr = VideoReader(video_file, ctx=cpu(0))
                        total_frame_num = len(vr)
                        avg_fps = round(vr.get_avg_fps()/self.data_args.video_fps)
                        # frame_idx = [i for i in range(0, len(vr), avg_fps)]
                        # sample_fps = round(vr.get_avg_fps() / self.data_args.video_fps)
                        # frame_idx = [i for i in range(0, len(vr), sample_fps)]

                        sample_fps = avg_fps
                        if self.data_args.frames_upbound > 0:
                            # import pdb;pdb.set_trace()
                            sample_fps = self.data_args.frames_upbound if total_frame_num > self.data_args.frames_upbound else total_frame_num
                            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
                            frame_idx = uniform_sampled_frames.tolist()
                        else:
                            # sample 1 frames/second
                            frame_idx = [i for i in range(0, total_frame_num, sample_fps)]
                        # video: (F, H, W, C)
                        video = vr.get_batch(frame_idx).asnumpy()
                        # Convert the list of frames to a numpy array if needed
                        video = np.array(video)

                        # using cv2
                        # cap = cv2.VideoCapture(video_file)

                        # total_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        # fps = cap.get(cv2.CAP_PROP_FPS)
                        # sample_fps = round(fps / self.data_args.video_fps)

                        # if self.data_args.frames_upbound > 0:
                        #     sample_fps = self.data_args.frames_upbound if total_frame_num > self.data_args.frames_upbound else total_frame_num
                        #     uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
                        #     frame_idx = uniform_sampled_frames.tolist()
                        # else:
                        #     # sample 1 frame/second
                        #     frame_idx = [i for i in range(0, total_frame_num, sample_fps)]

                        # video_frames = []
                        # for idx in frame_idx:
                        #     cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        #     ret, frame = cap.read()
                        #     if ret:
                        #         video_frames.append(frame)
                        #     else:
                        #         print(f"Failed to read frame at index {idx}: {video_file}")

                        # cap.release()
                    else:
                        frame_files = [os.path.join(video_file, f) for f in os.listdir(video_file) if os.path.isfile(os.path.join(video_file, f))]
                        frame_files.sort()  # Ensure the frames are sorted if they are named sequentially

                        # Determine the indices for uniformly sampling 10 frames
                        num_frames_to_sample = self.data_args.frames_upbound

                        total_frames = len(frame_files)

                        sampled_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)

                        # Read and store the sampled frames
                        video = []
                        for idx in sampled_indices:
                            frame_path = frame_files[idx]
                            try:
                                with Image.open(frame_path) as img:
                                    # Convert the PIL image to a numpy array if needed
                                    # frame = np.array(img.convert('RGB'))
                                    frame = img.convert('RGB')
                                    video.append(frame)
                            except IOError:
                                print(f"Failed to read frame at path: {frame_path}")

                    processor = self.data_args.image_processor
                    image = processor.preprocess(video, return_tensors='pt')['pixel_values']
                    # import pdb;pdb.set_trace()
                    # image_tensors = []
                    # for f in video:
                    #     cur_image,cur_size = self.process_image(f)
                    #     image_tensors.append(cur_image)
                    # image_tensors = torch.stack(image_tensors)  
                    image = [(image, video[0].size,"video")]
                    sources = preprocess_multimodal(
                        copy.deepcopy([e["conversations"] for e in sources]),
                        self.data_args)
                except:
                    import pdb;pdb.set_trace()
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        has_image = ('image' in self.list_data_dict[i]) or ('video' in self.list_data_dict[i])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=has_image,
            prompt=self.data_args.input_prompt,
            refine_prompt=self.data_args.refine_prompt)

        if 'prompt' in data_dict:
            prompt = data_dict['prompt']
        else:
            prompt = None
        
        if suffix == 'pkl':
            prompt = [query_prompt]

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # import pdb;pdb.set_trace()
        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif 'video' in self.list_data_dict[i]:
            # import pdb;pdb.set_trace()
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = [
                (
                    torch.zeros(1, 3, crop_size['height'], crop_size['width']),
                    (crop_size['width'], crop_size['height']),
                    "text"
                ),
            ]
        # prompt exist in the data
        # import pdb;pdb.set_trace()
        if prompt is not None:
            data_dict['prompt'] = prompt

        return data_dict
class EgoGPTDataset(LazySupervisedDataset):
    """Dataset for supervised fine-tuning."""
    
    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        

        super(LazySupervisedDataset, self).__init__()
        EgoGPTdata = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.EgoGPTdata = EgoGPTdata
        self.data_args = data_args
        # list_data_dict = json.load(open(data_path, "r"))
        # self._get_item(2)
        # rank0_print("Formatting inputs...Skip in lazy mode")
        # self.tokenizer = tokenizer
        # self.EgoGPTdata = list_data_dict
        # self.data_args = data_args

    def __len__(self):
        return len(self.EgoGPTdata['video_clip'])-1
        #(clip0,clip1) ... (clip13,clip14)

    # @property
    # def lengths(self):
    #     length_list = []
    #     for sample in self.EgoGPTdata:
    #         img_tokens = 128 if 'image' in sample else 0
    #         length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
    #     return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for key in self.EgoGPTdata.keys():
            cur_len=len(self.EgoGPTdata[key]['instruction'])+len(self.EgoGPTdata[key]['answer'])
            length_list.append(cur_len)
        return length_list

    def preprocess_imu(self,imu_paths,sample_mode='interpolate'):
        import torch.nn.functional as F
        def _downsample(imu,sample_mode):
            if sample_mode=="interpolate":
                imu=F.interpolate(imu.unsqueeze(0).unsqueeze(0),(2000,6),mode="bilinear").squeeze(0).squeeze(0)
            else:
                indices = sorted(np.random.choice(imu.shape[0], size=2000, replace=False))
                imu = imu[indices,: ]
            return imu

        imu=[]
        totensor=lambda x:torch.tensor(x)
        for imu_path in imu_paths:
            imu_raw_data=totensor(np.load(imu_path)[:,1:])
            imu_data=_downsample(imu_raw_data,sample_mode)
            imu.append(imu_data)
        imu=torch.stack(imu, dim=0)
        imu=imu.permute(0,2,1)
        imu=imu.to(torch.float32) #Convert to float32

        return imu

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # TODO: define number of retries somewhere else
        num_base_retries = 3
        num_final_retries = 300

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f'[try #{attempt_idx}] Failed to fetch sample {i}. Exception:', e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                sample_idx = random.choice(range(len(self)))
                sample = self._get_item(sample_idx)
                return sample
            except Exception as e:
                # no need to sleep
                print(f'[try other #{attempt_idx}] Failed to fetch sample {sample_idx}. Exception:', e)
                pass

        # still fail, most likely to be path issue or cloud disk issue, retry the same sample for longer
        for attempt_idx in range(num_final_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f'[final try #{attempt_idx}] Failed to fetch sample {i}. Exception:', e)
                time.sleep(1)

        # Finally raise exception on failing.
        assert False, "Failed to fetch sample."

    def preprocess_multimodal_egogptdata(self):
        mm_data=""
        mm_data+=DEFAULT_IMAGE_TOKEN+"\n"
        mm_data+=DEFAULT_AUDIO_TOKEN+"\n"
        mm_data+=DEFAULT_IMU_LEFT_TOKEN+"\n"
        mm_data+=DEFAULT_AUDIO_TOKEN+"\n"
        return mm_data

    def preprocess_egogptdata(self,
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    prompt: str = None,
    refine_prompt: bool = False,
) -> Dict:
        # conv = conversation_lib.default_conversation.copy()
        # conv.system=''  #clear system message to follow otter-EAI
        # roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        # conversations = []
        # conv.append_message(conv.roles[0], sources["instruction"])

        # conv.append_message(conv.roles[1], sources["answer"])

        # conversations.append(conv.get_prompt())

        input_ids=[1]
        # Tokenize conversations
        for token in sources.split("\n"):
            if token=="":
                continue
            if token==DEFAULT_IMAGE_TOKEN:
                input_ids.append(IMAGE_TOKEN_INDEX)
            if token==DEFAULT_IMU_LEFT_TOKEN:
                input_ids.append(IMU_LEFT_TOKEN_INDEX)
            if token==DEFAULT_IMU_RIGHT_TOKEN:
                input_ids.append(IMU_RIGHT_TOKEN_INDEX)
            if token==DEFAULT_AUDIO_TOKEN:
                input_ids.append(AUDIO_TOKEN_INDEX)                                                
            input_ids.append(13) #add "\n"

        targets = input_ids.clone()


        # Mask targets
        sep = conv.sep + conv.roles[1] + ": "
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(rounds):
                if rou == "":
                    break

                parts = rou.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep

                if has_image:
                    round_len = len(tokenizer_image_token(rou, tokenizer))
                    instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
                else:
                    round_len = len(tokenizer(rou).input_ids)
                    instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                    round_len -= 1
                    instruction_len -= 1

                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

                cur_len += round_len
            target[cur_len:] = IGNORE_INDEX

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )

        return dict(
            input_ids=input_ids,
            labels=targets,
        )

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        data=self.EgoGPTdata['video_clip']
        target_clip=f"clip{i}"
        label_clip=f"clip{i+1}"

        # device = "cuda" if torch.cuda.is_available() else "cpu"
        video_path=[data[target_clip]["video"],data[label_clip]["video"]]
        audio_path=[data[target_clip]["audio"],data[label_clip]["audio"]]
        imu_left_path=[data[target_clip]["imu_left"],data[label_clip]["imu_left"]]
        imu_right_path=[data[target_clip]["imu_right"],data[label_clip]["imu_right"]]

        video_feat=egogpt_processor.load_and_transform_video_data(video_path)
        audio_feat=egogpt_processor.load_and_transform_aria_audio_data(audio_path)
        imu_left_feat=self.preprocess_imu(imu_left_path,sample_mode='interpolate')
        imu_right_feat=self.preprocess_imu(imu_right_path,sample_mode='interpolate')

        data_dict = dict(
            video_input=video_feat[0],
            video_label=video_feat[1],
            audio_input=audio_feat[0],
            audio_label=audio_feat[1],
            imu_left_input=imu_left_feat[0],
            imu_left_label=imu_left_feat[1],
            imu_right_input=imu_right_feat[0],
            imu_right_label=imu_right_feat[1]
        ) 
        #TODO add text token
        # if 'prompt' in data_dict:
        #     prompt = data_dict['prompt']
        # else:
        #     prompt = None
        

        # if isinstance(i, int):
        #     data_dict = dict(input_ids=data_dict["input_ids"][0],
        #                      labels=data_dict["labels"][0])

        #     data_dict['image'] = image #add image sequence

        # if prompt is not None:
        #     data_dict['prompt'] = prompt

        return data_dict    
class MCDataset(LazySupervisedDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(MCDataset, self).__init__(data_path,tokenizer,data_args)
        # list_data_dict = json.load(open(data_path, "r"))

        # rank0_print("Formatting inputs...Skip in lazy mode")
        # self.tokenizer = tokenizer
        # self.EgoGPTdata = list_data_dict
        # self.data_args = data_args

    # def __len__(self):
    #     return len(self.EgoGPTdata)

    # @property
    # def lengths(self):
    #     length_list = []
    #     for sample in self.EgoGPTdata:
    #         img_tokens = 128 if 'image' in sample else 0
    #         length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
    #     return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for key in self.list_data_dict.keys():
            cur_len=len(self.list_data_dict[key]['instruction'])+len(self.list_data_dict[key]['answer'])
            length_list.append(cur_len)
        return length_list

    # def process_image(self, image_file):
    #     image_folder = self.data_args.image_folder
    #     processor = self.data_args.image_processor
    #     if not isinstance(image_file, np.ndarray):
    #         image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
    #     else:
    #         image = Image.fromarray(image_file).convert('RGB')
    #     image_size = image.size
    #     if self.data_args.image_aspect_ratio == 'highres':
    #         image = process_highres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
    #     elif self.data_args.image_aspect_ratio == 'anyres':
    #         image = process_anyres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
    #     elif self.data_args.image_aspect_ratio == 'crop_split':
    #         image = process_highres_image_crop_split(image, self.data_args)
    #     elif self.data_args.image_aspect_ratio == 'pad':
    #         def expand2square(pil_img, background_color):
    #             width, height = pil_img.size
    #             if width == height:
    #                 return pil_img
    #             elif width > height:
    #                 result = Image.new(pil_img.mode, (width, width), background_color)
    #                 result.paste(pil_img, (0, (width - height) // 2))
    #                 return result
    #             else:
    #                 result = Image.new(pil_img.mode, (height, height), background_color)
    #                 result.paste(pil_img, ((height - width) // 2, 0))
    #                 return result
    #         image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
    #         image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    #     else:
    #         image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    #     return image, image_size,"image"

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # TODO: define number of retries somewhere else
        num_base_retries = 3
        num_final_retries = 300

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f'[try #{attempt_idx}] Failed to fetch sample {i}. Exception:', e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                sample_idx = random.choice(range(len(self)))
                sample = self._get_item(sample_idx)
                return sample
            except Exception as e:
                # no need to sleep
                print(f'[try other #{attempt_idx}] Failed to fetch sample {sample_idx}. Exception:', e)
                pass

        # still fail, most likely to be path issue or cloud disk issue, retry the same sample for longer
        for attempt_idx in range(num_final_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f'[final try #{attempt_idx}] Failed to fetch sample {i}. Exception:', e)
                time.sleep(1)

        # Finally raise exception on failing.
        assert False, "Failed to fetch sample."

    def preprocess_multimodal_mcdata(self,data):
        data["instruction"]=DEFAULT_IMAGE_TOKEN+"\n"+data["instruction"]
        return data

    def preprocess_mcdata(self,
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    prompt: str = None,
    refine_prompt: bool = False,
) -> Dict:
        conv = conversation_lib.default_conversation.copy()
        conv.system=''  #clear system message to follow otter-EAI
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        conversations = []
        conv.append_message(conv.roles[0], sources["instruction"])

        conv.append_message(conv.roles[1], sources["answer"])

        conversations.append(conv.get_prompt())

        # Tokenize conversations

        if has_image:
            input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
        else:
            input_ids = tokenizer(
                conversations,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            ).input_ids

        targets = input_ids.clone()

        assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

        # Mask targets
        sep = conv.sep + conv.roles[1] + ": "
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(rounds):
                if rou == "":
                    break

                parts = rou.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep

                if has_image:
                    round_len = len(tokenizer_image_token(rou, tokenizer))
                    instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
                else:
                    round_len = len(tokenizer(rou).input_ids)
                    instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                    round_len -= 1
                    instruction_len -= 1

                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

                cur_len += round_len
            target[cur_len:] = IGNORE_INDEX

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )

        return dict(
            input_ids=input_ids,
            labels=targets,
        )

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        key_candidate=list(self.list_data_dict.keys())

        key=key_candidate[i]

        image_seq_path = self.list_data_dict[key]['image_ids']

        image_seq=[]
        for img in image_seq_path:
            data=np.array(Image.open(img).convert("RGB"))

            image_seq.append(data)
        video=np.stack(image_seq)

        processor = self.data_args.image_processor
        image = processor.preprocess(video, return_tensors='pt')['pixel_values']

        # image_tensors = torch.stack(image_tensors)  
        image = [(image, video[0].size,"video")]
        sources = self.preprocess_multimodal_mcdata(self.list_data_dict[key])

        # data_dict = preprocess(
        #     sources,
        #     self.tokenizer,
        #     has_image=True,
        #     prompt=self.data_args.input_prompt,
        #     refine_prompt=self.data_args.refine_prompt)
        data_dict = self.preprocess_mcdata(
            sources,
            self.tokenizer,
            has_image=True,
            prompt=self.data_args.input_prompt,
            refine_prompt=self.data_args.refine_prompt)        

        if 'prompt' in data_dict:
            prompt = data_dict['prompt']
        else:
            prompt = None
        

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

            data_dict['image'] = image #add image sequence

        if prompt is not None:
            data_dict['prompt'] = prompt

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids] 
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=batch_first,
            padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        # import pdb;pdb.set_trace()
        input_ids = [_input_ids[:self.tokenizer.model_max_length] for _input_ids in input_ids]
        labels = [_labels[:self.tokenizer.model_max_length] for _labels in labels]
        input_ids = self.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = self.pad_sequence(labels,
                                   batch_first=True,
                                   padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            # instances[1]['image'][0][0].shape
            # torch.Size([5, 3, 224, 224])
            images = [instance['image'] for instance in instances]
        
            batch['image_sizes'] = [im[1] for im_list in images for im in im_list]
            batch['modalities'] = [im[2] for im_list in images for im in im_list]
            images = [im[0] for im_list in images for im in im_list]
            # import pdb;pdb.set_trace()
            

            if all(x is not None and x.shape == images[0].shape for x in images):
                # Image: (N, P, C, H, W)
                # Video: (N, F, C, H, W)
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        # import pdb;pdb.set_trace()
        if 'prompt' in instances[0]:
            batch['prompts'] = [instance['prompt'] for instance in instances]

        return batch

@dataclass
class DataCollatorForEgoGPT(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids] 
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=batch_first,
            padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        video_input, video_label, audio_input,\
        audio_label, imu_left_input, imu_left_label, \
        imu_right_input, imu_right_label = tuple([instance[key] for instance in instances]
                                  for key in ('video_input', 'video_label', \
                                              'audio_input', 'audio_label', \
                                                'imu_left_input', 'imu_left_label',\
                                                     'imu_right_input', 'imu_right_label'))
        batch=dict(
            video_input=video_input,
            video_label=video_label,
            audio_input=audio_input,
            audio_label=audio_label,
            imu_left_input=imu_left_input,
            imu_left_label=imu_left_label,
            imu_right_input=imu_right_input,
            imu_right_label=imu_right_label
        )
        # # import pdb;pdb.set_trace()
        # input_ids = [_input_ids[:self.tokenizer.model_max_length] for _input_ids in input_ids]
        # labels = [_labels[:self.tokenizer.model_max_length] for _labels in labels]
        # input_ids = self.pad_sequence(
        #     input_ids,
        #     batch_first=True,
        #     padding_value=self.tokenizer.pad_token_id)
        # labels = self.pad_sequence(labels,
        #                            batch_first=True,
        #                            padding_value=IGNORE_INDEX)
        # batch = dict(
        #     input_ids=input_ids,
        #     labels=labels,
        #     attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        # )

        # if 'image' in instances[0]:
        #     # instances[1]['image'][0][0].shape
        #     # torch.Size([5, 3, 224, 224])
        #     images = [instance['image'] for instance in instances]
        
        #     batch['image_sizes'] = [im[1] for im_list in images for im in im_list]
        #     batch['modalities'] = [im[2] for im_list in images for im in im_list]
        #     images = [im[0] for im_list in images for im in im_list]
        #     # import pdb;pdb.set_trace()
            

        #     if all(x is not None and x.shape == images[0].shape for x in images):
        #         # Image: (N, P, C, H, W)
        #         # Video: (N, F, C, H, W)
        #         batch['images'] = torch.stack(images)
        #     else:
        #         batch['images'] = images

        # # import pdb;pdb.set_trace()
        # if 'prompt' in instances[0]:
        #     batch['prompts'] = [instance['prompt'] for instance in instances]

        return batch

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = MCDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def make_EgoGPT_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = EgoGPTDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForEgoGPT(tokenizer=tokenizer)
    # data_collator(instances=[train_dataset._get_item(1)])
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            model = LlavaMptForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args 
            )
        elif 'mixtral' in model_args.model_name_or_path.lower():
            model = LlavaMixtralForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation="flash_attention_2",
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
        elif 'mistral' in model_args.model_name_or_path.lower() or 'zephyr' in model_args.model_name_or_path.lower():
            model = LlavaMistralForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation="flash_attention_2",
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
        else:
            # finetune from a image trained model
            if not model_args.pretrain_mm_mlp_adapter:
                cfg_pretrained = AutoConfig.from_pretrained(model_args.model_name_or_path)
                overwrite_config = {}
                overwrite_config["mm_resampler_type"] = model_args.mm_resampler_type
                overwrite_config["mm_spatial_pool_stride"] = model_args.mm_spatial_pool_stride
                overwrite_config["mm_spatial_pool_out_channels"] = model_args.mm_spatial_pool_out_channels
                overwrite_config["mm_spatial_pool_mode"] = model_args.mm_spatial_pool_mode

                if "224" in model_args.vision_tower:
                    # suppose the length of text tokens is around 1000, from bo's report
                    least_token_number = data_args.frames_upbound*(16//model_args.mm_spatial_pool_stride)**2 + 1000
                else:
                    least_token_number = data_args.frames_upbound*(24//model_args.mm_spatial_pool_stride)**2 + 1000
                
                scaling_factor = math.ceil(least_token_number/4096)
                if scaling_factor >= 2:
                    overwrite_config["rope_scaling"] = {"factor": float(scaling_factor), "type": "linear"}
                    overwrite_config["max_sequence_length"] = 4096 * scaling_factor
                    overwrite_config["tokenizer_model_max_length"] = 4096 * scaling_factor
                    training_args.model_max_length = 4096 * scaling_factor

                print(f"Overwriting config with {overwrite_config}")
                for k, v in overwrite_config.items():
                    setattr(cfg_pretrained, k, v)
                if "egogpt" in model_args.vision_tower.lower(): #(Choiszt) Set egogpt vision tower
                    setattr(cfg_pretrained,"mm_vision_tower",model_args.vision_tower)
            # import pdb;pdb.set_trace()
            if transformers.__version__ == "4.31.0":
                model = LlavaLlamaForCausalLM.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir, config=cfg_pretrained, torch_dtype=(torch.bfloat16 if training_args.bf16 else None), **bnb_model_from_pretrained_args)
                print("Using LlavaLlamaForCausalLM from transformers==4.31.0")
            else:
                if model_args.pretrain_mm_mlp_adapter:
                    ## For the stage2 ft, we should not load the config of a pure vicuna model
                    model = LlavaLlamaForCausalLM.from_pretrained(
                        model_args.model_name_or_path, cache_dir=training_args.cache_dir, attn_implementation="flash_attention_2", torch_dtype=(torch.bfloat16 if training_args.bf16 else None), **bnb_model_from_pretrained_args
                    )
                elif "egogpt" in model_args.vision_tower.lower():
                    model = EgoGPTLlama.from_pretrained(
                        model_args.model_name_or_path, cache_dir=training_args.cache_dir, attn_implementation="flash_attention_2", config=cfg_pretrained, torch_dtype=(torch.bfloat16 if training_args.bf16 else None), **bnb_model_from_pretrained_args
                    )
                    # sample=torch.load("1.pt")
                    # model(**sample)
                else: #EgoGPT go with this loop
                    model = LlavaLlamaForCausalLM.from_pretrained(
                        model_args.model_name_or_path, cache_dir=training_args.cache_dir, attn_implementation="flash_attention_2", config=cfg_pretrained, torch_dtype=(torch.bfloat16 if training_args.bf16 else None), **bnb_model_from_pretrained_args
                    )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation="flash_attention_2",
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    elif 'mistral' in model_args.model_name_or_path.lower() or 'mixtral' in model_args.model_name_or_path.lower() or 'zephyr' in model_args.model_name_or_path.lower():
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="left"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if "egogpt" in model_args.vision_tower.lower():
        if model_args.vision_tower is not None:
            model.get_model().initialize_vision_modules(
                model_args=model_args,
                fsdp=training_args.fsdp
            )
            
            vision_tower = model.get_vision_tower()
            vision_tower.load_model()
            print("load imagebind model...")
            vision_tower.vision_tower.to(torch.float32) #Pretrained Imagebind only suit for float32

            # data_args.image_processor = vision_tower.image_processor
            data_args.is_multimodal = True

            model.config.image_aspect_ratio = data_args.image_aspect_ratio
            if data_args.image_grid_pinpoints is not None:
                data_args.image_grid_pinpoints = ast.literal_eval(data_args.image_grid_pinpoints)
            model.config.image_grid_pinpoints = data_args.image_grid_pinpoints
            model.config.image_crop_resolution = data_args.image_crop_resolution
            model.config.image_split_resolution = data_args.image_split_resolution
            model.config.tokenizer_padding_side = tokenizer.padding_side
            model.config.tokenizer_model_max_length = tokenizer.model_max_length

            model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
            model.config.tune_mm_vision_resampler = training_args.tune_mm_vision_resampler = model_args.tune_mm_vision_resampler
            if model_args.tune_mm_mlp_adapter or model_args.tune_mm_vision_resampler:
                model.requires_grad_(False)
            if model_args.tune_mm_mlp_adapter:
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = True
            if model_args.tune_mm_vision_resampler:
                for p in model.get_model().vision_resampler.parameters():
                    p.requires_grad = True

            model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
            if training_args.freeze_mm_mlp_adapter:
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = False

            model.config.freeze_mm_vision_resampler = training_args.freeze_mm_vision_resampler
            if training_args.freeze_mm_vision_resampler:
                for p in model.get_model().vision_resampler.parameters():
                    p.requires_grad = False

            model.config.unfreeze_mm_vision_tower = model_args.unfreeze_mm_vision_tower
            if model_args.unfreeze_mm_vision_tower:
                vision_tower.requires_grad_(True)

            if training_args.bits in [4, 8]:
                model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

            model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
            model.config.mm_projector_lr = training_args.mm_projector_lr
            model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr
            training_args.use_im_start_end = model_args.mm_use_im_start_end
            model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
            model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    else:
        if model_args.vision_tower is not None:
            model.get_model().initialize_vision_modules(
                model_args=model_args,
                fsdp=training_args.fsdp
            )
            
            vision_tower = model.get_vision_tower()
            vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

            data_args.image_processor = vision_tower.image_processor
            data_args.is_multimodal = True

            model.config.image_aspect_ratio = data_args.image_aspect_ratio
            if data_args.image_grid_pinpoints is not None:
                data_args.image_grid_pinpoints = ast.literal_eval(data_args.image_grid_pinpoints)
            model.config.image_grid_pinpoints = data_args.image_grid_pinpoints
            model.config.image_crop_resolution = data_args.image_crop_resolution
            model.config.image_split_resolution = data_args.image_split_resolution
            model.config.tokenizer_padding_side = tokenizer.padding_side
            model.config.tokenizer_model_max_length = tokenizer.model_max_length

            model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
            model.config.tune_mm_vision_resampler = training_args.tune_mm_vision_resampler = model_args.tune_mm_vision_resampler
            if model_args.tune_mm_mlp_adapter or model_args.tune_mm_vision_resampler:
                model.requires_grad_(False)
            if model_args.tune_mm_mlp_adapter:
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = True
            if model_args.tune_mm_vision_resampler:
                for p in model.get_model().vision_resampler.parameters():
                    p.requires_grad = True

            model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
            if training_args.freeze_mm_mlp_adapter:
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = False

            model.config.freeze_mm_vision_resampler = training_args.freeze_mm_vision_resampler
            if training_args.freeze_mm_vision_resampler:
                for p in model.get_model().vision_resampler.parameters():
                    p.requires_grad = False

            model.config.unfreeze_mm_vision_tower = model_args.unfreeze_mm_vision_tower
            if model_args.unfreeze_mm_vision_tower:
                vision_tower.requires_grad_(True)

            if training_args.bits in [4, 8]:
                model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

            model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
            model.config.mm_projector_lr = training_args.mm_projector_lr
            model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr
            training_args.use_im_start_end = model_args.mm_use_im_start_end
            model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
            model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    if "egogpt" in model_args.vision_tower.lower():

        data_module = make_EgoGPT_data_module(tokenizer=tokenizer,
                                                data_args=data_args)        

    else:
        data_module = make_supervised_data_module(tokenizer=tokenizer,
                                                data_args=data_args)
    trainer = EgoGPT_trainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
