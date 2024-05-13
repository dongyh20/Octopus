#    Copyright 2023 Haotian Liu
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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.models.llama.modeling_llama import *
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

from llava.model.multimodal_encoder.egogpt_imagebind.models.imagebind_model import ModalityType

class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        video_input: Optional[torch.FloatTensor] = None,
        video_label: Optional[torch.FloatTensor] = None,
        audio_input: Optional[torch.FloatTensor] = None,
        audio_label: Optional[torch.FloatTensor] = None,
        imu_left_input: Optional[torch.FloatTensor] = None,
        imu_left_label: Optional[torch.FloatTensor] = None,
        imu_right_input: Optional[torch.FloatTensor] = None,
        imu_right_label: Optional[torch.FloatTensor] = None,
        gaze: Optional[torch.FloatTensor] = None,#(choiszt)TODO to be updated soon
        speech:Optional[torch.FloatTensor] = None,
        caption: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        LlamaForCausalLM.__init__(self, config)
        self.model = LlavaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        prompts: Optional[List[str]] = None,
        modalities: Optional[List[str]] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                modalities,
                image_sizes,
                prompts
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        modalities = kwargs.pop("modalities", None)
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                modalities,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

class EgoGPTLlama(LlavaLlamaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        LlamaForCausalLM.__init__(self, config)
        self.model = LlavaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.ego_gpthead= nn.Linear(config.hidden_size, 1024, bias=False)
        self.normalize=nn.Softmax(dim=0)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        video_input: Optional[torch.FloatTensor] = None,
        video_label: Optional[torch.FloatTensor] = None,
        audio_input: Optional[torch.FloatTensor] = None,
        audio_label: Optional[torch.FloatTensor] = None,
        imu_left_input: Optional[torch.FloatTensor] = None,
        imu_left_label: Optional[torch.FloatTensor] = None,
        imu_right_input: Optional[torch.FloatTensor] = None,
        imu_right_label: Optional[torch.FloatTensor] = None,
        gaze: Optional[torch.FloatTensor] = None,#(choiszt)TODO to be updated soon
        speech:Optional[torch.FloatTensor] = None,
        caption: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        #out_feat: imagebind->projector
        #label_feat: imagebind
        out_feat,label_feat=self.prepare_inputs_labels_for_EgoGPT(
            video_input=video_input,
            video_label=video_label,
            audio_input=audio_input,
            audio_label=audio_label,
            imu_left_input=imu_left_input,
            imu_left_label=imu_left_label,
            imu_right_input=imu_right_input,
            imu_right_label=imu_right_label
            )

        hidden_states = out_feat

        for layers in self.model.layers:
            layer_outputs = layers(hidden_states)
            hidden_states = layer_outputs[0]

        hidden_states = self.model.norm(hidden_states)

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.ego_gpthead(hidden_states)
        logits = logits.float()
        
        loss = None
        if logits is not None:
            loss_fct = CrossEntropyLoss()
            
            loss = loss_fct(self.normalize(logits), label_feat)

        # if not return_dict:
        #     output = (logits,) + outputs[1:]
        #     return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
        )

    def prepare_inputs_labels_for_EgoGPT(
        self,
        video_input: Optional[torch.FloatTensor] = None,
        video_label: Optional[torch.FloatTensor] = None,
        audio_input: Optional[torch.FloatTensor] = None,
        audio_label: Optional[torch.FloatTensor] = None,
        imu_left_input: Optional[torch.FloatTensor] = None,
        imu_left_label: Optional[torch.FloatTensor] = None,
        imu_right_input: Optional[torch.FloatTensor] = None,
        imu_right_label: Optional[torch.FloatTensor] = None,
        gaze: Optional[torch.FloatTensor] = None,#(choiszt)TODO to be updated soon
        speech:Optional[torch.FloatTensor] = None,
        caption: Optional[torch.FloatTensor] = None,
    ):
        vision_tower = self.get_vision_tower()
        vision_tower.vision_tower.to(torch.float32)

        def _get_modalities_data(video,audio,imu_left,imu_right):
            modalities_data={}
            if len(video)==1: #batch_size=1
                if video:
                    modalities_data.update({ModalityType.VISION:video.unsqueeze(0).to(torch.float32)})
                if audio:
                    modalities_data.update({ModalityType.AUDIO:audio.unsqueeze(0).to(torch.float32)})
                if imu_left:
                    modalities_data.update({ModalityType.IMU_LEFT:imu_left.unsqueeze(0).to(torch.float32)})
                if imu_right:
                    modalities_data.update({ModalityType.IMU_RIGHT:imu_right.unsqueeze(0).to(torch.float32)})
            else:
                if video:
                    modalities_data.update({ModalityType.VISION:torch.stack(video).to(torch.float32)})
                if audio:
                    modalities_data.update({ModalityType.AUDIO:torch.stack(audio).to(torch.float32)})
                if imu_left:
                    modalities_data.update({ModalityType.IMU_LEFT:torch.stack(imu_left).to(torch.float32)})
                if imu_right:
                    modalities_data.update({ModalityType.IMU_RIGHT:torch.stack(imu_right).to(torch.float32)})
            return modalities_data

     
        input_modalities=_get_modalities_data(video_input,audio_input,imu_left_input,imu_right_input)
        label_modalities=_get_modalities_data(video_label,audio_label,imu_left_label,imu_right_label)
        egpgpt_feat=vision_tower.vision_tower(input_modalities)
        label_feat=vision_tower.vision_tower(label_modalities)

        aligned_feat = [[] for _ in range(len(video_input))]
        aligned_feat_out=[[] for _ in range(len(video_input))]
        #(choiszt) TODO add align options
        for _,feats in egpgpt_feat.items():
            for batch,feat in enumerate(feats):
                aligned_feat[batch].append(feat.unsqueeze(0))
        batch_feat=[]
        for feats in aligned_feat:
            batch_feat.append(torch.cat(feats))

        for _,feats in label_feat.items():
            for batch,feat in enumerate(feats):
                aligned_feat_out[batch].append(feat.unsqueeze(0))
        batch_feat_out=[]
        for feats in aligned_feat_out:
            batch_feat_out.append(torch.cat(feats))    

        out_feats=self.get_model().mm_projector(torch.stack(batch_feat).to(torch.bfloat16))

        return out_feats, torch.stack(batch_feat_out).to(torch.bfloat16)
    
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        modalities = kwargs.pop("modalities", None)
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                modalities,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs
    

if LlavaConfig.model_type == "llava":
    LlavaConfig.model_type = "llava_llama" # directly set to llava_dev to avoid conflict with HF's llava
    
AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
