#!/bin/bash
cd /mnt/bn/vl-research/workspace/yhzhang/llava-video-old

#  Get the installed version of transformers
installed_version=$(pip3 show transformers | grep Version | cut -d ' ' -f 2)

# Check if the installed version is not the latest
if [ "$installed_version" != "4.38.2" ]; then
    pip3 install transformers==4.38.2
fi

# Get the installed version of deepspeed
installed_version=$(pip3 show deepspeed | grep Version | cut -d ' ' -f 2)

# Check if the installed version is not the latest
if [ "$installed_version" != "0.12.2" ]; then
    pip3 install deepspeed==0.12.2
fi

# Install ninja if not installed
if ! pip3 show ninja > /dev/null 2>&1; then
    pip3 install ninja
fi

# Install flash-atten if not installed
if ! pip3 show flash-attn > /dev/null 2>&1; then
    pip3 install flash-attn --no-build-isolation
fi

# Install decord if not installed
if ! pip3 show decord > /dev/null 2>&1; then
    pip3 install decord
fi

# Install protobuf if not installed
if ! pip3 show protobuf > /dev/null 2>&1; then
    pip3 install protobuf 
fi

# Install torchvision if not installed
if ! pip3 show torchvision > /dev/null 2>&1; then
    pip3 install torchvision==0.16.0
fi

# Install timm if not installed
if ! pip3 show timm > /dev/null 2>&1; then
    pip3 install timm
fi

# Get the installed version of transformers
installed_version=$(pip3 show accelerate | grep Version | cut -d ' ' -f 2)

# Check if the installed version is not the latest
if [ "$installed_version" != "0.27.2" ]; then
    pip3 install accelerate==0.27.2
fi


# Install sentencepiece if not installed
if ! pip3 show sentencepiece > /dev/null 2>&1; then
    pip3 install sentencepiece==0.1.99
fi

pip3 install -U wandb

FRAMES_UPBOUND=$1
POOL_STRIDE=$2
MODEL_MAX_LENGTH=$3

################## project ##################
PROJECT_NAME=llava-vicuna_7B-mlp2x_gelu-llava_558k_with_webvid-spatial_pool4-video_fps1-ft-video_image_mix-frames_upbound_${FRAMES_UPBOUND}-pool_stride_${POOL_STRIDE}-model_max_length_${MODEL_MAX_LENGTH}-video_only


# wandb configure
wandb login $WANDB_API_KEY

export WANDB_NAME=$PROJECT_NAME

export WANDB_PROJECT=LLaVA_v1.6_video

wandb online

#llava_158k_detailv3_reinstall_gpt4v24k_wild15k_mixdocvqa_fixdca30k_fixsynden40k_sg40kt2k_ori_with_video_chatgpt_maxtime_5min

deepspeed --master_port 29501 llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path model_zoo/LLM/vicuna/7B-V1.5 \
    --version v1 \
    --data_path ./data/LLaMA-VID-Finetune/video_chatgpt_maxtime_5min.json \
    --image_folder ./data/LLaMA-VID-Finetune/ \
    --video_folder ./data/LLaMA-VID-Finetune/ \
    --vision_tower  openai/clip-vit-large-patch14 \
    --image_processor openai/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter /mnt/bn/vl-research/workspace/yhzhang/llava-video-old/work_dirs/llava-vicuna_7B-mlp2x_gelu-llava_558k_with_webvid-spatial_pool4-video_fps1-pt/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --video_fps 1 \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./work_dirs/$PROJECT_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length ${MODEL_MAX_LENGTH:-2048} \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --mm_resampler_type "spatial_pool" \
    --mm_spatial_pool_stride ${POOL_STRIDE:-4} \
    --mm_spatial_pool_out_channels 1024 \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "[(224, 448), (224, 672), (224, 896), (448, 448), (448, 224), (672, 224), (896, 224)]" \
    --mm_patch_merge_type spatial_unpad \
    --frames_upbound ${FRAMES_UPBOUND:-0} \
