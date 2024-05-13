#!/bin/bash
DIR="/mnt/bn/vl-research/workspace/yhzhang/llava-video-old"

# Check if the directory exists
if [ -d "$DIR" ]; then
    # If the directory exists, set BYTENAS to "vl-research"
    BYTENAS="vl-research"
else
    # If the directory does not exist, set BYTENAS to "vl-research-cn-lq"
    BYTENAS="vl-research-cn-lq"

    export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
    export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
    export HF_HOME=/mnt/bn/vl-research-cn-lq/.cache/huggingface
fi

DIR=/mnt/bn/${BYTENAS}/workspace/yhzhang/llava-video-old

cd ${DIR}

pip3 install --upgrade pip

# Get the installed version of transformers
installed_version=$(pip3 show transformers | grep Version | cut -d ' ' -f 2)

# # Check if the installed version is not the latest
# if [ "$installed_version" != "4.38.2" ]; then
#     pip3 install transformers==4.38.2
# fi
pip3 install git+https://github.com/huggingface/transformers.git@56b64bf1a51e29046bb3f8ca15839ff4d6a92c74


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

if [ "$MODEL_MAX_LENGTH" -ge 4096 ]
then
  LLM="vicuna-7b-v1-5-8k"
else
  LLM="vicuna-7b-v1-5"
fi

################## project ##################
PROJECT_NAME="llava-vicuna_7B-mlp2x_gelu-pretrain_blip558k_plain-336px-ft-video_image_mix-frames_upbound_${FRAMES_UPBOUND}-pool_stride_${POOL_STRIDE}-model_max_length_${MODEL_MAX_LENGTH}"


# wandb configure
wandb login $WANDB_API_KEY

export WANDB_NAME=$PROJECT_NAME

export WANDB_PROJECT=LLaVA_v1.6_video

wandb online

#llava_158k_detailv3_reinstall_gpt4v24k_wild15k_mixdocvqa_fixdca30k_fixsynden40k_sg40kt2k_ori_with_video_chatgpt_maxtime_5min

# 取 worker0 第一个 port
ports=(`echo $METIS_WORKER_0_PORT | tr ',' ' '`)
port=${ports[0]}
# port=30000

port_in_cmd="$(echo "${METIS_WORKER_0_PORT:-2222}"| awk -F',' '{print $1}')"

echo "total workers: ${ARNOLD_WORKER_NUM}"
echo "cur worker id: ${ARNOLD_ID}"
echo "gpus per worker: ${ARNOLD_WORKER_GPU}"
echo "master ip: ${METIS_WORKER_0_HOST}"
echo "master port: ${port}"
echo "master port in cmd: ${port_in_cmd}"

export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
# export NCCL_IB_HCA=${ARNOLD_RDMA_DEVICE}
export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO\

echo $((4 / ARNOLD_WORKER_NUM))

echo ${LLM}

torchrun --nproc_per_node="${ARNOLD_WORKER_GPU}" --nnodes="${ARNOLD_WORKER_NUM}" --node_rank="${ARNOLD_ID}" --master_addr="${METIS_WORKER_0_HOST}" --master_port="${port_in_cmd}" \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /mnt/bn/${BYTENAS}/checkpoints/vicuna/${LLM} \
    --version v1 \
    --data_path /mnt/bn/${BYTENAS}/data/llava_video/meta/finetune/llava_158k_detailv3_reinstall_gpt4v24k_wild15k_mixdocvqa_fixdca30k_fixsynden40k_sg40kt2k_ori_with_video_chatgpt_maxtime_5min.json \
    --image_folder /mnt/bn/${BYTENAS}/data/llava_data \
    --video_folder /mnt/bn/${BYTENAS}/data/llava_video \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --image_processor openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter /mnt/bn/${BYTENAS}/checkpoints/projectors/ds_llava-vicuna-7b-v1-5-clip_large_336px-mlp2x_gelu-pretrain_blip558k_plain/mm_projector.bin \
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
    --gradient_accumulation_steps $((4 / ARNOLD_WORKER_NUM)) \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 0 \
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
    --image_grid_pinpoints "[(336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008)]" \
    --mm_patch_merge_type spatial_unpad \
    --frames_upbound ${FRAMES_UPBOUND:-0}
