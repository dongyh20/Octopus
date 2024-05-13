# sh scripts/Otter_MPT3B_EAI.sh

export PYTHONPATH=.
RUN_NAME="Otter_MPT7B_EAI_$(date +'%Y-%m-%d-%H-%M-%S')"
GPU=2
WORKERS=$((${GPU}*2))

echo "Using ${GPU} GPUs and ${WORKERS} workers"
echo "Running ${RUN_NAME}"

accelerate launch --config_file=pipeline/accelerate_configs/accelerate_config_zero3.yaml \
    --num_processes=${GPU} --main_process_port=29500 \
    pipeline/train/instruction_following.py \
    --pretrained_model_name_or_path=luodian/OTTER-Video-LLaMA7B-DenseCaption \
    --customized_config=scripts/Otter_MPT7B_EAI.json \
    --training_data_yaml=scripts/EAI_Recipe.yaml \
    --model_name=otter \
    --instruction_format=simple \
    --batch_size=4 \
    --num_epochs=3 \
    --report_to_wandb \
    --wandb_entity=ntu-slab \
    --external_save_dir=checkpoints \
    --save_hf_model \
    --run_name=${RUN_NAME} \
    --wandb_project=octopus \
    --workers=${WORKERS} \
    --lr_scheduler=cosine \
    --learning_rate=1e-5 \
    --warmup_steps_ratio=0.01 \
    --max_seq_len=1600 \
    --resample_frames=10 \
    --resize_embedding \
    --keep_symbols

# cd ./checkpoints;
# source ~/.zshrc;
# azcopy_upload ./${RUN_NAME} otter/checkpoints/