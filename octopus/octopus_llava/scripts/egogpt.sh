PROJECT_NAME="egogpt_4.9"

export WANDB_API_KEY="36f2c08bcd41ad272c2e92b45b5c271985f01cf4"
wandb login $WANDB_API_KEY

export WANDB_NAME=$PROJECT_NAME

export WANDB_PROJECT=EgoGPT

wandb online

deepspeed --master_port 29503 --include localhost:4,5,6,7 llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path weight/llava-v1.6-vicuna-7b \
    --version v1 \
    --data_path data/aria_dataset/0322_2/0322_2_result.json \
    --image_folder ./data/mc \
    --video_folder ./data/mc \
    --vision_tower egogpt \
    --image_processor openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --video_fps 1 \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./work_dirs/egogpt \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --mm_resampler_type spatial_pool \
    --mm_spatial_pool_stride 2 \
    --mm_spatial_pool_out_channels 1024 \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "[(336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008)]" \
    --mm_patch_merge_type spatial_unpad \
    --frames_upbound 0