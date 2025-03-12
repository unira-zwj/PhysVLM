#!/bin/bash

MODEL_NAME_OR_PATH=./checkpoints/physvlm-qwen2-3B
TOWER_NAME_OR_PATH=google/siglip-so400m-patch14-384

# full finetune
DEEPSPEED_FILE=./scripts/train/zero2.json
VERSION=qwen2
TRAIN_ANNOTATION_PATH=
TRAIN_DATA_IMAGE_FOLDER=
OUTPUT_DIR=./checkpoints/physvlm-qwen2-3B-lora

# CUDA_VISIBLE_DEVICES=0
deepspeed --hostfile=${1} physvlm/train/train_mem.py \
    --deepspeed ${DEEPSPEED_FILE} \
    --lora_enable True \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --version ${VERSION} \
    --train_stage 2 \
    --data_path ${TRAIN_ANNOTATION_PATH} \
    --image_folder ${TRAIN_DATA_IMAGE_FOLDER} \
    --vision_tower ${TOWER_NAME_OR_PATH} \
    --depth_tower ${TOWER_NAME_OR_PATH} \
    --image_aspect_ratio None \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \