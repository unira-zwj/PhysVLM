#!/bin/bash

MODEL_NAME_OR_PATH=Qwen/Qwen2.5-3B-Instruct
TOWER_NAME_OR_PATH=google/siglip-so400m-patch14-384

STAGE1_DEEPSPEED_FILE=./scripts/train/zero2.json
STAGE1_VERSION=qwen2
STAGE1_TRAIN_ANNOTATION_PATH=./datasets/physvlm_pretrain.json
STAGE1_TRAIN_DATA_IMAGE_FOLDER=/mnt/usb2/datasets
STAGE1_OUTPUT_DIR=/mnt/usb2/zhouweijie/ws/physvlm/physvlm_checkpoints/physvlm-qwen2-3B-pretrain

deepspeed --hostfile=${1} physvlm/train/train_mem.py \
    --deepspeed ${STAGE1_DEEPSPEED_FILE} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --version ${STAGE1_VERSION} \
    --train_stage 1 \
    --data_path ${STAGE1_TRAIN_ANNOTATION_PATH} \
    --image_folder ${STAGE1_TRAIN_DATA_IMAGE_FOLDER} \
    --vision_tower ${TOWER_NAME_OR_PATH} \
    --depth_tower ${TOWER_NAME_OR_PATH} \
    --image_aspect_ratio None \
    --bf16 True \
    --output_dir ${STAGE1_OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to none \
