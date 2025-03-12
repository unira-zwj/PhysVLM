#!/bin/bash

MODEL_NAME_OR_PATH=Qwen/Qwen2.5-3B-Instruct
TOWER_NAME_OR_PATH=google/siglip-so400m-patch14-384

STAGE1_OUTPUT_DIR=/mnt/usb2/zhouweijie/ws/physvlm/physvlm_checkpoints/physvlm-qwen2-3B-pretrain

STAGE2_DEEPSPEED_FILE=./scripts/train/zero2.json
STAGE2_VERSION=qwen2
STAGE2_TRAIN_ANNOTATION_PATH=./datasets/physvlm_ft_vid.json
STAGE2_TRAIN_DATA_IMAGE_FOLDER=/mnt/usb2/datasets
STAGE2_OUTPUT_DIR=/mnt/usb2/zhouweijie/ws/physvlm/physvlm_checkpoints/physvlm-qwen2-3B-vid-plan

deepspeed --hostfile=${1} physvlm/train/train_mem.py \
    --deepspeed ${STAGE2_DEEPSPEED_FILE} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --version ${STAGE2_VERSION} \
    --train_stage 2 \
    --data_path ${STAGE2_TRAIN_ANNOTATION_PATH} \
    --image_folder ${STAGE2_TRAIN_DATA_IMAGE_FOLDER} \
    --vision_tower ${TOWER_NAME_OR_PATH} \
    --depth_tower ${TOWER_NAME_OR_PATH} \
    --pretrain_mm_mlp_adapter ${STAGE1_OUTPUT_DIR}/projector/mm_projector.bin\
    --image_aspect_ratio None \
    --bf16 True \
    --output_dir ${STAGE2_OUTPUT_DIR} \
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
    --model_max_length 3074 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to none \