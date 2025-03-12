#!/bin/bash

SPLIT="mmbench_dev_20230712"

# # llava-robothink-Llama-3-8B-stage2
# python -m llava.eval.model_vqa_mmbench \
#     --model-path ./checkpoints/llava-robothink-Llama-3-8B-stage2 \
#     --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
#     --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/llava-robothink-Llama-3-8B-stage2.jsonl \
#     --single-pred-prompt \
#     --temperature 0 \
#     --conv-mode llama3

# mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

# python scripts/convert_mmbench_for_submission.py \
#     --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
#     --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
#     --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
#     --experiment llava-robothink-Llama-3-8B-stage2


# # llava-robothink-Llama-3-8B-stage2-siglip
# python -m llava.eval.model_vqa_mmbench \
#     --model-path ./checkpoints/llava-robothink-Llama-3-8B-stage2-siglip \
#     --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
#     --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/llava-robothink-Llama-3-8B-stage2-siglip.jsonl \
#     --single-pred-prompt \
#     --temperature 0 \
#     --conv-mode llama3

# mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

# python scripts/convert_mmbench_for_submission.py \
#     --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
#     --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
#     --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
#     --experiment llava-robothink-Llama-3-8B-stage2-siglip


# # llava-robothink-Llama-3-8B-stage2-siglip-compress-2
# python -m llava.eval.model_vqa_mmbench \
#     --model-path ./checkpoints/llava-robothink-Llama-3-8B-stage2-siglip-compress-2 \
#     --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
#     --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/llava-robothink-Llama-3-8B-stage2-siglip-compress-2.jsonl \
#     --single-pred-prompt \
#     --temperature 0 \
#     --conv-mode llama3

# mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

# python scripts/convert_mmbench_for_submission.py \
#     --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
#     --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
#     --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
#     --experiment llava-robothink-Llama-3-8B-stage2-siglip-compress-2


# llava-robothink-Llama-3-8B-stage2-siglip-compress-4
python -m llava.eval.model_vqa_mmbench \
    --model-path ./checkpoints/llava-robothink-Llama-3-8B-stage2-siglip-compress-4 \
    --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/llava-robothink-Llama-3-8B-stage2-siglip-compress-4.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode llama3

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment llava-robothink-Llama-3-8B-stage2-siglip-compress-4


# llava-robothink-Llama-3-8B-stage2-siglip-compress-4-sharegpt
python -m llava.eval.model_vqa_mmbench \
    --model-path ./checkpoints/llava-robothink-Llama-3-8B-stage2-siglip-compress-4-sharegpt \
    --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/llava-robothink-Llama-3-8B-stage2-siglip-compress-4-sharegpt.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode llama3

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment llava-robothink-Llama-3-8B-stage2-siglip-compress-4-sharegpt