#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/llava-robothink-Llama-3-8B-stage2 \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/llava-robothink-Llama-3-8B.jsonl \
    --temperature 0 \
    --conv-mode llama3

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment llava-v1.5-13b

cd eval_tool

python calculation.py --results_dir answers/llava-v1.5-13b


# llava-robothink-Llama-3-8B-stage2-siglip
# python -m llava.eval.model_vqa_loader \
#     --model-path ./checkpoints/llava-robothink-Llama-3-8B-stage2-siglip \
#     --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
#     --image-folder ./playground/data/eval/textvqa/train_images \
#     --answers-file ./playground/data/eval/textvqa/answers/llava-robothink-Llama-3-8B-siglip.jsonl \
#     --temperature 0 \
#     --conv-mode llama3