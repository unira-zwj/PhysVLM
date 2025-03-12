#!/bin/bash

python -m evaluation.model_vqa_science \
    --model-path /data/disk0/output_cp/robothink-Llama3-8B-siglip-compress4-sharegpt-robovqa-stage2 \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/robothink-llama3_8b-siglip-compress-4-sharegpt-robovqa.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode llama3

python evaluation/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/robothink-llama3_8b-siglip-compress-4-sharegpt-robovqa.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/robothink-llama3_8b-siglip-compress-4-sharegpt-robovqa_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/robothink-llama3_8b-siglip-compress-4-sharegpt-robovqa_result.json
