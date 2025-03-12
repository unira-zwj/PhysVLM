#!/bin/bash

# # llava-robothink-Llama-3-8B-stage2
# python -m llava.eval.model_vqa \
#     --model-path ./checkpoints/llava-robothink-Llama-3-8B-stage2 \
#     --question-file ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
#     --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
#     --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/llava-robothink-Llama-3-8B-stage2.jsonl \
#     --temperature 0 \
#     --conv-mode llama3

# mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews


# # llava-robothink-Llama-3-8B-stage2-siglip
# python -m llava.eval.model_vqa \
#     --model-path ./checkpoints/llava-robothink-Llama-3-8B-stage2-siglip \
#     --question-file ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
#     --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
#     --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/llava-robothink-Llama-3-8B-stage2-siglip.jsonl \
#     --temperature 0 \
#     --conv-mode llama3

# mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews


# llava-robothink-Llama-3-8B-stage2-siglip-compress-4-sharegpt-interleaved-robovqa
python -m llava.eval.model_vqa \
    --model-path ./checkpoints/llava-robothink-Llama-3-8B-stage2-siglip-compress-4-sharegpt-interleaved-robovqa \
    --question-file ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
    --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/llava-robothink-Llama-3-8B-stage2-siglip-compress-4-sharegpt-interleaved-robovqa.jsonl \
    --temperature 0 \
    --conv-mode llama3

mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews


python llava/eval/eval_gpt_review_bench.py \
    --question playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --context playground/data/eval/llava-bench-in-the-wild/context.jsonl \
    --rule llava/eval/table/rule.json \
    --answer-list \
        playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
        playground/data/eval/llava-bench-in-the-wild/answers/llava-robothink-Llama-3-8B-stage2-siglip-compress-4-sharegpt-interleaved-robovqa.jsonl \
    --output \
        playground/data/eval/llava-bench-in-the-wild/reviews/llava-robothink-Llama-3-8B-stage2-siglip-compress-4-sharegpt-interleaved-robovqa.jsonl

python llava/eval/summarize_gpt_review.py -f playground/data/eval/llava-bench-in-the-wild/reviews/llava-robothink-Llama-3-8B-stage2-siglip-compress-4-sharegpt-interleaved-robovqa.jsonl
