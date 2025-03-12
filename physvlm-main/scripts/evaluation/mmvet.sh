#!/bin/bash

# # llava-robothink-Llama-3-8B-stage2
# python -m llava.eval.model_vqa \
#     --model-path ./checkpoints/llava-robothink-Llama-3-8B-stage2 \
#     --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
#     --image-folder ./playground/data/eval/mm-vet/images \
#     --answers-file ./playground/data/eval/mm-vet/answers/llava-robothink-Llama-3-8B-stage2.jsonl \
#     --temperature 0 \
#     --conv-mode llama3

# mkdir -p ./playground/data/eval/mm-vet/results

# python scripts/convert_mmvet_for_eval.py \
#     --src ./playground/data/eval/mm-vet/answers/llava-robothink-Llama-3-8B-stage2.jsonl \
#     --dst ./playground/data/eval/mm-vet/results/llava-robothink-Llama-3-8B-stage2.json


# llava-robothink-Llama-3-8B-stage2-clip-only-visual
python -m llava.eval.model_vqa \
    --model-path ./checkpoints/llava-robothink-Llama-3-8B-stage2-clip-only-visual \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/llava-robothink-Llama-3-8B-stage2-clip-only-visual.jsonl \
    --temperature 0 \
    --conv-mode llama3

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/llava-robothink-Llama-3-8B-stage2-clip-only-visual.jsonl \
    --dst ./playground/data/eval/mm-vet/results/llava-robothink-Llama-3-8B-stage2-clip-only-visual.json


# # llava-robothink-Llama-3-8B-stage2-siglip
# python -m llava.eval.model_vqa \
#     --model-path ./checkpoints/llava-robothink-Llama-3-8B-stage2-siglip \
#     --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
#     --image-folder ./playground/data/eval/mm-vet/images \
#     --answers-file ./playground/data/eval/mm-vet/answers/llava-robothink-Llama-3-8B-stage2-siglip.jsonl \
#     --temperature 0 \
#     --conv-mode llama3

# mkdir -p ./playground/data/eval/mm-vet/results

# python scripts/convert_mmvet_for_eval.py \
#     --src ./playground/data/eval/mm-vet/answers/llava-robothink-Llama-3-8B-stage2-siglip.jsonl \
#     --dst ./playground/data/eval/mm-vet/results/llava-robothink-Llama-3-8B-stage2-siglip.json


# # llava-robothink-Llama-3-8B-stage2-siglip-compress-2
# python -m llava.eval.model_vqa \
#     --model-path ./checkpoints/llava-robothink-Llama-3-8B-stage2-siglip-compress-2 \
#     --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
#     --image-folder ./playground/data/eval/mm-vet/images \
#     --answers-file ./playground/data/eval/mm-vet/answers/llava-robothink-Llama-3-8B-stage2-siglip-compress-2.jsonl \
#     --temperature 0 \
#     --conv-mode llama3

# mkdir -p ./playground/data/eval/mm-vet/results

# python scripts/convert_mmvet_for_eval.py \
#     --src ./playground/data/eval/mm-vet/answers/llava-robothink-Llama-3-8B-stage2-siglip-compress-2.jsonl \
#     --dst ./playground/data/eval/mm-vet/results/llava-robothink-Llama-3-8B-stage2-siglip-compress-2.json


# # llava-robothink-Llama-3-8B-stage2-siglip-compress-4
# python -m llava.eval.model_vqa \
#     --model-path ./checkpoints/llava-robothink-Llama-3-8B-stage2-siglip-compress-4 \
#     --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
#     --image-folder ./playground/data/eval/mm-vet/images \
#     --answers-file ./playground/data/eval/mm-vet/answers/llava-robothink-Llama-3-8B-stage2-siglip-compress-4.jsonl \
#     --temperature 0 \
#     --conv-mode llama3

# mkdir -p ./playground/data/eval/mm-vet/results

# python scripts/convert_mmvet_for_eval.py \
#     --src ./playground/data/eval/mm-vet/answers/llava-robothink-Llama-3-8B-stage2-siglip-compress-4.jsonl \
#     --dst ./playground/data/eval/mm-vet/results/llava-robothink-Llama-3-8B-stage2-siglip-compress-4.json


# # llava-robothink-Llama-3-8B-stage2-siglip-compress-4-sharegpt
# python -m llava.eval.model_vqa \
#     --model-path ./checkpoints/llava-robothink-Llama-3-8B-stage2-siglip-compress-4-sharegpt \
#     --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
#     --image-folder ./playground/data/eval/mm-vet/images \
#     --answers-file ./playground/data/eval/mm-vet/answers/llava-robothink-Llama-3-8B-stage2-siglip-compress-4-sharegpt.jsonl \
#     --temperature 0 \
#     --conv-mode llama3

# mkdir -p ./playground/data/eval/mm-vet/results

# python scripts/convert_mmvet_for_eval.py \
#     --src ./playground/data/eval/mm-vet/answers/llava-robothink-Llama-3-8B-stage2-siglip-compress-4-sharegpt.jsonl \
#     --dst ./playground/data/eval/mm-vet/results/llava-robothink-Llama-3-8B-stage2-siglip-compress-4-sharegpt.json



# llava-robothink-Llama-3-8B-stage2-siglip-compress-4-sharegpt-interleaved
python -m llava.eval.model_vqa \
    --model-path ./checkpoints/llava-robothink-Llama-3-8B-stage2-siglip-compress-4-sharegpt-interleaved \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/llava-robothink-Llama-3-8B-stage2-siglip-compress-4-sharegpt-interleaved.jsonl \
    --temperature 0 \
    --conv-mode llama3

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/llava-robothink-Llama-3-8B-stage2-siglip-compress-4-sharegpt-interleaved.jsonl \
    --dst ./playground/data/eval/mm-vet/results/llava-robothink-Llama-3-8B-stage2-siglip-compress-4-sharegpt-interleaved.json


# # llava-robothink-Llama-3-8B-stage2-siglip-compress-4-sharegpt-interleaved-robovqa
# python -m llava.eval.model_vqa \
#     --model-path ./checkpoints/llava-robothink-Llama-3-8B-stage2-siglip-compress-4-sharegpt-interleaved-robovqa \
#     --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
#     --image-folder ./playground/data/eval/mm-vet/images \
#     --answers-file ./playground/data/eval/mm-vet/answers/llava-robothink-Llama-3-8B-stage2-siglip-compress-4-sharegpt-interleaved-robovqa.jsonl \
#     --temperature 0 \
#     --conv-mode llama3

# mkdir -p ./playground/data/eval/mm-vet/results

# python scripts/convert_mmvet_for_eval.py \
#     --src ./playground/data/eval/mm-vet/answers/llava-robothink-Llama-3-8B-stage2-siglip-compress-4-sharegpt-interleaved-robovqa.jsonl \
#     --dst ./playground/data/eval/mm-vet/results/llava-robothink-Llama-3-8B-stage2-siglip-compress-4-sharegpt-interleaved-robovqa.json