#!/bin/bash

# python -m llava.eval.model_vqa_loader \
#     --model-path ./checkpoints/llava-robothink-Llama-3-8B-stage2 \
#     --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
#     --image-folder ./playground/data/eval/vizwiz/test \
#     --answers-file ./playground/data/eval/vizwiz/answers/llava-robothink-Llama-3-8B.jsonl \
#     --temperature 0 \
#     --conv-mode llama3

# python scripts/convert_vizwiz_for_submission.py \
#     --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
#     --result-file ./playground/data/eval/vizwiz/answers/llava-robothink-Llama-3-8B.jsonl \
#     --result-upload-file ./playground/data/eval/vizwiz/answers_upload/llava-robothink-Llama-3-8B.json

# llava-robothink-Llama-3-8B-stage2-siglip-compress-4-sharegpt-interleaved
python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/llava-robothink-Llama-3-8B-stage2-siglip-compress-4-sharegpt-interleaved \
    --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder ./playground/data/eval/vizwiz/test \
    --answers-file ./playground/data/eval/vizwiz/answers/llava-robothink-Llama-3-8B-siglip-compress-4-sharegpt-interleaved.jsonl \
    --temperature 0 \
    --conv-mode llama3

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/llava-robothink-Llama-3-8B-siglip-compress-4-sharegpt-interleaved.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/llava-robothink-Llama-3-8B-siglip-compress-4-sharegpt-interleaved.json


# # llava-robothink-Llama-3-8B-stage2-siglip
# python -m llava.eval.model_vqa_loader \
#     --model-path ./checkpoints/llava-robothink-Llama-3-8B-stage2-siglip \
#     --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
#     --image-folder ./playground/data/eval/vizwiz/test \
#     --answers-file ./playground/data/eval/vizwiz/answers/llava-robothink-Llama-3-8B-siglip.jsonl \
#     --temperature 0 \
#     --conv-mode llama3

# python scripts/convert_vizwiz_for_submission.py \
#     --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
#     --result-file ./playground/data/eval/vizwiz/answers/llava-robothink-Llama-3-8B-siglip.jsonl \
#     --result-upload-file ./playground/data/eval/vizwiz/answers_upload/llava-robothink-Llama-3-8B-siglip.json


# # llava-robothink-Llama-3-8B-stage2-siglip-compress-2
# python -m llava.eval.model_vqa_loader \
#     --model-path ./checkpoints/llava-robothink-Llama-3-8B-stage2-siglip-compress-2 \
#     --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
#     --image-folder ./playground/data/eval/vizwiz/test \
#     --answers-file ./playground/data/eval/vizwiz/answers/llava-robothink-Llama-3-8B-siglip-compress-2.jsonl \
#     --temperature 0 \
#     --conv-mode llama3

# python scripts/convert_vizwiz_for_submission.py \
#     --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
#     --result-file ./playground/data/eval/vizwiz/answers/llava-robothink-Llama-3-8B-siglip-compress-2.jsonl \
#     --result-upload-file ./playground/data/eval/vizwiz/answers_upload/llava-robothink-Llama-3-8B-siglip-compress-2.json


# # llava-robothink-Llama-3-8B-stage2-siglip-compress-4
# python -m llava.eval.model_vqa_loader \
#     --model-path ./checkpoints/llava-robothink-Llama-3-8B-stage2-siglip-compress-4 \
#     --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
#     --image-folder ./playground/data/eval/vizwiz/test \
#     --answers-file ./playground/data/eval/vizwiz/answers/llava-robothink-Llama-3-8B-siglip-compress-4.jsonl \
#     --temperature 0 \
#     --conv-mode llama3

# python scripts/convert_vizwiz_for_submission.py \
#     --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
#     --result-file ./playground/data/eval/vizwiz/answers/llava-robothink-Llama-3-8B-siglip-compress-4.jsonl \
#     --result-upload-file ./playground/data/eval/vizwiz/answers_upload/llava-robothink-Llama-3-8B-siglip-compress-4.json


# # llava-robothink-Llama-3-8B-stage2-siglip-compress-4-sharegpt
# python -m llava.eval.model_vqa_loader \
#     --model-path ./checkpoints/llava-robothink-Llama-3-8B-stage2-siglip-compress-4-sharegpt \
#     --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
#     --image-folder ./playground/data/eval/vizwiz/test \
#     --answers-file ./playground/data/eval/vizwiz/answers/llava-robothink-Llama-3-8B-siglip-compress-4-sharegpt.jsonl \
#     --temperature 0 \
#     --conv-mode llama3

# python scripts/convert_vizwiz_for_submission.py \
#     --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
#     --result-file ./playground/data/eval/vizwiz/answers/llava-robothink-Llama-3-8B-siglip-compress-4-sharegpt.jsonl \
#     --result-upload-file ./playground/data/eval/vizwiz/answers_upload/llava-robothink-Llama-3-8B-siglip-compress-4-sharegpt.json


# # llava-robothink-Llama-3-8B-stage2-siglip-compress-4-sharegpt-interleaved
# python -m llava.eval.model_vqa_loader \
#     --model-path ./checkpoints/llava-robothink-Llama-3-8B-stage2-siglip-compress-4-sharegpt-interleaved \
#     --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
#     --image-folder ./playground/data/eval/vizwiz/test \
#     --answers-file ./playground/data/eval/vizwiz/answers/llava-robothink-Llama-3-8B-siglip-compress-4-sharegpt-interleaved.jsonl \
#     --temperature 0 \
#     --conv-mode llama3

# python scripts/convert_vizwiz_for_submission.py \
#     --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
#     --result-file ./playground/data/eval/vizwiz/answers/llava-robothink-Llama-3-8B-siglip-compress-4-sharegpt-interleaved.jsonl \
#     --result-upload-file ./playground/data/eval/vizwiz/answers_upload/llava-robothink-Llama-3-8B-siglip-compress-4-sharegpt-interleaved.json


# # llava-robothink-Llama-3-8B-stage2-siglip-compress-4-sharegpt-interleaved-robovqa
# python -m llava.eval.model_vqa_loader \
#     --model-path ./checkpoints/llava-robothink-Llama-3-8B-stage2-siglip-compress-4-sharegpt-interleaved-robovqa \
#     --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
#     --image-folder ./playground/data/eval/vizwiz/test \
#     --answers-file ./playground/data/eval/vizwiz/answers/llava-robothink-Llama-3-8B-siglip-compress-4-sharegpt-interleaved-robovqa.jsonl \
#     --temperature 0 \
#     --conv-mode llama3

# python scripts/convert_vizwiz_for_submission.py \
#     --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
#     --result-file ./playground/data/eval/vizwiz/answers/llava-robothink-Llama-3-8B-siglip-compress-4-sharegpt-interleaved-robovqa.jsonl \
#     --result-upload-file ./playground/data/eval/vizwiz/answers_upload/llava-robothink-Llama-3-8B-siglip-compress-4-sharegpt-interleaved-robovqa.json

