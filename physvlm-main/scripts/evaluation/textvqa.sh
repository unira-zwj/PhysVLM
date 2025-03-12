#!/bin/bash

# python -m llava.eval.model_vqa_loader \
#     --model-path ./checkpoints/llava-robothink-Llama-3-8B-stage2 \
#     --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
#     --image-folder ./playground/data/eval/textvqa/train_images \
#     --answers-file ./playground/data/eval/textvqa/answers/llava-robothink-Llama-3-8B.jsonl \
#     --temperature 0 \
#     --conv-mode llama3

# python -m llava.eval.eval_textvqa \
#     --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
#     --result-file ./playground/data/eval/textvqa/answers/llava-robothink-Llama-3-8B.jsonl

python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/llava-robothink-Llama-3-8B-stage2-siglip-compress-4-sharegpt-interleaved \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/llava-robothink-Llama-3-8B-siglip-compress-4-sharegpt-interleaved.jsonl \
    --temperature 0 \
    --conv-mode llama3

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/llava-robothink-Llama-3-8B-siglip-compress-4-sharegpt-interleaved.jsonl


# # llava-robothink-Llama-3-8B-stage2-siglip
# python -m llava.eval.model_vqa_loader \
#     --model-path ./checkpoints/llava-robothink-Llama-3-8B-stage2-siglip \
#     --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
#     --image-folder ./playground/data/eval/textvqa/train_images \
#     --answers-file ./playground/data/eval/textvqa/answers/llava-robothink-Llama-3-8B-siglip.jsonl \
#     --temperature 0 \
#     --conv-mode llama3

# python -m llava.eval.eval_textvqa \
#     --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
#     --result-file ./playground/data/eval/textvqa/answers/llava-robothink-Llama-3-8B-siglip.jsonl 


# # llava-robothink-Llama-3-8B-stage2-siglip-compress-2
# python -m llava.eval.model_vqa_loader \
#     --model-path ./checkpoints/llava-robothink-Llama-3-8B-stage2-siglip-compress-2 \
#     --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
#     --image-folder ./playground/data/eval/textvqa/train_images \
#     --answers-file ./playground/data/eval/textvqa/answers/llava-robothink-Llama-3-8B-siglip-compress-2.jsonl \
#     --temperature 0 \
#     --conv-mode llama3

# python -m llava.eval.eval_textvqa \
#     --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
#     --result-file ./playground/data/eval/textvqa/answers/llava-robothink-Llama-3-8B-siglip-compress-2.jsonl 


# # llava-robothink-Llama-3-8B-stage2-siglip-compress-4
# python -m llava.eval.model_vqa_loader \
#     --model-path ./checkpoints/llava-robothink-Llama-3-8B-stage2-siglip-compress-4 \
#     --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
#     --image-folder ./playground/data/eval/textvqa/train_images \
#     --answers-file ./playground/data/eval/textvqa/answers/llava-robothink-Llama-3-8B-siglip-compress-4.jsonl \
#     --temperature 0 \
#     --conv-mode llama3

# python -m llava.eval.eval_textvqa \
#     --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
#     --result-file ./playground/data/eval/textvqa/answers/llava-robothink-Llama-3-8B-siglip-compress-4.jsonl 


# # llava-robothink-Llama-3-8B-stage2-siglip-compress-4-sharegpt
# python -m llava.eval.model_vqa_loader \
#     --model-path ./checkpoints/llava-robothink-Llama-3-8B-stage2-siglip-compress-4-sharegpt \
#     --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
#     --image-folder ./playground/data/eval/textvqa/train_images \
#     --answers-file ./playground/data/eval/textvqa/answers/llava-robothink-Llama-3-8B-siglip-compress-4-sharegpt.jsonl \
#     --temperature 0 \
#     --conv-mode llama3

# python -m llava.eval.eval_textvqa \
#     --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
#     --result-file ./playground/data/eval/textvqa/answers/llava-robothink-Llama-3-8B-siglip-compress-4-sharegpt.jsonl 


# # llava-robothink-Llama-3-8B-stage2-siglip-compress-4-sharegpt-interleaved
# python -m llava.eval.model_vqa_loader \
#     --model-path ./checkpoints/llava-robothink-Llama-3-8B-stage2-siglip-compress-4-sharegpt-interleaved \
#     --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
#     --image-folder ./playground/data/eval/textvqa/train_images \
#     --answers-file ./playground/data/eval/textvqa/answers/llava-robothink-Llama-3-8B-siglip-compress-4-sharegpt-interleaved.jsonl \
#     --temperature 0 \
#     --conv-mode llama3

# python -m llava.eval.eval_textvqa \
#     --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
#     --result-file ./playground/data/eval/textvqa/answers/llava-robothink-Llama-3-8B-siglip-compress-4-sharegpt-interleaved.jsonl 


# # llava-robothink-Llama-3-8B-stage2-siglip-compress-4-sharegpt-interleaved-robovqa
# python -m llava.eval.model_vqa_loader \
#     --model-path ./checkpoints/llava-robothink-Llama-3-8B-stage2-siglip-compress-4-sharegpt-interleaved-robovqa \
#     --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
#     --image-folder ./playground/data/eval/textvqa/train_images \
#     --answers-file ./playground/data/eval/textvqa/answers/llava-robothink-Llama-3-8B-siglip-compress-4-sharegpt-interleaved-robovqa.jsonl \
#     --temperature 0 \
#     --conv-mode llama3

# python -m llava.eval.eval_textvqa \
#     --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
#     --result-file ./playground/data/eval/textvqa/answers/llava-robothink-Llama-3-8B-siglip-compress-4-sharegpt-interleaved-robovqa.jsonl 
