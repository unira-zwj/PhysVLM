import argparse
import torch
import json
import warnings
import cv2
import numpy as np
from PIL import Image
import os
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import shutil

from robothink.constants import IMAGE_TOKEN_INDEX, DEPTH_TOKEN_INDEX
from robothink.conversation import conv_templates, SeparatorStyle
from robothink.model.builder import load_pretrained_model
from robothink.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
    disable_torch_init
)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

warnings.filterwarnings('ignore')


class MultiFrameInfer:
    def __init__(self, model_path, device="cuda:0", conv_mode="qwen2", temperature=0.01, top_p=None, max_new_tokens=256):        
        disable_torch_init()
        
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, None, model_name, device=device
        )
        
        self.conv_mode = conv_mode
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        
    def load_image(self, image_file):
        if image_file.startswith("http") or image_file.startswith("https"):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
        return image
    
    def load_sp_map(self, sp_map):
        if sp_map.startswith("http") or sp_map.startswith("https"):
            response = requests.get(sp_map)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(sp_map).convert("RGB")
        return image
    
    def load_depth(self, depth_file):
        depth_image = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        # 确保深度图像为浮点型以便进行处理
        depth_image = depth_image.astype(np.float32)
        # print(list(depth_image))
        # 定义深度范围并进行裁剪
        max_depth_value = np.max(depth_image)
        min_depth_value = np.min(depth_image)
        # max_depth_value = min(3000, int(max_depth_value))  # 最大深度值（毫米）
        # min_depth_value = max(0, int(min_depth_value))     # 最小深度值（毫米）
        depth_image = np.clip(depth_image, min_depth_value, max_depth_value)
        
        # 使用固定的最小值和最大值进行归一化
        depth_normalized = (depth_image - min_depth_value) / (max_depth_value - min_depth_value)
        depth_normalized = np.clip(depth_normalized, 0, 1)

        # 将归一化的深度值转换为8位整数
        depth_normalized_uint8 = (depth_normalized * 255).astype(np.uint8)

        # 应用 COLORMAP_JET 将深度图像转换为伪彩色图像
        depth_colored = cv2.applyColorMap(depth_normalized_uint8, cv2.COLORMAP_JET)
        
        cv2.imwrite("./depth_tmp.jpg", depth_colored)
        
        depth_colored = Image.fromarray(cv2.cvtColor(depth_colored,cv2.COLOR_BGR2RGB)) 
        return depth_colored

    def inference(self, image_path_list, depth_path_list, query):
        if depth_path_list[0] == "":
            depth_path_list = None
        image_list = []
        depth_list = []
        for image_path in image_path_list:
            image = self.load_image(image_path) # [3, 384, 384]
            image_list.append(image)
        
        image_tensor = process_images(image_list, self.image_processor, self.model.config)
        image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
        
        if depth_path_list is not None:
            for depth_path in depth_path_list:
                depth = self.load_sp_map(depth_path)
                depth_list.append(depth)
            
            depth_tensor = process_images(depth_list, self.image_processor, self.model.config)
            depth_tensor = [image.to(self.model.device, dtype=torch.float16) for image in depth_tensor]
                    
            tmp = "<image>\n<depth>\n" * len(image_list)
        else:
            tmp = "<image>\n" * len(image_list)
            depth_tensor = image_tensor
        
        inp = tmp + query
        print(inp)
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, DEPTH_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stop_str = conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        try:
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    depth_images=depth_tensor,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    num_beams=1,
                    max_new_tokens=self.max_new_tokens,
                    use_cache=True,
                    # stopping_criteria=[stopping_criteria],
                )
            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].replace("<|endoftext|>", "").strip()
            print(outputs)
        except Exception as e:
            print(e)
        return outputs

    
class InferenceRequest(BaseModel):
    image_paths: List[str]
    depth_paths: Optional[List[str]] = None
    query: str

# Initialize the inference model
model_path = "../physvlm_checkpoints/physvlm-qwen2-3B"
inference_model = MultiFrameInfer(model_path=model_path)

@app.post("/v1/inference")
async def run_inference(request: InferenceRequest):
    try:
        response = inference_model.inference(
            image_path_list=request.image_paths,
            depth_path_list=request.depth_paths,
            query=request.query
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)