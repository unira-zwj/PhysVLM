# Modified from LLaVA: https://github.com/haotian-liu/LLaVA.git
#
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings

from transformers import AutoTokenizer, AutoConfig, BitsAndBytesConfig
import torch
from physvlm.model import LlavaLlamaForCausalLM, LlavaQwen2ForCausalLM
from physvlm.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'
        
    # assert ('llava' in model_name.lower() or 'physvlm' in model_name.lower() or 'walle' in model_name.lower()), "!!!!! unkown model name {}!!!!".format(model_name)

    print(f'model_base is {model_base}\n')
    
    if 'lora' in model_name.lower() and model_base is None:
        warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        
    if 'lora' in model_name.lower() and model_base is not None:
        print('Loading physvlm from base model...')
        from physvlm.model.language_model.llava_qwen import LlavaConfig
        lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        model = LlavaQwen2ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
             
        token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

        print('Loading additional physvlm weights...')
        if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
            non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
        else:
            assert os.path.exists(os.path.join(model_path, 'base_model', 'non_lora_trainables.bin')), "!!!!! non_lora_trainables.bin not found !!!!!"
            # # this is probably from HF Hub
            # from huggingface_hub import hf_hub_download
            # def load_from_hf(repo_id, filename, subfolder=None):
            #     cache_file = hf_hub_download(
            #         repo_id=repo_id,
            #         filename=filename,
            #         subfolder=subfolder)
            #     return torch.load(cache_file, map_location='cpu')
            # non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)

        from peft import PeftModel
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_path)
        print('Merging LoRA weights...')
        model = model.merge_and_unload()
        print('Model is loaded...')
        
    elif model_base is not None:
        print('---physvlm-Qwen2---')
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        cfg_pretrained = AutoConfig.from_pretrained(model_base)
        model = LlavaQwen2ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
        print("---tokenizer---\n", tokenizer)
        print("---model---\n", model)

        mm_projector_weights = torch.load(os.path.join(model_path, 'projector/mm_projector.bin'), map_location='cpu')
        mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
        model.load_state_dict(mm_projector_weights, strict=False)
    else:
        print('---physvlm-Qwen2---')
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = LlavaQwen2ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        print("---tokenizer---\n", tokenizer)
        print("---model---\n", model)
        return None

    image_processor = None

    if 'physvlm' in model_name.lower() or 'llava' in model_name.lower() or 'qwen' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        depth_tower = model.get_depth_tower()
        if not vision_tower.is_loaded:
            print("********************* not vision_tower is_loaded")
            vision_tower.load_model(device_map=device_map)
        if not depth_tower.is_loaded:
            print("********************* not depth_tower is_loaded")
            depth_tower.load_model(device_map=device_map)
        if device_map != 'auto':
            vision_tower.to(device=device_map, dtype=torch.float16)
            depth_tower.to(device=device, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len