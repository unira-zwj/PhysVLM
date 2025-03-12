#
# physvlm: Embodied Large Multi-modal Model with Self-Constraints
#    2024/05/20
# Modified from LLaVA: https://github.com/haotian-liu/LLaVA.git
#
# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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


import pathlib
from typing import Dict, Optional
import os
import logging

import torch
import transformers
import tokenizers

from physvlm.train.llava_trainer import LLaVATrainer
from physvlm import conversation as conversation_lib
from physvlm.model import LlavaLlamaForCausalLM, LlavaQwen2ForCausalLM
from physvlm.arguments import ModelArguments, DataArguments, TrainingArguments, LoraArguments
from physvlm.dataloader.dataloader import LazySupervisedDataset, DataCollatorForSupervisedDataset
from physvlm.train.utils import (safe_save_model_for_hf_trainer, \
                                            smart_tokenizer_and_embedding_resize, \
                                            unlock_vit,
                                            set_model_config,
                                            find_all_linear_names,
                                            get_peft_state_maybe_zero_3,
                                            get_peft_state_non_lora_maybe_zero_3)

# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_flash_sdp(False)
# torch.backends.cuda.enable_math_sdp(True)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

local_rank = None
def rank0_print(*args):
    """Prints messages only if the local rank is 0."""
    if local_rank == 0:
        logger.info(" ".join(map(str, args)))


from packaging import version

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, 
                                data_args: DataArguments) -> Dict[str, Optional[LazySupervisedDataset]]:
    try:
        train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
    except Exception as e:
        logger.error(f"Error in make_supervised_data_module: {e}")
        raise


def initialize_model(model_args: ModelArguments, 
                     training_args: TrainingArguments, 
                     attn_implementation: Optional[str]) -> transformers.PreTrainedModel:
    try:
        if "llama" in model_args.model_name_or_path.lower():
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            )
        elif "qwen" in model_args.model_name_or_path.lower():
            model = LlavaQwen2ForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            )
        return model
    except Exception as e:
        logger.error(f"Error in initialize_model: {e}")
        raise


def configure_special_tokens(model, model_args, tokenizer):
    try:
        if model_args.version == "llama3":
            if tokenizer.pad_token is None:
                smart_tokenizer_and_embedding_resize(special_tokens_dict=dict(pad_token="<pad>"), tokenizer=tokenizer, model=model)
        elif model_args.version == "qwen2":
            tokenizer.add_special_tokens({'unk_token': '<|extra_0>|', 'eos_token': '<|endoftext>|'})
            tokenizer.pad_token = tokenizer.unk_token
        else:
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.unk_token
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    except Exception as e:
        logger.error(f"Error in configure_special_tokens: {e}")
        raise


def train(attn_implementation=None):
    global local_rank

    try:
        parser = transformers.HfArgumentParser((ModelArguments, 
                                                DataArguments, 
                                                TrainingArguments, 
                                                LoraArguments))
        model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()
        local_rank = training_args.local_rank

        assert (model_args.vision_tower is not None and model_args.depth_tower is not None), "!!!!! visual tower or depth tower miss !!!!!!"
        assert (model_args.version in ["plain", "llama3", "qwen2"]), "!!!!! model_args.version must be plain, llama3 or qwen2!!!!!"
        model = initialize_model(model_args, training_args, attn_implementation)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
        
        configure_special_tokens(model, model_args, tokenizer)
        
        model.config.use_cache = False

        if training_args.gradient_checkpointing:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
                
        if lora_args.lora_enable:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=lora_args.lora_r,
                lora_alpha=lora_args.lora_alpha,
                target_modules=find_all_linear_names(model),
                lora_dropout=lora_args.lora_dropout,
                bias=lora_args.lora_bias,
                task_type="CAUSAL_LM",
            )
            if lora_args.bits == 16:
                if training_args.bf16:
                    model.to(torch.bfloat16)
                if training_args.fp16:
                    model.to(torch.float16)
            rank0_print("Adding LoRA adapters...")
            model = get_peft_model(model, lora_config)

        if model_args.vision_tower is not None:
            model.get_model().initialize_vision_modules(
                model_args=model_args,
                fsdp=training_args.fsdp
            )

            vision_tower = model.get_vision_tower()
            vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

            depth_tower = model.get_depth_tower()
            depth_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

            data_args.image_processor = vision_tower.image_processor
            data_args.is_multimodal = True

            set_model_config(model, tokenizer, data_args, model_args, training_args)

            rank0_print(f'*************** training stage: {training_args.train_stage} ***************')
            if training_args.train_stage == 1:
                model.config.freeze_llm = True
                model.config.freeze_vision_encoder = False
                model.config.freeze_vision_mlp = False
                model.config.freeze_depth_encoder = True
                model.config.freeze_depth_mlp = True
            elif training_args.train_stage == 2:
                model.config.freeze_llm = True
                model.config.freeze_vision_encoder = True
                model.config.freeze_vision_mlp = True
                model.config.freeze_depth_encoder = False
                model.config.freeze_depth_mlp = False
            elif training_args.train_stage == 3:
                model.config.freeze_llm = False
                model.config.freeze_vision_encoder = False
                model.config.freeze_vision_mlp = False
                model.config.freeze_depth_encoder = False
                model.config.freeze_depth_mlp = False

            model.requires_grad_(False)
            if not model.config.freeze_llm:
                rank0_print('*************** unfreeze llm ***************')
                model.model.requires_grad_(True)
            if not model.config.freeze_vision_encoder:
                rank0_print('*************** unfreeze vision_encoder ***************')
                unlock_vit(training_args, model_args, vision_tower)
            if not model.config.freeze_depth_encoder:
                rank0_print('*************** unfreeze depth_encoder ***************')
                unlock_vit(training_args, model_args, depth_tower)
            if not model.config.freeze_vision_mlp:
                rank0_print('*************** unfreeze vision_mlp ***************')
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = True
            if not model.config.freeze_depth_mlp:
                rank0_print('*************** unfreeze depth_mlp ***************')
                for p in model.get_model().mm_depth_projector.parameters():
                    p.requires_grad = True
                for p in model.get_model().mm_share_projector.parameters():
                    p.requires_grad = True
                for p in model.get_model().sequence_compressor.parameters():
                    p.requires_grad = True

            model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

            training_args.use_im_start_end = model_args.mm_use_im_start_end

        if lora_args.bits in [4, 8]:
            from peft.tuners.lora import LoraLayer
            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    if training_args.bf16:
                        module = module.to(torch.bfloat16)
                if 'norm' in name:
                    module = module.to(torch.float32)
                if 'lm_head' in name or 'embed_tokens' in name:
                    if hasattr(module, 'weight'):
                        if training_args.bf16 and module.weight.dtype == torch.float32:
                            module = module.to(torch.bfloat16)

        total_params = sum(p.numel() for p in model.parameters())
        rank0_print(f"Total parameters: {total_params}")
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        rank0_print(f"Trainable parameters: {trainable_params}")

        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
        trainer = LLaVATrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
        trainer.save_state()
        model.config.use_cache = True

        if lora_args.lora_enable:
            state_dict = get_peft_state_maybe_zero_3(
                model.named_parameters(), lora_args.lora_bias
            )
            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                model.named_parameters()
            )
            if training_args.local_rank == 0 or training_args.local_rank == -1:
                model.config.save_pretrained(training_args.output_dir)
                model.save_pretrained(training_args.output_dir, state_dict=state_dict)
                torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
        else:
            safe_save_model_for_hf_trainer(trainer=trainer,
                                        output_dir=training_args.output_dir)
            
    except Exception as e:
        logger.error(f"Error in train function: {e}")
        raise


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}")
        raise
