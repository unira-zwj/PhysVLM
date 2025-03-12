import logging
import os
import torch
import transformers
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

from physvlm.arguments import *


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

        
# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def maybe_zero_3(param, ignore_status=False, name=None):
    """
    Detach and clone the parameter, handling DeepSpeed Zero-3 if necessary.
    
    Args:
        param (torch.nn.Parameter): The parameter to process.
        ignore_status (bool): Whether to ignore the Zero-3 status.
        name (str): The name of the parameter (for logging purposes).
    
    Returns:
        torch.Tensor: The detached and cloned parameter.
    """
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE and not ignore_status:
            logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_state_maybe_zero_3(named_params, keys_to_match):
    """
    Get the state of parameters matching the given keys, handling DeepSpeed Zero-3 if necessary.
    
    Args:
        named_params (iterable): Iterable of named parameters.
        keys_to_match (list): List of keys to match.
    
    Returns:
        dict: Dictionary of matched parameters.
    """
    matched_params = {k: t for k, t in named_params if any(key in k for key in keys_to_match)}
    return {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in matched_params.items()}


def find_all_linear_names(model):
    """
    Find all linear module names in the model, excluding certain multimodal keywords.
    
    Args:
        model (torch.nn.Module): The model to search.
    
    Returns:
        list: List of linear module names.
    """
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'mm_depth_projector', 
                           'vision_tower', 'vision_resampler',
                           'depth_tower']
    for name, module in model.named_modules():
        if any(keyword in name for keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    lora_module_names.discard('lm_head')  # needed for 16-bit
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """
    Save the model state for Hugging Face Trainer, handling DeepSpeed Zero-3 if necessary.
    
    Args:
        trainer (transformers.Trainer): The trainer instance.
        output_dir (str): The directory to save the model.
    """
    def save_vision_tower():
        if trainer.deepspeed:
            torch.cuda.synchronize()
        if getattr(trainer.model.get_vision_tower().image_processor, 'save_pretrained', False):
            trainer.model.get_vision_tower().image_processor.save_pretrained(
                os.path.join(output_dir, 'vision_tower'))
        trainer.model.get_vision_tower().vision_tower.vision_model.config.save_pretrained(
            os.path.join(output_dir, 'vision_tower'))
        weight_to_save = get_state_maybe_zero_3(
            trainer.model.get_vision_tower().vision_tower.named_parameters(), [''])
        if trainer.args.local_rank in [0, -1]:
            torch.save(weight_to_save, 
                       os.path.join(output_dir, 'vision_tower/pytorch_model.bin'))
    
    def save_depth_tower():
        if trainer.deepspeed:
            torch.cuda.synchronize()
        if getattr(trainer.model.get_depth_tower().image_processor, 'save_pretrained', False):
            trainer.model.get_depth_tower().image_processor.save_pretrained(
                os.path.join(output_dir, 'depth_tower'))
        trainer.model.get_depth_tower().vision_tower.vision_model.config.save_pretrained(
            os.path.join(output_dir, 'depth_tower'))
        weight_to_save = get_state_maybe_zero_3(
            trainer.model.get_depth_tower().vision_tower.named_parameters(), [''])
        if trainer.args.local_rank in [0, -1]:
            torch.save(weight_to_save, 
                       os.path.join(output_dir, 'depth_tower/pytorch_model.bin'))
    
    def save_mm_mlp(keys_to_match):
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])
        weight_to_save = get_state_maybe_zero_3(trainer.model.named_parameters(), 
                                                keys_to_match)
        trainer.model.config.save_pretrained(output_dir)
        if not os.path.exists(os.path.join(output_dir, 'projector')):
            os.makedirs(os.path.join(output_dir, 'projector'))
        if trainer.args.local_rank in [0, -1]:
            torch.save(weight_to_save, 
                        os.path.join(output_dir, 'projector/mm_projector.bin'))

    if getattr(trainer.args, "train_stage", -1) == 1:
        keys_to_match = ['mm_projector', 'mm_depth_projector']
        save_mm_mlp(keys_to_match)
        print("*********************** save mm adapter mlp")
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        # save_vision_tower()
        # save_depth_tower()
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)



def unlock_vit(training_args: TrainingArguments, 
               model_args: ModelArguments, 
               vision_tower: torch.nn.Module):
    """
    Unlocks the vision transformer (ViT) for training.

    Args:
        training_args: Training arguments.
        model_args: Model arguments.
        vision_tower: The vision transformer model.
    """
    lr_of_vit = training_args.vision_tower_lr if \
        training_args.vision_tower_lr is not None and training_args.vision_tower_lr != 0 \
            else training_args.learning_rate
    for n, p in vision_tower.named_parameters():
        if model_args.tune_vit_from_layer != -1:
            if 'vision_tower.vision_model.encoder.layers.' in n:
                layer_id = int(n.split('vision_tower.vision_model.encoder.layers.')[-1].split('.')[0])
                if layer_id >= model_args.tune_vit_from_layer:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
            else:
                p.requires_grad = False
        else:
            p.requires_grad = True


def lock_vit(training_args: TrainingArguments, 
               model_args: ModelArguments, 
               vision_tower: torch.nn.Module):
    """
    Unlocks the vision transformer (ViT) for training.

    Args:
        training_args: Training arguments.
        model_args: Model arguments.
        vision_tower: The vision transformer model.
    """
    for n, p in vision_tower.named_parameters():
        p.requires_grad = False
    
    
def set_model_config(model, tokenizer, data_args, model_args, training_args):
    model.config.image_aspect_ratio = data_args.image_aspect_ratio

    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length

    model.config.freeze_llm = training_args.freeze_llm
    model.config.freeze_vision_encoder = training_args.freeze_vision_encoder
    model.config.freeze_depth_encoder = training_args.freeze_depth_encoder
    model.config.freeze_vision_mlp = training_args.freeze_vision_mlp
    model.config.freeze_depth_mlp = training_args.freeze_depth_mlp
    model.config.train_stage = training_args.train_stage
    model.config.encoder_type = model_args.encoder_type

    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_projector_lr = training_args.mm_projector_lr
    # model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.config.pad_token_id = tokenizer.pad_token_id  
    model.vision_tower_lr = training_args.vision_tower_lr
    
    