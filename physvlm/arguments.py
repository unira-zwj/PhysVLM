from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field
import transformers


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m", metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})
    version: Optional[str] = field(default="qwen2", metadata={"help": "Version of the model"})
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None, metadata={"help": "Path to the pretrained multimodal MLP adapter"})
    
    vision_tower: Optional[str] = field(default=None, metadata={"help": "Path to the vision tower model"})
    mm_projector_type: Optional[str] = field(default='mlp2x_gelu', metadata={"help": "Type of the multimodal projector"})
    mm_use_im_start_end: bool = field(default=False, metadata={"help": "Whether to use image start and end tokens"})
    # mm_use_im_patch_token: bool = field(default=False, metadata={"help": "Whether to use image patch tokens"})
    mm_patch_merge_type: Optional[str] = field(default='flat', metadata={"help": "Type of patch merging"})
    tune_vit_from_layer: Optional[int] = field(default=-1, metadata={"help": "Layer to start tuning the vision transformer from"})

    depth_tower: Optional[str] = field(default=None, metadata={"help": "Path to the depth tower model"})
    mm_depth_projector_type: Optional[str] = field(default='mlp2x_gelu', metadata={"help": "Type of the depth projector"})
    mm_share_projector_type: Optional[str] = field(default='mlp2x_gelu', metadata={"help": "Type of the shared projector"})
    sequence_compressor_strid: Optional[int] = field(default=-1, metadata={"help": "Stride for the sequence compressor"})
    encoder_type: Optional[str] = field(default='siglip', metadata={"help": "encoder type select, siglip or clip"})


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and evaluation.
    """
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = field(default=True, metadata={"help": "Whether to use lazy preprocessing"})
    is_multimodal: bool = field(default=True, metadata={"help": "Whether the data is multimodal"})
    is_interleaved: bool = field(default=False, metadata={"help": "Whether the data is interleaved"})
    only_visual: bool = field(default=False, metadata={"help": "Whether to use only visual data"})
    image_folder: Optional[str] = field(default=None, metadata={"help": "Path to the image folder"})
    image_aspect_ratio: str = field(default='pad', metadata={"help": "Aspect ratio of the images"})
    rgb_mask_probability: float = field(default=0.5, metadata={"help": "Probability of masking RGB images"})
    depth_mask_probability: float = field(default=0.3, metadata={"help": "Probability of masking depth images"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """
    Arguments pertaining to the training configuration.
    """
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Directory to store the cache"})
    optim: str = field(default="adamw_torch", metadata={"help": "Optimizer to use"})
    remove_unused_columns: bool = field(default=False, metadata={"help": "Whether to remove unused columns"})
    mpt_attn_impl: Optional[str] = field(default="triton", metadata={"help": "Attention implementation to use"})
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."})
    mm_projector_lr: Optional[float] = field(default=None, metadata={"help": "Learning rate for the multimodal projector"})
    group_by_modality_length: bool = field(default=True, metadata={"help": "Whether to group by modality length"})
    vision_tower_lr: Optional[float] = field(default=None, metadata={"help": "Learning rate for the vision tower"})
    train_stage: Optional[int] = field(default=-1, metadata={"help": "Train stage select"})

    freeze_llm: bool = field(default=False, metadata={"help": "Whether to freeze the llm"})
    freeze_vision_encoder: bool = field(default=False, metadata={"help": "Whether to tune the depth tower"})
    freeze_depth_encoder: bool = field(default=False, metadata={"help": "Whether to tune the vision tower"})
    freeze_vision_mlp: bool = field(default=False, metadata={"help": "Whether to freeze the vision_mlp"})
    freeze_depth_mlp: bool = field(default=False, metadata={"help": "Whether to freeze the depth_mlp"})


@dataclass
class LoraArguments:
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )