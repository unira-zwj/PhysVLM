import os
from .clip_encoder import CLIPVisionTower
from .siglip_encoder import SigLipVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    encoder_type = getattr(vision_tower_cfg, 'encoder_type', None)
    is_absolute_path_exists = os.path.exists(vision_tower)
    if encoder_type == 'siglip':
        if is_absolute_path_exists or vision_tower.startswith("google") or vision_tower.startswith('bczhou'):
            return SigLipVisionTower(vision_tower, vision_tower_cfg, **kwargs)
    elif encoder_type == 'clip':
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    raise ValueError(f'Unknown vision tower: {vision_tower}')


def build_depth_tower(depth_tower_cfg, **kwargs):
    depth_tower = getattr(depth_tower_cfg, 'mm_depth_tower', getattr(depth_tower_cfg, 'depth_tower', None))
    encoder_type = getattr(depth_tower_cfg, 'encoder_type', None)
    is_absolute_path_exists = os.path.exists(depth_tower)
    if encoder_type == 'siglip':
        if is_absolute_path_exists or depth_tower.startswith("google") or depth_tower.startswith('bczhou'):
            return SigLipVisionTower(depth_tower, depth_tower_cfg, **kwargs)
    elif encoder_type == 'clip':
        return CLIPVisionTower(depth_tower, depth_tower_cfg, **kwargs)
    raise ValueError(f'Unknown depth tower: {depth_tower}')