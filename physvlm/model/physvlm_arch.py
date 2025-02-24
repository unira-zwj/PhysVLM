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


from abc import ABC, abstractmethod
import copy
import os

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower, build_depth_tower
from .multimodal_projector.builder import build_vision_projector, build_depth_projector
from .sequence_compressor.builder import build_sequence_compressor

from physvlm.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEPTH_TOKEN_INDEX,\
                            DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from physvlm.mm_utils import get_anyres_image_grid_shape



class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        self.tokenizer = None

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=False)
            self.mm_projector = build_vision_projector(config)
            self.down_sample = build_sequence_compressor(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )
                
        if hasattr(config, "mm_depth_tower"):
            self.depth_tower = build_depth_tower(config, delay_load=False)
            self.mm_depth_projector = build_depth_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower
    
    def get_depth_tower(self):
        depth_tower = getattr(self, 'depth_tower', None)
        if type(depth_tower) is list:
            depth_tower = depth_tower[0]
        return depth_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        depth_tower = model_args.depth_tower
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower
        self.config.mm_depth_tower = depth_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()
        
        if self.get_depth_tower() is None:
            depth_tower = build_depth_tower(model_args)
            if fsdp is not None and len(fsdp) > 0:
                self.depth_tower = [depth_tower]
            else:
                self.depth_tower = depth_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                depth_tower = self.depth_tower[0]
            else:
                depth_tower = self.depth_tower
            depth_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_depth_projector_type = getattr(model_args, 'mm_depth_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True
                
        if getattr(self, 'mm_depth_projector', None) is None:
            self.mm_depth_projector = build_depth_projector(self.config)
            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_depth_projector.parameters():
                p.requires_grad = True
        
        if getattr(self, 'self.down_sample', None) is None:
            self.down_sample = build_sequence_compressor(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.self.down_sample.parameters():
                p.requires_grad = True

        # stage2 加载训练好的 projector 
        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            self.mm_depth_projector.load_state_dict(get_w(mm_projector_weights, 'mm_depth_projector'))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        vision_tower = self.get_model().get_vision_tower()
        return vision_tower
    
    def get_depth_tower(self):
        depth_tower = self.get_model().get_depth_tower()
        return depth_tower

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        return image_features
    
    def encode_depths(self, depths):
        depth_features = self.get_model().get_depth_tower()(depths)
        return depth_features
    
    def encode_images_projector(self, image_features):
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def encode_depths_projector(self, depth_features):
        depth_features = self.get_model().mm_depth_projector(depth_features)
        return depth_features

    def encode_down_sample(self, features):
        features = self.get_model().down_sample(features)
        return features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, depths, image_sizes=None
    ):
        # print(input_ids[0], input_ids.shape, attention_mask.shape, labels.shape, images.shape, depths.shape)
        # print(input_ids[0])
        vision_tower = self.get_vision_tower()
        depth_tower = self.get_depth_tower()
        if (vision_tower is None and depth_tower is None) or (images is None and depths is None) or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # 处理图像输入
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images] 
            # [bs, list_len, C, H, W]
            concat_images = torch.cat([image for image in images], dim=0) # [bs*list_len, C, H, W]
            tmp_image_features = self.encode_images(concat_images) # [bs*list_len, l, c]
            down_sample_image_features = self.encode_down_sample(tmp_image_features)
            tmp_image_projector_feature = self.encode_images_projector(down_sample_image_features)
            split_sizes = [image.shape[0] for image in images]
            # tmp_image_projector_feature = torch.split(tmp_image_projector_feature, split_sizes, dim=0) 
            # [bs, list_len, l, c]
            
            if type(depths) is list or depths.ndim == 5:
                depths = [x.unsqueeze(0) if x.ndim == 3 else x for x in depths]
            concat_depths = torch.cat([depth for depth in depths], dim=0) 
            tmp_depth_features = self.encode_depths(concat_depths)
            down_sample_depth_features = self.encode_down_sample(tmp_depth_features)
            # print(down_sample_image_features.shape, down_sample_depth_features.shape)
            cat_feature = torch.cat([down_sample_image_features, down_sample_depth_features], dim=-1)
            tmp_depth_features_projector = self.encode_depths_projector(cat_feature)
            # tmp_depth_features_projector = torch.split(tmp_depth_features_projector, split_sizes, dim=0)
            # print(tmp_depth_features_projector.shape)
        else:
            tmp_image_features = self.encode_images(images) # [mini_b, l, c] [16, 729, 1152]
            down_sample_image_features = self.encode_down_sample(tmp_image_features) # [16, 243, 1152]
            tmp_image_projector_feature = self.encode_images_projector(down_sample_image_features)  # [16, 243, 896]
            
            tmp_depth_features = self.encode_depths(depths) # [mini_b, l, c] [16, 729, 1152]
            down_sample_depth_features = self.encode_down_sample(tmp_depth_features) # [16, 243, 1152]
            cat_feature = torch.cat([down_sample_image_features, down_sample_depth_features], dim=-1) # [16, 243, 2304]
            tmp_depth_features_projector = self.encode_depths_projector(cat_feature) # [16, 243, 1152]

        # # 处理深度图像输入
        # if type(depths) is list or depths.ndim == 5:
        #     if type(depths) is list:
        #         depths = [x.unsqueeze(0) if x.ndim == 3 else x for x in depths]
        #     concat_depths = torch.cat([depth for depth in depths], dim=0) 
        #     tmp_depth_features = self.encode_depths(concat_depths)
        #     down_sample_depth_features = self.encode_down_sample(tmp_depth_features)
        #     cat_feature = torch.cat([down_sample_image_features, down_sample_depth_features], dim=-1)
        #     tmp_depth_features_projector = self.encode_depths_projector(cat_feature)
        #     tmp_depth_features_projector = torch.split(tmp_depth_features_projector, split_sizes, dim=0)
        # else:
        #     tmp_depth_features = self.encode_depths(depths) # [mini_b, l, c] [16, 729, 1152]
        #     down_sample_depth_features = self.encode_down_sample(tmp_depth_features) # [16, 243, 1152]
        #     cat_feature = torch.cat([down_sample_image_features, down_sample_depth_features], dim=-1) # [16, 243, 2304]
        #     tmp_depth_features_projector = self.encode_depths_projector(cat_feature) # [16, 243, 1152]

        # ===========================================================================
        # print(tmp_image_projector_feature.shape, tmp_depth_features_projector.shape)
        image_features = []
        depth_features = []
        # 分别保存图像和深度图像的特征
        for i, img_feature in enumerate(tmp_image_projector_feature):
            image_features.append(img_feature)
        for i, depth_feature in enumerate(tmp_depth_features_projector):
            depth_features.append(depth_feature)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'train_stage', -1) == 1 and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0  # 初始化图像特征索引
        cur_depth_idx = 0  # 初始化深度图像特征索引

        for batch_idx, cur_input_ids in enumerate(input_ids):
            # 分别计算图像和深度图像的数量
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum() # .item()
            num_depths = (cur_input_ids == DEPTH_TOKEN_INDEX).sum() # .item()

            # 如果既没有图像也没有深度图像 token，直接跳过图像和深度图像特征
            if num_images == 0 and num_depths == 0:
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                new_input_embeds.append(cur_input_embeds_1)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                cur_depth_idx += 1
                continue

            # 找到特殊 token（图像或深度图像）的索引和类型
            special_tokens = []
            for idx in torch.where(
                (cur_input_ids == IMAGE_TOKEN_INDEX) | (cur_input_ids == DEPTH_TOKEN_INDEX)
            )[0].tolist():
                token_type = 'image' if cur_input_ids[idx] == IMAGE_TOKEN_INDEX else 'depth'
                special_tokens.append((idx, token_type))

            # 按索引排序特殊 token
            special_tokens.sort(key=lambda x: x[0])

            # 提取特殊 token 的索引
            special_token_indices = [idx for idx, _ in special_tokens]

            # 包含开始和结束位置
            split_positions = [-1] + special_token_indices + [cur_input_ids.shape[0]]

            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []

            for i in range(len(split_positions) - 1):
                start_idx = split_positions[i] + 1
                end_idx = split_positions[i + 1]
                if start_idx < end_idx:
                    cur_input_ids_noim.append(cur_input_ids[start_idx:end_idx])
                    cur_labels_noim.append(cur_labels[start_idx:end_idx])

            if len(cur_input_ids_noim) > 0:
                split_sizes = [x.shape[0] for x in cur_labels_noim]
                cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
                cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            else:
                cur_input_embeds_no_im = []

            # 记录初始索引
            initial_image_idx = cur_image_idx
            initial_depth_idx = cur_depth_idx

            # 处理文本段和插入对应的特征
            cur_new_input_embeds = []
            cur_new_labels = []

            segment_count = len(split_positions) - 1
            # for x in image_features:
            #     print(x.shape)
            for i in range(segment_count):
                # 添加当前文本段的嵌入和标签
                if i < len(cur_input_embeds_no_im):
                    cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                    cur_new_labels.append(cur_labels_noim[i])

                # 检查是否存在特殊 token
                if i < len(special_tokens):
                    _, token_type = special_tokens[i]
                    if token_type == 'image':
                        # 插入图像特征
                        cur_image_features = image_features[cur_image_idx]
                        cur_image_idx += 1
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    elif token_type == 'depth':
                        # 插入深度图像特征
                        cur_depth_features = depth_features[cur_depth_idx]
                        cur_depth_idx += 1
                        cur_new_input_embeds.append(cur_depth_features)
                        cur_new_labels.append(torch.full((cur_depth_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    # print(len(image_features), cur_image_idx)

            # 在处理完一个样本后，根据 num_images 和 num_depths，跳过填充的特征
            if num_images == 0:
                # 如果没有图像占位符，但特征被填充了，需要跳过这些特征
                cur_image_idx = initial_image_idx + 1
            if num_depths == 0:
                # 如果没有深度图像占位符，但特征被填充了，需要跳过这些特征
                cur_depth_idx = initial_depth_idx + 1

            # 确保特征与设备对齐
            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            # for x in cur_new_input_embeds:
            #     print(x.shape)
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)



        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        # print(new_input_embeds.shape)
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
    

    # def initialize_vision_tokenizer(self, model_args, tokenizer):
    #     self.tokenizer = tokenizer

    #     if model_args.mm_use_im_patch_token:
    #         tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    #         self.resize_token_embeddings(len(tokenizer))

    #     if model_args.mm_use_im_start_end:
    #         num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    #         self.resize_token_embeddings(len(tokenizer))

    #         if num_new_tokens > 0:
    #             input_embeddings = self.get_input_embeddings().weight.data
    #             output_embeddings = self.get_output_embeddings().weight.data

    #             input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
    #                 dim=0, keepdim=True)
    #             output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
    #                 dim=0, keepdim=True)

    #             input_embeddings[-num_new_tokens:] = input_embeddings_avg
    #             output_embeddings[-num_new_tokens:] = output_embeddings_avg

    #         if (not self.config.freeze_depth_mlp) or (not self.config.freeze_vision_mlp):
    #             for p in self.get_input_embeddings().parameters():
    #                 p.requires_grad = True
    #             for p in self.get_output_embeddings().parameters():
    #                 p.requires_grad = False

    #         if model_args.pretrain_mm_mlp_adapter:
    #             mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
    #             embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
    #             assert num_new_tokens == 2
    #             if input_embeddings.shape == embed_tokens_weight.shape:
    #                 input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
    #             elif embed_tokens_weight.shape[0] == num_new_tokens:
    #                 input_embeddings[-num_new_tokens:] = embed_tokens_weight
    #             else:
    #                 raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
    #     elif model_args.mm_use_im_patch_token:
    #         if (not self.config.freeze_depth_mlp) or (not self.config.freeze_vision_mlp):
    #             for p in self.get_input_embeddings().parameters():
    #                 p.requires_grad = False
    #             for p in self.get_output_embeddings().parameters():
    #                 p.requires_grad = False
