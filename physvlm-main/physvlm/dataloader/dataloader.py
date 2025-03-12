import json
from PIL import Image
from PIL import ImageFile
import os
import copy
from typing import Dict, Sequence
from dataclasses import dataclass
import numpy as np 
import random

import torch
from torch.utils.data import Dataset
import transformers

from physvlm.arguments import DataArguments
from physvlm.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_DEPTH_TOKEN, \
                            DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from physvlm import conversation as conversation_lib
from physvlm.mm_utils import tokenizer_image_token

ImageFile.LOAD_TRUNCATED_IMAGES = True



def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    """
    Preprocesses multimodal data by handling image tokens and formatting sentences accordingly.

    Args:
        sources (Sequence[str]): A sequence of source strings to be processed.
        data_args (DataArguments): Data arguments containing multimodal settings.

    Returns:
        Dict: Processed sources with appropriate image token handling.
    """
    is_multimodal = data_args.is_multimodal

    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if isinstance(sentence["value"], list):
                sentence["value"] = str(sentence["value"])

            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token2 = DEFAULT_DEPTH_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = f"{DEFAULT_IM_START_TOKEN}{replace_token}{DEFAULT_IM_END_TOKEN}"
                replace_token2 = f"{DEFAULT_IM_START_TOKEN}{replace_token2}{DEFAULT_IM_END_TOKEN}"
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token).replace(DEFAULT_DEPTH_TOKEN, replace_token2)

    return sources

def preprocess_qwen2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)

    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) + 1
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids) + 1
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1
                
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX  # instruction_len is before the answer

            cur_len += round_len

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    assert (conversation_lib.default_conversation.version == "qwen2"), "!!!!!!!!!!conversation_lib error!!!!!!!!!!"  # FIXME
    if conversation_lib.default_conversation.version.startswith("qwen2"):
        return preprocess_qwen2(sources, tokenizer, has_image=has_image)


class LazySupervisedDataset(Dataset):
    """
    Dataset for supervised fine-tuning.

    This dataset handles both text and image data, including depth images. It supports lazy loading of data from a JSON file,
    and processes images and text data as required for training.

    Attributes:
        IMG_TOKENS (int): Number of tokens to represent an image.
        IMG_DEPTH_TOKENS (int): Number of tokens to represent a depth image.
        list_data_dict (list): List of data samples loaded from the JSON file.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to process text data.
        data_args (DataArguments): Additional arguments for data processing.
    """
    
    IMG_TOKENS = 243
    IMG_DEPTH_TOKENS = 486

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments):
        """
        Initialize the dataset.

        Args:
            data_path (str): Path to the JSON file containing the dataset.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer to process text data.
            data_args (DataArguments): Additional arguments for data processing.
        """
        super(LazySupervisedDataset, self).__init__()
        with open(data_path, "r") as f:
            self.list_data_dict = json.load(f)
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.list_data_dict)

    @property
    def lengths(self):
        """
        Calculate the lengths of the conversations including image tokens.

        Returns:
            list: List of lengths for each sample.
        """
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = self.IMG_TOKENS*len(sample['image']) if 'image' in sample else (self.IMG_DEPTH_TOKENS*len(sample['depth']) if 'depth' in sample else 0)
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        """
        Calculate the lengths of the conversations with a sign indicating the presence of an image.

        Returns:
            list: List of signed lengths for each sample.
        """
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list
    
    @property
    def depthes(self):
        """
        Check if depth images are present in the samples.

        Returns:
            list: List indicating the presence of depth images for each sample.
        """
        return [1 if 'depth' in sample else 0 for sample in self.list_data_dict]

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        sources = [sources]  # Ensure sources is a list

        has_image = 'image' in sources[0]
        has_depth = 'depth' in sources[0]

        if has_image:
            images, depths = self._process_images(sources[0])
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
        else:
            images, depths = None, None
            sources = copy.deepcopy([e["conversations"] for e in sources])

        data_dict = preprocess(sources, self.tokenizer, has_image=has_image)
        data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # assert (has_image and not has_depth) or (not has_image and has_depth), "Either both or neither of image and depth should be present"
        
        if has_image:
            data_dict['images'] = images  # List of image tensors
        elif self.data_args.is_multimodal:
            crop_size = self.data_args.image_processor.crop_size
            data_dict['images'] = [torch.zeros(3, crop_size['height'], crop_size['width'])]
            # data_dict['images'] = None

        if has_depth:
            data_dict['depths'] = depths  # List of depth tensors
        elif self.data_args.is_multimodal:
            crop_size = self.data_args.image_processor.crop_size
            data_dict['depths'] = [torch.zeros(3, crop_size['height'], crop_size['width'])] * len(data_dict['images'])
            # data_dict['depths'] = None

        return data_dict

    
    def _random_block_mask(self, image, block_size=(16, 16), mask_ratio=0.3, mask_color=(128, 128, 128), mask_probability=1.0):
        """
        对输入的RGB图像进行随机分块掩码处理，并将掩码区域设置为灰色。

        参数:
        - image: 输入的RGB图像，Pillow Image对象
        - block_size: 每个块的大小，默认为 (16, 16)
        - mask_ratio: 掩码的比例，即被掩码的块的比例，默认为 0.3
        - mask_color: 掩码区域的颜色，默认为灰色 (128, 128, 128)
        - mask_probability: 使用掩码策略的概率，默认为 1.0（即总是使用掩码）

        返回:
        - mask_image: 掩码后的RGB图像，Pillow Image对象
        """
        # 如果随机数大于mask_probability，则不进行掩码处理，直接返回原图
        if random.random() > mask_probability:
            return image

        # 将Pillow图像转换为NumPy数组
        image_np = np.array(image)
        
        # 获取图像的高度和宽度
        H, W, C = image_np.shape
        
        # 计算块的数量
        num_blocks_x = W // block_size[1]
        num_blocks_y = H // block_size[0]
        
        # 创建一个掩码图像的副本
        mask_image_np = image_np.copy()
        
        # 生成所有块的坐标
        blocks = [(i, j) for i in range(num_blocks_y) for j in range(num_blocks_x)]
        
        # 随机选择需要掩码的块数量
        num_mask_blocks = int(mask_ratio * len(blocks))
        mask_blocks = random.sample(blocks, num_mask_blocks)
        
        # 对选中的块进行掩码处理
        for block_y, block_x in mask_blocks:
            y_start = block_y * block_size[0]
            y_end = y_start + block_size[0]
            x_start = block_x * block_size[1]
            x_end = x_start + block_size[1]
            
            # 将这个块的像素值设置为指定的灰色
            mask_image_np[y_start:y_end, x_start:x_end, :] = mask_color
        
        # 将NumPy数组转换回Pillow图像
        mask_image = Image.fromarray(mask_image_np)
        
        return mask_image
    
    def _process_images(self, source):
        image_folder = self.data_args.image_folder

        # Ensure images are in a list
        image_paths = source['image']
        if not isinstance(image_paths, list):
            image_paths = [image_paths]

        images = []

        for image_file in image_paths:
            img = Image.open(os.path.join(image_folder, image_file)).convert('RGB')

            if 'depth' in source:
                # Apply random block mask if needed
                img = self._random_block_mask(img, mask_probability=self.data_args.rgb_mask_probability)
            
            # Preprocess the image
            if self.data_args.image_aspect_ratio == 'pad':
                img = self._expand2square(img, tuple(int(x * 255) for x in self.data_args.image_processor.image_mean))
                img = self.data_args.image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0]
            else:
                img = self.data_args.image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0]

            images.append(img)

        depths = None
        if 'depth' in source:
            # Ensure depths are in a list
            depth_paths = source['depth']
            if not isinstance(depth_paths, list):
                depth_paths = [depth_paths]
            depths = []

            for depth_file in depth_paths:
                depth_img = Image.open(os.path.join(image_folder, depth_file)).convert('RGB')

                # Apply random block mask if needed
                depth_img = self._random_block_mask(depth_img, mask_probability=self.data_args.depth_mask_probability)

                # Preprocess the depth image
                if self.data_args.image_aspect_ratio == 'pad':
                    depth_img = self._expand2square(depth_img, tuple(int(x * 255) for x in self.data_args.image_processor.image_mean))
                    depth_img = self.data_args.image_processor.preprocess(depth_img, return_tensors='pt')['pixel_values'][0]
                else:
                    depth_img = self.data_args.image_processor.preprocess(depth_img, return_tensors='pt')['pixel_values'][0]

                depths.append(depth_img)

        return images, depths
            

    def _expand2square(self, pil_img, background_color):
        """
        Expand an image to a square by padding with a background color.

        Args:
            pil_img (PIL.Image.Image): Image to expand.
            background_color (tuple): Background color to use for padding.

        Returns:
            PIL.Image.Image: Expanded image.
        """
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result
            

def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Preprocesses data based on the conversation version specified in the default conversation settings.

    Args:
        sources (Sequence[str]): A sequence of source strings to be processed.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to be used for tokenizing the input.
        has_image (bool): Flag indicating if the data contains images.

    Returns:
        Dict: Processed input IDs and labels.
    """
    assert conversation_lib.default_conversation.version in ["qwen2"], \
        f"!!!!!Unsupported conversation version: {conversation_lib.default_conversation.version}"
    if conversation_lib.default_conversation.version == "qwen2":
        return preprocess_qwen2(sources, tokenizer, has_image=has_image)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """
    Collate examples for supervised fine-tuning.

    This collator handles padding of input sequences and labels, and manages image and depth image data if present.

    Attributes:
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to process text data.
    """

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of instances.

        Args:
            instances (Sequence[Dict]): List of instances to collate.

        Returns:
            Dict[str, torch.Tensor]: Collated batch.
        """
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        for tmp in input_ids:
            if len(tmp) > self.tokenizer.model_max_length:
                print(f"Warning: Input length is: {len(tmp)}. Cut to {self.tokenizer.model_max_length}.")
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        
        if 'images' in instances[0]:
            # Collect images lists
            images_list = [instance['images'] for instance in instances]
            # Find the maximum number of images in a sample
            max_num_images = max(len(imgs) for imgs in images_list)
            # Pad images to have same number per sample
            padded_images = []
            for imgs in images_list:
                num_imgs = len(imgs)
                if num_imgs < max_num_images:
                    # Create padding images of zeros
                    pad_img = torch.zeros_like(imgs[0])
                    imgs.extend([pad_img] * (max_num_images - num_imgs))
                padded_images.append(torch.stack(imgs))
            batch['images'] = torch.stack(padded_images)  # Shape: [batch_size, max_num_images, C, H, W]
            
        if 'depths' in instances[0]:
            # Collect depths lists
            depths_list = [instance['depths'] for instance in instances]
            # Find the maximum number of depths in a sample
            max_num_depths = max(len(deps) for deps in depths_list)
            # Pad depths to have same number per sample
            padded_depths = []
            for deps in depths_list:
                num_deps = len(deps)
                if num_deps < max_num_depths:
                    # Create padding depths of zeros
                    pad_deps = torch.zeros_like(deps[0])
                    deps.extend([pad_img] * (max_num_depths - num_deps))
                padded_depths.append(torch.stack(deps))
            batch['depths'] = torch.stack(padded_depths)  # Shape: [batch_size, max_num_depths, C, H, W]

        return batch