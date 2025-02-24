from typing import Dict, Sequence
import copy

import torch
import transformers

from physvlm.arguments import DataArguments
from physvlm.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, \
                            DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from physvlm import conversation as conversation_lib
from physvlm.mm_utils import tokenizer_image_token



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
    is_interleaved = data_args.is_interleaved
    only_visual = data_args.only_visual

    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value'] and not is_interleaved:
                image_count = sentence['value'].count(DEFAULT_IMAGE_TOKEN)
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()

                if image_count == 1:
                    sentence['value'] = f"{DEFAULT_IMAGE_TOKEN}\n{sentence['value']}".strip()
                elif image_count == 2:
                    if not only_visual:
                        sentence['value'] = f"{DEFAULT_IMAGE_TOKEN}\n{DEFAULT_IMAGE_TOKEN}\n{sentence['value']}".strip()
                    else:
                        sentence['value'] = f"{DEFAULT_IMAGE_TOKEN}\n{sentence['value']}".strip()

                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, f'<Image>{DEFAULT_IMAGE_TOKEN}</Image>')

            if isinstance(sentence["value"], list):
                sentence["value"] = str(sentence["value"])

            if only_visual and is_interleaved:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = f"{DEFAULT_IMAGE_TOKEN}\n{sentence['value']}".strip()

            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = f"{DEFAULT_IM_START_TOKEN}{replace_token}{DEFAULT_IM_END_TOKEN}"
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_common_style(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    conv,
    roles: Dict[str, str],
    has_image: bool,
    sep_style: str, # MPT or TWO
    sep,
    sep2=None
) -> Dict:
    """
    Preprocesses data in a common style for different conversation formats.

    Args:
        sources (Sequence[str]): A sequence of source strings to be processed.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to be used for tokenizing the input.
        conv: Conversation object containing conversation settings.
        roles (Dict[str, str]): Dictionary mapping roles to their respective identifiers.
        has_image (bool): Flag indicating if the data contains images.
        sep_style (str): Separator style, either "MPT" or "TWO".
        sep: Separator token.
        sep2: Secondary separator token, if applicable.

    Returns:
        Dict: Processed input IDs and labels.
    """
    conversations = []

    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        if sep_style == "MPT":
            rounds = conversation.split(sep)
            re_rounds = [sep.join(rounds[:3])]
            for conv_idx in range(3, len(rounds), 2):
                re_rounds.append(sep.join(rounds[conv_idx:conv_idx + 2]))
        elif sep_style == "TWO":
            rounds = conversation.split(sep2)
        else:
            raise ValueError(f"Unsupported separator style: {sep_style}")

        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds if sep_style == "TWO" else re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) + 1
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
            else:
                round_len = len(tokenizer(rou).input_ids) + 1
                instruction_len = len(tokenizer(parts[0]).input_ids)

            if sep2 is None:
                if i > 0:
                    round_len -= 1
                    instruction_len -= 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

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


def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Preprocesses data for the LLaMA3 conversation format.

    Args:
        sources (Sequence[str]): A sequence of source strings to be processed.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to be used for tokenizing the input.
        has_image (bool): Flag indicating if the data contains images.

    Returns:
        Dict: Processed input IDs and labels.
    """
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    sep = conv.sep + conv.roles[1]
    return preprocess_common_style(sources, tokenizer, conv, roles, has_image, "MPT", sep)


def preprocess_qwen2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Preprocesses data for the Qwen2 conversation format.

    Args:
        sources (Sequence[str]): A sequence of source strings to be processed.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to be used for tokenizing the input.
        has_image (bool): Flag indicating if the data contains images.

    Returns:
        Dict: Processed input IDs and labels.
    """
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    sep = conv.sep + conv.roles[1] + ": "
    sep2 = conv.sep2
    return preprocess_common_style(sources, tokenizer, conv, roles, has_image, "TWO", sep, sep2)


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Preprocesses plain data by concatenating source strings and handling image tokens.

    Args:
        sources (Sequence[str]): A sequence of source strings to be processed.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to be used for tokenizing the input.

    Returns:
        Dict: Processed input IDs and labels.
    """
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


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
    assert conversation_lib.default_conversation.version in ["plain", "llama3", "qwen2"], \
        f"!!!!!Unsupported conversation version: {conversation_lib.default_conversation.version}"
    
    if conversation_lib.default_conversation.version == "plain":
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.version == "llama3":
        return preprocess_llama3(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "qwen2":
        return preprocess_qwen2(sources, tokenizer, has_image=has_image)
