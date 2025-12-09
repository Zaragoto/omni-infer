import asyncio
import json
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Awaitable, Iterable
from functools import cache, lru_cache, partial
from pathlib import Path
from typing import (Any, Callable, Generic, Literal, Optional, TypeVar, Union,
                    cast)

import jinja2.nodes
import transformers.utils.chat_template_utils as hf_chat_utils
# yapf conflicts with isort for this block
# yapf: disable
from openai.types.chat import (ChatCompletionAssistantMessageParam,
                               ChatCompletionContentPartImageParam,
                               ChatCompletionContentPartInputAudioParam)
from openai.types.chat import (
    ChatCompletionContentPartParam as OpenAIChatCompletionContentPartParam)
from openai.types.chat import (ChatCompletionContentPartRefusalParam,
                               ChatCompletionContentPartTextParam)
from openai.types.chat import (
    ChatCompletionMessageParam as OpenAIChatCompletionMessageParam)
from openai.types.chat import (ChatCompletionMessageToolCallParam,
                               ChatCompletionToolMessageParam)
from openai.types.chat.chat_completion_content_part_input_audio_param import (
    InputAudio)
from pydantic import TypeAdapter
# yapf: enable
from transformers import (PreTrainedTokenizer, PreTrainedTokenizerFast,
                          ProcessorMixin)
# pydantic needs the TypedDict from typing_extensions
from typing_extensions import Required, TypeAlias, TypedDict

from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalDataDict
from vllm.multimodal.utils import MediaConnector
# yapf: disable
from vllm.transformers_utils.chat_templates import (
    get_chat_template_fallback_path)
# yapf: enable
from vllm.transformers_utils.processor import cached_get_processor
from vllm.transformers_utils.tokenizer import AnyTokenizer, MistralTokenizer
from vllm.utils import deprecate_kwargs, random_uuid

ModalityStr = Literal["image", "audio", "video", "image_embeds"]
def _placeholder_str_add_pangu(self, modality: ModalityStr,
                         current_count: int) -> Optional[str]:
        # TODO: Let user specify how to insert image tokens into prompt
        # (similar to chat template)
    hf_config = self._model_config.hf_config
    model_type = hf_config.model_type

    if modality in ("image", "image_embeds"):
        if model_type == "chatglm":
            return "<|begin_of_image|><|endoftext|><|end_of_image|>"
        if model_type in ("phi3_v", "phi4mm"):
            return f"<|image_{current_count}|>"
        if model_type in ("minicpmo", "minicpmv"):
            return "(<image>./</image>)"
        if model_type in ("blip-2", "florence2", "fuyu", "paligemma",
                              "pixtral", "mistral3"):
                # These models do not use image tokens in the prompt
            return None
        if model_type == "qwen":
            return f"Picture {current_count}: <img></img>"
        if model_type.startswith("llava"):
            return self._cached_token_str(self._tokenizer,
                                              hf_config.image_token_index)

        if model_type in ("aya_vision", "chameleon", "deepseek_vl_v2",
                              "internvl_chat", "ovis", "skywork_chat",
                              "NVLM_D", "h2ovl_chat", "idefics3", "smolvlm"):
            return "<image>"
        if model_type in ("mllama", "llama4"):
            return "<|image|>"
        if model_type in ("qwen2_vl", "qwen2_5_vl"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if model_type == "qwen2_5_omni":
            return "<|vision_start|><|IMAGE|><|vision_end|>"
        if model_type == "molmo":
            return ""
        if model_type == "aria":
            return "<|fim_prefix|><|img|><|fim_suffix|>"
        if model_type == "gemma3":
            return "<start_of_image>"
        if model_type == "kimi_vl":
            return "<|media_start|>image<|media_content|><|media_pad|><|media_end|>" # noqa: E501
        if model_type == "pangu_v5_vl":
            return ""
        if model_type == "openpangu_vl":
            return ""
        if model_type == "openpangu_omni":
            return ""
        if model_type == "pangu_v5_omni":
            return ""

        raise TypeError(f"Unknown {modality} model type: {model_type}")
    elif modality == "audio":
        if model_type in ("ultravox", "granite_speech"):
            return "<|audio|>"
        if model_type == "phi4mm":
            return f"<|audio_{current_count}|>"
        if model_type in ("qwen2_audio", "qwen2_5_omni"):
            return (f"Audio {current_count}: "
                        f"<|audio_bos|><|AUDIO|><|audio_eos|>")
        if model_type == "minicpmo":
            return "(<audio>./</audio>)"
        raise TypeError(f"Unknown model type: {model_type}")
    elif modality == "video":
        if model_type == "internvl_chat":
            return "<video>"
        if model_type in ("qwen2_vl", "qwen2_5_vl"):
            return "<|vision_start|><|video_pad|><|vision_end|>"
        if model_type == "qwen2_5_omni":
            return "<|vision_start|><|VIDEO|><|vision_end|>"
        if model_type in ("minicpmo", "minicpmv"):
            return "(<video>./</video>)"
        if model_type.startswith("llava"):
            return self._cached_token_str(self._tokenizer,
                                              hf_config.video_token_index)
        raise TypeError(f"Unknown {modality} model type: {model_type}")
    else:
        raise TypeError(f"Unknown modality: {modality}")

