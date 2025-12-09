# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.33.2/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Rotary Positional Embeddings."""
from typing import Any, Dict, Optional, Tuple, Union, List, Literal
import math
import torch_npu
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.platforms import current_platform
from vllm.distributed import get_pp_group
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding as GPURotaryEmbedding
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding as GPUMRotaryEmbedding
from vllm.model_executor.layers.rotary_embedding import DynamicNTKScalingRotaryEmbedding
from vllm.model_executor.layers.rotary_embedding import YaRNScalingRotaryEmbedding as GPUYaRNScalingRotaryEmbedding
from vllm.model_executor.layers.rotary_embedding import DeepseekScalingRotaryEmbedding as DeepseekScalingRotaryEmbeddingGPU
from vllm.model_executor.layers.rotary_embedding import (_yarn_find_correction_dim,
                                            _apply_rotary_emb_torch,
                                            _yarn_find_correction_range,
                                            _yarn_linear_ramp_mask,
                                            _yarn_get_mscale,
                                            _rotate_neox,
                                            _rotate_gptj)
from omni.layers.utils import ConditionalTNGScope
from omni.models.config_loader.loader import model_extra_config

SCALE_FACTOR = 8
LOW_FREQ_FACTOR = 1
HIGH_FREQ_FACTOR = 4
OLD_CONTEXT_LEN = 8192
ROPE_ROTARY_FACTOR = 64

NEOX_ROTARY_COEFF = 2


class RotaryEmbeddingTorchNpu(torch.nn.Module):
    _compute_inv_freq = GPURotaryEmbedding._compute_inv_freq

    def __init__(self,
                 head_size: int,
                 rotary_dim: int,
                 max_position_embeddings: int = 2048,
                 base: int = 10000,
                 is_neox_style: bool = False,
                 dtype: torch.dtype = None,
                 q_hidden_size=8192):
        super().__init__()
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.max_len = self.max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.rotary_coeff = NEOX_ROTARY_COEFF if is_neox_style else rotary_dim

        self.head_size = head_size
        self.cos, self.sin = self._compute_cos_sin_cache()
        self.cache_cos = None
        self.cache_sin = None
        self.cache_pos_shape = None

        cache = self._compute_cos_sin_cache_alt()
        cache = cache.to(dtype)
        self.cos_sin_cache: torch.Tensor
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_cos_sin_cache_alt(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base).npu()
        t = torch.arange(self.max_len, device=inv_freq.device, dtype=inv_freq.dtype)
        # Adapt: adapt for ascend rope
        if self.is_neox_style:
            freqs = torch.einsum("i,j->ij", t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
        else:
            freqs = torch.outer(t, inv_freq).float()
            emb = torch.stack((freqs, freqs), dim=-1)
        cos = torch.cos(emb).to(dtype=torch.get_default_dtype())
        sin = torch.sin(emb).to(dtype=torch.get_default_dtype())

        return cos, sin
        # Adapt end.
    
    def get_cos_sin(self, positions: torch.Tensor, offsets: Optional[torch.Tensor] = None):
        positions = torch.add(positions, offsets) if offsets is not None else positions
        cos = self.cos[positions].view(-1, 1, 1, self.cos.shape[-1]) # bnsd
        sin = self.sin[positions].view(-1, 1, 1, self.sin.shape[-1])
        return cos, sin

    # use small ops
    def apply_rotary_pos_emb(self, x, cos, sin):
        x1, x2 = torch.chunk(x, 2, -1)
        x_new = torch.cat((-x2, x1), dim=-1)
        output = cos * x + sin * x_new
        return output

    def apply_rotary_emb_torch(
            self,
            x: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
            is_neox_style: bool,
    ) -> torch.Tensor:
        cos = cos.unsqueeze(-2).to(x.dtype)
        sin = sin.unsqueeze(-2).to(x.dtype)
        if is_neox_style:
            x1, x2 = torch.chunk(x, 2, dim=-1)
        else:
            x1 = x[..., ::2]
            x2 = x[..., 1::2]
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        if is_neox_style:
            return torch.cat((o1, o2), dim=-1)
        else:
            return torch.stack((o1, o2), dim=-1).flatten(-2)

    def _forward_native(self, position_ids, query, key):
        position = position_ids.flatten()
        num_tokens = position_ids.shape[0]
        cos_sin = self.cos_sin_cache.index_select(0, position_ids)
        cos, sin = cos_sin.chunk(2, dim=-1)
        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., :self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        query_rot = self.apply_rotary_emb_torch(query_rot, cos, sin, self.is_neox_style)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)
        if key is not None:
            key_shape = key.shape
            key = key.view(num_tokens, -1, self.head_size)
            key_rot = key[..., :self.rotary_dim]
            key_pass = key[..., self.rotary_dim:]
            key_rot = self.apply_rotary_emb_torch(key_rot, cos, sin, self.is_neox_style)
            key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

    # use ascend_ops to deal with torch_npu.npu_apply_rotary_pos_emb last dim is not 128 bug
    def _forward_ascend_ops_and_small_ops(self, position_ids, query, key):
        cos = torch.index_select(self.cos, dim=0, index=position_ids)
        sin = torch.index_select(self.sin, dim=0, index=position_ids)
        query = query.view(*query.shape[:-1], -1, self.head_size).contiguous()
        key = key.view(*key.shape[:-1], -1, self.head_size).contiguous()
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
        q_embed = self.apply_rotary_pos_emb(query, cos, sin)
        k_embed = self.apply_rotary_pos_emb(key, cos, sin)
        return q_embed.flatten(-2), k_embed.flatten(-2)

    # use torch_npu fused ops
    def _forward_fused_ops(self, position_ids, query, key, cos=None, sin=None):
        # adapt to TND format
        if cos is None or sin is None:
            cos = torch.index_select(self.cos, dim=0, index=position_ids.view(-1)).unsqueeze(1)
            sin = torch.index_select(self.sin, dim=0, index=position_ids.view(-1)).unsqueeze(1)
        else:
            cos = cos.squeeze(2)
            sin = sin.squeeze(2)
        # head_dim use class variable, repair head_dim convert to symbol in dynamo
        query = query.view(*query.shape[:-1], -1, self.head_size).contiguous()
        key = key.view(*key.shape[:-1], -1, self.head_size).contiguous()

        # npu_apply_rotary_pos_emb replace npu_rotary_mul, npu_rotary_mul will not support muti batch size
        q_embed, k_embed = torch_npu.npu_apply_rotary_pos_emb(query, key, cos, sin, 'TND')

        # Flatten results
        q_embed_flat = q_embed.flatten(1,2)
        k_embed_flat = k_embed.flatten(1,2)

        return q_embed_flat, k_embed_flat


    def forward(self, position_ids, query, key, cos=None, sin=None):
        # adapt chatglm : dim = head_size / 2
        if self.rotary_dim < self.head_size:
            q_embed, k_embed = self._forward_native(position_ids, query, key)
        elif self.rotary_dim != 128:
            # use ascend_ops to deal with torch_npu.npu_apply_rotary_pos_emb last dim is not 128 bug
            q_embed, k_embed = self._forward_ascend_ops_and_small_ops(position_ids, query, key)
        else:
            q_embed, k_embed = self._forward_fused_ops(position_ids, query, key, cos, sin)
        return q_embed, k_embed


class YaRNScalingRotaryEmbedding(RotaryEmbeddingTorchNpu):
    """RotaryEmbedding extended with YaRN method.

    Credits to Peng et al. github.com/jquesnelle/yarn
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: torch.dtype,
        *,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        q_hidden_size: int = 8192,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        # Get n-d magnitude scaling corrected for interpolation
        self.mscale = float(
            _yarn_get_mscale(self.scaling_factor) * self.attn_factor)
        super().__init__(head_size, rotary_dim, max_position_embeddings, base,
                         is_neox_style, dtype, q_hidden_size)

    _compute_inv_freq = GPUYaRNScalingRotaryEmbedding._compute_inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.scaling_factor).npu()
        self.max_len = self.max_position_embeddings * self.scaling_factor
        t = torch.arange(self.max_len, device=inv_freq.device, dtype=inv_freq.dtype)
        # Adapt: adapt for ascend rope
        if self.is_neox_style:
            freqs = torch.einsum("i,j->ij", t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
        else:
            freqs = torch.outer(t, inv_freq).float()
            emb = torch.stack((freqs, freqs), dim=-1)
            emb = emb.reshape(emb.shape[0], -1)

        emb_cos = torch.cos(emb) * self.mscale
        emb_sin = torch.sin(emb) * self.mscale
        cos = emb_cos.to(dtype=torch.get_default_dtype())
        sin = emb_sin.to(dtype=torch.get_default_dtype())
        return cos, sin
        # Adapt end.


class LinearScalingRotaryEmbedding(RotaryEmbeddingTorchNpu):
    """RotaryEmbedding extended with linear scaling.

    Credits to the Reddit user /u/kaiokendev
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: torch.dtype,
        q_hidden_size: int = 8192,
    ) -> None:
        self.scaling_factor = scaling_factor
        super().__init__(head_size, rotary_dim, max_position_embeddings, base,
                         is_neox_style, dtype, q_hidden_size)

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.base).npu()
        self.max_len = self.max_position_embeddings * self.scaling_factor
        t = torch.arange(self.max_len, device=inv_freq.device, dtype=inv_freq.dtype)
        t = t / self.scaling_factor
        # Adapt: adapt for ascend rope
        if self.is_neox_style:
            freqs = torch.einsum("i,j->ij", t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
        else:
            freqs = torch.outer(t, inv_freq).float()
            emb = torch.stack((freqs, freqs), dim=-1)
            emb = emb.reshape(emb.shape[0], -1)
        cos = torch.cos(emb).to(dtype=torch.get_default_dtype())
        sin = torch.sin(emb).to(dtype=torch.get_default_dtype())
        return cos, sin
        # Adapt end.


class ExtendedRotaryEmbedding(RotaryEmbeddingTorchNpu):

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
        inv_freqs = super()._compute_inv_freq(base)
        return self.apply_scaling(inv_freqs)

    def apply_scaling(self, freqs: torch.Tensor):
        low_freq_wavelen = OLD_CONTEXT_LEN / LOW_FREQ_FACTOR
        high_freq_wavelen = OLD_CONTEXT_LEN / HIGH_FREQ_FACTOR
        new_freqs = []
        for freq in freqs:
            wavelen = 2 * math.pi / freq
            if wavelen < high_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > low_freq_wavelen:
                new_freqs.append(freq / SCALE_FACTOR)
            else:
                smooth = (OLD_CONTEXT_LEN / wavelen - LOW_FREQ_FACTOR) / (
                    HIGH_FREQ_FACTOR - LOW_FREQ_FACTOR)
                new_freqs.append((1 - smooth) * freq / SCALE_FACTOR +
                                 smooth * freq)
        return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

class DeepseekScalingRotaryEmbedding(DeepseekScalingRotaryEmbeddingGPU):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: torch.dtype,
        *,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1,
        mscale_all_dim: float = 0,
    ) -> None:
        super().__init__(head_size, rotary_dim, max_position_embeddings, base,
                         is_neox_style, scaling_factor, dtype, extrapolation_factor=extrapolation_factor,
                         attn_factor=attn_factor,beta_fast=beta_fast,beta_slow=beta_slow,
                         mscale=mscale, mscale_all_dim=mscale_all_dim)

        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=current_platform.device_type,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        dim = self.rotary_dim

        freq_extra = 1.0 / (
            self.base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        freq_inter = 1.0 / (
            self.scaling_factor
            * self.base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

        low, high = _yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.max_position_embeddings,
        )
        inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(low, high, dim // 2, dtype=torch.float32).to(
            device=device
        )
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(seq_len * self.scaling_factor, device=device, dtype=torch.float32)

        freqs = torch.outer(t, inv_freq)

        # _mscale = float(
        #     yarn_get_mscale(self.scaling_factor, self.mscale)
        #     / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        # )

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", (emb.cos() * self.mscale).to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", (emb.sin() * self.mscale).to(dtype), persistent=False
        )

    def _compute_inv_freq(self, scaling_factor: float) -> torch.Tensor:
        pos_freqs = self.base ** (torch.arange(
            0, self.rotary_dim, 2, dtype=torch.float, device=current_platform.device_type) /
                                  self.rotary_dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

        low, high = _yarn_find_correction_range(self.beta_fast, self.beta_slow,
                                                self.rotary_dim, self.base,
                                                self.max_position_embeddings)
        # Get n-d rotational scaling corrected for extrapolation
        inv_freq_mask = (1 - _yarn_linear_ramp_mask(
            low, high, self.rotary_dim // 2,
            dtype=torch.float)) * self.extrapolation_factor
        inv_freq_mask = inv_freq_mask.npu()
        inv_freq = inv_freq_interpolation * (
                1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.scaling_factor)
        t = torch.arange(self.max_position_embeddings * self.scaling_factor,
                         device=inv_freq.device,
                         dtype=torch.float32)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = (freqs.cos() * self.mscale)
        sin = (freqs.sin() * self.mscale)
        if self.is_neox_style:
            cos = cos.repeat(1, 2)
            sin = sin.repeat(1, 2)
        else:
            cos = cos.repeat_interleave(2, dim=-1)
            sin = sin.repeat_interleave(2, dim=-1)
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def get_cos_sin(self, positions: torch.Tensor, offsets: Optional[torch.Tensor] = None):
        positions = torch.add(positions, offsets) if offsets is not None else positions
        cos = self.cos_cached[positions].view(-1, 1, 1, self.cos_cached.shape[-1])
        sin = self.sin_cached[positions].view(-1, 1, 1, self.sin_cached.shape[-1])
        return cos, sin

    def forward(
            self,
            positions: torch.Tensor,
            query: torch.Tensor,
            key: torch.Tensor,
            offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
        # adapt use split_rope_cat when deepseek_yarn rope rotary_dim = 64
        bs, _, hidden_size = query.shape

        cos_sin = self.cos_sin_cache[torch.add(positions, offsets)
        if offsets is not None else positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        # Adapt: adapt cos and sin shape
        cos = cos.view(-1, 1, cos.shape[-1])
        sin = sin.view(-1, 1, sin.shape[-1])
        # Adapt end.
        rotate_fn = _rotate_neox if self.is_neox_style else _rotate_gptj
        query_rot = query * cos + rotate_fn(query) * sin
        if key is not None:
            key_rot = key * cos + rotate_fn(key) * sin

        query = query_rot
        key = key_rot
        return query, key


class QwenRotaryEmbedding(torch.nn.Module):

    def __init__(self,
                 head_size: int,
                 rotary_dim: int,
                 max_position_embeddings: int = 2048,
                 base: int = 10000,
                 is_neox_style: bool = True,
                 dtype: torch.dtype = None):
        super().__init__()
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.max_len = self.max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style

        self.head_size = head_size
        cos, sin = QwenRotaryEmbedding.compute_full_cos_sin(self.base, self.rotary_dim, self.max_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @staticmethod
    def compute_full_cos_sin(base: Union[int, float], rotary_dim: int, max_len: int) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """Compute the cos and sin cache."""
        inv_freq = QwenRotaryEmbedding.compute_inv_freq(base, rotary_dim)
        t = torch.arange(max_len, device=inv_freq.device, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = torch.cos(emb).to(dtype=torch.get_default_dtype())
        sin = torch.sin(emb).to(dtype=torch.get_default_dtype())

        return cos, sin

    @staticmethod
    def compute_inv_freq(base: Union[int, float], rotary_dim: int) -> torch.Tensor:
        """Compute the inverse frequency."""
        inv_freq = 1.0 / (base ** (torch.arange(
            0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        return inv_freq

    # use small ops
    def apply_rotary_pos_emb(self, x, cos, sin):
        x1, x2 = torch.chunk(x, 2, -1)
        x_new = torch.cat((-x2, x1), dim=-1)
        output = cos * x + sin * x_new
        return output

    def get_cos_sin(self, positions: torch.Tensor, offsets: Optional[torch.Tensor] = None):
        positions = torch.add(positions, offsets) if offsets is not None else positions
        cos = self.cos[positions].view(-1, self.cos.shape[-1])
        sin = self.sin[positions].view(-1, self.sin.shape[-1])
        return cos, sin

    def forward(self, position_ids, query, key, cos, sin):
        """
        Args:
            position_ids: [num_tokens, ]
            query: [num_tokens, num_heads * head_size]
            key: [num_tokens, num_heads * head_size]
        """

        if self.rotary_dim != 128:
            query = query.view(*query.shape[:-1], -1, self.head_size).contiguous()
            key = key.view(*key.shape[:-1], -1, self.head_size).contiguous()
            cos = cos.unsqueeze(-2)
            sin = sin.unsqueeze(-2)
            q_embed = self.apply_rotary_pos_emb(query, cos, sin)
            k_embed = self.apply_rotary_pos_emb(key, cos, sin)
            q_embed = q_embed.flatten(-2)
            k_embed = k_embed.flatten(-2)
        else:
            # shape to bsnd
            cos = cos.unsqueeze(1).unsqueeze(1)
            sin = sin.unsqueeze(1).unsqueeze(1)

            query = query.view(query.shape[0], 1, -1, self.head_size)
            key = key.view(key.shape[0], 1, -1, self.head_size)

            q_embed, k_embed = torch_npu.npu_apply_rotary_pos_emb(query, key, cos, sin)

            q_embed = q_embed.view(q_embed.shape[0], -1)
            k_embed = k_embed.view(k_embed.shape[0], -1)

        return q_embed, k_embed


class QwenMRotaryEmbedding(GPUMRotaryEmbedding):
    """Rotary Embedding with Multimodal Sections."""

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """PyTorch-native implementation equivalent to forward().

        Args:
            positions:
                [num_tokens,] (text only) or
                [3, num_tokens] (T/H/W positions with multimodal inputs)
            query: [num_tokens, num_heads * head_size]
            key: [num_tokens, num_kv_heads * head_size]
        """
        assert positions.ndim == 1 or positions.ndim == 2
        assert key is not None

        num_tokens = positions.shape[-1]
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if positions.ndim == 2:
            assert self.mrope_section

            cos = torch.cat([
                m[i]
                for i, m in enumerate(cos.split(self.mrope_section, dim=-1))
            ],
                            dim=-1)
            sin = torch.cat([
                m[i]
                for i, m in enumerate(sin.split(self.mrope_section, dim=-1))
            ],
                            dim=-1)

        query_shape = query.shape
        query = query.reshape(num_tokens, -1, self.head_size)
        query_rot = query[..., :self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        query_rot = _apply_rotary_emb_torch(query_rot, cos, sin, self.is_neox_style)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_shape = key.shape
        key = key.reshape(num_tokens, -1, self.head_size)
        key_rot = key[..., :self.rotary_dim]
        key_pass = key[..., self.rotary_dim:]
        key_rot = _apply_rotary_emb_torch(key_rot, cos, sin, self.is_neox_style)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

class PanguProMoERotaryEmbedding(RotaryEmbeddingTorchNpu):
    _compute_inv_freq = GPURotaryEmbedding._compute_inv_freq

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)

    @staticmethod
    def compute_inv_freq(base: Union[int, float], rotary_dim: int) -> torch.Tensor:
        """Compute the inverse frequency."""
        inv_freq = 1.0 / (base ** (torch.arange(
            0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        return inv_freq
    
    @staticmethod
    def compute_full_cos_sin_alt(base: Union[int, float], rotary_dim: int, max_len: int) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """Compute the cos and sin cache."""
        inv_freq = RotaryEmbeddingTorchNpu.compute_inv_freq(base, rotary_dim).npu()
        t = torch.arange(max_len, device=inv_freq.device, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        return cos, sin
    
    def get_cos_sin(self, positions: torch.Tensor, offsets: Optional[torch.Tensor] = None):
        positions = torch.add(positions, offsets) if offsets is not None else positions
        cos_sin = self.cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)
        return cos, sin

    def apply_rotary_emb_torch(
            self,
            x: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
            is_neox_style: bool,
    ) -> torch.Tensor:
        cos = cos.unsqueeze(-2).to(x.dtype)
        sin = sin.unsqueeze(-2).to(x.dtype)
        if is_neox_style:
            x1, x2 = torch.chunk(x, 2, dim=-1)
        else:
            # x1 = x[..., ::2]
            # x2 = x[..., 1::2]
            x = x.view(*x.shape[:-1], x.shape[-1] // 2, 2)
            x1, x2 = x.split([1, 1], dim=-1)
            x1 = x1.squeeze(-1)
            x2 = x2.squeeze(-1)
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        if is_neox_style:
            return torch.cat((o1, o2), dim=-1)
        else:
            return torch.stack((o1, o2), dim=-1).flatten(-2)

    def _forward_native(self, position_ids, query, key, cos=None, sin=None):
        position = position_ids.flatten()
        num_tokens = position_ids.shape[0]

        if cos is None or sin is None:
            cos_sin = self.cos_sin_cache.index_select(0, position_ids)
            cos, sin = cos_sin.chunk(2, dim=-1)

        with ConditionalTNGScope(multi_stream=model_extra_config.operator_opt_config.moe_multi_stream_tune, stream_id='11'):
            query_shape = query.shape
            query = query.view(num_tokens, -1, self.head_size) # TND
            # query_rot = query[..., :self.rotary_dim]
            # query_pass = query[..., self.rotary_dim:]
            query_rot, query_pass = query.split([self.rotary_dim, query.shape[-1] - self.rotary_dim], dim=-1)
            query_rot = self.apply_rotary_emb_torch(query_rot, cos, sin, self.is_neox_style)
            query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        if key is not None:
            with ConditionalTNGScope(multi_stream=model_extra_config.operator_opt_config.moe_multi_stream_tune, stream_id='22'):
                key_shape = key.shape
                key = key.view(num_tokens, -1, self.head_size)
                # key_rot = key[..., :self.rotary_dim]
                # key_pass = key[..., self.rotary_dim:]
                key_rot, key_pass = key.split([self.rotary_dim, key.shape[-1] - self.rotary_dim], dim=-1)
                key_rot = self.apply_rotary_emb_torch(key_rot, cos, sin, self.is_neox_style)
                key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key


    def forward(self, position_ids, query, key, cos=None, sin=None):
        # adapt chatglm : dim = head_size / 2
        if self.rotary_dim < self.head_size:
            q_embed, k_embed = self._forward_native(position_ids, query, key, cos, sin)
        elif self.rotary_dim != 128:
            # use ascend_ops to deal with torch_npu.npu_apply_rotary_pos_emb last dim is not 128 bug
            q_embed, k_embed = self._forward_ascend_ops_and_small_ops(position_ids, query, key)
        else:
            q_embed, k_embed = self._forward_fused_ops(position_ids, query, key, cos, sin)
        return q_embed, k_embed

_ROPE_DICT: Dict[Tuple, nn.Module] = {}

# MRotaryEmbedding with interleaved
class MRotaryEmbeddingInterleaved(GPUMRotaryEmbedding):
    """Rotary Embedding with Multimodal Sections and Interleaved Support."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        mrope_section: Optional[List[int]] = None,
        mrope_interleaved: bool = True,
        rotary_mode: Literal["half", "interleaved"] = "half",
        num_hidden_layers_cache: int = 1
    ) -> None:
        # Enlarge max_position_embeddings for video inputs
        self.cache_max_position_num = max_position_embeddings
        super().__init__(
            head_size,
            rotary_dim,
            self.cache_max_position_num,
            base,
            is_neox_style,
            dtype,
        )

        self.mrope_section = mrope_section
        self.mrope_interleaved = mrope_interleaved
        self.rotary_mode = rotary_mode

        if self.mrope_section is None:
            raise ValueError("mrope_section cannot be None.")
        if sum(self.mrope_section) != rotary_dim // 2:
            raise ValueError("Sum of mrope_section must equal rotary_dim // 2.")
        if not self.mrope_interleaved:
            raise ValueError("mrope_interleaved must be True when mrope_section is provided.")

        # Generate interleaved indices
        if len(mrope_section) == 2:
            h_num, w_num = mrope_section[0], mrope_section[1]
            mrope_dim = self.get_mrope_interleaved_id_list(h_num, w_num, 0)
        elif len(mrope_section) == 3:
            t_num, h_num, w_num = mrope_section[0], mrope_section[1], mrope_section[2]
            mrope_dim = self.get_mrope_interleaved_id_list(t_num, h_num, w_num, force_last=True)
        else:
            raise AssertionError("Cannot support the length of mrope section is not 2 or 3.")

        mrope_dim = mrope_dim * 2
        self.mrope_dim = mrope_dim

        self.layer_cache = None
        self.layer_counts = 0
        self.num_hidden_layers_cache = num_hidden_layers_cache

    def _rebuild_pos_emb(
        self,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Interleave the rotary embedding"""
        cos_sin = self.cos_sin_cache[positions]
        mrope_section_3d = [1] * len(self.mrope_dim)
        mrope_dim = self.mrope_dim
        cos_sin = torch.cat(
            [m[mrope_dim[i]] for i, m in enumerate(cos_sin.split(mrope_section_3d, dim=-1))],
            dim=-1,
        )
        return cos_sin, torch.arange(cos_sin.shape[0], device=positions.device)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with interleaved rotary embedding."""
        if self.layer_counts % self.num_hidden_layers_cache == 0:
            cos_sin, positions = self._rebuild_pos_emb(positions)
            self.layer_cache = (cos_sin, positions)
            self.layer_counts = 0
        else:
            cos_sin, positions = self.layer_cache
        self.layer_counts += 1

        import torch_npu

        mrope_section = [0, 0, 0] if positions.ndim == 1 else self.mrope_section

        num_tokens = query.shape[0]

        cos, sin = cos_sin.chunk(2, dim=-1)

        if positions.ndim == 2:
            assert self.mrope_section

            cos = torch.cat(
                [m[i] for i, m in enumerate(cos.split(self.mrope_section, dim=-1))],
                dim=-1,
            )
            sin = torch.cat(
                [m[i] for i, m in enumerate(sin.split(self.mrope_section, dim=-1))],
                dim=-1,
            )
            

        query_shape = query.shape
        query = query.reshape(num_tokens, -1, self.head_size)
        query_rot = query[..., :self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        query_rot = _apply_rotary_emb_torch(query_rot, cos, sin, self.is_neox_style)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_shape = key.shape
        key = key.reshape(num_tokens, -1, self.head_size)
        key_rot = key[..., :self.rotary_dim]
        key_pass = key[..., self.rotary_dim:]
        key_rot = _apply_rotary_emb_torch(key_rot, cos, sin, self.is_neox_style)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)


        return query, key

    @staticmethod
    def get_mrope_interleaved_id_list(a: int, b: int, c: int, force_last: bool = False) -> List[int]:
        """
        Generate an interleaved list of indices for multi-modal rotary embedding.

        Args:
            a: Number of indices for first modality
            b: Number of indices for second modality
            c: Number of indices for third modality
            force_last: Whether to force the last element to be from the first modality

        Returns:
            List of interleaved indices
        """
        if force_last:
            a -= 1

        counts = {0: a, 1: b, 2: c}
        placed = {k: 0 for k in counts}
        rem = counts.copy()
        seq: List[int] = []
        last = None

        total = a + b + c
        for _ in range(total):
            # Candidates: remaining > 0 and ≠ last
            cands = [k for k in rem if rem[k] > 0 and k != last]
            if not cands:
                # If only last remains, relax the condition
                cands = [k for k in rem if rem[k] > 0]

            # Select the rarest candidate
            try:
                best = min(cands, key=lambda k: (placed[k] / counts[k], k))
            except KeyError:
                best = 0

            seq.append(best)
            placed[best] += 1
            rem[best] -= 1
            last = best

        if force_last:
            seq.append(0)

        return seq



def get_rope(
        head_size: int,
        rotary_dim: int,
        max_position: int,
        base: int,
        is_neox_style: bool = True,
        rope_scaling: Optional[Dict[str, Any]] = None,
        dtype: Optional[torch.dtype] = None,
        partial_rotary_factor: float = 1.0,
        dual_chunk_attention_config: Optional[dict[str, Any]] = None,
        num_hidden_layers_cache: int = 1
):
    if dtype is None:
        dtype = torch.get_default_dtype()
    if partial_rotary_factor < 1.0:
        rotary_dim = int(rotary_dim * partial_rotary_factor)

    if rope_scaling is not None:
        rope_scaling_tuple = {
            k:tuple(v) if isinstance(v, list) else v for k,v in rope_scaling.items()
        }
        rope_scaling_args = tuple(rope_scaling_tuple.items())
    else:
        rope_scaling_args = None

    if partial_rotary_factor < 1.0:
        rotary_dim = int(rotary_dim * partial_rotary_factor)

    key = (head_size, rotary_dim, max_position, base, is_neox_style,
           rope_scaling_args)
    
    if key in _ROPE_DICT:
        return _ROPE_DICT[key]
    # Adapt:
    # 1. do not support su
    # 2. support llama3.1 and deepseek_v2
    if rope_scaling is not None:
        # adapt Replacing legacy 'type' key with 'rope_type' in 0.6.3
        scaling_type = rope_scaling["rope_type"]
        if scaling_type != "su":
            if "factor" in rope_scaling:
                scaling_factor = rope_scaling["factor"]
        if scaling_type == "linear":
            rotary_emb = LinearScalingRotaryEmbedding(head_size, rotary_dim,
                                                      max_position, base,
                                                      is_neox_style,
                                                      scaling_factor, dtype)
        elif scaling_type == "yarn":
            original_max_position = rope_scaling[
                "original_max_position_embeddings"]
            extra_kwargs = {
                k: v
                for k, v in rope_scaling.items()
                if k in ("extrapolation_factor", "attn_factor", "beta_fast",
                         "beta_slow")
            }
            rotary_emb = YaRNScalingRotaryEmbedding(head_size, rotary_dim,
                                                    original_max_position,
                                                    base, is_neox_style,
                                                    scaling_factor, dtype,
                                                    **extra_kwargs)
        elif scaling_type == "deepseek_yarn":
            original_max_position = rope_scaling[
                "original_max_position_embeddings"]
            extra_kwargs = {
                k: v
                for k, v in rope_scaling.items()
                if k in ("extrapolation_factor", "attn_factor", "beta_fast",
                         "beta_slow", "mscale", "mscale_all_dim")
            }
            rotary_emb = DeepseekScalingRotaryEmbedding(
                head_size, rotary_dim, original_max_position, base,
                is_neox_style, scaling_factor, dtype, **extra_kwargs)
        elif scaling_type == "llama3":
            rotary_emb = ExtendedRotaryEmbedding(head_size, rotary_dim,
                                                     max_position, base,
                                                     is_neox_style, dtype)
        elif scaling_type == "qwen":
            if 'mrope_section' in rope_scaling:
                rotary_emb = QwenMRotaryEmbedding(
                    head_size, 
                    rotary_dim, 
                    max_position, 
                    base,
                    is_neox_style,
                    dtype,
                    mrope_section=rope_scaling["mrope_section"]
                )
            else:
                rotary_emb = QwenRotaryEmbedding(
                        head_size, 
                        rotary_dim, 
                        max_position, 
                        base,
                        is_neox_style)
        elif scaling_type == "pangu":
            if 'mrope_section' in rope_scaling:
                num_hidden_layers_cache = 1 if get_pp_group().world_size > 1 else num_hidden_layers_cache
                rotary_emb = MRotaryEmbeddingInterleaved(
                        head_size,
                        rotary_dim,
                        max_position,
                        base,
                        is_neox_style,
                        dtype,
                        mrope_section=rope_scaling["mrope_section"],
                        mrope_interleaved=True,
                        rotary_mode=rope_scaling.get("rotary_mode", "half"),
                        num_hidden_layers_cache=num_hidden_layers_cache
                    )
            else:
                rotary_emb = QwenRotaryEmbedding(
                        head_size, 
                        rotary_dim, 
                        max_position, 
                        base,
                        is_neox_style)
        elif  scaling_type == "dynamic":
            rotary_emb = DynamicNTKScalingRotaryEmbedding(
                        head_size, 
                        rotary_dim, 
                        max_position, 
                        base,
                        is_neox_style,
                        scaling_factor, 
                        dtype)
            
        elif scaling_type == "gemma_default":
            if "mrope_section" in rope_scaling:
                rotary_emb = GPUMRotaryEmbedding(
                    head_size,
                    rotary_dim,
                    max_position,
                    base,
                    is_neox_style,
                    dtype,
                    mrope_section=rope_scaling["mrope_section"],
                )
            else:
                rotary_emb = RotaryEmbeddingTorchNpu(
                    head_size,
                    rotary_dim,
                    max_position,
                    base,
                    is_neox_style,
                    dtype,
                )

        elif scaling_type == "pangu_pro_moe":
            rotary_emb = PanguProMoERotaryEmbedding(head_size, rotary_dim, max_position, base,
                                                is_neox_style)
                                                
        else:
            scaling_type = rope_scaling["type"]
            raise ValueError(f"Unknown RoPE scaling type {scaling_type}, only support linear and dynamic now")
    else:
        rotary_emb = RotaryEmbeddingTorchNpu(head_size, rotary_dim, max_position, base,
                                             is_neox_style)
    _ROPE_DICT[key] = rotary_emb
    return rotary_emb

