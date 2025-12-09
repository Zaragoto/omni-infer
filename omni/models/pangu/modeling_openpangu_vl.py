#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Adapted from vllm/model_executor/models/qwen2_5_vl.py
# Copyright 2025 The vLLM team.
# Copyright 2025 The Qwen Team.
#
# This file is a part of the vllm-ascend project.
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

from functools import partial, lru_cache
from typing import Callable, Literal, Optional, TypedDict, Union
from collections.abc import Iterable, Mapping, Sequence
import math
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu
from einops import rearrange
from torchvision.transforms.v2 import functional

from vllm.config import VllmConfig
from vllm.distributed import parallel_state, tensor_model_parallel_all_gather, tensor_model_parallel_all_reduce
from vllm.distributed import utils as dist_utils
from vllm.model_executor.models.interfaces import (MultiModalEmbeddings, SupportsLoRA,
                         SupportsMultiModal, SupportsPP)
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.quantization.gptq import GPTQConfig
from vllm.model_executor.layers.quantization.gptq_marlin import GPTQMarlinConfig
from vllm.model_executor.layers.activation import _ACTIVATION_REGISTRY
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VLDummyInputsBuilder,
    Qwen2_5_VLMultiModalProcessor,
    Qwen2_5_VLProcessingInfo,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader 
from vllm.model_executor.models.utils import (AutoWeightsLoader, WeightsMapper,
                    init_vllm_registered_model, maybe_prefix,
                    merge_multimodal_embeddings)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.inputs import MultiModalKwargs
from vllm.multimodal.processing import PromptUpdate, PromptReplacement
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.attention import Attention, AttentionType, AttentionMetadata
from vllm.compilation.decorators import support_torch_compile

from transformers.utils import logging

from .processor_openpangu_vl import OpenPanguVLProcessor
from .pangu_dense import PanguEmbeddedForCausalLM
from omni.layers.attention.backend.attention import AscendAttentionState
from omni.layers.layernorm import RMSNorm
from omni.layers.linear import (
    RowParallelFlashCommLinear,
    QKVParallelFlashCommLinear,
    ColumnParallelFlashCommLinear,
    MergedColumnParallelFlashCommLinear)

logger = logging.get_logger(__name__)

class OpenPanguVisionAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        enable_vit_sp: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_size_per_attention_head = dist_utils.divide(
            projection_size, num_heads)
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.enable_vit_sp = enable_vit_sp #序列并行默认开启
        if self.enable_vit_sp:
            self.sp_size = parallel_state.get_tensor_model_parallel_world_size()
            self.num_attention_heads_per_partition = dist_utils.divide(
            num_heads, self.sp_size)
        else:
            self.sp_size = 1
            self.num_attention_heads_per_partition = dist_utils.divide(
            num_heads, self.tp_size)

        self.qkv = QKVParallelFlashCommLinear(
            hidden_size=embed_dim,
            head_size=self.hidden_size_per_attention_head,
            total_num_heads=num_heads,
            total_num_kv_heads=num_heads,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv")
        self.proj = RowParallelFlashCommLinear(
            input_size=projection_size,
            output_size=embed_dim,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            quant_config=quant_config,
            prefix=f"{prefix}.proj")
        self.scale_value = self.hidden_size_per_attention_head**-0.5

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        true_seq: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x = x.view(-1, x.shape[-1])
        x, bias = self.qkv(x)
        if bias is not None:
            x = x + bias

        if self.enable_vit_sp and self.sp_size > 1:
            # Transfer shape (b, s/sp, h, d) to (b, s, h/sp, d)
            x = rearrange(x, 's (b t h d) -> (b t) s h d',
            b=1,
            t=3,
            h=self.num_attention_heads_per_partition * self.sp_size)

            batch_size, shard_seqlen, head_num, head_dim = x.shape
            seq_len = shard_seqlen * self.sp_size
            shard_head_num = head_num // self.sp_size
            x = x.reshape(batch_size, shard_seqlen, self.sp_size, shard_head_num, head_dim).transpose(0,2).contiguous()
            x_all_to_all = torch.empty_like(x)
            torch.distributed.all_to_all_single(x_all_to_all, x)
            x_all_to_all = x_all_to_all.reshape(seq_len, batch_size, shard_head_num, head_dim).transpose(0,1).contiguous()
            cur_seq = x_all_to_all.shape[1]
            x_all_to_all = x_all_to_all[:, :true_seq, :, :]
            x = rearrange(x_all_to_all,
                '(b t) s h d-> s b (t h d)',
                b=1,
                t=3,
                h=self.num_attention_heads_per_partition)
        else:
            x = x.unsqueeze(1)
        
        q, k, v = x.chunk(3, dim=2)
        batch_size = q.shape[1]

        q, k, v = (rearrange(x, "s b (n d) -> b s n d", d=self.hidden_size_per_attention_head).contiguous()
                   for x in (q, k, v))
        q = torch_npu.npu_rotary_mul(q, cos, sin)
        k = torch_npu.npu_rotary_mul(k, cos, sin)

        q, k, v = [
            rearrange(x, "b s h d -> (b s) h d").contiguous()
            for x in (q, k, v)
        ]

        head_num = q.shape[1]
        actual_seq_len = tuple(cu_seqlens[1:].cpu().numpy().tolist())
        attn_out = torch_npu.npu_fusion_attention(
            q, k, v, head_num,
            scale=1.0 / math.sqrt(q.shape[-1]),
            keep_prob=1,
            input_layout="TND",
            actual_seq_qlen=actual_seq_len,
            actual_seq_kvlen=actual_seq_len,
            pre_tockens=2147483647,
            next_tockens=2147483647,
            sparse_mode=0)[0]

        if self.enable_vit_sp and self.sp_size > 1:
            # Transfer shape (s, h/sp, d) to (s/sp, h, d)
            padding = (0, 0, 0, 0, 0, cur_seq - true_seq)
            attn_out = F.pad(attn_out, padding)
            seq_len, shard_head_num, head_dim = attn_out.shape
            head_num = shard_head_num * self.sp_size
            shard_seqlen = seq_len // self.sp_size

            attn_out = attn_out.reshape(self.sp_size, shard_seqlen, shard_head_num, head_dim).transpose(1, 2).contiguous()
            attn_out_all_to_all = torch.empty_like(attn_out)
            torch.distributed.all_to_all_single(attn_out_all_to_all, attn_out)
            attn_out = attn_out_all_to_all.reshape(head_num, shard_seqlen, head_dim).transpose(0, 1).contiguous()

        attn_out = rearrange(attn_out, "(b s) h d -> (s b) (h d)", b=batch_size).contiguous()

        output, bias = self.proj(attn_out)
        if bias is not None:
            output = output + bias
        return output.unsqueeze(1)


class OpenPanguVisionMLP(nn.Module):

    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 bias: bool = False,
                 act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
                 vision_config = None,
                 tp_size: int = 1,
                 tp_rank: int = 0,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.hidden_act = vision_config.hidden_act
        ################ TODO From BF 
        if self.hidden_act == "silu":
            if hidden_features % tp_size != 0:
                hidden_features = (hidden_features + tp_size - 1) // tp_size * tp_size
            self.gate_up_proj = MergedColumnParallelFlashCommLinear(
                in_features,
                [hidden_features] * 2,
                tp_size=tp_size,
                tp_rank=tp_rank,
                bias=bias,
                quant_config=quant_config,
                prefix=f"{prefix}.gate_up_proj",)
        else:
            self.up_proj = ColumnParallelFlashCommLinear(
                in_features,
                hidden_features,
                tp_size=tp_size,
                tp_rank=tp_rank,
                bias=bias,
                quant_config=quant_config,
                prefix=f"{prefix}.up_proj")
        
        self.down_proj = RowParallelFlashCommLinear(
                            hidden_features,
                            in_features,
                            tp_size=tp_size,
                            tp_rank=tp_rank,
                            bias=bias,
                            quant_config=quant_config,
                            prefix=f"{prefix}.down_proj")
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor):
        x = x.view(-1, x.shape[-1])
        if self.hidden_act == "silu":
            x, _ = self.gate_up_proj(x)
            x = torch_npu.npu_swiglu(x)
        else:
            x, _ = self.up_proj(x)
            x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x.unsqueeze(1)


class OpenPanguVisionBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        tp_size: int = 1,
        tp_rank: int = 0,
        enable_vit_sp: bool = True,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        vision_config = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = OpenPanguVisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            tp_size=tp_size,
            tp_rank=tp_rank,
            enable_vit_sp=enable_vit_sp,
            quant_config=quant_config,
            prefix=f"{prefix}.attn")
        self.mlp = OpenPanguVisionMLP(dim,
                                     mlp_hidden_dim,
                                     act_fn=act_fn,
                                     bias=True,
                                     vision_config=vision_config,
                                     tp_size=tp_size,
                                     tp_rank=tp_rank,
                                     quant_config=quant_config,
                                     prefix=f"{prefix}.mlp")
        
    def forward(self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor], cu_seqlens: torch.Tensor, true_seq: int,
                cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        if residual is None:
            residual = hidden_states
            hidden_states = self.norm1(hidden_states)
        else:
            hidden_states, residual = self.norm1(hidden_states, residual)
        hidden_states = self.attn(hidden_states, cu_seqlens=cu_seqlens, true_seq=true_seq, cos=cos, sin=sin)

        # Fully Connected
        hidden_states, residual = self.norm2(hidden_states, residual)
        hidden_states = self.mlp(hidden_states) 
        return hidden_states, residual


class OpenPanguVisionRotaryEmbedding(nn.Module):

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.inv_freq = 1.0 / (theta**(torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self._seq_len_cached = 0
        self._freqs_cached = None

    def update_freqs_cache(self, seqlen: int) -> None:
        if seqlen > self._seq_len_cached:
            seqlen *= 2
            self._seq_len_cached = seqlen
            seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(seq, self.inv_freq)
            self._freqs_cached = freqs

    def forward(self, seqlen: int) -> torch.Tensor:
        self.update_freqs_cache(seqlen)
        return self._freqs_cached[:seqlen]


class OpenPanguVisionPatchEmbed(nn.Module):

    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.hidden_size = hidden_size
        self.input_size = self.patch_size * self.patch_size * in_channels * self.temporal_patch_size

        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = nn.Conv3d(in_channels,
                              hidden_size,
                              kernel_size=kernel_size,
                              stride=kernel_size,
                              bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_size:
            x = torch.cat([x, x], dim=-1)
        x = x.matmul(
            self.proj.weight.data.view(self.hidden_size, -1).transpose(0, 1))
        return x


class OpenPanguVisionPatchMerger(nn.Module):

    def __init__(
        self,
        d_model: int,
        context_dim: int,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        spatial_merge_size: int = 2,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: int = 1,
        tp_rank: int = 0,
        merge_parallel: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.merge_parallel = merge_parallel
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = norm_layer(context_dim)
        self.mlp = nn.ModuleList(
            [ColumnParallelFlashCommLinear(self.hidden_size,
                                        self.hidden_size,
                                        tp_size=tp_size,
                                        tp_rank=tp_rank,
                                        bias=True,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}.mlp.0"),
            nn.GELU(),
            RowParallelFlashCommLinear(self.hidden_size,
                            d_model,
                            tp_size=tp_size,
                            tp_rank=tp_rank,
                            bias=True,
                            quant_config=quant_config,
                            prefix=f"{prefix}.mlp.2")]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_q(x)
        x = x.view(-1, self.hidden_size)

        mlp_fc1, mlp_act, mlp_fc2 = self.mlp
        x, _ = mlp_fc1(x)
        x = mlp_act(x)
        if self.merge_parallel:
            out, _ = mlp_fc2(x, reduce_type=None)
        else:
            out, _ = mlp_fc2(x)
        return out


class OpenPanguVisionTransformer(nn.Module):
    def __init__(
        self,
        vision_config,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        interleaved=False,
        use_data_parallel: bool = False,
    ) -> None:
        self.use_data_parallel = use_data_parallel
        super().__init__()
        self.enable_vit_sp = True #序列并行默认使能
        if self.enable_vit_sp:
            self.tp_size = 1
            self.tp_rank = 0
        else:
            self.tp_size = parallel_state.get_tensor_model_parallel_world_size()
            self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads
        self.window_size = vision_config.window_size
        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.fullatt_block_indexes = vision_config.fullatt_block_indexes
        self.spatial_merge_unit = self.spatial_merge_size**2

        norm_layer = partial(RMSNorm, eps=norm_eps)
        self.interleaved = interleaved
        self.out_hidden_size = vision_config.out_hidden_size
        self.hidden_act = vision_config.hidden_act

        head_dim = self.hidden_size // self.num_heads
        self.rotary_pos_emb = OpenPanguVisionRotaryEmbedding(head_dim // 2)
        self.patch_embed = OpenPanguVisionPatchEmbed(
            patch_size=vision_config.patch_size,
            temporal_patch_size=vision_config.temporal_patch_size,
            in_channels=vision_config.in_channels,
            hidden_size=self.hidden_size,
        )
        self.blocks = nn.ModuleList(
            [
                OpenPanguVisionBlock(
                    dim=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_hidden_dim=vision_config.intermediate_size,
                    act_fn=_ACTIVATION_REGISTRY[vision_config.hidden_act], ############ TODO From BF
                    tp_size=self.tp_size,
                    tp_rank=self.tp_rank,
                    enable_vit_sp=self.enable_vit_sp,
                    vision_config=vision_config,
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                    prefix=f"{prefix}.blocks.{layer_idx}",
                )
                for layer_idx in range(vision_config.depth)
            ]
        )
        self.hidden_size_per_attention_head = dist_utils.divide(self.hidden_size, self.num_heads)

        self.select_layer = getattr(vision_config, "mm_unit_vision_select_layer", [-1, -3])
        self.select_index = [vision_config.depth + i for i in self.select_layer]
        self.select_index = self.select_index[::-1]
        self.select_layer = [-1 * (i + 1) for i in range(len(self.select_index))]
       
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()
        self.local_merger = None
        merge_parallel = False
        merge_tp_size = self.tp_size
        merge_tp_rank = self.tp_rank
        self.num_merger = len(self.select_layer)
        if self.world_size % self.num_merger == 0 and not self.enable_vit_sp:
            merge_parallel = True
            merge_tp_size = self.world_size // self.num_merger
            merge_tp_rank = self.rank % merge_tp_size

        
        self.take_indices = self.select_index

        self.final_layernorm = RMSNorm(self.hidden_size, eps=norm_eps)
        self.merger = nn.ModuleList(
            [
                OpenPanguVisionPatchMerger(
                    d_model=vision_config.out_hidden_size,
                    context_dim=self.hidden_size,
                    norm_layer=norm_layer,
                    spatial_merge_size=self.spatial_merge_size,
                    quant_config=quant_config,
                    tp_size = merge_tp_size,
                    tp_rank = merge_tp_rank,
                    merge_parallel = merge_parallel,
                    prefix=f"{prefix}.merger",
                )
                for i in range(len(self.select_layer))
            ]
        )
        if merge_parallel:
            self.merger_idx = self.rank // self.tp_size
            self.local_merger = self.merger[self.merger_idx]
            
    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def cal_cos_sin(self, rotary_pos_emb):
        cos = rotary_pos_emb.cos()
        sin = rotary_pos_emb.sin()

        if not self.interleaved:
            cos_new = torch.cat((cos, cos), dim=-1)
            sin_new = torch.cat((sin, sin), dim=-1)
        else:
            cos_new = rearrange(torch.stack((cos, cos), dim=-1), "... d two -> ...(d two)", two=2)
            sin_new = rearrange(torch.stack((sin, sin), dim=-1), "... d two -> ...(d two)", two=2)
        cos_new = cos_new.reshape(1, -1, 1, self.hidden_size_per_attention_head)
        sin_new = sin_new.reshape(1, -1, 1, self.hidden_size_per_attention_head)
        return cos_new, sin_new

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        see https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_vl/modular_qwen2_5_vl.py#L209 # noqa: E501
        for details.
        """
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            hpos_ids = (
                hpos_ids.reshape(
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                .permute(0, 2, 1, 3)
                .flatten()
            )
            wpos_ids = (
                wpos_ids.reshape(
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                .permute(0, 2, 1, 3)
                .flatten()
            )
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb


    def get_window_index(self, grid_thw):
        """
        see https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_vl/modular_qwen2_5_vl.py#L238 # noqa: E501
        for details.
        """
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h = grid_h // self.spatial_merge_size
            llm_grid_w = grid_w // self.spatial_merge_size
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)
        return window_index, cu_window_seqlens

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        seq_len, _ = x.size()
        rotary_pos_emb = []
        window_index: list = []
        cu_window_seqlens: list = [torch.tensor([0], dtype=torch.int32)]
        cu_seqlens: list = []

        hidden_states = x.to(device=self.device, dtype=self.dtype)
        hidden_states = self.patch_embed(hidden_states)

        window_index_id = 0
        cu_window_seqlens_last = 0
        for t, h, w in grid_thw:
            t, h, w = int(t), int(h), int(w)
            llm_h = h // self.spatial_merge_size
            llm_w = w // self.spatial_merge_size

            (
                rotary_pos_emb_thw,
                window_index_thw,
                cu_seqlens_window_thw,
                cu_seqlens_thw,
            ) = self.get_rope_by_thw(t, h, w)

            window_index.append(window_index_thw + window_index_id)
            window_index_id += (t * llm_h * llm_w)

            cu_seqlens_window_thw = (cu_seqlens_window_thw +
                                     cu_window_seqlens_last)
            cu_window_seqlens_last = cu_seqlens_window_thw[-1]
            cu_window_seqlens.append(cu_seqlens_window_thw)

            rotary_pos_emb.append(rotary_pos_emb_thw)

            cu_seqlens.append(cu_seqlens_thw)

        rotary_pos_emb = torch.cat(rotary_pos_emb)
        window_index = torch.cat(window_index)
        cu_window_seqlens = torch.cat(cu_window_seqlens)
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
        cu_seqlens = torch.cat(cu_seqlens)
        cu_seqlens = torch.cumsum(cu_seqlens, dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), "constant", 0)

        cu_seqlens = cu_seqlens.to(device=self.device, non_blocking=True)
        cu_window_seqlens = cu_window_seqlens.to(device=self.device,
                                                 non_blocking=True)
        window_index = window_index.to(device=hidden_states.device,
                                       non_blocking=True)

        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)

        cos, sin = self.cal_cos_sin(rotary_pos_emb)
        if self.enable_vit_sp and self.world_size > 1:
            merge_size = self.spatial_merge_size**2
            padding_size = math.ceil(math.ceil(seq_len / self.world_size) / merge_size) * merge_size *  self.world_size - seq_len
            if padding_size > 0:
                padding = torch.zeros(padding_size,
                                    *hidden_states.size()[1:],
                                    dtype=hidden_states.dtype,
                                    device=hidden_states.device)
                hidden_states = torch.cat([hidden_states, padding], dim=0)
            hidden_states = hidden_states.chunk(self.world_size, dim=0)[self.rank]
        hidden_states = hidden_states.unsqueeze(1)
        res = None
        
        if self.local_merger:
            #enable merger parallel
            for layer_num, blk in enumerate(self.blocks):
                if layer_num in self.fullatt_block_indexes:
                    cu_seqlens_now = cu_seqlens
                else:
                    cu_seqlens_now = cu_window_seqlens
                hidden_states, res = blk(hidden_states, res, cu_seqlens=cu_seqlens_now, true_seq=seq_len, cos=cos, sin=sin)
                if layer_num == self.take_indices[self.select_layer[self.merger_idx]]:
                    local_feather, _ = self.final_layernorm(hidden_states, res)
            hidden_states = self.local_merger(local_feather)
            if self.world_size > 1:
                hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        else:
            intermediates = []
            for layer_num, blk in enumerate(self.blocks):
                if layer_num in self.fullatt_block_indexes:
                    cu_seqlens_now = cu_seqlens
                else:
                    cu_seqlens_now = cu_window_seqlens
                hidden_states, res = blk(hidden_states, res, cu_seqlens=cu_seqlens_now, true_seq=seq_len, cos=cos, sin=sin)
                if layer_num in self.take_indices:
                    ln_hs, _ = self.final_layernorm(hidden_states, res)
                    intermediates.append(ln_hs)
        
            image_embeddings_list = []
            for idx, sl in enumerate(self.select_layer):
                image_embeddings_list.append(self.merger[idx](intermediates[sl]))
            hidden_states = sum(image_embeddings_list)

        if self.enable_vit_sp and self.world_size > 1:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, dim=0)
            if padding_size:
                hidden_states = hidden_states[:-padding_size // merge_size]
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]
        return hidden_states

    def rotary_pos_emb_thw(self, t, h, w):
        hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
        wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
        hpos_ids = hpos_ids.reshape(
            h // self.spatial_merge_size,
            self.spatial_merge_size,
            w // self.spatial_merge_size,
            self.spatial_merge_size,
        ).permute(0, 2, 1, 3).flatten()
        wpos_ids = wpos_ids.reshape(
            h // self.spatial_merge_size,
            self.spatial_merge_size,
            w // self.spatial_merge_size,
            self.spatial_merge_size,
        ).permute(0, 2, 1, 3).flatten()
        pos_ids = torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1)
        max_size = max(h, w)
        rotary_pos_emb_full = self.rotary_pos_emb(max_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            rotary_pos_emb.shape[0] // self.spatial_merge_unit,
            self.spatial_merge_unit, -1)

        return rotary_pos_emb

    def get_window_index_thw(self, grid_t, grid_h, grid_w):
        vit_merger_window_size = (self.window_size //
                                  self.spatial_merge_size // self.patch_size)

        llm_grid_h = grid_h // self.spatial_merge_size
        llm_grid_w = grid_w // self.spatial_merge_size
        index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
            grid_t, llm_grid_h, llm_grid_w)
        pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
        pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
        num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
        num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
        index_padded = F.pad(index, (0, pad_w, 0, pad_h), 'constant', -100)
        index_padded = index_padded.reshape(grid_t, num_windows_h,
                                            vit_merger_window_size,
                                            num_windows_w,
                                            vit_merger_window_size)
        index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
            grid_t, num_windows_h * num_windows_w, vit_merger_window_size,
            vit_merger_window_size)
        seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
        index_padded = index_padded.reshape(-1)
        index_new = index_padded[index_padded != -100]
        cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit
        cu_seqlens_tmp = cu_seqlens_tmp.to(dtype=torch.int32)
        cu_seqlens_tmp = torch.unique_consecutive(cu_seqlens_tmp)

        return index_new, cu_seqlens_tmp

    @lru_cache(maxsize=1024)  # noqa: B019
    def get_rope_by_thw(self, t, h, w):
        window_index_thw, cu_seqlens_window_thw = self.get_window_index_thw(
            t, h, w)
        rotary_pos_emb_thw = self.rotary_pos_emb_thw(t, h, w)
        rotary_pos_emb_thw = rotary_pos_emb_thw[window_index_thw, :, :]
        rotary_pos_emb_thw = rotary_pos_emb_thw.flatten(start_dim=0, end_dim=1)
        cu_seqlens_thw = torch.repeat_interleave(
            torch.tensor([h * w], dtype=torch.int32), t)
        return (rotary_pos_emb_thw, window_index_thw, cu_seqlens_window_thw,
                cu_seqlens_thw)

    def load_weights(self, weights) -> set[str]:
        def _padding_weight(name: str, w: torch.Tensor) -> torch.Tensor:
            if "gate_proj" in name or "up_proj" in name:
                dim, size = 0, w.size(0)
            elif "down_proj" in name:
                dim, size = 1, w.size(-1)
            else:
                return w
            pad_len = -size % self.tp_size
            if pad_len == 0:
                return w
            pad = [0] * (w.ndim * 2)
            pad[-(dim + 1) * 2 + 1] = pad_len
            return F.pad(w, pad, mode='constant', value=0)
        stacked_params_mapping = [
            ("attn.qkv.", "attn.q.", "q"),
            ("attn.qkv.", "attn.k.", "k"),
            ("attn.qkv.", "attn.v.", "v"),
        ]
        if self.hidden_act == "silu":
            stacked_params_mapping.extend([
                ("gate_up_proj", "gate_proj", 0),
                ("gate_up_proj", "up_proj", 1),
            ])
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if self.hidden_act == "silu":
                loaded_weight = _padding_weight(name, loaded_weight)
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

    def _get_tp_group(self) -> None:
        if not self.use_data_parallel:
            return parallel_state.get_tp_group()

        world_size = torch.distributed.get_world_size()
        tensor_model_parallel_size = 1
        group_ranks = torch.arange(world_size).view(-1, tensor_model_parallel_size).unbind(0)
        group_ranks = [x.tolist() for x in group_ranks]

        # creates tp process group containing only a subset of gpu ranks
        local_rank = parallel_state.get_world_group().local_rank
        tp_backend = torch.distributed.get_backend(parallel_state.get_tp_group().device_group)
        return parallel_state.init_model_parallel_group(group_ranks, local_rank, tp_backend)


class ProjectionSingle(nn.Module):
    def __init__(self, i_hidden_size: int, t_hidden_size: int):
        super().__init__()
        self.act = F.silu
        self.fc1 = nn.Linear(i_hidden_size, t_hidden_size, bias=True)

    def forward(self, hidden_states):
        x = self.act(hidden_states)
        return self.fc1(x)


class OpenPanguVLProcessingInfo(Qwen2_5_VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.model_config.hf_config

    def get_hf_processor(
        self,
        *,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        size: Optional[dict[str, int]] = None,
        fps: Optional[Union[float, list[float]]] = None,
        **kwargs: object,
    ):
        if fps is not None:
            kwargs["fps"] = fps

        return self.ctx.get_hf_processor(
            OpenPanguVLProcessor,
            image_processor=self.get_image_processor(
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                size=size,
                use_fast=kwargs.get("use_fast", True),
                do_rescale=False,
                do_normalize=False
            ),
            **kwargs,
        )


def get_load_balance_assignment(
    sizes: list[int],
    num_gpus: int = 2,
) -> tuple[list[int], list[int], list[int]]:
    """
    see https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/vision.py#L253 for details.
    """

    n_samples = len(sizes)

    # Handle edge cases
    if n_samples == 0:
        return [], [0] * num_gpus, [0] * num_gpus

    # Use greedy algorithm - balance by total size, not sample count
    gpu_assignments = [list[int]() for _ in range(num_gpus)]
    gpu_loads = [0] * num_gpus  # This tracks total SIZE, not sample count

    # Sort indices by size (largest first for better load balancing)
    large_to_small_indices = sorted(range(n_samples), key=lambda i: sizes[i], reverse=True)

    for idx in large_to_small_indices:
        # Find GPU with minimum current load (by total size)
        min_gpu = min(range(num_gpus), key=lambda i: gpu_loads[i])
        gpu_assignments[min_gpu].append(idx)
        gpu_loads[min_gpu] += sizes[idx]

    # Create shuffle indices and counts
    shuffle_indices = list[int]()
    gpu_sample_counts = list[int]()
    for gpu_id in range(num_gpus):
        shuffle_indices.extend(gpu_assignments[gpu_id])
        gpu_sample_counts.append(len(gpu_assignments[gpu_id]))

    return (shuffle_indices, gpu_sample_counts, gpu_loads)


def run_dp_sharded_mrope_vision_model(
    vision_model: torch.nn.Module,
    pixel_values: torch.Tensor,
    grid_thw_list,
    *,
    rope_type: Literal["rope_3d", "rope_2d"],
) -> tuple[torch.Tensor, ...]:
    """
    https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/vision.py#L322 for details.
    """
    grid_thw_list = grid_thw_list.tolist()
    tp_size = parallel_state.get_tensor_model_parallel_world_size()

    tp_rank_local = parallel_state.get_tensor_model_parallel_rank()

    patches_per_image = [math.prod(grid_thw) for grid_thw in grid_thw_list]
    cum_patches_per_image = [0, *itertools.accumulate(patches_per_image)]

    # Get load balancing assignment with all metadata
    (image_to_tp_rank, gpu_sample_counts, grouped_pixel_values_len) = get_load_balance_assignment(
        patches_per_image, tp_size
    )

    cum_gpu_sample_counts = [0, *itertools.accumulate(gpu_sample_counts)]

    image_idxs_local = image_to_tp_rank[cum_gpu_sample_counts[tp_rank_local] : cum_gpu_sample_counts[tp_rank_local + 1]]

    # Get the pixel values for the local images based on the image_idxs_local
    if len(image_idxs_local) > 0:
        pixel_values_local = torch.cat(
            [pixel_values[cum_patches_per_image[i] : cum_patches_per_image[i + 1]] for i in image_idxs_local]
        )
    else:
        # Handle case where this rank has no images
        pixel_values_local = torch.empty(
            (0, pixel_values.shape[1]),
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        )
    if rope_type == "rope_2d":
        embed_dim_reduction_factor = vision_model.merge_kernel_size[0] * vision_model.merge_kernel_size[1]
    else:
        embed_dim_reduction_factor = vision_model.spatial_merge_size * vision_model.spatial_merge_size

    # Find the max length across all ranks
    # The output embedding of every DP rank has to be
    # padded to this length for tensor_model_parallel_all_gather
    # to work
    max_len_per_rank = max(grouped_pixel_values_len) // embed_dim_reduction_factor
    local_grid_thw_list = [grid_thw_list[i] for i in image_idxs_local]

    # Run the vision model on the local pixel_values_local
    if rope_type == "rope_2d":
        if pixel_values_local.shape[0] > 0:
            image_embeds_local = vision_model(pixel_values_local, torch.tensor(local_grid_thw_list))
            if isinstance(image_embeds_local, list):
                image_embeds_local = torch.cat(image_embeds_local, dim=0)
        else:
            out_dim = getattr(vision_model.config, "hidden_size", None)
            image_embeds_local = torch.empty(
                (0, embed_dim_reduction_factor, out_dim),
                device=pixel_values.device,
                dtype=pixel_values.dtype,
            )
    else:
        if pixel_values_local.shape[0] > 0:
            image_embeds_local = vision_model(pixel_values_local, torch.tensor(local_grid_thw_list))
        else:
            # Handle empty case
            image_embeds_local = torch.empty(
                (0, vision_model.out_hidden_size),
                device=pixel_values.device,
                dtype=pixel_values.dtype,
            )

    # Pad the output based on max_len_per_rank
    # for tensor_model_parallel_all_gather to work
    current_len = image_embeds_local.shape[0]
    if current_len < max_len_per_rank:
        padding_size = max_len_per_rank - current_len
        if rope_type == "rope_2d":
            padding = torch.empty(
                (
                    padding_size,
                    image_embeds_local.shape[1],
                    image_embeds_local.shape[2],
                ),
                dtype=image_embeds_local.dtype,
                device=image_embeds_local.device,
            )
        else:
            padding = torch.empty(
                (padding_size, image_embeds_local.shape[1]),
                dtype=image_embeds_local.dtype,
                device=image_embeds_local.device,
            )
        image_embeds_local_padded = torch.cat([image_embeds_local, padding], dim=0)
    else:
        image_embeds_local_padded = image_embeds_local

    # Do all_gather to collect embeddings from all ranks
    gathered_embeds = tensor_model_parallel_all_gather(image_embeds_local_padded, dim=0)

    # Remove padding and reconstruct per-rank embeddings
    rank_embeddings = list[torch.Tensor]()
    for rank in range(tp_size):
        start_idx = rank * max_len_per_rank
        end_idx = start_idx + (grouped_pixel_values_len[rank] // embed_dim_reduction_factor)
        rank_embeddings.append(gathered_embeds[start_idx:end_idx])

    patches_per_output_image = [(patch_size // embed_dim_reduction_factor) for patch_size in patches_per_image]

    # Reconstruct embeddings in the original order
    original_order_embeddings = [None] * len(grid_thw_list)
    current_idx = 0
    for rank in range(tp_size):
        count = gpu_sample_counts[rank]
        if count > 0:
            # Get images assigned to this rank in shuffled order
            rank_images = image_to_tp_rank[current_idx : current_idx + count]

            rank_embed = rank_embeddings[rank]
            # Split rank embeddings back to individual images
            embed_start = 0
            for img_idx in rank_images:
                img_patches = patches_per_output_image[img_idx]
                original_order_embeddings[img_idx] = rank_embed[embed_start : embed_start + img_patches]
                embed_start += img_patches
            current_idx += count
    out_embeddings = tuple(embed for embed in original_order_embeddings if embed is not None)
    if len(out_embeddings) != len(original_order_embeddings):
        raise ValueError("Found unassigned embeddings")

    return torch.concat(out_embeddings)


class OpenPanguVLImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor


class OpenPanguVLImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    image_embeds: torch.Tensor
    image_grid_thw: torch.Tensor


class OpenPanguVLVideoPixelInputs(TypedDict):
    type: Literal["pixel_values_videos"]
    pixel_values_videos: torch.Tensor
    video_grid_thw: torch.Tensor
    second_per_grid_ts: torch.Tensor


class OpenPanguVLVideoEmbeddingInputs(TypedDict):
    type: Literal["video_embeds"]
    video_embeds: torch.Tensor
    video_grid_thw: torch.Tensor


class OpenPanguVLMultiModalProcessor(Qwen2_5_VLMultiModalProcessor):

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, any],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_processor = self.info.get_image_processor(
            **hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()
        image_token = hf_processor.image_token
        video_token = hf_processor.video_token
        vision_start_token = hf_processor.vision_start_token
        vision_end_token = hf_processor.vision_end_token
        image_token_id = vocab[image_token]
        video_token_id = vocab[video_token]
        vision_start_token_id = vocab[vision_start_token]
        vision_end_token_id = vocab[vision_end_token]
        placeholder = {
            "image": image_token_id,
            "video": video_token_id,
        }

        merge_length = image_processor.merge_size**2

        def get_replacement_pangu_v5_vision(item_idx: int, modality: str):
            grid_thw = out_mm_kwargs[f"{modality}_grid_thw"][item_idx]
            assert isinstance(grid_thw, torch.Tensor)
            if modality == "video":
                grid_t, grid_h, grid_w = grid_thw
                video_seq_length_per_time = (grid_h * grid_w).item() // merge_length
                token_id_per_time = [vision_start_token_id] + [video_token_id] * video_seq_length_per_time + \
                                    [vision_end_token_id]
                total_token_id = token_id_per_time * grid_t
                return total_token_id[1:-1]
            token_id = image_token_id
            num_tokens = int(grid_thw.prod()) // merge_length
            return [token_id] * num_tokens

        return [
            PromptReplacement(
                modality=modality,
                target=[placeholder[modality]],
                replacement=partial(get_replacement_pangu_v5_vision,
                                    modality=modality),
            ) for modality in ("image", "video")
        ]


class OpenPanguVLDummyInputsBuilder(Qwen2_5_VLDummyInputsBuilder):
    pass

@support_torch_compile
@MULTIMODAL_REGISTRY.register_processor(
    OpenPanguVLMultiModalProcessor,
    info=OpenPanguVLProcessingInfo,
    dummy_inputs=OpenPanguVLDummyInputsBuilder,
)
class OpenPanguVLForConditionalGeneration(nn.Module, SupportsMultiModal,
                                         SupportsLoRA, SupportsPP):

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "language_model.model.",
            "model.visual.": "visual.",
            "lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
        })

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):

        super().__init__()
        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        # self.use_data_parallel = vllm_config.parallel_config.enable_multimodal_encoder_data_parallel
        self.use_data_parallel = False
        self.config = config
        self.multimodal_config = multimodal_config
        quant_config = vllm_config.quant_config
        self.visual = OpenPanguVisionTransformer(
            vision_config=config.vision_config,
            norm_eps=getattr(config.vision_config, "rms_norm_eps", 1e-6),
            quant_config=self._maybe_ignore_quant_config(quant_config),
            prefix=maybe_prefix(prefix, "visual"),
            use_data_parallel=self.use_data_parallel,
        )
        self.visual.vision_projection = ProjectionSingle(config.vision_config.out_hidden_size, config.hidden_size)

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix("openpangu", "language_model"),
            architectures=["PanguEmbeddedForCausalLM"],
        )
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors
        self._image_post_process_config(config.vision_config, vllm_config.model_config)
    
    def _image_post_process_config(self, vision_config, model_config):
        processor = MULTIMODAL_REGISTRY.create_processor(model_config)
        self.channel = vision_config.in_channels
        self.patch_size = vision_config.patch_size
        self.temporal_patch_size = vision_config.temporal_patch_size
        self.do_rescale = True
        self.do_normalize = True
        self.rescale_factor = processor.info.get_hf_processor().image_processor.rescale_factor
        self.image_mean = tuple(processor.info.get_hf_processor().image_processor.image_mean)
        self.image_std = tuple(processor.info.get_hf_processor().image_processor.image_std)

    def _maybe_ignore_quant_config(self, quant_config: QuantizationConfig):
            if isinstance(quant_config, (GPTQConfig, GPTQMarlinConfig)):
                return None
            return quant_config

    def _validate_and_reshape_mm_tensor(self, mm_input: object,
                                    name: str) -> torch.Tensor:
        if not isinstance(mm_input, (torch.Tensor, list)):
            raise ValueError(f"Incorrect type of {name}. "
                                f"Got type: {type(mm_input)}")
        if isinstance(mm_input, torch.Tensor):
            if mm_input.ndim == 2:
                return mm_input
            if mm_input.ndim != 3:
                raise ValueError(f"{name} should be 2D or batched 3D tensor. "
                                    f"Got ndim: {mm_input.ndim} "
                                    f"(shape={mm_input.shape})")
            return torch.concat(list(mm_input))
        else:
            return torch.concat(mm_input)
        
    def _parse_and_validate_image_input(self, **kwargs: object):
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            pixel_values = self._validate_and_reshape_mm_tensor(
                pixel_values, "image pixel values")
            image_grid_thw = self._validate_and_reshape_mm_tensor(
                image_grid_thw, "image grid_thw")

            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image pixel values. "
                                 f"Got type: {type(pixel_values)}")

            return OpenPanguVLImagePixelInputs(type="pixel_values",
                                              pixel_values=pixel_values,
                                              image_grid_thw=image_grid_thw)

        if image_embeds is not None:
            image_embeds = self._validate_and_reshape_mm_tensor(
                image_embeds, "image embeds")
            image_grid_thw = self._validate_and_reshape_mm_tensor(
                image_grid_thw, "image grid_thw")

            if not isinstance(image_embeds, torch.Tensor):
                raise ValueError("Incorrect type of image embeddings. "
                                 f"Got type: {type(image_embeds)}")
            return OpenPanguVLImageEmbeddingInputs(
                type="image_embeds",
                image_embeds=image_embeds,
                image_grid_thw=image_grid_thw)
        
    def _parse_and_validate_video_input(self, **kwargs: object):
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_embeds = kwargs.pop("video_embeds", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        second_per_grid_ts = kwargs.pop("second_per_grid_ts", None)

        if pixel_values_videos is None and video_embeds is None:
            return None

        if pixel_values_videos is not None:
            pixel_values_videos = self._validate_and_reshape_mm_tensor(
                pixel_values_videos, "video pixel values")
            video_grid_thw = self._validate_and_reshape_mm_tensor(
                video_grid_thw, "video grid_thw")

            return OpenPanguVLVideoPixelInputs(
                type="pixel_values_videos",
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
            )

        if video_embeds is not None:
            video_embeds = self._validate_and_reshape_mm_tensor(
                video_embeds, "video embeds")
            video_grid_thw = self._validate_and_reshape_mm_tensor(
                video_grid_thw, "video grid_thw")

            if not isinstance(video_embeds, torch.Tensor):
                raise ValueError("Incorrect type of video embeddings. "
                                 f"Got type: {type(video_embeds)}")
            return OpenPanguVLVideoEmbeddingInputs(
                type="video_embeds",
                video_embeds=video_embeds,
                video_grid_thw=video_grid_thw)
        
    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        mm_input_by_modality = {}
        for input_key in kwargs:
            if input_key in ("pixel_values", "image_embeds"
                             ) and "image" not in mm_input_by_modality:
                mm_input_by_modality[
                    "image"] = self._parse_and_validate_image_input(**kwargs)
            if input_key in ("pixel_values_videos", "video_embeds"
                             ) and "video" not in mm_input_by_modality:
                mm_input_by_modality[
                    "video"] = self._parse_and_validate_video_input(**kwargs)
        return mm_input_by_modality
    
    def get_language_model(self) -> torch.nn.Module:
        return self.language_model
    
    def get_multimodal_embeddings(
            self, **kwargs: object) -> Optional[MultiModalEmbeddings]:

        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(
            **kwargs)
        if not mm_input_by_modality:
            return None

        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "image":
                vision_embeddings = self._process_image_input(multimodal_input)
                multimodal_embeddings += vision_embeddings
            if modality == "video":
                video_embeddings = self._process_video_input(multimodal_input)
                multimodal_embeddings += video_embeddings
        return multimodal_embeddings
    
    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                [self.config.image_token_id, self.config.video_token_id])
        return inputs_embeds

    def get_input_embeddings_v0(
        self,
        input_ids: torch.Tensor,
        image_input = None,
        video_input = None,
    ) -> torch.Tensor:
        inputs_embeds = self.get_input_embeddings(input_ids)
        if image_input is not None:
            image_embeds = self._process_image_input(image_input)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                image_embeds,
                placeholder_token_id=self.config.image_token_id,
            )

        if video_input is not None:
            video_embeds = self._process_video_input(video_input)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                video_embeds,
                placeholder_token_id=self.config.video_token_id,
            )
        return inputs_embeds
    
    @lru_cache(maxsize=10)
    def _fuse_mean_std_and_rescale_factor(self, do_normalize, image_mean, image_std, do_rescale, rescale_factor, device):
        if do_rescale and do_normalize:
            image_mean = torch.tensor(image_mean, device=device) * (1.0 / rescale_factor)
            image_std = torch.tensor(image_std, device=device) * (1.0 / rescale_factor)
            do_rescale = False
        return image_mean, image_std, do_rescale
    
    def rescale_and_normalize(self, images, do_rescale, rescale_factor, do_normalize, image_mean, image_std):
        """
        Rescale and normalize images.
        """
        image_mean, image_std, do_rescale = self._fuse_mean_std_and_rescale_factor(
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            device=images.device
        )
        # if/elif as we use fused rescale and normalize if both are set to True
        if do_normalize:
            origin_dtype = images.dtype
            images = functional.normalize(images.to(torch.float32), image_mean, image_std).to(origin_dtype)
        elif do_rescale:
            images = images * rescale_factor
        return images

    def _process_image_input(self, image_input) -> tuple[torch.Tensor, ...]:
        grid_thw = image_input["image_grid_thw"]
        if grid_thw.ndim != 2:
            raise ValueError(f"grid_thw.ndim must be 2, but it is {grid_thw.ndim}")

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.visual.dtype)
        else:
            pixel_values = image_input["pixel_values"].type(self.visual.dtype)
            pixel_values = pixel_values.reshape(-1, self.channel, self.patch_size, self.patch_size)
            pixel_values = self.rescale_and_normalize(pixel_values, self.do_rescale, 
                                                self.rescale_factor, self.do_normalize, self.image_mean, self.image_std)
            pixel_values = pixel_values.reshape(-1, self.channel * self.temporal_patch_size * self.patch_size * self.patch_size)
            if self.use_data_parallel:
                image_embeds = run_dp_sharded_mrope_vision_model(
                    self.visual, pixel_values, grid_thw, rope_type="rope_3d"
                )
            else:
                image_embeds = self.visual(pixel_values, grid_thw=grid_thw)

            image_embeds = self.visual.vision_projection(image_embeds)
        # Split concatenated embeddings for each image item.
        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size
        return image_embeds.split(sizes.tolist())

    def _process_video_input(
        self,
        video_input
    ) -> torch.Tensor:
        grid_thw = video_input["video_grid_thw"]
        if grid_thw.ndim != 2:
            raise ValueError(f"grid_thw.ndim must be 2, but it is {grid_thw.ndim}")

        if video_input["type"] == "video_embeds":
            video_embeds = video_input["video_embeds"].type(self.visual.dtype)
        else:
            pixel_values_videos = video_input["pixel_values_videos"].type(self.visual.dtype)
            if self.use_data_parallel:
                video_embeds = run_dp_sharded_mrope_vision_model(
                    self.visual, pixel_values_videos, grid_thw, rope_type="rope_3d"
                )
            else:
                video_embeds = self.visual(pixel_values_videos, grid_thw=grid_thw)

            video_embeds = self.visual.vision_projection(video_embeds)
        # Split concatenated embeddings for each video item.
        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size

        return video_embeds.split(sizes.tolist())

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            image_input = self._parse_and_validate_image_input(**kwargs)
            video_input = self._parse_and_validate_video_input(**kwargs)

            if image_input is None and video_input is None:
                inputs_embeds = None
            else:
                inputs_embeds = self.get_input_embeddings_v0(
                    input_ids, image_input=image_input, video_input=video_input
                )
                input_ids = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            attn_metadata=attn_metadata,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states
    
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states,
                                                  sampling_metadata)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
 
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="visual.merger.",
            tower_model="visual.",
        )

    def should_use_eager_mode(self, *args, **kwargs):
        attn_metadata = kwargs.get("attn_metadata", None)
        if not attn_metadata:
            return True
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[self.language_model.model.layers[self.language_model.model.start_layer].layer_name]
        return attn_metadata.attn_state != AscendAttentionState.DecodeOnly