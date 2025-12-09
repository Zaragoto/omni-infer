# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Inference-only Longcat-Flash model."""
from typing import Iterable, Optional, Tuple
import os
import concurrent.futures
import logging
import torch
import torch_npu
from torch import nn
import torchair as tng
from transformers import PretrainedConfig
torch._logging.set_logs(recompiles=True)

from sglang.srt.distributed import get_tensor_model_parallel_world_size, tensor_model_parallel_all_gather
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import (
    BumpAllocator,
    LazyValue,
    add_prefix,
)
from omni.adaptors.sglang.distributed import get_mlp_tp_group
from omni.adaptors.sglang.layers.attention.deepseek_mla import DeepseekMLA
from omni.adaptors.sglang.layers.moe.longcat_moe import LongcatFlashMoE
from omni.adaptors.sglang.layers.moe.fused_moe.layer import FusedMoE
from omni.adaptors.sglang.layers.layernorm import RMSNorm
from omni.adaptors.sglang.models.deepseek.deepseek_v3 import ParallelDeepseekMLP
from omni.adaptors.sglang.layers.vocab_parallel_embedding import VocabParallelEmbedding, ParallelLMHead
from omni.models.config_loader.loader import model_extra_config
from omni.adaptors.sglang.utils import ConditionalTNGScope
logger = logging.getLogger(__name__)

"""MLP module activation split length, split by 64G VRAM, need to confirm the optimal split length based on sequence length and performance"""
SEQ_SPLIT_LENGTH_BEFORE_ALL_GATHER = 64

MAX_ATTN_PREFETCH_SIZE = 18
MB_TO_BYTES = 1024 * 1024

class LongcatFlashMLP(ParallelDeepseekMLP):
    def __init__(self,*args, **kwargs,):
        super().__init__(*args, **kwargs)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        use_reduce_scatter: bool = False,
        return_down_hidden_status: bool = False,
        **kwargs):
        x = hidden_states
        x = get_mlp_tp_group().all_gather(x, dim=0)
        
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up, self.quant_symbol)
        down_hidden_status, _ = self.down_proj(x, skip_all_reduce=use_reduce_scatter)
        x = down_hidden_status
        x = get_mlp_tp_group().reduce_scatter_(x)
        
        if return_down_hidden_status:
            return x, down_hidden_status
        return x

class LongcatFlashDecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        is_nextn: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        self.enable_dp_attention = global_server_args_dict["enable_dp_attention"]
        self.speculative_algorithm = global_server_args_dict["speculative_algorithm"]
        self.layer_id = layer_id
        self.is_nextn = is_nextn
        self.quant_symbol = False
        self.use_super_kernel = model_extra_config.operator_opt_config.use_super_kernel
        self.use_mla_prolog = model_extra_config.operator_opt_config.use_mlaprolog
        self.enable_multi_stream = model_extra_config.operator_opt_config.enable_scmoe_multi_stream
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        self.self_attn = nn.ModuleList(
            [
                DeepseekMLA(
                    config=config,
                    hidden_size=self.hidden_size,
                    num_heads=config.num_attention_heads,
                    qk_nope_head_dim=config.qk_nope_head_dim,
                    qk_rope_head_dim=config.qk_rope_head_dim,
                    v_head_dim=config.v_head_dim,
                    q_lora_rank=config.q_lora_rank if hasattr(config, "q_lora_rank") else None,
                    kv_lora_rank=config.kv_lora_rank,
                    rope_theta=getattr(config, "rope_theta", 10000),
                    rope_scaling=getattr(config, "rope_scaling", None),
                    rope_is_neox_style=True,
                    max_position_embeddings=getattr(config, "max_position_embeddings", 8192),
                    quant_config=quant_config,
                    layer_id=layer_id,
                    reduce_results=False,
                    prefix=add_prefix(f"self_attn.{i}", prefix)
                )
                for i in range(2)
            ]
        )

        self.input_layernorm = nn.ModuleList(
            [RMSNorm(config.hidden_size, eps=config.rms_norm_eps) for i in range(2)]
        )
        self.post_attention_layernorm = nn.ModuleList(
            [RMSNorm(config.hidden_size, eps=config.rms_norm_eps) for i in range(2)]
        )

        self.mlps = nn.ModuleList(
            [
                LongcatFlashMLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.ffn_hidden_size,
                    hidden_act=config.hidden_act,
                    quant_config=quant_config,
                    prefix=add_prefix(f"mlps.{i}", prefix),
                    tp_size=get_mlp_tp_group().world_size,
                    tp_rank=get_mlp_tp_group().rank_in_group,
                )
                for i in range(2)
            ]
        )

        self.mlp = LongcatFlashMoE(
            config=config,
            layer_id=self.layer_id,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

    def forward_multi_stream(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        zero_allocator: BumpAllocator,
        next_layer = None,
        **kwargs,
    ) -> torch.Tensor:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm[0](hidden_states)
        else:
            # Adapt: adapt for w8a8 dynamic, do quant
            # Combines residual add and rmsnorm
            hidden_states, residual = self.input_layernorm[0](
                hidden_states, residual, quant_symbol=self.quant_symbol and not self.use_mla_prolog)
            # Adapt end.
        hidden_states = self.self_attn[0](
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            zero_allocator=zero_allocator,
        )
        hidden_states_norm, residual = self.post_attention_layernorm[0](hidden_states, residual)
    
        with ConditionalTNGScope(multi_stream=True, stream_id="longcat_scmoe", core_num="8|16"):
            moe_hidden_states = self.mlp(hidden_states_norm, forward_batch)

        with tng.scope.limit_core_num(16, 32):
            hidden_states, down_hidden_status = self.mlps[0](hidden_states_norm, forward_batch,return_down_hidden_status=True)
            use_prefetch = model_extra_config.operator_opt_config.use_prefetch
            # prefetch self.q_a_proj.weight, self.q_b_proj.weight
            self._npu_prefetch(use_prefetch, self.self_attn[1].q_a_proj.weight, down_hidden_status)
            self._npu_prefetch(use_prefetch, self.self_attn[1].q_b_proj.weight, down_hidden_status)
                
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm[1](hidden_states)
            else:
                # Adapt: adapt for w8a8 dynamic, do quant
                # Combines residual add and rmsnorm
                hidden_states, residual = self.input_layernorm[1](
                    hidden_states, residual, quant_symbol=self.quant_symbol and not self.use_mla_prolog)

            hidden_states = self.self_attn[1](
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                zero_allocator=zero_allocator,
            )

            hidden_states, residual = self.post_attention_layernorm[1](hidden_states, residual)

            hidden_states, down_hidden_status = self.mlps[1](hidden_states, forward_batch, return_down_hidden_status=True)
            if next_layer is not None:
                self._npu_prefetch(use_prefetch, next_layer.self_attn[0].q_a_proj.weight, down_hidden_status)
                self._npu_prefetch(use_prefetch, next_layer.self_attn[0].q_b_proj.weight, down_hidden_status, 36 * MB_TO_BYTES)
                self._npu_prefetch(use_prefetch, next_layer.self_attn[0].kv_a_proj_with_mqa.weight, down_hidden_status,7 * MB_TO_BYTES)
            
            try:
                tng.scope.npu_wait_tensor(hidden_states, moe_hidden_states)
            except:
                pass

            if isinstance(moe_hidden_states, (tuple, list)):
                assert len(moe_hidden_states) == 2
                # 0 is the shared expert hidden_states, 1 is the routing expert hidden_states, add operation cannot be placed in the super kernel
                moe_hidden_states = moe_hidden_states[0] + moe_hidden_states[1]
            hidden_states = moe_hidden_states + hidden_states

        return hidden_states, residual

    def _npu_prefetch(self, switch_flag, weight, depend, size=MAX_ATTN_PREFETCH_SIZE * MB_TO_BYTES, offset=0):
        if not switch_flag:
            return None
        return torch_npu.npu_prefetch(weight, depend, size, offset)
        
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        zero_allocator: BumpAllocator,
        next_layer = None,
        **kwargs,
    ) -> torch.Tensor:
        is_prefill = forward_batch.is_extend_in_batch
        if self.enable_multi_stream and not is_prefill:
            return self.forward_multi_stream(
                positions, hidden_states, forward_batch, residual, zero_allocator, next_layer, **kwargs
            )

        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm[0](hidden_states)
        else:
            # Adapt: adapt for w8a8 dynamic, do quant
            # Combines residual add and rmsnorm
            hidden_states, residual = self.input_layernorm[0](
                hidden_states, residual, quant_symbol=self.quant_symbol and not self.use_mla_prolog)
            # Adapt end.
        hidden_states = self.self_attn[0](
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            zero_allocator=zero_allocator,
        )
        hidden_states, residual = self.post_attention_layernorm[0](hidden_states, residual)

        moe_hidden_states = self.mlp(hidden_states, forward_batch)
        if isinstance(moe_hidden_states, (tuple, list)):
            assert len(moe_hidden_states) == 2
            # 0 is the shared expert hidden_states, 1 is the routing expert hidden_states, add operation cannot be placed in the super kernel
            moe_hidden_states = moe_hidden_states[0] + moe_hidden_states[1]
        try:
            tng.scope.npu_wait_tensor(hidden_states, moe_hidden_states)
        except:
            pass
        hidden_states = self.mlps[0](hidden_states, forward_batch)

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm[1](hidden_states)
        else:
            # Adapt: adapt for w8a8 dynamic, do quant
            # Combines residual add and rmsnorm
            hidden_states, residual = self.input_layernorm[1](
                hidden_states, residual, quant_symbol=self.quant_symbol and not self.use_mla_prolog)

        hidden_states = self.self_attn[1](
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            zero_allocator=zero_allocator,
        )

        hidden_states, residual = self.post_attention_layernorm[1](hidden_states, residual)

        hidden_states = self.mlps[1](hidden_states, forward_batch)
 
        hidden_states = moe_hidden_states + hidden_states

        return hidden_states, residual


class LongcatFlashModel(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = ""
    ):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.prefix = f"{prefix}.layers"
        self.postfix = ".self_attn.attn"
        self.tp_size = get_tensor_model_parallel_world_size()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            enable_tp=not global_server_args_dict["enable_dp_attention"],
        )
        
        self.layers = nn.ModuleList(
            [
                LongcatFlashDecoderLayer(
                    config,
                    layer_id,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{layer_id}", prefix),
                )
                for layer_id in range(config.num_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids, reduce=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        total_num_layers = len(self.layers)
        device = input_embeds.device if input_embeds is not None else input_ids.device
        zero_allocator = BumpAllocator(
            buffer_size=total_num_layers * 2 * (2 if forward_batch.can_run_tbo else 1),
            dtype=torch.float32,
            device=device,
        )

        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds

        residual = None

        for i in range(total_num_layers):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
                zero_allocator,
                next_layer=self.layers[i + 1] if i < total_num_layers - 1 else None,
                )

        if not forward_batch.is_prefill_idle:
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)

        if forward_batch.is_extend_in_batch :
            hidden_states = tensor_model_parallel_all_gather(hidden_states, dim=0)
        return hidden_states


class LongcatFlashForCausalLM(nn.Module):
    packed_modules_mapping = {}

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        # for quark model load
        # Fuse q_a_proj and kv_a_proj_with_mqa along output dimension when q_lora_rank is not None
        self.fuse_qkv_a_proj = model_extra_config.operator_opt_config.merge_qkv
        if self.fuse_qkv_a_proj:
            self.packed_modules_mapping["fused_qkv_a_proj_with_mqa"] = [
                "q_a_proj",
                "kv_a_proj_with_mqa",
            ]

        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config
        self.model = LongcatFlashModel(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=global_server_args_dict["enable_dp_lm_head"],
        )
        self.logits_processor = LogitsProcessor(config)

        self._routed_experts_weights_of_layer = LazyValue(
            lambda: {
                layer_id: layer.mlp.get_moe_weights()
                for layer_id, layer in enumerate(self.model.layers)
                if isinstance(layer.mlp, LongcatFlashMoE)
            }
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    @property
    def routed_experts_weights_of_layer(self):
        return self._routed_experts_weights_of_layer.value

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        if forward_batch.is_extend_in_batch :
            last_index = torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
            pruned_states = hidden_states[last_index]
            if not forward_batch.capture_hidden_mode.need_capture():
                hidden_states = None
        else:
            pruned_states = hidden_states
        logits = self.compute_lmhead(pruned_states)
        return LogitsProcessorOutput(
            next_token_logits=logits,
            hidden_states=hidden_states,
        )

    def compute_lmhead(
            self,
            hidden_states: torch.Tensor,
            selected_indices: Optional[torch.Tensor] = None,
            embedding_bias: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if selected_indices is not None:
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
            if hidden_states.shape[0] != selected_indices.shape[0]:
                hidden_states = hidden_states.index_select(0, selected_indices)

        # Get the logits for the next tokens.
        logits = self.lm_head(hidden_states, embedding_bias)
        return logits

    def post_load_weights(self, is_nextn=False, weight_names=None):
        # Perform post-processing after loading weights
        if is_nextn:
            layer_ids = [self.config.num_hidden_layers]
        else:
            if weight_names is None:
                layer_ids = range(self.config.num_hidden_layers)
            else:
                layer_ids = set()
                for name in weight_names:
                    if "mtp" in name:
                        continue
                    if "kv_b_proj" in name:
                        layer_id = int(name.split(".")[2])
                        if layer_id < self.config.num_hidden_layers:
                            layer_ids.add(layer_id)

        for layer_id in layer_ids:
            for i in range(2):
                self_attn = (
                    self.model.layers[layer_id].self_attn[i]
                    if not is_nextn
                    else self.model.decoder.self_attn
                )
                if self.config.mla_scale_q_lora:
                    self_attn.q_a_layernorm.weight.data *= (
                        self.config.hidden_size / self.config.q_lora_rank
                    ) ** 0.5
                if self.config.mla_scale_kv_lora:
                    self_attn.kv_a_layernorm.weight.data *= (
                        self.config.hidden_size / self.config.kv_lora_rank
                    ) ** 0.5
                if self_attn.w_kc is not None and self_attn.w_vc is not None:
                    self_attn.w_kc = torch.nn.Parameter(self_attn.w_kc.contiguous(), requires_grad=False)
                    self_attn.w_vc = torch.nn.Parameter(self_attn.w_vc.contiguous(), requires_grad=False)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):
        if is_nextn:
            if hasattr(self.config, "num_nextn_predict_layers"):
                num_nextn_layers = self.config.num_nextn_predict_layers
                assert num_nextn_layers == 1, "Only 1 nextn layer is supported"
                # compatible with old design
                nextn_layer_id = (
                    0
                    if self.config.num_layers == 1
                    else self.config.num_layers
                )
            else:
                raise ValueError("num_nextn_predict_layers is not in the config")

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
        )

        # Fuse q_a_proj and kv_a_proj_with_mqa along output dimension when q_lora_rank is not None
        fuse_qkv_a_proj = model_extra_config.operator_opt_config.merge_qkv
        cached_a_proj = {} if fuse_qkv_a_proj else None

        if is_nextn:
            nextn_layer_prefix = f"model.layers.{nextn_layer_id}"
            nextn_spec_weight_names = [
                "shared_head.norm",
                "eh_proj",
                "enorm",
                "hnorm",
            ]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            params_dict = dict(self.named_parameters())
            weight_names = []
            for name, loaded_weight in weights:
                weight_names.append(name)

                if not is_nextn:
                    if hasattr(self.config, "num_nextn_predict_layers"):
                        num_nextn_layers = self.config.num_nextn_predict_layers
                        if num_nextn_layers > 0 and name.startswith("model.layers"):
                            name_list = name.split(".")
                            if (
                                len(name_list) >= 3
                                and int(name_list[2]) >= self.config.num_layers
                            ):
                                continue
                else:
                    if not name.startswith(nextn_layer_prefix):
                        continue

                    # Use shared head and embed weights from target model
                    if "shared_head.head" in name or "embed_tokens" in name:
                        continue

                    is_decoder = True
                    # For nextn specific weights
                    for weight_name in nextn_spec_weight_names:
                        if weight_name in name:
                            name = name.replace(nextn_layer_prefix, "model")
                            is_decoder = False
                            break
                    # For decoder layer weights
                    if is_decoder:
                        name = name.replace(nextn_layer_prefix, "model.decoder")

                if "rotary_emb.inv_freq" in name:
                    continue
                if "weight_offset" in name:
                    continue  # NPU not support for weight_offset now.

                for param_name, weight_name, shard_id in stacked_params_mapping:
                    # Skip non-stacked layers and experts (experts handled below).
                    if weight_name not in name:
                        continue
                    # We have mlp.experts[0].gate_proj in the checkpoint.
                    # Since we handle the experts below in expert_params_mapping,
                    # we need to skip here BEFORE we update the name, otherwise
                    # name will be updated to mlp.experts[0].gate_up_proj, which
                    # will then be updated below in expert_params_mapping
                    # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                    if ("mlp.experts." in name) and name not in params_dict:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name not in params_dict:
                        logger.warning(f"{name} not found in params_dict.")
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    futures.append(
                        executor.submit(weight_loader, param, loaded_weight, shard_id)
                    )
                    break
                else:
                    for mapping in expert_params_mapping:
                        param_name, weight_name, expert_id, shard_id = mapping
                        if weight_name not in name:
                            continue
                        name = name.replace(weight_name, param_name)
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        futures.append(
                            executor.submit(
                                weight_loader,
                                param,
                                loaded_weight,
                                name,
                                shard_id=shard_id,
                                expert_id=expert_id,
                            )
                        )
                        break
                    else:
                        # Skip loading extra bias for GPTQ models.
                        if name.endswith(".bias") and name not in params_dict:
                            continue
                        if fuse_qkv_a_proj and (
                            "q_a_proj" in name or "kv_a_proj_with_mqa" in name
                        ):
                            cached_a_proj[name] = loaded_weight
                            q_a_proj_name = (
                                name
                                if "q_a_proj" in name
                                else name.replace("kv_a_proj_with_mqa", "q_a_proj")
                            )
                            kv_a_proj_name = (
                                name
                                if "kv_a_proj_with_mqa" in name
                                else name.replace("q_a_proj", "kv_a_proj_with_mqa")
                            )

                            # When both q_a_proj and kv_a_proj_with_mqa has been cached, load the fused weight to parameter
                            if (
                                q_a_proj_name in cached_a_proj
                                and kv_a_proj_name in cached_a_proj
                            ):
                                q_a_proj_weight = cached_a_proj[q_a_proj_name]
                                kv_a_proj_weight = cached_a_proj[kv_a_proj_name]
                                cat_dim = 0
                                if self.quant_config is not None and (
                                    self.quant_config.get_name() == "awq"
                                    or self.quant_config.get_name() == "awq_marlin"
                                    or self.quant_config.get_name() == "moe_wna16"
                                ):
                                    cat_dim = 1
                                fused_weight = torch.cat(
                                    [q_a_proj_weight, kv_a_proj_weight], dim=cat_dim
                                )
                                param_name = (
                                    name.replace(
                                        "q_a_proj", "fused_qkv_a_proj_with_mqa"
                                    )
                                    if "q_a_proj" in name
                                    else name.replace(
                                        "kv_a_proj_with_mqa",
                                        "fused_qkv_a_proj_with_mqa",
                                    )
                                )
                                param = params_dict[param_name]

                                weight_loader = getattr(
                                    param, "weight_loader", default_weight_loader
                                )
                                futures.append(
                                    executor.submit(weight_loader, param, fused_weight)
                                )
                                cached_a_proj.pop(q_a_proj_name)
                                cached_a_proj.pop(kv_a_proj_name)
                        else:
                            if (
                                "k_scale" in name or "v_scale" in name
                            ) and name not in params_dict:
                                # modelopt attn kv scale is named differently
                                for scale in ["k_scale", "v_scale"]:
                                    if scale in name:
                                        name = name.replace(
                                            f"{scale[0]}_proj", "attn_mqa"
                                        )
                                        break
                            if name not in params_dict:
                                # modelopt ckpt contains not needed weights for MTP module:
                                # model.decoder.self_attn.attn_mqa.v_scale and
                                # model.decoder.self_attn.attn_mqa.k_scale
                                logger.warning(f"{name} not found in params_dict.")
                                continue
                            param = params_dict[name]
                            weight_loader = getattr(
                                param, "weight_loader", default_weight_loader
                            )
                            futures.append(
                                executor.submit(weight_loader, param, loaded_weight)
                            )

            # Wait for all tasks to complete and raise any exceptions.
            for future in concurrent.futures.as_completed(futures):
                future.result()

        self.post_load_weights(is_nextn=is_nextn, weight_names=weight_names)

    def get_embed_and_head(self):
        return self.model.embed_tokens, self.lm_head

    def set_embed_and_head(self, embed, head):
        self.model.embed_tokens = embed
        self.lm_head = head

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_layers,
            num_logical_experts=config.n_routed_experts,
            num_groups=1,
        )

EntryClass = LongcatFlashForCausalLM