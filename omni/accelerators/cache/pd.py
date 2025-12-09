from typing_extensions import override
import torch
import os
from llm_datadist.v2.llm_types import Cache, CacheDesc, BlocksCacheKey
import os
USE_NEW_DATADIST = os.getenv("ENABLE_DYNAMIC_LLMDATADIST", "0") == "1"
if USE_NEW_DATADIST:
    from omni.accelerators.pd.llmdatadist_manager_v1 import (
        LLMDataDistManager,
        TORCH_DTYPE_TO_NPU_DTYPE,
        unzip_kv_cache_dict,
        logger,
    )
else:
    from omni.accelerators.pd.llmdatadist_manager import (
        LLMDataDistManager,
        TORCH_DTYPE_TO_NPU_DTYPE,
        unzip_kv_cache_dict,
        logger,
    )
from . import kv_cache_interface as itfc

class OmniBiGroupDataDistManager(LLMDataDistManager):
    def __init__(self, vllm_config, local_host_ip=0, host_port=0):
        if USE_NEW_DATADIST:
            super().__init__(vllm_config, local_host_ip, host_port)
        else:
            super().__init__(vllm_config)
        self.registered_kv_caches: list[list[Cache]] = [[], []]

    @override
    def register_memory(self, kv_caches: dict[str, torch.Tensor]):
        if any(len(group_cache) > 0 for group_cache in self.registered_kv_caches):
            raise ValueError("Attr `registered_kv_caches` must be empty before register kv_caches.")
        # NOTE: flatten_kv_caches is a nested list like [[k1,k2,...,kL], [v1,v2,...,vL]]
        # if KV is just one tensor, then it's [[kv1,kv2,...,kvL]]
        flatten_kv_caches: list[list[torch.Tensor]] = unzip_kv_cache_dict(kv_caches)
        num_layers = len(flatten_kv_caches[0])
        if self.data_dist_config.kv_producer_pp_size > 1:
            PATTERN = itfc.PATTERN
            # partition layer indices into full and omni
            if self.data_dist_config.is_prefill:
                # 1. 获取rank
                # 2. 拿到对应的stage的layer_start和end
                # 3. PATTERN
                pp_rank = self.rank // self.prefill_tp_dp_size
                prefill_pp_partitions = self.data_dist_config.kv_producer_pp_partitions
                pp_start_layer_idx = sum(prefill_pp_partitions[:pp_rank])
                pp_end_layer_idx = pp_start_layer_idx + prefill_pp_partitions[pp_rank]
                PATTERN = itfc.PATTERN[pp_start_layer_idx : pp_end_layer_idx]

            full_layer_idx = [i for i in range(num_layers) if PATTERN[i] == 0]
            omni_layer_idx = [i for i in range(num_layers) if PATTERN[i] == 1]
        else:
            full_layer_idx = [i for i in range(num_layers) if itfc.PATTERN[i] == 0]
            omni_layer_idx = [i for i in range(num_layers) if itfc.PATTERN[i] == 1]
        layer_idx = [full_layer_idx, omni_layer_idx]

        # check validity
        if len(full_layer_idx) == 0:
            raise RuntimeError("Error! No full attention layer is found.")
        if len(omni_layer_idx) == 0:
            raise RuntimeError("Error! No omni attention layer is found.")
        if len(full_layer_idx) + len(omni_layer_idx) != num_layers:
            raise RuntimeError(f"Error! {len(full_layer_idx)=}, {len(omni_layer_idx)=}, but {num_layers=}.")

        # check shape
        full_shape = flatten_kv_caches[0][full_layer_idx[0]].shape
        omni_shape = flatten_kv_caches[0][omni_layer_idx[0]].shape
        if any(flatten_kv_caches[0][j].shape != full_shape for j in full_layer_idx):
            raise RuntimeError(f"Error! Not all full attn cache shape match {full_shape}.")
        if any(flatten_kv_caches[0][j].shape != omni_shape for j in omni_layer_idx):
            raise RuntimeError(f"Error! Not all omni attn cache shape match {omni_shape}.")

        # logging
        logger.warning("Trying to register grouped KV caches for OMNI attention, with "
                       f"{len(full_layer_idx)} full attn layers and {len(omni_layer_idx)} omni attn layers.")
        if self.data_dist_config.kv_producer_pp_size > 1:
            if self.data_dist_config.is_prefill:
                self._register_caches_prefill(flatten_kv_caches, layer_idx)
            else:
                self._register_caches_decode(flatten_kv_caches, layer_idx)
        else:
            N = len(flatten_kv_caches)
            used_ids = set()

            for model_id, sub_kv_caches in enumerate(flatten_kv_caches):
                # sub_kv_caches is a list of Tensors, whose length is number of layers

                for flag in range(len(self.registered_kv_caches)):
                    group_kv_caches = [sub_kv_caches[j] for j in layer_idx[flag]]
                    cache_desc = CacheDesc(num_tensors=len(group_kv_caches), shape=tuple(group_kv_caches[0].shape),
                                        data_type=TORCH_DTYPE_TO_NPU_DTYPE[group_kv_caches[0].dtype])
                    cache_addrs = [int(item.data_ptr()) for item in group_kv_caches]

                    if self.data_dist_config.is_prefill:
                        # NOTE: when assigning model_id to cache_key, we consider KV group information
                        # e.g., if registered_kv_caches = [[K_full, V_full], [K_omni, V_omni]]
                        # then model_ids should be [[0, 1], [2, 3]]
                        cur_id = flag * N + model_id
                        if cur_id in used_ids:
                            raise RuntimeError(f"Error! ID already used. {N=}, {model_id=}, {used_ids=}, {cur_id=}.")
                        used_ids.add(cur_id)
                        cache_key = BlocksCacheKey(self.data_dist_engine.cluster_id, model_id=cur_id)
                    else:
                        cache_key = None

                    cache = self.data_dist_engine.cache_manager.register_blocks_cache(cache_desc, cache_addrs, cache_key)
                    self.registered_kv_caches[flag].append(cache)
        logger.error(f" ***** registered_kv_caches num:{sum([len(group_kv_caches) for group_kv_caches in self.registered_kv_caches])}")
    
    def _register_caches_prefill(self, flatten_kv_caches, layer_idx):
        # model_id related
        N = len(flatten_kv_caches)
        used_ids = set()

        for model_id, sub_kv_caches in enumerate(flatten_kv_caches):
            # sub_kv_caches is a list of Tensors, whose length is number of layers

            for flag in range(len(self.registered_kv_caches)):
                group_kv_caches = [sub_kv_caches[j] for j in layer_idx[flag]]
                cache_desc = CacheDesc(num_tensors=len(group_kv_caches), shape=tuple(group_kv_caches[0].shape),
                                       data_type=TORCH_DTYPE_TO_NPU_DTYPE[group_kv_caches[0].dtype])
                cache_addrs = [int(item.data_ptr()) for item in group_kv_caches]

                # NOTE: when assigning model_id to cache_key, we consider KV group information
                # e.g., if registered_kv_caches = [[K_full, V_full], [K_omni, V_omni]]
                # then model_ids should be [[0, 1], [2, 3]]
                cur_id = flag * N + model_id
                if cur_id in used_ids:
                    raise RuntimeError(f"Error! ID already used. {N=}, {model_id=}, {used_ids=}, {cur_id=}.")
                used_ids.add(cur_id)
                cache_key = BlocksCacheKey(self.data_dist_engine.cluster_id, model_id=cur_id)

                cache = self.data_dist_engine.cache_manager.register_blocks_cache(cache_desc, cache_addrs, cache_key)
                self.registered_kv_caches[flag].append(cache)
    
    def _register_caches_decode(self, flatten_kv_caches, layer_idx):
        prefill_pp_partitions = self.data_dist_config.kv_producer_pp_partitions
        for flag in range(len(self.registered_kv_caches)):
            cnt_layer_num = 0
            layer_idx_start = 0
            for cur_pp_stage_layer_num in prefill_pp_partitions:
                cur_pp_stage_kv_caches = []
                layer_idx_end = layer_idx_start
                while layer_idx_end < len(layer_idx[flag]) and cnt_layer_num <= layer_idx[flag][layer_idx_end] < cnt_layer_num + cur_pp_stage_layer_num:
                    layer_idx_end += 1
                flag_stage_layer_idx = layer_idx[flag][layer_idx_start : layer_idx_end]
                layer_idx_start = layer_idx_end
                for sub_kv_caches in flatten_kv_caches:
                    # sub_kv_caches is a list of Tensors, whose length is number of layers
                    # flag_stage_layer_idx = layer_idx[flag][cnt_layer_num : cnt_layer_num + cur_pp_stage_layer_num]
                    group_kv_caches = [sub_kv_caches[j] for j in flag_stage_layer_idx]
                    # group_kv_caches = [sub_kv_caches[j] for j in layer_idx[flag] if cnt_layer_num <= j < cnt_layer_num + cur_pp_stage_layer_num]
                    cache_desc = CacheDesc(num_tensors=len(group_kv_caches), shape=tuple(group_kv_caches[0].shape),
                                           data_type=TORCH_DTYPE_TO_NPU_DTYPE[group_kv_caches[0].dtype])
                    cache_addrs = [int(item.data_ptr()) for item in group_kv_caches]

                    cache = self.data_dist_engine.cache_manager.register_blocks_cache(cache_desc, cache_addrs, None)
                    cur_pp_stage_kv_caches.append(cache)
                self.registered_kv_caches[flag].append(cur_pp_stage_kv_caches)
                cnt_layer_num += cur_pp_stage_layer_num

    @override
    def pull_kv(self, src_blocks: list[int], tgt_blocks: list[list[int]], prompt_cluster_id: int, prefill_dp_rank: int=0):
        """Pull KV Caches for both full and omni attention layers. The input `tgt_blocks`
        is a list of lists of ints like [[blk1,...,blk100], [blk1,blk2,blk3]], where the
        first sublist is the block table for full attention layers while the second is
        for omni. In contrast, `src_blocks` is a list corresponding to the block table
        allocated for all layers during prefill.
        """
        if os.getenv("ENABLE_PD_MOCKUP", "0") == "1":
            return
        if isinstance(src_blocks[0], int):
            src_blocks = [src_blocks] * len(tgt_blocks)
        torch.npu.set_device(f"npu:{self.local_rank}")
        sink, recent = itfc.SINK, itfc.RECENT
        omni_max_blocks = sink + recent
        N = len(self.registered_kv_caches[0])
        used_ids = set()
        if self.data_dist_config.kv_producer_pp_size > 1:
            for flag in range(len(self.registered_kv_caches)):
                group_src_blocks: list[int] = src_blocks[flag]
                group_tgt_blocks: list[int] = tgt_blocks[flag]
                for pp_stage_ind, cur_pp_stage_kv_caches in enumerate(self.registered_kv_caches[flag]):
                    for model_id, kv_cache in enumerate(cur_pp_stage_kv_caches):
                        cur_id = flag * N + model_id
                        cluster_id_pp_offset = pp_stage_ind * self.prefill_tp_dp_size

                        prompt_cache_key = BlocksCacheKey(
                            prompt_cluster_id=prompt_cluster_id + cluster_id_pp_offset, model_id=cur_id)
                        if flag == 0:
                            if USE_NEW_DATADIST:
                                ret = self._pull_blocks(prompt_cache_key, kv_cache,
                                            group_src_blocks, group_tgt_blocks)
                            else:
                                self._pull_blocks(prompt_cache_key, kv_cache,
                                                group_src_blocks, group_tgt_blocks)
                        else:
                            if len(group_tgt_blocks) == 0:
                                continue
                            tmp_src, tmp_tgt = group_src_blocks, group_tgt_blocks
                            if len(group_src_blocks) < omni_max_blocks:
                                tmp_tgt = group_tgt_blocks[:len(group_src_blocks)]
                            elif len(group_src_blocks) > omni_max_blocks:
                                tmp_src = group_src_blocks[:sink] + group_src_blocks[-recent:]
                            if len(tmp_src) != len(tmp_tgt):
                                raise RuntimeError("src and tgt cannot match for omni kv caches. "
                                                f"{src_blocks=}, {tgt_blocks=}, "
                                                f"{tmp_src=}, {tmp_tgt=}, "
                                                f"{len(tmp_src)=}, {len(tmp_tgt)=}.")
                            if USE_NEW_DATADIST:
                                ret = self._pull_blocks(prompt_cache_key, kv_cache,
                                                tmp_src, tmp_tgt)
                            else:
                                self._pull_blocks(prompt_cache_key, kv_cache,
                                                tmp_src, tmp_tgt)
                        if USE_NEW_DATADIST:
                            if not ret:
                                self._refresh_link(prompt_cluster_id, prefill_dp_rank, self.rank)
                                ret_updated = self._pull_blocks(prompt_cache_key, kv_cache,
                                                tmp_src, tmp_tgt)
                                if not ret_updated:
                                    raise RuntimeError(f"Failed to pull kv even if rebuild the kv link!")
        else:
            for flag in range(len(self.registered_kv_caches)):
                group_src_blocks: list[int] = src_blocks[flag]
                group_tgt_blocks: list[int] = tgt_blocks[flag]
                for model_id, kv_cache in enumerate(self.registered_kv_caches[flag]):
                    cur_id = flag * N + model_id
                    if cur_id in used_ids:
                        raise RuntimeError(f"Error! ID already pulled. {N=}, {model_id=}, {used_ids=}, {cur_id=}.")
                    used_ids.add(cur_id)

                    prompt_cache_key = BlocksCacheKey(
                        prompt_cluster_id=prompt_cluster_id, model_id=cur_id)
                    if flag == 0:
                        if USE_NEW_DATADIST:
                            ret = self._pull_blocks(prompt_cache_key, kv_cache,
                                            group_src_blocks, group_tgt_blocks)
                        else:
                            self._pull_blocks(prompt_cache_key, kv_cache,
                                            group_src_blocks, group_tgt_blocks)
                    else:
                        if len(group_tgt_blocks) == 0:
                            continue
                        tmp_src, tmp_tgt = group_src_blocks, group_tgt_blocks
                        if len(group_src_blocks) < omni_max_blocks:
                            tmp_tgt = group_tgt_blocks[:len(group_src_blocks)]
                        elif len(group_src_blocks) > omni_max_blocks:
                            tmp_src = group_src_blocks[:sink] + group_src_blocks[-recent:]
                        if len(tmp_src) != len(tmp_tgt):
                            raise RuntimeError("src and tgt cannot match for omni kv caches. "
                                            f"{src_blocks=}, {tgt_blocks=}, "
                                            f"{len(tmp_src)=}, {len(tmp_tgt)=}.")
                        if USE_NEW_DATADIST:
                            ret = self._pull_blocks(prompt_cache_key, kv_cache,
                                            tmp_src, tmp_tgt)
                        else:
                            self._pull_blocks(prompt_cache_key, kv_cache,
                                            tmp_src, tmp_tgt)
                    if USE_NEW_DATADIST:
                        if not ret:
                            self._refresh_link(prompt_cluster_id, prefill_dp_rank, self.rank)
                            ret_updated = self._pull_blocks(prompt_cache_key, kv_cache,
                                            tmp_src, tmp_tgt)
                            if not ret_updated:
                                raise RuntimeError(f"Failed to pull kv even if rebuild the kv link!")

