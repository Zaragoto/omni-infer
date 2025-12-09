# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import json
from collections.abc import Iterator
import math
import threading
from typing import TYPE_CHECKING, Any, Optional, Union, Mapping, Dict, List
import zmq
import os
import pickle
import threading
import time
from typing import TYPE_CHECKING, Any, Optional

import zmq
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.envs import VLLM_RPC_TIMEOUT
from vllm.logger import logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput

from omni.accelerators.pd.utils import get_config_from_dict_or_env

if TYPE_CHECKING:
    from vllm.config import VllmConfig, KVTransferConfig
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.request import Request
from vllm.v1.request import Request
from vllm.utils import round_down
from dataclasses import dataclass
from collections import defaultdict
import torch
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size,
    get_tp_group, get_pp_group)

from vllm.utils import get_open_port
from vllm.v1.request import RequestStatus
import queue
from concurrent.futures import ThreadPoolExecutor
from omni.tools.profiler.trace.tracing import get_local_ip

GET_META_MSG = b"get_meta_msg"

thread_dump_path = os.environ.get("VLLM_THREAD_DUMP_PATH", "/tmp/vllm_thread_info")
BLOCK_RELEASE_DELAY = int(os.environ.get("BLOCK_RELEASE_DELAY", 600))  # seconds, use to free blocks when the request is finished for a long time 

BASE_PORT = int(os.environ.get("BASE_PORT", "15077"))
ZMQ_BASE_PORT = int(os.environ.get("ZMQ_BASE_PORT", "17555"))
ZMQ_DECODE_PUSH_REQUEST_TO_PREFILL_PORT = int(os.environ.get("ZMQ_DECODE_PUSH_REQUEST_TO_PREFILL_PORT", "17556"))
ZMQ_PREFILL_PUSH_INFORMATION_TO_DECODE_BASE_PORT = int(os.environ.get("ZMQ_PREFILL_PUSH_INFORMATION_TO_DECODE_BASE_PORT", "17557"))

from omni.accelerators.pd.llmdatadist_manager import LLMDataDistManager, LLMDataDistConfig
from omni.tools.profiler.apply_profiler_patches import patch_request
patch_request()


@dataclass
class ReqMeta:
    local_block_ids: list[int]
    remote_block_ids: list[int]
    remote_host: str
    remote_cluster_id: str
    spec_token_ids: Optional[list[int]]
    remote_dp_rank: Optional[int]
    remote_request_id: Optional[str]
    trace_headers: Optional[Mapping[str, str]] = None

@dataclass
class ReqMetaPrefill:
    finish_time: float

class DatadistConnectorMetadata(KVConnectorMetadata):
    """Metadata for datadist connector."""

    def __init__(self):
        self.requests: dict[str, ReqMeta] = {}

    def add_new_req(
        self,
        request_id: str,
        local_block_ids: list[int],
        kv_transfer_params: dict[str, Any],
        trace_headers: Optional[Mapping[str, str]] = None,
    ):
        self.requests[request_id] = ReqMeta(
            local_block_ids=local_block_ids,
            remote_block_ids=kv_transfer_params["remote_block_ids"],
            remote_host=kv_transfer_params["remote_host_ip"],
            remote_cluster_id=kv_transfer_params["remote_cluster_id"],
            spec_token_ids=kv_transfer_params["spec_token_ids"],
            remote_dp_rank=kv_transfer_params.get("remote_dp_rank", 0),
            remote_request_id=kv_transfer_params.get("remote_request_id", None),
            trace_headers=trace_headers  or {},
        )

class DatadistConnectorMetadataPrefill(KVConnectorMetadata):
    """Metadata for datadist connector."""

    def __init__(self):
        self.requests: dict[str, ReqMetaPrefill] = {}

    def add_new_req(
        self,
        request_id: str,
        finish_time: float,
    ):
        self.requests[request_id] = ReqMetaPrefill(
            finish_time=finish_time
        )


class LLMDataDistConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole):
        if vllm_config.kv_transfer_config is None:
            raise RuntimeError("vllm_config.kv_transfer_config cannot be None")

        if vllm_config.model_config.is_deepseek_mla:
            vllm_config.kv_transfer_config.kv_parallel_size = 1
            logger.info("Set kv_parallel_size to 1 when use deepseek mla model.")

        self.datadist_config = LLMDataDistConfig(vllm_config, ignore_load_rank=True)
        self.cluster_id_start = self.datadist_config.cluster_id_start
        self.host_ip = self.datadist_config.local_group.host_ip
        # Introduce the environment variable VLLM_LLMDATADIST_ZMQ_PORT to resolve ZMQ connection conflicts during
        # multi-P deployments on the same machine.
        # This variable should not be set separately unless specifically required for this scenario.
        self.host_port = get_config_from_dict_or_env(vllm_config.kv_transfer_config, "kv_port",
                                                     "VLLM_LLMDATADIST_ZMQ_PORT", "5568", int)
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        self.host_port += dp_rank
        self.is_prefill = vllm_config.kv_transfer_config.kv_role == "kv_producer"

        if role == KVConnectorRole.SCHEDULER:
            if self.is_prefill:
                self.connector_scheduler = PrefillConnectorScheduler(vllm_config, self.cluster_id_start, self.host_ip, str(self.host_port), str(self.host_port + 1000))
            else:
                self.connector_scheduler = DecodeConnectorScheduler(vllm_config, str(self.host_port + 2000))
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            if self.is_prefill:
                self.connector_worker = PrefillConnectorWorker(vllm_config, str(self.host_ip), str(self.host_port), str(self.host_port + 1000))
            else:
                self.connector_worker = DecodeConnectorWorker(vllm_config, str(self.host_ip), self.cluster_id_start, str(self.host_port + 2000))
            self.connector_scheduler = None

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> tuple[int, bool]:
        if self.connector_scheduler is None:
            raise RuntimeError("self.connector_scheduler cannot be None")
        return self.connector_scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        if self.connector_scheduler is None:
            raise RuntimeError("self.connector_scheduler cannot be None")
        return self.connector_scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(
            self,
            scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        if self.connector_scheduler is None:
            raise RuntimeError("self.connector_scheduler cannot be None")
        return self.connector_scheduler.build_connector_metadata(scheduler_output)

    def request_finished(
            self,
            request: "Request",
            block_ids: list[int],
            spec_token_ids: Optional[list[int]] = []
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        if self.connector_scheduler is None:
            raise RuntimeError("self.connector_scheduler cannot be None")
        return self.connector_scheduler.request_finished(request, block_ids, spec_token_ids)

    ############################################################
    # Worker Side Methods
    ############################################################
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        if self.connector_worker is None:
            raise RuntimeError("self.connector_worker cannot be None")
        return self.connector_worker.register_kv_caches(kv_caches)

    def get_finished(self,
                     finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """Get the finished recving and sending requests."""
        if self.connector_worker is None:
            raise RuntimeError("self.connector_worker cannot be None")
        return self.connector_worker.get_finished(self._connector_metadata)

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        if self.connector_worker is None:
            raise RuntimeError("self.connector_worker cannot be None")
        if not isinstance(self._connector_metadata, Union[DatadistConnectorMetadata, DatadistConnectorMetadataPrefill]):
            raise RuntimeError("self._connector_metadata must be an instance of DatadistConnectorMetadata or DatadistConnectorMetadataPrefill")
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Connector does not do layerwise saving."""
        pass

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """Connector does not save explicitly."""
        pass

    def wait_for_save(self):
        """Connector does not save explicitly."""
        pass

    def pop_trace_headers(self, req_id: str) -> dict:
        if self.connector_scheduler is None:
            raise RuntimeError("self.connector_scheduler cannot be None")                             
        return self.connector_scheduler.pop_trace_headers(req_id)

class PrefillConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, vllm_config, cluster_id_start: str, host_ip: str, host_port: str, trace_p_port: str):
        self.vllm_config = vllm_config
        self.cluster_id_start = cluster_id_start
        self.host_ip = host_ip
        self.host_port = host_port
        self.trace_p_port=trace_p_port
        logger.info("Initializing LLMDataDist Scheduler %s %s %s", cluster_id_start, host_ip, host_port)
        # initialize the dict to save requests finish time
        self.requests_finish_time = dict()
        # req_id -> headers
        self.sending_trace_headers: dict[str, dict] = {}

        self._transfer_lock = threading.Lock()
        self.ctx = zmq.Context()
        self.pull_socket = self.ctx.socket(zmq.PULL)
        self.pull_socket.bind(f"tcp://{host_ip}:{trace_p_port}")
        if os.getenv("PROFILING_NAMELIST", None):
            self._listener_thread = threading.Thread(target=self._listen_worker_headers, daemon=True)
            self._listener_thread.start()
        
        # initialize the dict to save requests from decode
        # each request is formatted as {request id:{ip:decode's address, rank: decode's rank}}
        requestInfoType = Union[str, int]
        self.decode_requests_dict: Dict[str, Dict[str, requestInfoType]] = {}
        self._decode_requests_lock = threading.Lock()

        # add a pull socket to pull the decode request
        self.ctx = zmq.Context()
        self.input_socket = self.ctx.socket(zmq.PULL)
        self.input_socket.bind(f"tcp://{self.host_ip}:{ZMQ_DECODE_PUSH_REQUEST_TO_PREFILL_PORT}")
        logger.info(f"PrefillConnectorScheduler bind tcp://{self.host_ip}:{ZMQ_DECODE_PUSH_REQUEST_TO_PREFILL_PORT}")
        
        # add decode socket map
        self.decode_zmq_socket_map: Dict[str, zmq.Socket] = {}

        # start a thread to receive requests
        thread_name = "prefill_receives_decode_requests"
        self.receive_decode_requests_thread = threading.Thread(target = self.receive_and_record_decode_requests, daemon = True, name = thread_name)
        self.receive_decode_requests_thread.start()

        self._prefill_pending = {}                # request_id -> payload
        self._prefill_pending_lock = threading.Lock()
        self._prefill_cv = threading.Condition(self._prefill_pending_lock)
        self._prefill_sender_stop = threading.Event()
        self._prefill_sender_thread = threading.Thread(target=self._prefill_sender_loop, daemon=True)
        self._prefill_sender_thread.start()

    def _listen_worker_headers(self):
        while True:
            try:
                msg_str = self.pull_socket.recv_string()
                msg_list = json.loads(msg_str)
                for msg in msg_list:
                    req_id = msg['remote_request_id'] or msg['request_id']
                    headers = msg.get('trace_headers', {})
                    with self._transfer_lock:
                        self.sending_trace_headers[req_id] = headers
            except Exception as e:
                logger.error(f"Failed to receive worker header (P): {e}")
                time.sleep(1)

    def pop_trace_headers(self, req_id: str) -> dict:
        with self._transfer_lock:
            trace_headers = self.sending_trace_headers.pop(req_id, {})
            if trace_headers:
                return trace_headers

    def receive_and_record_decode_requests(self):
        logger.info("Prefill begins to receive decode requests")
        while True:
            try:
                if self.input_socket.poll(timeout = 100) > 0:
                    decode_request_msg = self.input_socket.recv_string()
                    decode_request = json.loads(decode_request_msg)
                    logger.debug(f"Received:{decode_request}")
                    with self._decode_requests_lock:
                        self.decode_requests_dict.update(decode_request)
            except Exception as e:
                logger.error(f"Failed to receive the decode request message")
                
    def _prefill_sender_loop(self):
        poll_interval = 1.0
        while not self._prefill_sender_stop.is_set():
            # wait for pending or timeout
            with self._prefill_cv:
                if not self._prefill_pending:
                    self._prefill_cv.wait(timeout=poll_interval)
                pending_keys = list(self._prefill_pending.keys())

            if not pending_keys:
                continue

            for req_id in pending_keys:
                if self._prefill_sender_stop.is_set():
                    break

                # Try to grab decode request under its own lock
                with self._decode_requests_lock:
                    if req_id not in self.decode_requests_dict:
                        # decode info not ready yet — continue to next pending req
                        continue
                    decode_request = self.decode_requests_dict.pop(req_id)

                decode_ip = decode_request.get("ip")
                decode_rank = decode_request.get("rank")
                if decode_ip is None or decode_rank is None:
                    logger.warning("Bad decode_request for %s: %s", req_id, decode_request)
                    continue

                path = f"tcp://{decode_ip}:{ZMQ_PREFILL_PUSH_INFORMATION_TO_DECODE_BASE_PORT + decode_rank}"

                try:
                    # Ensure socket creation/usage is done in this thread only
                    if path in self.decode_zmq_socket_map:
                        socket_ = self.decode_zmq_socket_map[path]
                    else:
                        socket_ = self.ctx.socket(zmq.PUSH)
                        socket_.connect(path)
                        self.decode_zmq_socket_map[path] = socket_
                        logger.info("Create new socket path:%s", path)

                    # Fetch payload under prefill lock to ensure atomicity
                    with self._prefill_cv:
                        payload = self._prefill_pending.get(req_id)
                        if payload is None:
                            continue

                    data_to_be_sent = {req_id: payload}
                    json_data = json.dumps(data_to_be_sent)
                    socket_.send_string(json_data)
                    logger.info("Send required data for request %s to path:%s", req_id, path)

                    # metadata call if desired
                    try:
                        if hasattr(self, "metadata") and self.metadata is not None:
                            self.metadata.add_new_req(
                                request_id=req_id,
                                local_block_ids=payload.get("remote_block_ids"),
                                kv_transfer_params=payload,
                            )
                    except Exception:
                        logger.exception("metadata.add_new_req failed for %s", req_id)

                    # remove from pending after successful send
                    with self._prefill_cv:
                        self._prefill_pending.pop(req_id, None)

                except Exception as e:
                    logger.error("Failed to send required data for request %s to %s: %s", req_id, path, e)
                    # leave payload in pending for retry later
                    continue

    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> tuple[int, bool]:
        return 0, False

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        pass

    def build_connector_metadata(
            self,
            scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        metadata = DatadistConnectorMetadataPrefill()
        # add requests finish time to metadata, to pass to worker connector
        metadata.requests = {req_id: ReqMetaPrefill(finish_time=finish_time)
                     for req_id, finish_time in self.requests_finish_time.items()}
        self.requests_finish_time.clear()
        return metadata

    def request_finished(
            self,
            request: "Request",
            block_ids: list[int],
            spec_token_ids: Optional[list[int]] = []
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """
        if request.status != RequestStatus.FINISHED_LENGTH_CAPPED:
            return False, None

        delay_free_blocks = len(block_ids) > 0
        # record the finish time of the request
        if delay_free_blocks:
            self.requests_finish_time[request.request_id] = time.monotonic()

        payload = dict(
            remote_block_ids=block_ids,
            remote_cluster_id=self.cluster_id_start,
            remote_host_ip=f"tcp://{self.host_ip}:{self.host_port}",
            spec_token_ids=spec_token_ids,
            remote_dp_rank=self.vllm_config.parallel_config.data_parallel_rank,
            remote_request_id=request.request_id
        )

        if delay_free_blocks:
            # Use the Condition to guard and notify; do NOT nest with the raw lock
            with self._prefill_cv:
                self._prefill_pending[request.request_id] = payload
                # notify background thread that new item available
                self._prefill_cv.notify()
        return delay_free_blocks, payload


class PrefillConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, vllm_config: "VllmConfig", host_ip: str, host_port: str, trace_p_port: str):
        # Metadata.
        self.host_ip = host_ip
        self.host_port = host_port
        self.trace_p_port = trace_p_port
        self.rank = get_tensor_model_parallel_rank()
        if self.rank == 0 and get_pp_group().is_last_rank:
            self.ctx = zmq.Context()
            self.input_socket = self.ctx.socket(zmq.constants.PULL)
            self.input_socket.bind(f"tcp://{self.host_ip}:{self.host_port}")
            logger.info(f"ConnectWorker bind tcp://{self.host_ip}:{self.host_port}")
            self._transfer_lock = threading.Lock()
            self.receive_req_list = []
            thread_name = "prefill_connector_get_pulled_kv_req_list"
            self.thread = threading.Thread(target=self.get_pulled_kv_req_list, daemon=True, name=thread_name)
            self.thread.start()
            dump_thread_to_file(self.thread, thread_name, thread_dump_path)

        # check whether omni attention is enabled
        manager_cls = LLMDataDistManager
        if vllm_config.additional_config and "enable_omni_attn" in vllm_config.additional_config:
            # do import only when necessary
            from omni.accelerators.cache import OmniBiGroupDataDistManager, check_omni_attn_cmd_arg
            use_omni_attn_mgr = check_omni_attn_cmd_arg(vllm_config.additional_config)
            if use_omni_attn_mgr:
                manager_cls = OmniBiGroupDataDistManager
                logger.warning(f"PrefillingConnector is using Omni datadist manager for KV transfer.")
        self.datadist_manager = manager_cls(vllm_config)

        # initialize the dict to save requests finish time
        self.requests_finish_time = dict()

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self.datadist_manager.register_memory(kv_caches)
        self.datadist_manager.register_link()
        pass

    def start_load_kv(self, metadata: DatadistConnectorMetadataPrefill):
        pass

    def get_finished(self, metadata: DatadistConnectorMetadataPrefill) -> tuple[set[str], set[str]]:
        """
        Get requests that are done sending or recving.
        """
        all_done_sending: set[str] = set()
        all_done_recving: set[str] = set()
        if self.rank == 0 and get_pp_group().is_last_rank:
            # Update requests_finish_time with new finish times from metadata
            with self._transfer_lock:
                self.requests_finish_time.update(
                    {req_id: meta.finish_time for req_id, meta in metadata.requests.items()}
                )
                current_time = time.monotonic()
                # Identify requests whose finish time exceeds BLOCK_RELEASE_DELAY
                out_date_reqs = []
                for req_id, finish_time in self.requests_finish_time.items():
                    if current_time - finish_time > BLOCK_RELEASE_DELAY:
                        out_date_reqs.append(req_id)
                    else:
                        # Since the dict is ordered by finish_time, we can break early
                        break
                for req_id in out_date_reqs:
                    logger.warning(
                        f"Request {req_id} is out of date, finish time: {self.requests_finish_time[req_id]}. Freeing blocks now."
                    )
                    all_done_sending.add(req_id)
                    del self.requests_finish_time[req_id]

            if len(self.receive_req_list) == 0:
                return all_done_sending, all_done_recving

            with self._transfer_lock:
                for item in self.receive_req_list:
                    req_id = item.get('remote_request_id')#item['remote_request_id']
                    headers = item.get('trace_headers', {})
                    logger.debug(f"Get_finished: request {req_id}")
                    all_done_sending.add(req_id)
                    # if the request's kv has been received, remove it from requests_finish_time
                    if req_id in self.requests_finish_time:
                        del self.requests_finish_time[req_id]
                self.receive_req_list.clear()

        return all_done_sending, all_done_recving

    def get_pulled_kv_req_list(self):
        path_p = f"tcp://{self.host_ip}:{self.trace_p_port}"
        socket_p = self.ctx.socket(zmq.PUSH)
        socket_p.connect(path_p)
        while True:
            try:
                if self.input_socket.poll(timeout=10) > 0:
                    message = self.input_socket.recv_string()
                    id_list = json.loads(message)  # Parse the received JSON string into a list
                    logger.debug("Received: %s", id_list)
                    with self._transfer_lock:
                        self.receive_req_list.extend(id_list)
                    if os.getenv("PROFILING_NAMELIST", None):
                        json_data = json.dumps(id_list)
                        socket_p.send_string(json_data)
            except Exception as e:
                logger.error("get pulled kv req list failed: %s", e)


class DecodeConnectorScheduler:
    """Implementation of Scheduler side methods"""
    def __init__(self, vllm_config: VllmConfig, trace_d_port: str):
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self._reqs_need_recv: dict[str, tuple[Request, list[int]]] = {}
        self.processed_request: set[str] = set()
        self.ctx = zmq.Context()
        self.zmq_socket_map = {}

        self.host_ip = get_local_ip()
        self.trace_d_port = trace_d_port
        self.recving_trace_headers: dict[str, dict] = {}
        self._transfer_lock = threading.Lock()
        self.ctx = zmq.Context()
        self.pull_socket = self.ctx.socket(zmq.PULL)
        self.pull_socket.bind(f"tcp://{self.host_ip}:{self.trace_d_port}")

        if os.getenv("PROFILING_NAMELIST", None):
            self._listener_thread = threading.Thread(target=self._listen_worker_headers, daemon=True)
            self._listener_thread.start()

        additional_config = vllm_config.additional_config
        if additional_config:
            self.async_pull_kv = additional_config.get("async_pull_kv", False)
        else:
            self.async_pull_kv = False

        if self.async_pull_kv:
            self.context = zmq.Context()
            self.pub = self.context.socket(zmq.PUB)
            kv_rank = self.vllm_config.kv_transfer_config.kv_rank
            self.pub.bind(f"ipc:///tmp/sched-pub-{kv_rank}-{vllm_config.parallel_config.data_parallel_rank_local}")

        # initialize a dict to record remote information
        infoType = Union[None, str, List[int]]
        remoteInfoType = Dict[str, infoType]
        self.prefill_info_dict: Dict[str, remoteInfoType] = {} #prefill info list
        self._prefill_info_dict_lock = threading.Lock()

        # add a pull socket to pull the prefill information
        self.ctx = zmq.Context()
        self.input_socket = self.ctx.socket(zmq.PULL)
        self.local_ip = get_local_ip()
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank_local
        self.input_socket.bind(f"tcp://{self.local_ip}:{ZMQ_PREFILL_PUSH_INFORMATION_TO_DECODE_BASE_PORT + self.dp_rank}")
        logger.info(f"DecodeConnectorScheduler bind tcp://{self.local_ip}:{ZMQ_PREFILL_PUSH_INFORMATION_TO_DECODE_BASE_PORT + self.dp_rank}")

        # add prefill socket map
        self.prefill_zmq_socket_map: Dict[str, zmq:socket] = {}

        # start a thread to receive information
        thread_name = "decode_receives_prefill_information"
        self.receive_prefill_information_thread = threading.Thread(target = self.receive_and_record_prefill_information, daemon = True, name = thread_name)
        self.receive_prefill_information_thread.start()

    def _listen_worker_headers(self):
        while True:
            try:
                msg_str = self.pull_socket.recv_string()
                msg_list = json.loads(msg_str)
                for msg in msg_list:
                    req_id = msg['remote_request_id']
                    headers = msg.get('trace_headers', {})
                    with self._transfer_lock:
                        self.recving_trace_headers[req_id] = headers
            except Exception as e:
                logger.error(f"Failed to receive worker header (D): {e}")
                time.sleep(1)
                
    def receive_and_record_prefill_information(self):
        logger.info("Decode begins to receive prefill information")
        while True:
            try:
                if self.input_socket.poll(timeout = 100) > 0:
                    prefill_information_msg = self.input_socket.recv_string()
                    prefill_information = json.loads(prefill_information_msg)
                    logger.debug(f"Received:{prefill_information}")
                    with self._prefill_info_dict_lock:
                        self.prefill_info_dict.update(prefill_information)
            except Exception as e:
                logger.error(f"Failed to receive the prefill information:{e}")

    def pop_trace_headers(self, req_id: str) -> dict:
        with self._transfer_lock:
            trace_headers = self.recving_trace_headers.pop(req_id, {})
            if trace_headers:
                return trace_headers

    def _send_pulled_kv_req_list(self, path, data):
        if path in self.zmq_socket_map:
            socket = self.zmq_socket_map[path]
        else:
            socket = self.ctx.socket(zmq.PUSH)
            socket.connect(path)
            self.zmq_socket_map[path] = socket
            logger.info(f"create new socket path:{path}")

        try:
            json_data = json.dumps(data)
            socket.send_string(json_data)
            logger.info(f"send string {json_data} path:{path}")
        except Exception as e:
            logger.error(f"Failed to send reqest_id {json_data} to prefill: {e}")

    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> tuple[int, bool]:
        if request.request_id in self.processed_request:
            return 0, False
        params = request.kv_transfer_params
        prefill_ip = params["remote_cluster_id"]
        req_id = params["remote_request_id"]

        path = f"tcp://{prefill_ip}:{ZMQ_DECODE_PUSH_REQUEST_TO_PREFILL_PORT}"
        if path in self.prefill_zmq_socket_map:
            socket_ = self.prefill_zmq_socket_map[path]
        else:
            socket_ = self.ctx.socket(zmq.PUSH)
            socket_.connect(path)
            self.prefill_zmq_socket_map[path] = socket_
            logger.info(f"Create new socket path:{path}")
        try:
            decode_inf = {req_id : {"ip" : self.local_ip , "rank" : self.dp_rank}}
            json_data = json.dumps(decode_inf)
            socket_.send_string(json_data)
            self.prefill_info_dict[req_id] = {}
            logger.info(f"Send request to prefill {path}")
        except Exception as e:
            logger.error(f"Failed to send request to prefill {path}:{e}")
        
        if params is None:
            return 0, False
        logger.debug(
            "DatadistConnector get_num_new_matched_tokens: "
            "num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens, params)

        if num_computed_tokens % self.block_size != 0:
            raise RuntimeError("num_computed_tokens must be divisible by self.block_size")
        rounded_num_prompt_tokens = self._round_up(
            len(request.prompt_token_ids), self.block_size)
        count = max(rounded_num_prompt_tokens - num_computed_tokens, 0)
        return count, count > 0

    def _round_up(self, x: int, y: int) -> int:
        return ((x + y - 1) // y) * y

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        logger.debug(f"Request id {request.request_id}: blocks length is {len(blocks.blocks)}")
        params = request.kv_transfer_params
        logger.debug(
            "DatadistConnector update_state_after_alloc: "
            "num_external_tokens=%s, kv_transfer_params=%s",
            num_external_tokens, params)

        self.processed_request.add(request.request_id)
        if params is not None:
            if params.get("remote_block_ids"):
                if all(p in params for p in ("remote_cluster_id", "remote_host_ip")):
                    self._reqs_need_recv[request.request_id] = (
                        request, blocks.get_unhashed_block_ids())
                else:
                    logger.warning(
                        "Got invalid KVTransferParams: %s.", params)

    def build_connector_metadata(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        metadata = DatadistConnectorMetadata()
        for req_id, (req, blocks) in list(self._reqs_need_recv.items()):
            if req.kv_transfer_params is None:
                logger.warning(f"For request {req_id}: kv_transfer_params now is None")
                continue
            with self._prefill_info_dict_lock:
                val_tmp = self.prefill_info_dict.get(req_id, None)
                if val_tmp:
                    prefill_information = self.prefill_info_dict.pop(req_id)
                else:
                    prefill_information = None
            if prefill_information:
                req.kv_transfer_params = prefill_information
                params = req.kv_transfer_params
                block_ids = None
                if params.get("remote_block_ids"):
                    if all(p in params for p in ("remote_cluster_id", "remote_host_ip")):
                        block_ids = blocks.get_unhashed_block_ids()
                    else:
                        logger.warning("Got invalid KVTransferParams: %s.", params)
                self.processed_request.add(req_id)
                metadata.add_new_req(
                    request_id=req_id,
                    local_block_ids=block_ids,
                    kv_transfer_params=req.kv_transfer_params,
                )
                req.kv_transfer_params = None
                self._reqs_need_recv.pop(req_id, None)
            else:
                continue

        if self.async_pull_kv:
            # Fast-path publish (scheduler_output may be None on fast path)
            if scheduler_output is None and metadata.requests:
                serialized_data = pickle.dumps(metadata)
                self.pub.send(serialized_data)

        return metadata

    def request_finished(
            self,
            request: "Request",
            block_ids: list[int],
            spec_token_ids: Optional[list[int]] = []
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        if request.request_id in self.processed_request:
            self.processed_request.remove(request.request_id)
        if request.status == RequestStatus.FINISHED_ABORTED and request.kv_transfer_params is not None:
            self._send_pulled_kv_req_list(request.kv_transfer_params.get("remote_host_ip"), [{'request_id': request.request_id, 'trace_headers': request.trace_headers or {}}])
        return False, None


class DecodeConnectorWorker:
    """Worker implementation for datadist."""

    def __init__(self, vllm_config: "VllmConfig", host_ip: str, cluster_id_start: int, trace_d_port: str):
        self.vllm_config = vllm_config
        self.cluster_id_start = cluster_id_start
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank_local
        self.tp_rank = get_tensor_model_parallel_rank()
        additional_config = vllm_config.additional_config
        if additional_config:
            self.async_pull_kv = additional_config.get("async_pull_kv", False)
            self.multi_thread_pull_kv = additional_config.get("multi_thread_pull_kv", False)
            self.multi_rank_pull_kv = additional_config.get("multi_rank_pull_kv", False)
        else:
            self.async_pull_kv = False
            self.multi_thread_pull_kv = False
            self.multi_rank_pull_kv = False
        if self.multi_rank_pull_kv:
            self.multi_thread_pull_kv = True
        if vllm_config.parallel_config.tensor_parallel_size > 1 and self.multi_rank_pull_kv:
            raise ValueError("multi_rank_pull_kv are not supported when tp > 1.")

        # check whether omni attention is enabled
        manager_cls = LLMDataDistManager
        if vllm_config.additional_config and "enable_omni_attn" in vllm_config.additional_config:
            # do import only when necessary
            from omni.accelerators.cache import OmniBiGroupDataDistManager, check_omni_attn_cmd_arg
            use_omni_attn_mgr = check_omni_attn_cmd_arg(vllm_config.additional_config)
            if use_omni_attn_mgr:
                manager_cls = OmniBiGroupDataDistManager
                logger.warning(f"DecodeConnector is using Omni datadist manager for KV transfer.")
        self.datadist_manager = manager_cls(vllm_config)

        self._recving_transfers: list = []
        self._done_recving_count: defaultdict[str, int] = defaultdict(lambda: 0)

        self._pull_kv_lock = threading.Lock()
        self.queues = {} # cluster_id -> queue.Queue
        self.threads = {} # cluster_id -> threading.Thread

        self._transfer_lock = threading.Lock()
        self.host_ip = host_ip
        self.trace_d_port = trace_d_port

        self.ctx = zmq.Context()
        self.zmq_socket_map = {}

        if self.async_pull_kv:
            # dp_rank = vllm_config.parallel_config.data_parallel_rank_local
            thread_name = f"async_pull_kv_{self.dp_rank}"
            self.thread_on_fast_path_req = threading.Thread(target=self.on_fast_path_req, daemon=True, name=thread_name)
            self.thread_on_fast_path_req.start()
            logger.warning(f"DecodeConnectorWorker initialized with self.async_pull_kv enabled.")

            # Write thread name and native_id to file
            dump_thread_to_file(self.thread_on_fast_path_req, thread_name, thread_dump_path)

        if self.multi_thread_pull_kv and self.vllm_config.parallel_config.tensor_parallel_size > 1:
            self.tp_sync_path = f"ipc:///tmp/tp-sync-dp{self.vllm_config.parallel_config.data_parallel_rank}"
            if get_tensor_model_parallel_rank() == 0:
                self.input_socket = self.ctx.socket(zmq.constants.PULL)
                self.input_socket.bind(self.tp_sync_path)
                logger.info(f"ConnectWorker bind {self.tp_sync_path}")

                self.tp_sync_req_dict = {}
                thread_name = f"decode_connector_sync_pulled_tp_kvcache_and_send_dp{self.vllm_config.parallel_config.data_parallel_rank}"
                self.sync_thread = threading.Thread(target=self.sync_pulled_tp_kvcache_and_send, daemon=True,
                                                    name=thread_name)
                self.sync_thread.start()
                dump_thread_to_file(self.sync_thread, thread_name, thread_dump_path)

    def sync_pulled_tp_kvcache_and_send(self):
        while True:
            try:
                if self.input_socket.poll(timeout=10) > 0:
                    data = self.input_socket.recv_json()
                    request_id = data[0].get("request_id")
                    remote_host_ip = data[0].get("remote_host_ip")
                    remote_request_id = data[0].get("remote_request_id", None)
                    trace_headers = data[0].get("trace_headers", {})
                    # if request_id not in dict, set to 0, else do nothing
                    self.tp_sync_req_dict.setdefault(request_id, 0)
                    self.tp_sync_req_dict[request_id] += 1
                    logger.debug(f"{request_id} finish pull kv {self.tp_sync_req_dict[request_id]} times.")
                    if self.tp_sync_req_dict[request_id] == self.vllm_config.parallel_config.tensor_parallel_size:
                        self.tp_sync_req_dict.pop(request_id)
                        self._send_pulled_kv_req_list(remote_host_ip, [{'remote_request_id': remote_request_id, 'trace_headers': trace_headers or {}}])
                        with self._transfer_lock:
                            self._recving_transfers.append(request_id)
            except Exception as e:
                logger.error("Sync pulled kv when tp > 1 and send failed: %s", e)

    def on_fast_path_req(self):
        context = zmq.Context()
        sub = context.socket(zmq.SUB)
        kv_rank = self.vllm_config.kv_transfer_config.kv_rank
        sub.connect(f"ipc:///tmp/sched-pub-{kv_rank}-{self.vllm_config.parallel_config.data_parallel_rank_local}")
        sub.setsockopt_string(zmq.SUBSCRIBE, "")

        while True:
            serialized_data = sub.recv()
            metadata = pickle.loads(serialized_data)
            for req_id, meta in metadata.requests.items():
                if (len(meta.local_block_ids) > 0) and (len(meta.remote_block_ids) > 0):
                    self.start_load_kv(metadata)
                    if self.tp_rank == 0:
                        logger.info(
                            "Received fast path request for request %s with "
                            "local_block_ids: %s, remote_block_ids: %s.",
                            req_id,
                            len(meta.local_block_ids),
                            len(meta.remote_block_ids)
                        )

    def worker(self, cluster_id):
        q = self.queues[cluster_id]
        time.sleep(0)
        while True:
            task = q.get()
            if task is None:
                continue
            try:
                self._read_blocks(**task)
            except Exception as e:
                logger.error("KV transfer task failed in thread %s: %s", cluster_id, e)
                patch_data = [{'request_id': task['request_id'], 'trace_headers': task.get('trace_headers', {})}]
                self._send_pulled_kv_req_list(task['remote_host_ip'], patch_data)
                raise RuntimeError(f"Failed to pull kv for request:{task['request_id']} from cluster:{cluster_id}.")
            q.task_done()

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self.datadist_manager.register_memory(kv_caches)
        self.datadist_manager.register_link()
        # put multi-thread_pull_kv and multi_rank_pull_kv related registered_link_infos into queues
        if self.multi_rank_pull_kv or self.multi_thread_pull_kv:
            # In multi_rank_pull_kv mode, we create a thread for each P rank's cluster_id
            logger.info(f" ***** registered_link_infos: {self.datadist_manager.registered_link_infos}")
            for (cluster_id_start, prefill_dp_rank, d_rank), cluster_ids in self.datadist_manager.registered_link_infos.items():
                if d_rank != self.datadist_manager.rank:
                    continue
                for idx_count, cluster_id in enumerate(cluster_ids):
                    with self._pull_kv_lock:
                        if cluster_id in self.queues:
                            continue
                        q = queue.Queue()
                        self.queues[cluster_id] = q
                        thread_name = f"thread_pull_kv_dp_rank_{self.dp_rank}_cluster_id_{cluster_id}"
                        t = threading.Thread(target=self.worker, args=(cluster_id,), daemon=True, name=thread_name)
                        t.start()
                        self.threads[cluster_id] = t
                        logger.debug(f" ***** Created a new thread for pulling kv from cluster {cluster_id}.")

                        # Write thread name and native_id to file
                        dump_thread_to_file(t, thread_name, thread_dump_path)
        else:
            # In single thread pull kv mode, we use a single thread to pull kv
            logger.info(" ***** Using single thread to pull kv.")
            max_concurrents = 1
            self.executor = ThreadPoolExecutor(max_workers=max_concurrents)

        logger.debug("Finish register_kv_caches.")

    # Now go asynchronous pull_kv
    def start_load_kv(self, metadata: DatadistConnectorMetadata):
        logger.debug(f" ***** start_load_kv: {len(metadata.requests)}")
        futures = []
        for req_id, meta in metadata.requests.items():
            # if the local_block_ids is empty, skip pulling kv for the request
            if len(meta.local_block_ids) == 0:
                if self.tp_rank == 0:
                    logger.info(f" ***** Request {req_id} has 0 local blocks, skip load kv.")
                continue
            # If local_block_ids is a flat list of int, omni-attention is not used
            # and we can directly use the local_block_ids and remote_block_ids
            if isinstance(meta.local_block_ids[0], int):
                # local_block_ids (kv blocks in D) is more than remote_block_ids (kv blocks in P)
                # leaded by lookahead num, which is used by eagle and multi step
                if len(meta.remote_block_ids) < len(meta.local_block_ids):
                    meta.local_block_ids = meta.local_block_ids[:len(meta.remote_block_ids)]
                    logger.debug("look ahead token num is greater than 0")
                # If remote_block_ids is more than local_block_ids, we only need the last N remote blocks
                # where N is the number of local blocks
                elif len(meta.remote_block_ids) > len(meta.local_block_ids):
                    meta.remote_block_ids = meta.remote_block_ids[-len(meta.local_block_ids):]
                if self.tp_rank == 0:
                    logger.info(
                        " ***** start_load_kv for request %s "
                        "Num local_block_ids: %s. Num remote_block_ids: %s.",
                        req_id,
                        len(meta.local_block_ids),
                        len(meta.remote_block_ids)
                    )
            # If local_block_ids is a list of lists (e.g., [[], []]), omni-attention is used
            # local_block_ids[0] is a list of local block ids for uncompressed layers
            # local_block_ids[1] is a list of local block ids for compressed layers
            elif isinstance(meta.local_block_ids[0], list):
                # If local_block_ids[0] is a list of lists, we need to ensure that remote_block_ids
                # is a list of lists as well, where each sublist corresponds to the local_block
                meta.remote_block_ids = [meta.remote_block_ids] * len(meta.local_block_ids)
                # If local_block_ids[0] is empty, skip pulling kv for the request
                if len(meta.local_block_ids[0]) == 0:
                    if self.tp_rank == 0:
                        logger.info(f" ***** Request {req_id} has 0 local blocks, skip load kv.")
                    continue
                # remote_block_ids in P is less than local_block_ids[0] in D, 
                # leaded by lookahead num, which is used by eagle and multi step
                elif len(meta.remote_block_ids[0]) < len(meta.local_block_ids[0]):
                    meta.local_block_ids[0] = meta.local_block_ids[0][:len(meta.remote_block_ids[0])]
                    logger.debug("look ahead token num is greater than 0")
                # If remote_block_ids in P is more than local_block_ids[0] in D, we only need the last N remote blocks
                elif len(meta.remote_block_ids[0]) > len(meta.local_block_ids[0]):
                    meta.remote_block_ids[0] = meta.remote_block_ids[0][-len(meta.local_block_ids[0]):]
                if self.tp_rank == 0:
                    logger.info(
                        " ***** start_load_kv for request %s "
                        "Num local_block_ids: %s. Num remote_block_ids: %s.",
                        req_id,
                        len(meta.local_block_ids[0]),
                        len(meta.remote_block_ids[0])
                    )
            # handle the unexpected case where local_block_ids is not a list of int or list of lists
            else:
                logger.error(f"Unexpected type for meta.local_block_ids[0]: {type(meta.local_block_ids[0])}")
                raise RuntimeError(f"Unexpected type for meta.local_block_ids[0]: {type(meta.local_block_ids[0])}")
            cluster_ids = self.datadist_manager.get_real_remote_cluster_ids(meta)
            if self.multi_rank_pull_kv:
                # If multi_rank_pull_kv is enabled, each DP rank will pull kv from multiple P ranks
                # and the cluster_ids are obtained from registered_link_infos
                # If the local_block_ids is a flat list of int, we can directly use it
                # As multi_rank_pull_kv is designed to pull kv from two P ranks,
                # we split the local_block_ids and remote_block_ids into two parts
                if not isinstance(meta.local_block_ids[0], list):
                    block_thre = len(meta.local_block_ids) // 2
                # If the local_block_ids is a flat list of list, only split the blocks for uncompressed layers
                else:
                    block_thre = len(meta.local_block_ids[0]) // 2
                for idx_cluster, cluster_id in enumerate(cluster_ids):
                    if not isinstance(meta.local_block_ids[0], list):
                        if idx_cluster == 0:
                            local_blocks = meta.local_block_ids[:block_thre]
                            remote_blocks = meta.remote_block_ids[:block_thre]
                            len_local_blocks = len(local_blocks)
                        else:
                            local_blocks = meta.local_block_ids[block_thre:]
                            remote_blocks = meta.remote_block_ids[block_thre:]
                            len_local_blocks = len(local_blocks)
                    else:
                        if idx_cluster == 0:
                            # For uncompressed layers, split the local_block_ids[0] and remote_block_ids
                            # For compressed layers, only pull kv from the second P rank
                            local_blocks = [meta.local_block_ids[0][:block_thre], []]
                            # remote_blocks need to be split as well for getting kv blocks for compressed layers in P
                            remote_blocks = [meta.remote_block_ids[0][:block_thre], []]
                            len_local_blocks = len(local_blocks[0])
                        else:
                            local_blocks = [meta.local_block_ids[0][block_thre:], meta.local_block_ids[1]]
                            remote_blocks = [meta.remote_block_ids[0][block_thre:], meta.remote_block_ids[1]]
                            len_local_blocks = len(local_blocks[0])
                    if len_local_blocks > 0:
                        task = {
                            'request_id': req_id,
                            'remote_request_id': meta.remote_request_id,
                            'dst_cluster_id': cluster_id,
                            'local_block_ids': local_blocks,
                            'remote_block_ids': remote_blocks,
                            'remote_host_ip': meta.remote_host,
                            'trace_headers': meta.trace_headers  or {},
                        }
                        logger.warning(f"*********** dst cluster_id is {cluster_id}.")
                        self.queues[cluster_id].put(task)
            elif self.multi_thread_pull_kv:
                task = {
                    'request_id': req_id,
                    'remote_request_id': meta.remote_request_id,
                    'dst_cluster_id': cluster_ids[0],
                    'local_block_ids': meta.local_block_ids,
                    'remote_block_ids': meta.remote_block_ids,
                    'remote_host_ip': meta.remote_host,
                    'trace_headers': meta.trace_headers  or {},
                }

                self.queues[cluster_ids[0]].put(task)
            else:
                # Use ThreadPoolExecutor to handle the task
                future = self.executor.submit(
                    self._read_blocks,
                    local_block_ids=meta.local_block_ids,
                    remote_block_ids=meta.remote_block_ids,
                    dst_cluster_id=cluster_ids[0],
                    request_id=req_id,
                    remote_request_id=meta.remote_request_id,
                    remote_host_ip=meta.remote_host,
                    trace_headers=meta.trace_headers,
                )
                futures.append(future)

        if not self.multi_thread_pull_kv:
            for future in futures:
                future.add_done_callback(handle_exception)

    def _read_blocks(
        self,
        local_block_ids: list[int],
        remote_block_ids: list[int],
        dst_cluster_id: str,
        request_id: str,
        remote_request_id: str,
        remote_host_ip: str,
        trace_headers: Optional[Mapping[str, str]] = None
    ):
        start = time.time()
        if hasattr(self.vllm_config.model_config.hf_config, 'param_sink_with_value'):
            local_block_ids.insert(0, 0)
            remote_block_ids.insert(0, 0)
        self.datadist_manager.pull_kv(remote_block_ids, local_block_ids, dst_cluster_id)

        if self.vllm_config.parallel_config.tensor_parallel_size == 1:
            # tp=1, send to prefill tp rank0 directly.
            self._send_pulled_kv_req_list(remote_host_ip, [{'remote_request_id': remote_request_id, 'trace_headers': trace_headers or {}}])
            with self._transfer_lock:
                self._recving_transfers.append(request_id)
        else:
            if self.multi_thread_pull_kv:
                # tp>1, send to decode to rank0 firstly.
                self._send_pulled_kv_req_list(
                    self.tp_sync_path,
                    {
                        "request_id": request_id,
                        "remote_request_id": remote_request_id,
                        "remote_host_ip": remote_host_ip,
                        'trace_headers': trace_headers or {}
                    }
                )
            else:
                torch.distributed.barrier(group=get_tp_group().cpu_group)
                if get_tensor_model_parallel_rank() == 0:
                    self._send_pulled_kv_req_list(remote_host_ip, [{'remote_request_id': remote_request_id, 'trace_headers': trace_headers or {}}])
                with self._transfer_lock:
                    self._recving_transfers.append(request_id)
        logger.debug(f" ***** read block, req_id:{request_id}, local_block_ids:{local_block_ids}, remote_block_ids:{remote_block_ids}")
        cost = time.time() - start
        if self.tp_rank == 0:
            logger.info(f" ***** read block, req_id:{request_id}, cost:{cost:.6f}")


    def _send_pulled_kv_req_list(self, path, data):
        if path in self.zmq_socket_map:
            socket = self.zmq_socket_map[path]
        else:
            socket = self.ctx.socket(zmq.PUSH)
            socket.connect(path)
            self.zmq_socket_map[path] = socket
            logger.info(f"create new socket path:{path}")

        path_d = f"tcp://{self.host_ip}:{self.trace_d_port}"
        socket_d = self.ctx.socket(zmq.PUSH)
        socket_d.connect(path_d)

        try:
            json_data = json.dumps(data)
            socket.send_string(json_data)
            logger.info(f"send string {json_data} path:{path}")
            if os.getenv("PROFILING_NAMELIST", None):
                socket_d.send_string(json_data)
        except Exception as e:
            logger.error(f"Failed to send reqest_id {json_data} to prefill: {e}")

    def get_finished(self, metadata: DatadistConnectorMetadata) -> tuple[set[str], set[str]]:
        # for decode size, done_sending is no need
        all_done_sending: set[str] = set()
        with self._transfer_lock:
            all_done_recving = self._pop_done_transfers(self._recving_transfers)
        if len(all_done_recving) > 0:
            logger.debug(
                "Get_finished: %s requests done recving", len(all_done_recving))

        return all_done_sending, all_done_recving

    def _pop_done_transfers(self, transfers: list) -> set[str]:
        done_req_ids: set[str] = set()
        for req_id in transfers:
            done_req_ids.add(req_id)
        self._recving_transfers.clear()
        return done_req_ids

def handle_exception(future):
    if future.exception():
        logger.error(f"Exception occurred in future: {future.exception()}")
        raise future.exception()

def dump_thread_to_file(thread, thread_name: str, folder_path: str):

    timeout = 5  # seconds
    start_time = time.time()
    while not hasattr(thread, "native_id"):
        if time.time() - start_time > timeout:
            logger.error(f"Timeout waiting for thread {thread_name} to have native_id.")
            return
        time.sleep(0.005)

    # Ensure the folder exists
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create folder {folder_path}: {e}")
            return

    file_path = os.path.join(folder_path, thread_name)
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(str(thread.native_id))
    except Exception as e:
        logger.error(f"Failed to write thread info to {file_path}: {e}")
