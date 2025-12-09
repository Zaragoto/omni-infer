# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import json
import os
import sys
import pickle
import queue
import socket
import struct
import subprocess
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Tuple, List, Dict, Union
from dataclasses import dataclass, field

import torch
import zmq
import uuid
import msgpack
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size,
    get_tp_group)
from vllm.envs import VLLM_RPC_TIMEOUT
from vllm.logger import init_logger
from vllm.utils import get_open_port

from omni.accelerators.pd.utils import get_config_from_dict_or_env
# from omni.accelerators.pd.llmdatadist_manager import LLMDataDistManager, LLMDataDistConfig

if TYPE_CHECKING:
    from vllm.config import KVTransferConfig
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.request import Request
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.model_executor.models.utils import extract_layer_index

from omni.models.config_loader.loader import model_extra_config

GET_META_MSG = b"get_meta_msg"
logger = init_logger(__name__)

# seconds, used to free blocks after a delay once the request is finished
BLOCK_RELEASE_DELAY = 3000
PER_REQUEST_CONNECTION = 4

BASE_DIR = os.path.dirname(__file__)
OX_PATH = os.environ.get("OX_PATH", os.path.join(BASE_DIR, "ox2/ox")) # call the ox for layerwise kv transfer
OX_LOG_PATH = os.environ.get("OX_LOG_PATH", os.path.join("/data/ox_log"))

# Cluster/P-node configuration
P_NODE_LIST = os.environ.get("P_NODE_LIST", "7.150.13.67,7.150.14.143")

CLUSTER_LIST = [part.strip() for part in P_NODE_LIST.split(';') if part.strip()]
CLUSTER_SIZE = [len(part.split(',')) for part in CLUSTER_LIST][0]

NODE_IP_SPECS = [ip.strip()
              for segment in P_NODE_LIST.split(';')
              for ip in segment.split(',')
              if ip.strip()]

BASE_PORT = int(os.environ.get("BASE_PORT", "15077"))
ZMQ_BASE_PORT = int(os.environ.get("ZMQ_BASE_PORT", "17555"))
ZMQ_DECODE_PUSH_REQUEST_TO_PREFILL_PORT = int(os.environ.get("ZMQ_DECODE_PUSH_REQUEST_TO_PREFILL_PORT", "17556"))
ZMQ_PREFILL_PUSH_INFORMATION_TO_DECODE_BASE_PORT = int(os.environ.get("ZMQ_PREFILL_PUSH_INFORMATION_TO_DECODE_BASE_PORT", "17557"))
ZMQ_LAYER_PORT = int(os.environ.get("ZMQ_LAYER_PORT", "25555")) # port for ox server to receive layerwise kv transfer

P_NODE_PORT_LIST = ';'.join(
    ','.join(f"{h.strip()}:{BASE_PORT}" for h in grp.split(',') if h.strip())
    for grp in P_NODE_LIST.split(';') if grp.strip()
)

@dataclass
class ReqMeta:
    local_block_ids: List[List[int]]
    remote_block_ids: List[int]
    remote_host: str
    remote_cluster_id: str
    spec_token_ids: Optional[List[int]]
    remote_dp_rank: Optional[int]
    remote_request_id: Optional[str]


@dataclass
class ReqMetaPrefill:
    finish_time: float


class DatadistConnectorMetadata(KVConnectorMetadata):
    """Metadata for datadist connector (decode path)."""

    def __init__(self):
        self.requests: Dict[str, ReqMeta] = {}

    def add_new_req(
        self,
        request_id: str,
        local_block_ids: List[List[int]],
        kv_transfer_params: Dict[str, Any],
    ):
        self.requests[request_id] = ReqMeta(
            local_block_ids=local_block_ids,
            remote_block_ids=kv_transfer_params["remote_block_ids"],
            remote_host=kv_transfer_params["remote_host_ip"],
            remote_cluster_id=kv_transfer_params["remote_cluster_id"],
            spec_token_ids=kv_transfer_params["spec_token_ids"],
            remote_dp_rank=kv_transfer_params.get("remote_dp_rank", 0),
            remote_request_id=kv_transfer_params.get("remote_request_id"),
        )


class DatadistConnectorMetadataPrefill(KVConnectorMetadata):
    """Metadata for datadist connector (prefill path)."""

    def __init__(self):
        self.requests: Dict[str, ReqMetaPrefill] = {}

    def add_new_req(
        self,
        request_id: str,
        finish_time: float,
    ):
        self.requests[request_id] = ReqMetaPrefill(finish_time=finish_time)


class DTypeUtils:
    """
    Static helper for converting common dtype strings
    to their corresponding byte sizes.
    """

    # Central registry: dtype alias -> byte size
    _MAP: Dict[str, int] = {
        # 1 byte
        "int8": 1,
        "uint8": 1,
        "byte": 1,
        # 2 bytes
        "int16": 2,
        "uint16": 2,
        "fp16": 2,
        "bf16": 2,
        # 4 bytes
        "int32": 4,
        "uint32": 4,
        "fp32": 4,
        "float32": 4,
        # 8 bytes
        "int64": 8,
        "uint64": 8,
        "fp64": 8,
        "float64": 8,
    }

    @staticmethod
    def size(dtype: str) -> int:
        """
        Return the number of bytes for a given dtype string.

        Args:
            dtype: Case-insensitive dtype alias, e.g. 'bf16', 'FP32'.

        Returns:
            Byte size (positive int).

        Raises:
            ValueError: If the dtype is not recognised.
        """
        key = dtype.lower()
        if key not in DTypeUtils._MAP:
            raise ValueError(
                f"Unsupported data type: {dtype}. "
                f"Supported types: {list(DTypeUtils._MAP.keys())}"
            )
        return DTypeUtils._MAP[key]

    @staticmethod
    def supported() -> list[str]:
        """Return a list of all supported dtype aliases."""
        return list(DTypeUtils._MAP.keys())


class RouterDealerClient:
    def __init__(self, server_address="tcp://localhost:5555"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)

        client_id = f"client_{uuid.uuid4().hex[:8]}".encode('utf-8')
        self.socket.setsockopt(zmq.IDENTITY, client_id)

        self.socket.connect(server_address)
        logger.warning(f"Connected to server at {server_address} with ID: {client_id.decode()}")

    def send_request(self, request_id: str, cluster_id: int, src_id_list: List[int], dst_id_list: List[int], rank_id: int) -> bool:
        try:
            request_data = {
                'request_id': request_id,
                'table_id': rank_id,
                'src_block_ids': src_id_list,
                'dst_block_ids': dst_id_list,
                'cluster_id': cluster_id
            }
            packed_data = msgpack.packb(request_data)
            self.socket.send(packed_data)
            return True
        except Exception as e:
            logger.warning(f"Error sending request {request_id}: {e}")
            return False

    def receive_response(self, timeout: int = 1000) -> Optional[Dict]:
        try:
            if self.socket.poll(timeout, zmq.POLLIN):
                response_data = self.socket.recv()
                response = msgpack.unpackb(response_data)
                return response
        except Exception as e:
            logger.warning(f"Error receiving response: {e}")
        return None

    def close(self):
        self.socket.close()
        self.context.term()
        logger.warning("Client closed")


@dataclass
class _SendItem:
    request_id: str
    cluster_id: int
    src_ids: List[int]
    dst_ids: List[int]
    rank_id: int


@dataclass
class PendingReq:
    request_id: str
    local_block_ids: List[List[int]]
    remote_block_ids: List[int]
    dst_cluster_id: str
    remote_request_id: Optional[str]
    remote_host_ip: str
    dp_rank: int
    t_submit: float = field(default_factory=time.time)
    t_sent: float = 0.0
    t_resp: float = 0.0


class _ZMQSendProxy:
    def __init__(self, send_q: "queue.Queue[_SendItem]"):
        self._q = send_q

    def send_request(self, request_id: str, cluster_id: int, src_id_list: List[int], dst_id_list: List[int], rank_id: int) -> bool:
        self._q.put(_SendItem(
            request_id=request_id,
            cluster_id=int(cluster_id),
            src_ids=list(src_id_list),
            dst_ids=list(dst_id_list),
            rank_id=int(rank_id),
        ))
        return True


class LLMDataDistConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole):
        if vllm_config.kv_transfer_config is None:
            raise RuntimeError("vllm_config.kv_transfer_config cannot be None")

        if vllm_config.model_config.is_deepseek_mla:
            vllm_config.kv_transfer_config.kv_parallel_size = 1
            logger.info("Set kv_parallel_size to 1 when using deepseek MLA model.")

        # self.datadist_config = LLMDataDistConfig(vllm_config, ignore_load_rank=True)
        self.is_prefill = vllm_config.kv_transfer_config.kv_role == "kv_producer"
        if self.is_prefill:
            target_ip = get_local_ip()
            if target_ip not in NODE_IP_SPECS:
                raise ValueError(f"Local IP {target_ip} not found in P_NODE_LIST {P_NODE_LIST}")
            node_idx = NODE_IP_SPECS.index(target_ip)
            self.cluster_id_start = node_idx // CLUSTER_SIZE
            self.host_ip = NODE_IP_SPECS[self.cluster_id_start * CLUSTER_SIZE]
            # Resolve ZMQ port conflicts in multi-P deployments on the same machine.
            self.host_port = get_config_from_dict_or_env(
                vllm_config.kv_transfer_config, "kv_port",
                "VLLM_LLMDATADIST_ZMQ_PORT", "5568", int)
            dp_rank = vllm_config.parallel_config.data_parallel_rank
            self.host_port += dp_rank
        else:
            # in decode instance, these twos are not used, just send some random thing to it
            self.host_ip = "127.0.0.1"
            self.cluster_id_start = 0

        if role == KVConnectorRole.SCHEDULER:
            if self.is_prefill:
                self.connector_scheduler = PrefillConnectorScheduler(
                    vllm_config, self.cluster_id_start, self.host_ip, str(self.host_port))
            else:
                self.connector_scheduler = DecodeConnectorScheduler(vllm_config)
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            if self.is_prefill:
                self.connector_worker = PrefillConnectorWorker(
                    vllm_config, str(self.host_ip), str(self.host_port))
            else:
                self.connector_worker = DecodeConnectorWorker(
                    vllm_config, str(self.host_ip), self.cluster_id_start)
            self.connector_scheduler = None

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> Tuple[int, bool]:
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
            block_ids: List[int],
            spec_token_ids: Optional[List[int]] = None
    ) -> Tuple[bool, Optional[dict]]:
        if self.connector_scheduler is None:
            raise RuntimeError("self.connector_scheduler cannot be None")
        return self.connector_scheduler.request_finished(request, block_ids, spec_token_ids or [])

    ############################################################
    # Worker Side Methods
    ############################################################
    def register_kv_caches(self, kv_pool_mmap_path, data_type, block_len_dtype, omni_cache=None):
        data_type = 'bf16'
        if self.connector_worker is None:
            raise RuntimeError("self.connector_worker cannot be None")
        return self.connector_worker.register_kv_caches(kv_pool_mmap_path, data_type, block_len_dtype, omni_cache)

    def get_finished(self,
                     finished_req_ids: set[str]) -> Tuple[set[str], set[str]]:
        """Get the finished recving and sending requests."""
        if self.connector_worker is None:
            raise RuntimeError("self.connector_worker cannot be None")
        # finished_req_ids is currently not used; we forward internal metadata
        return self.connector_worker.get_finished(self._connector_metadata)

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        if self.connector_worker is None:
            raise RuntimeError("self.connector_worker cannot be None")
        if not isinstance(self._connector_metadata, (DatadistConnectorMetadata, DatadistConnectorMetadataPrefill)):
            raise RuntimeError("self._connector_metadata must be DatadistConnectorMetadata or DatadistConnectorMetadataPrefill")
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Connector does not do layerwise saving."""
        pass

    # this function is called now for layerwise send kv in prefill instance 
    def save_kv_layer(self, layer_name: str,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        if self.connector_worker is None:
            raise RuntimeError("self.connector_scheduler cannot be None")
        if self.is_prefill:
            return self.connector_worker.save_kv_layer(layer_name, attn_metadata)
        else:
            pass

    def wait_for_save(self):
        """Connector does not save explicitly."""
        pass


class PrefillConnectorScheduler:
    """Implementation of Scheduler side methods (prefill)."""

    def __init__(self, vllm_config, cluster_id_start: str, host_ip: str, host_port: str):
        self.vllm_config = vllm_config
        self.cluster_id_start = cluster_id_start
        self.host_ip = host_ip
        self.host_port = host_port
        logger.info("Initializing LLMDataDist Scheduler %s %s %s", cluster_id_start, host_ip, host_port)
        # initialize the dict to save requests finish time
        self.requests_finish_time: Dict[str, float] = {}

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

    def receive_and_record_decode_requests(self):
        logger.info("Prefill begins to receive decode requests")
        while True:
            try:
                if self.input_socket.poll(timeout = 100) > 0:
                    decode_request_msg = self.input_socket.recv_string()
                    decode_request = json.loads(decode_request_msg)
                    logger.warning(f"Received:{decode_request}")
                    with self._decode_requests_lock:
                        self.decode_requests_dict.update(decode_request)
            except Exception as e:
                logger.error(f"Failed to receive the decode request message")


    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> Tuple[int, bool]:
        return 0, False

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        # get block ids from blocks
        logger.warning(f"====== {request=}: kv blocks from update_state_after_alloc : {blocks.get_block_ids()} ======")
        block_ids = blocks.get_block_ids()[0]
        spec_token_ids = []
        # the payload for pull kv metadata should be sent to decode side immediately after allocation
        payload = dict(
            remote_block_ids=block_ids,
            remote_cluster_id=self.cluster_id_start,
            remote_host_ip=f"tcp://{self.host_ip}:{self.host_port}",
            spec_token_ids=spec_token_ids,
            remote_dp_rank=self.vllm_config.parallel_config.data_parallel_rank,
            remote_request_id=request.request_id,
        )
        
        delay_free_blocks = len(block_ids) > 0
        
        if delay_free_blocks:
            with self._prefill_cv:
                self._prefill_pending[request.request_id] = payload
                self._prefill_cv.notify()

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
            block_ids: List[int],
            spec_token_ids: Optional[List[int]] = None
    ) -> Tuple[bool, Optional[dict]]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """
        spec_token_ids = spec_token_ids or []
        if request.status != RequestStatus.FINISHED_LENGTH_CAPPED:
            return False, None

        delay_free_blocks = len(block_ids) > 0
        if delay_free_blocks:
            self.requests_finish_time[request.request_id] = time.monotonic()

        payload = dict(
            remote_block_ids=block_ids,
            remote_cluster_id=self.cluster_id_start,
            remote_host_ip=f"tcp://{self.host_ip}:{self.host_port}",
            spec_token_ids=spec_token_ids,
            remote_dp_rank=self.vllm_config.parallel_config.data_parallel_rank,
            remote_request_id=request.request_id,
        )

        return delay_free_blocks, payload

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


class PrefillConnectorWorker:
    """Implementation of Worker side methods (prefill)."""

    def __init__(self, vllm_config: "VllmConfig", host_ip: str, host_port: str):
        # Metadata.
        self.host_ip = host_ip
        self.host_port = host_port
        self.vllm_config = vllm_config
        self.rank = get_tensor_model_parallel_rank()
        # need tp_rank_local so that each P node can send kv to its corresponding decode node
        self.tp_rank_local = self.rank % (self.vllm_config.parallel_config.tensor_parallel_size // CLUSTER_SIZE)
        if self.rank == 0:
            self.ctx = zmq.Context()
            self.input_socket = self.ctx.socket(zmq.constants.PULL)
            self.input_socket.bind(f"tcp://{self.host_ip}:{self.host_port}")
            logger.info(f"ConnectWorker bind tcp://{self.host_ip}:{self.host_port}")
            self._transfer_lock = threading.Lock()
            self.receive_req_list: List[str] = []
            thread_name = "prefill_connector_get_pulled_kv_req_list"
            self.thread = threading.Thread(target=self.get_pulled_kv_req_list, daemon=True, name=thread_name)
            self.thread.start()

        # # check whether omni attention is enabled
        # from omni.accelerators.cache import OmniBiGroupDataDistManager, check_omni_attn_cmd_arg
        # use_omni_attn_mgr = check_omni_attn_cmd_arg(vllm_config.additional_config)
        # # if use_omni_attn_mgr or False:
        # #     manager_cls = OmniBiGroupDataDistManager
        # #     logger.warning("PrefillingConnector is using Omni datadist manager for KV transfer.")
        # #     self.datadist_manager = manager_cls(vllm_config)
        # # else:
        # manager_cls = LLMDataDistManager
        # self.datadist_manager = manager_cls(vllm_config)

        # initialize the dict to save requests finish time
        self.requests_finish_time: Dict[str, float] = {}

        # each P node starts an zmq PUSH socket to send layerwise kv to ox server
        if self.tp_rank_local == 0:
            ctx_layer = zmq.Context()
            self.sock_layer = ctx_layer.socket(zmq.PUSH)
            self.sock_layer.RCVTIMEO = 5000  # ms
            self.sock_layer.SNDTIMEO = 2000
            self.sock_layer.connect(f"tcp://{self.host_ip}:{ZMQ_LAYER_PORT}")


    def register_kv_caches(self, kv_pool_mmap_path, data_type, block_len_dtype, omni_cache=None):
        logger.warning(f" ======= OX parameters for P server: {kv_pool_mmap_path=}, {data_type=}, {block_len_dtype=}")
        self._start_p_server_kv_transfer(kv_pool_mmap_path, data_type, block_len_dtype, omni_cache)

    def _start_p_server_kv_transfer(self, kv_pool_mmap_path, data_type, block_len_dtype, omni_cache):
        data_type_size = DTypeUtils.size(data_type)
        if self.tp_rank_local == 0:
            cmd = [
                str(OX_PATH),
                "--addr", f"0.0.0.0:{BASE_PORT}",
                "--block-table-shm", str(kv_pool_mmap_path),
                "--num-blocks", str(omni_cache.num_blocks),
                "--num-layers", str(omni_cache.num_layers),
                "--tokens-per-block", str(omni_cache.node_block_size),
                "--dims",  ",".join(map(str, omni_cache.head_sizes)),
                "--zmq-port", str(ZMQ_LAYER_PORT) # add new zmq port parameter for ox server to receive layerwise kv transfer flag
                # "--block-size", str(block_len_dtype * data_type_size), # no block size parameter now
            ]
            logger.warning(f"<<<Executing {cmd}")

            proc = subprocess.Popen(cmd,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    text=True,
                                    bufsize=1)
            q = queue.Queue()

            t_read = threading.Thread(target=stdout_reader, args=(proc.stdout, q))
            t_print = threading.Thread(target=stdout_printer, args=(q, ))
            t_read.daemon = True
            t_print.daemon = True
            t_read.start()
            t_print.start()

            ok, not_ready = _wait_ports(host_ports=[("127.0.0.1", BASE_PORT)], timeout_sec=60)
            if not ok:
                stop_logged_process(proc)
                raise RuntimeError(f"[ERROR] P not ready: {sorted(list(not_ready))}")
            logger.info("[READY] P node is ready")

    def start_load_kv(self, metadata: DatadistConnectorMetadataPrefill):
        pass

    # newly added function for prefill instance to send layerwise kv to ox server
    def save_kv_layer(self, layer_idx: int,
                      attn_metadata: "AttentionMetadata") -> None:
        """ do this operaiton only when tp_rank_local == 0 """
        if self.tp_rank_local == 0:
            block_table_tensor = attn_metadata.prefill.block_table
            block_tables = [sub[sub != 0].tolist() for sub in block_table_tensor]
            if isinstance(block_tables[0], int):
                block_tables = [block_tables]
            for block_table in block_tables:
                if block_table != []:
                    # only "layer_id" and "block_ids" are used in ox server now, other fields need to be removed later
                    cmd = {
                            "table_id": 0,
                            "layer_id": layer_idx,
                            "block_ids": block_table,
                            "rank": 0
                        }
                    packed = msgpack.packb(cmd)
                    try:
                        self.sock_layer.send(packed, zmq.NOBLOCK)
                        logger.warning(f"[ZMQ controller] sent layer {layer_idx} to ox server for {block_table}")
                    except Exception as e:
                        logger.warning("[ZMQ controller] failed to send:", e)
                        break
        else:
            pass

    def get_finished(self, metadata: DatadistConnectorMetadataPrefill) -> Tuple[set[str], set[str]]:
        """
        Get requests that are done sending or recving.
        """
        all_done_sending: set[str] = set()
        all_done_recving: set[str] = set()
        if self.rank == 0:
            # Update requests_finish_time with new finish times from metadata
            with self._transfer_lock:
                self.requests_finish_time.update(
                    {req_id: meta.finish_time for req_id, meta in metadata.requests.items()}
                )
                current_time = time.monotonic()
                # Identify requests whose finish time exceeds BLOCK_RELEASE_DELAY
                out_date_reqs: List[str] = []
                for req_id, finish_time in list(self.requests_finish_time.items()):
                    if current_time - finish_time > BLOCK_RELEASE_DELAY:
                        out_date_reqs.append(req_id)
                for req_id in out_date_reqs:
                    logger.warning(
                        f"Request {req_id} is out of date, finish time: {self.requests_finish_time[req_id]}. Freeing blocks now."
                    )
                    all_done_sending.add(req_id)
                    del self.requests_finish_time[req_id]

            if len(self.receive_req_list) == 0:
                return all_done_sending, all_done_recving

            with self._transfer_lock:
                for req_id in self.receive_req_list:
                    logger.debug(f"Get_finished: request {req_id}")
                    all_done_sending.add(req_id)
                    # if the request's kv has been received, remove it from requests_finish_time
                    if req_id in self.requests_finish_time:
                        del self.requests_finish_time[req_id]
                self.receive_req_list.clear()

        return all_done_sending, all_done_recving

    def get_pulled_kv_req_list(self):
        while True:
            try:
                # pyzmq Socket.poll timeout is in milliseconds; check every 1s
                if self.input_socket.poll(timeout=100) > 0:
                    message = self.input_socket.recv_string()
                    id_list = json.loads(message)  # Parse the received JSON string into a list
                    logger.debug("Received: %s", id_list)
                    with self._transfer_lock:
                        self.receive_req_list.extend(id_list)
            except Exception as e:
                logger.error("get pulled kv req list failed: %s", e)


class DecodeConnectorScheduler:
    """Implementation of Scheduler side methods (decode)."""

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self._reqs_need_recv: Dict[str, Tuple[Request, List[int]]] = {}
        self.processed_request: set[str] = set()

        additional_config = vllm_config.additional_config or {}
        self.async_pull_kv = additional_config.get("async_pull_kv", False)

        if self.async_pull_kv:
            self.context = zmq.Context()
            self.pub = self.context.socket(zmq.PUB)
            self.pub.bind(f"ipc:///tmp/sched-pub-{vllm_config.parallel_config.data_parallel_rank_local}")

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

    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> Tuple[int, bool]:
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

        if params is not None:
            self._reqs_need_recv[request.request_id] = (request, blocks)

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
            block_ids: List[int],
            spec_token_ids: Optional[List[int]] = None
    ) -> Tuple[bool, Optional[dict]]:
        if request.request_id in self.processed_request:
            self.processed_request.remove(request.request_id)
        return False, None


class DecodeConnectorWorker:
    """Worker implementation for datadist (decode)."""

    # _h2d_wait = threading.Event()

    def __init__(self, vllm_config: "VllmConfig", host_ip: str, cluster_id_start: int):
        self.vllm_config = vllm_config
        self.cluster_id_start = cluster_id_start
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank_local
        additional_config = vllm_config.additional_config or {}
        self.async_pull_kv = additional_config.get("async_pull_kv", False)

        # from omni.accelerators.cache import OmniBiGroupDataDistManager, check_omni_attn_cmd_arg
        # use_omni_attn_mgr = check_omni_attn_cmd_arg(vllm_config.additional_config)
        # # if use_omni_attn_mgr or False:
        # #     manager_cls = OmniBiGroupDataDistManager
        # #     logger.warning("DecodeConnector is using Omni datadist manager for KV transfer.")
        # #     self.datadist_manager = manager_cls(vllm_config)
        # # else:
        # manager_cls = LLMDataDistManager
        # self.datadist_manager = manager_cls(vllm_config)

        self._recving_transfers: List[str] = []
        self._done_recving_count: defaultdict[str, int] = defaultdict(lambda: 0)

        self._pull_kv_lock = threading.Lock()
        self.queues: Dict[str, queue.Queue] = {}     # cluster_id -> Queue
        self.threads: Dict[str, threading.Thread] = {}  # cluster_id -> Thread

        self._transfer_lock = threading.Lock()

        self.ctx = zmq.Context()
        self.zmq_socket_map: Dict[str, zmq.Socket] = {}

        logger.info(" ***** Using single thread to pull kv.")
        max_concurrents = 1
        self.executor = ThreadPoolExecutor(max_workers=max_concurrents)

        self.omni_cache = None

        if self.async_pull_kv:
            thread_name = f"async_pull_kv_{self.dp_rank}"
            self.thread_on_fast_path_req = threading.Thread(
                target=self.on_fast_path_req, daemon=True, name=thread_name)
            self.thread_on_fast_path_req.start()
            logger.warning("DecodeConnectorWorker initialized with self.async_pull_kv enabled.")

        self._transfer_lock = getattr(self, "_transfer_lock", threading.Lock())
        self._recving_transfers = getattr(self, "_recving_transfers", [])
        self._endpoint = f"tcp://127.0.0.1:{ZMQ_BASE_PORT}"
        self.zmq_client = None
        self._send_q: "queue.Queue[_SendItem]" = queue.Queue()
        self._resp_thread: Optional[threading.Thread] = None
        self._resp_stop = threading.Event()

        import queue as _q
        self._h2d_q: "_q.Queue[PendingReq]" = _q.Queue(maxsize=1024)
        self._h2d_stop = threading.Event()
        self._h2d_thread: Optional[threading.Thread] = None

        self._pending: Dict[str, PendingReq] = {}
        self._pending_lock = threading.Lock()

        # self.h2d_stream = torch.npu.Stream()

    def on_fast_path_req(self):
        context = zmq.Context()
        sub = context.socket(zmq.SUB)
        sub.connect(f"ipc:///tmp/sched-pub-{self.vllm_config.parallel_config.data_parallel_rank_local}")
        sub.setsockopt_string(zmq.SUBSCRIBE, "")

        while True:
            serialized_data = sub.recv()
            metadata: DatadistConnectorMetadata = pickle.loads(serialized_data)
            for req_id, meta in metadata.requests.items():
                if (len(meta.local_block_ids) > 0) and (len(meta.remote_block_ids) > 0):
                    self.start_load_kv(metadata)
                    logger.info(
                        "Received fast path request for request %s with "
                        "local_block_ids: %s, remote_block_ids: %s.",
                        req_id,
                        len(meta.local_block_ids),
                        len(meta.remote_block_ids)
                    )

    def register_kv_caches(self, kv_pool_mmap_path, data_type, block_len_dtype, omni_cache):
        logger.warning(f" ======= OX parameters for D client: {omni_cache.num_blocks=};{kv_pool_mmap_path=}, {data_type=}, {block_len_dtype=}")
        # kv_dims_scale = [kv_dim_tmp * omni_cache.head_size for kv_dim_tmp in omni_cache.head_size_ratio]
        # kv_dims_int = [kv_dim_tmp // sum(omni_cache.head_size_ratio) for kv_dim_tmp in kv_dims_scale]
        # kv_dims = ", ".join(map(str, kv_dims_int))
        if omni_cache.dp_local_rank == 0:
            self.block_len_dtype = block_len_dtype
            end_port = f"{ZMQ_BASE_PORT + omni_cache.dp_local_rank}"
            data_type_size = DTypeUtils.size(data_type)
            cmd = [
                str(OX_PATH),
                "--shard-list", str(P_NODE_PORT_LIST),
                "--zmq-port", end_port,
                "--block-table-shm", str(kv_pool_mmap_path),
                "--num-block-tables", str(omni_cache.dp_world_size_local),
                "--num-blocks", str(omni_cache.num_blocks),
                "--num-layers", str(omni_cache.num_layers),
                "--tokens-per-block", str(omni_cache.block_size),
                "--num-layers", str(omni_cache.num_layers),
                "--num-connections-per-req", str(PER_REQUEST_CONNECTION),
                "--dims",  ",".join(map(str, omni_cache.head_sizes)),
                # "--block-size", str(block_len_dtype * data_type_size),
                # "--kv-layers", str(omni_cache.num_layers),
                # "--kv-tokens", str(omni_cache.block_size),
                # "--kv-dims", kv_dims
            ]

            proc = subprocess.Popen(cmd,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT,
                                        text=True,
                                        bufsize=1)
            q = queue.Queue()

            t_read = threading.Thread(target=stdout_reader, args=(proc.stdout, q))
            t_print = threading.Thread(target=stdout_printer, args=(q, OX_LOG_PATH))
            t_read.daemon = True
            t_print.daemon = True
            t_read.start()
            t_print.start()
        self.zmq_client = None
        self.omni_cache = omni_cache


    # Now go asynchronous pull_kv
    def start_load_kv(self, metadata: DatadistConnectorMetadata):
        logger.info(f" ***** start_load_kv: {len(metadata.requests)}")
        futures = []
        for req_id, meta in metadata.requests.items():
            # if the local_block_ids is empty, skip pulling kv for the request
            if len(meta.local_block_ids) == 0:
                logger.info(f" ***** Request {req_id} has 0 local blocks, skip load kv.")
                continue
            # If local_block_ids is a flat list of int, omni-attention is not used
            # and we can directly use the local_block_ids and remote_block_ids
            if isinstance(meta.local_block_ids[0], int):
                # Adjust for lookahead tokens (eagle/multistep)
                if len(meta.remote_block_ids) < len(meta.local_block_ids):
                    meta.local_block_ids = meta.local_block_ids[:len(meta.remote_block_ids)]
                    logger.debug("look ahead token num is greater than 0")
                # If remote_block_ids is more than local_block_ids, we only need the last N remote blocks
                # where N is the number of local blocks
                elif len(meta.remote_block_ids) > len(meta.local_block_ids):
                    meta.remote_block_ids = meta.remote_block_ids[-len(meta.local_block_ids):]
                logger.info(
                    " ***** start_load_kv for request %s "
                    "Num local_block_ids: %s. Num remote_block_ids: %s.",
                    req_id,
                    len(meta.local_block_ids),
                    len(meta.remote_block_ids)
                )
            # If local_block_ids is a list of lists (omni-attention case)
            elif isinstance(meta.local_block_ids[0], List):
                # If local_block_ids[0] is a list of lists, we need to ensure that remote_block_ids
                # is a list of lists as well, where each sublist corresponds to the local_block
                meta.remote_block_ids = [meta.remote_block_ids] * len(meta.local_block_ids)
                # If local_block_ids[0] is empty, skip pulling kv for the request
                if len(meta.local_block_ids[0]) == 0:
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

            # cluster_ids = self.datadist_manager.get_real_remote_cluster_ids(meta)
            # Use ThreadPoolExecutor to handle the task
            # TODO:now not support omni-attention case yet
            future = self.executor.submit(
                self._read_blocks,
                local_block_ids=meta.local_block_ids,
                remote_block_ids=meta.remote_block_ids if isinstance(meta.remote_block_ids[0], int) else meta.remote_block_ids[0],
                dst_cluster_id=meta.remote_cluster_id,
                request_id=req_id,
                remote_request_id=meta.remote_request_id,
                remote_host_ip=meta.remote_host,
            )
            futures.append(future)

        for future in futures:
            future.add_done_callback(handle_exception)

    def _read_blocks(
        self,
        local_block_ids: List[List[int]],
        remote_block_ids: List[int],
        dst_cluster_id: str,
        request_id: str,
        remote_request_id: Optional[str],
        remote_host_ip: str
    ):
        # only for debug, need to be removed after a stable version is ready
        logger.warning(f"++++++ {local_block_ids=}; {remote_block_ids=}; {dst_cluster_id=}; {request_id=}; {remote_request_id=}; {remote_host_ip=} ++++++")
        logger.warning("\n========== PY DEBUG (_read_blocks entry) ==========")
        logger.warning(">>> PY DEBUG: sending to ox:", {
            "local_block_ids": local_block_ids,
            "remote_block_ids": remote_block_ids,
            "dst_cluster_id": dst_cluster_id,
            "request_id": request_id,
            "remote_request_id": remote_request_id,
            "remote_host_ip": remote_host_ip,
        })
        logger.warning(">>> PY DEBUG TYPES:", {
            "type_local_block_ids": type(local_block_ids),
            "type_remote_block_ids": type(remote_block_ids),
            "type_dst_cluster_id": type(dst_cluster_id),
            "type_request_id": type(request_id),
            "type_remote_request_id": type(remote_request_id),
        })
        logger.warning("===================================================\n")
        start = time.time()

        logger.warning(" ***** AAAA read block-send request: req_id:%s", request_id)
        self._ensure_resp_thread_started()

        with self._pending_lock:
            self._pending[request_id] = PendingReq(
                request_id=request_id,
                local_block_ids=local_block_ids,
                remote_block_ids=remote_block_ids,
                dst_cluster_id=dst_cluster_id,
                remote_request_id=remote_request_id,
                remote_host_ip=remote_host_ip,
                dp_rank=self.omni_cache.dp_local_rank,
                t_submit=start,
            )

        # only for debug, need to be removed after a stable version is ready
        final_payload={
            "request_id": request_id,
            "cluster_id": int(dst_cluster_id),
            "src_id_list": remote_block_ids,
            "dst_id_list": local_block_ids[0],
            "rank_id": self.omni_cache.dp_local_rank,
        }
        logger.warning("\n========== PY FINAL SENT TO OX ==========")
        logger.warning(final_payload)
        logger.warning("FINAL TYPES:", {k: type(v) for k, v in final_payload.items()})
        logger.warning("\n=========================================")

        logger.warning(" ***** Before read block-send request: req_id:%s", request_id)

        self.zmq_client.send_request(
            request_id=request_id,
            cluster_id=int(dst_cluster_id),
            src_id_list=remote_block_ids,
            dst_id_list=local_block_ids[0],
            rank_id=self.omni_cache.dp_local_rank,
        )

        enqueue_cost = time.time() - start
        logger.warning(" ***** read block-send request (enqueued): req_id:%s, cost:%.6f",
                       request_id, enqueue_cost)

    def _ensure_resp_thread_started(self):
        if not (self._resp_thread and self._resp_thread.is_alive()):
            self._resp_stop.clear()
            self._resp_thread = threading.Thread(
                target=self._get_zmq_response, name="ZMQ-Recv-Thread", daemon=True
            )
            self._resp_thread.start()
            self.zmq_client = _ZMQSendProxy(self._send_q)
            logger.info("ZMQ receive thread started at %s", self._endpoint)

        if not (self._h2d_thread and self._h2d_thread.is_alive()):
            self._h2d_stop.clear()
            self._h2d_thread = threading.Thread(
                target=self._h2d_worker, name="H2D-Worker", daemon=True
            )
            self._h2d_thread.start()
            logger.info("H2D worker thread started")

    def _get_zmq_response(self):
        client = RouterDealerClient(self._endpoint)
        try:
            while not self._resp_stop.is_set():
                # send queued requests
                while True:
                    try:
                        item: _SendItem = self._send_q.get_nowait()
                    except queue.Empty:
                        break
                    t0 = time.time()
                    ok = client.send_request(
                        request_id=item.request_id,
                        cluster_id=item.cluster_id,
                        src_id_list=item.src_ids,
                        dst_id_list=item.dst_ids,
                        rank_id=item.rank_id,
                    )
                    if not ok:
                        logger.error("Send failed for req_id=%s", item.request_id)
                        with self._pending_lock:
                            self._pending.pop(item.request_id, None)
                    else:
                        logger.warning("Sent req_id=%s in %.6f s", item.request_id, time.time() - t0)
                    self._send_q.task_done()

                resp = client.receive_response(timeout=50)
                if resp is None:
                    continue
                if not isinstance(resp, dict):
                    raise RuntimeError(f"Invalied responds format!")

                req_id = resp.get("request_id")
                if not isinstance(req_id, str):
                    raise RuntimeError(f"Invalied responds format!")

                # parse layer tag if provided in request id (format you used elsewhere)
                layer_id = None
                if "#L" in req_id:
                    req_id, layer_part = req_id.split("#L", 1)
                    try:
                        layer_id = int(layer_part)
                    except Exception:
                        layer_id = None

                success = bool(resp.get("success"))

                if not req_id:
                    logger.error("Received response without request_id: %s", resp)
                    continue

                with self._pending_lock:
                    ctx = self._pending.get(req_id)
                    if ctx:
                        ctx.t_resp = time.time()

                if not ctx:
                    logger.warning("Orphan response for unknown req_id=%s", req_id)
                    continue

                # keep original timing log
                self._log_network_timing(ctx)

                if not success:
                    logger.error("Failed to pull kv for request %s", req_id)
                    with self._pending_lock:
                        self._pending.pop(req_id, None)
                    continue
                else:
                    if layer_id is not None and layer_id >= 0:
                        logger.warning(f"======= done ox pull kv for layer {layer_id} for req {req_id} ======")

                # - If this response contains a layer tag (#L<layer>), do per-layer H2D and track progress.
                # - If there's no layer tag, IGNORE it (per your request).
                try:
                    if layer_id is not None and layer_id >= 0:
                        # Per-layer response: perform H2D for this layer and try finalize.
                        try:
                            self._post_success_layer(ctx, layer_id)
                        except Exception as e:
                            logger.exception("Per-layer H2D failed for req_id=%s layer=%s: %s", req_id, layer_id, e)
                        # we handled a per-layer response; continue loop
                        continue
                    else:
                        # No layer tag -> per user request we ignore final responses entirely.
                        logger.debug("Ignoring final/no-layer ZMQ response for req_id=%s (per configuration)", req_id)
                        continue
                except Exception as e:
                    logger.exception("Error handling ZMQ response for req_id=%s layer=%s: %s", req_id, layer_id, e)
                    # don't crash the recv loop
        finally:
            client.close()

    def _post_success_layer(self, ctx: 'PendingReq', layer_id: int):
        """
        Perform H2D copy for a single layer of ctx.
        After success, mark layer done and try finalize if all layers finished.
        This function does the synchronous synchronize_h2d for the single layer.
        """
        # Wait for any global H2D precondition as in full _post_success
        # DecodeConnectorWorker._h2d_wait.wait()
        t_h2d_start = time.time()

        # Extract per-layer block ids from ctx.local_block_ids
        try:
            if not hasattr(ctx, "local_block_ids") or ctx.local_block_ids is None:
                raise RuntimeError("ctx.local_block_ids missing")
        except Exception as e:
            logger.exception("Cannot determine layer block ids for req_id=%s layer=%s: %s",
                             getattr(ctx, "request_id", "<unknown>"), layer_id, e)
            return

        # Attempt synchronize for that layer
        try:
            self.omni_cache.synchronize_h2d_layerwise(ctx.local_block_ids, layer_id)
            t_h2d_end = time.time()
            logger.warning(" ***** Time cost of per-layer synchronize_h2d is %.3f ms (req_id:%s layer:%d)",
                           (t_h2d_end - t_h2d_start) * 1000.0,
                           getattr(ctx, "request_id", "<unknown>"),
                           layer_id)
        except Exception as e:
            logger.exception("synchronize_h2d failed for req_id=%s layer=%s: %s",
                             getattr(ctx, "request_id", "<unknown>"), layer_id, e)
            return

        # mark layer copied on ctx (thread-safe with pending_lock)
        try:
            with self._pending_lock:
                if not hasattr(ctx, "_layers_done") or ctx._layers_done is None:
                    ctx._layers_done = set()
                ctx._layers_done.add(layer_id)
                logger.warning(f"======= done h2d copy for layer {layer_id} ======")

                # lazily record total layers if available
                if not hasattr(ctx, "_layers_total") or ctx._layers_total is None:
                    ctx._layers_total = self.omni_cache.num_layers

                # try to finalize if all layers done
                self._maybe_finalize(ctx)
        except Exception:
            logger.exception("Failed to record per-layer completion for req_id=%s layer=%s",
                             getattr(ctx, "request_id", "<unknown>"), layer_id)

    def _maybe_finalize(self, ctx: 'PendingReq'):
        """
        If all expected layers have been copied, perform finalization actions:
        - send pulled kv req list (if remote_request_id present)
        - append ctx.request_id to _recving_transfers under _transfer_lock
        - remove ctx from pending and prebuilt tables
        This function is idempotent and thread-safe (holds _pending_lock when modifying shared state).
        """
        req_id = getattr(ctx, "request_id", None)
        if req_id is None:
            return False

        # We require a known total or derive it from ctx._layers_total; if unknown, we cannot finalize.
        total_layers = getattr(ctx, "_layers_total", None)
        layers_done = getattr(ctx, "_layers_done", set())

        if total_layers is None:
            # cannot decide yet; do nothing
            return False

        if not isinstance(layers_done, (set, list)):
            return False

        if len(layers_done) < total_layers:
            # not all layers done yet
            return False

        # At this point, all layers are done -> perform final actions once.
        # Use a flag to make this idempotent.
        if getattr(ctx, "_finalized", False):
            return True

        try:
            # mark finalized early to avoid races
            ctx._finalized = True

            tp_size = self.vllm_config.parallel_config.tensor_parallel_size
            if tp_size == 1:
                if ctx.remote_request_id is not None:
                    try:
                        self._send_pulled_kv_req_list(ctx.remote_host_ip, [ctx.remote_request_id])
                    except Exception:
                        logger.exception("Failed _send_pulled_kv_req_list for req_id=%s", req_id)
                with self._transfer_lock:
                    self._recving_transfers.append(ctx.request_id)
            else:
                # barrier & sending as original
                torch.distributed.barrier(group=get_tp_group().cpu_group)
                if get_tensor_model_parallel_rank() == 0 and ctx.remote_request_id is not None:
                    try:
                        self._send_pulled_kv_req_list(ctx.remote_host_ip, [ctx.remote_request_id])
                    except Exception:
                        logger.exception("Failed _send_pulled_kv_req_list in TP mode for req_id=%s", req_id)
                with self._transfer_lock:
                    self._recving_transfers.append(ctx.request_id)

            # cleanup pending and prebuilt tables
            with self._pending_lock:
                self._pending.pop(req_id, None)

            logger.debug("Finalized req_id=%s after per-layer copies", req_id)
            return True
        except Exception as e:
            logger.exception("Failed to finalize req_id=%s after per-layer copies: %s", req_id, e)
            # if finalization failed, unset finalized flag to allow retry
            try:
                ctx._finalized = False
            except Exception:
                pass
            return False

    def _h2d_worker(self):
        while not self._h2d_stop.is_set():
            try:
                ctx: PendingReq = self._h2d_q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                # Existing full-path behavior (unchanged)
                self._post_success(ctx)
            except Exception as e:
                logger.exception("H2D worker error on req_id=%s: %s", ctx.request_id, e)
            finally:
                with self._pending_lock:
                    self._pending.pop(ctx.request_id, None)
                if hasattr(self, "_prebuilt_block_tables"):
                    self._prebuilt_block_tables.pop(ctx.request_id, None)
                self._h2d_q.task_done()

    def _log_network_timing(self, ctx: 'PendingReq'):
        t_submit = ctx.t_submit
        t_sent = ctx.t_sent if ctx.t_sent > 0 else t_submit
        t_resp = ctx.t_resp if ctx.t_resp > 0 else time.time()

        cost_submit_to_send = (t_sent - t_submit) * 1000.0
        cost_send_to_resp = (t_resp - t_sent) * 1000.0
        cost_submit_to_resp = (t_resp - t_submit) * 1000.0

        num_blocks = len(ctx.local_block_ids[0]) if ctx.local_block_ids else 0
        logger.warning(
            " ***** Pull kv timing (network only): req_id:%s, num_blocks:%d, "
            "submit->send: %.3f ms, send->resp: %.3f ms, submit->resp: %.3f ms",
            ctx.request_id, num_blocks, cost_submit_to_send, cost_send_to_resp, cost_submit_to_resp
        )

    def _post_success(self, ctx: 'PendingReq'):
        # unchanged full-path behavior (kept as in your original code)
        t_submit = ctx.t_submit
        t_resp = ctx.t_resp if ctx.t_resp > 0 else time.time()

        if self.omni_cache is None or self.omni_cache.device_cache is None:
            raise RuntimeError("Error! omni_cache is None or device_cache is None.")

        # DecodeConnectorWorker._h2d_wait.wait()
        t_h2d_start = time.time()

        # execute full synchronize (this may re-copy already-copied layers; acceptable but can be optimized later)
        self.omni_cache.synchronize_h2d(ctx.local_block_ids, CLUSTER_SIZE)

        t_h2d_end = time.time()
        logger.warning(" ***** Time cost of decode synchronize_h2d is %.3f ms (req_id:%s)",
                       (t_h2d_end - t_h2d_start) * 1000.0, ctx.request_id)

        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        if tp_size == 1:
            if ctx.remote_request_id is not None:
                self._send_pulled_kv_req_list(ctx.remote_host_ip, [ctx.remote_request_id])
            with self._transfer_lock:
                self._recving_transfers.append(ctx.request_id)
        else:
            torch.distributed.barrier(group=get_tp_group().cpu_group)
            if get_tensor_model_parallel_rank() == 0 and ctx.remote_request_id is not None:
                self._send_pulled_kv_req_list(ctx.remote_host_ip, [ctx.remote_request_id])
            with self._transfer_lock:
                self._recving_transfers.append(ctx.request_id)

        total_cost = (time.time() - t_submit)
        logger.warning(" ***** read block total: req_id:%s, cost:%.6f s", ctx.request_id, total_cost)

    def stop_zmq_thread(self, wait: bool = True):
        self._resp_stop.set()
        self._h2d_stop.set()
        # DecodeConnectorWorker._h2d_wait.set()
        if self._resp_thread and wait:
            self._resp_thread.join(timeout=2.0)
        if self._h2d_thread and wait:
            self._h2d_thread.join(timeout=2.0)
        self._resp_thread = None
        self._h2d_thread = None

    def _send_pulled_kv_req_list(self, path: str, data: List[str]):
        if path in self.zmq_socket_map:
            socket_ = self.zmq_socket_map[path]
        else:
            socket_ = self.ctx.socket(zmq.PUSH)
            socket_.connect(path)
            self.zmq_socket_map[path] = socket_
            logger.info("create new socket path:%s", path)

        try:
            json_data = json.dumps(data)
            socket_.send_string(json_data)
            logger.info("send string %s path:%s", json_data, path)
        except Exception as e:
            logger.error("Failed to send request_ids to prefill: %s", e)

    def get_finished(self, metadata: DatadistConnectorMetadata) -> Tuple[set[str], set[str]]:
        # for decode side, done_sending is not needed
        all_done_sending: set[str] = set()
        with self._transfer_lock:
            all_done_recving = self._pop_done_transfers(self._recving_transfers)
        if len(all_done_recving) > 0:
            logger.debug("Get_finished: %s requests done recving", len(all_done_recving))

        return all_done_sending, all_done_recving

    def _pop_done_transfers(self, transfers: List[str]) -> set[str]:
        done_req_ids: set[str] = set()
        for req_id in transfers:
            done_req_ids.add(req_id)
        transfers.clear()
        return done_req_ids

def handle_exception(future):
    if future.exception():
        logger.error("Exception occurred in future: %s", future.exception())
        # Re-raise on the caller thread if someone waits on the future elsewhere
        raise future.exception()

def stop_logged_process(proc, timeout=5.0):
    try:
        if proc.poll() is None:
            proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=timeout)
    finally:
        try:
            if proc.stdout:
                proc.stdout.close()
        except Exception:
            pass
        try:
            if proc.stderr:
                proc.stderr.close()
        except Exception:
            pass

def _wait_ports(host_ports, timeout_sec=1000, interval=0.1):
    remaining = set(host_ports)
    deadline = time.time() + timeout_sec

    def can_connect(h, p, to):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(to)
        try:
            s.connect((h, p))
            return True
        except Exception:
            return False
        finally:
            try:
                s.close()
            except Exception:
                pass

    while remaining and time.time() < deadline:
        ready = [hp for hp in list(remaining) if can_connect(hp[0], hp[1], interval)]
        for hp in ready:
            remaining.discard(hp)
        if remaining:
            time.sleep(interval)
    return len(remaining) == 0, remaining

def stdout_reader(pipe, q):
    for line in iter(pipe.readline, ''):
        q.put(line)
    q.put(None)
    pipe.close()

def stdout_printer(q: queue.Queue,
                   file_path: Optional[str] = None) -> None:
    file_obj = None
    try:
        if file_path is not None:
            os.makedirs(file_path, exist_ok=True)
            log_file = os.path.join(
                file_path,
                f'ox_log_d_client.log'
                # f'ox_log_{datetime.now():%Y%m%d_%H%M%S}.log'
            )

            file_obj = open(log_file, 'w', buffering=1)

        while True:
            try:
                line = q.get(timeout=0.1)
            except queue.Empty:
                continue
            if line is None:
                break

            if file_obj:
                file_obj.write(line)
                file_obj.flush()
            else:
                sys.__stdout__.write(line)
                sys.__stdout__.flush()
    finally:
        if file_obj is not None:
            file_obj.close()

def decode_h2d_trigger():
    DecodeConnectorWorker._h2d_wait.set()
    DecodeConnectorWorker._h2d_wait.clear()

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip