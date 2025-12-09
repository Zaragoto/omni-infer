from typing import List, Tuple, Dict
import threading
import time
from collections import deque

import torch
import numpy as np
from vllm.config import VllmConfig
from vllm.distributed import get_tp_group, get_pp_group
from vllm.logger import logger

from ems import Ems, CcKvOption, EmsException, EmsErrorCode, EmsConfig, CcConfig_v1
from ems.cc_v1.cc_config import KVCacheType

from omni.adaptors.vllm.ems.ems_env import EmsEnv
from omni.adaptors.vllm.ems.zmq_comm import ZmqComm, CommMethod, SocketType


class EmsAdapter:
    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.is_mla = vllm_config.model_config.use_mla
        logger.info(f"[EMS] EmsAdapter init with vllm_config: {vllm_config}, is_mla: {self.is_mla}")

        self.rank = vllm_config.parallel_config.rank
        self.world_size = vllm_config.parallel_config.world_size
        self.block_size = vllm_config.cache_config.block_size

        self.load_futures = {}
        self.save_futures = {}

        self.zmq_cfg = self.get_zmq_cfg()
        self.zmq_comm = None
        self.init_zmq_comm()

        self._inited = False
        self._need_reinit = False
        self._registered = False
        self._pending_kv_caches = None

        self.ems_cfg = self.get_ems_cfg()
        self.context_caching = None
        self.init_ems()

        self.task_manager = None
        self.init_task_manager()


    def get_zmq_cfg(self, comm_method="tcp", zmq_ip="127.0.0.1", zmq_port="50500") -> Dict:
        local_dp_rank = self.vllm_config.parallel_config.data_parallel_rank_local
        comm_method = self.vllm_config.kv_transfer_config.kv_connector_extra_config.get("comm_method", comm_method)
        zmq_ip = self.vllm_config.kv_transfer_config.kv_connector_extra_config.get("ems_zmq_ip", zmq_ip)
        zmq_port = self.vllm_config.kv_transfer_config.kv_connector_extra_config.get("ems_zmq_port", zmq_port)

        used_port = str(int(zmq_port) + local_dp_rank)
        socket_type = SocketType.Receiver if self.rank == 0 else SocketType.Sender
        if comm_method == "tcp":
            used_comm_method = CommMethod.TCP
        elif comm_method == "ipc":
            used_comm_method = CommMethod.IPC
        else:
            logger.error(f"[EMS] undefined ZMQ communication method: \"{comm_method}\", use TCP instead.")
            used_comm_method = CommMethod.TCP

        zmq_cfg = {
                "ip": zmq_ip,
                "port": used_port,
                "dp_rank": local_dp_rank,
                "comm_method": used_comm_method,
                "socket_type": socket_type,
        }
        return zmq_cfg


    def get_ems_cfg(self) -> EmsConfig:
        tp_group = get_tp_group()
        pp_group = get_pp_group()
        cc_config_v1 = CcConfig_v1(
            rank_id=tp_group.rank,
            device_id=tp_group.local_rank,
            model_id=EmsEnv.model_id,
            tp_world_size=tp_group.world_size,
            pp_world_size=pp_group.world_size,
            rank_in_tp_group=tp_group.rank_in_group,
            rank_in_pp_group=pp_group.rank_in_group,
            llm_engine=f"{EmsEnv.llm_engine}@{EmsEnv.service_name}"
        )
        if self.is_mla:
            cc_config_v1.kvcache_type = KVCacheType.MLA

        ems_cfg = EmsConfig(
            access_id=EmsEnv.access_id,
            access_key=EmsEnv.access_key,
            cc_config_v1=cc_config_v1
        )
        return ems_cfg


    def init_zmq_comm(self) -> None:
        try:
            self.zmq_comm = ZmqComm(
                ip=self.zmq_cfg["ip"],
                port=self.zmq_cfg["port"],
                dp_rank=self.zmq_cfg["dp_rank"],
                comm_method=self.zmq_cfg["comm_method"],
                socket_type=self.zmq_cfg["socket_type"],
            )
            logger.info(f"[EMS] init ZMQ communication succeeded, "
                        f"address: \"{self.zmq_comm.addr}\", role: {self.zmq_comm.socket_type.name}).")

            if self.rank == 0:
                self.zmq_thread_lock = threading.Lock()
                self.req_info = {}
                self.rank0_thread = threading.Thread(target=self.rank0_task, daemon=True, name="ems_worker_comm")
                self.rank0_thread.start()
                logger.info(f"[EMS] ZMQ communication subthread \"ems_worker_comm\" started on rank 0.")

        except Exception as e:
            logger.error(f"[EMS] init ZMQ communication failed, error: {e}.")
            raise


    def init_ems(self) -> None:
        try:
            Ems.init(self.ems_cfg)
            self.context_caching = Ems.get_cc()
            self._inited = True
            logger.info(f"[EMS][Init] EmsConnector init succeed, EMS ready.")
        except EmsException as e:
            if e.status_code() == EmsErrorCode.EMS_RECOVERD_ERROR:
                self._need_reinit = True
            logger.error(f"[EMS][Init] EmsConnector init fail, error: {e}.")
        if not self._inited and self.context_caching:
            logger.warning(f"[EMS][Init][degraded] EmsConnector init failed but get cc, reason=EMS not ready.")

    def init_task_manager(self) -> None:
        try:
            self.task_manager = PeriodicTaskManager(check_fn=self._check_health)
            logger.info(f"[EMS] init periodic task manager succeeded.")
        except Exception as e:
            logger.error(f"[EMS] init periodic task manager failed, error: {e}.")
            raise


    def rank0_task(self) -> None:
        while True:
            try:
                res = self.zmq_comm.recv()
            except Exception as e:
                logger.error(f"[EMS] ZMQ recv() failed, error: {e}.")
            else:
                if res is not None:
                    self.update_req_info(res)


    def update_req_info(self, data) -> None:
        with self.zmq_thread_lock:
            for req_id, value in data["item"]:
                if req_id in self.req_info:
                    self.req_info[req_id]["ranks"].add(
                        data["from_rank"]
                    )
                    self.req_info[req_id]["earliest"] = min(
                        data["timestamp"], self.req_info[req_id]["earliest"]
                    )
                    self.req_info[req_id]["value"] = min(
                        value, self.req_info[req_id]["value"]
                    )
                else:
                    self.req_info[req_id] = {
                        "ranks": {data["from_rank"]},
                        "earliest": data["timestamp"],
                        "value": value,
                    }

    def close_zmq_comm(self) -> None:
        self.zmq_comm.terminate()
    

    def register_kv_caches(self, kv_caches: dict[str, Tuple[torch.Tensor]]) -> None:
        if not self._inited:
            self._pending_kv_caches = kv_caches
            logger.warning("[EMS][Register] EMS not ready, skip register_kvcache.")
            return

        kv_caches_list: List[List[torch.Tensor]] = []
        for _, kv_caches_layer in kv_caches.items():
            num_kv = len(kv_caches_layer)
            kv_tensor_list: List[torch.Tensor] = [kv_caches_layer[idx] for idx in range(num_kv)]
            kv_caches_list.append(kv_tensor_list)
        
        try:
            self.context_caching.register_kvcache(kv_caches_list)
            self._registered = True
            self._pending_kv_caches = None

            # shape of kv_caches_list: [layers, [k_v_index, [gpu_blocks, block_size, heads, head_size]]]
            ds_layers = len(kv_caches_list)
            ds_k_v_index = len(kv_caches_list[0])
            ds_gpu_blocks, ds_block_size, ds_heads, ds_head_size = list(kv_caches_list[0][0].shape)
            logger.info(f"[EMS][Register] layer_num={ds_layers}, block_size={ds_block_size}, "
                        f"is_mla={self.is_mla}, kvcache_dim={ds_k_v_index}")

        except EmsException as e:
            logger.error(f"[EMS][Register] register kvcache error: {e}.")


    def _cal_block_offsets(self, block_ids: List[int], block_size: int) -> List[int]:
        return [block_size] * len(block_ids)
    

    def _cal_block_slot_mapping(self, block_ids: List[int], block_size: int) -> List[int]:
        block_ids_np = np.array(block_ids)
        range_arr = np.arange(block_size)
        result_matrix = block_ids_np[:, np.newaxis] * block_size + range_arr
        flattened_result = result_matrix.ravel()
        return flattened_result.tolist()
    

    def async_load(self, req_id: str, block_hashes: List[int], block_ids: List[int], num_computed_blocks: int) -> None:
        future = None
        submit_time = time.perf_counter()

        if not self._check_params(req_id, block_hashes, block_ids, "AsyncLoad"):
            self.load_futures[req_id] = (future, submit_time, num_computed_blocks)
            return
        
        option = CcKvOption(
            write_rcache=EmsEnv.ems_enable_write_rcache,
            read_local_only=EmsEnv.ems_enable_read_local_only,
            timeout=EmsEnv.ems_timeout
        )
        offsets = self._cal_block_offsets(block_ids, self.block_size)
        slot_mapping = self._cal_block_slot_mapping(block_ids, self.block_size)

        try:
            future = self.context_caching.async_load(
                slot_mapping=slot_mapping,
                hashes=block_hashes,
                offsets=offsets,
                option=option
            )
            logger.info(f"[EMS][AsyncLoad][start] req_id={req_id} timeout_ms={EmsEnv.ems_timeout} "
                        f"planned_blocks={len(block_ids)} first_block_hash={block_hashes[0]}, future=submitted")
        except EmsException as e:
            self._process_exception(e)
            logger.error(f"[EMS][AsyncLoad][error] req_id={req_id} timeout_ms={EmsEnv.ems_timeout} ..., err={e}")
        self.load_futures[req_id] = (future, submit_time, num_computed_blocks)


    def async_save(self, req_id: str, block_hashes: List[int], block_ids: List[int]) -> None:
        future = None
        submit_time = time.perf_counter()

        if not self._check_params(req_id, block_hashes, block_ids, "AsyncSave"):
            self.save_futures[req_id] = (future, submit_time)
            return
        
        option = CcKvOption(
            write_rcache=EmsEnv.ems_enable_write_rcache,
            read_local_only=EmsEnv.ems_enable_read_local_only,
            timeout=EmsEnv.ems_timeout
        )
        offsets = self._cal_block_offsets(block_ids, self.block_size)
        slot_mapping = self._cal_block_slot_mapping(block_ids, self.block_size)

        try:
            future = self.context_caching.async_save(
                slot_mapping=slot_mapping,
                hashes=block_hashes,
                offsets=offsets,
                option=option
            )
            logger.info(f"[EMS][AsyncSave][start] req_id={req_id} timeout_ms={EmsEnv.ems_timeout} "
                        f"planned_blocks={len(block_ids)} first_block_hash={block_hashes[0]}, future=submitted")
        except EmsException as e:
            self._process_exception(e)
            logger.error(f"[EMS][AsyncSave][error] req_id={req_id} timeout_ms={EmsEnv.ems_timeout} ..., err={e}")
        self.save_futures[req_id] = (future, submit_time)


    def get_finished_load_reqs(self) -> List[Tuple[str, int]]:
        finished_reqs = []

        for req_id, (future, submit_time, num_computed_blocks) in list(self.load_futures.items()):
            if future and not self.context_caching.is_ready(future):
                continue

            num_success_tokens = 0
            try:
                if future:
                    result = self.context_caching.get_result(future)
                    cost_ms = 1e3 * (time.perf_counter() - submit_time)
                    logger.info(f"[EMS][GetResult] Req {req_id} async load done, success_blocks={result.success}, "
                                f"total_blocks_num={result.total}, status=SUCCESS, cost_ms={cost_ms:.2f}")
                    self.task_manager.update_req_stat("LOAD", result.success, cost_ms)
                    self.task_manager.update_hit_stat(result.success, result.total)
                    num_success_tokens = result.success * self.block_size
                else:
                    cost_ms = 1e3 * (time.perf_counter() - submit_time)
                    logger.info(f"[EMS][GetResult] Req {req_id} async load done, success_blocks={0}, "
                                f"total_blocks_num={0}, status=EMS_INTERNAL_ERROR, cost_ms={cost_ms:.2f}")
                    self.task_manager.update_req_stat("LOAD", 0, cost_ms)
            except EmsException as e:
                cost_ms = 1e3 * (time.perf_counter() - submit_time)
                self._process_exception(e)
                logger.info(f"[EMS][GetResult] Req {req_id} async load done, success_blocks={0}, "
                            f"total_blocks_num={0}, status={e.status_code().name}, cost_ms={cost_ms:.2f}")
            num_total_computed_tokens = num_success_tokens + num_computed_blocks * self.block_size
            finished_reqs.append((req_id, num_total_computed_tokens))
            self.load_futures.pop(req_id)

        updated_reqs = self.update_finished_reqs(finished_reqs)
        return updated_reqs


    def update_finished_reqs(self, finished_reqs) -> List[Tuple[str, int]]:
        data = {
            "from_rank": self.rank,
            "timestamp": time.perf_counter(),
            "item": finished_reqs,
        }

        if self.rank == 0:
            self.update_req_info(data)

            global_finished_reqs = []
            with self.zmq_thread_lock:
                to_pop = []
                for req_id in self.req_info:
                    if len(self.req_info[req_id]["ranks"]) == self.world_size:
                        to_pop.append(req_id)
                        global_finished_reqs.append((req_id, self.req_info[req_id]["value"]))
                for req_id in to_pop:
                    self.req_info.pop(req_id)
            return global_finished_reqs
        else:
            try:
                self.zmq_comm.send(data)
            except Exception as e:
                logger.error(f"[EMS] ZMQ send() failed, error: {e}.")
            return finished_reqs
    

    def get_finished_save_reqs(self) -> List[Tuple[str, int]]:
        finished_reqs = []

        for req_id, (future, submit_time) in list(self.save_futures.items()):
            if future and not self.context_caching.is_ready(future):
                continue

            num_success_tokens = 0
            try:
                if future:
                    result = self.context_caching.get_result(future)
                    cost_ms = 1e3 * (time.perf_counter() - submit_time)
                    logger.info(f"[EMS][GetResult] Req {req_id} async save done, success_blocks={result.success}, "
                                f"total_blocks_num={result.total}, status=SUCCESS, cost_ms={cost_ms:.2f}")
                    self.task_manager.update_req_stat("SAVE", result.success, cost_ms)
                    num_success_tokens = result.success * self.block_size
                else:
                    cost_ms = 1e3 * (time.perf_counter() - submit_time)
                    logger.info(f"[EMS][GetResult] Req {req_id} async save done, success_blocks={0}, "
                                f"total_blocks_num={0}, status=EMS_INTERNAL_ERROR, cost_ms={cost_ms:.2f}")
                    self.task_manager.update_req_stat("SAVE", 0, cost_ms)
            except EmsException as e:
                cost_ms = 1e3 * (time.perf_counter() - submit_time)
                self._process_exception(e)
                logger.info(f"[EMS][GetResult] Req {req_id} async save done, success_blocks={0}, "
                            f"total_blocks_num={0}, status={e.status_code().name}, cost_ms={cost_ms:.2f}")
            finished_reqs.append((req_id, num_success_tokens))
            self.save_futures.pop(req_id)
        
        return finished_reqs
    

    def sync_save_reqs(self) -> None:
        for req_id, (future, submit_time) in self.save_futures.items():
            if future is None:
                continue

            try:
                result = self.context_caching.get_result(future)
                cost_ms = 1e3 * (time.perf_counter() - submit_time)
                logger.info(f"[EMS][GetResult] Req {req_id} async save done, success_blocks={result.success}, "
                            f"total_blocks_num={result.total}, status=SUCCESS, cost_ms={cost_ms:.2f}")
                self.task_manager.update_req_stat("SAVE", result.success, cost_ms)
            except EmsException as e:
                cost_ms = 1e3 * (time.perf_counter() - submit_time)
                self._process_exception(e)
                logger.info(f"[EMS][GetResult] Req {req_id} async save done, success_blocks={0}, "
                            f"total_blocks_num={0}, status={e.status_code().name}, cost_ms={cost_ms:.2f}")
            
        self.save_futures.clear()


    def _check_health(self) -> bool:
        is_health = Ems.check_health()
        if is_health:
            logger.info("[EMS] EMS health status is ok.")
        else:
            logger.info("[EMS] EMS health status is abnormal.")

        if not self._inited and is_health and self._need_reinit:
            logger.info(f"[EMS][Init] re-init during health check.")
            self.init_ems()
            if self._inited:
                logger.info("[EMS][Init] re-init succeed during health check.")
                if (not self._registered) and (self._pending_kv_caches is not None):
                    self.register_kv_caches(self._pending_kv_caches)
                    logger.info(f"[EMS][Register] re-register pending kvcache succeed.")

        return is_health


    def _check_params(self, req_id: str, block_hashes: List[int], block_ids: List[int], called_at: str) -> bool:
        if not self._inited or not self._registered:
            logger.warning(f"[EMS][{called_at}][degraded] req_id={req_id} reason=EMS not ready.")
            return False

        if not self.task_manager.get_status():
            logger.warning(f"[EMS][{called_at}][skip] req_id={req_id} reason=Unhealthy EMS")
            return False

        if (len(block_hashes) == 0 or len(block_ids) == 0) or (len(block_hashes) != len(block_ids)):
            logger.error(f"[EMS] req {req_id} has invalid block_hashes or block_ids: "
                         f"len(block_hashes) == {len(block_hashes)}, len(block_ids) == {len(block_ids)}.")
            return False
        
        return True


    def _process_exception(self, e: EmsException) -> None:
        if e.status_code() == EmsErrorCode.EMS_INVALID_ARGUMENT:
            return
        self.task_manager.reset_status()


class PeriodicTaskManager:
    HEALTH_CHECK_INTERVAL = 10  # 检查间隔 (秒)
    FLAPPING_WINDOW = 60  # 震荡检测窗口 (秒)
    FLAPPING_LIMIT = 3  # 窗口内允许的最大状态变更次数
    SUCCESS_THRESHOLD = 3  # 恢复健康所需的连续成功次数

    PRINT_INTERVAL = 30  # 检查间隔 (秒)

    def __init__(self, check_fn):
        self.check_fn = check_fn

        self._ems_ok = False
        self._consecutive_success_count = 0
        # 滑动窗口震荡检测记录
        self._change_history: deque = deque()

        self.last_log_time = time.perf_counter()
        self.stat = { # [sum, min, max], init min with a large number: 2**30 (int) or 1e9 (float)
            "LOAD": {"count": 0, "block_nums": [0, 2**30, 0], "cost_times": [0.0, 1e9, 0.0]},
            "SAVE": {"count": 0, "block_nums": [0, 2**30, 0], "cost_times": [0.0, 1e9, 0.0]},
            "HIT": {"num_hit_blocks": 0, "num_total_blocks": 0},
        }

        self.thread_lock = threading.Lock()
        self.start_loop()

    def get_status(self) -> bool:
        return self._ems_ok

    def reset_status(self) -> None:
        if not self._ems_ok:
            return

        logger.info(f"[EMS] Ems health status reset to False")
        self._ems_ok = False

    def check_health_status(self) -> None:
        is_healthy = False
        try:
            is_healthy = self.check_fn()
        except Exception as e:
            logger.error(f"[EMS] EMS health check failed, error: {e}.")

        self._process_check_result(is_healthy)

    def _process_check_result(self, is_healthy: bool):
        if is_healthy:
            self._handle_success()
        else:
            self._handle_failure()

    def _handle_success(self):
        self._consecutive_success_count += 1

        if not self._ems_ok:
            if self._consecutive_success_count >= self.SUCCESS_THRESHOLD:
                # 尝试切换健康状态为True，这会受到震荡检测的严格拦截
                if self._try_switch_status(new_status=True):
                    pass
                else:
                    # 被震荡拦截，重置计数器
                    self._consecutive_success_count = 0
        else:
            self._consecutive_success_count = min(self._consecutive_success_count, self.SUCCESS_THRESHOLD)

    def _handle_failure(self):
        self._consecutive_success_count = 0

        if self._ems_ok:
            # 尝试切换健康状态为False，忽略震荡，立即切换
            self._try_switch_status(new_status=False)

    def _try_switch_status(self, new_status: bool) -> bool:
        current_time = time.monotonic()

        # 1. 清理滑动窗口
        self._clean_flapping_history(current_time)

        # 2. 检查震荡
        is_flapping = len(self._change_history) >= self.FLAPPING_LIMIT

        if is_flapping:
            if new_status is True:
                # 震荡中尝试恢复被拦截
                logger.warning(
                    f"[EMS] EMS status flapping detected ({len(self._change_history)} changes in {self.FLAPPING_WINDOW}s). "
                    f"Blocking recovery (switch to True)."
                )
                return False
            else:
                logger.info(
                    f"[EMS] EMS status flapping detected, but forcing status to False (Fail Fast strategy)."
                )

        # 3. 执行切换
        logger.info(f"[EMS] EMS health status changing: {self._ems_ok} -> {new_status}")
        self._ems_ok = new_status

        # 4. 记录变更历史
        self._change_history.append(current_time)

        return True

    def _clean_flapping_history(self, current_time: float):
        threshold_time = current_time - self.FLAPPING_WINDOW
        while self._change_history and self._change_history[0] < threshold_time:
            self._change_history.popleft()

    def update_req_stat(self, event: str, block_num: int, cost_time: float) -> None:
        with self.thread_lock:
            self.stat[event]["count"] += 1
            self.stat[event]["block_nums"][0] += block_num
            self.stat[event]["block_nums"][1] = min(block_num, self.stat[event]["block_nums"][1])
            self.stat[event]["block_nums"][2] = max(block_num, self.stat[event]["block_nums"][2])
            self.stat[event]["cost_times"][0] += cost_time
            self.stat[event]["cost_times"][1] = min(cost_time, self.stat[event]["cost_times"][1])
            self.stat[event]["cost_times"][2] = max(cost_time, self.stat[event]["cost_times"][2])

    def update_hit_stat(self, num_hit_blocks: int, num_total_blocks: int) -> None:
        with self.thread_lock:
            self.stat["HIT"]["num_hit_blocks"] += num_hit_blocks
            self.stat["HIT"]["num_total_blocks"] += num_total_blocks

    def print_stat(self) -> None:
        with self.thread_lock:
            stat = self.stat
            self.stat = { # [sum, min, max], init min with a large number: 2**30 (int) or 1e9 (float)
                "LOAD": {"count": 0, "block_nums": [0, 2**30, 0], "cost_times": [0.0, 1e9, 0.0]},
                "SAVE": {"count": 0, "block_nums": [0, 2**30, 0], "cost_times": [0.0, 1e9, 0.0]},
                "HIT": {"num_hit_blocks": 0, "num_total_blocks": 0},
            }

        load = stat["LOAD"]["count"]
        avg_load_block_num = stat["LOAD"]["block_nums"][0] / load if load else 0.0
        min_load_block_num = stat["LOAD"]["block_nums"][1] if load else 0
        max_load_block_num = stat["LOAD"]["block_nums"][2] if load else 0
        avg_load_cost_time = stat["LOAD"]["cost_times"][0] / load if load else 0.0
        min_load_cost_time = stat["LOAD"]["cost_times"][1] if load else 0.0
        max_load_cost_time = stat["LOAD"]["cost_times"][2] if load else 0.0

        save = stat["SAVE"]["count"]
        avg_save_block_num = stat["SAVE"]["block_nums"][0] / save if save else 0.0
        min_save_block_num = stat["SAVE"]["block_nums"][1] if save else 0
        max_save_block_num = stat["SAVE"]["block_nums"][2] if save else 0
        avg_save_cost_time = stat["SAVE"]["cost_times"][0] / save if save else 0.0
        min_save_cost_time = stat["SAVE"]["cost_times"][1] if save else 0.0
        max_save_cost_time = stat["SAVE"]["cost_times"][2] if save else 0.0

        num_hit_blocks = stat["HIT"]["num_hit_blocks"]
        num_total_blocks = stat["HIT"]["num_total_blocks"]
        hit_rate = 100 * (num_hit_blocks / num_total_blocks) if num_total_blocks else 0.0

        logger.info(f"[EMS][{self.PRINT_INTERVAL}Sec Summary] req(load={load},save={save}) "
                    f"LOAD(avg_block_num={avg_load_block_num:.1f} [min:{min_load_block_num}, max:{max_load_block_num}]; "
                    f"avg_cost_ms={avg_load_cost_time:.1f} [min:{min_load_cost_time:.1f}, max:{max_load_cost_time:.1f}]) "
                    f"SAVE(avg_block_num={avg_save_block_num:.1f} [min:{min_save_block_num}, max:{max_save_block_num}]; "
                    f"avg_cost_ms={avg_save_cost_time:.1f} [min:{min_save_cost_time:.1f}, max:{max_save_cost_time:.1f}]) "
                    f"HIT(hit_rate={hit_rate:.1f} [hit:{num_hit_blocks}, total:{num_total_blocks}])")

    def task_loop(self) -> None:
        while True:
            time.sleep(self.HEALTH_CHECK_INTERVAL)
            self.check_health_status()
            cur_time = time.perf_counter()
            if cur_time - self.last_log_time > self.PRINT_INTERVAL:
                self.last_log_time = cur_time
                self.print_stat()

    def start_loop(self) -> None:
        logger.info("[EMS] EMS start periodic tasks.")
        self.check_health_status()
        self.periodic_task_thread = threading.Thread(target=self.task_loop, daemon=True, name="ems_task_loop")
        self.periodic_task_thread.start()
        logger.info(f"[EMS] periodic tasks subthread \"ems_task_loop\" started.")
