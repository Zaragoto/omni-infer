import unittest
import time
import threading
import logging
from collections import deque
from unittest.mock import MagicMock, patch, ANY


from omni.adaptors.vllm.ems.ems_adapter import PeriodicTaskManager

class TestPeriodicTaskManager(unittest.TestCase):

    def setUp(self):
        """测试前的环境准备"""
        # 1. Mock 核心依赖
        self.mock_check_fn = MagicMock(return_value=True)

        # 2. Patch 系统函数，防止副作用
        # 拦截 Logger
        self.patcher_log_info = patch('logging.Logger.info')
        self.mock_log_info = self.patcher_log_info.start()
        self.patcher_logger = patch('logging.getLogger')
        self.mock_logger = self.patcher_logger.start()

        # 拦截 Thread，防止 __init__ 真的启动后台死循环线程
        self.patcher_thread = patch('threading.Thread')
        self.mock_thread = self.patcher_thread.start()

        # 拦截时间函数 - 用于精确控制震荡逻辑
        self.patcher_monotonic = patch('time.monotonic')
        self.mock_monotonic = self.patcher_monotonic.start()
        self.mock_monotonic.return_value = 1000.0

        # 拦截时间函数 - 用于控制打印频率
        self.patcher_perf = patch('time.perf_counter')
        self.mock_perf = self.patcher_perf.start()
        self.mock_perf.return_value = 1000.0

        # 拦截 sleep，加快测试速度
        self.patcher_sleep = patch('time.sleep')
        self.mock_sleep = self.patcher_sleep.start()

    def tearDown(self):
        """测试后的清理"""
        self.patcher_log_info.stop()
        self.patcher_logger.stop()
        self.patcher_thread.stop()
        self.patcher_monotonic.stop()
        self.patcher_perf.stop()
        self.patcher_sleep.stop()

    def create_manager(self):
        """
        辅助函数：创建一个干净的 Manager 实例。
        因为 __init__ 会自动调用一次 check_health_status，可能会污染 mock 的调用记录或初始状态。
        """
        mgr = PeriodicTaskManager(self.mock_check_fn)

        # 重置内部状态，确保测试从已知的“干净”状态开始
        mgr._ems_ok = False
        mgr._consecutive_success_count = 0
        mgr._change_history.clear()

        # 重置统计信息
        # 注意：这里需要模拟 print_stat 后的重置状态，但保留基本结构
        mgr.print_stat()  # 调用一次打印来重置统计数据为初始值

        # 重置 mock 调用记录
        self.mock_check_fn.reset_mock()
        return mgr

    # ==========================================
    # 1. 初始化与线程测试
    # ==========================================

    def test_init_starts_thread(self):
        """测试初始化时是否正确启动了守护线程"""
        self.mock_check_fn.return_value = False
        mgr = PeriodicTaskManager(self.mock_check_fn)

        self.mock_thread.assert_called_once()
        _, kwargs = self.mock_thread.call_args
        self.assertTrue(kwargs.get('daemon'), "线程应该是 daemon 模式")
        self.mock_thread.return_value.start.assert_called_once()

    # ==========================================
    # 2. 健康检查状态逻辑测试
    # ==========================================

    def test_health_check_success_threshold(self):
        """测试：只有连续成功达到 SUCCESS_THRESHOLD 次，状态才变为 True"""
        mgr = self.create_manager()
        mgr.SUCCESS_THRESHOLD = 3

        self.mock_check_fn.return_value = True

        # 第 1 次成功
        mgr.check_health_status()
        self.assertFalse(mgr.get_status())
        self.assertEqual(mgr._consecutive_success_count, 1)

        # 第 2 次成功
        mgr.check_health_status()
        self.assertFalse(mgr.get_status())

        # 第 3 次成功 -> 状态翻转
        mgr.check_health_status()
        self.assertTrue(mgr.get_status())

        # 第 4 次成功 -> 计数器保持在阈值，不无限增长
        mgr.check_health_status()
        self.assertEqual(mgr._consecutive_success_count, 3)

    def test_health_check_fail_fast(self):
        """测试：Fail Fast 策略，一次失败立即导致状态变为 False"""
        mgr = self.create_manager()
        # 预设为健康状态
        mgr._ems_ok = True
        mgr._consecutive_success_count = 3

        # 模拟检查失败
        self.mock_check_fn.return_value = False
        mgr.check_health_status()

        self.assertFalse(mgr.get_status())
        self.assertEqual(mgr._consecutive_success_count, 0)

    # ==========================================
    # 3. 防震荡逻辑测试 (Flapping)
    # ==========================================

    def test_flapping_blocks_recovery(self):
        """测试：震荡期间阻止状态恢复为 True"""
        mgr = self.create_manager()
        mgr.FLAPPING_LIMIT = 3
        mgr.FLAPPING_WINDOW = 60
        mgr.SUCCESS_THRESHOLD = 1  # 设为1以便快速测试

        mgr._ems_ok = True  # 初始健康

        # --- 模拟快速震荡 ---
        # 1. 变为 False
        self.mock_monotonic.return_value = 1001
        mgr._process_check_result(is_healthy=False)

        # 2. 变为 True
        self.mock_monotonic.return_value = 1002
        mgr._process_check_result(is_healthy=True)

        # 3. 变为 False
        self.mock_monotonic.return_value = 1003
        mgr._process_check_result(is_healthy=False)

        self.assertEqual(len(mgr._change_history), 3)
        self.assertFalse(mgr.get_status())

        # --- 尝试恢复 ---
        # 4. 尝试变为 True (此时还在窗口内)
        self.mock_monotonic.return_value = 1004
        mgr._process_check_result(is_healthy=True)

        # 断言：状态应该依然是 False (被拦截)
        self.assertFalse(mgr.get_status(), "震荡期间应阻止上线")
        # 断言：拦截不应计入变更历史 (历史记录长度不变)
        self.assertEqual(len(mgr._change_history), 3)

    def test_flapping_allows_failure(self):
        """测试：震荡期间允许 Fail Fast (变为 False)"""
        mgr = self.create_manager()
        mgr.FLAPPING_LIMIT = 3

        # 填满历史，且当前状态是 True
        mgr._ems_ok = True
        mgr._change_history.extend([1001, 1002, 1003])

        self.mock_monotonic.return_value = 1004

        # 尝试变为 False
        mgr._process_check_result(is_healthy=False)

        # 断言：允许下线
        self.assertFalse(mgr.get_status())
        # 断言：下线会计入历史
        self.assertEqual(len(mgr._change_history), 4)

    def test_flapping_window_expiry(self):
        """测试：震荡窗口过期后，允许恢复"""
        mgr = self.create_manager()
        mgr.FLAPPING_WINDOW = 60
        mgr.FLAPPING_LIMIT = 2
        mgr.SUCCESS_THRESHOLD = 1

        # 历史记录 [1000, 1005]
        mgr._ems_ok = False
        mgr._change_history.extend([1000, 1005])

        # 时间来到 1010 (仍在窗口内 1010-1000 < 60)
        self.mock_monotonic.return_value = 1010
        mgr._process_check_result(is_healthy=True)
        self.assertFalse(mgr.get_status())

        # 时间来到 1061 (最早的 1000 过期)
        self.mock_monotonic.return_value = 1061
        mgr._process_check_result(is_healthy=True)

        self.assertTrue(mgr.get_status())

    # ==========================================
    # 4. 统计信息聚合测试 (Stats)
    # ==========================================

    def test_update_req_stat_calculation(self):
        """测试 LOAD/SAVE 统计信息的累加、最大值、最小值计算是否正确"""
        mgr = self.create_manager()

        # 1. 第一次更新 LOAD
        # block_num=10, cost=0.5
        mgr.update_req_stat("LOAD", 10, 0.5)

        load_stat = mgr.stat["LOAD"]
        self.assertEqual(load_stat["count"], 1)
        self.assertEqual(load_stat["block_nums"][0], 10)  # sum
        self.assertEqual(load_stat["block_nums"][1], 10)  # min
        self.assertEqual(load_stat["block_nums"][2], 10)  # max

        # 2. 第二次更新 LOAD (更大值)
        # block_num=20, cost=1.0
        mgr.update_req_stat("LOAD", 20, 1.0)

        self.assertEqual(load_stat["count"], 2)
        self.assertEqual(load_stat["block_nums"][0], 30)  # sum (10+20)
        self.assertEqual(load_stat["block_nums"][1], 10)  # min (10)
        self.assertEqual(load_stat["block_nums"][2], 20)  # max (20)
        self.assertEqual(load_stat["cost_times"][0], 1.5)  # sum (0.5+1.0)

        # 3. 第三次更新 LOAD (更小值)
        # block_num=5, cost=0.1
        mgr.update_req_stat("LOAD", 5, 0.1)

        self.assertEqual(load_stat["block_nums"][1], 5)  # min updated
        self.assertEqual(load_stat["cost_times"][1], 0.1)  # min updated

    def test_update_hit_stat(self):
        """测试 HIT 统计的累加"""
        mgr = self.create_manager()
        mgr.update_hit_stat(10, 20)
        mgr.update_hit_stat(5, 5)

        self.assertEqual(mgr.stat["HIT"]["num_hit_blocks"], 15)
        self.assertEqual(mgr.stat["HIT"]["num_total_blocks"], 25)

    def test_print_stat_logs_and_resets(self):
        """测试 print_stat 是否产生日志，并在打印后重置数据"""
        mgr = self.create_manager()

        # 填充一些数据
        mgr.update_req_stat("LOAD", 100, 1.0)
        mgr.update_hit_stat(10, 20)

        # 验证数据已存在
        self.assertEqual(mgr.stat["LOAD"]["count"], 1)
        self.assertEqual(mgr.stat["HIT"]["num_hit_blocks"], 10)

        # 调用打印
        mgr.print_stat()

        # 1. 首先断言 info 方法确实被调用了
        self.assertTrue(self.mock_log_info.called, "logger.info 应该被调用")

        # 2. 获取调用参数
        # 当 patch 类方法时，第一个参数 args[0] 是实例本身(self/logger对象)，第二个参数 args[1] 才是消息
        args, _ = self.mock_log_info.call_args
        log_msg = args[0]

        self.assertIn("LOAD(avg_block_num=100.0", log_msg)
        self.assertIn("hit_rate=50.0", log_msg)

        # 验证 2: 统计数据被重置
        self.assertEqual(mgr.stat["LOAD"]["count"], 0)
        self.assertEqual(mgr.stat["LOAD"]["block_nums"][0], 0)
        self.assertEqual(mgr.stat["HIT"]["num_hit_blocks"], 0)

    # ==========================================
    # 5. 主循环定时逻辑测试
    # ==========================================

    def test_task_loop_intervals(self):
        """测试主循环：仅当时间间隔达到 PRINT_INTERVAL 时才打印日志"""
        mgr = self.create_manager()
        mgr.PRINT_INTERVAL = 30
        mgr.HEALTH_CHECK_INTERVAL = 10

        # Mock print_stat，这样我们可以简单地计算它被调用的次数
        mgr.print_stat = MagicMock()

        # 1. 第一次模拟循环：时间只过了 10 秒
        self.mock_perf.return_value = 1000.0 + 10.0

        # 模拟 task_loop 内部的一轮逻辑
        time.sleep(mgr.HEALTH_CHECK_INTERVAL)
        mgr.check_health_status()
        cur_time = time.perf_counter()
        if cur_time - mgr.last_log_time > mgr.PRINT_INTERVAL:
            mgr.last_log_time = cur_time
            mgr.print_stat()

        # 断言：时间没到，不应打印
        mgr.print_stat.assert_not_called()

        # 2. 第二次模拟循环：时间又过了 25 秒 (总共 35 秒 > 30)
        self.mock_perf.return_value = 1000.0 + 35.0

        # 模拟 loop 逻辑
        cur_time = time.perf_counter()
        if cur_time - mgr.last_log_time > mgr.PRINT_INTERVAL:
            mgr.last_log_time = cur_time
            mgr.print_stat()

        # 断言：时间到了，应该打印
        mgr.print_stat.assert_called_once()
        # 断言：最后打印时间更新了
        self.assertEqual(mgr.last_log_time, 1035.0)

    def test_reset_status(self):
        """测试 reset_status 方法"""
        mgr = self.create_manager()
        mgr._ems_ok = True

        mgr.reset_status()
        self.assertFalse(mgr.get_status())

        # 如果已经是 False，调用 reset_status 不应产生额外的日志
        self.mock_logger.return_value.info.reset_mock()
        mgr.reset_status()
        self.mock_logger.return_value.info.assert_not_called()


if __name__ == '__main__':
    unittest.main()