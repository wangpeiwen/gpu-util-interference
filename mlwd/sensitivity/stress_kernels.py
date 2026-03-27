"""
合成压力核 Python ctypes 封装。

封装 libpython_interface.so 中的 4 类压力核函数，
供干扰敏感度采集使用。
"""

import ctypes
from pathlib import Path
from typing import Optional


class StressKernels:
    """四类合成压力核的 Python 接口。"""

    def __init__(self, lib_path: str):
        self.lib = ctypes.CDLL(lib_path)
        self._setup_argtypes()

    def _setup_argtypes(self):
        # CU 压力核 (已有): run_fp32_fma_kernel(int, int, long long, int)
        self.lib.run_fp32_fma_kernel.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_longlong, ctypes.c_int
        ]
        self.lib.run_fp32_fma_kernel.restype = None

        # BS 压力核: run_tb_scheduler_stress(int, int, long long, int)
        self.lib.run_tb_scheduler_stress.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_longlong, ctypes.c_int
        ]
        self.lib.run_tb_scheduler_stress.restype = None

        # L2 压力核: run_l2_cache_stress(int, int, long long, long long, int)
        self.lib.run_l2_cache_stress.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_int
        ]
        self.lib.run_l2_cache_stress.restype = None

        # BW 压力核: run_mem_bw_stress(int, int, long long, long long, int)
        self.lib.run_mem_bw_stress.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_int
        ]
        self.lib.run_mem_bw_stress.restype = None

    def run_bs_stress(self, num_tb: int = 160, num_threads: int = 1024,
                      num_itrs: int = 100000, num_runs: int = 4):
        """Block Scheduler 压力核：高频 launch sleep_kernel。"""
        self.lib.run_tb_scheduler_stress(num_tb, num_threads, num_itrs, num_runs)

    def run_cu_stress(self, num_tb: int = 80, num_threads: int = 128,
                      num_itrs: int = 500000, num_runs: int = 4):
        """Compute Unit 压力核：密集 FMA 运算。"""
        self.lib.run_fp32_fma_kernel(num_tb, num_threads, num_itrs, num_runs)

    def run_l2_stress(self, num_tb: int = 40, num_threads: int = 1024,
                      num_itrs: int = 10000, num_bytes: int = 6 * 1024 * 1024,
                      num_runs: int = 4):
        """L2 Cache 压力核：L2 大小工作集反复读写。"""
        self.lib.run_l2_cache_stress(num_tb, num_threads, num_itrs, num_bytes, num_runs)

    def run_bw_stress(self, num_tb: int = 80, num_threads: int = 1024,
                      num_itrs: int = 50, num_bytes: int = 2 * 1024 * 1024 * 1024,
                      num_runs: int = 4):
        """Memory Bandwidth 压力核：大块连续显存读写。"""
        self.lib.run_mem_bw_stress(num_tb, num_threads, num_itrs, num_bytes, num_runs)
