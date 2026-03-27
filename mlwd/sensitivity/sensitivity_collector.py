"""
四维干扰敏感度采集器。

对每个实验点，分别与 4 类压力核共置运行 vLLM 推理，
测量性能退化比例 σ_bs, σ_cu, σ_l2, σ_bw。
"""

import time
import threading
from dataclasses import dataclass
from typing import Optional, Callable

from .stress_kernels import StressKernels
from ..config import StressKernelConfig


@dataclass
class SensitivityResult:
    """四维干扰敏感度向量。"""
    sigma_bs: Optional[float] = None
    sigma_cu: Optional[float] = None
    sigma_l2: Optional[float] = None
    sigma_bw: Optional[float] = None
    baseline_latency_ms: Optional[float] = None


def _run_stress_in_background(stress_fn: Callable, stop_event: threading.Event,
                              repeat_count: int = 100):
    """在后台线程中循环运行压力核，直到 stop_event 被设置。"""
    while not stop_event.is_set():
        stress_fn()


def _measure_inference_latency(run_inference_fn: Callable, num_runs: int) -> float:
    """运行推理并返回中位数时延 (ms)。"""
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        run_inference_fn()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies.append(elapsed_ms)

    latencies.sort()
    mid = len(latencies) // 2
    if num_runs % 2 == 1:
        return latencies[mid]
    return (latencies[mid - 1] + latencies[mid]) / 2


def collect_sensitivity(run_inference_fn: Callable,
                        stress_kernels: StressKernels,
                        stress_config: StressKernelConfig,
                        num_runs: int = 5,
                        warmup_runs: int = 2) -> SensitivityResult:
    """
    采集四维干扰敏感度。

    Args:
        run_inference_fn: 无参数的推理函数，每次调用执行一次推理
        stress_kernels: StressKernels 实例
        stress_config: 压力核参数配置
        num_runs: 每个维度的测量次数
        warmup_runs: 预热次数
    """
    result = SensitivityResult()

    # Step 1: 测量基线时延 (无干扰)
    print("  [Sensitivity] Measuring baseline (no interference)...")
    # warmup
    for _ in range(warmup_runs):
        run_inference_fn()

    baseline = _measure_inference_latency(run_inference_fn, num_runs)
    result.baseline_latency_ms = baseline
    print(f"  [Sensitivity] Baseline: {baseline:.2f} ms")

    if baseline <= 0:
        print("  [Sensitivity] Warning: baseline latency <= 0, skipping")
        return result

    # Step 2: 逐维度测量干扰敏感度
    stress_fns = {
        "bs": lambda: stress_kernels.run_bs_stress(
            stress_config.bs_num_tb, stress_config.bs_num_threads,
            stress_config.bs_num_itrs, 1),
        "cu": lambda: stress_kernels.run_cu_stress(
            stress_config.cu_num_tb, stress_config.cu_num_threads,
            stress_config.cu_num_itrs, 1),
        "l2": lambda: stress_kernels.run_l2_stress(
            stress_config.l2_num_tb, stress_config.l2_num_threads,
            stress_config.l2_num_itrs, stress_config.l2_num_bytes, 1),
        "bw": lambda: stress_kernels.run_bw_stress(
            stress_config.bw_num_tb, stress_config.bw_num_threads,
            stress_config.bw_num_itrs, stress_config.bw_num_bytes, 1),
    }

    for dim_name, stress_fn in stress_fns.items():
        print(f"  [Sensitivity] Measuring σ_{dim_name}...")

        stop_event = threading.Event()
        stress_thread = threading.Thread(
            target=_run_stress_in_background,
            args=(stress_fn, stop_event),
            daemon=True,
        )

        # warmup with stress
        stress_thread.start()
        for _ in range(warmup_runs):
            run_inference_fn()

        # measure
        stressed_latency = _measure_inference_latency(run_inference_fn, num_runs)

        stop_event.set()
        stress_thread.join(timeout=30)

        sigma = (stressed_latency - baseline) / baseline
        sigma = max(sigma, 0.0)  # 敏感度不应为负

        print(f"  [Sensitivity] σ_{dim_name} = {sigma:.4f} "
              f"(baseline={baseline:.2f}ms, stressed={stressed_latency:.2f}ms)")

        setattr(result, f"sigma_{dim_name}", sigma)

    return result
