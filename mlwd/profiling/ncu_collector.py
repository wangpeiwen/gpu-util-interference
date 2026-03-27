"""
Nsight Compute 采集器。

对每个实验点调用 ncu 子进程，采集 CI、L2 命中率、IPC。
"""

import subprocess
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict

from .ncu_metrics import NCU_METRIC_LIST, parse_ncu_csv, KernelMetrics
from .kernel_classifier import classify_kernel, KernelCategory


@dataclass
class NCUResult:
    """单个实验点的 ncu 采集结果。"""
    ci_attn: Optional[float] = None
    ci_ffn: Optional[float] = None
    l2_attn: Optional[float] = None
    l2_ffn: Optional[float] = None
    ipc: Optional[float] = None


def run_ncu_profile(model: str, quantization: str, tp_degree: int,
                    batch_size: int, seq_len: int, phase: str,
                    vllm_runner_path: str, output_dir: str = "/tmp/mlwd_ncu",
                    launch_count: int = 3) -> NCUResult:
    """
    对单个实验点运行 ncu profiling。

    通过 ncu 包装 vllm_runner.py 子进程，采集 kernel 级硬件 counter。
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(
        output_dir,
        f"ncu_{model.replace('/', '_')}_{quantization}_tp{tp_degree}_b{batch_size}_s{seq_len}_{phase}.csv"
    )

    # 构造 ncu 命令
    cmd = [
        "ncu",
        "--csv",
        "--metrics", NCU_METRIC_LIST,
        "--kernel-name-base", "demangled",
        "--launch-count", str(launch_count),
        "--target-processes", "all",
        "--log-file", csv_path,
        sys.executable, vllm_runner_path,
        "--model", model,
        "--quantization", quantization,
        "--tp", str(tp_degree),
        "--batch_size", str(batch_size),
        "--seq_len", str(seq_len),
        "--phase", phase,
        "--num_runs", "1",
        "--warmup_runs", "1",
    ]

    print(f"[NCU] Running: {' '.join(cmd[:8])}...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

    if result.returncode != 0:
        print(f"[NCU] Error (rc={result.returncode}): {result.stderr[:500]}")
        return NCUResult()

    # 解析 CSV 输出
    try:
        with open(csv_path, "r") as f:
            csv_content = f.read()
    except FileNotFoundError:
        # ncu 可能直接输出到 stdout
        csv_content = result.stdout

    kernel_metrics = parse_ncu_csv(csv_content)
    if not kernel_metrics:
        print("[NCU] Warning: no kernel metrics parsed")
        return NCUResult()

    return _aggregate_by_category(kernel_metrics)


def _aggregate_by_category(kernel_metrics: list) -> NCUResult:
    """按 Attention/FFN 分类聚合 kernel 指标。"""
    attn_kernels = []
    ffn_kernels = []
    all_ipcs = []

    for km in kernel_metrics:
        cat = classify_kernel(km.kernel_name)
        if cat == KernelCategory.ATTENTION:
            attn_kernels.append(km)
        elif cat == KernelCategory.FFN:
            ffn_kernels.append(km)
        all_ipcs.append(km.ipc)

    result = NCUResult()

    if attn_kernels:
        cis = [k.compute_intensity for k in attn_kernels if k.dram_bytes > 0]
        l2s = [k.l2_hit_rate for k in attn_kernels]
        result.ci_attn = sum(cis) / len(cis) if cis else None
        result.l2_attn = sum(l2s) / len(l2s) if l2s else None

    if ffn_kernels:
        cis = [k.compute_intensity for k in ffn_kernels if k.dram_bytes > 0]
        l2s = [k.l2_hit_rate for k in ffn_kernels]
        result.ci_ffn = sum(cis) / len(cis) if cis else None
        result.l2_ffn = sum(l2s) / len(l2s) if l2s else None

    if all_ipcs:
        result.ipc = sum(all_ipcs) / len(all_ipcs)

    return result
