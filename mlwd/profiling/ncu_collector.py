"""
Nsight Compute 采集器。

对每个实验点调用 ncu 子进程，采集 CI、L2 命中率、IPC。
"""

import subprocess
import os
import shutil
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict

from .ncu_metrics import NCU_METRIC_LIST, parse_ncu_csv, KernelMetrics
from .kernel_classifier import classify_kernel, KernelCategory


def _find_ncu() -> str:
    """查找 ncu 可执行文件路径。"""
    ncu = shutil.which("ncu")
    if ncu:
        return ncu
    candidates = [
        "/opt/nvidia/nsight-compute/2025.1.1/ncu",
        "/opt/nvidia/nsight-compute/2025.1.1/target/linux-desktop-glibc_2_11_3-x64/ncu",
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    raise FileNotFoundError("ncu not found. Install Nsight Compute or set PATH.")


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
                    launch_count: int = 10,
                    launch_skip: int = 500) -> NCUResult:
    """
    对单个实验点运行 ncu profiling。

    通过 ncu 包装 vllm_runner.py 子进程，采集 kernel 级硬件 counter。
    使用 application-level replay 避免 kernel replay 导致显存翻倍。
    使用 launch-skip 跳过模型加载阶段的 kernel。
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(
        output_dir,
        f"ncu_{model.replace('/', '_')}_{quantization}_tp{tp_degree}_b{batch_size}_s{seq_len}_{phase}.csv"
    )

    # 强制 vLLM 使用单进程 V0 引擎，避免 ncu 无法跟踪多进程
    env = os.environ.copy()
    env["VLLM_USE_V1"] = "0"

    # 构造 ncu 命令
    # --replay-mode application: 重放整个应用而非单个 kernel，避免显存翻倍
    # --launch-skip: 跳过模型加载阶段的 kernel（权重加载、初始化等）
    # --launch-count: 只采集少量推理阶段的 kernel
    ncu_bin = _find_ncu()

    cmd = [
        ncu_bin,
        "--csv",
        "--metrics", NCU_METRIC_LIST,
        "--kernel-name-base", "demangled",
        "--replay-mode", "application",
        "--launch-skip", str(launch_skip),
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
        "--gpu_mem", "0.85",
    ]

    print(f"[NCU] Running: {' '.join(cmd[:8])}...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, env=env)

    if result.returncode != 0:
        # ncu 的 stderr 经常混入 FutureWarning 等无害输出，检查是否有实际错误
        stderr = result.stderr
        # 如果 CSV 文件已生成，忽略 returncode 继续解析
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            print(f"[NCU] Warning: rc={result.returncode} but CSV exists, attempting parse")
        else:
            print(f"[NCU] Error (rc={result.returncode}):\n{stderr[-1000:]}")
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
