"""
Nsight Systems 采集器。

对每个实验点调用 nsys 子进程进行时间线追踪，
然后导出 SQLite 并解析执行模式特征。
"""

import subprocess
import os
import sys
from typing import Optional

from .nsys_parser import parse_nsys_sqlite, NSysResult


def run_nsys_profile(model: str, quantization: str, tp_degree: int,
                     batch_size: int, seq_len: int, phase: str,
                     vllm_runner_path: str, output_dir: str = "/tmp/mlwd_nsys") -> NSysResult:
    """
    对单个实验点运行 nsys profiling。

    步骤:
    1. nsys profile → .nsys-rep 文件
    2. nsys export --type sqlite → .sqlite 文件
    3. 解析 SQLite 提取执行模式特征
    """
    os.makedirs(output_dir, exist_ok=True)

    base_name = f"nsys_{model.replace('/', '_')}_{quantization}_tp{tp_degree}_b{batch_size}_s{seq_len}_{phase}"
    rep_path = os.path.join(output_dir, base_name)
    sqlite_path = rep_path + ".sqlite"

    # Step 1: nsys profile
    profile_cmd = [
        "nsys", "profile",
        "-o", rep_path,
        "--force-overwrite", "true",
        "--trace", "cuda,nvtx",
        "--sample", "none",
        "--cpuctxsw", "none",
        sys.executable, vllm_runner_path,
        "--model", model,
        "--quantization", quantization,
        "--tp", str(tp_degree),
        "--batch_size", str(batch_size),
        "--seq_len", str(seq_len),
        "--phase", phase,
        "--num_runs", "3",
        "--warmup_runs", "1",
    ]

    print(f"[NSYS] Profiling: {base_name}...")
    result = subprocess.run(profile_cmd, capture_output=True, text=True, timeout=1800)

    if result.returncode != 0:
        print(f"[NSYS] Profile error (rc={result.returncode}): {result.stderr[:500]}")
        return NSysResult()

    # nsys 可能自动添加后缀
    rep_file = rep_path + ".nsys-rep"
    if not os.path.exists(rep_file):
        # 尝试查找实际生成的文件
        for f in os.listdir(output_dir):
            if f.startswith(base_name) and f.endswith(".nsys-rep"):
                rep_file = os.path.join(output_dir, f)
                break

    if not os.path.exists(rep_file):
        print(f"[NSYS] Warning: .nsys-rep file not found at {rep_file}")
        return NSysResult()

    # Step 2: nsys export to SQLite
    export_cmd = [
        "nsys", "export",
        "--type", "sqlite",
        "--output", sqlite_path,
        "--force-overwrite", "true",
        rep_file,
    ]

    print(f"[NSYS] Exporting to SQLite...")
    result = subprocess.run(export_cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print(f"[NSYS] Export error: {result.stderr[:500]}")
        return NSysResult()

    if not os.path.exists(sqlite_path):
        print(f"[NSYS] Warning: SQLite file not found at {sqlite_path}")
        return NSysResult()

    # Step 3: 解析 SQLite
    print(f"[NSYS] Parsing trace...")
    nsys_result = parse_nsys_sqlite(sqlite_path)

    # 清理临时文件
    for f in [rep_file, sqlite_path]:
        try:
            os.remove(f)
        except OSError:
            pass

    return nsys_result
