"""
Trace 解析脚本：从 nsys/ncu 生成的 trace 文件中提取 MLWD 特征，
合并到 mlwd_results.json。

Usage:
    # 解析 nsys trace（先导出 SQLite）
    nsys export --type sqlite --output /tmp/mlwd_trace.sqlite /tmp/mlwd_trace.nsys-rep
    python mlwd/parse_traces.py --nsys /tmp/mlwd_trace.sqlite --output mlwd_output/mlwd_results.json

    # 解析 ncu CSV
    python mlwd/parse_traces.py --ncu /tmp/ncu_out.csv --output mlwd_output/mlwd_results.json

    # 同时解析两者
    python mlwd/parse_traces.py --nsys /tmp/mlwd_trace.sqlite --ncu /tmp/ncu_out.csv \
      --output mlwd_output/mlwd_results.json
"""

import argparse
import json
import os
import sqlite3
import statistics
from typing import Dict, List, Optional

# 直接导入解析模块（避免包依赖问题）
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlwd.profiling.kernel_classifier import classify_kernel, KernelCategory
from mlwd.profiling.ncu_metrics import parse_ncu_csv


def parse_nsys_trace(sqlite_path: str, meta_path: str = None) -> Dict[str, dict]:
    """
    解析 nsys SQLite trace，按 kernel 时间间隔中的大 gap 分段。

    由于 NVTX marker（主进程 CPU 时间）和 kernel（子进程 GPU 时间）
    在不同时间域，无法直接用 NVTX 时间范围匹配 kernel。
    改为：用 kernel 之间的大时间间隔作为分段边界，
    再用 NVTX 的顺序给每段命名。

    Args:
        sqlite_path: nsys 导出的 SQLite 文件
        meta_path: run_profiling.py 输出的元数据 JSON（可选，用于验证）
    """
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row

    # 获取 NVTX ranges 的顺序（用于命名）
    nvtx_keys = []
    try:
        str_ids = {}
        for r in conn.execute("SELECT id, value FROM StringIds").fetchall():
            str_ids[r[0]] = r[1]

        rows = conn.execute("""
            SELECT text, start FROM NVTX_EVENTS
            WHERE eventType = 59 OR eventType = 60
            ORDER BY start ASC
        """).fetchall()

        seen = set()
        for r in rows:
            text = r["text"]
            if isinstance(text, int):
                text = str_ids.get(text, str(text))
            # 去掉 _runN 后缀，去重
            parts = str(text).rsplit("_run", 1)
            key = parts[0] if len(parts) == 2 else str(text)
            if key not in seen:
                seen.add(key)
                nvtx_keys.append(key)
    except sqlite3.OperationalError:
        pass

    # 获取所有 CUDA kernel
    try:
        kernels = conn.execute("""
            SELECT s.value as name, k.start, k.end, (k.end - k.start) as duration_ns
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON k.demangledName = s.id
            ORDER BY k.start ASC
        """).fetchall()
    except sqlite3.OperationalError:
        try:
            kernels = conn.execute("""
                SELECT s.value as name, k.start, k.end, (k.end - k.start) as duration_ns
                FROM CUPTI_ACTIVITY_KIND_KERNEL k
                JOIN StringIds s ON k.shortName = s.id
                ORDER BY k.start ASC
            """).fetchall()
        except sqlite3.OperationalError:
            conn.close()
            print("[NSYS] Cannot find kernel trace table")
            return {}

    conn.close()

    if not kernels:
        print("[NSYS] No kernels found in trace")
        return {}

    print(f"[NSYS] Total kernels: {len(kernels)}, NVTX keys: {nvtx_keys}")

    # 按大 gap 分段
    segments = _split_by_gaps(kernels)
    print(f"[NSYS] Found {len(segments)} segments by gap detection")

    # 跳过第一段（模型加载 + warmup），剩余段与 NVTX keys 对齐
    # run_profiling.py 的结构：每个实验点有 warmup + num_runs 次推理
    # warmup 和实际 run 之间的 gap 较小，实验点之间的 gap 较大
    # 所以大 gap 分出的段对应每个实验点（包含 warmup + runs）

    results = {}
    for i, seg_kernels in enumerate(segments):
        if i < len(nvtx_keys):
            key = nvtx_keys[i]
        else:
            key = f"segment_{i}"

        features = _compute_segment_features(seg_kernels)
        if features:
            results[key] = features

    return results


def _split_by_gaps(kernels, min_gap_ratio: float = 10.0) -> List[list]:
    """
    按 kernel 之间的大时间间隔分段。

    大 gap 定义：间隔 > 中位间隔 * min_gap_ratio
    """
    if len(kernels) <= 1:
        return [list(kernels)]

    # 计算所有间隔
    gaps = []
    for i in range(1, len(kernels)):
        gap = kernels[i]["start"] - kernels[i-1]["end"]
        gaps.append(gap)

    # 用中位数的倍数作为阈值
    sorted_gaps = sorted(gaps)
    median_gap = sorted_gaps[len(sorted_gaps) // 2]
    threshold = max(median_gap * min_gap_ratio, 100_000_000)  # 至少 100ms

    # 分段
    segments = []
    current = [kernels[0]]
    for i in range(1, len(kernels)):
        if gaps[i-1] > threshold:
            segments.append(current)
            current = [kernels[i]]
        else:
            current.append(kernels[i])
    segments.append(current)

    return segments


def _compute_segment_features(kernels) -> Optional[dict]:
    """从一段 kernel 列表计算 MLWD 执行模式特征。"""
    if not kernels:
        return None

    attn_durations = []
    ffn_durations = []
    all_durations = []
    categories = []

    for k in kernels:
        name = k["name"]
        dur_us = k["duration_ns"] / 1000.0
        cat = classify_kernel(name)

        all_durations.append(dur_us)
        categories.append(cat)

        if cat == KernelCategory.ATTENTION:
            attn_durations.append(dur_us)
        elif cat == KernelCategory.FFN:
            ffn_durations.append(dur_us)

    result = {}

    # t_attn, t_ffn
    if attn_durations:
        result["t_attn"] = round(statistics.mean(attn_durations), 4)
        result["t_attn_std"] = round(statistics.stdev(attn_durations), 4) if len(attn_durations) > 1 else 0.0
    if ffn_durations:
        result["t_ffn"] = round(statistics.mean(ffn_durations), 4)
        result["t_ffn_std"] = round(statistics.stdev(ffn_durations), 4) if len(ffn_durations) > 1 else 0.0

    # g_launch
    if len(kernels) > 1:
        intervals = []
        for i in range(1, len(kernels)):
            iv = (kernels[i]["start"] - kernels[i-1]["start"]) / 1000.0
            if iv > 0:
                intervals.append(iv)
        if intervals:
            result["g_launch"] = round(statistics.mean(intervals), 4)

    # r_attn, r_ffn
    total_time = sum(all_durations)
    if total_time > 0:
        result["r_attn"] = round(sum(attn_durations) / total_time, 6)
        result["r_ffn"] = round(sum(ffn_durations) / total_time, 6)

    # f_switch
    if len(kernels) > 1:
        trace_dur_s = (kernels[-1]["end"] - kernels[0]["start"]) / 1e9
        if trace_dur_s > 0:
            switches = 0
            prev_compute = None
            for cat in categories:
                if cat == KernelCategory.OTHER:
                    continue
                is_compute = (cat == KernelCategory.FFN)
                if prev_compute is not None and is_compute != prev_compute:
                    switches += 1
                prev_compute = is_compute
            result["f_switch"] = round(switches / trace_dur_s, 4)

    # kernel 统计
    result["num_kernels"] = len(kernels)
    result["num_attn_kernels"] = len(attn_durations)
    result["num_ffn_kernels"] = len(ffn_durations)

    return result


def parse_ncu_trace(csv_path: str) -> dict:
    """解析 ncu CSV，提取 CI, L2, IPC。"""
    with open(csv_path, "r") as f:
        csv_content = f.read()

    kernel_metrics = parse_ncu_csv(csv_content)
    if not kernel_metrics:
        print("[NCU] No kernel metrics parsed")
        return {}

    attn_cis, attn_l2s = [], []
    ffn_cis, ffn_l2s = [], []
    all_ipcs = []

    for km in kernel_metrics:
        cat = classify_kernel(km.kernel_name)
        if cat == KernelCategory.ATTENTION:
            if km.dram_bytes > 0:
                attn_cis.append(km.compute_intensity)
            attn_l2s.append(km.l2_hit_rate)
        elif cat == KernelCategory.FFN:
            if km.dram_bytes > 0:
                ffn_cis.append(km.compute_intensity)
            ffn_l2s.append(km.l2_hit_rate)
        all_ipcs.append(km.ipc)

    result = {}
    if attn_cis:
        result["ci_attn"] = round(statistics.mean(attn_cis), 6)
    if attn_l2s:
        result["l2_attn"] = round(statistics.mean(attn_l2s), 6)
    if ffn_cis:
        result["ci_ffn"] = round(statistics.mean(ffn_cis), 6)
    if ffn_l2s:
        result["l2_ffn"] = round(statistics.mean(ffn_l2s), 6)
    if all_ipcs:
        result["ipc"] = round(statistics.mean(all_ipcs), 6)

    result["num_profiled_kernels"] = len(kernel_metrics)
    return result


def merge_to_json(output_path: str, new_data: Dict[str, dict], prefix: str = ""):
    """合并新数据到已有的 mlwd_results.json。"""
    existing = {}
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            existing = json.load(f)

    for key, features in new_data.items():
        full_key = f"{prefix}{key}" if prefix else key
        if full_key not in existing:
            existing[full_key] = {}
        existing[full_key].update(features)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    print(f"Merged {len(new_data)} entries into {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Parse nsys/ncu traces into MLWD features")
    parser.add_argument("--nsys", type=str, help="nsys 导出的 SQLite 文件路径")
    parser.add_argument("--ncu", type=str, help="ncu 输出的 CSV 文件路径")
    parser.add_argument("--output", type=str, default="mlwd_output/mlwd_results.json",
                        help="输出 JSON 路径")
    parser.add_argument("--prefix", type=str, default="",
                        help="key 前缀（如模型名）")
    args = parser.parse_args()

    if not args.nsys and not args.ncu:
        parser.error("至少指定 --nsys 或 --ncu 之一")

    if args.nsys:
        print(f"[NSYS] Parsing: {args.nsys}")
        nsys_data = parse_nsys_trace(args.nsys)
        print(f"[NSYS] Extracted {len(nsys_data)} segments")
        for key, features in nsys_data.items():
            print(f"  {key}: {len(features)} features")
        merge_to_json(args.output, nsys_data, args.prefix)

    if args.ncu:
        print(f"\n[NCU] Parsing: {args.ncu}")
        ncu_data = parse_ncu_trace(args.ncu)
        print(f"[NCU] Extracted: {ncu_data}")
        # ncu 通常是单个实验点，用 prefix 作为 key
        if ncu_data:
            key = args.prefix or "ncu_profile"
            merge_to_json(args.output, {key: ncu_data})


if __name__ == "__main__":
    main()
