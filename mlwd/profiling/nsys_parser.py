"""
Nsight Systems SQLite 导出解析器。

从 nsys 导出的 SQLite 数据库中提取 kernel 时延、launch 间隔、
时间占比和计算-访存交替频率。
"""

import sqlite3
import statistics
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .kernel_classifier import classify_kernel, KernelCategory


@dataclass
class NSysResult:
    """单个实验点的 nsys 解析结果。"""
    t_attn: Optional[float] = None       # Attention 平均时延 (μs)
    t_ffn: Optional[float] = None        # FFN 平均时延 (μs)
    g_launch: Optional[float] = None     # 平均 launch 间隔 (μs)
    r_attn: Optional[float] = None       # Attention 时间占比
    r_ffn: Optional[float] = None        # FFN 时间占比
    f_switch: Optional[float] = None     # 计算-访存交替频率 (次/秒)
    t_attn_std: Optional[float] = None
    t_ffn_std: Optional[float] = None


@dataclass
class KernelTrace:
    """单个 kernel 执行记录。"""
    name: str
    start_ns: int
    end_ns: int
    duration_ns: int
    category: KernelCategory = KernelCategory.OTHER


def parse_nsys_sqlite(db_path: str, ci_threshold: float = 2.0) -> NSysResult:
    """
    解析 nsys 导出的 SQLite 数据库。

    Args:
        db_path: nsys 导出的 .sqlite 文件路径
        ci_threshold: 区分 compute-bound / memory-bound 的 CI 阈值
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # 查询 CUDA kernel 执行记录
    # nsys SQLite schema: CUPTI_ACTIVITY_KIND_KERNEL 表
    try:
        rows = conn.execute("""
            SELECT
                demangledName as name,
                start as start_ns,
                end as end_ns,
                (end - start) as duration_ns
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            ORDER BY start ASC
        """).fetchall()
    except sqlite3.OperationalError:
        # 尝试备用表名
        try:
            rows = conn.execute("""
                SELECT
                    shortName as name,
                    start as start_ns,
                    end as end_ns,
                    (end - start) as duration_ns
                FROM CUPTI_ACTIVITY_KIND_KERNEL
                ORDER BY start ASC
            """).fetchall()
        except sqlite3.OperationalError:
            conn.close()
            print("[NSYS] Warning: cannot find kernel trace table")
            return NSysResult()

    conn.close()

    if not rows:
        return NSysResult()

    # 构建 kernel trace 列表
    traces = []
    for r in rows:
        name = r["name"]
        cat = classify_kernel(name)
        traces.append(KernelTrace(
            name=name,
            start_ns=r["start_ns"],
            end_ns=r["end_ns"],
            duration_ns=r["duration_ns"],
            category=cat,
        ))

    return _compute_features(traces)


def _compute_features(traces: List[KernelTrace]) -> NSysResult:
    """从 kernel trace 列表计算 MLWD 执行模式特征。"""
    result = NSysResult()

    # 分类统计时延
    attn_durations_us = [t.duration_ns / 1000.0 for t in traces if t.category == KernelCategory.ATTENTION]
    ffn_durations_us = [t.duration_ns / 1000.0 for t in traces if t.category == KernelCategory.FFN]

    if attn_durations_us:
        result.t_attn = statistics.mean(attn_durations_us)
        result.t_attn_std = statistics.stdev(attn_durations_us) if len(attn_durations_us) > 1 else 0.0

    if ffn_durations_us:
        result.t_ffn = statistics.mean(ffn_durations_us)
        result.t_ffn_std = statistics.stdev(ffn_durations_us) if len(ffn_durations_us) > 1 else 0.0

    # Launch 间隔: 相邻 kernel 的 start 时间差
    if len(traces) > 1:
        intervals_us = []
        for i in range(1, len(traces)):
            interval = (traces[i].start_ns - traces[i - 1].start_ns) / 1000.0
            if interval > 0:
                intervals_us.append(interval)
        if intervals_us:
            result.g_launch = statistics.mean(intervals_us)

    # 时间占比
    total_gpu_time_ns = sum(t.duration_ns for t in traces)
    if total_gpu_time_ns > 0:
        attn_time_ns = sum(t.duration_ns for t in traces if t.category == KernelCategory.ATTENTION)
        ffn_time_ns = sum(t.duration_ns for t in traces if t.category == KernelCategory.FFN)
        result.r_attn = attn_time_ns / total_gpu_time_ns
        result.r_ffn = ffn_time_ns / total_gpu_time_ns

    # 计算-访存交替频率
    # 简化分类: Attention 视为 memory-bound, FFN (GEMM) 视为 compute-bound
    if len(traces) > 1:
        trace_duration_s = (traces[-1].end_ns - traces[0].start_ns) / 1e9
        if trace_duration_s > 0:
            switches = 0
            prev_is_compute = None
            for t in traces:
                if t.category == KernelCategory.OTHER:
                    continue
                is_compute = (t.category == KernelCategory.FFN)
                if prev_is_compute is not None and is_compute != prev_is_compute:
                    switches += 1
                prev_is_compute = is_compute
            result.f_switch = switches / trace_duration_s

    return result
