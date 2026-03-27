"""
Nsight Compute metric 名映射与 CSV 解析。

V100 (SM 70) 使用的 metric 名与 SM 80+ 不同，此模块处理兼容性。
"""

import csv
import io
from dataclasses import dataclass
from typing import Dict, List, Optional


# V100 (SM 70) ncu metric 名称
V100_METRICS = {
    # FLOPs: FMA 指令数 (每条 FMA = 2 FLOPs)
    "flops_fma": "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "flops_fadd": "sm__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "flops_fmul": "sm__sass_thread_inst_executed_op_fmul_pred_on.sum",
    # FP16 (half precision) FLOPs
    "flops_hfma": "sm__sass_thread_inst_executed_op_hfma_pred_on.sum",
    "flops_hadd": "sm__sass_thread_inst_executed_op_hadd_pred_on.sum",
    "flops_hmul": "sm__sass_thread_inst_executed_op_hmul_pred_on.sum",
    # DRAM bytes
    "dram_bytes_read": "dram__bytes_read.sum",
    "dram_bytes_write": "dram__bytes_write.sum",
    "dram_bytes": "dram__bytes.sum",
    # L2 cache
    "l2_hit": "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
    "l2_total": "lts__t_sectors_srcunit_tex_op_read.sum",
    # IPC
    "ipc": "sm__inst_executed.avg.per_cycle_active",
    # Kernel duration
    "duration": "gpu__time_duration.sum",
}

# 所有需要采集的 metric 列表
NCU_METRIC_LIST = ",".join([
    V100_METRICS["flops_fma"],
    V100_METRICS["flops_fadd"],
    V100_METRICS["flops_fmul"],
    V100_METRICS["flops_hfma"],
    V100_METRICS["flops_hadd"],
    V100_METRICS["flops_hmul"],
    V100_METRICS["dram_bytes"],
    V100_METRICS["l2_hit"],
    V100_METRICS["l2_total"],
    V100_METRICS["ipc"],
    V100_METRICS["duration"],
])


@dataclass
class KernelMetrics:
    """单个 kernel 的 ncu 采集结果。"""
    kernel_name: str
    flops: float = 0.0       # 总浮点运算次数
    dram_bytes: float = 0.0  # DRAM 传输字节数
    l2_hit_rate: float = 0.0 # L2 命中率
    ipc: float = 0.0         # 每周期指令数
    duration_ns: float = 0.0 # kernel 执行时间 (ns)

    @property
    def compute_intensity(self) -> float:
        """计算强度 CI = FLOP / Byte。"""
        if self.dram_bytes == 0:
            return float("inf")
        return self.flops / self.dram_bytes


def parse_ncu_csv(csv_content: str) -> List[KernelMetrics]:
    """解析 ncu --csv 输出，返回每个 kernel 的指标。"""
    results = []

    # ncu CSV 输出可能有前导信息行，找到实际 CSV 头
    lines = csv_content.strip().split("\n")
    header_idx = None
    for i, line in enumerate(lines):
        if '"Kernel Name"' in line or "Kernel Name" in line:
            header_idx = i
            break

    if header_idx is None:
        return results

    csv_text = "\n".join(lines[header_idx:])
    reader = csv.DictReader(io.StringIO(csv_text))

    # 按 kernel 名聚合
    kernel_data: Dict[str, Dict] = {}

    for row in reader:
        name = row.get("Kernel Name", "").strip('"')
        metric_name = row.get("Metric Name", "").strip('"')
        metric_value = row.get("Metric Value", "0").strip('"').replace(",", "")

        try:
            value = float(metric_value)
        except ValueError:
            continue

        if name not in kernel_data:
            kernel_data[name] = {"name": name}

        data = kernel_data[name]

        # 映射 metric 到字段
        if metric_name == V100_METRICS["flops_fma"]:
            data["flops_fma"] = data.get("flops_fma", 0) + value
        elif metric_name == V100_METRICS["flops_fadd"]:
            data["flops_fadd"] = data.get("flops_fadd", 0) + value
        elif metric_name == V100_METRICS["flops_fmul"]:
            data["flops_fmul"] = data.get("flops_fmul", 0) + value
        elif metric_name == V100_METRICS["flops_hfma"]:
            data["flops_hfma"] = data.get("flops_hfma", 0) + value
        elif metric_name == V100_METRICS["flops_hadd"]:
            data["flops_hadd"] = data.get("flops_hadd", 0) + value
        elif metric_name == V100_METRICS["flops_hmul"]:
            data["flops_hmul"] = data.get("flops_hmul", 0) + value
        elif metric_name == V100_METRICS["dram_bytes"]:
            data["dram_bytes"] = value
        elif metric_name == V100_METRICS["l2_hit"]:
            data["l2_hit"] = value
        elif metric_name == V100_METRICS["l2_total"]:
            data["l2_total"] = value
        elif metric_name == V100_METRICS["ipc"]:
            data["ipc"] = value
        elif metric_name == V100_METRICS["duration"]:
            data["duration_ns"] = value

    # 转换为 KernelMetrics
    for name, data in kernel_data.items():
        # FP32 FLOPs: FMA=2, ADD=1, MUL=1
        fp32_flops = (data.get("flops_fma", 0) * 2
                      + data.get("flops_fadd", 0)
                      + data.get("flops_fmul", 0))
        # FP16 FLOPs: 同理
        fp16_flops = (data.get("flops_hfma", 0) * 2
                      + data.get("flops_hadd", 0)
                      + data.get("flops_hmul", 0))
        total_flops = fp32_flops + fp16_flops

        l2_hit = data.get("l2_hit", 0)
        l2_total = data.get("l2_total", 0)
        l2_hit_rate = l2_hit / l2_total if l2_total > 0 else 0.0

        results.append(KernelMetrics(
            kernel_name=name,
            flops=total_flops,
            dram_bytes=data.get("dram_bytes", 0),
            l2_hit_rate=l2_hit_rate,
            ipc=data.get("ipc", 0),
            duration_ns=data.get("duration_ns", 0),
        ))

    return results
