"""
MLWD 实验矩阵定义与 V100 硬件参数常量。

实验矩阵 M = {(model, framework, config, b, s, phi)}
"""

from dataclasses import dataclass, field
from itertools import product
from typing import List, Tuple


# ── V100 硬件参数 ──────────────────────────────────────────────
V100_NUM_SMS = 80
V100_L2_CACHE_BYTES = 6 * 1024 * 1024  # 6 MB
V100_HBM_BW_GBS = 900  # GB/s
V100_CUDA_ARCH = 70


# ── 实验矩阵默认值 ────────────────────────────────────────────
DEFAULT_MODELS = ["meta-llama/Llama-2-7b-hf", "Qwen/Qwen2-7B"]
DEFAULT_FRAMEWORK = "vllm"
DEFAULT_QUANTIZATIONS = ["fp16"]
DEFAULT_TP_DEGREES = [1]
DEFAULT_BATCH_SIZES = [1, 4, 16]
DEFAULT_SEQ_LENGTHS = [32, 64, 128, 512, 2048]
DEFAULT_PHASES = ["prefill", "decode"]


@dataclass
class StressKernelConfig:
    """合成压力核参数配置（针对 V100）。"""
    # Block Scheduler 压力核: sleep_kernel 高频 launch
    bs_num_tb: int = 160          # 2x SMs
    bs_num_threads: int = 1024
    bs_num_itrs: int = 100000

    # Compute Unit 压力核: fma_fp32_ilp4
    cu_num_tb: int = 80           # = NUM_SMS
    cu_num_threads: int = 128
    cu_num_itrs: int = 500000

    # L2 Cache 压力核: copy_kernel + L2 大小工作集
    l2_num_tb: int = 40
    l2_num_threads: int = 1024
    l2_num_bytes: int = V100_L2_CACHE_BYTES
    l2_num_itrs: int = 10000

    # Memory Bandwidth 压力核: copy_kernel + 大工作集
    bw_num_tb: int = 80
    bw_num_threads: int = 1024
    bw_num_bytes: int = 2 * 1024 * 1024 * 1024  # 2 GB
    bw_num_itrs: int = 50


@dataclass
class ExperimentConfig:
    """完整实验配置。"""
    models: List[str] = field(default_factory=lambda: DEFAULT_MODELS.copy())
    framework: str = DEFAULT_FRAMEWORK
    quantizations: List[str] = field(default_factory=lambda: DEFAULT_QUANTIZATIONS.copy())
    tp_degrees: List[int] = field(default_factory=lambda: DEFAULT_TP_DEGREES.copy())
    batch_sizes: List[int] = field(default_factory=lambda: DEFAULT_BATCH_SIZES.copy())
    seq_lengths: List[int] = field(default_factory=lambda: DEFAULT_SEQ_LENGTHS.copy())
    phases: List[str] = field(default_factory=lambda: DEFAULT_PHASES.copy())

    # profiling 参数
    ncu_num_runs: int = 3
    nsys_num_runs: int = 5
    sensitivity_num_runs: int = 5
    warmup_runs: int = 2

    stress: StressKernelConfig = field(default_factory=StressKernelConfig)

    def iter_deployment_configs(self):
        """遍历所有部署配置 (model, framework, quantization, tp)。"""
        for model, quant, tp in product(self.models, self.quantizations, self.tp_degrees):
            yield model, self.framework, quant, tp

    def iter_experiment_points(self):
        """遍历完整实验矩阵 (model, framework, config, b, s, phase)。"""
        for model, framework, quant, tp in self.iter_deployment_configs():
            for b, s, phase in product(self.batch_sizes, self.seq_lengths, self.phases):
                yield {
                    "model": model,
                    "framework": framework,
                    "quantization": quant,
                    "tp_degree": tp,
                    "batch_size": b,
                    "seq_len": s,
                    "phase": phase,
                }

    def total_points(self) -> int:
        n_configs = len(self.models) * len(self.quantizations) * len(self.tp_degrees)
        n_inputs = len(self.batch_sizes) * len(self.seq_lengths) * len(self.phases)
        return n_configs * n_inputs
