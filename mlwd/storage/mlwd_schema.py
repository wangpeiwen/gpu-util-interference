"""
MLWD 数据类定义与 SQLite schema。
"""

from dataclasses import dataclass, fields, asdict
from typing import Optional


@dataclass
class DeploymentConfig:
    """部署配置主键。"""
    model: str
    framework: str = "vllm"
    quantization: str = "fp16"
    tp_degree: int = 1
    gpu_model: str = "V100"
    cuda_version: str = ""
    config_id: Optional[int] = None


@dataclass
class OperatorProfile:
    """MLWD 第一层：算子运行时画像（15 维）。"""
    config_id: int
    batch_size: int
    seq_len: int
    phase: str  # 'prefill' or 'decode'

    # 资源竞争强度 (4)
    ci_attn: Optional[float] = None
    ci_ffn: Optional[float] = None
    l2_attn: Optional[float] = None
    l2_ffn: Optional[float] = None

    # 干扰敏感度 (4)
    sigma_bs: Optional[float] = None
    sigma_cu: Optional[float] = None
    sigma_l2: Optional[float] = None
    sigma_bw: Optional[float] = None

    # 执行模式 (7)
    t_attn: Optional[float] = None      # μs
    t_ffn: Optional[float] = None       # μs
    g_launch: Optional[float] = None    # μs
    r_attn: Optional[float] = None
    r_ffn: Optional[float] = None
    f_switch: Optional[float] = None
    ipc: Optional[float] = None

    # 统计量
    t_attn_std: Optional[float] = None
    t_ffn_std: Optional[float] = None
    baseline_latency_ms: Optional[float] = None

    id: Optional[int] = None

    def to_vector(self):
        """返回 15 维 MLWD 第一层向量。"""
        return [
            self.ci_attn, self.ci_ffn, self.l2_attn, self.l2_ffn,
            self.sigma_bs, self.sigma_cu, self.sigma_l2, self.sigma_bw,
            self.t_attn, self.t_ffn, self.g_launch,
            self.r_attn, self.r_ffn, self.f_switch, self.ipc,
        ]


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS deployment_configs (
    config_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model TEXT NOT NULL,
    framework TEXT NOT NULL DEFAULT 'vllm',
    quantization TEXT NOT NULL DEFAULT 'fp16',
    tp_degree INTEGER NOT NULL DEFAULT 1,
    gpu_model TEXT,
    cuda_version TEXT,
    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model, framework, quantization, tp_degree)
);

CREATE TABLE IF NOT EXISTS mlwd_operator_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_id INTEGER NOT NULL REFERENCES deployment_configs(config_id),
    batch_size INTEGER NOT NULL,
    seq_len INTEGER NOT NULL,
    phase TEXT NOT NULL,
    -- 资源竞争强度
    ci_attn REAL, ci_ffn REAL, l2_attn REAL, l2_ffn REAL,
    -- 干扰敏感度
    sigma_bs REAL, sigma_cu REAL, sigma_l2 REAL, sigma_bw REAL,
    -- 执行模式
    t_attn REAL, t_ffn REAL, g_launch REAL,
    r_attn REAL, r_ffn REAL, f_switch REAL, ipc REAL,
    -- 统计量
    t_attn_std REAL, t_ffn_std REAL,
    baseline_latency_ms REAL,
    UNIQUE(config_id, batch_size, seq_len, phase)
);
"""
