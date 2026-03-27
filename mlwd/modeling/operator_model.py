"""
参数化算子模型：基于 (b, s) 的 MLWD 特征回归外推。

对 MLWD 第一层中的每个特征，训练轻量级回归模型，
支持对未采样的 (b, s) 组合进行预测。
"""

import json
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


FEATURE_NAMES = [
    "ci_attn", "ci_ffn", "l2_attn", "l2_ffn",
    "sigma_bs", "sigma_cu", "sigma_l2", "sigma_bw",
    "t_attn", "t_ffn", "g_launch",
    "r_attn", "r_ffn", "f_switch", "ipc",
]


@dataclass
class OperatorModel:
    """单个特征的回归模型。"""
    feature_name: str
    coefficients: np.ndarray  # [bias, b, s, b*s, b^2, s^2]
    r_squared: float = 0.0

    def predict(self, b: float, s: float) -> float:
        """预测给定 (b, s) 下的特征值。"""
        x = np.array([1, b, s, b * s, b ** 2, s ** 2])
        return float(x @ self.coefficients)


def _build_design_matrix(bs_pairs: List[Tuple[int, int]]) -> np.ndarray:
    """构建多项式设计矩阵 [1, b, s, b*s, b^2, s^2]。"""
    X = []
    for b, s in bs_pairs:
        X.append([1, b, s, b * s, b ** 2, s ** 2])
    return np.array(X)


def train_operator_models(profiles: list, phase: str) -> Dict[str, OperatorModel]:
    """
    对给定 phase 的所有 profile 数据训练回归模型。

    Args:
        profiles: OperatorProfile 列表
        phase: 'prefill' or 'decode'

    Returns:
        {feature_name: OperatorModel} 字典
    """
    # 过滤指定 phase
    phase_profiles = [p for p in profiles if p.phase == phase]
    if len(phase_profiles) < 3:
        print(f"  [Model] Not enough data for {phase} (need >= 3, got {len(phase_profiles)})")
        return {}

    bs_pairs = [(p.batch_size, p.seq_len) for p in phase_profiles]
    X = _build_design_matrix(bs_pairs)

    models = {}
    for feat in FEATURE_NAMES:
        y_values = [getattr(p, feat) for p in phase_profiles]
        # 跳过全 None 的特征
        if all(v is None for v in y_values):
            continue

        # 用 0 填充 None（或跳过）
        valid_mask = [v is not None for v in y_values]
        if sum(valid_mask) < 3:
            continue

        X_valid = X[np.array(valid_mask)]
        y_valid = np.array([v for v, m in zip(y_values, valid_mask) if m])

        # OLS: coefficients = (X^T X)^{-1} X^T y
        try:
            coeffs, residuals, rank, sv = np.linalg.lstsq(X_valid, y_valid, rcond=None)
        except np.linalg.LinAlgError:
            continue

        # R^2
        y_pred = X_valid @ coeffs
        ss_res = np.sum((y_valid - y_pred) ** 2)
        ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        models[feat] = OperatorModel(
            feature_name=feat,
            coefficients=coeffs,
            r_squared=r2,
        )
        print(f"  [Model] {feat} ({phase}): R²={r2:.4f}")

    return models


def save_models(models: Dict[str, Dict[str, OperatorModel]], output_path: str):
    """保存所有模型到文件。"""
    serializable = {}
    for key, phase_models in models.items():
        serializable[key] = {
            feat: {
                "feature_name": m.feature_name,
                "coefficients": m.coefficients.tolist(),
                "r_squared": m.r_squared,
            }
            for feat, m in phase_models.items()
        }

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  [Model] Saved to {output_path}")


def load_models(input_path: str) -> Dict[str, Dict[str, OperatorModel]]:
    """从文件加载模型。"""
    with open(input_path, "r") as f:
        data = json.load(f)

    models = {}
    for key, phase_models in data.items():
        models[key] = {
            feat: OperatorModel(
                feature_name=d["feature_name"],
                coefficients=np.array(d["coefficients"]),
                r_squared=d["r_squared"],
            )
            for feat, d in phase_models.items()
        }
    return models


def predict_profile(models: Dict[str, OperatorModel],
                    batch_size: int, seq_len: int) -> Dict[str, float]:
    """用回归模型预测给定 (b, s) 的所有特征。"""
    return {
        feat: model.predict(batch_size, seq_len)
        for feat, model in models.items()
    }
