"""
MLWD 数据可视化 — 贴合 chap3 主旨。

图表设计围绕 chap3 的三个核心论点：
1. Prefill/Decode 阶段异构性：相同 (b,s) 下两阶段的干扰敏感度差异显著
2. 四维资源竞争的细粒度差异：不同维度的敏感度分布不同，不能用单一指标替代
3. MLWD 特征随 (b,s) 的变化规律：验证 MLWD 能捕捉动态参数对算子画像的影响

Usage:
    PYTHONPATH=. python mlwd/visualize.py
    PYTHONPATH=. python mlwd/visualize.py --data mlwd_output/mlwd_complete.json --output mlwd_output/plots
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties

# 字体设置：中文仿宋/宋体，英文 Times New Roman
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['mathtext.fontset'] = 'stix'  # 数学公式用 STIX (类 Times)

# 查找可用的中文字体
_CN_FONT = None
for name in ['FangSong', 'STFangsong', 'SimSun', 'STSong', 'Noto Sans CJK SC',
             'WenQuanYi Micro Hei', 'Source Han Sans SC']:
    try:
        fp = FontProperties(family=name)
        if fp.get_name() != name:
            continue
        _CN_FONT = name
        break
    except Exception:
        continue

if _CN_FONT:
    matplotlib.rcParams['font.sans-serif'] = [_CN_FONT, 'Times New Roman', 'DejaVu Sans']
    print(f"Using Chinese font: {_CN_FONT}")
else:
    matplotlib.rcParams['font.sans-serif'] = ['SimSun', 'FangSong', 'Times New Roman', 'DejaVu Sans']
    print("Warning: Chinese font not found, falling back to default")

matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']


def load_data(path):
    with open(path) as f:
        return json.load(f)


def fig1_phase_heterogeneity(data, output_dir):
    """
    图1: Prefill vs Decode 阶段异构性（chap3 核心动机）
    """
    dims = ["sigma_bs", "sigma_cu", "sigma_l2", "sigma_bw"]
    dim_labels = ["线程块\n调度器", "计算\n单元", "L2\n缓存", "显存\n带宽"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)

    for row, b in enumerate([1, 4]):
        for col, s in enumerate([32, 64, 128]):
            ax = axes[row][col]
            p_key = f"b{b}_s{s}_prefill"
            d_key = f"b{b}_s{s}_decode"

            p_vals = [data.get(p_key, {}).get(d, 0) or 0 for d in dims]
            d_vals = [data.get(d_key, {}).get(d, 0) or 0 for d in dims]

            x = np.arange(len(dims))
            w = 0.35
            bars_p = ax.bar(x - w/2, p_vals, w, label="Prefill",
                           color="#D32F2F", alpha=0.85)
            bars_d = ax.bar(x + w/2, d_vals, w, label="Decode",
                           color="#1976D2", alpha=0.85)

            for bar in bars_p:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                            f'{h:.2f}', ha='center', va='bottom', fontsize=7)
            for bar in bars_d:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                            f'{h:.2f}', ha='center', va='bottom', fontsize=7)

            ax.set_xticks(x)
            ax.set_xticklabels(dim_labels, fontsize=8)
            ax.set_title(f'b={b}, s={s}', fontsize=11, family='Times New Roman')
            if col == 0:
                ax.set_ylabel('干扰敏感度 ($\\sigma$)', fontsize=10)
            if row == 0 and col == 2:
                ax.legend(fontsize=9, loc='upper right')
            ax.grid(True, alpha=0.2, axis='y')
            ax.set_ylim(0, max(max(p_vals), max(d_vals)) * 1.15)

    fig.suptitle('Prefill 与 Decode 阶段的四维干扰敏感度对比',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "fig1_phase_heterogeneity.png"), dpi=200)
    plt.close()
    print("  fig1_phase_heterogeneity.png")


def fig2_sensitivity_heatmap(data, output_dir):
    """
    图2: 四维干扰敏感度热力图

    行按 (phase, batch_size, seq_len) 排列，列为四个资源维度。
    直观展示不同工作负载配置下的敏感度分布差异。
    """
    dims = ["sigma_bs", "sigma_cu", "sigma_l2", "sigma_bw"]
    dim_labels = ["线程块调度器", "计算单元", "L2 缓存", "显存带宽"]

    # 按 phase 分组，phase 内按 b, s 排序
    order = []
    for phase in ["prefill", "decode"]:
        for b in [1, 4]:
            for s in [32, 64, 128]:
                order.append(f"b{b}_s{s}_{phase}")

    matrix = []
    labels = []
    for k in order:
        if k in data:
            row = [data[k].get(d, 0) or 0 for d in dims]
            matrix.append(row)
            d = data[k]
            labels.append(f"b={d['batch_size']}, s={d['seq_len']}, {d['phase']}")

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(matrix, cmap="RdYlBu_r", aspect="auto", vmin=0)

    ax.set_xticks(range(len(dim_labels)))
    ax.set_xticklabels(dim_labels, fontsize=10)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)

    # 分隔线：prefill 和 decode 之间
    n_prefill = sum(1 for k in order if "prefill" in k and k in data)
    ax.axhline(y=n_prefill - 0.5, color='black', linewidth=2)
    ax.text(-0.8, n_prefill/2 - 0.5, 'Prefill', ha='center', va='center',
            fontsize=10, fontweight='bold', rotation=90)
    ax.text(-0.8, n_prefill + (len(labels) - n_prefill)/2 - 0.5, 'Decode',
            ha='center', va='center', fontsize=10, fontweight='bold', rotation=90)

    for i in range(len(labels)):
        for j in range(len(dims)):
            val = matrix[i, j]
            color = "white" if val > matrix.max() * 0.65 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=color)

    plt.colorbar(im, ax=ax, label="干扰敏感度 ($\\sigma$)", shrink=0.8)
    ax.set_title("MLWD 干扰敏感度画像", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig2_sensitivity_heatmap.png"), dpi=200)
    plt.close()
    print("  fig2_sensitivity_heatmap.png")


def fig3_sensitivity_vs_seqlen(data, output_dir):
    """
    图3: 干扰敏感度随 (b, s) 变化趋势

    验证 MLWD 第二层（请求动态状态）的必要性：
    算子画像随 batch_size 和 seq_len 变化。
    """
    dims = ["sigma_bs", "sigma_cu", "sigma_l2", "sigma_bw"]
    dim_labels = ["$\\sigma_{bs}$", "$\\sigma_{cu}$", "$\\sigma_{l2}$", "$\\sigma_{bw}$"]
    colors = ["#D32F2F", "#1976D2", "#388E3C", "#F57C00"]
    markers = ["o", "s", "^", "D"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    configs = [
        (0, 0, "prefill", 1, "Prefill, b=1"),
        (0, 1, "prefill", 4, "Prefill, b=4"),
        (1, 0, "decode", 1, "Decode, b=1"),
        (1, 1, "decode", 4, "Decode, b=4"),
    ]

    for row, col, phase, b, title in configs:
        ax = axes[row][col]
        seq_lens = [32, 64, 128]

        for i, (dim, label) in enumerate(zip(dims, dim_labels)):
            vals = []
            for s in seq_lens:
                key = f"b{b}_s{s}_{phase}"
                vals.append(data.get(key, {}).get(dim, 0) or 0)

            ax.plot(seq_lens, vals, f"{markers[i]}-", color=colors[i],
                    label=label, linewidth=2, markersize=7)

        ax.set_xlabel("序列长度 (s)", fontsize=10)
        ax.set_ylabel("干扰敏感度 ($\\sigma$)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(seq_lens)

    fig.suptitle("干扰敏感度随 (batch_size, seq_len) 的变化趋势",
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "fig3_sensitivity_vs_seqlen.png"), dpi=200)
    plt.close()
    print("  fig3_sensitivity_vs_seqlen.png")


def fig4_execution_pattern(data, output_dir):
    """
    图4: 执行模式特征对比（MLWD 第一层第三组）

    对比不同 (b, s, phase) 下的 Attention/FFN 时间占比和交替频率，
    展示 Kernel 级执行模式的差异。
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 子图1: Attention vs FFN 时间占比堆叠柱状图
    ax = axes[0]
    keys = []
    r_attn_vals = []
    r_ffn_vals = []
    r_other_vals = []
    for b in [1, 4]:
        for s in [32, 128]:
            for phase in ["prefill", "decode"]:
                key = f"b{b}_s{s}_{phase}"
                if key in data:
                    keys.append(f"b{b}s{s}\n{phase[:3]}")
                    ra = data[key].get("r_attn", 0) or 0
                    rf = data[key].get("r_ffn", 0) or 0
                    r_attn_vals.append(ra)
                    r_ffn_vals.append(rf)
                    r_other_vals.append(max(0, 1 - ra - rf))

    x = np.arange(len(keys))
    ax.bar(x, r_ffn_vals, label="FFN", color="#1976D2", alpha=0.85)
    ax.bar(x, r_attn_vals, bottom=r_ffn_vals, label="Attention", color="#D32F2F", alpha=0.85)
    ax.bar(x, r_other_vals, bottom=[a+f for a, f in zip(r_attn_vals, r_ffn_vals)],
           label="Other", color="#BDBDBD", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, fontsize=8)
    ax.set_ylabel("时间占比", fontsize=10)
    ax.set_title("Attention/FFN 时间占比", fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)

    # 子图2: Kernel 平均时延
    ax = axes[1]
    keys2 = []
    t_attn_vals = []
    t_ffn_vals = []
    for b in [1, 4]:
        for s in [32, 128]:
            key_p = f"b{b}_s{s}_prefill"
            if key_p in data and data[key_p].get("t_ffn"):
                keys2.append(f"b{b}s{s}")
                t_attn_vals.append(data[key_p].get("t_attn", 0) or 0)
                t_ffn_vals.append(data[key_p].get("t_ffn", 0) or 0)

    if keys2:
        x2 = np.arange(len(keys2))
        w = 0.35
        ax.bar(x2 - w/2, t_attn_vals, w, label="$\\bar{t}_{attn}$", color="#D32F2F", alpha=0.85)
        ax.bar(x2 + w/2, t_ffn_vals, w, label="$\\bar{t}_{ffn}$", color="#1976D2", alpha=0.85)
        ax.set_xticks(x2)
        ax.set_xticklabels(keys2, fontsize=9)
        ax.set_ylabel("时延 ($\\mu$s)", fontsize=10)
        ax.set_title("Kernel 平均时延", fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2, axis='y')

    # 子图3: 计算-访存交替频率
    ax = axes[2]
    keys3 = []
    fswitch_vals = []
    for b in [1, 4]:
        for s in [32, 64, 128]:
            for phase in ["prefill", "decode"]:
                key = f"b{b}_s{s}_{phase}"
                if key in data and data[key].get("f_switch") is not None:
                    keys3.append(f"b{b}s{s}\n{phase[:3]}")
                    fswitch_vals.append(data[key]["f_switch"])

    if keys3:
        x3 = np.arange(len(keys3))
        colors3 = ["#D32F2F" if "pre" in k else "#1976D2" for k in keys3]
        ax.bar(x3, fswitch_vals, color=colors3, alpha=0.85)
        ax.set_xticks(x3)
        ax.set_xticklabels(keys3, fontsize=7)
        ax.set_ylabel("$f_{switch}$ (次/秒)", fontsize=10)
        ax.set_title("计算-访存交替频率", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.2, axis='y')
        # 图例
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color="#D32F2F", label="Prefill"),
                          Patch(color="#1976D2", label="Decode")], fontsize=9)

    fig.suptitle("MLWD 执行模式特征", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(os.path.join(output_dir, "fig4_execution_pattern.png"), dpi=200)
    plt.close()
    print("  fig4_execution_pattern.png")


def fig5_baseline_scaling(data, output_dir):
    """
    图5: Baseline 时延随 (b, s) 的缩放关系

    展示 Prefill 和 Decode 的时延缩放差异，
    说明请求动态状态（第二层）对调度决策的重要性。
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, phase in enumerate(["prefill", "decode"]):
        ax = axes[idx]
        for b, color, marker in [(1, "#D32F2F", "o"), (4, "#1976D2", "s")]:
            seq_lens = []
            baselines = []
            for s in [32, 64, 128]:
                key = f"b{b}_s{s}_{phase}"
                bl = data.get(key, {}).get("baseline_ms")
                if bl:
                    seq_lens.append(s)
                    baselines.append(bl)

            ax.plot(seq_lens, baselines, f"{marker}-", color=color,
                    label=f"batch={b}", linewidth=2, markersize=8)
            for s, bl in zip(seq_lens, baselines):
                ax.annotate(f"{bl:.0f}ms", (s, bl), textcoords="offset points",
                           xytext=(0, 10), ha='center', fontsize=8)

        ax.set_xlabel("序列长度 (s)", fontsize=11)
        ax.set_ylabel("基线时延 (ms)", fontsize=11)
        ax.set_title(f"{phase.capitalize()}", fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([32, 64, 128])

    fig.suptitle("基线时延随 (batch_size, seq_len) 的缩放关系",
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(os.path.join(output_dir, "fig5_baseline_scaling.png"), dpi=200)
    plt.close()
    print("  fig5_baseline_scaling.png")


def main():
    parser = argparse.ArgumentParser(description="MLWD Data Visualization")
    parser.add_argument("--data", type=str, default="mlwd_output/mlwd_complete.json")
    parser.add_argument("--output", type=str, default="mlwd_output/plots")
    args = parser.parse_args()

    data = load_data(args.data)
    os.makedirs(args.output, exist_ok=True)

    print(f"Loaded {len(data)} entries\n")
    print("Generating plots:")

    fig1_phase_heterogeneity(data, args.output)
    fig2_sensitivity_heatmap(data, args.output)
    fig3_sensitivity_vs_seqlen(data, args.output)
    fig4_execution_pattern(data, args.output)
    fig5_baseline_scaling(data, args.output)

    print(f"\nAll plots saved to {args.output}/")


if __name__ == "__main__":
    main()
