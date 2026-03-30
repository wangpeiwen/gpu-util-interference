"""
生成 chap3 用的 MLWD 15 维向量示例表格（LaTeX 格式）。

15 维定义：
  资源竞争强度 (4): CI_attn, CI_ffn, L2_attn, L2_ffn
  干扰敏感度 (4): σ_bs, σ_cu, σ_l2, σ_bw
  执行模式 (7): t_attn, t_ffn, g_launch, r_attn, r_ffn, f_switch, IPC
"""

import json

with open("mlwd_output/mlwd_complete.json") as f:
    data = json.load(f)

# 15 维字段映射
FIELDS = [
    ("CI_attn",    "ci_attn",   "FLOP/Byte"),
    ("CI_ffn",     "ci_ffn",    "FLOP/Byte"),
    ("L2_attn",    "l2_attn",   ""),
    ("L2_ffn",     "l2_ffn",    ""),
    ("t_attn",     "t_attn",    "μs"),
    ("t_ffn",      "t_ffn",     "μs"),
    ("g_launch",   "g_launch",  "μs"),
    ("σ_bs",       "sigma_bs",  ""),
    ("σ_cu",       "sigma_cu",  ""),
    ("σ_l2",       "sigma_l2",  ""),
    ("σ_bw",       "sigma_bw",  ""),
    ("r_attn",     "r_attn",    ""),
    ("r_ffn",      "r_ffn",     ""),
    ("f_switch",   "f_switch",  "次/秒"),
    ("IPC",        "ipc",       ""),
]

# 选两个代表性实验点：b1_s128 prefill vs decode
examples = [
    ("b1_s128_prefill", "Prefill (b=1, s=128)"),
    ("b1_s128_decode",  "Decode (b=1, s=128)"),
    ("b4_s128_prefill", "Prefill (b=4, s=128)"),
    ("b4_s128_decode",  "Decode (b=4, s=128)"),
]

print("=" * 80)
print("MLWD 15 维向量示例（Qwen2.5-7B-Instruct, V100, FP16, vLLM）")
print("=" * 80)

# 打印表格
header = f"{'特征':<12} {'单位':<10}"
for _, label in examples:
    header += f" {label:>22}"
print(header)
print("-" * len(header))

for name, key, unit in FIELDS:
    row = f"{name:<12} {unit:<10}"
    for exp_key, _ in examples:
        val = data.get(exp_key, {}).get(key)
        if val is not None:
            if isinstance(val, float):
                if val > 100:
                    row += f" {val:>22.1f}"
                elif val > 1:
                    row += f" {val:>22.4f}"
                else:
                    row += f" {val:>22.6f}"
            else:
                row += f" {str(val):>22}"
        else:
            row += f" {'—':>22}"
    print(row)

# 生成 LaTeX 表格
print("\n\n% ===== LaTeX 表格 =====")
print(r"\begin{table}[htbp]")
print(r"\centering")
print(r"\caption{Qwen2.5-7B-Instruct 在 V100 上的 MLWD 向量示例}")
print(r"\label{tab:mlwd_example}")
print(r"\small")
print(r"\renewcommand{\arraystretch}{1.2}")
print(r"\begin{tabular}{l c c c c c}")
print(r"\toprule")
print(r"特征 & 单位 & \makecell{Prefill\\(b=1,s=128)} & \makecell{Decode\\(b=1,s=128)} & \makecell{Prefill\\(b=4,s=128)} & \makecell{Decode\\(b=4,s=128)} \\")
print(r"\midrule")

group_labels = {
    0: r"\multicolumn{6}{l}{\textit{资源竞争强度}} \\",
    4: r"\multicolumn{6}{l}{\textit{执行模式}} \\",
    7: r"\multicolumn{6}{l}{\textit{干扰敏感度}} \\",
    11: r"\multicolumn{6}{l}{\textit{执行模式（续）}} \\",
}

# 重新排列为 chap3 的三组顺序
ordered = [
    # 资源竞争强度
    ("$\\mathrm{CI}_{\\mathrm{attn}}$", "ci_attn", "FLOP/Byte"),
    ("$\\mathrm{CI}_{\\mathrm{ffn}}$", "ci_ffn", "FLOP/Byte"),
    ("$\\mathrm{L2}_{\\mathrm{attn}}$", "l2_attn", ""),
    ("$\\mathrm{L2}_{\\mathrm{ffn}}$", "l2_ffn", ""),
    # 干扰敏感度
    ("$\\sigma_{\\mathrm{bs}}$", "sigma_bs", ""),
    ("$\\sigma_{\\mathrm{cu}}$", "sigma_cu", ""),
    ("$\\sigma_{\\mathrm{l2}}$", "sigma_l2", ""),
    ("$\\sigma_{\\mathrm{bw}}$", "sigma_bw", ""),
    # 执行模式
    ("$\\bar{t}_{\\mathrm{attn}}$", "t_attn", "$\\mu$s"),
    ("$\\bar{t}_{\\mathrm{ffn}}$", "t_ffn", "$\\mu$s"),
    ("$\\bar{g}_{\\mathrm{launch}}$", "g_launch", "$\\mu$s"),
    ("$r_{\\mathrm{attn}}$", "r_attn", ""),
    ("$r_{\\mathrm{ffn}}$", "r_ffn", ""),
    ("$f_{\\mathrm{switch}}$", "f_switch", "次/秒"),
    ("$\\overline{\\mathrm{IPC}}$", "ipc", ""),
]

group_headers = [
    (0, "资源竞争强度"),
    (4, "干扰敏感度"),
    (8, "执行模式"),
]

for i, (name, key, unit) in enumerate(ordered):
    # 组标题
    for gi, glabel in group_headers:
        if i == gi:
            print(f"\\midrule")
            print(f"\\multicolumn{{6}}{{l}}{{\\textit{{{glabel}}}}} \\\\")

    vals = []
    for exp_key, _ in examples:
        val = data.get(exp_key, {}).get(key)
        if val is not None:
            if isinstance(val, float):
                if val > 100:
                    vals.append(f"{val:.1f}")
                elif val > 1:
                    vals.append(f"{val:.4f}")
                else:
                    vals.append(f"{val:.4f}")
            else:
                vals.append(str(val))
        else:
            vals.append("—")

    print(f"{name} & {unit} & {' & '.join(vals)} \\\\")

print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")

# 打印 15 维向量
print("\n\n% ===== 向量表示 =====")
for exp_key, label in examples:
    d = data.get(exp_key, {})
    vec = []
    for _, key, _ in ordered:
        val = d.get(key)
        if val is not None:
            vec.append(f"{val:.4f}" if isinstance(val, float) else str(val))
        else:
            vec.append("—")
    print(f"\n% {label}")
    print(f"% W = [{', '.join(vec)}]")
