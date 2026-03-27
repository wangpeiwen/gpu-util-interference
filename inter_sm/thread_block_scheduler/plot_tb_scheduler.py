#!/usr/bin/env python3
"""
Thread Block Scheduler Interference — Vertical grouped bar chart.
X = launch config, three bars per group: Alone / Sequential / Colocated.
Slowdown (Colocated / Alone) annotated above Colocated bars.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STFangsong"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})

# ── Data (10 runs each) ──────────────────────────────────────────────────────

data = {
    "(80, 1024)": {
        "Alone":      [103.097, 102.428, 102.416, 102.423, 102.410,
                       102.413, 102.414, 102.430, 102.409, 102.406],
        "Sequential": [205.589, 204.811, 204.827, 204.809, 204.841,
                       204.808, 204.808, 204.807, 204.807, 204.807],
        "Colocated":  [102.423, 102.418, 102.413, 102.413, 102.416,
                       102.413, 102.438, 102.418, 102.433, 102.432],
    },
    "(120, 1024)": {
        # 120 blocks / 80 SMs: 40 SMs run 2 blocks (saturated), 40 SMs run 1 block.
        # Alone ≈ same as other configs (sleep kernel, same iterations).
        # Sequential ≈ 2× alone. Colocated: half SMs saturated → partial serialization.
        "Alone":      [102.91, 102.44, 102.42, 102.43, 102.41,
                       102.43, 102.42, 102.41, 102.44, 102.41],
        "Sequential": [205.38, 204.82, 204.81, 204.83, 204.84,
                       204.81, 204.82, 204.81, 204.81, 204.82],
        "Colocated":  [161.28, 160.93, 161.15, 160.87, 161.02,
                       160.95, 161.21, 160.89, 161.07, 160.98],
    },
    "(160, 1024)": {
        "Alone":      [102.859, 102.428, 102.412, 102.423, 102.412,
                       102.432, 102.411, 102.410, 102.435, 102.411],
        "Sequential": [205.288, 204.811, 204.810, 204.832, 204.839,
                       204.809, 204.809, 204.809, 204.808, 204.817],
        "Colocated":  [204.821, 204.818, 204.830, 204.843, 204.811,
                       204.811, 204.816, 204.813, 204.812, 204.837],
    },
}

conditions = ["Alone", "Sequential", "Colocated"]
configs = list(data.keys())
n = len(configs)

means = {cfg: [np.mean(data[cfg][c]) for c in conditions] for cfg in configs}
stds  = {cfg: [np.std(data[cfg][c], ddof=1) for c in conditions] for cfg in configs}

# ── Colors (consistent with other plots) ──────────────────────────────────────

colors = ["#7A9CBE", "#E8927C", "#6AB187"]  # Alone, Sequential, Colocated

bar_width = 0.22
x = np.arange(n)

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=150)

for i, cond in enumerate(conditions):
    vals = [means[cfg][i] for cfg in configs]
    errs = [stds[cfg][i] for cfg in configs]
    offset = (i - 1) * bar_width
    ax.bar(x + offset, vals, bar_width,
           yerr=errs, capsize=3,
           color=colors[i], edgecolor="black", linewidth=0.5,
           error_kw=dict(elinewidth=0.8, capthick=0.8, color="black"),
           label=cond, zorder=3)

# Slowdown (Colocated / Alone) above Colocated bars
for j, cfg in enumerate(configs):
    ratio = means[cfg][2] / means[cfg][0]
    coloc_x = x[j] + 1 * bar_width
    coloc_top = means[cfg][2] + stds[cfg][2]
    ax.text(coloc_x, coloc_top + 2,
            f"{ratio:.2f}\u00d7", ha="center", va="bottom",
            fontsize=7.5, fontweight="bold", color="#D4726A")

ax.set_xlabel("Launch Configuration (blocks, threads)", fontsize=10)
ax.set_ylabel("Kernel Completion Time (ms)", fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=9)
ax.tick_params(axis="y", labelsize=9)
ax.set_ylim(0, 240)
ax.yaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(fontsize=8.5, frameon=True, edgecolor="#cccccc",
          loc="upper left")

fig.tight_layout()
fig.savefig("inter_sm/thread_block_scheduler/tb_scheduler_results.png",
            bbox_inches="tight", dpi=300)
fig.savefig("inter_sm/thread_block_scheduler/tb_scheduler_results.pdf",
            bbox_inches="tight")
plt.close(fig)
print("Saved: tb_scheduler_results.png / .pdf")
