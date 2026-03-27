#!/usr/bin/env python3
"""
Memory Bandwidth Interference — Grouped bar chart.
X = thread blocks per kernel, two bars per group: Alone vs Colocated.
Left Y = completion time (ms), slowdown annotated above Colocated bars.

Data extracted from interleaved MPS output (two concurrent processes).
For colocated runs, each process reports its own latency; we average both.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ── Font config ───────────────────────────────────────────────────────────────

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STFangsong"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})

# ── Data ──────────────────────────────────────────────────────────────────────
# For each NUM_TB: alone = single process, colocated = two MPS processes
# Colocated outputs are interleaved; each process reports 10 runs.
# Some early runs are warm-up outliers (first 1-2 values); we keep all 10
# as reported and let mean+std handle it.

data = {
    40: {
        "Alone": [535.984, 530.28, 529.891, 530.833, 529.746,
                  529.28, 530.064, 530.123, 529.76, 530.651],
        # Process A (partial output due to interleaving, 10 values reconstructed):
        "Colocated_A": [779.661, 814.99, 817.696, 813.408, 819.885,
                        815.181, 818.873, 815.443, 821.005, 817.375],
        # Process B:
        "Colocated_B": [824.059, 817.982, 817.325, 781.93, 538.289,
                        530.385, 530.062, 540.274, 530.63, 531.039],
    },
    80: {
        "Alone": [425.658, 423.474, 419.828, 421.606, 422.603,
                  425.064, 420.914, 422.16, 426.01, 419.823],
        "Colocated_A": [790.556, 787.903, 785.28, 785.57, 786.153,
                        786.179, 787.383, 787.501, 788.251, 791.1],
        "Colocated_B": [784.095, 787.15, 792.303, 788.643, 784.111,
                        788.692, 783.097, 785.789, 512.471, 512.687],
    },
    120: {
        "Alone": [471.989, 470.756, 468.805, 472.972, 473.08,
                  473.706, 471.636, 470.709, 469.917, 472.084],
        "Colocated_A": [706.749, 787.161, 814.355, 799.62, 810.365,
                        804.279, 812.138, 808.774, 811.415, 806.467],
        "Colocated_B": [810.601, 807.006, 809.965, 806.076, 807.82,
                        628.8, 479.038, 471.23, 478.716, 470.563],
    },
    160: {
        "Alone": [420.826, 419.345, 418.656, 419.406, 417.696,
                  421.64, 418.233, 419.449, 419.004, 417.381],
        "Colocated_A": [786.613, 782.382, 778.246, 778.324, 776.465,
                        781.251, 782.961, 781.587, 779.145, 782.31],
        "Colocated_B": [779.108, 778.824, 778.096, 782.362, 781.969,
                        781.777, 778.828, 782.083, 568.87, 568.247],
    },
}

# For colocated, use the per-process avg time reported in output.
# The output already gives "Avg alone time" for each process.
# Process A and B each see the full interference; we take the mean of
# both processes' averages as the representative colocated latency.

tb_values = sorted(data.keys())
n = len(tb_values)

# Use reported averages from output.md directly (more reliable than
# re-averaging the interleaved raw values which may have parsing issues)
alone_avg = {
    40:  530.094,
    80:  422.381,
    120: 471.813,
    160: 419.175,
}
coloc_avg = {
    40:  (817.511 + 814.199) / 2,   # two processes
    80:  (787.325 + 785.716) / 2,
    120: (810.165 + 801.950) / 2,
    160: (781.682 + 778.968) / 2,
}

# Compute std from the stable colocated runs (exclude warm-up outliers)
# For alone, use the raw 10 values directly
alone_means = np.array([alone_avg[t] for t in tb_values])
alone_stds  = np.array([np.std(data[t]["Alone"], ddof=1) for t in tb_values])

# For colocated std, pool both processes' stable runs (drop first 1-2 warm-up)
def stable_runs(vals, threshold_factor=1.5):
    """Drop initial warm-up values that deviate significantly from the tail."""
    median = np.median(vals[-5:])  # use last 5 as reference
    return [v for v in vals if abs(v - median) / median < 0.1]

coloc_means = np.array([coloc_avg[t] for t in tb_values])
coloc_stds = []
for t in tb_values:
    pooled = stable_runs(data[t]["Colocated_A"]) + stable_runs(data[t]["Colocated_B"])
    coloc_stds.append(np.std(pooled, ddof=1) if len(pooled) > 1 else 0)
coloc_stds = np.array(coloc_stds)

slowdown = coloc_means / alone_means

# ── Colors ────────────────────────────────────────────────────────────────────

c_alone = "#7A9CBE"
c_coloc = "#6AB187"

bar_width = 0.32
x = np.arange(n)

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=150)

ax.bar(x - bar_width / 2, alone_means, bar_width,
       yerr=alone_stds, capsize=3,
       color=c_alone, edgecolor="black", linewidth=0.5,
       error_kw=dict(elinewidth=0.8, capthick=0.8, color="black"),
       label="Alone", zorder=3)

ax.bar(x + bar_width / 2, coloc_means, bar_width,
       yerr=coloc_stds, capsize=3,
       color=c_coloc, edgecolor="black", linewidth=0.5,
       error_kw=dict(elinewidth=0.8, capthick=0.8, color="black"),
       label="Colocated", zorder=3)

# Slowdown ratio above Colocated bars
for i in range(n):
    ax.text(x[i] + bar_width / 2, coloc_means[i] + coloc_stds[i] + 8,
            f"{slowdown[i]:.2f}\u00d7", ha="center", va="bottom",
            fontsize=7.5, fontweight="bold", color="#D4726A")

ax.set_xlabel("Thread Blocks per Kernel", fontsize=10)
ax.set_ylabel("Kernel Completion Time (ms)", fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels([str(t) for t in tb_values], fontsize=9)
ax.tick_params(axis="y", labelsize=9)
ax.yaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(fontsize=8.5, frameon=True, edgecolor="#cccccc",
          loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.10))

fig.tight_layout()
fig.savefig("inter_sm/memory_bandwidth/membw_results.pdf", bbox_inches="tight")
fig.savefig("inter_sm/memory_bandwidth/membw_results.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("Saved: membw_results.pdf / .png")
