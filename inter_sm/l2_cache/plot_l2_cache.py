#!/usr/bin/env python3
"""
L2 Cache Interference — Grouped vertical bar chart with error bars.
X = copy size (MB), two bars per group: Alone vs Colocated.
Slowdown ratio annotated above each Colocated bar.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator

# ── Font config ───────────────────────────────────────────────────────────────

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STFangsong"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})

# ── Raw data (10 runs each) ──────────────────────────────────────────────────

data = {
    1: {
        "Alone":     [9.38842, 8.89347, 8.87373, 8.86867, 8.87411,
                      8.86438, 8.84806, 8.84314, 8.84634, 8.84067],
        "Colocated": [14.4292, 14.4215, 14.4279, 14.4247, 14.4237,
                      14.4222, 14.4211, 14.4226, 14.4207, 14.4240],
    },
    2: {
        "Alone":     [16.6987, 16.2814, 16.3202, 16.2897, 16.3436,
                      16.2443, 16.2799, 16.2318, 16.2222, 16.2908],
        "Colocated": [28.2908, 28.1279, 28.1474, 28.0858, 28.0724,
                      27.9787, 28.3046, 28.1573, 28.3413, 28.1947],
    },
    3: {
        "Alone":     [24.6907, 24.0451, 24.1426, 24.2146, 24.2668,
                      24.0460, 23.9178, 23.9067, 24.2751, 24.1475],
        "Colocated": [44.2814, 43.8171, 43.9259, 44.7405, 43.8859,
                      43.8116, 43.5443, 43.8660, 44.0437, 43.9646],
    },
    4: {
        "Alone":     [47.4653, 48.0364, 46.8966, 47.3324, 46.7510,
                      48.7716, 48.4581, 47.5006, 47.6362, 47.5253],
        "Colocated": [60.9746, 61.0785, 60.5923, 60.4618, 61.3777,
                      61.6642, 61.8152, 61.0406, 62.0819, 60.8884],
    },
    5: {
        "Alone":     [101.840, 99.3383, 102.902, 100.814, 99.4388,
                      99.8343, 102.930, 100.174, 99.3802, 99.7542],
        "Colocated": [121.269, 117.382, 117.467, 113.802, 122.419,
                      123.463, 115.643, 116.236, 116.648, 117.441],
    },
    6: {
        "Alone":     [274.797, 274.118, 274.329, 274.197, 273.222,
                      274.273, 273.483, 274.408, 273.913, 273.792],
        "Colocated": [347.659, 347.365, 348.769, 348.214, 346.289,
                      346.021, 345.921, 345.858, 345.927, 346.794],
    },
}

sizes = sorted(data.keys())
n = len(sizes)

alone_means = np.array([np.mean(data[s]["Alone"]) for s in sizes])
alone_stds  = np.array([np.std(data[s]["Alone"], ddof=1) for s in sizes])
coloc_means = np.array([np.mean(data[s]["Colocated"]) for s in sizes])
coloc_stds  = np.array([np.std(data[s]["Colocated"], ddof=1) for s in sizes])
slowdown    = coloc_means / alone_means

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
    ax.text(x[i] + bar_width / 2, coloc_means[i] + coloc_stds[i] + 4,
            f"{slowdown[i]:.2f}×", ha="center", va="bottom",
            fontsize=7.5, fontweight="bold", color="#D4726A")

ax.set_xlabel("Copy Size (MB)", fontsize=10)
ax.set_ylabel("Kernel Completion Time (ms)", fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels([str(s) for s in sizes], fontsize=9)
ax.tick_params(axis="y", labelsize=9)
ax.yaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(fontsize=8.5, frameon=True, edgecolor="#cccccc",
          loc="upper left")

fig.tight_layout()
fig.savefig("inter_sm/l2_cache/l2_cache_results.pdf", bbox_inches="tight")
fig.savefig("inter_sm/l2_cache/l2_cache_results.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("Saved: l2_cache_results.pdf / .png")
