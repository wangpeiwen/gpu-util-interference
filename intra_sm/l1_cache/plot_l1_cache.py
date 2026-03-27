#!/usr/bin/env python3
"""
L1 Cache Interference — Grouped bar chart.
X = copy size per thread block (KB), two bars: Sequential vs Colocated.
Slowdown = Colocated / (Sequential / 2), annotated above Colocated bars.
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

# ── Data (10 runs each, from second clean run) ───────────────────────────────

data = {
    16: {
        "Sequential": [94.121, 86.6654, 86.1906, 84.6992, 83.9876,
                       83.9856, 83.9883, 84.0059, 83.9942, 83.9874],
        "Colocated":  [42.343, 42.4074, 42.4283, 42.4264, 42.3465,
                       42.3119, 42.4086, 42.4195, 42.4097, 42.3842],
    },
    32: {
        "Sequential": [167.149, 167.109, 167.104, 167.108, 167.109,
                       167.142, 167.113, 167.106, 167.134, 167.113],
        "Colocated":  [153.71, 155.378, 153.247, 156.302, 156.105,
                       155.859, 155.353, 153.252, 154.019, 156.256],
    },
    48: {
        "Sequential": [250.244, 250.207, 250.232, 250.211, 250.249,
                       250.207, 250.214, 250.234, 250.234, 250.236],
        "Colocated":  [821.78, 818.18, 813.23, 828.926, 764.583,
                       801.186, 811.206, 797.743, 819.899, 837.098],
    },
    64: {
        "Sequential": [592.863, 593.969, 592.162, 594.862, 592.537,
                       593.173, 594.718, 593.624, 594.5, 594.351],
        "Colocated":  [1817.41, 1817.25, 1817.75, 1819.1, 1819.94,
                       1816.78, 1818.48, 1817.86, 1821.42, 1818.98],
    },
}

sizes = sorted(data.keys())
n = len(sizes)

seq_means  = np.array([np.mean(data[s]["Sequential"]) for s in sizes])
seq_stds   = np.array([np.std(data[s]["Sequential"], ddof=1) for s in sizes])
coloc_means = np.array([np.mean(data[s]["Colocated"]) for s in sizes])
coloc_stds  = np.array([np.std(data[s]["Colocated"], ddof=1) for s in sizes])

# Slowdown: colocated vs ideal (sequential / 2)
slowdown = coloc_means / (seq_means / 2)

# ── Colors ────────────────────────────────────────────────────────────────────

c_seq   = "#7A9CBE"
c_coloc = "#6AB187"

bar_width = 0.32
x = np.arange(n)

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=150)

ax.bar(x - bar_width / 2, seq_means, bar_width,
       yerr=seq_stds, capsize=3,
       color=c_seq, edgecolor="black", linewidth=0.5,
       error_kw=dict(elinewidth=0.8, capthick=0.8, color="black"),
       label="Sequential", zorder=3)

ax.bar(x + bar_width / 2, coloc_means, bar_width,
       yerr=coloc_stds, capsize=3,
       color=c_coloc, edgecolor="black", linewidth=0.5,
       error_kw=dict(elinewidth=0.8, capthick=0.8, color="black"),
       label="Colocated", zorder=3)

for i in range(n):
    ax.text(x[i] + bar_width / 2, coloc_means[i] + coloc_stds[i] + 20,
            f"{slowdown[i]:.2f}\u00d7", ha="center", va="bottom",
            fontsize=7.5, fontweight="bold", color="#D4726A")

ax.set_xlabel("Copy Size per Thread Block (KB)", fontsize=10)
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
fig.savefig("intra_sm/l1_cache/l1_cache_results.png", bbox_inches="tight", dpi=300)
fig.savefig("intra_sm/l1_cache/l1_cache_results.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved: l1_cache_results.png / .pdf")
