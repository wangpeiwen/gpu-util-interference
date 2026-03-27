#!/usr/bin/env python3
"""
Pipeline Interference — Grouped bar chart.
X = ILP degree, two bars: Sequential vs Colocated.
Slowdown = Colocated / (Sequential / 2).
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
    1: {
        "Sequential": [264.395, 241.988, 237.872, 237.878, 237.973,
                       237.910, 237.902, 237.880, 237.885, 237.885],
        "Colocated":  [124.361, 124.372, 124.379, 124.379, 124.383,
                       124.379, 124.398, 124.385, 124.382, 124.385],
    },
    2: {
        "Sequential": [247.300, 246.936, 246.930, 246.923, 246.938,
                       246.905, 246.907, 246.905, 246.937, 246.930],
        "Colocated":  [234.180, 234.201, 234.182, 234.205, 234.186,
                       234.199, 234.174, 234.175, 234.174, 234.172],
    },
    3: {
        "Sequential": [359.413, 358.640, 358.634, 358.662, 358.633,
                       358.642, 358.660, 358.644, 358.664, 358.633],
        "Colocated":  [346.016, 346.028, 346.052, 346.011, 346.012,
                       346.007, 346.017, 347.451, 346.017, 346.007],
    },
    4: {
        "Sequential": [474.385, 473.947, 473.973, 473.949, 473.970,
                       473.972, 473.949, 473.947, 473.947, 473.947],
        "Colocated":  [461.329, 461.361, 461.341, 461.355, 461.351,
                       461.352, 461.351, 461.832, 462.165, 461.325],
    },
}

ilps = sorted(data.keys())
n = len(ilps)

seq_means   = np.array([np.mean(data[i]["Sequential"]) for i in ilps])
seq_stds    = np.array([np.std(data[i]["Sequential"], ddof=1) for i in ilps])
coloc_means = np.array([np.mean(data[i]["Colocated"]) for i in ilps])
coloc_stds  = np.array([np.std(data[i]["Colocated"], ddof=1) for i in ilps])

slowdown = coloc_means / (seq_means / 2)

# ── Colors ────────────────────────────────────────────────────────────────────

c_seq   = "#7A9CBE"
c_coloc = "#6AB187"

bar_width = 0.30
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
    ax.text(x[i] + bar_width / 2, coloc_means[i] + coloc_stds[i] + 5,
            f"{slowdown[i]:.2f}\u00d7", ha="center", va="bottom",
            fontsize=7.5, fontweight="bold", color="#D4726A")

ax.set_xlabel("ILP Degree", fontsize=10)
ax.set_ylabel("Kernel Completion Time (ms)", fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels([str(i) for i in ilps], fontsize=9)
ax.tick_params(axis="y", labelsize=9)
ax.yaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(fontsize=8.5, frameon=True, edgecolor="#cccccc",
          loc="upper left")

fig.tight_layout()
fig.savefig("intra_sm/pipelines/pipelines_results.png", bbox_inches="tight", dpi=300)
fig.savefig("intra_sm/pipelines/pipelines_results.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved: pipelines_results.png / .pdf")
