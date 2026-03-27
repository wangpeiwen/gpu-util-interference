#!/usr/bin/env python3
"""
IPC (Warp Scheduler) Interference — Horizontal grouped bar chart.
Four conditions: Copy Alone, Compute Alone, Sequential, Colocated.
Shows that colocated time ≈ copy alone time despite complementary profiles,
demonstrating warp scheduler interference is minimal at ILP 4.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STFangsong"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})

# ── Data (10 runs each) ──────────────────────────────────────────────────────

data = {
    "Copy Alone": [333.669, 326.812, 326.367, 326.102, 325.524,
                   326.677, 326.517, 325.691, 327.792, 324.719],
    "Compute Alone\n(ILP 4)": [125.708, 125.257, 125.251, 125.250, 125.249,
                                125.251, 125.250, 125.278, 125.253, 125.250],
    "Sequential":  [457.754, 451.129, 450.601, 451.470, 451.206,
                    453.269, 450.772, 451.869, 453.059, 451.262],
    "Colocated":   [327.017, 329.105, 325.861, 327.814, 326.095,
                    326.208, 328.088, 327.576, 326.642, 327.913],
}

conditions = list(data.keys())
n = len(conditions)

means = np.array([np.mean(data[c]) for c in conditions])
stds  = np.array([np.std(data[c], ddof=1) for c in conditions])

# ── Colors ────────────────────────────────────────────────────────────────────

colors = ["#7A9CBE", "#E8927C", "#B8B0D0", "#6AB187"]

bar_height = 0.5
y = np.arange(n)

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6.0, 3.2), dpi=150)

bars = ax.barh(y, means, bar_height,
               xerr=stds, capsize=3,
               color=colors, edgecolor="black", linewidth=0.5,
               error_kw=dict(elinewidth=0.8, capthick=0.8, color="black"),
               zorder=3)

# Value labels at bar end
for i in range(n):
    ax.text(means[i] + stds[i] + 2, y[i], f"{means[i]:.1f} ms",
            va="center", ha="left", fontsize=7.5, color="#333333")

ax.set_xlabel("Kernel Completion Time (ms)", fontsize=10)
ax.set_yticks(y)
ax.set_yticklabels(conditions, fontsize=8.5)
ax.tick_params(axis="x", labelsize=9)
ax.xaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlim(0, 510)

fig.tight_layout()
fig.savefig("intra_sm/ipc/ipc_results.png", bbox_inches="tight", dpi=300)
fig.savefig("intra_sm/ipc/ipc_results.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved: ipc_results.png / .pdf")
