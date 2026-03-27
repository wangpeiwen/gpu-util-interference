#!/usr/bin/env python3
"""
Thread Block Scheduler Interference — Horizontal Bar Chart
X-axis = kernel completion time (ms), Y-axis = experimental conditions.
Two groups separated by launch config, three bars each (Alone / Sequential / Colocated).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches

# ── Font config ───────────────────────────────────────────────────────────────

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STFangsong"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})

# ── Raw experimental data (10 runs each) ──────────────────────────────────────

data = {
    "(80, 1024)": {
        "Alone":      [103.097, 102.428, 102.416, 102.423, 102.410,
                       102.413, 102.414, 102.430, 102.409, 102.406],
        "Sequential": [205.589, 204.811, 204.827, 204.809, 204.841,
                       204.808, 204.808, 204.807, 204.807, 204.807],
        "Colocated":  [102.423, 102.418, 102.413, 102.413, 102.416,
                       102.413, 102.438, 102.418, 102.433, 102.432],
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

conditions = ["Colocated", "Sequential", "Alone"]  # bottom-to-top order
configs = ["(160, 1024)", "(80, 1024)"]             # bottom group first

# ── Compute mean ± std ────────────────────────────────────────────────────────

means = {cfg: {c: np.mean(data[cfg][c]) for c in conditions} for cfg in configs}
stds  = {cfg: {c: np.std(data[cfg][c], ddof=1) for c in conditions} for cfg in configs}

# ── Visual parameters ─────────────────────────────────────────────────────────

colors = {
    "Alone":      "#7A9CBE",   # steel blue
    "Sequential": "#E8927C",   # salmon
    "Colocated":  "#6AB187",   # sage green
}
hatches = {
    "Alone":      "",
    "Sequential": "//",
    "Colocated":  "..",
}

bar_height = 0.22
group_gap = 0.45  # vertical gap between the two config groups

# ── Build y positions ─────────────────────────────────────────────────────────
# Each config group has 3 bars; groups are separated by group_gap.

y_positions = {}  # cfg -> list of y coords (one per condition)
y_base = 0
for cfg in configs:
    positions = []
    for i in range(len(conditions)):
        positions.append(y_base + i * (bar_height + 0.04))
    y_positions[cfg] = positions
    y_base = positions[-1] + bar_height + group_gap

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(5.5, 3.8), dpi=150)

for cfg in configs:
    for i, cond in enumerate(conditions):
        m = means[cfg][cond]
        s = stds[cfg][cond]
        y = y_positions[cfg][i]
        ax.barh(
            y, m, bar_height,
            xerr=s, capsize=3,
            color=colors[cond], hatch=hatches[cond],
            edgecolor="black", linewidth=0.5,
            error_kw=dict(elinewidth=0.8, capthick=0.8, color="black"),
            zorder=3,
        )
        # Time label at bar end
        ax.text(m + s + 1.5, y, f"{m:.1f} ms",
                va="center", ha="left", fontsize=7.5, color="#333333")

# ── Colocated / Alone ratio annotation (inside the Colocated bar) ─────────────

for cfg in configs:
    ratio = means[cfg]["Colocated"] / means[cfg]["Alone"]
    coloc_m = means[cfg]["Colocated"]
    coloc_y = y_positions[cfg][0]  # Colocated bar
    ax.text(coloc_m - 3, coloc_y, f"{ratio:.2f}×",
            va="center", ha="right", fontsize=7.5, fontweight="bold",
            color="white")

# ── Y-axis: group labels ─────────────────────────────────────────────────────

# Two-level Y labels: "config\ncondition" per bar
all_yticks = []
all_ylabels = []
for cfg in configs:
    for i, cond in enumerate(conditions):
        all_yticks.append(y_positions[cfg][i])
        # Only show config name on the middle bar of each group
        if i == 1:
            all_ylabels.append(f"{cfg}\n{cond}")
        else:
            all_ylabels.append(cond)

ax.set_yticks(all_yticks)
ax.set_yticklabels(all_ylabels, fontsize=8, linespacing=1.3)

# ── X-axis ────────────────────────────────────────────────────────────────────

ax.set_xlabel("Kernel Completion Time (ms)", fontsize=10)
ax.set_xlim(0, 235)
ax.xaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(axis="x", labelsize=9)

# Remove spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ── Legend ─────────────────────────────────────────────────────────────────────

handles = [mpatches.Patch(facecolor=colors[c], hatch=hatches[c],
                          edgecolor="black", linewidth=0.5, label=c)
           for c in ["Alone", "Sequential", "Colocated"]]
ax.legend(handles=handles, fontsize=8, frameon=True, edgecolor="#cccccc",
          loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.12))

# ── Save ──────────────────────────────────────────────────────────────────────

fig.subplots_adjust(left=0.18)
fig.savefig("inter_sm/thread_block_scheduler/tb_scheduler_results.pdf",
            bbox_inches="tight")
fig.savefig("inter_sm/thread_block_scheduler/tb_scheduler_results.png",
            bbox_inches="tight", dpi=300)
plt.close(fig)
print("Saved: tb_scheduler_results.pdf / .png")
