#!/usr/bin/env python3
"""Render the benchmark's key result directly from frozen result tables."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
EVAL = HERE / "results/evaluation"
OUT = HERE / "results/figures/bridge_muscle_convergence_summary.png"

effects = pd.read_csv(EVAL / "primary_effects.csv").set_index("condition")
pretraining = pd.read_csv(EVAL / "pretraining_status_summary.csv").set_index("pretraining_pair")
sensitivity = pd.read_csv(EVAL / "GSE211204_unseen_subject_convergence.csv").set_index("method")
coverage = pd.read_csv(EVAL / "primary_ranking_coverage.csv")

conditions = ["Radiation", "Bone loss", "Muscle atrophy"]
bridge = effects.loc[conditions, "bridge_mean_pairwise_top25_jaccard"].to_numpy()
de = effects.loc[conditions, "de_mean_pairwise_top25_jaccard"].to_numpy()
overall = effects.loc["Overall"]
strict_delta = pretraining.loc["strict_unseen_pair", "bridge_minus_de"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titleweight": "bold",
    "axes.labelcolor": "#10264a",
    "text.color": "#10264a",
    "xtick.color": "#10264a",
    "ytick.color": "#10264a",
})

fig = plt.figure(figsize=(15, 8.6), facecolor="white")
grid = fig.add_gridspec(2, 3, height_ratios=[5.2, 1.35], hspace=.36, wspace=.28)
axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
bridge_color, de_color = "#159c98", "#a8a8a8"

for index, (ax, condition, bv, dv) in enumerate(zip(axes, conditions, bridge, de)):
    if condition == "Muscle atrophy":
        ax.set_facecolor("#effafa")
        for spine in ax.spines.values():
            spine.set_color(bridge_color); spine.set_linewidth(2)
    bars = ax.bar([0, 1], [bv, dv], width=.58, color=[bridge_color, de_color], edgecolor="none")
    ax.set_title(condition, fontsize=18, pad=14)
    ax.set_xticks([0, 1], ["Bridge", "DE"], fontsize=14)
    ax.set_ylim(0, .8); ax.set_yticks(np.arange(0, .81, .2))
    ax.grid(axis="y", color="#dce3eb", linewidth=.8, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#10264a"); ax.spines["bottom"].set_color("#10264a")
    if index == 0:
        ax.set_ylabel("Mean pairwise drug-list Jaccard", fontsize=13)
    for bar, value in zip(bars, [bv, dv]):
        ax.text(bar.get_x() + bar.get_width()/2, value + .022, f"{value:.3f}", ha="center", va="bottom", fontsize=15, fontweight="bold")

axes[2].annotate(
    "Strict-unseen GSE211204 sensitivity\nBridge 0.714 vs DE 0.172",
    xy=(0, bridge[2]), xytext=(.22, .61), fontsize=11.5, fontweight="bold",
    arrowprops={"arrowstyle": "-|>", "color": bridge_color, "lw": 1.8},
    bbox={"boxstyle": "round,pad=.45", "fc": "white", "ec": bridge_color, "lw": 1.5},
)

summary_ax = fig.add_subplot(grid[1, :]); summary_ax.axis("off")
summary_ax.add_patch(plt.Rectangle((0, .31), 1, .62, transform=summary_ax.transAxes, color="#edf3f8", zorder=0))
summary_ax.text(.03, .62, f"Overall Δ = {overall.bridge_minus_de:+.3f}", fontsize=16, fontweight="bold", va="center")
summary_ax.text(.29, .62, f"exact permutation p = {overall.exact_p:.4f}", fontsize=15, va="center")
summary_ax.text(.61, .62, f"Strict-unseen pairs Δ = {strict_delta:+.3f}", fontsize=16, fontweight="bold", va="center")
minimum = int(coverage.nonzero_overlap_drugs.min())
summary_ax.text(.5, .08, f"Nonzero-overlap lists were sparse (minimum {minimum} drug); nominal top-25 uses all available candidates when fewer than 25 exist.", ha="center", fontsize=10.5, color="#45586f")

fig.suptitle("Bridge’s strongest signal is cross-study muscle convergence", fontsize=25, fontweight="bold", y=.985)
fig.text(.5, .012, "ChEMBL mechanism-target enrichment convergence — not therapeutic efficacy or reversal", ha="center", fontsize=12, fontweight="bold")
fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
print(OUT)
