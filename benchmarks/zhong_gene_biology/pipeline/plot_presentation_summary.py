#!/usr/bin/env python3
"""Create the presentation summary figure from frozen Zhong result tables."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "benchmarks" / "zhong_gene_biology"
RESULTS = BENCH / "results"
FIGURES = BENCH / "figures"

COLORS = ["#8A8F98", "#2878B5", "#D95319"]


def main() -> None:
    single = pd.read_csv(RESULTS / "single_gene_zhong_matched_summary.csv")
    pair = pd.read_csv(RESULTS / "final_pair_best_conditions.csv")

    def single_values(task: str) -> np.ndarray:
        values = (
            single[single["Task"] == task]
            .set_index("Representation")
            .loc[["E0", "E1", "E2"], "Mean_AUROC"]
            .to_numpy(float)
        )
        return values

    go = single_values("GO")
    omim = single_values("OMIM")
    pair_lookup = pair.set_index("Task")["AUROC"]
    pair_values = pair_lookup.loc[["Synthetic lethal", "TF-target", "Negative genetic"]].to_numpy(float)

    # Guard the frozen plotted values against accidental source-table drift.
    assert np.allclose(go, [0.5177, 0.7722, 0.7811], atol=5e-5)
    assert np.allclose(omim, [0.4891, 0.7354, 0.7596], atol=5e-5)
    assert np.allclose(pair_values, [0.707, 0.670, 0.580], atol=5e-4)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.titlesize": 20,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.4), sharey=True)
    panels = [
        (
            "Gene Function (GO)",
            "Mean AUROC across 56 GO Biological Process terms",
            ["Static", "Transformer", "GTEx Context"],
            go,
            COLORS,
        ),
        (
            "Disease Association (OMIM)",
            "Mean AUROC across 103 OMIM terms",
            ["Static", "Transformer", "GTEx Context"],
            omim,
            COLORS,
        ),
        (
            "Gene–Gene Relationships",
            "Highest observed condition per pair task",
            ["Synthetic lethal", "TF → target", "Negative genetic"],
            pair_values,
            ["#4472C4", "#5B9BD5", "#9DC3E6"],
        ),
    ]

    for i, (ax, (title, subtitle, labels, values, colors)) in enumerate(zip(axes, panels)):
        x = np.arange(len(labels))
        bars = ax.bar(x, values, width=0.64, color=colors, edgecolor="none")
        ax.set_ylim(0, 1)
        ax.set_xticks(x, labels)
        ax.tick_params(axis="x", rotation=0, pad=9)
        ax.set_title(title, fontweight="bold", pad=31)
        ax.text(
            0.5,
            1.025,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=12.5,
            color="#4A4A4A",
        )
        precision = 4 if i < 2 else 3
        ax.bar_label(
            bars,
            labels=[f"{v:.{precision}f}" for v in values],
            padding=5,
            fontsize=15,
            fontweight="bold",
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines["left"].set_color("#777777")
        ax.spines["bottom"].set_color("#777777")
        ax.tick_params(colors="#333333")
        ax.grid(False)
        if i == 0:
            ax.set_ylabel("AUROC")
        if i == 2:
            ax.set_xticklabels(labels, fontsize=12)

    fig.text(
        0.5,
        0.025,
        "Pair-task bars show best observed AUROC across evaluated representation/operation conditions.",
        ha="center",
        va="bottom",
        fontsize=11.5,
        color="#4A4A4A",
    )
    fig.subplots_adjust(left=0.06, right=0.99, top=0.82, bottom=0.18, wspace=0.18)

    FIGURES.mkdir(parents=True, exist_ok=True)
    png = FIGURES / "presentation_zhong_benchmark_summary.png"
    pdf = FIGURES / "presentation_zhong_benchmark_summary.pdf"
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    print(f"wrote {png}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
