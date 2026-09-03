#!/usr/bin/env python3
"""Create publication figures for the completed Task 1A geometry benchmark.

This is a read-only plotting utility: all values are loaded from
``results/task1a_geometry_summary.csv``.  The kNN comparison uses k=1
consistently rather than selecting k using target-cohort performance.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"
FIGURES = RESULTS / "figures" / "task1a_geometry"
SUMMARY = RESULTS / "task1a_geometry_summary.csv"

METHODS = [
    ("Raw expression +\ncentroid cosine", "raw_expression", "centroid_cosine"),
    ("Joint PCA +\ncentroid cosine", "joint_pca", "centroid_cosine"),
    ("BridgeRNA +\ncentroid cosine", "bridgerna", "centroid_cosine"),
    ("BridgeRNA +\ncentered cosine", "bridgerna", "centroid_cosine_centered"),
    ("BridgeRNA +\nkNN (k=1)", "bridgerna", "knn_cosine_k1"),
    ("BridgeRNA +\nlinear probe", "bridgerna", "linear_softmax_probe"),
    ("BridgeRNA +\nshallow MLP", "bridgerna", "shallow_mlp_probe"),
]

DIRECTIONS = [("human_to_mouse", "Human→Mouse"), ("mouse_to_human", "Mouse→Human")]
COHORTS = {
    "complete_11_tissue": ("11-tissue fully unseen benchmark", 9.1),
    "replicated_5_tissue": ("5-tissue replicated benchmark", 20.0),
}
COLORS = {"Human→Mouse": "#2878B5", "Mouse→Human": "#E07A1F"}


def load_results() -> pd.DataFrame:
    if not SUMMARY.exists():
        raise FileNotFoundError(f"Missing completed Task 1A results: {SUMMARY}")
    frame = pd.read_csv(SUMMARY)
    required = {"cohort", "representation", "readout", "direction", "top1_accuracy"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Task 1A summary lacks columns: {sorted(missing)}")
    keys = frame[["cohort", "representation", "readout", "direction"]]
    if keys.duplicated().any():
        raise ValueError("Task 1A summary contains duplicate result keys")
    return frame


def accuracy(frame: pd.DataFrame, cohort: str, representation: str, readout: str, direction: str) -> float:
    selected = frame[
        (frame["cohort"] == cohort)
        & (frame["representation"] == representation)
        & (frame["readout"] == readout)
        & (frame["direction"] == direction)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one value for {cohort}/{representation}/{readout}/{direction}; found {len(selected)}"
        )
    return 100.0 * float(selected.iloc[0]["top1_accuracy"])


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)


def save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.png", dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_cohort(frame: pd.DataFrame, cohort: str, title: str, chance: float) -> None:
    labels = [item[0] for item in METHODS]
    values = {
        direction_label: [
            accuracy(frame, cohort, representation, readout, direction)
            for _, representation, readout in METHODS
        ]
        for direction, direction_label in DIRECTIONS
    }
    y = np.arange(len(labels))
    height = 0.34
    fig, ax = plt.subplots(figsize=(9.2, 5.7))
    for offset, (_, direction_label) in zip((-height / 2, height / 2), DIRECTIONS):
        bars = ax.barh(
            y + offset,
            values[direction_label],
            height,
            color=COLORS[direction_label],
            label=direction_label,
        )
        ax.bar_label(bars, labels=[f"{v:.1f}%" for v in values[direction_label]], padding=3, fontsize=8.5)
    ax.axvline(chance, color="#333333", linestyle="--", linewidth=1.2, label=f"Chance ({chance:.1f}%)")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Tissue prediction accuracy (%)", fontsize=10)
    ax.set_title(title, fontsize=13, weight="bold", loc="left")
    ax.legend(frameon=False, ncol=3, loc="lower right", fontsize=8.5)
    style_axis(ax)
    fig.tight_layout()
    stem = "task1a_methods_11_tissue" if cohort == "complete_11_tissue" else "task1a_methods_5_tissue"
    save(fig, stem)


def plot_improvements(frame: pd.DataFrame) -> None:
    alternatives = [
        ("Centered cosine", "centroid_cosine_centered"),
        ("kNN (k=1)", "knn_cosine_k1"),
        ("Linear probe", "linear_softmax_probe"),
        ("Shallow MLP", "shallow_mlp_probe"),
    ]
    rows = []
    for cohort, (cohort_title, _) in COHORTS.items():
        for direction, direction_label in DIRECTIONS:
            baseline = accuracy(frame, cohort, "bridgerna", "centroid_cosine", direction)
            for label, readout in alternatives:
                rows.append(
                    {
                        "cohort": cohort_title.replace(" benchmark", ""),
                        "direction": direction_label,
                        "method": label,
                        "delta": accuracy(frame, cohort, "bridgerna", readout, direction) - baseline,
                    }
                )
    changes = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharex=True)
    y = np.arange(len(alternatives))
    height = 0.34
    for ax, (cohort, (title, _)) in zip(axes, COHORTS.items()):
        subset = changes[changes["cohort"] == title.replace(" benchmark", "")]
        for offset, (_, direction_label) in zip((-height / 2, height / 2), DIRECTIONS):
            vals = [
                float(subset[(subset.direction == direction_label) & (subset.method == label)].delta.iloc[0])
                for label, _ in alternatives
            ]
            bars = ax.barh(y + offset, vals, height, color=COLORS[direction_label], label=direction_label)
            ax.bar_label(
                bars,
                labels=[f"{value:+.1f}" for value in vals],
                padding=3,
                fontsize=8.5,
            )
        ax.axvline(0, color="#333333", linewidth=1.0)
        ax.set_yticks(y, [label for label, _ in alternatives])
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11, weight="bold")
        ax.set_xlabel("Accuracy change from BridgeRNA centroid cosine (percentage points)", fontsize=9)
        style_axis(ax)
    axes[1].legend(frameon=False, loc="lower right", fontsize=8.5)
    fig.suptitle("BridgeRNA readout improvement", fontsize=13, weight="bold", x=0.07, ha="left")
    fig.tight_layout()
    save(fig, "task1a_bridgerna_readout_improvement")


def main() -> None:
    frame = load_results()
    for cohort, (title, chance) in COHORTS.items():
        plot_cohort(frame, cohort, title, chance)
    plot_improvements(frame)
    print(f"Saved publication figures to {FIGURES}")


if __name__ == "__main__":
    main()
