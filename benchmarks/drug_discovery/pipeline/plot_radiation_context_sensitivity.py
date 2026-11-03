#!/usr/bin/env python3
"""Publication figure for the frozen radiation-context sensitivity analysis."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "results/radiation_context_sensitivity"


def main():
    pairs = pd.read_csv(OUT / "contrast_pair_similarity.csv")
    fibro = pd.read_csv(OUT / "fibroblast_comparison.csv")
    core = pd.read_csv(OUT / "radiation_core_genes.csv")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    bridge_color, de_color = "#2457C5", "#D35A3A"
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), constrained_layout=True)

    ax = axes[0]
    rng = np.random.default_rng(297090)
    for method, color, offset in [("Bridge top-500", bridge_color, -0.07), ("DE top-500", de_color, 0.07)]:
        subset = pairs[pairs.analysis.eq(method)]
        for distance, group in subset.groupby("context_distance"):
            x = distance + offset + rng.uniform(-0.025, 0.025, len(group))
            ax.scatter(x, group.jaccard, s=22, color=color, alpha=.48, edgecolor="none")
            ax.plot([distance + offset - .055, distance + offset + .055],
                    [group.jaccard.mean()] * 2, color=color, lw=3)
    ax.set_xticks([1, 2, 3, 4], ["Same\nsystem", "Similar\ncell", "Different cell,\nsimilar exposure",
                                  "Different cell\n+ exposure"])
    ax.set_ylabel("Top-500 gene Jaccard")
    ax.set_title("A  Agreement decays with context distance", loc="left", weight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    select = pairs[(pairs.analysis.isin(["Bridge top-500", "DE top-500"])) &
                   (((pairs.contrast_1.eq("GSE297090_gamma")) & (pairs.contrast_2.eq("GSE297090_proton"))) |
                    ((pairs.contrast_1.eq("GSE297560_gamma")) & (pairs.contrast_2.eq("GSE297560_proton"))) |
                    ((pairs.contrast_1.eq("GSE297560_iron")) & (pairs.contrast_2.eq("GSE297560_silicon"))))].copy()
    select["label"] = np.where(select.contrast_1.str.contains("297090"), "Organoid\nγ vs proton",
                       np.where(select.contrast_1.str.endswith("gamma"), "Fibroblast\nγ vs proton", "Fibroblast\nFe vs Si"))
    pivot = select.pivot(index="label", columns="analysis", values="jaccard").loc[
        ["Organoid\nγ vs proton", "Fibroblast\nγ vs proton", "Fibroblast\nFe vs Si"]]
    x = np.arange(len(pivot)); width=.35
    ax.bar(x-width/2, pivot["Bridge top-500"], width, color=bridge_color, label="Bridge")
    ax.bar(x+width/2, pivot["DE top-500"], width, color=de_color, label="DE")
    ax.set_xticks(x, pivot.index)
    ax.set_ylabel("Top-500 gene Jaccard")
    ax.set_title("B  Within-system replication", loc="left", weight="bold")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    fp = fibro.pivot(index="osdr_modality", columns="method", values="jaccard").loc[["gamma", "proton", "iron", "silicon"]]
    x = np.arange(len(fp))
    ax.bar(x-width/2, fp["Bridge"], width, color=bridge_color, label="Bridge NHDF block")
    ax.bar(x+width/2, fp["DE"], width, color=de_color, label="DE pooled 6-cell†")
    ax.set_xticks(x, ["γ", "proton", "Fe", "Si"])
    ax.set_ylabel("Terrestrial–OSDR Jaccard")
    ax.set_title("C  Fibroblast-matched sensitivity", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=8)
    ax.text(.02, .98, "†not a cell-specific DE comparator", transform=ax.transAxes, va="top", fontsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Radiation-response recurrence is context dependent", fontsize=13, weight="bold")
    for suffix in ["png", "svg"]:
        fig.savefig(OUT / f"radiation_context_sensitivity.{suffix}", dpi=300 if suffix == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
