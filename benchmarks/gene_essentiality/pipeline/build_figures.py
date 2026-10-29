#!/usr/bin/env python3
"""Build final figures from saved per-gene metrics only."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BENCH = Path(__file__).resolve().parent.parent
ANALYSIS = BENCH / "results/analysis"
FIGURES = BENCH / "results/figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def main() -> None:
    metrics = pd.read_csv(ANALYSIS / "per_gene_metrics.csv")
    order = ["target_expression", "pca_context", "bulkformer_contextual",
             "bridge_contextual", "bridge_plus_pca"]
    labels = ["Target expression", "PCA context", "BulkFormer", "Bridge", "Bridge + PCA"]
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    arrays = [metrics.loc[metrics.representation.eq(name), "pcc"].dropna().to_numpy() for name in order]
    parts = axis.violinplot(arrays, showmeans=False, showmedians=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("#4C78A8"); body.set_alpha(.7)
    axis.axhline(0, color="black", linewidth=.8, linestyle="--")
    axis.set_xticks(range(1, len(labels) + 1), labels, rotation=20, ha="right")
    axis.set_ylabel("Per-gene held-out PCC")
    axis.set_title("Common 14,415-gene evaluation universe")
    figure.tight_layout(); figure.savefig(FIGURES / "per_gene_pcc_distributions.png", dpi=300)
    figure.savefig(FIGURES / "per_gene_pcc_distributions.pdf"); plt.close(figure)

    wide = metrics.pivot(index="gene_id", columns="representation", values="pcc")
    for comparator, label, filename in [
        ("pca_context", "PCA context", "bridge_vs_pca_pcc"),
        ("target_expression", "Target expression", "bridge_vs_target_expression_pcc"),
    ]:
        valid = wide[[comparator, "bridge_contextual"]].dropna()
        figure, axis = plt.subplots(figsize=(6.5, 6.5))
        axis.scatter(valid[comparator], valid.bridge_contextual, s=5, alpha=.25, rasterized=True)
        limits = [float(valid.min().min() - .02), float(valid.max().max() + .02)]
        axis.plot(limits, limits, color="black", linewidth=1, linestyle="--")
        axis.set(xlabel=f"{label} per-gene PCC", ylabel="Bridge per-gene PCC",
                 xlim=limits, ylim=limits, title=f"Matched genes (n={len(valid):,})")
        figure.tight_layout(); figure.savefig(FIGURES / f"{filename}.png", dpi=300)
        figure.savefig(FIGURES / f"{filename}.pdf"); plt.close(figure)


if __name__ == "__main__":
    main()
