#!/usr/bin/env python3
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from common import RESULTS


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    summary = pd.read_csv(RESULTS / "primary_results.csv").sort_values("mean_pcc")
    labels = {"raw_expression": "Raw expression", "pca_128": "PCA-128",
              "pca_512": "PCA-512", "bulkformer_50m": "BulkFormer-50M",
              "bulkformer_147m": "BulkFormer-147M", "bridge_45.6m": "Bridge-45.6M",
              "pca128_bridge": "PCA-128 + Bridge"}
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([labels[x] for x in summary.representation], summary.mean_pcc, color="#4472C4")
    ax.axvline(.373, color="#C44E52", linestyle="--", label="Published BulkFormer 0.373\n(not protocol-equivalent)")
    ax.set(xlabel="Mean per-screen PCC", ylabel="")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / "primary_mean_pcc.png", dpi=200)
    fig.savefig(RESULTS / "primary_mean_pcc.pdf")
    plt.close(fig)

    metrics = pd.read_csv(RESULTS / "per_drug_metrics.csv")
    keep = ["pca_512", "bulkformer_50m", "bulkformer_147m", "bridge_45.6m"]
    plot = metrics.loc[metrics.representation.isin(keep)].copy()
    fig, ax = plt.subplots(figsize=(11, 6))
    values = [plot.loc[plot.representation.eq(name), "pcc"].to_numpy() for name in keep]
    ax.boxplot(values, vert=False, tick_labels=[labels[x] for x in keep], showfliers=False)
    ax.axvline(0, color="black", linewidth=1)
    ax.set(xlabel="Per-screen held-out-cell-line PCC", ylabel="")
    fig.tight_layout()
    fig.savefig(RESULTS / "per_drug_pcc_distribution.png", dpi=200)
    fig.savefig(RESULTS / "per_drug_pcc_distribution.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
