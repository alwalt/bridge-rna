#!/usr/bin/env python3
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import RESULTS, WORK


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    summary = pd.read_csv(RESULTS / "primary_results.csv").sort_values("mean_pcc")
    labels = {"raw_expression": "Raw expression",
              **{f"pca_{count}": f"PCA-{count}" for count in range(1, 9)},
              "pca_16": "PCA-16", "pca_32": "PCA-32", "pca_64": "PCA-64",
              "pca_128": "PCA-128", "pca_256": "PCA-256", "pca_512": "PCA-512",
              "bulkformer_50m": "BulkFormer-50M",
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
    keep = ["pca_8", "pca_512", "bulkformer_50m", "bulkformer_147m", "bridge_45.6m"]
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

    pca = summary.loc[summary.representation.str.fullmatch(r"pca_\d+")].copy()
    pca["components"] = pca.representation.str.extract(r"(\d+)").astype(int)
    pca = pca.sort_values("components")
    pca[["components", "mean_pcc", "median_pcc", "mean_scc", "rmse", "evaluable_drugs"]].to_csv(
        RESULTS / "pca_component_sensitivity.csv", index=False)
    bridge_pcc = float(summary.loc[summary.representation.eq("bridge_45.6m"), "mean_pcc"].iloc[0])
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(pca.components, pca.mean_pcc, marker="o", linewidth=2.2, color="#4472C4",
            label="PCA + ridge")
    ax.axhline(bridge_pcc, color="#D62728", linestyle="--",
               label=f"Bridge-45.6M ({bridge_pcc:.3f})")
    ax.set(xscale="log", xlabel="PCA components (log scale)", ylabel="Mean per-screen OOF PCC",
           xticks=pca.components.to_list())
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(RESULTS / "pca_component_sensitivity.png", dpi=200)
    fig.savefig(RESULTS / "pca_component_sensitivity.pdf")
    plt.close(fig)

    # A pooled raw-IC50 scatter would be dominated by between-screen offsets.
    # Standardize truth and predictions within each screen before pooling so
    # the figure summarizes within-screen held-out-cell-line discrimination.
    from evaluate import matrices

    _, screens, targets = matrices()
    predictions = np.load(WORK / "bridge_45.6m_oof_predictions.npy")
    pooled_truth, pooled_prediction = [], []
    for index, _ in enumerate(screens):
        keep = np.isfinite(targets[:, index]) & np.isfinite(predictions[:, index])
        truth = targets[keep, index]
        prediction = predictions[keep, index]
        truth_sd, prediction_sd = truth.std(ddof=1), prediction.std(ddof=1)
        if truth_sd <= 0 or prediction_sd <= 0:
            continue
        pooled_truth.append((truth - truth.mean()) / truth_sd)
        pooled_prediction.append((prediction - prediction.mean()) / prediction_sd)
    pooled_truth = np.concatenate(pooled_truth)
    pooled_prediction = np.concatenate(pooled_prediction)
    slope, intercept = np.polyfit(pooled_truth, pooled_prediction, 1)
    bridge_summary = summary.loc[summary.representation.eq("bridge_45.6m")].iloc[0]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(pooled_truth, pooled_prediction, s=5, alpha=.035,
               color="#4C78A8", edgecolors="none", rasterized=True)
    limits = (-4, 4)
    xline = np.linspace(*limits, 200)
    ax.plot(xline, intercept + slope * xline, color="#D62728", linewidth=2.5,
            label=f"Linear fit (slope = {slope:.3f})")
    ax.set(xlim=limits, ylim=limits,
           xlabel="Measured IC50 (within-screen z-score)",
           ylabel="Bridge-predicted IC50 (within-screen z-score)",
           title="Bridge-45.6M held-out-cell-line predictions")
    ax.text(.03, .97,
            f"355 screens · {len(pooled_truth):,} observations\n"
            f"Mean per-screen PCC = {bridge_summary.mean_pcc:.3f}\n"
            f"Median per-screen PCC = {bridge_summary.median_pcc:.3f}",
            transform=ax.transAxes, va="top", ha="left",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": .9})
    ax.legend(loc="lower right", fontsize=11, frameon=True)
    fig.tight_layout()
    fig.savefig(RESULTS / "bridge_oof_ic50_regression.png", dpi=200)
    fig.savefig(RESULTS / "bridge_oof_ic50_regression.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
