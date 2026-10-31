#!/usr/bin/env python3
"""Create publication-style plots exclusively from saved benchmark results."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

COLORS = {
    "Cosine": "#4C78A8",
    "CSLS(k=5)": "#72B7B2",
    "CSLS(k=10)": "#54A24B",
    "CSLS(k=50)": "#9D755D",
    "LocalScaling(k=10)": "#F58518",
    "NICDM(k=10)": "#E45756",
    "Empirical-null cosine (global)": "#B279A2",
}


def save(fig, stem):
    fig.tight_layout()
    fig.savefig(FIGURES / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def method_comparison():
    d = pd.read_csv(RESULTS / "ranked_method_comparison.csv")
    keep = ["Cosine", "CSLS(k=5)", "CSLS(k=10)", "LocalScaling(k=10)", "NICDM(k=10)"]
    d = d.set_index("method").loc[keep].reset_index()
    metrics = [("exact_r1", "Exact R@1"), ("exact_r10", "Exact R@10"),
               ("same_gse_r1", "Same-GSE R@1"), ("source_r1_known", "Source R@1")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), sharey=True)
    y = np.arange(len(d))
    for ax, (column, title) in zip(axes, metrics):
        values = d[column].to_numpy()
        ax.barh(y, values, color=[COLORS[m] for m in d.method])
        ax.axvline(d.loc[d.method.eq("Cosine"), column].iloc[0], color="black", ls="--", lw=1)
        ax.set(title=title, xlim=(0, 1), xlabel="Fraction")
        ax.grid(axis="x", alpha=.2)
        for i, value in enumerate(values):
            ax.text(value + .012, i, f"{value:.1%}", va="center", fontsize=8)
    axes[0].set_yticks(y, d.method); axes[0].invert_yaxis()
    for ax in axes[1:]: ax.tick_params(labelleft=False)
    fig.suptitle("Retrieval gains trade against biological-neighborhood agreement", y=1.03)
    save(fig, "ranked_method_comparison")


def density_strata():
    d = pd.read_csv(RESULTS / "density_stratified_metrics.csv")
    methods = ["Cosine", "CSLS(k=10)", "LocalScaling(k=10)", "NICDM(k=10)"]
    strata = ["sparse", "intermediate", "dense"]
    d = d[d.method.isin(methods)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)
    x = np.arange(len(strata)); width = .19
    for j, method in enumerate(methods):
        sub = d.set_index(["method", "density_stratum"]).loc[method].reindex(strata)
        for ax, metric, title in zip(axes, ["exact_r1", "exact_r10"], ["Exact R@1", "Exact R@10"]):
            ax.bar(x + (j - 1.5) * width, sub[metric], width, color=COLORS[method], label=method)
            ax.set(title=title, ylabel="Recall", xticks=x, xticklabels=strata, ylim=(0, 1))
            ax.grid(axis="y", alpha=.2)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Local corrections help most consistently in dense neighborhoods")
    save(fig, "density_stratified_retrieval")


def hubness():
    d = pd.read_csv(RESULTS / "hubness_summary.csv")
    methods = ["Cosine", "CSLS(k=10)", "LocalScaling(k=10)", "NICDM(k=10)"]
    d = d[d.method.isin(methods) & d.k.isin([1, 10, 100])]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    x = np.arange(3); width = .19
    for j, method in enumerate(methods):
        sub = d[d.method.eq(method)].set_index("k").reindex([1, 10, 100])
        axes[0].bar(x + (j - 1.5) * width, sub.max_nk, width, color=COLORS[method], label=method)
        axes[1].bar(x + (j - 1.5) * width, sub.top_10_hubs_assignment_fraction, width, color=COLORS[method])
    axes[0].set(title="Largest hub", ylabel="Maximum $N_k$", xticks=x, xticklabels=["k=1", "k=10", "k=100"])
    axes[1].set(title="Top-ten hub concentration", ylabel="Assignment fraction", xticks=x,
                xticklabels=["k=1", "k=10", "k=100"])
    axes[0].legend(frameon=False, fontsize=8)
    for ax in axes: ax.grid(axis="y", alpha=.2)
    fig.suptitle("Local normalization reduces hub concentration")
    save(fig, "hubness_comparison")


def null_calibration():
    d = pd.read_csv(RESULTS / "empirical_null_calibration.csv")
    order = ["sparse", "intermediate", "dense"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    data = [d.loc[d.density_stratum.eq(s), "empirical_p"] for s in order]
    axes[0].boxplot(data, tick_labels=order, showfliers=False)
    axes[0].set_yscale("log"); axes[0].set(ylabel="Global empirical P-value", title="Null calibration varies by density")
    for s, color in zip(order, ["#4C78A8", "#F2CF5B", "#E45756"]):
        sub = d[d.density_stratum.eq(s)]
        axes[1].scatter(sub.density_k10, sub.exact_cosine, s=14, alpha=.55, label=s, color=color)
    axes[1].set(xlabel="Mean cosine to 10 nearest references", ylabel="Exact-pair cosine",
                title="Exact similarity follows local density")
    axes[1].legend(frameon=False)
    for ax in axes: ax.grid(alpha=.2)
    save(fig, "empirical_null_density_calibration")


def rank_changes():
    d = pd.read_parquet(RESULTS / "per_query_ranked_results.parquet")
    wide = d.pivot(index="query_index", columns="method", values="exact_rank")
    methods = ["CSLS(k=10)", "LocalScaling(k=10)", "NICDM(k=10)"]
    strata = (d[d.method.eq("Cosine")].set_index("query_index").density_stratum
              .reindex(wide.index))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), sharex=True, sharey=True)
    lim = max(2, np.quantile(np.abs(np.log10(wide[methods].clip(lower=1)) -
                                    np.log10(wide["Cosine"].clip(lower=1)).to_numpy()[:, None]), .99))
    for ax, method in zip(axes, methods):
        change = np.log10(wide["Cosine"].clip(lower=1)) - np.log10(wide[method].clip(lower=1))
        for s, color in zip(["sparse", "intermediate", "dense"], ["#4C78A8", "#F2CF5B", "#E45756"]):
            vals = change[strata.eq(s)]
            ax.hist(vals, bins=np.linspace(-lim, lim, 31), alpha=.55, color=color, label=s)
        ax.axvline(0, color="black", lw=1)
        ax.set(title=method, xlabel="$log_{10}$(cosine rank / method rank)")
    axes[0].set_ylabel("Queries"); axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Rank changes include both improvements and degradations")
    save(fig, "rank_change_distributions")


if __name__ == "__main__":
    method_comparison()
    density_strata()
    hubness()
    null_calibration()
    rank_changes()
    print(f"Saved figure suite to {FIGURES}")
