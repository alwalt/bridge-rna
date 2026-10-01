#!/usr/bin/env python3
"""Build a compact four-panel paper summary from saved benchmark results."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "results/paper_summary"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    matched = pd.read_csv(HERE / "results/matched_gtex_tcga/gtex_tcga_summary.csv")
    coex = pd.read_csv(HERE / "results/coexpression_baseline/coexpression_vs_bridge_summary.csv")
    conditional = pd.read_csv(HERE / "results/l12_conditional_function/l12_conditional_results.csv")
    modules = pd.read_csv(HERE / "results/contextual_gene_modules/contextual_gene_module_summary.csv")
    controls = pd.read_csv(HERE / "results/contextual_gene_modules/expression_control_summary.csv")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    colors = {"GO": "#2878B5", "KEGG": "#E07A1F"}
    layer_order = ["L0_static", "L1_early", "L6_middle", "L12_final"]
    layer_labels = ["L0", "L1", "L6", "L12"]

    # A: functional-neighborhood organization across depth.
    ax = axes[0, 0]
    depth = matched[(matched.geometry == "mean_centered_cosine") & (matched.k == 25)]
    for dataset, linestyle in (("GTEx", "-"), ("TCGA", "--")):
        for library in ("GO", "KEGG"):
            frame = depth[(depth.dataset == dataset) & (depth.library == library)].set_index("layer").loc[layer_order]
            ax.plot(layer_labels, frame.neighbor_fold, marker="o", linestyle=linestyle,
                    color=colors[library], label=f"{dataset} {library}")
    ax.axhline(1, color="black", linestyle=":", linewidth=1)
    ax.set(title="A  Functional neighborhoods across depth", ylabel="Fold over random", xlabel="Layer")
    ax.legend(fontsize=8, ncol=2)

    # B: matched coexpression comparison at k=25.
    ax = axes[0, 1]
    methods = ["Pearson", "Spearman", "Bridge L0", "Bridge L1", "Bridge L6", "Bridge L12"]
    positions = np.arange(len(methods))
    offsets = {("GTEx", "GO"): -0.24, ("GTEx", "KEGG"): -0.08,
               ("TCGA", "GO"): 0.08, ("TCGA", "KEGG"): 0.24}
    markers = {"GTEx": "o", "TCGA": "s"}
    for dataset in ("GTEx", "TCGA"):
        for library in ("GO", "KEGG"):
            frame = coex[(coex.dataset == dataset) & (coex.library == library) & (coex.k == 25)].set_index("method").loc[methods]
            x = positions + offsets[(dataset, library)]
            ax.errorbar(x, frame.fold_over_null,
                        yerr=np.vstack([frame.fold_over_null-frame.ci_low, frame.ci_high-frame.fold_over_null]),
                        fmt=markers[dataset], linestyle="none", capsize=2, color=colors[library],
                        label=f"{dataset} {library}")
    ax.axhline(1, color="black", linestyle=":", linewidth=1)
    ax.set(title="B  Bridge versus coexpression (k=25)", ylabel="Fold over random")
    ax.set_xticks(positions, ["Pearson", "Spearman", "L0", "L1", "L6", "L12"], rotation=25)
    ax.legend(fontsize=8, ncol=2)

    # C: independent L12 odds ratios.
    ax = axes[1, 0]
    ordered = conditional.set_index(["dataset", "library"]).loc[
        [(d, l) for d in ("GTEx", "TCGA") for l in ("GO", "KEGG")]
    ]
    labels = ["GTEx\nGO", "GTEx\nKEGG", "TCGA\nGO", "TCGA\nKEGG"]
    x = np.arange(4)
    ax.errorbar(x, ordered.l12_odds_ratio_per_sd,
                yerr=np.vstack([ordered.l12_odds_ratio_per_sd-ordered.l12_or_ci_low,
                                ordered.l12_or_ci_high-ordered.l12_odds_ratio_per_sd]),
                fmt="o", capsize=4, color="#6F4E7C")
    ax.axhline(1, color="black", linestyle=":", linewidth=1)
    ax.set(title="C  L12 contribution after expression controls",
           ylabel="Adjusted odds ratio per SD L12 cosine")
    ax.set_xticks(x, labels)

    # D: original and linearly TPM-residualized L12 modules.
    ax = axes[1, 1]
    original = modules[modules.layer == "L12_final"].set_index("dataset")
    residual = controls[(controls.control == "linear_tpm_residual") &
                        (controls.layer == "L12_final")].set_index("dataset")
    categories = ["GO terms", "KEGG pathways", "Stability ARI"]
    x = np.arange(3)
    width = 0.18
    for dataset_index, dataset in enumerate(("GTEx", "TCGA")):
        values = [100 * residual.loc[dataset, "significant_go_terms"] /
                  original.loc[dataset, "significant_go_terms"],
                  100 * residual.loc[dataset, "significant_kegg_terms"] /
                  original.loc[dataset, "significant_kegg_terms"],
                  100 * residual.loc[dataset, "stability_ari"] /
                  original.loc[dataset, "stability_ari"]]
        offset = (dataset_index - 0.5) * width
        ax.bar(x + offset, values, width, label=dataset,
               color="#59A14F" if dataset == "GTEx" else "#E15759")
    ax.set(title="D  Functional modules after linear TPM removal",
           ylabel="Residual/original retained (%)")
    ax.axhline(100, color="black", linestyle=":", linewidth=1)
    ax.set_xticks(x, categories)
    ax.legend(fontsize=8)

    for extension in ("png", "pdf"):
        fig.savefig(OUT / f"bridge_contextual_biology_summary.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
