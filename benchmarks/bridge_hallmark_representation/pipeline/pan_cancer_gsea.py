#!/usr/bin/env python3
"""Pan-cancer Bridge pathway deltas versus matched whole/vocabulary GSEA."""
from __future__ import annotations

import json

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from common import CONFIG, RESULTS, resolve, read_gmt, read_vocab, save_json
from de_gsea import de_table, gsea, norm

OUT = RESULTS / "pan_cancer_gsea"


def overlap_fraction(left, right, k):
    a = set(np.argsort(-np.abs(np.asarray(left)))[:k])
    b = set(np.argsort(-np.abs(np.asarray(right)))[:k])
    return len(a & b) / k


def correlation_metrics(contrast, comparison, left_name, right_name, left, right):
    left = np.asarray(left, float); right = np.asarray(right, float)
    return {
        "contrast": contrast, "comparison": comparison,
        "left_quantity": left_name, "right_quantity": right_name,
        "pearson": stats.pearsonr(left, right).statistic,
        "spearman": stats.spearmanr(left, right).statistic,
        "direction_agreement": np.mean(np.sign(left) == np.sign(right)),
        "top5_overlap": overlap_fraction(left, right, 5),
        "top10_overlap": overlap_fraction(left, right, 10),
    }


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(RESULTS / "tcga_sample_manifest.parquet")
    profiles = pd.read_parquet(RESULTS / "tcga_zero_shot_hallmark_profiles.parquet")
    if profiles.sample_id.tolist() != meta.sample_id.tolist():
        raise AssertionError("Frozen Bridge profile/sample ordering mismatch")
    pathway_order = [column for column in profiles if column != "sample_id"]
    if len(pathway_order) != 50:
        raise AssertionError(len(pathway_order))
    expression = np.load(resolve(CONFIG["tcga_full_expression"]), mmap_mode="r")
    with h5py.File(resolve(CONFIG["tcga_full_h5"]), "r") as handle:
        genes = [norm(value.decode() if isinstance(value, bytes) else value) for value in handle["meta/genes"][:]]
    if expression.shape != (len(meta), len(genes)):
        raise AssertionError((expression.shape, len(meta), len(genes)))
    bridge_genes = set(read_vocab())
    gene_sets = read_gmt()
    merged = meta.merge(profiles, on="sample_id", validate="one_to_one", sort=False)
    gsea_tables, pathway_rows, manifests = [], [], []

    for contrast, definition in CONFIG["tcga_pan_cancer_contrasts"].items():
        group_a, group_b = definition["groups"]
        mask_a = meta.cohort.eq(group_a).to_numpy()
        mask_b = meta.cohort.eq(group_b).to_numpy()
        if mask_a.sum() < 20 or mask_b.sum() < 20:
            raise AssertionError(f"Insufficient cohort for {contrast}: {mask_a.sum()}, {mask_b.sum()}")
        de = de_table(expression, genes, mask_a, mask_b)
        reduced = de[de.gene.isin(bridge_genes)].copy()
        de.to_parquet(OUT / f"de_{contrast}_whole.parquet", index=False)
        reduced.to_parquet(OUT / f"de_{contrast}_bridge_vocab.parquet", index=False)
        whole = gsea(de, gene_sets, "whole", contrast)
        vocab = gsea(reduced, gene_sets, "bridge_15165", contrast)
        gsea_tables.extend([whole, vocab])
        delta = (
            merged.loc[merged.cohort.eq(group_a), pathway_order].mean()
            - merged.loc[merged.cohort.eq(group_b), pathway_order].mean()
        )
        whole_index = whole.set_index("hallmark").reindex(pathway_order)
        vocab_index = vocab.set_index("hallmark").reindex(pathway_order)
        if whole_index.nes.isna().any() or vocab_index.nes.isna().any():
            raise AssertionError(f"Missing Hallmark result in {contrast}")
        for hallmark in pathway_order:
            pathway_rows.append({
                "contrast": contrast, "contrast_family": definition["family"],
                "contrast_type": "subtype_vs_subtype", "group_a": group_a, "group_b": group_b,
                "hallmark": hallmark, "bridge_delta": delta[hallmark],
                "whole_gsea_nes": whole_index.loc[hallmark, "nes"],
                "whole_gsea_fdr": whole_index.loc[hallmark, "gsea_fdr"],
                "vocab_gsea_nes": vocab_index.loc[hallmark, "nes"],
                "vocab_gsea_fdr": vocab_index.loc[hallmark, "gsea_fdr"],
                "bridge_whole_direction_agree": np.sign(delta[hallmark]) == np.sign(whole_index.loc[hallmark, "nes"]),
                "bridge_vocab_direction_agree": np.sign(delta[hallmark]) == np.sign(vocab_index.loc[hallmark, "nes"]),
                "whole_vocab_direction_agree": np.sign(whole_index.loc[hallmark, "nes"]) == np.sign(vocab_index.loc[hallmark, "nes"]),
            })
        manifests.append({
            "contrast": contrast, "contrast_family": definition["family"],
            "contrast_type": "subtype_vs_subtype", "group_a": group_a, "group_b": group_b,
            "n_a": int(mask_a.sum()), "n_b": int(mask_b.sum()),
            "patient_disjoint": bool(meta.loc[mask_a, "patient_id"].isin(meta.loc[mask_b, "patient_id"]).sum() == 0),
            "de_method": "Welch t statistic on log1p CPM",
            "ranking_statistic": "signed Welch t statistic",
            "whole_genes": len(de), "bridge_intersection_genes": len(reduced),
            "gsea_permutations": CONFIG["gsea_permutations"],
        })

    gsea_all = pd.concat(gsea_tables, ignore_index=True)
    pathways = pd.DataFrame(pathway_rows)
    manifest = pd.DataFrame(manifests)
    gsea_all.to_csv(OUT / "conventional_gsea_results.csv", index=False)
    pathways.to_csv(OUT / "pathway_comparison.csv", index=False)
    manifest.to_csv(OUT / "contrast_manifest.csv", index=False)
    summarize(pathways, pathway_order)


def summarize(pathways, pathway_order):
    metric_rows = []
    for contrast, frame in pathways.groupby("contrast", sort=False):
        metric_rows.extend([
            correlation_metrics(contrast, "bridge_vs_whole", "bridge_delta", "whole_gsea_nes", frame.bridge_delta, frame.whole_gsea_nes),
            correlation_metrics(contrast, "bridge_vs_vocab", "bridge_delta", "vocab_gsea_nes", frame.bridge_delta, frame.vocab_gsea_nes),
            correlation_metrics(contrast, "whole_vs_vocab", "whole_gsea_nes", "vocab_gsea_nes", frame.whole_gsea_nes, frame.vocab_gsea_nes),
        ])
        whole_sig = set(frame.loc[frame.whole_gsea_fdr < .05, "hallmark"])
        vocab_sig = set(frame.loc[frame.vocab_gsea_fdr < .05, "hallmark"])
        for row in metric_rows[-3:]:
            row.update({
                "whole_significant": len(whole_sig), "vocab_significant": len(vocab_sig),
                "significant_both": len(whole_sig & vocab_sig),
                "significant_union": len(whole_sig | vocab_sig),
                "significant_jaccard": len(whole_sig & vocab_sig) / max(len(whole_sig | vocab_sig), 1),
            })
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "per_contrast_agreement.csv", index=False)

    summary_rows = []
    for comparison, frame in metrics.groupby("comparison", sort=False):
        summary_rows.append({
            "comparison": comparison, "contrasts": len(frame),
            "median_pearson": frame.pearson.median(),
            "pearson_iqr_low": frame.pearson.quantile(.25), "pearson_iqr_high": frame.pearson.quantile(.75),
            "median_spearman": frame.spearman.median(),
            "spearman_iqr_low": frame.spearman.quantile(.25), "spearman_iqr_high": frame.spearman.quantile(.75),
            "median_direction_agreement": frame.direction_agreement.median(),
            "positive_spearman_contrasts": int(frame.spearman.gt(0).sum()),
            "positive_spearman_proportion": frame.spearman.gt(0).mean(),
            "median_top5_overlap": frame.top5_overlap.median(),
            "median_top10_overlap": frame.top10_overlap.median(),
        })
    pd.DataFrame(summary_rows).to_csv(OUT / "pan_contrast_summary.csv", index=False)

    pathway_summary = pathways.groupby("hallmark", as_index=False).agg(
        bridge_whole_spearman=("bridge_delta", lambda x: np.nan),
        bridge_whole_direction_rate=("bridge_whole_direction_agree", "mean"),
        bridge_vocab_direction_rate=("bridge_vocab_direction_agree", "mean"),
        whole_vocab_direction_rate=("whole_vocab_direction_agree", "mean"),
    )
    for index, row in pathway_summary.iterrows():
        block = pathways[pathways.hallmark.eq(row.hallmark)]
        pathway_summary.loc[index, "bridge_whole_spearman"] = stats.spearmanr(block.bridge_delta, block.whole_gsea_nes).statistic
        pathway_summary.loc[index, "bridge_vocab_spearman"] = stats.spearmanr(block.bridge_delta, block.vocab_gsea_nes).statistic
        pathway_summary.loc[index, "whole_vocab_spearman"] = stats.spearmanr(block.whole_gsea_nes, block.vocab_gsea_nes).statistic
        pathway_summary.loc[index, "bridge_whole_mean_sign_product"] = np.mean(np.sign(block.bridge_delta) * np.sign(block.whole_gsea_nes))
        pathway_summary.loc[index, "bridge_vocab_mean_sign_product"] = np.mean(np.sign(block.bridge_delta) * np.sign(block.vocab_gsea_nes))
    pathway_summary["agreement_class"] = np.select(
        [pathway_summary.bridge_whole_direction_rate.ge(.8) & pathway_summary.bridge_vocab_direction_rate.ge(.8),
         pathway_summary.bridge_whole_direction_rate.le(.2) & pathway_summary.bridge_vocab_direction_rate.le(.2)],
        ["consistent_agreement", "consistent_disagreement"], default="context_dependent",
    )
    pathway_summary.to_csv(OUT / "pathway_wise_agreement.csv", index=False)
    plot_results(pathways, metrics, pathway_order)
    save_json(OUT / "provenance.json", {
        "bridge_scores": "existing frozen GTEx-trained per-sample 50-Hallmark predictions",
        "bridge_delta": "mean(group A) minus mean(group B)",
        "expression": "TCGA full log1p(CPM) matrix aligned to the primary-tumor patient manifest",
        "de": "Welch t statistic; no covariates",
        "gsea": "gseapy prerank, weight 1, 1,000 permutations, seed 42",
        "universe_comparison": "identical samples/statistic/Hallmarks/settings; only ranked gene universe differs",
        "contrast_types": ["subtype_vs_subtype"],
        "tumor_vs_normal_status": "not run: frozen Bridge profiles contain primary tumors only",
        "bridge_frozen": True,
    })


def plot_results(pathways, metrics, pathway_order):
    contrast_order = list(CONFIG["tcga_pan_cancer_contrasts"])
    value_specs = [
        ("bridge_delta", "Bridge pathway delta", "bridge_delta_heatmap"),
        ("whole_gsea_nes", "Whole-transcriptome GSEA NES", "whole_gsea_nes_heatmap"),
        ("vocab_gsea_nes", "Bridge-vocabulary GSEA NES", "vocab_gsea_nes_heatmap"),
    ]
    limits = {column: np.nanmax(np.abs(pathways[column])) for column, _, _ in value_specs}
    for column, title, filename in value_specs:
        matrix = pathways.pivot(index="contrast", columns="hallmark", values=column).loc[contrast_order, pathway_order]
        fig, ax = plt.subplots(figsize=(18, 5))
        image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="RdBu_r",
                          vmin=-limits[column], vmax=limits[column])
        ax.set_yticks(np.arange(len(matrix.index)), matrix.index)
        ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=90)
        fig.colorbar(image, ax=ax, label=title)
        ax.set(title=title, xlabel="Hallmark", ylabel="TCGA subtype contrast")
        ax.tick_params(axis="x", labelsize=6)
        fig.tight_layout(); fig.savefig(OUT / f"{filename}.png", dpi=220); fig.savefig(OUT / f"{filename}.pdf"); plt.close(fig)

    for y, title, filename in [
        ("whole_gsea_nes", "Bridge delta versus whole-transcriptome GSEA", "bridge_vs_whole_all_contrasts"),
        ("vocab_gsea_nes", "Bridge delta versus vocabulary-matched GSEA", "bridge_vs_vocab_all_contrasts"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 7))
        for contrast, frame in pathways.groupby("contrast", sort=False):
            ax.scatter(frame.bridge_delta, frame[y], s=22, alpha=.65, label=contrast)
        ax.axhline(0, color="grey", linewidth=.7); ax.axvline(0, color="grey", linewidth=.7)
        ax.set(xlabel="Frozen Bridge group delta", ylabel=y, title=title)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout(); fig.savefig(OUT / f"{filename}.png", dpi=220); fig.savefig(OUT / f"{filename}.pdf"); plt.close(fig)

    subset = metrics[metrics.comparison.isin(["bridge_vs_whole", "bridge_vs_vocab"])]
    labels = ["bridge_vs_whole", "bridge_vs_vocab"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, metric in zip(axes, ["spearman", "direction_agreement"]):
        values = [subset.loc[subset.comparison.eq(label), metric].to_numpy() for label in labels]
        axis.boxplot(values, tick_labels=labels)
        for position, points in enumerate(values, 1):
            axis.scatter(np.full(len(points), position), points, color="black", zorder=3)
    axes[0].set(title="Ranking agreement across contrasts", xlabel="", ylabel="Spearman")
    axes[1].set(title="Direction agreement across contrasts", xlabel="", ylabel="Fraction of 50 Hallmarks")
    fig.tight_layout(); fig.savefig(OUT / "pan_contrast_agreement_summary.png", dpi=220); fig.savefig(OUT / "pan_contrast_agreement_summary.pdf"); plt.close(fig)


if __name__ == "__main__":
    run()
