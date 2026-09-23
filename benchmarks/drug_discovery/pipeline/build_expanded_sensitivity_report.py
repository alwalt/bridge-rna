#!/usr/bin/env python3
"""Build report and figure for expanded ChEMBL and complementarity sensitivity."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "results/expanded_chembl_sensitivity"


def main():
    comp = pd.read_csv(OUT / "expanded_vs_restrictive.csv")
    metrics = pd.read_csv(OUT / "partition_metrics.csv")
    paths = pd.read_csv(OUT / "partition_pathway_summary.csv")
    deletion = pd.read_csv(OUT / "partition_deletion_summary.csv")
    recurrence = pd.read_csv(OUT / "recurrent_drugs.csv")
    targets = pd.read_csv(OUT / "recurrent_drug_targets.csv")
    repeated = pd.read_csv(OUT / "recurrent_partition_genes.csv")
    matrix = pd.read_csv(HERE / "results/dataset_matrix.csv")[["condition", "dataset"]]
    deletion = deletion.merge(matrix, on="dataset")

    agg = metrics.groupby(["condition", "partition"], as_index=False).agg(
        recurrence_fraction=("recurrent_in_any_other_fraction", "mean"),
        osdr_jaccard=("osdr_gene_jaccard", "mean"), external_overlap=("external_overlap_count", "sum"),
        external_min_q=("external_fdr", "min"), targetable_fraction=("targetable_gene_fraction", "mean"),
        significant_drugs=("significant_drug_enrichments", "sum"))
    dagg = deletion.groupby(["condition", "partition"], as_index=False).agg(
        deletion_change=("absolute_score_change", "mean"),
        deletion_per100=("absolute_change_per_100_genes", "mean"),
        fraction_remaining=("fraction_signal_remaining", "mean"))
    overall = metrics.groupby("partition", as_index=False).agg(
        recurrence_fraction=("recurrent_in_any_other_fraction", "mean"),
        osdr_jaccard=("osdr_gene_jaccard", "mean"), targetable_fraction=("targetable_gene_fraction", "mean"),
        significant_drugs=("significant_drug_enrichments", "sum"))
    overall = overall.merge(deletion.groupby("partition", as_index=False).agg(
        deletion_change=("absolute_score_change", "mean"),
        deletion_per100=("absolute_change_per_100_genes", "mean"),
        fraction_remaining=("fraction_signal_remaining", "mean")), on="partition")
    agg.to_csv(OUT / "partition_condition_summary.csv", index=False)
    overall.to_csv(OUT / "partition_overall_summary.csv", index=False)

    muscle_targets = targets[(targets.condition.eq("Muscle atrophy")) & targets.analysis.eq("Bridge")]
    universal = sorted(muscle_targets.groupby("target_gene").dataset.nunique().loc[lambda x: x.eq(3)].index)
    muscle_drugs = recurrence[(recurrence.condition.eq("Muscle atrophy")) & recurrence.analysis.eq("Bridge")].drug_name.tolist()
    recurrent_path = paths[paths.scope.eq("recurrent_partition")].set_index("condition")
    rb = repeated[repeated.partition.eq("Bridge-only")].groupby("condition").agg(
        recurrent_genes=("gene", "size"), three_way=("dataset_count", lambda x: int((x == 3).sum())),
        includes_osdr=("includes_osdr", "sum"), external=("external_support", "sum"),
        targetable=("expanded_chembl_target", "sum"), actionable=("eligible_drug_count", lambda x: int((x > 0).sum())))

    lines = ["# Expanded ChEMBL Phase 2 sensitivity and Bridge–DE complementarity", "",
             "This is a new sensitivity layer over the frozen Phase 2 top-500 modules.",
             "Expression preprocessing, embeddings, condition vectors, Integrated",
             "Gradients, differential expression, and module construction were not rerun.", "",
             "## Expanded-universe drug convergence", "",
             "The Evaluation 2 ChEMBL 37 graph (4,389 drugs, 2,046 targets, 16,986",
             "edges) was filtered to the 366 drugs with >10 targets. Enrichment uses",
             "right-sided hypergeometric tests over the fixed 15,165-gene canonical",
             "background with BH correction. Convergence is the original mean of three",
             "pairwise top-25 nonzero-overlap drug-list Jaccards.", "",
             "| Condition | Expanded Bridge | Expanded DE | Expanded difference | exact p | Restrictive Bridge | Restrictive DE |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in comp.itertuples():
        lines.append(f"| {row.condition} | {row.bridge_mean_pairwise_top25_jaccard_expanded:.3f} | {row.de_mean_pairwise_top25_jaccard_expanded:.3f} | {row.bridge_minus_de_expanded:.3f} | {row.exact_p_expanded:.3f} | {row.bridge_mean_pairwise_top25_jaccard_restrictive:.3f} | {row.de_mean_pairwise_top25_jaccard_restrictive:.3f} |")
    lines += ["", "The expanded resource changes the headline substantially. Muscle",
              "remains the strongest Bridge condition and Bridge remains far above DE",
              "(0.164 vs 0.0068), but Bridge's absolute muscle Jaccard falls 77% from",
              "0.714. Thus the original magnitude was highly dependent on the sparse",
              "direct-mechanism graph; the qualitative Bridge>DE ordering survives.", "",
              f"Three Bridge drugs recur across all muscle datasets: {', '.join(muscle_drugs)}.",
              f"Their same-gene recurrence across all three datasets comprises {', '.join(universal)}.",
              "This broadens the strict universal structure beyond ERBB2/PDE5A to",
              "mitochondrial complex-I (NDUFS6), MAP3K20, and PDE4B, but remains a",
              "compact five-target/three-drug structure rather than a broad drug program.", "",
              "## Bridge/DE complementarity", "",
              "| Condition | Partition | Recurs in another dataset | OSDR Jaccard | External overlaps | minimum external q | Targetable | Mean deletion loss |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in agg.itertuples():
        d = dagg[(dagg.condition.eq(row.condition)) & dagg.partition.eq(row.partition)].iloc[0]
        lines.append(f"| {row.condition} | {row.partition} | {row.recurrence_fraction:.3f} | {row.osdr_jaccard:.3f} | {row.external_overlap} | {row.external_min_q:.3g} | {row.targetable_fraction:.3f} | {d.deletion_change:.4f} |")

    lines += ["", "Across all conditions, Bridge-only genes recur in another independent",
              "dataset at 0.435 on average, versus 0.117 for DE-only genes. Their mean",
              "terrestrial–OSDR Jaccard is 0.159 versus 0.016. Targetability is similar",
              "but slightly higher for Bridge-only genes (0.163 vs 0.146).", "",
              "Observed partition deletion is also representation-consistent: Bridge-only",
              "masking changes the frozen projection by 0.0635 on average (0.0143 per",
              "100 genes), versus 0.0246 (0.00549 per 100 genes) for DE-only masking.",
              "Shared genes have the largest loss per gene. These are effect sizes, not",
              "partition-level significance tests; the frozen Phase 2 whole-module",
              "expression-matched deletion null remains the inferential control.", "",
              "## Repeated Bridge-only biology", "",
              "| Condition | Recurrent genes | Present in all 3 | Includes OSDR | External support | ChEMBL targets | Targeted by eligible drug | Top pathway | adjusted p |",
              "|---|---:|---:|---:|---:|---:|---:|---|---:|"]
    for condition, row in rb.iterrows():
        pathway = recurrent_path.loc[condition]
        lines.append(f"| {condition} | {row.recurrent_genes} | {row.three_way} | {row.includes_osdr} | {row.external} | {row.targetable} | {row.actionable} | {pathway.top_term_name} | {pathway.top_adjusted_pvalue:.3g} |")

    lines += ["", "The clearest condition-specific evidence is:", "",
              "- Bone: Bridge-only genes are externally enriched in all three datasets",
              "  after FDR correction and converge on extracellular-matrix organization.",
              "  Recurrent Bridge-only drugs include luteolin, ocriplasmin, and",
              "  collagenase *C. histolyticum*; these are target recurrences, not efficacy claims.",
              "- Muscle: Bridge-only genes are enriched for the skeletal-muscle-atrophy",
              "  reference in all three datasets and converge on muscle cytoskeleton.",
              "  Repeated genes include LAMB2, ITGA7, DES, NEB, TPM2, ACTA1, ERBB2,",
              "  PDE5A, and PDE4B. BMS-690514 and pentoxifylline recur specifically from",
              "  Bridge-only partitions.",
              "- Radiation: Bridge-only recurrence and pathway coherence are strong, but",
              "  the narrow 39-gene canonical DisGeNET Radiation Damage reference provides",
              "  no corroborating Bridge-only overlap. External support is therefore",
              "  unresolved rather than negative evidence against the representation.", "",
              "## Answer to the central question", "",
              "Qualified yes for bone and muscle: Bridge prioritizes gene sets that DE does",
              "not, that recur much more strongly across independent and OSDR datasets,",
              "carry frozen-representation signal, match external condition genes, form",
              "coherent pathways, and contain recurrent druggable targets. The conclusion",
              "is not universal (radiation lacks adequate external-reference confirmation),",
              "and pharmacological actionability is modest: no direction-aware reversal was",
              "tested, expanded-universe drug convergence is smaller than the restrictive",
              "result, and target enrichment is not evidence of therapeutic efficacy.", ""]
    (OUT / "expanded_sensitivity_report.md").write_text("\n".join(lines))

    # Compact publication figure.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(3); width = .18
    for i, (label, col, color) in enumerate([
        ("Bridge restrictive", "bridge_mean_pairwise_top25_jaccard_restrictive", "#8bb7df"),
        ("Bridge expanded", "bridge_mean_pairwise_top25_jaccard_expanded", "#2166ac"),
        ("DE restrictive", "de_mean_pairwise_top25_jaccard_restrictive", "#e6b873"),
        ("DE expanded", "de_mean_pairwise_top25_jaccard_expanded", "#b35806")]):
        axes[0].bar(x + (i-1.5)*width, comp[col], width, label=label, color=color)
    axes[0].set_xticks(x, ["Radiation", "Bone", "Muscle"])
    axes[0].set_ylabel("Mean pairwise top-25 drug Jaccard")
    axes[0].set_title("Drug convergence is resource-sensitive")
    axes[0].legend(frameon=False, fontsize=8)
    pivot = overall.set_index("partition")
    xx = np.arange(3)
    axes[1].bar(xx-.18, pivot.loc[["Bridge-only","Shared","DE-only"], "recurrence_fraction"], .36, label="Any-study recurrence", color="#3b7ea1")
    axes[1].bar(xx+.18, pivot.loc[["Bridge-only","Shared","DE-only"], "osdr_jaccard"], .36, label="Terrestrial–OSDR Jaccard", color="#7fb069")
    axes[1].set_xticks(xx, ["Bridge-only", "Shared", "DE-only"])
    axes[1].set_ylabel("Mean gene recurrence")
    axes[1].set_title("Bridge-only genes reproduce across studies")
    axes[1].legend(frameon=False, fontsize=8)
    fig.text(.5, .01, "Expanded ChEMBL sensitivity; target enrichment is not therapeutic efficacy", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(OUT / "expanded_sensitivity_summary.png", dpi=300)
    fig.savefig(OUT / "expanded_sensitivity_summary.pdf")


if __name__ == "__main__":
    main()
