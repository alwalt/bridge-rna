#!/usr/bin/env python3
"""Build the integrated three-evaluation benchmark report and figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"
EVAL = RESULTS / "evaluation"
PHASE2 = Path(__file__).resolve().parents[2] / "drug_discovery/results"
LABEL = {
    "GSE138614": "MS active lesion",
    "GSE101794": "Crohn ileum",
    "GSE72509": "SLE blood",
}


def sci(value):
    if pd.isna(value):
        return "NA"
    return f"{value:.2e}"


def rank_text(value):
    return "excluded" if pd.isna(value) else str(int(value))


def genes_text(value):
    return "—" if pd.isna(value) or not str(value).strip() else str(value)


def main():
    disease = pd.read_csv(EVAL / "disease_gene_recovery.csv")
    drugs = pd.read_csv(EVAL / "published_drug_recovery.csv")
    top = pd.read_csv(EVAL / "top_independent_drugs.csv")
    cohorts = pd.read_csv(RESULTS / "locked_cohorts.csv")
    primary = disease[(disease.module_size.eq(500)) &
                      disease.reference_definition.eq("all_v7_associations")]

    lines = [
        "# Integrated BridgeRNA biological and drug-recovery benchmark",
        "",
        "This report separates de Weerd replication-style evaluations from the",
        "previously completed cross-study space-condition extension. The frozen",
        "ExpressionPerformer checkpoint, natural `log1p(TPM)` input, signed",
        "leave-donor-out Integrated Gradients attribution, and top-500 primary",
        "module definition are unchanged.", "",
        "## Evaluation 1: Disease-gene recovery", "",
        "The primary reference is exact-CUI DisGeNET v7.0 intersected with each",
        "dataset's expressed canonical-gene universe. This is the release closest",
        "to de Weerd, but not an exact reconstruction: their filtered gene lists",
        "were not published and their reported reference totals differ markedly.", "",
        "| Disease | Method | Recovered / reference | OR | raw p | BH q | de Weerd VAE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for accession in ["GSE138614", "GSE101794", "GSE72509"]:
        for method in ["Bridge", "DE"]:
            row = primary[(primary.accession.eq(accession)) & primary.method.eq(method)].iloc[0]
            published = (f"{int(row.deweerd_vae_recovered)}/{int(row.deweerd_vae_reference_genes)}, "
                         f"OR {row.deweerd_vae_odds_ratio:.2f}" if method == "Bridge" else "—")
            lines.append(f"| {LABEL[accession]} | {method} | {int(row.known_genes_recovered)}/{int(row.disgenet_genes_in_universe)} "
                         f"| {row.odds_ratio:.2f} | {sci(row.pvalue)} | {sci(row.fdr)} | {published} |")

    lines += ["", "Module-size sensitivity covers 50–500 genes in increments of 50;",
              "the complete machine-readable table reports raw p, BH q, overlap, and",
              "odds ratio for every size. A DisGeNET score ≥0.1 sensitivity is kept",
              "separate from the all-association primary analysis.", "",
              "At top 500, DE recovers more reference genes than Bridge for MS",
              "(87 vs 72) and Crohn (79 vs 70), while Bridge recovers more for SLE",
              "(106 vs 72). Therefore these cohorts do not support a general claim",
              "that Bridge improves disease-gene recovery over DE; the advantage is",
              "disease-specific and strongest for SLE.", "",
              "All three studies are overwhelmingly pretraining-exposed (39/41 MS,",
              "71/74 Crohn, 116/117 SLE in ARCHS4 train/validation), so this evaluates",
              "recovery/reproducibility rather than unseen-study generalization.", "",
              "## Evaluation 2: Published drug recovery", "",
              "Exact DrugBank replication was not authorized. De Weerd tested 328",
              "DrugBank drugs with >10 targets against 16,600 proteins. The frozen",
              "ChEMBL 37 graph has 985 approved small-molecule mechanism drugs, 323",
              "targets, and only one drug with >10 targets. Thus the strict analogue",
              "cannot meaningfully compare published-drug recovery. The ≥3-target",
              "ChEMBL endpoint and ≥1-target reference audit are labeled sensitivities.", "",
              "| Disease | Published drug (#) | ChEMBL targets | Bridge rank | DE rank | Bridge overlap | DE overlap | Bridge p/q | DE p/q |",
              "|---|---|---:|---:|---:|---|---|---:|---:|" ]
    for accession in ["GSE138614", "GSE101794", "GSE72509"]:
        subset = drugs[drugs.accession.eq(accession)]
        for drug, group in subset.groupby("reference_drug", sort=False):
            bridge = group[group.method.eq("Bridge")].iloc[0]
            de = group[group.method.eq("DE")].iloc[0]
            def chosen_rank(row):
                for column in ["strict_rank", "sensitivity_ge3_rank", "audit_ge1_rank"]:
                    if column in row and pd.notna(row[column]):
                        return rank_text(row[column])
                return "absent"
            lines.append(f"| {LABEL[accession]} | {drug.title()} ({int(bridge.deweerd_reported_rank)}) "
                         f"| {int(bridge.target_count)} | {chosen_rank(bridge)} | {chosen_rank(de)} "
                         f"| {genes_text(bridge.overlap_genes)} | {genes_text(de.overlap_genes)} "
                         f"| {sci(bridge.pvalue)}/{sci(bridge.fdr)} | {sci(de.pvalue)}/{sci(de.fdr)} |")

    lines += ["", "Ranks in the table use the strict endpoint when eligible, then the",
              "≥3 sensitivity, then the ≥1 reference-audit universe. Missing compounds",
              "are absent from the approved direct-mechanism graph; below-threshold",
              "ranks are not primary benchmark recoveries. The full table contains",
              "target counts, raw p, FDR, and endpoint-specific ranks.", "",
              "Only Sunitinib overlaps a top-500 module (Bridge: CSF1R; raw",
              "p=0.282, FDR=1.0), and it is not significant. Ibrutinib,",
              "zanubrutinib, gemcitabine, and fostamatinib are represented but have",
              "zero module-target overlap for both methods. Consequently, this",
              "ChEMBL audit does not establish that Bridge reproduces de Weerd's",
              "published drug signals better than DE.", "",
              "Top independently generated nonzero-overlap drugs (≥3 targets):", ""]
    for accession in ["GSE138614", "GSE101794", "GSE72509"]:
        for method in ["Bridge", "DE"]:
            names = top[(top.accession.eq(accession)) & top.method.eq(method)].head(5).drug_name.tolist()
            lines.append(f"- {LABEL[accession]}, {method}: {', '.join(names) if names else 'none'}")

    phase = pd.read_csv(PHASE2 / "evaluation/primary_effects.csv")
    lines += ["", "## Evaluation 3: Cross-study space-condition convergence", "",
              "This is the previously completed Bridge-specific extension, not a de",
              "Weerd replication. No Phase 2 analysis was rerun.", "",
              "| Condition | Bridge Jaccard | DE Jaccard | Difference | exact permutation p |",
              "|---|---:|---:|---:|---:|"]
    for row in phase.itertuples():
        lines.append(f"| {row.condition} | {row.bridge_mean_pairwise_top25_jaccard:.3f} "
                     f"| {row.de_mean_pairwise_top25_jaccard:.3f} | {row.bridge_minus_de:.3f} | {row.exact_p:.3f} |")
    lines += ["", "The saved muscle interpretation remains: Bridge Jaccard 0.714; four",
              "drugs recur across all three studies; universal target recurrence is",
              "primarily ERBB2 and PDE5A. Removing ERBB2 lowers Jaccard to 0.500,",
              "removing PDE5A lowers it to 0.619, and PDE10A supplies only",
              "spaceflight-specific ibudilast support. Corrected GO/KEGG enrichment",
              "of the strict two-gene recurrent set is not significant. This is a",
              "low-dimensional target recurrence amplified by ChEMBL connectivity,",
              "not evidence of therapeutic efficacy.", "",
              "## Overall conclusion", "",
              "The three evaluations answer different questions and must not be pooled.",
              "Disease-gene recovery compares representation-derived modules with known",
              "biology and favors Bridge only in SLE at the primary endpoint; published-",
              "drug recovery is strongly limited by non-equivalent",
              "DrugBank/ChEMBL coverage; cross-study Jaccard tests reproducibility across",
              "space-condition cohorts. None is a direction-aware reversal test or",
              "evidence that an enriched drug is therapeutically effective.", ""]
    (RESULTS / "integrated_benchmark_summary.md").write_text("\n".join(lines))

    # Compact result figure.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    x = np.arange(3)
    width = 0.34
    for offset, method, color in [(-width / 2, "Bridge", "#2b6cb0"), (width / 2, "DE", "#b7791f")]:
        values = [primary[(primary.accession.eq(a)) & primary.method.eq(method)].odds_ratio.iloc[0]
                  for a in ["GSE138614", "GSE101794", "GSE72509"]]
        ax.bar(x + offset, values, width, label=method, color=color)
    published = [primary[(primary.accession.eq(a)) & primary.method.eq("Bridge")].deweerd_vae_odds_ratio.iloc[0]
                 for a in ["GSE138614", "GSE101794", "GSE72509"]]
    ax.scatter(x, published, marker="D", color="#333333", label="Published VAE*", zorder=3)
    ax.axhline(1, color="#777777", lw=1, ls="--")
    ax.set_xticks(x, ["MS", "Crohn", "SLE"])
    ax.set_ylabel("Disease-gene Fisher odds ratio")
    ax.set_title("Top-500 disease-gene recovery")
    ax.legend(frameon=False, fontsize=8)
    ax.text(0.01, -0.20, "*Published VAE uses a non-reconstructable filtered reference.",
            transform=ax.transAxes, fontsize=7)

    ax = axes[1]
    present = drugs[(drugs.method.eq("Bridge")) & drugs.present_in_chembl_mechanism_resource]
    total = drugs[drugs.method.eq("Bridge")].groupby("accession").size().reindex(ACCESSIONS := ["GSE138614", "GSE101794", "GSE72509"])
    available = present.groupby("accession").size().reindex(ACCESSIONS, fill_value=0)
    eligible3 = present.groupby("accession").sensitivity_ge3_rank.apply(lambda x: x.notna().sum()).reindex(ACCESSIONS, fill_value=0)
    ax.bar(x, total, color="#dddddd", label="Published top-5")
    ax.bar(x, available, color="#718096", label="Present in ChEMBL graph")
    ax.bar(x, eligible3, color="#2b6cb0", label="Eligible at ≥3 targets")
    ax.set_xticks(x, ["MS", "Crohn", "SLE"])
    ax.set_ylim(0, 5.5)
    ax.set_ylabel("Number of five highlighted drugs")
    ax.set_title("Reference-drug coverage, not efficacy")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "integrated_benchmark_summary.png", dpi=300, bbox_inches="tight")
    fig.savefig(RESULTS / "integrated_benchmark_summary.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
