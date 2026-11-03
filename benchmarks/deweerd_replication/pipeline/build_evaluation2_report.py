#!/usr/bin/env python3
"""Render the expanded-ChEMBL Evaluation 2 report from frozen outputs."""

from pathlib import Path
import json
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "results/evaluation2_expanded"


def fmt(value):
    return "—" if pd.isna(value) else f"{value:.3g}"


def rank(value):
    return "ineligible" if pd.isna(value) else str(int(value))


def genes(value):
    return "—" if pd.isna(value) or not str(value).strip() else str(value)


def main():
    qc = json.loads((OUT / "drug_target_universe_qc.json").read_text())
    audit = pd.read_csv(OUT / "published_drug_recovery.csv")
    comparison = pd.read_csv(OUT / "published_drug_method_comparison.csv")
    top = pd.read_csv(OUT / "top_independent_drugs.csv")
    enrichment = pd.read_parquet(OUT / "drug_enrichment_gt10.parquet")
    broad, direct, published = qc["broad_universe"], qc["frozen_direct_comparator"], qc["deweerd_drugbank"]

    lines = [
        "# Evaluation 2 — published and independent drug recovery", "",
        "This evaluation reuses the frozen top-500 Bridge and DE modules. It did not",
        "rerun attribution, differential expression, Evaluation 1, or the Phase 2",
        "space-condition benchmark.", "",
        "## Final drug-target universe", "",
        "The final approximation uses ChEMBL 37 clinical-stage parent drugs",
        "(`max_phase >= 2`; cells, genes, and vaccine components excluded). Targets",
        "are human protein components with a canonical HGNC symbol. Evidence is either",
        "a curated ChEMBL drug mechanism or a high-confidence human single-protein",
        "binding assay (`confidence_score = 9`, pChEMBL ≥ 6, valid, non-duplicate).",
        "Complex and family mechanisms are expanded only through ChEMBL's explicit",
        "component membership. Drug–gene edges are parent-normalized and deduplicated.", "",
        f"- Broad graph: {broad['drugs_with_edges']:,} drugs, {broad['targets']:,} targets, {broad['edges']:,} edges.",
        f"- Strict endpoint: {broad['drugs_gt10_targets']:,} drugs with >10 targets.",
        f"- Restrictive Phase 2 comparator: {direct['drugs']:,} drugs, {direct['targets']:,} targets, {direct['edges']:,} edges; {direct['drugs_gt10_targets']} drug with >10 targets.",
        f"- de Weerd DrugBank setup: {published['reported_tested_drugs']} tested drugs and {published['protein_background']:,}-protein background.",
        "- Fisher background here: the fixed 15,165 canonical genes, identical for",
        "  Bridge and DE; BH correction is performed separately within each",
        "  disease × method family of 366 eligible drugs.", "",
        "The 366-drug eligible set is much closer in scale to DrugBank's 328 than the",
        "restrictive graph, but it is not an exact replication: ChEMBL bioactivity and",
        "component-expanded mechanisms are not equivalent to DrugBank target curation,",
        "and the gene backgrounds differ.", "",
        "Target-count quantiles (all drugs with edges): " + ", ".join(
            f"{float(k)*100:.0f}%={v:g}" for k, v in broad["target_count_quantiles"].items()), "",
        "## Published drugs", "",
        "`Strict rank/p/q` applies only to >10-target drugs. `Audit rank/p/q` includes",
        "all drugs with at least one mapped target and is descriptive only. A ChEMBL",
        "entity without a qualifying canonical protein edge is not counted as a Bridge",
        "failure.", "",
        "| Disease | de Weerd drug (#) | Target status | Targets | Method | Strict rank | Overlap | Strict p/q | Audit rank/p/q |",
        "|---|---|---|---:|---|---:|---|---:|---:|",
    ]
    for (disease, drug), group in audit.groupby(["disease", "reference_drug"], sort=False):
        for row in group.itertuples():
            if row.represented_in_universe:
                status = "eligible" if row.eligible_gt10 else "mapped, ≤10"
            elif row.present_in_chembl_clinical_scope:
                status = "entity, no eligible gene edge"
            else:
                status = "absent"
            lines.append(
                f"| {disease} | {drug} ({row.deweerd_reported_rank}) | {status} | {row.target_count} | {row.method} "
                f"| {rank(row.rank)} | {genes(row.overlap_genes)} | {fmt(row.pvalue)}/{fmt(row.fdr)} "
                f"| {rank(row.audit_all_rank)}/{fmt(row.audit_all_pvalue)}/{fmt(row.audit_all_fdr)} |"
            )

    present = audit[audit.method.eq("Bridge")]
    clinical_n = int(present.groupby("reference_drug").present_in_chembl_clinical_scope.max().sum())
    mapped_n = int(present.groupby("reference_drug").represented_in_universe.max().sum())
    eligible_n = int(present.groupby("reference_drug").eligible_gt10.max().sum())
    pooled = comparison[comparison.accession.eq("ALL")].iloc[0]
    lines += ["", "## Bridge versus DE", "",
              f"Coverage is {clinical_n}/15 as ChEMBL clinical entities, {mapped_n}/15 with",
              f"usable target edges, and {eligible_n}/15 at the strict >10-target endpoint.",
              "The five strictly eligible reference drugs are ibrutinib, zanubrutinib,",
              "gemcitabine, enzastaurin, and sunitinib.", "",
              f"Across those five drugs, mean Bridge-minus-DE rank percentile is {pooled.mean_bridge_minus_de_rank_percentile:.3f}",
              f"(exact paired sign-flip p={pooled.exact_two_sided_sign_flip_pvalue:.3f}).",
              "This is a negligible effect slightly favoring DE, with no evidence that",
              "Bridge preferentially recovers the published drug set. No highlighted",
              "drug is significant after FDR correction for either method.", "",
              "Disease-specific comparison:", "",
              "| Disease | Eligible reference drugs | Bridge−DE rank-percentile | Exact p |",
              "|---|---:|---:|---:|"]
    for row in comparison[~comparison.accession.eq("ALL")].itertuples():
        lines.append(f"| {row.disease} | {row.eligible_reference_drugs} | {fmt(row.mean_bridge_minus_de_rank_percentile)} | {fmt(row.exact_two_sided_sign_flip_pvalue)} |")

    lines += ["", "## Independently highest-ranked drugs", "",
              "These are enrichment rankings, not therapeutic recommendations or",
              "direction-aware reversal results.", "",
              "| Disease | Method | Top five drugs (raw p; FDR) |",
              "|---|---|---|"]
    for (disease, method), group in top.groupby(["disease", "method"], sort=False):
        items = [f"{r.drug_name} ({r.pvalue:.3g}; {r.fdr:.3g})" for r in group.head(5).itertuples()]
        lines.append(f"| {disease} | {method} | {'; '.join(items)} |")

    sig = enrichment[enrichment.fdr.lt(.05)]
    lines += ["", "## Statistical result", "",
              f"Of six disease × method families, {len(sig)} drug results pass FDR < 0.05."]
    if len(sig):
        for row in sig.itertuples():
            lines.append(f"- {row.disease}, {row.method}: {row.drug_name}, overlap {row.overlap_count}/{row.target_count}, raw p={row.pvalue:.3g}, FDR={row.fdr:.3g}.")
    lines += ["", "The sole FDR-significant result is not one of de Weerd's highlighted",
              "drugs. No Bridge drug enrichment passes FDR. Target enrichment does not",
              "establish drug efficacy, direction of effect, or expression reversal.", "",
              "## Exact-replication limitation", "",
              "No authorized DrugBank export is available. The redistributed object in",
              "the paper's code materials does not establish downstream data rights.",
              "Consequently this is a reproducible ChEMBL approximation, not an exact",
              "DrugBank replication. Exact replication requires an authorized versioned",
              "DrugBank export and the authors' precise drug/protein identifier mapping.", ""]
    (OUT / "evaluation2_report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
