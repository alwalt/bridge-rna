#!/usr/bin/env python3
"""Audit frozen expanded-ChEMBL pharmacology for fixed Bridge radiation sets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parents[1]
CONTEXT = HERE / "results/radiation_context_sensitivity"
OUT = HERE / "results/radiation_pharmacology"
DEWEERD = HERE.parent / "deweerd_replication/results/evaluation2_expanded"
CANONICAL = Path("/home/walt/bridge-rna/data/ensembl/canonical_genes.csv")
OSD = ["GSE297560_gamma", "GSE297560_proton", "GSE297560_iron", "GSE297560_silicon"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def enrichment(module: set[str], universe: set[str], drugs: pd.Series) -> pd.DataFrame:
    rows = []
    module = module & universe
    for (drug_id, drug_name), targets in drugs.items():
        targets = targets & universe
        overlap = module & targets
        m, n, k, x = len(universe), len(module), len(targets), len(overlap)
        pvalue = float(hypergeom.sf(x - 1, m, k, n))
        denominator = (n - x) * (k - x)
        odds = x * (m - n - k + x) / denominator if denominator else (np.inf if x else np.nan)
        rows.append({"drug_id": drug_id, "drug_name": drug_name, "target_count": k,
                     "overlap_count": x, "supporting_genes": ";".join(sorted(overlap)),
                     "odds_ratio": odds, "pvalue": pvalue})
    result = pd.DataFrame(rows)
    result["fdr"] = multipletests(result.pvalue, method="fdr_bh")[1]
    result = result.sort_values(["pvalue", "overlap_count", "drug_name"],
                                ascending=[True, False, True]).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    return result


def mechanism_text(group: pd.DataFrame) -> str:
    rows = []
    for row in group.drop_duplicates(["evidence_type", "action_type"]).itertuples():
        action = row.action_type if pd.notna(row.action_type) and str(row.action_type).strip() else "unspecified action"
        if row.evidence_type == "mechanism":
            rows.append(f"curated mechanism: {action}")
        else:
            rows.append(f"high-confidence binding: {action}")
    return "; ".join(sorted(set(rows)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    canonical = set(pd.read_csv(CANONICAL).gene_symbol.astype(str))
    core = pd.read_csv(CONTEXT / "radiation_core_genes.csv")
    recurrent = pd.read_csv(CONTEXT / "recurrent_bridge_only_genes.csv")
    block = pd.read_csv(CONTEXT / "gse184119_frozen_block_rankings.csv.gz")
    bridge = pd.read_parquet(HERE / "results/bridge/contrast_gene_scores.parquet")
    edges_path = DEWEERD / "chembl37_broad_drug_target_edges.csv.gz"
    evidence_path = DEWEERD / "chembl37_broad_evidence.parquet"
    edges = pd.read_csv(edges_path)
    evidence = pd.read_parquet(evidence_path)

    all_drugs = edges.groupby(["drug_id", "drug_name"]).target_gene.agg(set)
    eligible = all_drugs[all_drugs.map(len).gt(10)]
    eligible_ids = {key[0] for key in eligible.index}
    targetable = set(edges.target_gene)

    strict = set(core[(core.method.eq("Bridge")) & core.classification.eq("strict_universal")].gene)
    cross = set(core[(core.method.eq("Bridge")) & core.classification.isin(
        ["strict_universal", "majority_cross_study_core"])].gene)
    all_three_bo = set(recurrent[recurrent.dataset_count.eq(3)].gene)

    nhdf = set(block[block.contrast_id.eq("GSE184119_NHDF_block")].gene)
    osd_sets = {}
    for contrast in OSD:
        frame = bridge[bridge.contrast_id.eq(contrast)].sort_values(
            ["absolute_bridge_score", "gene"], ascending=[False, True])
        osd_sets[contrast] = set(frame.head(500).gene)
    fibro = nhdf & set().union(*osd_sets.values())

    sets = {
        "strict_radiation_core": strict,
        "cross_study_radiation_core": cross,
        "all_three_study_bridge_only": all_three_bo,
        "matched_fibroblast_program": fibro,
    }

    # Radiation-context occurrence strings for every gene.
    occurrence = {}
    for row in core[core.method.eq("Bridge")].itertuples():
        occurrence[row.gene] = str(row.contrasts)
    for gene in fibro:
        contexts = ["GSE184119_NHDF_block"] + [contrast for contrast, genes in osd_sets.items() if gene in genes]
        occurrence[gene] = ";".join(contexts)

    # Collapse ChEMBL evidence by drug-gene without inventing action direction.
    mechanism = (evidence[evidence.drug_id.isin(eligible_ids)]
                 .groupby(["drug_id", "drug_name", "target_gene"], as_index=False)
                 .apply(mechanism_text, include_groups=False)
                 .rename(columns={None: "mechanism_action"}))
    if "mechanism_action" not in mechanism:
        mechanism = mechanism.rename(columns={0: "mechanism_action"})

    enrichment_parts = []
    gene_drug_parts = []
    set_summary = []
    for set_name, genes in sets.items():
        enriched = enrichment(genes, canonical, eligible)
        enriched.insert(0, "gene_set", set_name)
        enrichment_parts.append(enriched)
        nonzero = enriched[enriched.overlap_count.gt(0)].copy()
        for row in nonzero.itertuples():
            for gene in str(row.supporting_genes).split(";"):
                gene_drug_parts.append({
                    "gene_set": set_name,
                    "bridge_gene": gene,
                    "drug_id": row.drug_id,
                    "drug_name": row.drug_name,
                    "drug_target_count": row.target_count,
                    "drug_overlap_count": row.overlap_count,
                    "pvalue": row.pvalue,
                    "fdr": row.fdr,
                    "rank": row.rank,
                    "radiation_contrasts": occurrence.get(gene, ""),
                    "radiation_studies": ";".join(sorted({x.split("_")[0] for x in occurrence.get(gene, "").split(";") if x})),
                })
        set_summary.append({
            "gene_set": set_name,
            "gene_count": len(genes),
            "chembl_targetable_gene_count": len(genes & targetable),
            "eligible_connected_gene_count": len({g for targets in eligible for g in (genes & targets)}),
            "eligible_drugs_with_overlap": len(nonzero),
            "multi_gene_drugs": int(nonzero.overlap_count.ge(2).sum()),
            "fdr_lt_0_05": int(nonzero.fdr.lt(.05).sum()),
            "minimum_pvalue": nonzero.pvalue.min() if len(nonzero) else np.nan,
            "minimum_fdr": nonzero.fdr.min() if len(nonzero) else np.nan,
        })

    all_enrichment = pd.concat(enrichment_parts, ignore_index=True)
    all_enrichment.to_csv(OUT / "drug_enrichment_all.csv.gz", index=False, compression="gzip")
    all_enrichment[all_enrichment.overlap_count.gt(0)].to_csv(
        OUT / "drug_enrichment_nonzero.csv", index=False)
    gene_drug = pd.DataFrame(gene_drug_parts).merge(
        mechanism, left_on=["drug_id", "drug_name", "bridge_gene"],
        right_on=["drug_id", "drug_name", "target_gene"], how="left").drop(columns="target_gene")
    gene_drug["mechanism_action"] = gene_drug.mechanism_action.fillna("expanded ChEMBL edge; action not specified")
    gene_drug.to_csv(OUT / "bridge_gene_drug_mechanism_mapping.csv.gz", index=False, compression="gzip")

    # One drug row with exact support and mechanisms per gene set.
    drug_rows = []
    for (set_name, drug_id, drug_name), group in gene_drug.groupby(["gene_set", "drug_id", "drug_name"]):
        first = group.iloc[0]
        drug_rows.append({
            "gene_set": set_name, "drug_id": drug_id, "drug_name": drug_name,
            "support_type": "multiple recurrent targets" if len(group) > 1 else "single recurrent target",
            "supporting_genes": ";".join(sorted(group.bridge_gene.unique())),
            "supporting_gene_mechanisms": " | ".join(
                f"{row.bridge_gene}: {row.mechanism_action}" for row in group.sort_values("bridge_gene").itertuples()),
            "supporting_datasets": ";".join(sorted(set(";".join(group.radiation_studies).split(";")) - {""})),
            "target_count": first.drug_target_count, "overlap_count": first.drug_overlap_count,
            "pvalue": first.pvalue, "fdr": first.fdr, "rank": int(first["rank"]),
        })
    pd.DataFrame(drug_rows).sort_values(["gene_set", "rank"]).to_csv(
        OUT / "drug_support_compact.csv", index=False)
    pd.DataFrame(set_summary).to_csv(OUT / "gene_set_summary.csv", index=False)

    # Save exact gene memberships, including non-targetable genes.
    membership_rows = []
    for set_name, genes in sets.items():
        for gene in sorted(genes):
            membership_rows.append({"gene_set": set_name, "gene": gene,
                                    "chembl_targetable": gene in targetable,
                                    "eligible_drug_connected": any(gene in targets for targets in eligible),
                                    "radiation_contrasts": occurrence.get(gene, "")})
    pd.DataFrame(membership_rows).to_csv(OUT / "fixed_gene_set_membership.csv.gz", index=False, compression="gzip")

    provenance = {
        "analysis": "interpretation/audit of fixed Bridge radiation gene sets and frozen expanded ChEMBL 37",
        "definitions": {
            "strict_radiation_core": "15 genes present in all seven frozen Bridge top-500 modules",
            "cross_study_radiation_core": "strict plus majority-cross-study classification; 237 genes",
            "all_three_study_bridge_only": "Bridge-only genes with dataset_count == 3; 84 genes",
            "matched_fibroblast_program": "frozen NHDF-block top-500 intersected with the union of four frozen OSD-993 fibroblast top-500 modules",
        },
        "drug_eligibility": ">10 canonical human targets in frozen expanded ChEMBL 37 graph",
        "universe": "15,165 canonical genes",
        "statistics": "right-sided hypergeometric; BH FDR separately within each gene set over all 366 eligible drugs",
        "input_hashes": {
            "expanded_edges": sha256(edges_path), "expanded_evidence": sha256(evidence_path),
            "radiation_core": sha256(CONTEXT / "radiation_core_genes.csv"),
            "recurrent_bridge_only": sha256(CONTEXT / "recurrent_bridge_only_genes.csv"),
            "gse184119_blocks": sha256(CONTEXT / "gse184119_frozen_block_rankings.csv.gz"),
            "bridge_scores": sha256(HERE / "results/bridge/contrast_gene_scores.parquet"),
        },
        "protected_steps_rerun": {"Bridge": False, "DE": False, "attribution": False,
                                  "gene_modules": False, "expanded_ChEMBL_construction": False,
                                  "previous_benchmarks": False},
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
