#!/usr/bin/env python3
"""Evaluate disease-gene and published-drug recovery for locked cohorts."""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests


HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"
WORK = HERE / "work"
REFERENCE_DIR = WORK / "references"
CANONICAL = Path("/home/walt/bridge-rna/data/ensembl/canonical_genes.csv")
CHEMBL_EDGES = Path(__file__).resolve().parents[2] / "drug_discovery/results/chembl37_direct_human_mechanism_edges.csv.gz"
DISGENET_URL = "https://raw.githubusercontent.com/Uriyah3/proyecto-tesis/main/disgenet/all_gene_disease_associations.tsv.gz"
DISGENET_PATH = REFERENCE_DIR / "disgenet_v7_all_gene_disease_associations.tsv.gz"
ACCESSIONS = ["GSE138614", "GSE101794", "GSE72509"]
LABELS = {
    "GSE138614": "Multiple sclerosis — active lesion",
    "GSE101794": "Crohn's disease — ileum",
    "GSE72509": "Systemic lupus erythematosus — blood",
}
CUIS = {"GSE138614": "C0026769", "GSE101794": "C0010346", "GSE72509": "C0024141"}
PUBLISHED = {
    "GSE138614": {"reference_genes": 300, "recovered": 141, "odds_ratio": 10.028949, "pvalue": 3.343717e-69},
    "GSE101794": {"reference_genes": 168, "recovered": 61, "odds_ratio": 6.969721, "pvalue": 1.121464e-25},
    "GSE72509": {"reference_genes": 95, "recovered": 6, "odds_ratio": 0.670451, "pvalue": 0.8751437},
}
REFERENCE_DRUGS = {
    "GSE138614": ["MUROMONAB", "IBRUTINIB", "DACLIZUMAB", "ZANUBRUTINIB", "ALEMTUZUMAB"],
    "GSE101794": ["ZINC", "ZINC ACETATE", "DILMAPIMOD", "GLUCOSAMINE", "VX-702"],
    "GSE72509": ["GEMCITABINE", "ENZASTAURIN", "SUNITINIB", "FOSTAMATINIB", "CLADRIBINE"],
}
MODULE_SIZES = list(range(50, 501, 50))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_reference():
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    if not DISGENET_PATH.exists():
        cached = Path("/tmp/disgenet_v7_all.tsv.gz")
        if cached.exists():
            shutil.copy2(cached, DISGENET_PATH)
        else:
            urllib.request.urlretrieve(DISGENET_URL, DISGENET_PATH)


def fisher_summary(module, disease_genes, universe):
    module, disease_genes, universe = set(module), set(disease_genes), set(universe)
    module &= universe
    disease_genes &= universe
    a = len(module & disease_genes)
    b = len(module - disease_genes)
    c = len(disease_genes - module)
    d = len(universe - module - disease_genes)
    odds = (a * d / (b * c)) if b and c else (np.inf if a and d else np.nan)
    pvalue = float(hypergeom.sf(a - 1, len(universe), len(disease_genes), len(module)))
    return a, len(disease_genes), len(module), odds, pvalue


def drug_enrichment(module, universe, edges, minimum_targets):
    universe, module = set(universe), set(module) & set(universe)
    filtered = edges[edges.target_gene.isin(universe)]
    groups = filtered.groupby(["drug_id", "drug_name"]).target_gene.agg(lambda x: set(x))
    groups = groups[groups.map(len).ge(minimum_targets)]
    rows = []
    for (drug_id, drug_name), targets in groups.items():
        overlap = module & targets
        pvalue = float(hypergeom.sf(len(overlap) - 1, len(universe), len(targets), len(module)))
        rows.append({"drug_id": drug_id, "drug_name": drug_name, "target_count": len(targets),
                     "overlap_count": len(overlap), "overlap_genes": ";".join(sorted(overlap)),
                     "pvalue": pvalue})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["fdr"] = multipletests(frame.pvalue, method="fdr_bh")[1]
    frame = frame.sort_values(["fdr", "pvalue", "overlap_count", "drug_id"],
                              ascending=[True, True, False, True]).reset_index(drop=True)
    frame["rank"] = np.arange(1, len(frame) + 1)
    return frame


def main():
    fetch_reference()
    output = RESULTS / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    genes = pd.read_csv(CANONICAL).gene_symbol.astype(str).tolist()
    dgn = pd.read_csv(DISGENET_PATH, sep="\t", usecols=["geneSymbol", "diseaseId", "diseaseName", "score"])
    edges = pd.read_csv(CHEMBL_EDGES)

    rankings, universes, modules = {}, {}, {}
    disease_rows, module_rows = [], []
    for accession in ACCESSIONS:
        expression = np.load(WORK / f"prepared/{accession}_log1p_tpm.npy", mmap_mode="r")
        universes[accession] = [gene for gene, total in zip(genes, np.asarray(expression).sum(0)) if total > 0]
        disease_frame = dgn[dgn.diseaseId.eq(CUIS[accession])]
        disease_sets = {
            "all_v7_associations": set(disease_frame.geneSymbol.dropna().astype(str)),
            "v7_score_ge_0.1_sensitivity": set(disease_frame.loc[disease_frame.score.ge(0.1), "geneSymbol"].dropna().astype(str)),
        }
        for method, path, score_column in [
            ("Bridge", RESULTS / f"bridge/{accession}_gene_ranking.parquet", "signed_bridge_score"),
            ("DE", RESULTS / f"differential_expression/{accession}_gene_ranking.csv.gz", "t"),
        ]:
            ranking = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
            ranking = ranking[ranking.gene.isin(universes[accession])].sort_values("rank")
            rankings[(accession, method)] = ranking
            for size in MODULE_SIZES:
                module = ranking.head(size).gene.tolist()
                modules[(accession, method, size)] = module
                scores = ranking.head(size).set_index("gene")[score_column]
                for rank, gene in enumerate(module, 1):
                    module_rows.append({"accession": accession, "disease": LABELS[accession],
                                        "method": method, "module_size": size, "rank": rank,
                                        "gene": gene, "signed_score": float(scores[gene])})
                for reference_definition, disease in disease_sets.items():
                    recovered, reference_n, actual_n, odds, pvalue = fisher_summary(
                        module, disease, universes[accession])
                    disease_rows.append({"accession": accession, "disease": LABELS[accession],
                                         "method": method, "module_size": size,
                                         "reference_definition": reference_definition,
                                         "universe_genes": len(universes[accession]),
                                         "disgenet_cui": CUIS[accession], "disgenet_genes_in_universe": reference_n,
                                         "module_genes_in_universe": actual_n, "known_genes_recovered": recovered,
                                         "odds_ratio": odds, "pvalue": pvalue})
    disease_results = pd.DataFrame(disease_rows)
    # A single prespecified family per module size: 3 diseases x 2 methods.
    disease_results["fdr"] = np.nan
    for (_, size), indexes in disease_results.groupby(["reference_definition", "module_size"]).groups.items():
        disease_results.loc[indexes, "fdr"] = multipletests(
            disease_results.loc[indexes, "pvalue"], method="fdr_bh")[1]
    for accession in ACCESSIONS:
        mask = (disease_results.accession.eq(accession) & disease_results.module_size.eq(500)
                & disease_results.reference_definition.eq("all_v7_associations"))
        for key, value in PUBLISHED[accession].items():
            disease_results.loc[mask, f"deweerd_vae_{key}"] = value
    disease_results.to_csv(output / "disease_gene_recovery.csv", index=False)
    pd.DataFrame(module_rows).to_parquet(output / "gene_modules.parquet", index=False)

    enrichment_frames = []
    for accession in ACCESSIONS:
        for method in ("Bridge", "DE"):
            module = modules[(accession, method, 500)]
            for threshold, label in [(11, "deweerd_gt10_analogue"), (3, "chembl_contract_sensitivity"),
                                     (1, "reference_drug_audit_only")]:
                frame = drug_enrichment(module, universes[accession], edges, threshold)
                if not frame.empty:
                    frame.insert(0, "minimum_targets", threshold)
                    frame.insert(0, "drug_endpoint", label)
                    frame.insert(0, "method", method)
                    frame.insert(0, "disease", LABELS[accession])
                    frame.insert(0, "accession", accession)
                    enrichment_frames.append(frame)
    drug_results = pd.concat(enrichment_frames, ignore_index=True) if enrichment_frames else pd.DataFrame()
    drug_results.to_parquet(output / "drug_enrichment.parquet", index=False)
    drug_results.to_csv(output / "drug_enrichment.csv.gz", index=False, compression="gzip")
    top = drug_results[(drug_results.minimum_targets.eq(3)) & drug_results.overlap_count.gt(0)]
    top.groupby(["accession", "method"], group_keys=False).head(25).to_csv(
        output / "top_independent_drugs.csv", index=False)

    # Reference-drug audit: strict endpoint rank and the prespecified >=3 ChEMBL
    # sensitivity rank. Below-threshold compounds remain explicitly excluded.
    edge_groups = edges.groupby(["drug_id", "drug_name"]).target_gene.agg(lambda x: set(x))
    audit_rows = []
    for accession in ACCESSIONS:
        universe = set(universes[accession])
        name_groups = {name.upper(): (drug_id, targets & universe)
                       for (drug_id, name), targets in edge_groups.items()}
        for published_rank, requested in enumerate(REFERENCE_DRUGS[accession], 1):
            match = name_groups.get(requested)
            base = {"accession": accession, "disease": LABELS[accession],
                    "reference_drug": requested, "deweerd_reported_rank": published_rank,
                    "present_in_chembl_mechanism_resource": match is not None}
            if match is None:
                for method in ("Bridge", "DE"):
                    audit_rows.append({**base, "method": method, "drug_id": None,
                                       "target_count": 0, "eligible_gt10": False,
                                       "strict_rank": np.nan, "sensitivity_ge3_rank": np.nan,
                                       "overlap_genes": "", "pvalue": np.nan, "fdr": np.nan})
                continue
            drug_id, targets = match
            target_count = len(targets)
            for method in ("Bridge", "DE"):
                strict = drug_results[(drug_results.accession.eq(accession)) &
                                      (drug_results.method.eq(method)) &
                                      (drug_results.minimum_targets.eq(11)) &
                                      (drug_results.drug_id.eq(drug_id))]
                sensitivity = drug_results[(drug_results.accession.eq(accession)) &
                                           (drug_results.method.eq(method)) &
                                           (drug_results.minimum_targets.eq(3)) &
                                           (drug_results.drug_id.eq(drug_id))]
                audit_only = drug_results[(drug_results.accession.eq(accession)) &
                                          (drug_results.method.eq(method)) &
                                          (drug_results.minimum_targets.eq(1)) &
                                          (drug_results.drug_id.eq(drug_id))]
                chosen = strict if not strict.empty else sensitivity if not sensitivity.empty else audit_only
                audit_rows.append({**base, "method": method, "drug_id": drug_id,
                                   "target_count": target_count, "eligible_gt10": target_count > 10,
                                   "strict_rank": strict.iloc[0]["rank"] if not strict.empty else np.nan,
                                   "sensitivity_ge3_rank": sensitivity.iloc[0]["rank"] if not sensitivity.empty else np.nan,
                                   "audit_ge1_rank": audit_only.iloc[0]["rank"] if not audit_only.empty else np.nan,
                                   "overlap_genes": chosen.iloc[0].overlap_genes if not chosen.empty else "",
                                   "pvalue": chosen.iloc[0].pvalue if not chosen.empty else np.nan,
                                   "fdr": chosen.iloc[0].fdr if not chosen.empty else np.nan})
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(output / "published_drug_recovery.csv", index=False)

    counts = edges.groupby(["drug_id", "drug_name"]).target_gene.nunique()
    provenance = {
        "disgenet": {"version": "v7.0 (2020 public archive)", "source": DISGENET_URL,
                      "sha256": sha256(DISGENET_PATH), "disease_cuis": CUIS,
                      "association_scope": "all DisGeNET v7 associations for exact CUI; score >=0.1 sensitivity",
                      "comparability": "closest documented release; de Weerd's filtered gene lists were not published, so exact reference totals cannot be reconstructed"},
        "disease_gene_universe": "dataset-expressed canonical Bridge genes",
        "module_sizes": MODULE_SIZES, "primary_module_size": 500,
        "multiple_testing": "BH within each module-size family of six disease x method tests",
        "drug_resource": "frozen ChEMBL 37 approved small-molecule direct human mechanism targets from Phase 2",
        "drugbank_decision": "not used: no authorized local DrugBank license/export; the paper repository's redistributed R object does not establish downstream authorization",
        "drugbank_published_universe": {"protein_background": 16600, "minimum_targets": ">10", "tested_drugs_in_supplement_per_case": 328},
        "chembl_resource_coverage": {"drugs": int(len(counts)), "targets": int(edges.target_gene.nunique()),
                                     "edges": int(len(edges)), "drugs_gt10_targets": int((counts > 10).sum()),
                                     "drugs_ge3_targets": int((counts >= 3).sum())},
        "primary_drug_endpoint": "strict >10-target ChEMBL analogue; expected to be non-informative because only one resource drug qualifies",
        "secondary_drug_endpoint": "frozen >=3-target ChEMBL contract, clearly labeled sensitivity",
        "reference_drug_audit": "all >=1-target resource drugs tested solely to provide requested ranks/p/FDR for below-threshold highlighted drugs",
        "direction_aware_reversal": False,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
