#!/usr/bin/env python3
"""Evaluation 2 only: broaden ChEMBL 37 without changing frozen gene modules.

The universe definition is outcome-blind. It includes clinical-stage parent drugs
(ChEMBL max_phase >= 2; excluding cells, genes, and vaccine components) and human
single-protein targets supported either by a curated drug mechanism or by a
high-confidence binding assay (confidence 9, exact/upper-bound pChEMBL >= 6,
valid and non-duplicate). Reference-drug names are resolved only after the edge
universe is built, using exact normalized preferred names and ChEMBL synonyms.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests


HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"
OUT = RESULTS / "evaluation2_expanded"
DB = HERE / "work/chembl37_sqlite/chembl_37/chembl_37_sqlite/chembl_37.db"
CANONICAL = Path("/home/walt/bridge-rna/data/ensembl/canonical_genes.csv")
MODULES = RESULTS / "evaluation/gene_modules.parquet"
DIRECT_EDGES = HERE.parent / "drug_discovery/results/chembl37_direct_human_mechanism_edges.csv.gz"
ARCHIVE = HERE.parent / "drug_discovery/work/chembl/chembl_37_sqlite.tar.gz"

ACCESSIONS = ["GSE138614", "GSE101794", "GSE72509"]
LABELS = {
    "GSE138614": "Multiple sclerosis — active lesion",
    "GSE101794": "Crohn's disease — ileum",
    "GSE72509": "Systemic lupus erythematosus — blood",
}
REFERENCE_DRUGS = {
    "GSE138614": ["muromonab", "ibrutinib", "daclizumab", "zanubrutinib", "alemtuzumab"],
    "GSE101794": ["zinc", "zinc acetate", "dilmapimod", "glucosamine", "VX-702"],
    "GSE72509": ["gemcitabine", "enzastaurin", "sunitinib", "fostamatinib", "cladribine"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_name(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def fetch_edges(connection: sqlite3.Connection, canonical: set[str]):
    connection.executescript("""
        CREATE TEMP TABLE clinical_parents AS
        SELECT DISTINCT COALESCE(mh.parent_molregno, md.molregno) AS parent_molregno
        FROM molecule_dictionary md
        LEFT JOIN molecule_hierarchy mh ON mh.molregno = md.molregno
        WHERE md.max_phase >= 2
          AND md.molecule_type NOT IN ('Cell', 'Gene', 'Vaccine component');
        CREATE UNIQUE INDEX temp.idx_clinical_parent ON clinical_parents(parent_molregno);

        CREATE TEMP TABLE drug_members AS
        SELECT md.molregno,
               COALESCE(mh.parent_molregno, md.molregno) AS parent_molregno
        FROM molecule_dictionary md
        LEFT JOIN molecule_hierarchy mh ON mh.molregno = md.molregno
        JOIN clinical_parents cp
          ON cp.parent_molregno = COALESCE(mh.parent_molregno, md.molregno);
        CREATE INDEX temp.idx_drug_members_mol ON drug_members(molregno);
        CREATE INDEX temp.idx_drug_members_parent ON drug_members(parent_molregno);

        CREATE TEMP TABLE component_genes AS
        SELECT DISTINCT cs.component_id, csyn.component_synonym AS target_gene,
                        cs.accession
        FROM component_sequences cs
        JOIN component_synonyms csyn ON csyn.component_id = cs.component_id
        WHERE cs.tax_id = 9606 AND csyn.syn_type = 'GENE_SYMBOL';
        CREATE INDEX temp.idx_component_genes ON component_genes(component_id);
    """)
    drug_sql = """
        SELECT cp.parent_molregno, p.chembl_id AS drug_id,
               COALESCE(p.pref_name, p.chembl_id) AS drug_name,
               p.molecule_type, p.max_phase
        FROM clinical_parents cp
        JOIN molecule_dictionary p ON p.molregno = cp.parent_molregno
    """
    drugs = pd.read_sql_query(drug_sql, connection)

    mechanism_sql = """
        SELECT DISTINCT dmemb.parent_molregno, cg.target_gene, cg.accession,
               'mechanism' AS evidence_type,
               mec.direct_interaction, mec.action_type, NULL AS pchembl_value
        FROM drug_mechanism mec
        JOIN drug_members dmemb ON dmemb.molregno = mec.molregno
        JOIN target_dictionary td ON td.tid = mec.tid
        JOIN target_components tc ON tc.tid = td.tid
        JOIN component_genes cg ON cg.component_id = tc.component_id
        WHERE td.tax_id = 9606 AND td.target_type LIKE '%PROTEIN%'
    """
    mechanisms = pd.read_sql_query(mechanism_sql, connection)

    activity_sql = """
        SELECT DISTINCT dmemb.parent_molregno, cg.target_gene, cg.accession,
               'high_confidence_binding' AS evidence_type,
               NULL AS direct_interaction, act.action_type, act.pchembl_value
        FROM activities act
        JOIN drug_members dmemb ON dmemb.molregno = act.molregno
        JOIN assays ass ON ass.assay_id = act.assay_id
        JOIN target_dictionary td ON td.tid = ass.tid
        JOIN target_components tc ON tc.tid = td.tid
        JOIN component_genes cg ON cg.component_id = tc.component_id
        WHERE act.pchembl_value >= 6
          AND act.standard_relation IN ('=', '<', '<=')
          AND act.standard_flag = 1
          AND act.data_validity_comment IS NULL
          AND COALESCE(act.potential_duplicate, 0) = 0
          AND ass.assay_type = 'B'
          AND ass.confidence_score = 9
          AND td.tax_id = 9606
          AND td.target_type = 'SINGLE PROTEIN'
    """
    activities = pd.read_sql_query(activity_sql, connection)
    evidence = pd.concat([mechanisms, activities], ignore_index=True)
    evidence = evidence[evidence.target_gene.isin(canonical)].copy()
    evidence = evidence.merge(drugs, on="parent_molregno", how="inner")
    edges = (evidence.sort_values(["drug_id", "target_gene", "evidence_type"])
             .drop_duplicates(["drug_id", "target_gene"])
             .reset_index(drop=True))
    return drugs, evidence, edges


def aliases(connection: sqlite3.Connection, drugs: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_sql_query("""
        SELECT cp.parent_molregno, p.pref_name AS alias, 'PREF_NAME' AS alias_type
        FROM clinical_parents cp JOIN molecule_dictionary p ON p.molregno=cp.parent_molregno
        UNION ALL
        SELECT DISTINCT dmemb.parent_molregno, ms.synonyms AS alias, ms.syn_type AS alias_type
        FROM drug_members dmemb JOIN molecule_synonyms ms ON ms.molregno=dmemb.molregno
        WHERE ms.synonyms IS NOT NULL
    """, connection)
    frame = frame.merge(drugs[["parent_molregno", "drug_id", "drug_name"]], on="parent_molregno")
    frame["normalized_alias"] = frame.alias.map(normalize_name)
    return frame.drop_duplicates(["drug_id", "normalized_alias"])


def enrich(module: set[str], universe: set[str], edge_sets: pd.Series) -> pd.DataFrame:
    rows = []
    for (drug_id, drug_name), targets in edge_sets.items():
        targets = targets & universe
        overlap = module & targets
        n, K, N, k = len(universe), len(targets), len(module), len(overlap)
        p = float(hypergeom.sf(k - 1, n, K, N))
        non_targets = n - K
        non_module = n - N
        odds = (k * (non_targets - (N - k)) / ((N - k) * (K - k))) if (N-k) and (K-k) else (np.inf if k else np.nan)
        rows.append({"drug_id": drug_id, "drug_name": drug_name, "target_count": K,
                     "overlap_count": k, "overlap_genes": ";".join(sorted(overlap)),
                     "odds_ratio": odds, "pvalue": p})
    out = pd.DataFrame(rows)
    out["fdr"] = multipletests(out.pvalue, method="fdr_bh")[1]
    out["rank"] = out.pvalue.rank(method="min", ascending=True).astype(int)
    return out.sort_values(["pvalue", "overlap_count", "drug_name"], ascending=[True, False, True])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    canonical = set(pd.read_csv(CANONICAL).gene_symbol.astype(str))
    connection = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    drugs, evidence, edges = fetch_edges(connection, canonical)
    alias_frame = aliases(connection, drugs)
    connection.close()

    evidence.to_parquet(OUT / "chembl37_broad_evidence.parquet", index=False)
    edge_columns = ["drug_id", "drug_name", "target_gene", "accession", "molecule_type", "max_phase"]
    edges[edge_columns].to_csv(OUT / "chembl37_broad_drug_target_edges.csv.gz", index=False, compression="gzip")
    counts = edges.groupby(["drug_id", "drug_name"]).target_gene.nunique().rename("target_count").reset_index()
    counts.to_csv(OUT / "chembl37_broad_drug_target_counts.csv", index=False)

    modules = pd.read_parquet(MODULES)
    modules = modules[modules.module_size.eq(500)]
    edge_sets_all = edges.groupby(["drug_id", "drug_name"]).target_gene.agg(set)
    eligible_sets = edge_sets_all[edge_sets_all.map(len).gt(10)]
    enrichment = []
    audit_enrichment = []
    for accession in ACCESSIONS:
        for method in ["Bridge", "DE"]:
            module = set(modules[(modules.accession.eq(accession)) & modules.method.eq(method)].gene)
            result = enrich(module, canonical, eligible_sets)
            result.insert(0, "method", method)
            result.insert(0, "disease", LABELS[accession])
            result.insert(0, "accession", accession)
            enrichment.append(result)
            audit_result = enrich(module, canonical, edge_sets_all)
            audit_result.insert(0, "method", method)
            audit_result.insert(0, "disease", LABELS[accession])
            audit_result.insert(0, "accession", accession)
            audit_enrichment.append(audit_result)
    enrichment = pd.concat(enrichment, ignore_index=True)
    audit_enrichment = pd.concat(audit_enrichment, ignore_index=True)
    enrichment.to_parquet(OUT / "drug_enrichment_gt10.parquet", index=False)
    enrichment.to_csv(OUT / "drug_enrichment_gt10.csv.gz", index=False, compression="gzip")
    audit_enrichment.to_parquet(OUT / "drug_enrichment_all_audit.parquet", index=False)
    top = enrichment[enrichment.overlap_count.gt(0)].groupby(["accession", "method"], group_keys=False).head(25)
    top.to_csv(OUT / "top_independent_drugs.csv", index=False)

    # Audit names only after universe construction; no fuzzy matching or exceptions.
    alias_groups = alias_frame.groupby("normalized_alias").drug_id.agg(lambda x: sorted(set(x)))
    drug_lookup = drugs.drop_duplicates("drug_id").set_index("drug_id")
    audit = []
    for accession in ACCESSIONS:
        for reported_rank, requested in enumerate(REFERENCE_DRUGS[accession], 1):
            ids = alias_groups.get(normalize_name(requested), [])
            ids_with_edges = [drug_id for drug_id in ids if drug_id in edge_sets_all.index.get_level_values(0)]
            # Exact aliases that resolve to multiple ChEMBL parents are explicitly ambiguous.
            resolved = ids_with_edges[0] if len(ids_with_edges) == 1 else None
            for method in ["Bridge", "DE"]:
                base = {"accession": accession, "disease": LABELS[accession], "method": method,
                        "reference_drug": requested, "deweerd_reported_rank": reported_rank,
                        "exact_alias_candidate_ids": ";".join(ids),
                        "present_in_chembl_clinical_scope": len(ids) > 0,
                        "represented_in_universe": resolved is not None,
                        "ambiguous_exact_alias": len(ids_with_edges) > 1}
                if resolved is None:
                    audit.append({**base, "drug_id": None, "drug_name": None, "target_count": 0,
                                  "eligible_gt10": False, "rank": np.nan, "overlap_genes": "",
                                  "pvalue": np.nan, "fdr": np.nan, "audit_all_rank": np.nan,
                                  "audit_all_pvalue": np.nan, "audit_all_fdr": np.nan})
                    continue
                info = drug_lookup.loc[resolved]
                target_count = len(edge_sets_all.loc[(resolved, info.drug_name)])
                hit = enrichment[(enrichment.accession.eq(accession)) & enrichment.method.eq(method)
                                 & enrichment.drug_id.eq(resolved)]
                audit_hit = audit_enrichment[(audit_enrichment.accession.eq(accession))
                                             & audit_enrichment.method.eq(method)
                                             & audit_enrichment.drug_id.eq(resolved)].iloc[0]
                if hit.empty:
                    audit.append({**base, "drug_id": resolved, "drug_name": info.drug_name,
                                  "target_count": target_count, "eligible_gt10": False,
                                  "rank": np.nan, "overlap_genes": audit_hit.overlap_genes,
                                  "pvalue": np.nan, "fdr": np.nan,
                                  "audit_all_rank": int(audit_hit["rank"]),
                                  "audit_all_pvalue": audit_hit.pvalue,
                                  "audit_all_fdr": audit_hit.fdr})
                else:
                    row = hit.iloc[0]
                    audit.append({**base, "drug_id": resolved, "drug_name": info.drug_name,
                                  "target_count": target_count, "eligible_gt10": True,
                                  "rank": int(row["rank"]), "overlap_genes": row.overlap_genes,
                                  "pvalue": row.pvalue, "fdr": row.fdr,
                                  "audit_all_rank": int(audit_hit["rank"]),
                                  "audit_all_pvalue": audit_hit.pvalue,
                                  "audit_all_fdr": audit_hit.fdr})
    audit = pd.DataFrame(audit)
    audit.to_csv(OUT / "published_drug_recovery.csv", index=False)

    # Paired rank-percentile comparison on reference drugs eligible under the
    # prespecified >10-target rule. Positive values favor Bridge.
    comparisons = []
    n_tested = len(eligible_sets)
    for accession in ACCESSIONS + ["ALL"]:
        subset = audit[audit.eligible_gt10]
        if accession != "ALL":
            subset = subset[subset.accession.eq(accession)]
        paired = subset.pivot(index=["accession", "reference_drug"], columns="method", values="rank").dropna()
        if {"Bridge", "DE"}.issubset(paired.columns):
            diffs = ((n_tested - paired["Bridge"]) / max(n_tested - 1, 1)
                     - (n_tested - paired["DE"]) / max(n_tested - 1, 1)).to_numpy()
        else:
            diffs = np.array([])
        if len(diffs):
            observed = float(np.mean(diffs))
            null = []
            for mask in range(1 << len(diffs)):
                signs = np.array([1 if mask & (1 << i) else -1 for i in range(len(diffs))])
                null.append(float(np.mean(diffs * signs)))
            pvalue = float(np.mean(np.abs(null) >= abs(observed) - 1e-15))
        else:
            observed = pvalue = np.nan
        comparisons.append({"accession": accession, "disease": LABELS.get(accession, "Pooled"),
                            "eligible_reference_drugs": len(diffs),
                            "mean_bridge_minus_de_rank_percentile": observed,
                            "exact_two_sided_sign_flip_pvalue": pvalue})
    pd.DataFrame(comparisons).to_csv(OUT / "published_drug_method_comparison.csv", index=False)

    # Coverage comparison to the frozen direct-mechanism graph.
    direct = pd.read_csv(DIRECT_EDGES)
    direct_counts = direct.groupby(["drug_id", "drug_name"]).target_gene.nunique()
    quantiles = counts.target_count.quantile([0, .25, .5, .75, .9, .95, .99, 1]).to_dict()
    qc = {
        "resource": "ChEMBL 37 SQLite",
        "selection_frozen_before_reference_drug_audit": True,
        "inclusion_rules": {
            "drug": "parent-normalized ChEMBL molecule with max_phase >=2; exclude Cell, Gene, Vaccine component",
            "target": "human protein component with ChEMBL GENE_SYMBOL mapping present in frozen 15,165-gene canonical universe",
            "mechanism_evidence": "all curated drug_mechanism records mapped through human protein target components (single proteins, complexes, complex groups, families, and protein interactions); direct_interaction not required",
            "bioactivity_evidence": "binding assay, confidence_score=9, pChEMBL>=6, relation =/< /<=, standard_flag=1, no validity comment, not potential duplicate",
            "deduplication": "parent ChEMBL molecule x HGNC gene",
            "eligibility": ">10 targets before disease-module testing",
            "fisher_background": "frozen 15,165 canonical genes, identical for Bridge and DE",
        },
        "broad_universe": {"drugs_with_edges": int(counts.shape[0]), "targets": int(edges.target_gene.nunique()),
                           "edges": int(edges[["drug_id", "target_gene"]].drop_duplicates().shape[0]),
                           "drugs_gt10_targets": int(counts.target_count.gt(10).sum()),
                           "target_count_quantiles": {str(k): float(v) for k, v in quantiles.items()}},
        "evidence": {"mechanism_rows": int(evidence.evidence_type.eq("mechanism").sum()),
                     "binding_rows": int(evidence.evidence_type.eq("high_confidence_binding").sum())},
        "frozen_direct_comparator": {"drugs": int(direct_counts.shape[0]), "targets": int(direct.target_gene.nunique()),
                                     "edges": int(direct.shape[0]), "drugs_gt10_targets": int(direct_counts.gt(10).sum())},
        "deweerd_drugbank": {"reported_tested_drugs": 328, "protein_background": 16600,
                             "eligibility": ">10 known targets"},
        "archive_sha256": sha256(ARCHIVE),
        "frozen_module_file_sha256": sha256(MODULES),
        "frozen_direct_edge_file_sha256": sha256(DIRECT_EDGES),
        "edge_file_sha256": sha256(OUT / "chembl37_broad_drug_target_edges.csv.gz"),
        "direction_aware_reversal": False,
    }
    (OUT / "drug_target_universe_qc.json").write_text(json.dumps(qc, indent=2) + "\n")


if __name__ == "__main__":
    main()
