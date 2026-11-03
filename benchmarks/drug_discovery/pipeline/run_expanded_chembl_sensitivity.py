#!/usr/bin/env python3
"""Expanded-ChEMBL drug convergence and Bridge/DE complementarity sensitivity.

Reads the frozen top-500 Phase 2 modules; it does not reconstruct any module or
rerun expression, representation, attribution, or differential expression.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"
OUT = RESULTS / "expanded_chembl_sensitivity"
MODULE_FILE = RESULTS / "evaluation/gene_modules.parquet"
EDGE_FILE = HERE.parent / "deweerd_replication/results/evaluation2_expanded/chembl37_broad_drug_target_edges.csv.gz"
EDGE_QC = HERE.parent / "deweerd_replication/results/evaluation2_expanded/drug_target_universe_qc.json"
DISGENET = HERE.parent / "deweerd_replication/work/references/disgenet_v7_all_gene_disease_associations.tsv.gz"
CANONICAL = Path("/home/walt/bridge-rna/data/ensembl/canonical_genes.csv")

ORDER = {
    "Radiation": ["GSE297090", "GSE184119", "GSE297560"],
    "Bone loss": ["GSE189524", "GSE276529", "GSE273868"],
    "Muscle atrophy": ["GSE211204", "GSE113165", "GSE234465"],
}
ROLE = {studies[0]: "terrestrial_discovery" for studies in ORDER.values()}
ROLE.update({studies[1]: "terrestrial_replication" for studies in ORDER.values()})
ROLE.update({studies[2]: "osdr_validation" for studies in ORDER.values()})
DISGENET_CUI = {"Radiation": "C3828416", "Bone loss": "C0029456", "Muscle atrophy": "C0541794"}
DISGENET_NAME = {"Radiation": "Radiation Damage", "Bone loss": "Osteoporosis", "Muscle atrophy": "Skeletal muscle atrophy"}
PARTITIONS = ["Bridge-only", "Shared", "DE-only"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def enrich(module: set[str], universe: set[str], targets: pd.Series) -> pd.DataFrame:
    rows = []
    module &= universe
    for (drug_id, drug_name), genes in targets.items():
        genes &= universe
        k, K, n, M = len(module & genes), len(genes), len(module), len(universe)
        p = float(hypergeom.sf(k - 1, M, K, n))
        d = M - n - K + k
        odds = (k * d / ((n - k) * (K - k))) if (n-k) and (K-k) else (np.inf if k else np.nan)
        rows.append({"drug_id": drug_id, "drug_name": drug_name, "target_count": K,
                     "overlap_count": k, "overlap_genes": ";".join(sorted(module & genes)),
                     "odds_ratio": odds, "pvalue": p})
    out = pd.DataFrame(rows)
    out["fdr"] = multipletests(out.pvalue, method="fdr_bh")[1]
    out = out.sort_values(["fdr", "pvalue", "odds_ratio", "drug_id"],
                          ascending=[True, True, False, True]).reset_index(drop=True)
    out["drug_rank"] = np.arange(1, len(out) + 1)
    return out


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(len(a | b), 1)


def condition_score(rankings, condition, assignment, top_n=25):
    values = []
    for a, b in itertools.combinations(ORDER[condition], 2):
        aa = rankings[(assignment[a], a)][:top_n]
        bb = rankings[(assignment[b], b)][:top_n]
        values.append(jaccard(aa, bb))
    return float(np.mean(values))


def fisher_overlap(module, reference, universe):
    module, reference = set(module) & universe, set(reference) & universe
    k, K, n, M = len(module & reference), len(reference), len(module), len(universe)
    p = float(hypergeom.sf(k - 1, M, K, n))
    d = M - n - K + k
    odds = (k * d / ((n-k) * (K-k))) if (n-k) and (K-k) else (np.inf if k else np.nan)
    return k, K, odds, p


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    canonical = set(pd.read_csv(CANONICAL).gene_symbol.astype(str))
    modules_frame = pd.read_parquet(MODULE_FILE)
    frozen = modules_frame[modules_frame.module_size.eq(500)].copy()
    assert frozen.groupby(["method", "dataset"]).size().eq(500).all()
    modules = {(m, d): set(g.gene) for (m, d), g in frozen.groupby(["method", "dataset"])}

    edges = pd.read_csv(EDGE_FILE)
    all_targets = edges.groupby(["drug_id", "drug_name"]).target_gene.agg(set)
    eligible_targets = all_targets[all_targets.map(len).gt(10)]
    assert len(eligible_targets) == 366

    # Drug enrichment and top-25 nonzero-overlap convergence.
    enrichment_parts, rankings = [], {}
    for condition, studies in ORDER.items():
        for dataset in studies:
            for method in ["Bridge", "DE"]:
                result = enrich(set(modules[(method, dataset)]), canonical, eligible_targets)
                result.insert(0, "role", ROLE[dataset])
                result.insert(0, "condition", condition)
                result.insert(0, "dataset", dataset)
                result.insert(0, "method", method)
                enrichment_parts.append(result)
                rankings[(method, dataset)] = result.loc[result.overlap_count.gt(0), "drug_id"].tolist()
    enrichment = pd.concat(enrichment_parts, ignore_index=True)
    enrichment.to_parquet(OUT / "expanded_drug_enrichment.parquet", index=False)
    enrichment.to_csv(OUT / "expanded_drug_enrichment.csv.gz", index=False, compression="gzip")

    convergence_rows = []
    for condition, studies in ORDER.items():
        for method in ["Bridge", "DE"]:
            for a, b in itertools.combinations(studies, 2):
                aa, bb = rankings[(method, a)][:25], rankings[(method, b)][:25]
                convergence_rows.append({"condition": condition, "method": method,
                                         "dataset_1": a, "dataset_2": b,
                                         "role_1": ROLE[a], "role_2": ROLE[b],
                                         "top_n": 25, "overlap": len(set(aa) & set(bb)),
                                         "jaccard": jaccard(aa, bb)})
            sets = [set(rankings[(method, d)][:25]) for d in studies]
            convergence_rows.append({"condition": condition, "method": method,
                                     "dataset_1": "three_way", "dataset_2": "three_way",
                                     "role_1": "mixed", "role_2": "mixed", "top_n": 25,
                                     "overlap": len(set.intersection(*sets)),
                                     "jaccard": len(set.intersection(*sets)) / max(len(set.union(*sets)), 1)})
    convergence = pd.DataFrame(convergence_rows)
    convergence.to_csv(OUT / "expanded_convergence.csv", index=False)

    # Exact within-dataset method-label permutation, matching Phase 2.
    effects, null_rows = [], []
    for condition, studies in ORDER.items():
        b = condition_score(rankings, condition, {s: "Bridge" for s in studies})
        d = condition_score(rankings, condition, {s: "DE" for s in studies})
        observed, null = b-d, []
        for bits in itertools.product([0, 1], repeat=3):
            amap = {s: ("DE" if bit else "Bridge") for s, bit in zip(studies, bits)}
            bmap = {s: ("Bridge" if bit else "DE") for s, bit in zip(studies, bits)}
            value = condition_score(rankings, condition, amap) - condition_score(rankings, condition, bmap)
            null.append(value)
            null_rows.append({"condition": condition, "mask": "".join(map(str, bits)), "difference": value})
        p = sum(abs(x) >= abs(observed)-1e-15 for x in null) / len(null)
        effects.append({"condition": condition, "bridge_mean_pairwise_top25_jaccard": b,
                        "de_mean_pairwise_top25_jaccard": d, "bridge_minus_de": observed,
                        "exact_p": p, "permutations": len(null)})
    pd.DataFrame(effects).to_csv(OUT / "expanded_primary_effects.csv", index=False)
    pd.DataFrame(null_rows).to_csv(OUT / "expanded_method_label_permutation_null.csv", index=False)

    original = pd.read_csv(RESULTS / "evaluation/primary_effects.csv")
    comparison = pd.DataFrame(effects).merge(original, on="condition", suffixes=("_expanded", "_restrictive"))
    for metric in ["bridge_mean_pairwise_top25_jaccard", "de_mean_pairwise_top25_jaccard", "bridge_minus_de"]:
        comparison[f"delta_{metric}"] = comparison[f"{metric}_expanded"] - comparison[f"{metric}_restrictive"]
    comparison.to_csv(OUT / "expanded_vs_restrictive.csv", index=False)

    # Frozen top-500 complementarity partitions.
    partition_rows = []
    partition_sets = {}
    for condition, studies in ORDER.items():
        for dataset in studies:
            bridge, de = modules[("Bridge", dataset)], modules[("DE", dataset)]
            values = {"Bridge-only": bridge-de, "Shared": bridge & de, "DE-only": de-bridge}
            for partition, genes in values.items():
                partition_sets[(condition, dataset, partition)] = genes
                for gene in sorted(genes):
                    br = frozen[(frozen.dataset.eq(dataset)) & frozen.method.eq("Bridge") & frozen.gene.eq(gene)]
                    dr = frozen[(frozen.dataset.eq(dataset)) & frozen.method.eq("DE") & frozen.gene.eq(gene)]
                    partition_rows.append({"condition": condition, "dataset": dataset, "role": ROLE[dataset],
                                           "partition": partition, "gene": gene,
                                           "bridge_rank": int(br.iloc[0]["rank"]) if len(br) else np.nan,
                                           "bridge_score": float(br.iloc[0].signed_score) if len(br) else np.nan,
                                           "de_rank": int(dr.iloc[0]["rank"]) if len(dr) else np.nan,
                                           "de_score": float(dr.iloc[0].signed_score) if len(dr) else np.nan})
    pd.DataFrame(partition_rows).to_csv(OUT / "gene_partitions.csv.gz", index=False, compression="gzip")

    dgn = pd.read_csv(DISGENET, sep="\t", usecols=["geneSymbol", "diseaseId"])
    reference_sets = {condition: set(dgn.loc[dgn.diseaseId.eq(cui), "geneSymbol"].dropna().astype(str))
                      for condition, cui in DISGENET_CUI.items()}
    targetable = set(edges.target_gene)
    partition_metrics, partition_drugs = [], []
    for condition, studies in ORDER.items():
        osdr = studies[2]
        for dataset in studies:
            peers = [x for x in studies if x != dataset]
            for partition in PARTITIONS:
                genes = partition_sets[(condition, dataset, partition)]
                peer_sets = [partition_sets[(condition, p, partition)] for p in peers]
                recur_any = genes & set.union(*peer_sets)
                recur_all = genes & set.intersection(*peer_sets)
                if dataset == osdr:
                    osdr_match = genes & (partition_sets[(condition, studies[0], partition)] |
                                          partition_sets[(condition, studies[1], partition)])
                    osdr_j = float(np.mean([jaccard(genes, partition_sets[(condition, p, partition)]) for p in studies[:2]]))
                else:
                    osdr_match = genes & partition_sets[(condition, osdr, partition)]
                    osdr_j = jaccard(genes, partition_sets[(condition, osdr, partition)])
                k, K, odds, p = fisher_overlap(genes, reference_sets[condition], canonical)
                er = enrich(set(genes), canonical, eligible_targets)
                er.insert(0, "partition", partition); er.insert(0, "role", ROLE[dataset])
                er.insert(0, "dataset", dataset); er.insert(0, "condition", condition)
                partition_drugs.append(er)
                partition_metrics.append({"condition": condition, "dataset": dataset, "role": ROLE[dataset],
                                          "partition": partition, "partition_size": len(genes),
                                          "recurrent_in_any_other_count": len(recur_any),
                                          "recurrent_in_any_other_fraction": len(recur_any)/max(len(genes), 1),
                                          "recurrent_in_both_other_count": len(recur_all),
                                          "mean_pairwise_gene_jaccard": float(np.mean([jaccard(genes, x) for x in peer_sets])),
                                          "osdr_overlap_count": len(osdr_match), "osdr_gene_jaccard": osdr_j,
                                          "external_reference": DISGENET_NAME[condition], "external_reference_cui": DISGENET_CUI[condition],
                                          "external_reference_size": K, "external_overlap_count": k,
                                          "external_odds_ratio": odds, "external_pvalue": p,
                                          "targetable_gene_count": len(genes & targetable),
                                          "targetable_gene_fraction": len(genes & targetable)/max(len(genes), 1),
                                          "eligible_drugs_with_overlap": int(er.overlap_count.gt(0).sum()),
                                          "significant_drug_enrichments": int(er.fdr.lt(.05).sum()),
                                          "top_drug": er.iloc[0].drug_name, "top_drug_pvalue": er.iloc[0].pvalue,
                                          "top_drug_fdr": er.iloc[0].fdr})
    partition_metrics = pd.DataFrame(partition_metrics)
    partition_metrics["external_fdr"] = multipletests(partition_metrics.external_pvalue, method="fdr_bh")[1]
    partition_metrics.to_csv(OUT / "partition_metrics.csv", index=False)
    partition_drugs = pd.concat(partition_drugs, ignore_index=True)
    partition_drugs.to_parquet(OUT / "partition_drug_enrichment.parquet", index=False)

    # Repeated same-partition genes, emphasizing Bridge-only recurrence.
    repeated_rows = []
    for condition, studies in ORDER.items():
        for partition in PARTITIONS:
            counts = {}
            for dataset in studies:
                for gene in partition_sets[(condition, dataset, partition)]:
                    counts.setdefault(gene, []).append(dataset)
            for gene, datasets in counts.items():
                if len(datasets) >= 2:
                    repeated_rows.append({"condition": condition, "partition": partition, "gene": gene,
                                          "dataset_count": len(datasets), "datasets": ";".join(datasets),
                                          "includes_osdr": studies[2] in datasets,
                                          "external_support": gene in reference_sets[condition],
                                          "expanded_chembl_target": gene in targetable,
                                          "eligible_drug_count": int(sum(gene in x for x in eligible_targets))})
    repeated = pd.DataFrame(repeated_rows)
    repeated.to_csv(OUT / "recurrent_partition_genes.csv", index=False)

    # Recurrent drug/target mechanisms for whole modules and each partition.
    recurrence_rows, target_rows = [], []
    collection = {("Whole module", method, d): rankings[(method, d)][:25]
                  for method in ["Bridge", "DE"] for d in frozen.dataset.unique()}
    for condition, studies in ORDER.items():
        for partition in PARTITIONS:
            for dataset in studies:
                frame = partition_drugs[(partition_drugs.condition.eq(condition)) &
                                        (partition_drugs.dataset.eq(dataset)) &
                                        (partition_drugs.partition.eq(partition)) &
                                        partition_drugs.overlap_count.gt(0)]
                collection[(partition, partition, dataset)] = frame.head(25).drug_id.tolist()
        analyses = [("Whole module", m, m) for m in ["Bridge", "DE"]] + [(p, p, p) for p in PARTITIONS]
        for analysis, key_method, label in analyses:
            lists = [collection[(analysis, key_method, d)] for d in studies]
            shared = set.intersection(*map(set, lists))
            for drug_id in sorted(shared):
                name = edges.loc[edges.drug_id.eq(drug_id), "drug_name"].iloc[0]
                recurrence_rows.append({"condition": condition, "analysis": label, "drug_id": drug_id,
                                        "drug_name": name, "datasets": ";".join(studies)})
                for dataset in studies:
                    gene_set = modules[(label, dataset)] if label in ["Bridge", "DE"] else partition_sets[(condition, dataset, label)]
                    for gene in sorted(eligible_targets.loc[(drug_id, name)] & gene_set):
                        target_rows.append({"condition": condition, "analysis": label, "drug_id": drug_id,
                                           "drug_name": name, "dataset": dataset, "target_gene": gene})
    pd.DataFrame(recurrence_rows).to_csv(OUT / "recurrent_drugs.csv", index=False)
    pd.DataFrame(target_rows).to_csv(OUT / "recurrent_drug_targets.csv", index=False)

    provenance = {
        "analysis_type": "new expanded-ChEMBL sensitivity and frozen-module complementarity layer",
        "frozen_module_sha256": sha256(MODULE_FILE), "expanded_edge_sha256": sha256(EDGE_FILE),
        "expanded_edge_qc_sha256": sha256(EDGE_QC), "disgenet_sha256": sha256(DISGENET),
        "module_size": 500, "drug_target_rule": ">10 targets in expanded ChEMBL 37 universe",
        "drug_test": "right-sided hypergeometric; BH within each dataset x method/partition family",
        "convergence": "mean three pairwise top-25 nonzero-overlap drug Jaccards",
        "external_references": {c: {"cui": DISGENET_CUI[c], "name": DISGENET_NAME[c]} for c in ORDER},
        "protected_steps_rerun": {"expression": False, "embeddings": False, "condition_vectors": False,
                                  "integrated_gradients": False, "differential_expression": False,
                                  "gene_module_construction": False},
        "direction_aware_reversal": False,
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
