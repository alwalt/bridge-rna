#!/usr/bin/env python3
"""Decompose frozen Phase-2 muscle drug convergence into target support.

This is a post hoc diagnostic of saved top-500 / >=3-target enrichment results.
It does not rerun Bridge, differential expression, module construction, or drug
enrichment. Target-removal counterfactuals remove a gene from each observed
drug/module overlap and discard a drug only when no observed support remains.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import hypergeom


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "muscle_target_decomposition"
DATASETS = ["GSE211204", "GSE113165", "GSE234465"]
PRIMARY_SIZE = 500
PRIMARY_MIN_TARGETS = 3


def bh(pvalues: pd.Series) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def split_genes(value: str) -> set[str]:
    return {x for x in str(value).split(";") if x and x != "nan"}


def mean_pairwise_jaccard(drug_sets: dict[str, set[str]]) -> tuple[float, list[dict]]:
    rows = []
    for a, b in combinations(DATASETS, 2):
        inter = drug_sets[a] & drug_sets[b]
        union = drug_sets[a] | drug_sets[b]
        rows.append({
            "dataset_a": a,
            "dataset_b": b,
            "intersection": len(inter),
            "union": len(union),
            "jaccard": len(inter) / len(union) if union else np.nan,
        })
    return float(np.nanmean([r["jaccard"] for r in rows])), rows


def supported_sets(frame: pd.DataFrame, remove_target: str | None = None,
                   remove_drug: str | None = None) -> dict[str, set[str]]:
    result = {}
    for dataset in DATASETS:
        selected = set()
        for row in frame[frame.dataset.eq(dataset)].itertuples():
            if remove_drug and row.drug_id == remove_drug:
                continue
            genes = split_genes(row.overlap_genes)
            if remove_target:
                genes.discard(remove_target)
            if genes:
                selected.add(row.drug_id)
        result[dataset] = selected
    return result


def add_rank_scores(table: pd.DataFrame, method: str, ranking: pd.DataFrame) -> pd.DataFrame:
    score_col = "signed_bridge_score" if method == "Bridge" else "score"
    lookup = ranking.set_index(["dataset", "gene"])[["rank", score_col]]
    rows = []
    genes = sorted({g for value in table.overlap_genes for g in split_genes(value)})
    for gene in genes:
        for dataset in DATASETS:
            if (dataset, gene) in lookup.index:
                rec = lookup.loc[(dataset, gene)]
                rows.append({"method": method, "target_gene": gene, "dataset": dataset,
                             "rank": int(rec["rank"]), "signed_score": float(rec[score_col]),
                             "in_top_500": bool(int(rec["rank"]) <= 500)})
    return pd.DataFrame(rows)


def parse_gmt(path: Path):
    with path.open() as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 3:
                yield fields[0], fields[1], set(fields[2:])


def enrichment(query: set[str], universe: set[str], source: str, path: Path) -> pd.DataFrame:
    rows = []
    q = query & universe
    for term, description, genes in parse_gmt(path):
        term_genes = genes & universe
        overlap = q & term_genes
        if 10 <= len(term_genes) <= 500:
            p = hypergeom.sf(len(overlap) - 1, len(universe), len(term_genes), len(q))
            rows.append({"query": ";".join(sorted(q)), "source": source, "term": term,
                         "description": description, "term_size": len(term_genes),
                         "overlap_count": len(overlap), "overlap_genes": ";".join(sorted(overlap)),
                         "pvalue": p})
    result = pd.DataFrame(rows)
    if not result.empty:
        result["fdr"] = bh(result.pvalue)
        # Correct across the full tested library, then retain interpretable hits.
        result = result[result.overlap_count.gt(0)].sort_values(["fdr", "pvalue", "term"])
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    enrich = pd.read_parquet(RESULTS / "evaluation/drug_enrichment.parquet")
    primary = enrich[
        enrich.dataset.isin(DATASETS)
        & enrich.module_size.eq(PRIMARY_SIZE)
        & enrich.min_targets.eq(PRIMARY_MIN_TARGETS)
        & enrich.overlap_count.gt(0)
    ].copy()
    primary["overlap_genes"] = primary.overlap_genes.fillna("")

    bridge = primary[primary.method.eq("Bridge")].copy()
    de = primary[primary.method.eq("DE")].copy()
    bridge_sets = supported_sets(bridge)
    de_sets = supported_sets(de)
    baseline_bridge, bridge_pairs = mean_pairwise_jaccard(bridge_sets)
    baseline_de, de_pairs = mean_pairwise_jaccard(de_sets)

    shared_bridge = set.intersection(*bridge_sets.values())
    shared_de = set.intersection(*de_sets.values())
    shared_rows = bridge[bridge.drug_id.isin(shared_bridge)][
        ["drug_id", "drug_name", "dataset", "target_count", "overlap_count", "overlap_genes",
         "pvalue", "fdr", "drug_rank"]
    ].sort_values(["drug_name", "dataset"])
    shared_rows.to_csv(OUT / "shared_bridge_drugs_by_dataset.csv", index=False)

    # One compact row per shared drug with exact dataset-specific support.
    compact = []
    for (drug_id, drug_name), group in shared_rows.groupby(["drug_id", "drug_name"]):
        by_dataset = dict(zip(group.dataset, group.overlap_genes))
        union = sorted(set().union(*(split_genes(x) for x in group.overlap_genes)))
        strict = sorted(set.intersection(*(split_genes(x) for x in group.overlap_genes)))
        compact.append({
            "drug_id": drug_id, "drug_name": drug_name,
            **{f"{d}_overlap_genes": by_dataset[d] for d in DATASETS},
            "union_targets": ";".join(union), "targets_shared_all_datasets": ";".join(strict),
        })
    compact = pd.DataFrame(compact).sort_values("drug_name")
    compact.to_csv(OUT / "target_drug_dataset_compact.csv", index=False)

    # Collapse all observed Bridge candidates by the targets that support them.
    target_rows = []
    for row in bridge.itertuples():
        for gene in split_genes(row.overlap_genes):
            target_rows.append({"target_gene": gene, "drug_id": row.drug_id,
                                "drug_name": row.drug_name, "dataset": row.dataset})
    target_support = pd.DataFrame(target_rows)
    target_summary = (target_support.groupby("target_gene")
                      .agg(drug_count=("drug_id", "nunique"), drugs=("drug_name", lambda x: ";".join(sorted(set(x)))),
                           dataset_count=("dataset", "nunique"), datasets=("dataset", lambda x: ";".join(sorted(set(x)))),
                           drug_dataset_supports=("drug_id", "size"))
                      .reset_index().sort_values(["dataset_count", "drug_count", "target_gene"], ascending=[False, False, True]))
    target_summary.to_csv(OUT / "bridge_target_drug_dataset_summary.csv", index=False)
    target_support.sort_values(["target_gene", "drug_name", "dataset"]).to_csv(
        OUT / "bridge_target_drug_dataset_long.csv", index=False)

    # Target combinations supporting the same drug in one or more datasets.
    combos = (bridge.assign(target_combination=bridge.overlap_genes)
              .groupby("target_combination")
              .agg(drug_count=("drug_id", "nunique"), drugs=("drug_name", lambda x: ";".join(sorted(set(x)))),
                   dataset_count=("dataset", "nunique"), datasets=("dataset", lambda x: ";".join(sorted(set(x)))))
              .reset_index().sort_values(["drug_count", "dataset_count"], ascending=False))
    combos.to_csv(OUT / "bridge_target_combinations.csv", index=False)

    bridge_rank = pd.read_parquet(RESULTS / "bridge/study_gene_rankings.parquet")
    de_rank = pd.read_parquet(RESULTS / "differential_expression/study_gene_rankings.parquet")
    scores = pd.concat([add_rank_scores(bridge, "Bridge", bridge_rank),
                        add_rank_scores(de, "DE", de_rank)], ignore_index=True)
    scores.to_csv(OUT / "recurrent_target_ranks_scores.csv", index=False)

    # Counterfactuals for every observed target and the most-supported drug.
    sensitivity = [{"removal_type": "none", "removed": "none", "mean_jaccard": baseline_bridge,
                    "absolute_change": 0.0, "fraction_of_baseline_lost": 0.0}]
    for gene in sorted(target_support.target_gene.unique()):
        value, _ = mean_pairwise_jaccard(supported_sets(bridge, remove_target=gene))
        sensitivity.append({"removal_type": "target", "removed": gene, "mean_jaccard": value,
                            "absolute_change": value - baseline_bridge,
                            "fraction_of_baseline_lost": (baseline_bridge - value) / baseline_bridge})
    drug_support = (target_support.groupby(["drug_id", "drug_name"]).size()
                    .reset_index(name="target_dataset_supports")
                    .sort_values(["target_dataset_supports", "drug_name"], ascending=[False, True]))
    most_promiscuous_drug = drug_support.iloc[0]
    value, _ = mean_pairwise_jaccard(supported_sets(bridge, remove_drug=most_promiscuous_drug.drug_id))
    sensitivity.append({"removal_type": "drug", "removed": most_promiscuous_drug.drug_name,
                        "mean_jaccard": value, "absolute_change": value - baseline_bridge,
                        "fraction_of_baseline_lost": (baseline_bridge - value) / baseline_bridge})
    sensitivity = pd.DataFrame(sensitivity).sort_values(["removal_type", "mean_jaccard"])
    sensitivity.to_csv(OUT / "bridge_target_removal_sensitivity.csv", index=False)

    # DE recurrence has the same transparent long/compact representation.
    de_long = []
    for row in de.itertuples():
        for gene in split_genes(row.overlap_genes):
            de_long.append({"target_gene": gene, "drug_id": row.drug_id,
                            "drug_name": row.drug_name, "dataset": row.dataset})
    de_long = pd.DataFrame(de_long)
    de_summary = (de_long.groupby("target_gene")
                  .agg(drug_count=("drug_id", "nunique"), drugs=("drug_name", lambda x: ";".join(sorted(set(x)))),
                       dataset_count=("dataset", "nunique"), datasets=("dataset", lambda x: ";".join(sorted(set(x)))),
                       drug_dataset_supports=("drug_id", "size"))
                  .reset_index().sort_values(["dataset_count", "drug_count"], ascending=False))
    de_long.to_csv(OUT / "de_target_drug_dataset_long.csv", index=False)
    de_summary.to_csv(OUT / "de_target_drug_dataset_summary.csv", index=False)

    # GO/KEGG is deliberately limited to strict all-study recurrent targets.
    strict_targets = set.intersection(*[
        set(target_support[target_support.dataset.eq(d)].target_gene) for d in DATASETS
    ])
    union_shared_targets = set().union(*(split_genes(x) for x in shared_rows.overlap_genes))
    canonical_path = Path("/home/walt/bridge-rna/data/ensembl/canonical_genes.csv")
    canonical = pd.read_csv(canonical_path)
    gene_col = next(c for c in canonical.columns if "symbol" in c.lower() or c.lower() == "gene")
    universe = set(canonical[gene_col].dropna().astype(str))
    gmt_paths = {
        "GO_Biological_Process_2026": Path("/home/walt/bridge-rna/benchmarks/cross_species_exercise_response/results/per_study_ranked_gsea/GO_Biological_Process_2026.gmt"),
        "KEGG_2026": Path("/home/walt/bridge-rna/benchmarks/cross_species_exercise_response/results/per_study_ranked_gsea/KEGG_2026.gmt"),
    }
    go_frames = []
    for label, query in [("strict_all_study", strict_targets), ("shared_drug_target_union", union_shared_targets)]:
        for source, path in gmt_paths.items():
            frame = enrichment(query, universe, source, path)
            if not frame.empty:
                frame.insert(0, "query_label", label)
                go_frames.append(frame)
    pathway = pd.concat(go_frames, ignore_index=True) if go_frames else pd.DataFrame()
    pathway.to_csv(OUT / "recurrent_target_pathway_enrichment.csv", index=False)

    # Publication-ready diagnostic figure: exact support and leave-one-out effects.
    major = sensitivity[(sensitivity.removal_type.eq("none")) | sensitivity.removed.isin(
        ["ERBB2", "PDE5A", "PDE10A", most_promiscuous_drug.drug_name])].copy()
    major["label"] = major.apply(lambda r: "No removal" if r.removal_type == "none" else f"Remove {r.removed}", axis=1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1.35, 1]})
    matrix_genes = ["ERBB2", "PDE5A", "PDE10A"]
    shared_names = compact.drug_name.tolist()
    matrix = np.zeros((len(shared_names), len(matrix_genes)))
    annotations = np.empty(matrix.shape, dtype=object)
    annotations[:] = ""
    for i, drug in enumerate(shared_names):
        rows = shared_rows[shared_rows.drug_name.eq(drug)]
        for j, gene in enumerate(matrix_genes):
            ds = [r.dataset.replace("GSE", "") for r in rows.itertuples() if gene in split_genes(r.overlap_genes)]
            matrix[i, j] = len(ds)
            annotations[i, j] = ", ".join(ds)
    im = ax1.imshow(matrix, cmap="Blues", vmin=0, vmax=3, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax1.text(j, i, annotations[i, j] or "—", ha="center", va="center", fontsize=8,
                     color="white" if matrix[i, j] >= 2 else "#222222")
    ax1.set_xticks(range(len(matrix_genes)), matrix_genes)
    ax1.set_yticks(range(len(shared_names)), shared_names)
    ax1.set_title("Targets supporting drugs shared by all studies")
    ax1.set_xlabel("Cell text: supporting dataset accession suffix")
    colors = ["#555555" if x == "No removal" else "#2b6cb0" for x in major.label]
    ax2.barh(major.label, major.mean_jaccard, color=colors)
    ax2.axvline(baseline_bridge, color="#222222", ls="--", lw=1)
    ax2.set_xlim(0, 0.8)
    ax2.set_xlabel("Mean pairwise drug Jaccard")
    ax2.set_title("Support-removal sensitivity")
    for y, value in enumerate(major.mean_jaccard):
        ax2.text(value + 0.015, y, f"{value:.3f}", va="center", fontsize=9)
    fig.suptitle("Bridge muscle convergence is dominated by two recurrent targets", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "muscle_target_structure.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "muscle_target_structure.pdf", bbox_inches="tight")
    plt.close(fig)

    provenance = {
        "analysis": "frozen Phase-2 muscle target-support decomposition",
        "datasets": DATASETS, "module_size": PRIMARY_SIZE, "minimum_drug_targets": PRIMARY_MIN_TARGETS,
        "phase2_rerun": False, "bridge_baseline_mean_pairwise_jaccard": baseline_bridge,
        "de_baseline_mean_pairwise_jaccard": baseline_de,
        "shared_bridge_drug_count": len(shared_bridge), "shared_de_drug_count": len(shared_de),
        "strict_recurrent_bridge_targets": sorted(strict_targets),
        "shared_bridge_drug_target_union": sorted(union_shared_targets),
        "most_promiscuous_target_definition": "maximum observed drug x dataset support edges",
        "most_promiscuous_target": target_summary.iloc[0].target_gene,
        "most_promiscuous_drug_definition": "maximum observed target x dataset support edges; alphabetical tie break",
        "most_promiscuous_drug": most_promiscuous_drug.drug_name,
        "target_removal_contract": "remove target from saved overlap_genes and discard drug in a dataset only if no saved overlap remains",
        "inputs": ["evaluation/drug_enrichment.parquet", "bridge/study_gene_rankings.parquet",
                   "differential_expression/study_gene_rankings.parquet"],
        "bridge_pairwise": bridge_pairs, "de_pairwise": de_pairs,
        "pathway_universe_size": len(universe), "pathway_query_is_exploratory": True,
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
