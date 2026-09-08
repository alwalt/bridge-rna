#!/usr/bin/env python3
"""Build the RR1/RR3 contextual-graph case study from existing frozen caches."""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "benchmarks/rr1_rr3_contextual_graph_case_study"
SOURCE = ROOT / "benchmarks/frozen_sample_embedding_readout"
OUT, FIG = BENCH / "results", BENCH / "results/figures"
sys.path.insert(0, str(SOURCE / "pipeline"))
from stress_test_contextual_graphs import G, SPECS, group_vector, load  # noqa: E402

def response_graphs():
    manifest, neighbors, weights = load("task3")
    members = pd.read_csv(ROOT / "benchmarks/osdr_batch_effect_representation/results/task3b_contrast_sample_membership.csv")
    sample_index = dict(zip(manifest.sample_id.astype(str), range(len(manifest))))
    effects = {}
    audit_rows = []
    for name, (contrast_id, strict) in SPECS.items():
        query = members[members.contrast_id.eq(contrast_id)].copy()
        if name == "RR1_original": query = query[~query.sample_id.str.endswith("_M27")]
        if name == "RR1_remeasurement": query = query[~query.sample_id.str.endswith("_M29")]
        if strict: query = query[~query.sample_id.str.endswith("_F5")]
        audit_rows.extend(
            {"measurement": name, "contrast_id": contrast_id, **row}
            for row in query[["sample_id", "OSD", "condition"]].to_dict("records")
        )
        idx = {c: query[query.condition.eq(c)].sample_id.map(sample_index).to_numpy(int) for c in ["FLT", "GC"]}
        vector = group_vector(idx["FLT"], neighbors, weights, 10, "union") - group_vector(idx["GC"], neighbors, weights, 10, "union")
        codes, values = vector.indices, vector.data.astype(np.float64)
        left, right = codes // G, codes % G
        effects[name] = csr_matrix((np.r_[values, values], (np.r_[left, right], np.r_[right, left])), shape=(G, G))
    pd.DataFrame(audit_rows).to_csv(OUT / "sample_manifest.csv", index=False)
    return effects

def row_cosine(a, b):
    numerator = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    denominator = np.sqrt(np.asarray(a.multiply(a).sum(axis=1)).ravel() * np.asarray(b.multiply(b).sum(axis=1)).ravel())
    result = np.full(a.shape[0], np.nan)
    valid = denominator > 0
    result[valid] = numerator[valid] / denominator[valid]
    return result

def gene_sets():
    sets = {f"Hallmark: {n}": {str(g).upper() for g in v} for n, v in json.loads((ROOT / "data/gsea/hallmark_gene_sets.json").read_text()).items()}
    base = ROOT / "benchmarks/cross_species_exercise_response/results/per_study_ranked_gsea"
    for collection, filename in [("GO BP", "GO_Biological_Process_2026.gmt"), ("Reactome", "Reactome_Pathways_2024.gmt")]:
        for line in (base / filename).read_text().splitlines():
            fields = line.split("\t")
            if len(fields) >= 3: sets[f"{collection}: {fields[0]}"] = {g.upper() for g in fields[2:]}
    return sets

def enrich(selected, universe):
    rows = []
    for term, genes in gene_sets().items():
        members, overlap = genes & universe, genes & selected
        if len(members) >= 5 and overlap:
            rows.append({"term": term, "overlap": len(overlap), "set_size": len(members), "p_value": hypergeom.sf(len(overlap)-1, len(universe), len(members), len(selected))})
    result = pd.DataFrame(rows)
    if not result.empty:
        result["fdr"] = multipletests(result.p_value, method="fdr_bh")[1]
        result = result.sort_values(["fdr", "p_value", "term"])
    return result

def main():
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    stress = pd.read_csv(SOURCE / "results/graph_fingerprint/rr1_rr3_stress/cross_representation_comparison.csv")
    stress["comparison"] = stress.comparison.replace({"false_friend": "RR1↔RR3-39 false friend"})
    primary = stress[stress.representation.isin(["Bridge mean", "Graph signed-edge response"])].copy()
    primary["relationship"] = np.where(primary.comparison.str.contains("false"), "unrelated", "true replicate")
    primary.to_csv(OUT / "global_mean_vs_graph.csv", index=False)
    effects = response_graphs()
    true39 = row_cosine(effects["RR3_39_original"], effects["RR3_39_remeasurement"])
    true40 = row_cosine(effects["RR3_40_original"], effects["RR3_40_remeasurement"])
    false = row_cosine(effects["RR1_original"], effects["RR3_39_original"])
    genes = pd.read_csv(ROOT / "data/ensembl/canonical_genes.csv").sort_values("token_id").gene_symbol.astype(str)
    local = pd.DataFrame({"gene_symbol": genes, "rr3_39_true_local_cosine": true39, "rr3_40_true_local_cosine": true40, "false_friend_local_cosine": false})
    local["rr3_39_discrimination"] = local.rr3_39_true_local_cosine - local.false_friend_local_cosine
    local["rr3_40_discrimination"] = local.rr3_40_true_local_cosine - local.false_friend_local_cosine
    local["mean_true_discrimination"] = local[["rr3_39_discrimination", "rr3_40_discrimination"]].mean(axis=1)
    local = local.sort_values("mean_true_discrimination", ascending=False)
    local["rank"] = np.arange(1, len(local)+1)
    local.to_csv(OUT / "per_gene_neighborhood_discrimination.csv", index=False)
    local.head(50).to_csv(OUT / "top50_discriminative_gene_neighborhoods.csv", index=False)
    universe = set(genes.str.upper()); selected = set(local.head(250).gene_symbol.str.upper())
    enrich(selected, universe).to_csv(OUT / "top250_neighborhood_enrichment.csv", index=False)
    order = ["RR1", "RR3-39", "RR3-40", "RR1↔RR3-39 false friend"]
    representations = ["Bridge mean", "Graph signed-edge response"]
    y = np.arange(len(order)); height = 0.36
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for offset, representation, color in [(-height / 2, representations[0], "#9E9E9E"), (height / 2, representations[1], "#2878B5")]:
        values = [value if not subset.empty else np.nan for comparison in order
                  for subset in [primary[(primary.comparison == comparison) & (primary.representation == representation)]]
                  for value in [float(subset.similarity.iloc[0]) if not subset.empty else np.nan]]
        bars = ax.barh(y + offset, values, height=height, label=representation, color=color)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_yticks(y, order); ax.invert_yaxis(); ax.grid(axis="x", alpha=.25)
    ax.axvline(0, color="black", lw=.8); ax.set(title="Global pooling creates a false friend; contextual graphs restore ordering", xlabel="Within-representation response similarity", ylabel=""); ax.legend(title="")
    fig.tight_layout()
    for ext in ["png", "pdf"]: fig.savefig(FIG / f"global_mean_vs_graph.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    show = local.head(20).sort_values("mean_true_discrimination")
    fig, ax = plt.subplots(figsize=(9.5, 7.5)); ax.barh(show.gene_symbol, show.mean_true_discrimination, color="#2878B5"); ax.axvline(0, color="black", lw=.8)
    ax.set(xlabel="True-replicate minus false-friend local cosine", ylabel="", title="Gene neighborhoods favoring true RR3 replication"); fig.tight_layout()
    for ext in ["png", "pdf"]: fig.savefig(FIG / f"top_gene_neighborhoods.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    def value(rep, comp): return float(primary.query("representation == @rep and comparison == @comp").similarity.iloc[0])
    summary = {"primary_graph": "layer-12 contextual-cosine union-kNN, k=10", "parameters_selected_without_rr1_rr3": True, "global_mean_false_friend": value("Bridge mean", "RR1↔RR3-39 false friend"), "global_mean_rr3_39": value("Bridge mean", "RR3-39"), "graph_false_friend": value("Graph signed-edge response", "RR1↔RR3-39 false friend"), "graph_rr3_39": value("Graph signed-edge response", "RR3-39"), "graph_rr3_40": value("Graph signed-edge response", "RR3-40"), "limitations": ["graph and vector scores have different scales", "contextual edges are not causal interactions", "RR1 remains a cross-protocol failure"]}
    (OUT / "case_study_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__": main()
