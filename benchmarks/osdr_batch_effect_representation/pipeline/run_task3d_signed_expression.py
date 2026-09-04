#!/usr/bin/env python3
"""Signed FLT-minus-GC expression analysis for fixed Task 3D IG gene sets."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
RESULTS = HERE / "results"
IG_DIR = RESULTS / "task3d_mode_ig"
OUT = IG_DIR / "signed_expression"
FIGURES = OUT / "figures"


def expression_effects() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    genes = pd.read_csv(ROOT / "data/ensembl/canonical_genes.csv").gene_symbol.astype(str).str.upper().tolist()
    if len(genes) != 15165 or len(set(genes)) != 15165:
        raise AssertionError("Expected the exact 15,165-gene BridgeRNA vocabulary")
    expression = np.load(HERE / "work/bridgerna_log1p_tpm_inputs.npy", mmap_mode="r")
    manifest = pd.read_csv(RESULTS / "sample_manifest.csv")
    contrasts = pd.read_csv(RESULTS / "task3c_cluster_assignments.csv").sort_values(
        ["geometry_cluster", "heatmap_order"]).reset_index(drop=True)
    members = pd.read_csv(RESULTS / "task3b_contrast_sample_membership.csv").merge(
        manifest[["sample_id"]].reset_index(names="sample_index"), on="sample_id", validate="many_to_one")
    effects = []
    for contrast in contrasts.contrast_id:
        group = members[members.contrast_id.eq(contrast)]
        flt = group.loc[group.condition.eq("FLT"), "sample_index"].to_numpy(int)
        gc = group.loc[group.condition.eq("GC"), "sample_index"].to_numpy(int)
        effects.append(np.asarray(expression[flt]).mean(0) - np.asarray(expression[gc]).mean(0))
    return contrasts, np.stack(effects), genes


def gene_sets(rankings: pd.DataFrame) -> dict[str, list[str]]:
    top1 = rankings.query("mode == 1 and rank <= 100").sort_values("rank").gene_symbol_human.tolist()
    top2 = rankings.query("mode == 2 and rank <= 100").sort_values("rank").gene_symbol_human.tolist()
    shared = set(top1) & set(top2)
    return {"shared_top100": [x for x in top1 if x in shared],
            "mode_1_specific_top100": [x for x in top1 if x not in shared],
            "mode_2_specific_top100": [x for x in top2 if x not in shared]}


def summarize_genes(contrasts: pd.DataFrame, effects: np.ndarray, genes: list[str],
                    sets: dict[str, list[str]], rankings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = {gene: i for i, gene in enumerate(genes)}
    membership = {gene: name for name, values in sets.items() for gene in values}
    selected = list(dict.fromkeys(sum(sets.values(), [])))
    long_rows, summary_rows = [], []
    rank_lookup = rankings.set_index(["mode", "gene_symbol_human"])["rank"]
    for contrast_index, contrast in contrasts.iterrows():
        for gene in selected:
            long_rows.append({"contrast_id": contrast.contrast_id, "OSD": contrast.OSD,
                              "mode": int(contrast.geometry_cluster), "gene": gene,
                              "gene_set": membership[gene], "flt_minus_gc_log1p_tpm": effects[contrast_index, index[gene]]})
    long = pd.DataFrame(long_rows)
    for gene in selected:
        record = {"gene": gene, "gene_set": membership[gene],
                  "mode_1_ig_rank": rank_lookup.get((1, gene), np.nan),
                  "mode_2_ig_rank": rank_lookup.get((2, gene), np.nan)}
        for mode in (1, 2):
            values = long.query("gene == @gene and mode == @mode").flt_minus_gc_log1p_tpm.to_numpy()
            mean = values.mean()
            record.update({f"mode_{mode}_mean_change": mean, f"mode_{mode}_median_change": np.median(values),
                           f"mode_{mode}_sd_change": values.std(ddof=1),
                           f"mode_{mode}_direction_consistency": np.mean(np.sign(values) == np.sign(mean))})
        record["mode_mean_same_sign"] = np.sign(record["mode_1_mean_change"]) == np.sign(record["mode_2_mean_change"])
        summary_rows.append(record)
    return long, pd.DataFrame(summary_rows)


def extract_pathways(raw: dict) -> pd.DataFrame:
    queries = raw["meta"]["genes_metadata"]["query"]
    candidates = []
    for result in raw.get("result", []):
        if result["query"] not in {"mode_1_top100", "mode_2_top100"} or not result.get("significant", False):
            continue
        ensgs = queries[result["query"]]["ensgs"]
        mapping = queries[result["query"]]["mapping"]
        reverse = {ensg: gene for gene, identifiers in mapping.items() for ensg in identifiers}
        members = [reverse[ensg] for ensg, hit in zip(ensgs, result["intersections"]) if hit and ensg in reverse]
        if len(members) < 3 or result["name"].lower() in {"metabolism", "kegg root term"}:
            continue
        candidates.append({"query": result["query"], "mode": 1 if "mode_1" in result["query"] else 2,
                           "source": result["source"], "term_id": result["native"], "term": result["name"],
                           "adjusted_p": result["p_value"], "genes": members})
    candidates.sort(key=lambda x: x["adjusted_p"])
    selected = []
    # Greedy redundancy control makes a compact actual-term display without inventing themes.
    for candidate in candidates:
        genes = set(candidate["genes"])
        if any(len(genes & set(old["genes"])) / len(genes | set(old["genes"])) >= .6 for old in selected):
            continue
        selected.append(candidate)
        if len(selected) == 12:
            break
    return pd.DataFrame(selected)


def summarize_pathways(pathways: pd.DataFrame, contrasts: pd.DataFrame, effects: np.ndarray,
                       genes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = {gene: i for i, gene in enumerate(genes)}
    long_rows, summary_rows = [], []
    for pathway in pathways.itertuples(index=False):
        member_indices = [index[x] for x in pathway.genes if x in index]
        values = np.median(effects[:, member_indices], axis=1)
        for contrast_index, contrast in contrasts.iterrows():
            long_rows.append({"term_id": pathway.term_id, "term": pathway.term, "source": pathway.source,
                              "discovery_mode": pathway.mode, "contrast_id": contrast.contrast_id,
                              "OSD": contrast.OSD, "mode": int(contrast.geometry_cluster),
                              "member_genes": len(member_indices), "median_member_expression_change": values[contrast_index]})
        record = {"term_id": pathway.term_id, "term": pathway.term, "source": pathway.source,
                  "discovery_mode": pathway.mode, "adjusted_p": pathway.adjusted_p,
                  "member_genes": len(member_indices)}
        for mode in (1, 2):
            z = values[contrasts.geometry_cluster.eq(mode)]
            center = np.median(z)
            record.update({f"mode_{mode}_median_change": center,
                           f"mode_{mode}_direction_consistency": np.mean(np.sign(z) == np.sign(center))})
        record["mode_medians_same_sign"] = np.sign(record["mode_1_median_change"]) == np.sign(record["mode_2_median_change"])
        summary_rows.append(record)
    return pd.DataFrame(long_rows), pd.DataFrame(summary_rows)


def mode_comparison(gene_summary: pd.DataFrame, pathway_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gene_set, frame in gene_summary.groupby("gene_set"):
        rho = spearmanr(frame.mode_1_mean_change, frame.mode_2_mean_change).statistic if len(frame) > 2 else np.nan
        rows.append({"level": "gene", "set": gene_set, "features": len(frame),
                     "mode1_vs_mode2_spearman": rho,
                     "same_direction_fraction": frame.mode_mean_same_sign.mean(),
                     "opposite_direction_fraction": 1 - frame.mode_mean_same_sign.mean()})
    if len(pathway_summary):
        rows.append({"level": "pathway", "set": "selected_major_terms", "features": len(pathway_summary),
                     "mode1_vs_mode2_spearman": spearmanr(pathway_summary.mode_1_median_change,
                                                          pathway_summary.mode_2_median_change).statistic,
                     "same_direction_fraction": pathway_summary.mode_medians_same_sign.mean(),
                     "opposite_direction_fraction": 1 - pathway_summary.mode_medians_same_sign.mean()})
    return pd.DataFrame(rows)


def direction_consistency_summary(gene_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gene_set, frame in gene_summary.groupby("gene_set"):
        for mode in (1, 2):
            values = frame[f"mode_{mode}_direction_consistency"]
            rows.append({"gene_set": gene_set, "mode": mode, "genes": len(frame),
                         "median_direction_consistency": values.median(),
                         "genes_consistent_at_least_5_of_7": int((values >= 5 / 7).sum()),
                         "genes_consistent_at_least_6_of_7": int((values >= 6 / 7).sum()),
                         "genes_consistent_7_of_7": int((values == 1).sum())})
    return pd.DataFrame(rows)


def row_scale(matrix: pd.DataFrame) -> pd.DataFrame:
    denominator = matrix.abs().max(axis=1).replace(0, 1)
    return matrix.div(denominator, axis=0)


def plot_heatmap(gene_long: pd.DataFrame, pathway_long: pd.DataFrame, rankings: pd.DataFrame,
                 contrasts: pd.DataFrame) -> None:
    rank1 = rankings.query("mode == 1").set_index("gene_symbol_human")["rank"]
    rank2 = rankings.query("mode == 2").set_index("gene_symbol_human")["rank"]
    membership = gene_long[["gene", "gene_set"]].drop_duplicates().set_index("gene").gene_set
    shared = sorted(membership[membership.eq("shared_top100")].index,
                    key=lambda gene: min(rank1.get(gene, 99999), rank2.get(gene, 99999)))[:8]
    mode1 = sorted(membership[membership.eq("mode_1_specific_top100")].index,
                   key=lambda gene: rank1.get(gene, 99999))[:10]
    mode2 = sorted(membership[membership.eq("mode_2_specific_top100")].index,
                   key=lambda gene: rank2.get(gene, 99999))[:10]
    genes = shared + mode1 + mode2
    columns = contrasts.contrast_id.tolist()
    gene_matrix = gene_long[gene_long.gene.isin(genes)].pivot(index="gene", columns="contrast_id",
                                                              values="flt_minus_gc_log1p_tpm").loc[genes, columns]
    pathway_order = pathway_long[["term_id", "source", "term"]].drop_duplicates()
    pathway_matrix = pathway_long.pivot(index="term_id", columns="contrast_id",
                                        values="median_member_expression_change").loc[pathway_order.term_id, columns]
    pathway_matrix.index = [f"{row.source} | {row.term[:52]}" for row in pathway_order.itertuples(index=False)]
    labels = [f"C{i+1:02d}\nM{mode}" for i, mode in enumerate(contrasts.geometry_cluster)]
    fig, axes = plt.subplots(2, 1, figsize=(14, 13), gridspec_kw={"height_ratios": [2.2, 1]}, layout="constrained")
    gene_image = axes[0].imshow(row_scale(gene_matrix), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axes[0].set_xticks(np.arange(len(labels)), labels)
    axes[0].set_yticks(np.arange(len(gene_matrix)), gene_matrix.index)
    fig.colorbar(gene_image, ax=axes[0], label="Signed Δ / row max |Δ|", pad=.01)
    axes[0].set(title="Major shared and mode-specific IG genes", xlabel="", ylabel="Gene")
    pathway_image = axes[1].imshow(row_scale(pathway_matrix), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axes[1].set_xticks(np.arange(len(labels)), labels)
    axes[1].set_yticks(np.arange(len(pathway_matrix)), pathway_matrix.index)
    fig.colorbar(pathway_image, ax=axes[1], label="Signed median Δ / row max |Δ|", pad=.01)
    axes[1].set(title="Major nonredundant enriched terms (actual GO/KEGG/Reactome terms)",
                xlabel="Contrast and Task 3C mode", ylabel="Pathway")
    fig.suptitle("Signed FLT − GC expression changes in Task 3D IG genes and pathways", fontsize=15)
    fig.savefig(FIGURES / "signed_gene_pathway_heatmap.png", dpi=400, bbox_inches="tight")
    fig.savefig(FIGURES / "signed_gene_pathway_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    gene_matrix.to_csv(OUT / "heatmap_gene_raw_changes.csv")
    pathway_matrix.to_csv(OUT / "heatmap_pathway_raw_changes.csv")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    contrasts, effects, genes = expression_effects()
    rankings = pd.read_parquet(IG_DIR / "mode_ig_gene_rankings.parquet")
    sets = gene_sets(rankings)
    gene_long, gene_summary = summarize_genes(contrasts, effects, genes, sets, rankings)
    raw = json.loads((IG_DIR / "mode_ig_enrichment_raw.json").read_text())
    pathways = extract_pathways(raw)
    pathway_long, pathway_summary = summarize_pathways(pathways, contrasts, effects, genes)
    comparison = mode_comparison(gene_summary, pathway_summary)
    consistency = direction_consistency_summary(gene_summary)
    gene_long.to_parquet(OUT / "signed_gene_changes_by_contrast.parquet", index=False)
    gene_summary.to_csv(OUT / "signed_gene_direction_summary.csv", index=False)
    pathway_long.to_parquet(OUT / "signed_pathway_changes_by_contrast.parquet", index=False)
    pathway_summary.to_csv(OUT / "signed_pathway_direction_summary.csv", index=False)
    comparison.to_csv(OUT / "mode_direction_comparison.csv", index=False)
    consistency.to_csv(OUT / "gene_set_direction_consistency.csv", index=False)
    contrasts[["contrast_id", "geometry_cluster", "OSD", "mission"]].to_csv(OUT / "contrast_order.csv", index=False)
    plot_heatmap(gene_long, pathway_long, rankings, contrasts)
    provenance = {"expression": "cached exact 15,165-gene natural log1p(TPM) BridgeRNA input",
                  "contrast": "strict Task 3B mean(FLT) minus mean(GC)",
                  "gene_sets": {key: len(value) for key, value in sets.items()},
                  "pathway_direction": "median signed expression change among enriched-term genes present in its Top-100 IG query",
                  "pathway_selection": "12 adjusted-p-ranked actual terms with greedy gene-set Jaccard <0.6; generic roots removed",
                  "heatmap_scaling": "row divided by row maximum absolute value for display only; raw changes saved separately",
                  "rerun": {"IG": False, "deletion": False, "edgeR": False, "batch_correction": False}}
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("Signed Task 3D analysis complete")
    print(comparison.to_string(index=False))
    print("\nGene direction consistency\n", consistency.to_string(index=False))
    print("\nPathway direction summary")
    print(pathway_summary.to_string(index=False))


if __name__ == "__main__":
    main()
