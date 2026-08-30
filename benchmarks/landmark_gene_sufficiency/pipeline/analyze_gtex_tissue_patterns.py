#!/usr/bin/env python3
"""Characterize frozen Shared-451 and Top-921 genes across GTEx tissues."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import REPO_ROOT, RESULTS, sha256

GTEX_H5 = REPO_ROOT / "data/gtex/gtex_matrix.h5"
HGNC = REPO_ROOT / "data/annotations/hgnc/hgnc_complete_set_2026-08-27.tsv"
RANKING = RESULTS / "final_human_informative_gene_ranking.parquet"
SHARED = RESULTS / "shared_top_451_genes.parquet"


def norm_gene(value: object) -> str:
    return str(value).strip().split(".", 1)[0].upper()


def split_symbols(value: object) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    return [norm_gene(item) for item in str(value).split("|") if str(item).strip()]


def build_hgnc_mapping(source_symbols: list[str]) -> dict[str, str]:
    """Return unambiguous, one-to-one source-symbol to approved-symbol mappings."""
    frame = pd.read_csv(HGNC, sep="\t", dtype=str, low_memory=False)
    frame = frame.loc[frame.status.eq("Approved")]
    approved = {norm_gene(symbol): norm_gene(symbol) for symbol in frame.symbol}
    previous: dict[str, set[str]] = defaultdict(set)
    aliases: dict[str, set[str]] = defaultdict(set)
    for row in frame.itertuples(index=False):
        target = norm_gene(row.symbol)
        for symbol in split_symbols(row.prev_symbol):
            previous[symbol].add(target)
        for symbol in split_symbols(row.alias_symbol):
            aliases[symbol].add(target)
    mapped: dict[str, str] = {}
    for source in map(norm_gene, source_symbols):
        candidates = ({approved[source]} if source in approved else
                      previous[source] if len(previous[source]) == 1 else
                      aliases[source] if len(aliases[source]) == 1 else set())
        if len(candidates) == 1:
            mapped[source] = next(iter(candidates))
    reverse: dict[str, list[str]] = defaultdict(list)
    for source, target in mapped.items():
        reverse[target].append(source)
    return {source: target for source, target in mapped.items()
            if len(reverse[target]) == 1 or source == target}


def load_gtex_log1p_cpm(model_genes: list[str], chunk_size: int = 128
                        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load GTEx counts in model order and apply sample-wise log1p(CPM)."""
    with h5py.File(GTEX_H5, "r") as handle:
        source_genes = [norm_gene(value.decode() if isinstance(value, bytes) else value)
                        for value in handle["meta/genes"][:]]
        tissues = np.asarray([value.decode() if isinstance(value, bytes) else str(value)
                              for value in handle["meta/smtsd"][:]], dtype=object)
        crosswalk = build_hgnc_mapping(source_genes)
        approved_to_source = {crosswalk[source]: index for index, source in enumerate(source_genes)
                              if source in crosswalk}
        source_rows = np.asarray([approved_to_source.get(norm_gene(gene), -1)
                                  for gene in model_genes], dtype=int)
        observed = source_rows >= 0
        matrix = np.full((len(tissues), len(model_genes)), np.nan, dtype=np.float32)
        expression = handle["data/expression"]
        for start in range(0, len(tissues), chunk_size):
            stop = min(start + chunk_size, len(tissues))
            counts = np.asarray(expression[start:stop, :], dtype=np.float32)
            library_size = counts.sum(axis=1, keepdims=True)
            selected = counts[:, source_rows[observed]]
            cpm = np.divide(selected, library_size, out=np.zeros_like(selected),
                            where=library_size > 0) * 1e6
            matrix[start:stop, observed] = np.log1p(cpm)
            if stop == len(tissues) or stop % 1024 == 0:
                print(f"[GTEx] normalized {stop:,}/{len(tissues):,} samples", flush=True)
    mapping = pd.DataFrame({"gene": model_genes, "gtex_source_index": source_rows,
                            "observed_in_gtex": observed})
    return matrix, tissues, mapping


def tissue_medians(matrix: np.ndarray, tissues: np.ndarray, genes: list[str]) -> pd.DataFrame:
    unique_tissues = sorted(set(tissues))
    observed = np.isfinite(matrix).any(axis=0)
    values = np.full((len(genes), len(unique_tissues)), np.nan, dtype=np.float32)
    values[observed] = np.column_stack([
        np.median(matrix[tissues == tissue][:, observed], axis=0)
        for tissue in unique_tissues])
    return pd.DataFrame(values, index=pd.Index(genes, name="gene"), columns=unique_tissues)


def specificity_table(medians: pd.DataFrame, ranking: pd.DataFrame,
                      shared_genes: set[str], top_genes: set[str]) -> pd.DataFrame:
    maximum = medians.max(axis=1, skipna=True)
    mean = medians.mean(axis=1, skipna=True)
    highest = pd.Series(pd.NA, index=medians.index, dtype="string")
    observed = medians.notna().any(axis=1)
    highest.loc[observed] = medians.loc[observed].idxmax(axis=1)
    specificity = maximum.div(mean.where(mean > 0))
    output = ranking[["gene", "model_index", "information_score", "information_rank"]].copy()
    output = output.set_index("gene")
    output["maximum_tissue_median"] = maximum
    output["mean_tissue_median"] = mean
    output["tissue_specificity"] = specificity
    output["highest_expression_tissue"] = highest
    output["in_shared_451"] = output.index.isin(shared_genes)
    output["in_top_921"] = output.index.isin(top_genes)
    output["in_background"] = ~output.in_top_921
    return output.reset_index()


def group_summary(table: pd.DataFrame) -> pd.DataFrame:
    groups = {
        "Shared-451": table.in_shared_451,
        "Top-921": table.in_top_921,
        "Remaining/background": table.in_background,
    }
    rows = []
    for name, mask in groups.items():
        values = table.loc[mask, "tissue_specificity"].dropna()
        rows.append({"gene_set": name, "genes_in_set": int(mask.sum()),
                     "genes_with_gtex_specificity": len(values),
                     "mean_tissue_specificity": values.mean(),
                     "median_tissue_specificity": values.median(),
                     "tissue_specificity_q25": values.quantile(0.25),
                     "tissue_specificity_q75": values.quantile(0.75)})
    return pd.DataFrame(rows)


def save_heatmap(medians: pd.DataFrame, shared: pd.DataFrame) -> None:
    ordered = shared.sort_values(["mean_rank", "max_rank", "gene"]).gene.astype(str)
    values = medians.loc[ordered]
    row_mean = values.mean(axis=1)
    row_sd = values.std(axis=1, ddof=0).replace(0, np.nan)
    zscore = values.sub(row_mean, axis=0).div(row_sd, axis=0)
    zscore.to_parquet(RESULTS / "gtex_shared_451_tissue_median_row_zscore.parquet")
    figure_dir = RESULTS / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 16))
    image = ax.imshow(zscore.to_numpy(), aspect="auto", interpolation="nearest",
                      cmap="coolwarm", vmin=-3, vmax=3)
    ax.set_xticks(np.arange(len(zscore.columns)), labels=zscore.columns,
                  rotation=90, fontsize=7)
    ax.set_yticks([])
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("Row z-score")
    ax.set_xlabel("GTEx tissue")
    ax.set_ylabel("Shared-451 genes (ordered by mean information rank)")
    ax.set_title("GTEx tissue-median expression of Shared-451 informative genes")
    fig.tight_layout()
    fig.savefig(figure_dir / "gtex_shared_451_tissue_heatmap.png", dpi=300)
    fig.savefig(figure_dir / "gtex_shared_451_tissue_heatmap.pdf")
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ranking = pd.read_parquet(RANKING).sort_values("model_index")
    if len(ranking) != 15165 or ranking.gene.nunique() != 15165:
        raise AssertionError("Expected exactly 15,165 unique model genes")
    shared = pd.read_parquet(SHARED)
    if len(shared) != 451 or shared.gene.nunique() != 451:
        raise AssertionError("Expected the frozen Shared-451 gene set")
    top_genes = set(ranking.nsmallest(921, "information_rank").gene.astype(str))
    shared_genes = set(shared.gene.astype(str))

    matrix, tissues, mapping = load_gtex_log1p_cpm(ranking.gene.astype(str).tolist())
    medians = tissue_medians(matrix, tissues, ranking.gene.astype(str).tolist())
    medians.to_parquet(RESULTS / "gtex_tissue_median_expression.parquet")
    mapping.to_parquet(RESULTS / "gtex_model_gene_mapping.parquet", index=False)

    specificity = specificity_table(medians, ranking, shared_genes, top_genes)
    specificity.to_parquet(RESULTS / "gtex_gene_tissue_specificity.parquet", index=False)
    summary = group_summary(specificity)
    summary.to_parquet(RESULTS / "gtex_tissue_specificity_group_summary.parquet", index=False)
    summary.to_csv(RESULTS / "gtex_tissue_specificity_group_summary.csv", index=False)

    valid = specificity[["tissue_specificity", "information_score"]].dropna()
    correlation = spearmanr(valid.tissue_specificity, valid.information_score)
    correlation_table = pd.DataFrame([{"genes": len(valid),
        "spearman_rho": correlation.statistic, "p_value": correlation.pvalue}])
    correlation_table.to_parquet(RESULTS / "gtex_tissue_specificity_score_correlation.parquet",
                                 index=False)
    correlation_table.to_csv(RESULTS / "gtex_tissue_specificity_score_correlation.csv", index=False)

    shared_highest = specificity.loc[specificity.in_shared_451, ["gene", "model_index",
        "information_score", "information_rank", "highest_expression_tissue",
        "maximum_tissue_median", "mean_tissue_median", "tissue_specificity"]]
    shared_highest = shared_highest.sort_values("information_rank")
    shared_highest.to_parquet(RESULTS / "gtex_shared_451_highest_expression_tissue.parquet",
                              index=False)
    shared_highest.to_csv(RESULTS / "gtex_shared_451_highest_expression_tissue.csv", index=False)
    save_heatmap(medians, shared)

    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gtex_h5": str(GTEX_H5), "gtex_h5_sha256": sha256(GTEX_H5),
        "tissue_field": "meta/smtsd", "samples": len(tissues),
        "tissues": medians.shape[1], "model_gene_universe": len(ranking),
        "genes_observed_after_unique_hgnc_mapping": int(mapping.observed_in_gtex.sum()),
        "preprocessing": "sample-wise counts per million followed by natural log1p",
        "tissue_summary": "median log1p(CPM) within each detailed GTEx tissue",
        "specificity": "maximum tissue median divided by mean tissue median",
        "ranking": str(RANKING), "shared_gene_set": str(SHARED),
        "shared_genes": len(shared_genes), "top_genes": len(top_genes),
    }
    (RESULTS / "gtex_tissue_pattern_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)
    print("\nSpearman correlation\n" + correlation_table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
