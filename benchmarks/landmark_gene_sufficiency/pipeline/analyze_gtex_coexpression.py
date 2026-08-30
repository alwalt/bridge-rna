#!/usr/bin/env python3
"""Test whether informative reconstruction genes are broadly coexpressed in GTEx."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from analyze_gtex_tissue_patterns import GTEX_H5, RANKING, SHARED, load_gtex_log1p_cpm
from common import RESULTS


def coexpression_summaries(matrix: np.ndarray, device_name: str, block_size: int
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Compute exact per-gene summaries without retaining the correlation matrix."""
    finite = np.isfinite(matrix).all(axis=0)
    means = np.full(matrix.shape[1], np.nan, dtype=np.float32)
    means[finite] = matrix[:, finite].mean(axis=0)
    centered = matrix[:, finite] - means[finite]
    norms = np.linalg.norm(centered, axis=0)
    variable_local = np.isfinite(norms) & (norms > 0)
    valid_indices = np.flatnonzero(finite)[variable_local]
    normalized = centered[:, variable_local] / norms[variable_local]

    requested = torch.device(device_name)
    device = requested if requested.type != "cuda" or torch.cuda.is_available() else torch.device("cpu")
    values = torch.from_numpy(np.ascontiguousarray(normalized, dtype=np.float32)).to(device)
    genes = values.shape[1]
    mean_absolute = np.full(matrix.shape[1], np.nan, dtype=np.float64)
    count_over_half = np.full(matrix.shape[1], np.nan, dtype=np.float64)
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, genes, block_size):
            stop = min(start + block_size, genes)
            correlations = values[:, start:stop].T @ values
            local_rows = torch.arange(stop - start, device=device)
            correlations[local_rows, torch.arange(start, stop, device=device)] = 0.0
            absolute = correlations.abs()
            block_mean = absolute.sum(dim=1) / (genes - 1)
            block_count = (absolute > 0.5).sum(dim=1)
            model_rows = valid_indices[start:stop]
            mean_absolute[model_rows] = block_mean.cpu().numpy()
            count_over_half[model_rows] = block_count.cpu().numpy()
            elapsed = time.monotonic() - started
            print(f"[coexpression heartbeat] genes={stop:,}/{genes:,} "
                  f"elapsed={elapsed / 60:.1f}m device={device}", flush=True)
            del correlations, absolute, block_mean, block_count
    return mean_absolute, count_over_half, valid_indices, str(device)


def summarize_groups(table: pd.DataFrame) -> pd.DataFrame:
    groups = {
        "Shared-451": table.in_shared_451,
        "Top-921": table.in_top_921,
        "Remaining/background": table.in_background,
    }
    rows = []
    for name, mask in groups.items():
        subset = table.loc[mask]
        valid = subset.dropna(subset=["mean_absolute_correlation", "correlated_genes_abs_r_gt_0_5"])
        rows.append({"gene_set": name, "genes_in_set": int(mask.sum()),
                     "genes_with_coexpression": len(valid),
                     "median_mean_absolute_correlation": valid.mean_absolute_correlation.median(),
                     "median_correlated_genes_abs_r_gt_0_5":
                         valid.correlated_genes_abs_r_gt_0_5.median()})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--block-size", type=int, default=1024)
    args = parser.parse_args()

    ranking = pd.read_parquet(RANKING).sort_values("model_index")
    if len(ranking) != 15165 or ranking.gene.nunique() != 15165:
        raise AssertionError("Expected exactly 15,165 unique model genes")
    shared = set(pd.read_parquet(SHARED).gene.astype(str))
    if len(shared) != 451:
        raise AssertionError("Expected the frozen Shared-451 gene set")
    top = set(ranking.nsmallest(921, "information_rank").gene.astype(str))

    matrix, _, mapping = load_gtex_log1p_cpm(ranking.gene.astype(str).tolist())
    mean_absolute, count_over_half, valid_indices, device = coexpression_summaries(
        matrix, args.device, args.block_size)
    table = ranking[["gene", "model_index", "information_score", "information_rank"]].copy()
    table["mean_absolute_correlation"] = mean_absolute
    table["correlated_genes_abs_r_gt_0_5"] = pd.array(count_over_half, dtype="Int64")
    table["coexpression_partner_genes"] = pd.array(
        np.where(np.isfinite(mean_absolute), len(valid_indices) - 1, np.nan), dtype="Int64")
    table["in_shared_451"] = table.gene.isin(shared)
    table["in_top_921"] = table.gene.isin(top)
    table["in_background"] = ~table.in_top_921
    table.to_parquet(RESULTS / "gtex_gene_coexpression_summary.parquet", index=False)

    group_summary = summarize_groups(table)
    group_summary.to_parquet(RESULTS / "gtex_coexpression_group_summary.parquet", index=False)
    group_summary.to_csv(RESULTS / "gtex_coexpression_group_summary.csv", index=False)

    rows = []
    for measure in ("mean_absolute_correlation", "correlated_genes_abs_r_gt_0_5"):
        valid = table[[measure, "information_score"]].dropna()
        result = spearmanr(valid[measure], valid.information_score)
        rows.append({"coexpression_measure": measure, "genes": len(valid),
                     "spearman_rho": result.statistic, "p_value": result.pvalue})
    correlations = pd.DataFrame(rows)
    correlations.to_parquet(RESULTS / "gtex_coexpression_score_correlations.parquet", index=False)
    correlations.to_csv(RESULTS / "gtex_coexpression_score_correlations.csv", index=False)

    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "gtex_h5": str(GTEX_H5),
        "model_gene_universe": len(ranking), "gtex_samples": matrix.shape[0],
        "genes_with_valid_coexpression": len(valid_indices),
        "partners_per_valid_gene": len(valid_indices) - 1,
        "preprocessing": "sample-wise counts per million followed by natural log1p",
        "correlation": "Pearson across GTEx samples; self-correlation excluded",
        "threshold": "absolute Pearson r > 0.5", "compute_device": device,
        "block_size": args.block_size, "ranking": str(RANKING), "shared_gene_set": str(SHARED),
        "missing_or_constant_genes_retained_as_null": int(len(ranking) - len(valid_indices)),
    }
    (RESULTS / "gtex_coexpression_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n")
    print("\nGroup medians\n" + group_summary.to_string(index=False), flush=True)
    print("\nSpearman correlations\n" + correlations.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
