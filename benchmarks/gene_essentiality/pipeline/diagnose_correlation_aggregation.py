#!/usr/bin/env python3
"""Diagnose correlation aggregation using immutable saved OOF predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
CONFIG = json.loads((BENCH / "config.json").read_text())
WORK = Path(CONFIG["shared_work_root"])
RESULTS = BENCH / "results"
PREDICTIONS = WORK / "predictions"
ANALYSIS = RESULTS / "analysis"
PUBLISHED_PCC = 0.186

REPRESENTATIONS = {
    "target_expression": "target_expression_matched",
    "pca_context": "pca_matched",
    "bridge_contextual": "bridge_matched",
    "bulkformer_contextual": "bulkformer_matched",
    "bridge_plus_pca": "bridge_pca_matched",
    "bulkformer_native": "bulkformer_native",
}


def correlation(prediction: np.ndarray, truth: np.ndarray, method: str) -> float:
    valid = np.isfinite(prediction) & np.isfinite(truth)
    x = prediction[valid]
    y = truth[valid]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    if method == "pearson":
        return float(pearsonr(x, y).statistic)
    return float(spearmanr(x, y).statistic)


def load_oof(directory: Path, dependency: np.ndarray):
    manifest = pd.read_csv(directory / "target_manifest.csv")
    columns = manifest.dependency_column_index.to_numpy(dtype=int)
    truth = np.asarray(dependency[:, columns], dtype=np.float32)
    prediction = np.full(truth.shape, np.nan, dtype=np.float32)
    fold_for_row = np.full(len(truth), -1, dtype=np.int16)
    seen = np.zeros(len(truth), dtype=np.int8)
    for fold in range(int(CONFIG["cv_folds"])):
        rows = pd.read_csv(directory / f"fold_{fold:02d}_test_rows.csv").source_row.to_numpy(dtype=int)
        values = np.load(directory / f"fold_{fold:02d}_predictions.float32.npy")
        if values.shape != (len(rows), len(manifest)):
            raise AssertionError(f"Unexpected prediction shape for {directory.name} fold {fold}")
        prediction[rows] = values
        fold_for_row[rows] = fold
        seen[rows] += 1
    if not np.all(seen == 1) or np.any(fold_for_row < 0):
        raise AssertionError(f"Incomplete OOF coverage for {directory.name}")
    return manifest, prediction, truth, fold_for_row


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    dependency = np.load(WORK / "dependency.float32.npy", mmap_mode="r")
    existing_gene_metrics = {
        name: pd.read_csv(ANALYSIS / f"{name}_per_gene_metrics.csv")
        for name in REPRESENTATIONS
    }
    summaries, cell_rows, fold_rows = [], [], []
    for name, directory_name in REPRESENTATIONS.items():
        print(f"[aggregation] {name}", flush=True)
        manifest, prediction, truth, fold_for_row = load_oof(
            PREDICTIONS / directory_name, dependency
        )
        gene_metrics = existing_gene_metrics[name]
        if len(gene_metrics) != len(manifest):
            raise AssertionError(f"Gene metric/manifest mismatch for {name}")

        cell_pcc = np.array([
            correlation(prediction[row], truth[row], "pearson")
            for row in range(len(truth))
        ])
        cell_scc = np.array([
            correlation(prediction[row], truth[row], "spearman")
            for row in range(len(truth))
        ])
        for row in range(len(truth)):
            cell_rows.append({
                "representation": name,
                "source_row": row,
                "fold": int(fold_for_row[row]),
                "valid_genes": int(np.sum(np.isfinite(prediction[row]) & np.isfinite(truth[row]))),
                "pcc": cell_pcc[row],
                "scc": cell_scc[row],
            })

        global_pcc = correlation(prediction, truth, "pearson")
        global_scc = correlation(prediction, truth, "spearman")
        representation_fold_rows = []
        for fold in range(int(CONFIG["cv_folds"])):
            rows = np.flatnonzero(fold_for_row == fold)
            item = {
                "representation": name,
                "fold": fold,
                "n_cell_lines": len(rows),
                "valid_pairs": int(np.sum(np.isfinite(prediction[rows]) & np.isfinite(truth[rows]))),
                "global_pcc": correlation(prediction[rows], truth[rows], "pearson"),
                "global_scc": correlation(prediction[rows], truth[rows], "spearman"),
            }
            fold_rows.append(item)
            representation_fold_rows.append(item)
        fold_frame = pd.DataFrame(representation_fold_rows)

        summary = {
            "representation": name,
            "gene_count": len(manifest),
            "cell_line_count": len(truth),
            "valid_pairs": int(np.sum(np.isfinite(prediction) & np.isfinite(truth))),
            "mean_per_gene_pcc": gene_metrics.pcc.mean(),
            "median_per_gene_pcc": gene_metrics.pcc.median(),
            "mean_per_gene_scc": gene_metrics.scc.mean(),
            "median_per_gene_scc": gene_metrics.scc.median(),
            "mean_per_cell_line_pcc": np.nanmean(cell_pcc),
            "median_per_cell_line_pcc": np.nanmedian(cell_pcc),
            "mean_per_cell_line_scc": np.nanmean(cell_scc),
            "median_per_cell_line_scc": np.nanmedian(cell_scc),
            "global_pcc": global_pcc,
            "global_scc": global_scc,
            "mean_fold_global_pcc": fold_frame.global_pcc.mean(),
            "sd_fold_global_pcc": fold_frame.global_pcc.std(ddof=1),
            "mean_fold_global_scc": fold_frame.global_scc.mean(),
            "sd_fold_global_scc": fold_frame.global_scc.std(ddof=1),
        }
        pcc_fields = [
            "mean_per_gene_pcc", "median_per_gene_pcc",
            "mean_per_cell_line_pcc", "median_per_cell_line_pcc",
            "global_pcc", "mean_fold_global_pcc",
        ]
        nearest = min(pcc_fields, key=lambda field: abs(summary[field] - PUBLISHED_PCC))
        summary["nearest_pcc_aggregation_to_0_186"] = nearest
        summary["nearest_pcc_value"] = summary[nearest]
        summary["absolute_distance_from_0_186"] = abs(summary[nearest] - PUBLISHED_PCC)
        summaries.append(summary)

    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(ANALYSIS / "correlation_aggregation_summary.csv", index=False)
    pd.DataFrame(cell_rows).to_csv(ANALYSIS / "per_cell_line_correlations.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(ANALYSIS / "fold_global_correlations.csv", index=False)
    compact = summary_frame[[
        "representation", "mean_per_gene_pcc", "mean_per_cell_line_pcc",
        "global_pcc", "mean_fold_global_pcc",
    ]]
    print(compact.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
