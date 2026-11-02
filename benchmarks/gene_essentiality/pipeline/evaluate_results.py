#!/usr/bin/env python3
"""Assemble OOF predictions, compute per-gene metrics, paired tests, and figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
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
FIGURES = RESULTS / "figures"


REPRESENTATIONS = {
    "target_expression": "target_expression_matched",
    "pca_context": "pca_matched",
    "bridge_contextual": "bridge_matched",
    "bulkformer_contextual": "bulkformer_matched",
    "bridge_plus_pca": "bridge_pca_matched",
    "bulkformer_native": "bulkformer_native",
}


def safe_correlation(x: np.ndarray, y: np.ndarray, kind: str) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.ptp(x[valid]) == 0 or np.ptp(y[valid]) == 0:
        return np.nan
    return float(pearsonr(x[valid], y[valid]).statistic if kind == "pearson"
                 else spearmanr(x[valid], y[valid]).statistic)


def assemble(directory: Path, dependency: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    manifest = pd.read_csv(directory / "target_manifest.csv")
    target_columns = manifest.dependency_column_index.to_numpy(dtype=int)
    prediction = np.full((len(dependency), len(manifest)), np.nan, dtype=np.float32)
    seen = np.zeros(len(dependency), dtype=int)
    for fold in range(int(CONFIG["cv_folds"])):
        rows = pd.read_csv(directory / f"fold_{fold:02d}_test_rows.csv").source_row.to_numpy(dtype=int)
        values = np.load(directory / f"fold_{fold:02d}_predictions.float32.npy")
        if values.shape != (len(rows), len(manifest)):
            raise AssertionError(f"Shape mismatch in {directory.name} fold {fold}: {values.shape}")
        prediction[rows] = values; seen[rows] += 1
    if not np.all(seen == 1):
        raise AssertionError(f"OOF coverage failure in {directory.name}: {np.unique(seen, return_counts=True)}")
    truth = np.asarray(dependency[:, target_columns], dtype=np.float32)
    return manifest, prediction, truth


def per_gene_metrics(name: str, manifest: pd.DataFrame, prediction: np.ndarray,
                     truth: np.ndarray) -> pd.DataFrame:
    rows = []
    for gene in range(len(manifest)):
        valid = np.isfinite(truth[:, gene]) & np.isfinite(prediction[:, gene])
        rows.append({
            "representation": name,
            "gene_id": manifest.iloc[gene].gene_id,
            "gene_symbol": manifest.iloc[gene].get("bridge_gene_symbol", ""),
            "n_observed": int(valid.sum()),
            "pcc": safe_correlation(prediction[:, gene], truth[:, gene], "pearson"),
            "scc": safe_correlation(prediction[:, gene], truth[:, gene], "spearman"),
            "truth_mean": float(np.nanmean(truth[:, gene])),
            "truth_variance": float(np.nanvar(truth[:, gene], ddof=1)),
            "prediction_mean": float(np.nanmean(prediction[:, gene])),
            "prediction_variance": float(np.nanvar(prediction[:, gene], ddof=1)),
        })
    return pd.DataFrame(rows)


def training_statistics(manifest: pd.DataFrame, expression: np.ndarray,
                        dependency: np.ndarray, folds: pd.DataFrame) -> pd.DataFrame:
    stats = []
    for gene in manifest.itertuples():
        fold_values = []
        for fold in range(int(CONFIG["cv_folds"])):
            train = folds[(folds.fold == fold) & (folds.split == "train")].row_index.to_numpy(dtype=int)
            expr = expression[train, int(gene.expression_column_index)]
            dep = dependency[train, int(gene.dependency_column_index)]
            fold_values.append((float(np.mean(expr)), float(np.var(expr, ddof=1)), float(np.nanvar(dep, ddof=1))))
        values = np.asarray(fold_values)
        stats.append({"gene_id": gene.gene_id, "train_expression_mean": values[:, 0].mean(),
                      "train_expression_variance": values[:, 1].mean(),
                      "train_dependency_variance": values[:, 2].mean()})
    return pd.DataFrame(stats)


def bootstrap_delta(delta: np.ndarray, seed: int = 42) -> tuple[float, float]:
    delta = delta[np.isfinite(delta)]
    rng = np.random.default_rng(seed)
    means = np.empty(10_000)
    for start in range(0, len(means), 100):
        stop = min(start + 100, len(means))
        indices = rng.integers(0, len(delta), size=(stop - start, len(delta)))
        means[start:stop] = delta[indices].mean(axis=1)
    return tuple(np.quantile(means, [.025, .975]))


def sign_flip_p(delta: np.ndarray, seed: int = 42) -> float:
    delta = delta[np.isfinite(delta)]
    observed = abs(delta.mean())
    rng = np.random.default_rng(seed)
    exceed = 0; permutations = 100_000
    for _ in range(permutations // 1000):
        signs = rng.choice(np.array([-1.0, 1.0]), size=(1000, len(delta)))
        exceed += int(np.sum(np.abs((signs * delta).mean(axis=1)) >= observed))
    return (exceed + 1) / (permutations + 1)


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    dependency = np.load(WORK / "dependency.float32.npy", mmap_mode="r")
    expression = np.load(WORK / "expression_released_log2_tpm_plus_1.float32.npy", mmap_mode="r")
    folds = pd.read_csv(RESULTS / "cv_folds.csv")
    all_metrics = []
    for name, directory_name in REPRESENTATIONS.items():
        directory = PREDICTIONS / directory_name
        if not all((directory / f"fold_{fold:02d}_predictions.float32.npy").exists()
                   for fold in range(int(CONFIG["cv_folds"]))):
            print(f"[skip incomplete] {name}", flush=True); continue
        manifest, predictions, truth = assemble(directory, dependency)
        metrics = per_gene_metrics(name, manifest, predictions, truth)
        metrics.to_csv(ANALYSIS / f"{name}_per_gene_metrics.csv", index=False)
        all_metrics.append(metrics)
    if not all_metrics:
        raise RuntimeError("No complete prediction sets")
    combined = pd.concat(all_metrics, ignore_index=True)
    combined.to_csv(ANALYSIS / "per_gene_metrics.csv", index=False)
    summary = combined.groupby("representation").agg(
        gene_count=("gene_id", "size"), evaluable_pcc=("pcc", "count"),
        mean_pcc=("pcc", "mean"), median_pcc=("pcc", "median"),
        mean_scc=("scc", "mean"), median_scc=("scc", "median"),
        fraction_pcc_positive=("pcc", lambda x: float(np.mean(x.dropna() > 0))),
    ).reset_index()
    summary.to_csv(RESULTS / "primary_metrics.csv", index=False)

    common_manifest = pd.read_csv(RESULTS / "gene_overlap.csv")
    common_manifest = common_manifest[common_manifest.all_method_common].reset_index(drop=True)
    train_stats = training_statistics(common_manifest, expression, dependency, folds)
    common_metrics = combined[combined.gene_id.isin(common_manifest.gene_id)].merge(train_stats, on="gene_id")
    variance_rows = []
    for name, group in common_metrics.groupby("representation"):
        for metric in ("pcc", "scc"):
            for covariate in ("train_expression_mean", "train_expression_variance", "train_dependency_variance"):
                variance_rows.append({"representation": name, "metric": metric, "covariate": covariate,
                    "spearman_correlation": safe_correlation(group[metric].to_numpy(), group[covariate].to_numpy(), "spearman")})
    pd.DataFrame(variance_rows).to_csv(ANALYSIS / "variance_control_correlations.csv", index=False)

    wide = common_metrics.pivot(index="gene_id", columns="representation", values="pcc")
    comparisons = []
    for left, right in (("bridge_contextual", "bulkformer_contextual"),
                        ("bridge_contextual", "pca_context"),
                        ("bridge_plus_pca", "bridge_contextual"),
                        ("bridge_plus_pca", "pca_context")):
        if left not in wide or right not in wide:
            continue
        delta = (wide[left] - wide[right]).dropna().to_numpy()
        low, high = bootstrap_delta(delta)
        comparisons.append({"comparison": f"{left}_minus_{right}", "n_genes": len(delta),
            "mean_delta_pcc": delta.mean(), "median_delta_pcc": np.median(delta),
            "ci95_mean_low": low, "ci95_mean_high": high,
            "fraction_left_greater": np.mean(delta > 0), "fraction_right_greater": np.mean(delta < 0),
            "paired_sign_flip_p": sign_flip_p(delta)})
    pd.DataFrame(comparisons).to_csv(ANALYSIS / "paired_comparisons.csv", index=False)

    if "bridge_contextual" in wide and "bulkformer_contextual" in wide:
        figure, axis = plt.subplots(figsize=(6.5, 6.5))
        valid = wide[["bulkformer_contextual", "bridge_contextual"]].dropna()
        axis.scatter(valid.bulkformer_contextual, valid.bridge_contextual, s=5, alpha=.25, rasterized=True)
        limits = [min(valid.min()) - .02, max(valid.max()) + .02]
        axis.plot(limits, limits, color="black", linewidth=1, linestyle="--")
        axis.set(xlabel="BulkFormer local per-gene PCC", ylabel="Bridge local per-gene PCC",
                 xlim=limits, ylim=limits, title=f"Matched genes (n={len(valid):,})")
        figure.tight_layout(); figure.savefig(FIGURES / "bridge_vs_bulkformer_pcc.png", dpi=300)
        figure.savefig(FIGURES / "bridge_vs_bulkformer_pcc.pdf"); plt.close(figure)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
