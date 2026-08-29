#!/usr/bin/env python3
"""Run frozen-model reconstruction on unused ARCHS4 human and mouse samples."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import CONFIG, RESULTS, WORK, exact_mask
from model_adapters import predict, resolved_device
from run_imputation import apply_mask, score_prediction


DATASETS = ("archs4_holdout_human", "archs4_holdout_mouse")
DEFAULT_RATIOS = (0.15, 0.30, 0.50, 0.70, 0.90, 1.00)
DEFAULT_SEEDS = tuple(range(10))


def load_dataset(dataset: str):
    matrix = np.load(WORK / f"{dataset}_log1p_tpm.npy", mmap_mode="r")
    genes = pd.read_parquet(WORK / f"{dataset}_genes.parquet")
    selected = pd.read_parquet(RESULTS / "archs4_holdout_selected_samples.parquet")
    sample_ids = selected.loc[selected.dataset.eq(dataset), "sample_id"].astype(str).tolist()
    if len(sample_ids) != len(matrix):
        raise ValueError(f"{dataset}: {len(sample_ids)} IDs but {len(matrix)} matrix rows")
    return matrix, genes, sample_ids


def dataset_mask(sample_ids: list[str], genes: pd.DataFrame,
                 ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    eligible = genes.observed.to_numpy(bool)
    compact = exact_mask(sample_ids, genes.loc[eligible, "gene_index"].astype(str).tolist(), ratio, seed)
    mask = np.zeros((len(sample_ids), len(genes)), dtype=bool)
    mask[:, eligible] = compact
    return mask, eligible


def aggregate(condition_dir: Path) -> None:
    files = sorted(condition_dir.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"No ARCHS4 holdout results under {condition_dir}")
    per_sample = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    per_sample.to_parquet(RESULTS / "archs4_holdout_per_sample_results.parquet", index=False)
    per_seed = (per_sample.groupby(["dataset", "mask_ratio", "mask_seed"], as_index=False)
                .agg(pearson=("pearson", "mean"), spearman=("spearman", "mean"),
                     mse=("mse", "mean"), samples=("sample_id", "size"),
                     evaluated_genes=("evaluated_genes", "first")))
    per_seed.to_parquet(RESULTS / "archs4_holdout_per_seed_results.parquet", index=False)
    summary = (per_seed.groupby(["dataset", "mask_ratio"], as_index=False)
               .agg(pearson_mean=("pearson", "mean"), pearson_sd=("pearson", "std"),
                    spearman_mean=("spearman", "mean"), spearman_sd=("spearman", "std"),
                    mse_mean=("mse", "mean"), mse_sd=("mse", "std"),
                    seeds=("mask_seed", "size"), samples=("samples", "first"),
                    evaluated_genes=("evaluated_genes", "first")))
    summary.to_parquet(RESULTS / "archs4_holdout_summary_results.parquet", index=False)
    summary.to_csv(RESULTS / "archs4_holdout_summary_results.csv", index=False)
    print(summary.to_string(index=False))


def run(datasets: list[str], ratios: list[float], seeds: list[int]) -> None:
    condition_dir = WORK / "archs4_strict_unseen_condition_results"
    condition_dir.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        matrix, genes, sample_ids = load_dataset(dataset)
        for ratio in ratios:
            for seed in seeds:
                path = condition_dir / f"{dataset}__ours_45.6m__r{round(ratio*100):03d}__s{seed:02d}.parquet"
                if path.is_file():
                    print(f"reuse {path.name}", flush=True)
                    continue
                mask, eligible = dataset_mask(sample_ids, genes, ratio, seed)
                masked = apply_mask(matrix, mask)
                actual_ratio = float(np.mean(masked == float(CONFIG["mask_token"])))
                output = predict("ours_45.6m", masked, actual_ratio, resolved_device(),
                                 int(CONFIG["batch_sizes"]["ours_45.6m"]))
                score_prediction(
                    sample_ids, matrix, output, mask, dataset=dataset, method="ours_45.6m",
                    mask_ratio=ratio, mask_seed=seed, native_genes=matrix.shape[1],
                    evaluated_genes=int(eligible.sum()),
                ).to_parquet(path, index=False)
                print(f"completed {dataset} ratio={ratio:.2f} seed={seed}", flush=True)
    aggregate(condition_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--mask-ratios", nargs="+", type=float, default=list(DEFAULT_RATIOS))
    parser.add_argument("--mask-seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    args = parser.parse_args()
    run(args.datasets, args.mask_ratios, args.mask_seeds)


if __name__ == "__main__":
    main()
