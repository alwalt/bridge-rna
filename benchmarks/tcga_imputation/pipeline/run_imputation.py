#!/usr/bin/env python3
"""Validate and run shared/native-vocabulary TCGA imputation benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import CONFIG, RESULTS, WORK, exact_mask, row_metrics
from model_adapters import predict, resolved_device


MODELS = ("ours_45.6m", "bulkformer_50m", "bulkformer_147m")
MODEL_SPACE = {
    "ours_45.6m": "ours",
    "bulkformer_50m": "bulkformer",
    "bulkformer_147m": "bulkformer",
}


def load_inputs():
    samples = pd.read_parquet(WORK / "selected_tcga_samples.parquet")
    ours = np.load(WORK / "ours_log1p_tpm.npy", mmap_mode="r")
    bulk = np.load(WORK / "bulkformer_log1p_tpm.npy", mmap_mode="r")
    our_genes = pd.read_parquet(WORK / "ours_genes.parquet")
    bulk_genes = pd.read_parquet(WORK / "bulkformer_genes.parquet")
    shared = pd.read_parquet(WORK / "shared_genes.parquet")
    return samples, ours, bulk, our_genes, bulk_genes, shared


def native_mask(sample_ids: list[str], gene_table: pd.DataFrame,
                ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    eligible = gene_table["tcga_observed"].to_numpy(bool)
    genes = gene_table.loc[eligible, "gene"].astype(str).tolist()
    compact = exact_mask(sample_ids, genes, ratio, seed)
    native = np.zeros((len(sample_ids), len(gene_table)), dtype=bool)
    native[:, eligible] = compact
    return native, eligible


def shared_native_masks(sample_ids: list[str], shared: pd.DataFrame,
                        ratio: float, seed: int, our_size: int, bulk_size: int):
    compact = exact_mask(sample_ids, shared["gene"].astype(str).tolist(), ratio, seed)
    ours = np.zeros((len(sample_ids), our_size), dtype=bool)
    bulk = np.zeros((len(sample_ids), bulk_size), dtype=bool)
    ours[:, shared["ours_index"].to_numpy(int)] = compact
    bulk[:, shared["bulkformer_index"].to_numpy(int)] = compact
    return compact, ours, bulk


def apply_mask(matrix: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.asarray(matrix, dtype=np.float32).copy()
    result[mask] = float(CONFIG["mask_token"])
    return result


def score_prediction(sample_ids: list[str], truth: np.ndarray, prediction: np.ndarray,
                     mask: np.ndarray, **labels) -> pd.DataFrame:
    per_row = mask.sum(axis=1)
    if len(set(per_row.tolist())) != 1:
        raise ValueError("Masks must contain the same number of evaluated genes per sample")
    count = int(per_row[0])
    y = np.asarray(truth)[mask].reshape(len(truth), count)
    yhat = np.asarray(prediction)[mask].reshape(len(truth), count)
    pearson, spearman, mse = row_metrics(y, yhat)
    result = pd.DataFrame({"sample_id": sample_ids, "pearson": pearson,
                           "spearman": spearman, "mse": mse,
                           "masked_genes": count})
    for key, value in labels.items():
        result[key] = value
    return result


def load_frozen_baselines() -> dict[str, np.ndarray]:
    path = WORK / "frozen_training_gene_baselines.npz"
    if not path.is_file():
        raise RuntimeError(
            "Missing frozen training baselines. Run pipeline/prepare_frozen_baselines.py first."
        )
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def validate_one() -> None:
    samples, ours, bulk, our_genes, bulk_genes, shared = load_inputs()
    sample_ids = samples["sample_id"].astype(str).tolist()[:1]
    ratio, seed = float(CONFIG["mask_ratios"][0]), int(CONFIG["mask_seeds"][0])
    compact, our_mask, bulk_mask = shared_native_masks(
        sample_ids, shared, ratio, seed, ours.shape[1], bulk.shape[1]
    )
    device = resolved_device()
    report = {"status": "running", "sample_id": sample_ids[0], "mask_ratio": ratio,
              "mask_seed": seed, "shared_genes": len(shared), "device": str(device),
              "benchmarks": {}}
    for mode in ("shared_vocab", "native_vocab"):
        mode_report = {"models": {}}
        for model_name in MODELS:
            matrix = ours[:1] if MODEL_SPACE[model_name] == "ours" else bulk[:1]
            if mode == "shared_vocab":
                mask = our_mask if MODEL_SPACE[model_name] == "ours" else bulk_mask
            else:
                table = our_genes if MODEL_SPACE[model_name] == "ours" else bulk_genes
                mask, _ = native_mask(sample_ids, table, ratio, seed)
            masked = apply_mask(matrix, mask)
            actual_ratio = float(np.mean(masked == float(CONFIG["mask_token"])))
            output = predict(model_name, masked, actual_ratio, device,
                             int(CONFIG["batch_sizes"][model_name]))
            if output.shape != matrix.shape or not np.isfinite(output).all():
                raise ValueError(f"{model_name} invalid output: shape={output.shape}")
            metric = score_prediction(sample_ids, matrix, output, mask).iloc[0]
            mode_report["models"][model_name] = {
                "input_shape": list(masked.shape), "output_shape": list(output.shape),
                "masked_native_positions": int(mask.sum()), "mask_prob_passed": actual_ratio,
                "finite_output": True, "pearson": float(metric.pearson),
                "spearman": float(metric.spearman), "mse": float(metric.mse),
            }
        report["benchmarks"][mode] = mode_report
    report["status"] = "passed"
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "one_sample_validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def condition_masks(mode: str, model_name: str, sample_ids: list[str], ratio: float, seed: int,
                    our_genes: pd.DataFrame, bulk_genes: pd.DataFrame, shared: pd.DataFrame):
    if mode == "shared_vocab":
        _, our_mask, bulk_mask = shared_native_masks(
            sample_ids, shared, ratio, seed, len(our_genes), len(bulk_genes)
        )
        if MODEL_SPACE[model_name] == "ours":
            eligible = np.zeros(len(our_genes), bool); eligible[shared.ours_index] = True
            return our_mask, eligible
        eligible = np.zeros(len(bulk_genes), bool); eligible[shared.bulkformer_index] = True
        return bulk_mask, eligible
    table = our_genes if MODEL_SPACE[model_name] == "ours" else bulk_genes
    return native_mask(sample_ids, table, ratio, seed)


def run_full(mask_ratios: list[float] | None = None,
             mask_seeds: list[int] | None = None,
             models: list[str] | None = None,
             baselines_only: bool = False) -> None:
    validation = RESULTS / "one_sample_validation.json"
    if not validation.is_file() or json.loads(validation.read_text()).get("status") != "passed":
        raise RuntimeError("Run --validate-one successfully before --run")
    samples, ours, bulk, our_genes, bulk_genes, shared = load_inputs()
    frozen_baselines = load_frozen_baselines()
    sample_ids = samples["sample_id"].astype(str).tolist()
    condition_dir = WORK / "condition_results"
    condition_dir.mkdir(parents=True, exist_ok=True)
    ratios = list(map(float, mask_ratios or CONFIG["mask_ratios"]))
    seeds = list(map(int, mask_seeds or CONFIG["mask_seeds"]))
    selected_models = tuple(models or MODELS)
    if any(ratio <= 0 or ratio > 1 for ratio in ratios):
        raise ValueError(f"Mask ratios must be in (0, 1], received {ratios}")
    for mode in ("shared_vocab", "native_vocab"):
        for ratio in ratios:
            for seed in seeds:
                baseline_done = set()
                for model_name in selected_models:
                    space = MODEL_SPACE[model_name]
                    tag = f"{mode}__{model_name}__{space}__r{int(round(ratio * 100)):02d}__s{seed:02d}"
                    model_output = condition_dir / f"{tag}.parquet"
                    truth = ours if MODEL_SPACE[model_name] == "ours" else bulk
                    mask, eligible = condition_masks(
                        mode, model_name, sample_ids, ratio, seed,
                        our_genes, bulk_genes, shared,
                    )
                    if not baselines_only:
                        if model_output.is_file():
                            print(f"reuse {model_output.name}")
                        else:
                            masked = apply_mask(truth, mask)
                            actual_ratio = float(np.mean(masked == float(CONFIG["mask_token"])))
                            output = predict(model_name, masked, actual_ratio, resolved_device(),
                                             int(CONFIG["batch_sizes"][model_name]))
                            score_prediction(
                                sample_ids, truth, output, mask, benchmark=mode, method=model_name,
                                target_space=space, mask_ratio=ratio, mask_seed=seed,
                                native_genes=truth.shape[1], evaluated_genes=int(eligible.sum()),
                            ).to_parquet(model_output, index=False)
                    # One baseline result per native target space; both BulkFormer
                    # scales share the same truth and mask.
                    if space not in baseline_done:
                        for statistic in ("mean", "median"):
                            baseline_tag = (f"{mode}__train_gene_{statistic}__{space}__"
                                            f"r{int(round(ratio * 100)):02d}__s{seed:02d}")
                            baseline_output = condition_dir / f"{baseline_tag}.parquet"
                            if not baseline_output.is_file():
                                center = frozen_baselines[f"{space}_{statistic}"]
                                baseline = np.broadcast_to(center, truth.shape)
                                score_prediction(
                                    sample_ids, truth, baseline, mask, benchmark=mode,
                                    method=f"train_gene_{statistic}", target_space=space,
                                    mask_ratio=ratio, mask_seed=seed, native_genes=truth.shape[1],
                                    evaluated_genes=int(eligible.sum()),
                                ).to_parquet(baseline_output, index=False)
                        baseline_done.add(space)
                print(f"completed {mode} ratio={ratio:.2f} seed={seed}")
    condition_files = sorted(condition_dir.glob("*.parquet"))
    requested = {
        (mode, ratio, seed) for mode in ("shared_vocab", "native_vocab")
        for ratio in ratios for seed in seeds
    }
    completed = set()
    for path in condition_files:
        row = pd.read_parquet(path, columns=["benchmark", "mask_ratio", "mask_seed"]).iloc[0]
        completed.add((str(row.benchmark), float(row.mask_ratio), int(row.mask_seed)))
    missing = requested - completed
    if missing:
        raise RuntimeError(f"Requested conditions were not completed: {sorted(missing)}")
    per_sample = pd.concat((pd.read_parquet(path) for path in condition_files), ignore_index=True)
    # Legacy baselines estimated their centers from TCGA itself. Keep their
    # cache files for provenance, but never include them in regenerated tables.
    per_sample = per_sample.loc[~per_sample.method.isin(["gene_mean", "gene_median"])].copy()
    per_sample.to_parquet(RESULTS / "per_sample_results.parquet", index=False)
    per_seed = (per_sample.groupby(
        ["benchmark", "method", "target_space", "mask_ratio", "mask_seed"], as_index=False
    ).agg(pearson=("pearson", "mean"), spearman=("spearman", "mean"),
          mse=("mse", "mean"), samples=("sample_id", "size"),
          evaluated_genes=("evaluated_genes", "first")))
    per_seed.to_parquet(RESULTS / "per_seed_results.parquet", index=False)
    summary = (per_seed.groupby(
        ["benchmark", "method", "target_space", "mask_ratio"], as_index=False
    ).agg(pearson_mean=("pearson", "mean"), pearson_sd=("pearson", "std"),
          spearman_mean=("spearman", "mean"), spearman_sd=("spearman", "std"),
          mse_mean=("mse", "mean"), mse_sd=("mse", "std"),
          seeds=("mask_seed", "size"), evaluated_genes=("evaluated_genes", "first")))
    summary.to_csv(RESULTS / "summary_results.csv", index=False)
    summary.to_parquet(RESULTS / "summary_results.parquet", index=False)
    print(summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--validate-one", action="store_true")
    group.add_argument("--run", action="store_true")
    parser.add_argument("--mask-ratios", type=float, nargs="+", default=None,
                        help="Optional append-only ratios; e.g. --mask-ratios 0.7 0.9 1.0")
    parser.add_argument("--mask-seeds", type=int, nargs="+", default=None,
                        help="Optional seeds for this run; existing condition files are reused")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=None,
                        help="Optional model subset; e.g. --models ours_45.6m")
    parser.add_argument("--baselines-only", action="store_true",
                        help="Generate frozen baseline conditions without model inference")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    validate_one() if args.validate_one else run_full(
        args.mask_ratios, args.mask_seeds, args.models, args.baselines_only
    )
