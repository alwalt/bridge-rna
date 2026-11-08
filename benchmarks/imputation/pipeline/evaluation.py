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






