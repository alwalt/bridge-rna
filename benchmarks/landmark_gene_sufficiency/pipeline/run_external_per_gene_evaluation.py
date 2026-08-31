#!/usr/bin/env python3
"""Exploratory per-gene reconstruction scoring for frozen external panels."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

from common import CONFIG, REPO_ROOT, RESULTS, WORK

sys.path.insert(0, str(REPO_ROOT / "src"))
from fm_embed.model import load_expression_performer
from fm_embed.reconstruction import mask_except_panel, reconstruct


DATASETS = ("tcga_human", "osdr_mouse")
PANEL_IDS = (
    "human_top", "mouse_top", "human_bottom", "mouse_bottom", "l1000",
    "random_921_r00", "shared_451", "random_451_r00",
)


def load_selected_panels() -> pd.DataFrame:
    """Load only pre-existing frozen panels; never derive or rerank a panel here."""
    old = pd.read_parquet(RESULTS / "frozen_validation_panels.parquet")
    shared = pd.read_parquet(RESULTS / "frozen_shared_451_validation_panels.parquet")
    panels = pd.concat([old, shared], ignore_index=True, sort=False)
    panels = panels[panels.panel_id.isin(PANEL_IDS)].copy()
    found = set(panels.panel_id.unique())
    if found != set(PANEL_IDS):
        raise RuntimeError(f"Frozen panel mismatch; missing={sorted(set(PANEL_IDS)-found)}")
    counts = panels.groupby("panel_id").model_index.agg(["size", "nunique"])
    if not counts["size"].eq(counts["nunique"]).all():
        raise ValueError("A frozen panel contains duplicate model indices")
    expected = {panel: (451 if "451" in panel else 921) for panel in PANEL_IDS}
    actual = counts["size"].to_dict()
    if actual != expected:
        raise ValueError(f"Frozen panel sizes changed: {actual}")
    return panels


def columnwise_metrics(truth: np.ndarray, prediction: np.ndarray,
                       indices: np.ndarray, block_size: int = 512) -> pd.DataFrame:
    """Score each masked gene across samples, with blockwise rank computation."""
    rows = []
    for start in range(0, len(indices), block_size):
        idx = indices[start:start + block_size]
        left = np.asarray(truth[:, idx], dtype=np.float64)
        right = np.asarray(prediction[:, idx], dtype=np.float64)
        lc, rc = left - left.mean(0), right - right.mean(0)
        denom = np.sqrt((lc * lc).sum(0) * (rc * rc).sum(0))
        pearson = np.divide((lc * rc).sum(0), denom,
                            out=np.full(len(idx), np.nan), where=denom > 0)
        left_rank, right_rank = rankdata(left, axis=0), rankdata(right, axis=0)
        lrc = left_rank - left_rank.mean(0); rrc = right_rank - right_rank.mean(0)
        rank_denom = np.sqrt((lrc * lrc).sum(0) * (rrc * rrc).sum(0))
        spearman = np.divide((lrc * rrc).sum(0), rank_denom,
                             out=np.full(len(idx), np.nan), where=rank_denom > 0)
        rows.append(pd.DataFrame({"model_index": idx, "pearson": pearson,
            "spearman": spearman, "mse": np.mean((left - right) ** 2, axis=0),
            "truth_mean": left.mean(0), "truth_sd": left.std(0, ddof=1),
            "prediction_mean": right.mean(0), "prediction_sd": right.std(0, ddof=1)}))
    return pd.concat(rows, ignore_index=True)


def worker(dataset: str, device_name: str) -> None:
    panels = load_selected_panels()
    matrix = np.load(WORK / f"external_{dataset}_log1p_tpm.npy", mmap_mode="r")
    genes = pd.read_parquet(WORK / f"external_{dataset}_genes.parquet")
    gene_lookup = genes.set_index("model_index")
    eligible = set(genes.loc[genes.observed, "model_index"].astype(int))
    device = torch.device(device_name if device_name.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model, device = load_expression_performer(REPO_ROOT / "model/r7hnr92k/best_model.pt",
        REPO_ROOT / "model/r7hnr92k/config.json", 15165, str(device))
    cache = WORK / "external_per_gene_conditions" / dataset
    cache.mkdir(parents=True, exist_ok=True)
    manifest = panels[["panel_id", "panel_type", "selection_species", "replicate"]].drop_duplicates()
    for panel in manifest.itertuples(index=False):
        output = cache / f"{panel.panel_id}.parquet"
        if output.exists():
            print(f"[{dataset}] reuse {output.name}", flush=True)
            continue
        visible = np.sort(panels.loc[panels.panel_id.eq(panel.panel_id), "model_index"].to_numpy(int))
        score_indices = np.asarray(sorted(eligible - set(visible)), dtype=int)
        prediction = reconstruct(model, mask_except_panel(matrix, visible, float(CONFIG["mask_token"])),
            device, int(CONFIG["batch_size"]), f"{dataset} {panel.panel_id} per-gene")
        result = columnwise_metrics(matrix, prediction, score_indices)
        result.insert(0, "gene", result.model_index.map(gene_lookup.gene))
        result.insert(0, "replicate", panel.replicate)
        result.insert(0, "selection_species", panel.selection_species)
        result.insert(0, "panel_type", panel.panel_type)
        result.insert(0, "panel_id", panel.panel_id)
        result.insert(0, "dataset", dataset)
        result["samples"] = len(matrix)
        result["visible_genes"] = len(visible)
        result["masked_observed_genes"] = len(score_indices)
        result.to_parquet(output, index=False)
        print(f"[{dataset}] completed {panel.panel_id}: {len(result):,} masked genes", flush=True)


def aggregate() -> None:
    files = sorted((WORK / "external_per_gene_conditions").glob("*/*.parquet"))
    expected = len(DATASETS) * len(PANEL_IDS)
    if len(files) != expected:
        raise RuntimeError(f"Expected {expected} cache files, found {len(files)}")
    per_gene = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    per_gene.to_parquet(RESULTS / "external_per_gene_results.parquet", index=False)
    summary = per_gene.groupby(["dataset", "panel_id", "panel_type", "selection_species"],
        as_index=False, dropna=False).agg(
        genes_scored=("gene", "size"), samples=("samples", "first"),
        pearson_mean=("pearson", "mean"), pearson_median=("pearson", "median"),
        spearman_mean=("spearman", "mean"), spearman_median=("spearman", "median"),
        mse_mean=("mse", "mean"), mse_median=("mse", "median"))
    summary.to_parquet(RESULTS / "external_per_gene_summary.parquet", index=False)
    summary.to_csv(RESULTS / "external_per_gene_summary.csv", index=False)
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--worker-dataset", choices=DATASETS)
    parser.add_argument("--worker-device")
    args = parser.parse_args()
    if args.worker_dataset:
        worker(args.worker_dataset, args.worker_device or "cuda:0")
        return
    processes = []
    for index, dataset in enumerate(args.datasets):
        command = [sys.executable, str(Path(__file__).resolve()), "--worker-dataset", dataset,
                   "--worker-device", args.devices[index % len(args.devices)]]
        processes.append(subprocess.Popen(command))
    for process in processes:
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, process.args)
    aggregate()


if __name__ == "__main__":
    main()
