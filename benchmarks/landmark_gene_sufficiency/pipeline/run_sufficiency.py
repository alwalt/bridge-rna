#!/usr/bin/env python3
"""Run resumable fixed-panel sufficiency reconstruction with the frozen model."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import CONFIG, REPO_ROOT, RESULTS, WORK

sys.path.insert(0, str(REPO_ROOT / "src"))
from fm_embed.model import load_expression_performer
from fm_embed.reconstruction import mask_except_panel, reconstruct, score_masked_rows


DATASETS = ("archs4_holdout_human", "archs4_holdout_mouse")


def aggregate(condition_dir: Path, prefix: str) -> None:
    files = sorted(condition_dir.glob("*.parquet"))
    per_sample = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    per_sample.to_parquet(RESULTS / f"{prefix}per_sample_results.parquet", index=False)
    per_panel = (per_sample.groupby(
        ["dataset", "panel_type", "panel_id", "visible_gene_count", "replicate", "seed"],
        as_index=False, dropna=False
    ).agg(pearson=("pearson", "mean"), spearman=("spearman", "mean"),
          mse=("mse", "mean"), samples=("sample_id", "size"),
          masked_genes=("masked_genes", "first")))
    per_panel.to_parquet(RESULTS / f"{prefix}per_panel_results.parquet", index=False)
    summary = (per_panel.groupby(
        ["dataset", "panel_type", "visible_gene_count"], as_index=False
    ).agg(pearson_mean=("pearson", "mean"), pearson_sd=("pearson", "std"),
          spearman_mean=("spearman", "mean"), spearman_sd=("spearman", "std"),
          mse_mean=("mse", "mean"), mse_sd=("mse", "std"),
          replicates=("panel_id", "size"), samples=("samples", "first"),
          masked_genes=("masked_genes", "first")))
    summary.to_parquet(RESULTS / f"{prefix}summary_results.parquet", index=False)
    summary.to_csv(RESULTS / f"{prefix}summary_results.csv", index=False)
    print(summary.to_string(index=False))


def run(pilot: bool, panel_types: list[str] | None) -> None:
    panels = pd.read_parquet(RESULTS / "panel_manifest.parquet")
    panel_genes = pd.read_parquet(RESULTS / "panel_genes.parquet")
    if panel_types:
        panels = panels.loc[panels.panel_type.isin(panel_types)]
    if pilot:
        panels = panels.loc[panels.panel_type.eq("l1000") | panels.replicate.eq(0)]
    if panels.empty:
        raise ValueError("No panels match the requested filters")
    cohorts = pd.read_parquet(RESULTS / "cohort_manifest.parquet")
    device_name = str(CONFIG["device"])
    device = torch.device(device_name if device_name.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model, _ = load_expression_performer(
        REPO_ROOT / "model/r7hnr92k/best_model.pt",
        REPO_ROOT / "model/r7hnr92k/config.json", num_genes=15165, device=str(device),
    )
    condition_dir = WORK / ("condition_results_pilot" if pilot else "condition_results")
    condition_dir.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        species = dataset.rsplit("_", 1)[-1]
        sample_ids = cohorts.loc[(cohorts.role == "evaluation") & (cohorts.species == species),
                                 "sample_id"].astype(str).tolist()
        matrix = np.load(WORK / f"{dataset}_log1p_tpm.npy", mmap_mode="r")
        genes = pd.read_parquet(WORK / f"{dataset}_genes.parquet")
        if pilot:
            sample_ids, matrix = sample_ids[:100], matrix[:100]
        eligible = set(np.flatnonzero(genes.observed.to_numpy(bool)).tolist())
        for panel in panels.itertuples(index=False):
            path = condition_dir / f"{dataset}__{panel.panel_id}.parquet"
            if path.is_file():
                print(f"reuse {path.name}", flush=True)
                continue
            visible = np.sort(panel_genes.loc[panel_genes.panel_id.eq(panel.panel_id),
                                                "model_index"].to_numpy(int))
            score_indices = np.asarray(sorted(eligible - set(visible)), dtype=int)
            masked = mask_except_panel(matrix, visible, float(CONFIG["mask_token"]))
            prediction = reconstruct(model, masked, device, int(CONFIG["batch_size"]),
                                     f"{dataset} {panel.panel_id}")
            pearson, spearman, mse = score_masked_rows(matrix, prediction, score_indices)
            pd.DataFrame({
                "dataset": dataset, "species": species, "sample_id": sample_ids,
                "panel_type": panel.panel_type, "panel_id": panel.panel_id,
                "visible_gene_count": len(visible), "masked_genes": len(score_indices),
                "replicate": panel.replicate, "seed": panel.seed,
                "pearson": pearson, "spearman": spearman, "mse": mse,
            }).to_parquet(path, index=False)
            print(f"completed {dataset} {panel.panel_id}", flush=True)
    del model
    gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()
    aggregate(condition_dir, "pilot_" if pilot else "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="Use 100 samples and replicate 0")
    parser.add_argument("--panel-types", nargs="+", choices=["l1000", "random_curve", "random_l1000_matched"])
    args = parser.parse_args()
    run(args.pilot, args.panel_types)


if __name__ == "__main__":
    main()
