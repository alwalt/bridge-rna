#!/usr/bin/env python3
"""Validate frozen informative panels on external TCGA and OSDR datasets."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import CONFIG, REPO_ROOT, RESULTS, WORK

sys.path.insert(0, str(REPO_ROOT / "src"))
from fm_embed.model import load_expression_performer
from fm_embed.reconstruction import mask_except_panel, reconstruct, score_masked_rows

DATASETS = {"tcga_human": "human", "osdr_mouse": "mouse"}


def aggregate(cache_dir: str = "external_panel_conditions",
              output_prefix: str = "external_panel_validation") -> None:
    files = sorted((WORK / cache_dir).glob("*/*.parquet"))
    if not files:
        raise RuntimeError(f"No cached panel conditions under {WORK / cache_dir}")
    per_sample = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    per_sample.to_parquet(RESULTS / f"{output_prefix}_per_sample.parquet", index=False)
    per_panel = per_sample.groupby(["dataset", "panel_id", "panel_type", "selection_species", "replicate"],
        as_index=False, dropna=False).agg(pearson=("pearson", "mean"), spearman=("spearman", "mean"),
        mse=("mse", "mean"), samples=("sample_id", "size"),
        effective_visible_genes=("effective_visible_genes", "first"),
        observed_genes=("observed_genes", "first"), masked_genes=("masked_genes", "first"))
    per_panel.to_parquet(RESULTS / f"{output_prefix}_per_panel.parquet", index=False)
    summary = per_panel.groupby(["dataset", "panel_type", "selection_species"], as_index=False).agg(
        pearson_mean=("pearson", "mean"), pearson_sd=("pearson", "std"),
        spearman_mean=("spearman", "mean"), spearman_sd=("spearman", "std"),
        mse_mean=("mse", "mean"), mse_sd=("mse", "std"), replicates=("panel_id", "size"))
    summary.to_parquet(RESULTS / f"{output_prefix}_summary.parquet", index=False)
    summary.to_csv(RESULTS / f"{output_prefix}_summary.csv", index=False)
    if set(per_panel.panel_type) == {"shared_451", "random_451"}:
        comparisons = []
        for dataset, frame in per_panel.groupby("dataset"):
            fixed = frame.loc[frame.panel_type.eq("shared_451")].iloc[0]
            random = frame.loc[frame.panel_type.eq("random_451")]
            for metric, higher_is_better in (("pearson", True), ("spearman", True), ("mse", False)):
                values = random[metric].to_numpy(float)
                observed = float(fixed[metric])
                mean, sd = float(values.mean()), float(values.std(ddof=1))
                advantage = observed - mean if higher_is_better else mean - observed
                extreme = values >= observed if higher_is_better else values <= observed
                percentile = np.mean(values <= observed) if higher_is_better else np.mean(values >= observed)
                comparisons.append({"dataset": dataset, "metric": metric,
                    "shared_451": observed, "random_451_mean": mean, "random_451_sd": sd,
                    "directional_advantage": advantage,
                    "standardized_advantage": advantage / sd if sd > 0 else np.nan,
                    "shared_451_percentile": percentile,
                    "empirical_one_sided_p": (1 + int(extreme.sum())) / (len(values) + 1),
                    "random_replicates": len(values), "higher_is_better": higher_is_better})
        comparison = pd.DataFrame(comparisons)
        comparison.to_parquet(RESULTS / f"{output_prefix}_comparison.parquet", index=False)
        comparison.to_csv(RESULTS / f"{output_prefix}_comparison.csv", index=False)
        print("\nFixed-panel comparison\n" + comparison.to_string(index=False))
    print(summary.to_string(index=False))


def worker(dataset: str, device_name: str, panel_file: str = "frozen_validation_panels.parquet",
           cache_dir: str = "external_panel_conditions") -> None:
    panels = pd.read_parquet(RESULTS / panel_file)
    panel_manifest = panels[["panel_id", "panel_type", "selection_species", "replicate"]].drop_duplicates()
    matrix = np.load(WORK / f"external_{dataset}_log1p_tpm.npy", mmap_mode="r")
    genes = pd.read_parquet(WORK / f"external_{dataset}_genes.parquet")
    eligible = set(genes.loc[genes.observed, "model_index"].astype(int))
    samples = pd.read_parquet(RESULTS / f"external_{dataset}_samples.parquet").sort_values("row_index")
    if len(matrix) != len(samples): raise ValueError(f"{dataset}: sample order mismatch")
    device = torch.device(device_name if device_name.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model, device = load_expression_performer(REPO_ROOT / "model/r7hnr92k/best_model.pt",
        REPO_ROOT / "model/r7hnr92k/config.json", 15165, str(device))
    cache = WORK / cache_dir / dataset; cache.mkdir(parents=True, exist_ok=True)
    for panel in panel_manifest.itertuples(index=False):
        path = cache / f"{panel.panel_id}.parquet"
        if path.exists(): print(f"[{dataset}] reuse {path.name}", flush=True); continue
        visible = np.sort(panels.loc[panels.panel_id.eq(panel.panel_id), "model_index"].to_numpy(int))
        effective_visible = eligible & set(visible)
        score_indices = np.asarray(sorted(eligible - set(visible)), dtype=int)
        prediction = reconstruct(model, mask_except_panel(matrix, visible, float(CONFIG["mask_token"])),
            device, int(CONFIG["batch_size"]), f"{dataset} {panel.panel_id}")
        pearson, spearman, mse = score_masked_rows(matrix, prediction, score_indices)
        pd.DataFrame({"dataset": dataset, "sample_id": samples.sample_id.astype(str),
            "panel_id": panel.panel_id, "panel_type": panel.panel_type,
            "selection_species": panel.selection_species, "replicate": panel.replicate,
            "visible_genes": len(visible), "effective_visible_genes": len(effective_visible),
            "observed_genes": len(eligible), "masked_genes": len(score_indices),
            "pearson": pearson, "spearman": spearman, "mse": mse}).to_parquet(path, index=False)
        print(f"[{dataset}] completed {panel.panel_id}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--heartbeat-seconds", type=int, default=60)
    parser.add_argument("--panel-file", default="frozen_validation_panels.parquet")
    parser.add_argument("--cache-dir", default="external_panel_conditions")
    parser.add_argument("--output-prefix", default="external_panel_validation")
    parser.add_argument("--worker-dataset", choices=list(DATASETS)); parser.add_argument("--worker-device")
    args = parser.parse_args()
    if args.worker_dataset:
        worker(args.worker_dataset, args.worker_device or "cuda:0", args.panel_file, args.cache_dir)
        return
    processes = []
    for index, dataset in enumerate(args.datasets):
        device = args.devices[index % len(args.devices)]
        command = [sys.executable, str(Path(__file__).resolve()), "--worker-dataset", dataset,
                   "--worker-device", device, "--panel-file", args.panel_file,
                   "--cache-dir", args.cache_dir]
        processes.append((dataset, device, subprocess.Popen(command)))
    started = time.monotonic(); last = started
    while processes:
        time.sleep(10); remaining = []
        for dataset, device, process in processes:
            code = process.poll()
            if code is None: remaining.append((dataset, device, process))
            elif code: raise subprocess.CalledProcessError(code, process.args)
            else: print(f"[orchestrator] {dataset} finished on {device}", flush=True)
        processes = remaining
        if processes and time.monotonic() - last >= args.heartbeat_seconds:
            print(f"[orchestrator heartbeat] elapsed={(time.monotonic()-started)/60:.1f}m "
                  f"running={','.join(x[0] for x in processes)}", flush=True); last = time.monotonic()
    aggregate(args.cache_dir, args.output_prefix)


if __name__ == "__main__": main()
