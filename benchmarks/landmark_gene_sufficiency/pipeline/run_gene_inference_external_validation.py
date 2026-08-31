#!/usr/bin/env python3
"""Validate frozen gene-inference panels on external TCGA and OSDR samples."""

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
from run_external_per_gene_evaluation import columnwise_metrics

sys.path.insert(0, str(REPO_ROOT / "src"))
from fm_embed.model import load_expression_performer
from fm_embed.reconstruction import mask_except_panel, reconstruct


DATASETS = ("tcga_human", "osdr_mouse")
PANEL_FILE = RESULTS / "frozen_gene_inference_validation_panels.parquet"
CACHE = WORK / "gene_inference_external_conditions"
PREFIX = "gene_inference_external_validation"


def worker(dataset: str, device_name: str) -> None:
    panels = pd.read_parquet(PANEL_FILE)
    manifest = panels[["panel_id", "panel_type", "selection_species", "replicate"]].drop_duplicates()
    matrix = np.load(WORK / f"external_{dataset}_log1p_tpm.npy", mmap_mode="r")
    genes = pd.read_parquet(WORK / f"external_{dataset}_genes.parquet")
    lookup = genes.set_index("model_index")
    eligible = set(genes.loc[genes.observed, "model_index"].astype(int))
    device = torch.device(device_name if device_name.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model, device = load_expression_performer(REPO_ROOT / "model/r7hnr92k/best_model.pt",
        REPO_ROOT / "model/r7hnr92k/config.json", 15165, str(device))
    cache = CACHE / dataset; cache.mkdir(parents=True, exist_ok=True)
    for panel in manifest.itertuples(index=False):
        output = cache / f"{panel.panel_id}.parquet"
        if output.is_file():
            print(f"[{dataset}] reuse {output.name}", flush=True); continue
        visible = np.sort(panels.loc[panels.panel_id.eq(panel.panel_id), "model_index"].to_numpy(int))
        effective_visible = eligible & set(visible)
        score_indices = np.asarray(sorted(eligible - set(visible)), dtype=int)
        prediction = reconstruct(model,
            mask_except_panel(matrix, visible, float(CONFIG["mask_token"])), device,
            int(CONFIG["batch_size"]), f"{dataset} {panel.panel_id} per-gene validation")
        result = columnwise_metrics(matrix, prediction, score_indices)
        result.insert(0, "gene", result.model_index.map(lookup.gene))
        result.insert(0, "replicate", panel.replicate)
        result.insert(0, "selection_species", panel.selection_species)
        result.insert(0, "panel_type", panel.panel_type)
        result.insert(0, "panel_id", panel.panel_id)
        result.insert(0, "dataset", dataset)
        result["samples"] = len(matrix); result["visible_genes"] = len(visible)
        result["effective_visible_genes"] = len(effective_visible)
        result["masked_observed_genes"] = len(score_indices)
        result.to_parquet(output, index=False)
        print(f"[{dataset}] completed {panel.panel_id}: {len(result):,} masked genes", flush=True)


def aggregate() -> None:
    files = sorted(CACHE.glob("*/*.parquet"))
    expected = len(DATASETS) * 54
    if len(files) != expected:
        raise RuntimeError(f"Expected {expected} cached conditions, found {len(files)}")
    per_gene = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    per_gene.to_parquet(RESULTS / f"{PREFIX}_per_gene.parquet", index=False)
    per_panel = per_gene.groupby(["dataset", "panel_id", "panel_type", "selection_species", "replicate"],
        as_index=False, dropna=False).agg(genes_scored=("gene", "size"), samples=("samples", "first"),
        visible_genes=("visible_genes", "first"),
        effective_visible_genes=("effective_visible_genes", "first"),
        pearson=("pearson", "mean"), spearman=("spearman", "mean"), mse=("mse", "mean"))
    per_panel.to_parquet(RESULTS / f"{PREFIX}_per_panel.parquet", index=False)
    summary = per_panel.groupby(["dataset", "panel_type", "selection_species"],
        as_index=False, dropna=False).agg(visible_genes=("visible_genes", "first"),
        effective_visible_genes=("effective_visible_genes", "first"),
        pearson_mean=("pearson", "mean"), pearson_sd=("pearson", "std"),
        spearman_mean=("spearman", "mean"), spearman_sd=("spearman", "std"),
        mse_mean=("mse", "mean"), mse_sd=("mse", "std"), replicates=("panel_id", "size"))
    summary.to_parquet(RESULTS / f"{PREFIX}_summary.parquet", index=False)
    summary.to_csv(RESULTS / f"{PREFIX}_summary.csv", index=False)
    comparisons = []
    for dataset, frame in per_panel.groupby("dataset"):
        random = frame.loc[frame.panel_type.eq("random_1000")]
        for fixed in frame.loc[~frame.panel_type.eq("random_1000")].itertuples(index=False):
            for metric, higher in (("pearson", True), ("spearman", True), ("mse", False)):
                values = random[metric].to_numpy(float); observed = float(getattr(fixed, metric))
                mean, sd = float(values.mean()), float(values.std(ddof=1))
                advantage = observed - mean if higher else mean - observed
                extreme = values >= observed if higher else values <= observed
                comparisons.append({"dataset": dataset, "panel_id": fixed.panel_id,
                    "panel_type": fixed.panel_type, "selection_species": fixed.selection_species,
                    "visible_genes": fixed.visible_genes,
                    "effective_visible_genes": fixed.effective_visible_genes,
                    "metric": metric, "fixed_value": observed, "random_1000_mean": mean,
                    "random_1000_sd": sd, "directional_advantage": advantage,
                    "standardized_advantage": advantage / sd if sd else np.nan,
                    "empirical_one_sided_p": (1 + int(extreme.sum())) / (len(values) + 1),
                    "random_replicates": len(values), "higher_is_better": higher,
                    "size_matched_to_random": fixed.visible_genes == 1000})
    comparison = pd.DataFrame(comparisons)
    comparison.to_parquet(RESULTS / f"{PREFIX}_comparison.parquet", index=False)
    comparison.to_csv(RESULTS / f"{PREFIX}_comparison.csv", index=False)
    print("\nSummary\n" + summary.to_string(index=False), flush=True)
    print("\nFixed-panel comparisons\n" + comparison.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--worker-dataset", choices=DATASETS)
    parser.add_argument("--worker-device")
    args = parser.parse_args()
    if args.worker_dataset:
        worker(args.worker_dataset, args.worker_device or "cuda:0"); return
    processes = []
    for index, dataset in enumerate(args.datasets):
        command = [sys.executable, str(Path(__file__).resolve()), "--worker-dataset", dataset,
                   "--worker-device", args.devices[index % len(args.devices)]]
        processes.append((dataset, subprocess.Popen(command)))
    started = time.monotonic()
    while processes:
        time.sleep(10); active = []
        for dataset, process in processes:
            code = process.poll()
            if code is None: active.append((dataset, process))
            elif code: raise subprocess.CalledProcessError(code, process.args)
            else: print(f"[orchestrator] {dataset} complete", flush=True)
        processes = active
        if processes and int(time.monotonic() - started) % 60 < 10:
            print(f"[orchestrator heartbeat] elapsed={(time.monotonic()-started)/60:.1f}m "
                  f"running={','.join(x[0] for x in processes)}", flush=True)
    aggregate()


if __name__ == "__main__":
    main()
