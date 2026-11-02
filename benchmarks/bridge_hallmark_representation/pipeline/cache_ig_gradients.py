#!/usr/bin/env python3
"""Cache nested IG path gradients for assigned samples on one device."""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd
import torch

from common import CONFIG, REPO, RESULTS, WORK, read_vocab, resolve
from validate_integrated_gradients import gradients_on_nested_path, select_samples

sys.path.insert(0, str(REPO / "src"))
from fm_embed.model import load_expression_performer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--sample-indices", type=int, nargs="+", required=True)
    args = parser.parse_args()
    settings = CONFIG["ig_validation"]
    max_steps = settings["reference_steps"]
    meta = pd.read_parquet(RESULTS / "tcga_sample_manifest.parquet")
    samples = select_samples(meta)
    expression = np.load(resolve(CONFIG["tcga_bridge_expression"]), mmap_mode="r")
    probe = np.load(WORK / "frozen_gtex_hallmark_probe.npz")
    transfer = pd.read_csv(RESULTS / "gtex_tcga_probe_transfer_summary.csv")
    selected_pathways = transfer.nlargest(4, "pearson").hallmark.tolist()
    hallmarks = probe["hallmarks"].astype(str).tolist()
    indices = np.asarray([hallmarks.index(name) for name in selected_pathways])
    genes = read_vocab()
    model, device = load_expression_performer(
        resolve(CONFIG["checkpoint"]), resolve(CONFIG["model_config"]), 15165, args.device
    )
    model.requires_grad_(False)
    started = time.time()
    for count, sample_index in enumerate(args.sample_indices, 1):
        selected = samples.iloc[sample_index]
        sample = meta.iloc[int(selected.row)]
        cache = WORK / f"ig_nested_gradients_{sample.sample_id}_{max_steps}.npy"
        if cache.exists():
            gradients = np.load(cache, mmap_mode="r")
            if gradients.shape == (max_steps + 1, len(selected_pathways), len(genes)):
                print(f"[cache {args.device}] reuse {sample.sample_id}", flush=True)
                continue
        observed = np.array(expression[int(selected.row)], dtype=np.float32, copy=True)
        _, _, gradients = gradients_on_nested_path(
            model, observed, probe["weight"][indices], probe["bias"][indices],
            probe["mean"], probe["scale"], device, max_steps, settings["path_batch"],
        )
        np.save(cache, gradients.astype(np.float32))
        elapsed = time.time() - started
        print(f"[cache {args.device}] {count}/{len(args.sample_indices)} {sample.sample_id} elapsed={elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
