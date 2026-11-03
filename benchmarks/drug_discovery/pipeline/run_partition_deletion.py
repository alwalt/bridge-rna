#!/usr/bin/env python3
"""Mask frozen complementarity partitions in the frozen Bridge model."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_bridge as rb

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "results/expanded_chembl_sensitivity"
PREP = HERE / "work/prepared"


def scores(model, values, direction, device, mask=None):
    result = []
    target = torch.as_tensor(direction, device=device).view(1, -1)
    with torch.no_grad():
        for start in range(0, len(values), 8):
            batch = torch.as_tensor(np.array(values[start:start+8], copy=True), device=device)
            if mask is not None:
                batch[:, mask] = -10.0
            result.extend(rb.score(model, batch, target.expand(len(batch), -1)).cpu().numpy())
    return np.asarray(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--datasets", default="")
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    selected = set(args.datasets.split(",")) if args.datasets else None
    model, device, model_genes = rb.load_model(args.device)
    lookup = {gene: i for i, gene in enumerate(model_genes)}
    members = pd.read_parquet(HERE / "results/manifests/contrast_members.parquet")
    vectors = pd.read_parquet(HERE / "results/bridge/condition_vectors.parquet")
    partitions = pd.read_csv(OUT / "gene_partitions.csv.gz")
    rows = []
    for dataset, dataset_members in members.groupby("dataset", sort=True):
        if selected is not None and dataset not in selected:
            continue
        matrix = np.load(PREP / f"{dataset}_log1p_tpm.npy", mmap_mode="r")
        mapping = pd.read_csv(PREP / f"{dataset}_matrix_rows.csv").set_index("sample_id").matrix_row.to_dict()
        dataset_parts = {p: g.gene.tolist() for p, g in partitions[partitions.dataset.eq(dataset)].groupby("partition")}
        panels = []
        for partition, genes in dataset_parts.items():
            indexes = np.asarray([lookup[g] for g in genes], dtype=int)
            panels.append((partition, "observed", 0, indexes))
        effects = {key: [] for key in [(p, t, r) for p, t, r, _ in panels]}
        original_effects = []
        for contrast_id, contrast_members in dataset_members.groupby("contrast_id", sort=True):
            direction = vectors[vectors.contrast_id.eq(contrast_id)].sort_values("dimension").value.to_numpy(np.float32)
            cm = contrast_members.reset_index(drop=True)
            values = np.stack([matrix[mapping[s]] for s in cm.sample_id])
            original_scores = scores(model, values, direction, device)
            original, _ = rb.contrast_contributions(contrast_id, original_scores[:, None], cm)
            original_effects.append(float(original.mean()))
            for partition, panel_type, replicate, indexes in panels:
                masked_scores = scores(model, values, direction, device, indexes)
                masked, _ = rb.contrast_contributions(contrast_id, masked_scores[:, None], cm)
                effects[(partition, panel_type, replicate)].append(float(masked.mean()))
        original = float(np.mean(original_effects))
        for partition, panel_type, replicate, indexes in panels:
            masked = float(np.mean(effects[(partition, panel_type, replicate)]))
            rows.append({"dataset": dataset, "partition": partition, "panel_type": panel_type,
                         "replicate": replicate, "genes_masked": len(indexes),
                         "original_axis_effect": original, "masked_axis_effect": masked,
                         "absolute_score_change": abs(masked-original),
                         "absolute_change_per_100_genes": abs(masked-original)/max(len(indexes), 1)*100,
                         "fraction_signal_remaining": masked/original if abs(original) > 1e-12 else np.nan})
        print(f"partition deletion {dataset}", flush=True)
    frame = pd.DataFrame(rows)
    suffix = f"_{args.suffix}" if args.suffix else ""
    frame.to_csv(OUT / f"partition_deletion{suffix}.csv", index=False)
    frame.to_csv(OUT / f"partition_deletion_summary{suffix}.csv", index=False)
    (OUT / f"partition_deletion_provenance{suffix}.json").write_text(json.dumps({
        "frozen_model": True, "mask_token": -10, "random_panels_per_partition": 0,
        "interpretation": "observed effect-size analysis normalized by partition size; no partition-level null test",
        "context": "the frozen Phase 2 whole-top-500 deletion control retains five expression-matched random panels per dataset",
        "recomputed": ["masked and original projection scores for this sensitivity only"],
        "not_recomputed": ["expression", "embeddings", "condition vectors", "attribution", "DE", "gene modules"]
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
