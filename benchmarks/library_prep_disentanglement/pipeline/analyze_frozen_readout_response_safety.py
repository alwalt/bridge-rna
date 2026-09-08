#!/usr/bin/env python3
"""OSDR safety check for the selected frozen BridgeRNA mean+SD readout."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
T3 = REPO / "benchmarks/osdr_batch_effect_representation"
OUT = HERE / "results/task4_frozen_readout_selection"
WORK = HERE / "work/task4_frozen_readout_selection"
G = 15165

SPECS = {
    "RR1_original": ("C14__OSD-48__RR1-NASA__37-day", None),
    "RR1_remeasurement": ("C04__OSD-168__RR1-NASA__37-day", None),
    "RR3_39_original": ("C01__OSD-137__RR3__39-day", None),
    "RR3_39_remeasurement": ("C05__OSD-168__RR3__39-day", None),
    "RR3_40_original": ("C02__OSD-137__RR3__40-day", "strict"),
    "RR3_40_remeasurement": ("C06__OSD-168__RR3__40-day", None),
}


def say(message: str) -> None:
    print(f"[{time.strftime('%F %T')}] {message}", flush=True)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator else np.nan


def extract(device: str = "cuda:0", batch_size: int = 1) -> Path:
    output = WORK / "task3_layer12_mean_std.float32.npy"
    if output.exists():
        say(f"reusing {output}")
        return output
    WORK.mkdir(parents=True, exist_ok=True)
    x = np.load(T3 / "work/bridgerna_log1p_tpm_inputs.npy", mmap_mode="r")
    manifest = pd.read_csv(T3 / "results/sample_manifest.csv")
    if len(x) != len(manifest) or x.shape[1] != G:
        raise ValueError(f"Unexpected Task 3 input shape {x.shape} for {len(manifest)} samples")
    sys.path.insert(0, str(REPO / "benchmarks/tcga_downstream/pipeline"))
    from run_attention_pooling import load_frozen_encoder

    dev = torch.device(device)
    model = load_frozen_encoder(dev)
    result = np.lib.format.open_memmap(output, mode="w+", dtype="float32", shape=(len(x), 1024))
    ids = torch.arange(G, device=dev)
    started = time.time()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            stop = min(start + batch_size, len(x))
            values = torch.as_tensor(np.asarray(x[start:stop]), device=dev)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                hidden = model.gene_embedding(ids).unsqueeze(0) + model.ree(values)
                for layer in model.layers:
                    hidden = layer(hidden)
                pooled = torch.cat(
                    [hidden.float().mean(1), hidden.float().std(1, unbiased=False)], dim=1
                )
            result[start:stop] = pooled.cpu().numpy()
            if stop == len(x) or stop % 20 == 0:
                say(f"inference {stop}/{len(x)} elapsed={(time.time()-started)/60:.1f}m")
    result.flush()
    return output


def response_vectors(embeddings: np.ndarray) -> dict[str, np.ndarray]:
    manifest = pd.read_csv(T3 / "results/sample_manifest.csv")
    membership = pd.read_csv(T3 / "results/task3b_contrast_sample_membership.csv")
    index = dict(zip(manifest.sample_id.astype(str), range(len(manifest))))
    vectors = {}
    for name, (contrast_id, strict) in SPECS.items():
        rows = membership[membership.contrast_id.eq(contrast_id)].copy()
        if name == "RR1_original":
            rows = rows[~rows.sample_id.str.endswith("_M27")]
        if name == "RR1_remeasurement":
            rows = rows[~rows.sample_id.str.endswith("_M29")]
        if strict:
            rows = rows[~rows.sample_id.str.endswith("_F5")]
        flt = rows.loc[rows.condition.eq("FLT"), "sample_id"].map(index).to_numpy(int)
        gc = rows.loc[rows.condition.eq("GC"), "sample_id"].map(index).to_numpy(int)
        if not len(flt) or not len(gc):
            raise ValueError(f"Empty condition in {name}: FLT={len(flt)}, GC={len(gc)}")
        vectors[name] = np.asarray(embeddings[flt]).mean(0) - np.asarray(embeddings[gc]).mean(0)
    return vectors


def evaluate() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    original = np.load(T3 / "work/bridgerna_embeddings.npy", mmap_mode="r")
    selected = np.load(WORK / "task3_layer12_mean_std.float32.npy", mmap_mode="r")
    rows = []
    expected = {"RR1": -0.804, "RR3-39": 0.790, "RR3-40": 0.917}
    pairs = {
        "RR1": ("RR1_original", "RR1_remeasurement"),
        "RR3-39": ("RR3_39_original", "RR3_39_remeasurement"),
        "RR3-40": ("RR3_40_original", "RR3_40_remeasurement"),
    }
    for readout, matrix in [("current_mean", original), ("selected_mean_plus_std", selected)]:
        vectors = response_vectors(matrix)
        for comparison, (left, right) in pairs.items():
            a, b = vectors[left], vectors[right]
            rows.append(
                {
                    "readout": readout,
                    "comparison": comparison,
                    "cosine": cosine(a, b),
                    "spearman": float(spearmanr(a, b).statistic),
                    "norm_original": float(np.linalg.norm(a)),
                    "norm_remeasurement": float(np.linalg.norm(b)),
                }
            )
    results = pd.DataFrame(rows)
    baseline = results[results.readout.eq("current_mean")].set_index("comparison").cosine
    for name, target in expected.items():
        if abs(baseline[name] - target) > 0.015:
            raise AssertionError(f"{name} baseline {baseline[name]:.4f} does not reproduce ~{target:.3f}")
    wide = results.pivot(index="comparison", columns="readout", values="cosine").reset_index()
    wide["cosine_change"] = wide.selected_mean_plus_std - wide.current_mean
    results.to_csv(OUT / "osdr_response_safety.csv", index=False)
    wide.to_csv(OUT / "osdr_response_safety_comparison.csv", index=False)
    (OUT / "osdr_response_safety_provenance.json").write_text(
        json.dumps(
            {
                "backbone_frozen": True,
                "sample_memberships_changed": False,
                "contrast_definitions": "Exact strict mappings from analyze_multilayer_reproducibility.py",
                "selected_readout": "layer 12 token mean concatenated with population SD (1024D)",
                "baseline_validation_tolerance": 0.015,
            },
            indent=2,
        ) + "\n"
    )
    print(results.to_string(index=False))
    print("\nCosine comparison\n", wide.to_string(index=False))


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    extract(args.device, args.batch_size)
    evaluate()
    say("complete")


if __name__ == "__main__":
    main()
