#!/usr/bin/env python3
"""Prepare exact Bridge inputs and frozen representations for the OSDR atlas cohort."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import CONFIG, RESULTS, WORK, apply_probe, save_json, sha256

REPO = Path("/home/walt/bridge-rna")
OUT = RESULTS / "osdr_atlas"
ATLAS_WORK = WORK / "osdr_atlas"
OLD = REPO / "benchmarks/osdr_batch_effect_representation"


def log(message: str) -> None:
    stamp = pd.Timestamp.now(tz="UTC").isoformat()
    print(f"[{stamp}] {message}", flush=True)
    with (OUT / "representation_run.log").open("a") as stream:
        stream.write(f"[{stamp}] {message}\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); ATLAS_WORK.mkdir(parents=True, exist_ok=True)
    (OUT / "representation_run.log").write_text("")
    started = time.time()
    manifest = pd.read_parquet(OUT / "eligible_sample_manifest.parquet").sort_values("matrix_row").reset_index(drop=True)
    genes = pd.read_csv(RESULTS / "frozen/frozen_canonical_genes.csv")["gene"].astype(str).tolist()
    assert len(genes) == 15_165 and manifest.matrix_row.tolist() == list(range(len(manifest)))
    expression = np.lib.format.open_memmap(ATLAS_WORK / "log1p_tpm.float32.npy", mode="w+", dtype=np.float32,
                                           shape=(len(manifest), len(genes)))
    sys.path.insert(0, str(REPO / "src"))
    from fm_embed.sources.osdr import load_osdr_matrix
    for file_i, (counts_path, group) in enumerate(manifest.groupby("counts_path", sort=True), 1):
        frame, already_tpm = load_osdr_matrix(REPO / counts_path, group.counts_column.astype(str).tolist())
        assert already_tpm
        frame = frame.reindex(index=group.counts_column.astype(str), columns=genes, fill_value=0.0)
        expression[group.matrix_row.to_numpy(int)] = np.log1p(frame.to_numpy(np.float32))
        if file_i % 10 == 0 or file_i == manifest.counts_path.nunique():
            log(f"preprocessing {file_i}/{manifest.counts_path.nunique()} count matrices; samples complete={group.matrix_row.max()+1}/{len(manifest)}")
    expression.flush()
    assert np.isfinite(expression).all() and expression.min() >= 0

    # Compatibility witness against all exact study+sample matches in the prior cache.
    old_m = pd.read_csv(OLD / "results/sample_manifest.csv")
    old_x = np.load(OLD / "work/bridgerna_log1p_tpm_inputs.npy", mmap_mode="r")
    old_z = np.load(OLD / "work/bridgerna_embeddings.npy", mmap_mode="r")
    old_lookup = {(str(r.OSD).upper(), str(r.sample_id)): i for i, r in old_m.iterrows()}
    pairs = [(new_i, old_lookup[(r.study, str(r.sample_id))]) for new_i, r in manifest.iterrows()
             if (r.study, str(r.sample_id)) in old_lookup]
    assert len(pairs) == 112
    max_input_diff = max(float(np.max(np.abs(expression[i] - old_x[j]))) for i, j in pairs)
    if max_input_diff > 1e-6:
        raise AssertionError(f"cached input incompatibility: max_abs_difference={max_input_diff}")
    log(f"compatibility witness passed for {len(pairs)} cached rows; max input difference={max_input_diff}")

    from fm_embed.model import load_expression_performer
    model, device = load_expression_performer(CONFIG["checkpoint"], RESULTS / "frozen/bridge_model_config.json", 15165, "cuda:0")
    embeddings = np.lib.format.open_memmap(ATLAS_WORK / "bridge_embeddings.float32.npy", mode="w+", dtype=np.float32,
                                            shape=(len(manifest), 512))
    reused = {i: j for i, j in pairs}
    for i, j in pairs:
        embeddings[i] = old_z[j]
    missing = [i for i in range(len(manifest)) if i not in reused]
    log(f"inference start: reuse={len(reused)}, infer={len(missing)}, batch=8, device={device}; estimated 10-30 minutes")
    batch = 8
    infer_started = time.time()
    for left in range(0, len(missing), batch):
        rows = missing[left:left + batch]
        xb = torch.as_tensor(np.array(expression[rows], copy=True), device=device)
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            embeddings[rows] = model.encode(xb, normalize=False).float().cpu().numpy()
        done = min(left + batch, len(missing))
        if done % 80 == 0 or done == len(missing):
            elapsed = time.time() - infer_started; rate = done / elapsed
            log(f"inference {done}/{len(missing)}; elapsed={elapsed:.1f}s; rate={rate:.2f}/s; eta={(len(missing)-done)/rate:.1f}s")
    embeddings.flush()
    probe = np.load(RESULTS / "frozen/frozen_gtex_hallmark_probe.npz", allow_pickle=False)
    hallmark = apply_probe(np.asarray(embeddings), probe).astype(np.float32)
    np.save(ATLAS_WORK / "hallmark_profiles.float32.npy", hallmark)
    assert np.isfinite(embeddings).all() and np.isfinite(hallmark).all()
    save_json(OUT / "representation_provenance.json", {
        "samples": len(manifest), "expression_shape": list(expression.shape), "bridge_shape": list(embeddings.shape),
        "hallmark_shape": list(hallmark.shape), "cached_rows_reused": len(reused), "new_rows_inferred": len(missing),
        "cache_input_compatibility_max_abs_difference": max_input_diff,
        "preprocessing": "mouse raw counts -> mouse exon-length TPM -> one-to-one human orthologs -> canonical 15,165 genes -> natural log1p",
        "checkpoint": CONFIG["checkpoint"], "checkpoint_sha256": sha256(CONFIG["checkpoint"]),
        "probe_sha256": sha256(RESULTS / "frozen/frozen_gtex_hallmark_probe.npz"),
        "bridge_frozen": True, "hallmark_head_retrained": False, "runtime_seconds": time.time() - started})
    log(f"COMPLETE runtime={time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
