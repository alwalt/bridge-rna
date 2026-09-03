#!/usr/bin/env python3
"""Cache frozen r7hnr92k contextual gene tokens as a resumable float16 memmap."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO_ROOT, RESULTS, WORK  # noqa: E402
from run_attention_pooling import EXPRESSION, MODEL_CONFIG, load_frozen_encoder, verify_input_contract  # noqa: E402

CACHE_DIR = REPO_ROOT / "embeddings/tcga/ours_r7hnr92k_contextual"
TOKENS = CACHE_DIR / "contextual_tokens.float16.npy"
COMPLETED = CACHE_DIR / "completed_samples.npy"


def digest_strings(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode()); digest.update(b"\0")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--heartbeat-seconds", type=int, default=60)
    args = parser.parse_args()
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    verify_input_contract(); CACHE_DIR.mkdir(parents=True, exist_ok=True)
    expression = np.load(EXPRESSION, mmap_mode="r"); samples, genes = expression.shape
    if TOKENS.exists():
        tokens = np.load(TOKENS, mmap_mode="r+")
        if tokens.shape != (samples, genes, 512) or tokens.dtype != np.float16:
            raise RuntimeError(f"Unexpected cache: shape={tokens.shape}, dtype={tokens.dtype}")
    else:
        tokens = np.lib.format.open_memmap(TOKENS, mode="w+", dtype=np.float16,
            shape=(samples, genes, 512))
    done = np.load(COMPLETED) if COMPLETED.exists() else np.zeros(samples, dtype=bool)
    encoder = load_frozen_encoder(device); started = last = time.monotonic()
    pending = np.flatnonzero(~done)
    print(f"[cache] pending={len(pending):,}/{samples:,} device={device} output={TOKENS}", flush=True)
    for offset in range(0, len(pending), args.batch_size):
        rows = pending[offset:offset + args.batch_size]
        values = torch.as_tensor(np.asarray(expression[rows]), dtype=torch.float32, device=device)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16,
                                            enabled=device.type == "cuda"):
            hidden = encoder._encode_hidden(values)
        tokens[rows] = hidden.to(torch.float16).cpu().numpy(); done[rows] = True
        now = time.monotonic()
        if now - last >= args.heartbeat_seconds or offset + len(rows) == len(pending):
            tokens.flush(); np.save(COMPLETED, done)
            completed_now = min(offset + len(rows), len(pending)); elapsed = now - started
            rate = completed_now / max(elapsed, 1e-9)
            eta = (len(pending) - completed_now) / max(rate, 1e-9)
            print(f"[cache heartbeat] new={completed_now:,}/{len(pending):,} total={done.sum():,}/{samples:,} "
                  f"rate={rate:.2f}/s elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m", flush=True)
            last = now
    tokens.flush(); np.save(COMPLETED, done)
    cohort = pd.read_parquet(RESULTS / "cohort_manifest.parquet")
    cohort.to_parquet(CACHE_DIR / "sample_manifest.parquet", index=False)
    genes_table = pd.read_parquet(WORK / "ours_genes.parquet").reset_index(names="token_index")
    genes_table.to_parquet(CACHE_DIR / "gene_manifest.parquet", index=False)
    config = json.loads(MODEL_CONFIG.read_text())
    provenance = {"shape": [samples, genes, 512], "dtype": "float16",
        "checkpoint_run_id": config["run_id"], "normalization": config["normalization"],
        "gene_order_sha256": digest_strings(genes_table.gene),
        "sample_order_sha256": digest_strings(cohort.sample_id), "complete": bool(done.all())}
    (CACHE_DIR / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"[cache] complete={done.all()} size={TOKENS.stat().st_size/1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
