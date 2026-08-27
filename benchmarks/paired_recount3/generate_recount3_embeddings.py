#!/usr/bin/env python3
"""Encode paired recount3 log1p(TPM) with the frozen foundation model."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fm_embed.encode import encode_matrix  # noqa: E402
from fm_embed.model import load_expression_performer  # noqa: E402

HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expression", type=Path,
        default=HERE / "outputs/paired_expression/recount3_log1p_tpm.npy",
    )
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "model/r7hnr92k/best_model.pt")
    parser.add_argument("--model-config", type=Path, default=REPO_ROOT / "model/r7hnr92k/config.json")
    parser.add_argument(
        "--output", type=Path,
        default=HERE / "outputs/recount3_embeddings.npy",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expression = np.load(args.expression, mmap_mode="r")
    if expression.ndim != 2:
        raise ValueError("Expression matrix must be samples x genes")
    model, device = load_expression_performer(
        args.checkpoint, args.model_config, expression.shape[1], args.device
    )
    embeddings = encode_matrix(
        model, device, np.asarray(expression), batch_size=args.batch_size,
        normalize=False, label="recount3 paired encoding",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, embeddings.astype(np.float32))
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expression": str(args.expression.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "model_config": str(args.model_config.resolve()),
        "samples": embeddings.shape[0], "embedding_dim": embeddings.shape[1],
        "l2_normalized": False, "device": str(device),
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved {embeddings.shape} embeddings to {args.output}")


if __name__ == "__main__":
    main()
