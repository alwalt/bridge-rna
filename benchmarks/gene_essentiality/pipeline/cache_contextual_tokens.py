#!/usr/bin/env python3
"""Cache frozen contextual target-gene tokens in shared external storage."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
CONFIG = json.loads((BENCH / "config.json").read_text())
ASSET_ROOT = Path(CONFIG["bridge_root"])
WORK = Path(CONFIG["shared_work_root"])
RESULTS = BENCH / "results"


def say(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def python_rc_compatibility() -> None:
    if not hasattr(sys, "get_int_max_str_digits"):
        def get_int_max_str_digits() -> int:
            return 4300
        sys.get_int_max_str_digits = get_int_max_str_digits  # type: ignore[attr-defined]
    if not hasattr(sys, "set_int_max_str_digits"):
        def set_int_max_str_digits(maxdigits: int) -> None:
            del maxdigits
        sys.set_int_max_str_digits = set_int_max_str_digits  # type: ignore[attr-defined]


def build_bridge_input(expression: np.ndarray) -> np.ndarray:
    mapping = pd.read_csv(RESULTS / "expression_gene_overlap.csv")
    output = np.full((len(expression), 15165), -10.0, dtype=np.float32)
    mapped = mapping[mapping.bridge_representable]
    output[:, mapped.bridge_token_index.astype(int)] = expression[:, mapped.expression_column_index.astype(int)]
    return output


def build_bulkformer_input(expression: np.ndarray) -> tuple[np.ndarray, float]:
    vocab = pd.read_csv(Path(CONFIG["bulkformer_root"]) / "data/bulkformer_gene_info.csv").ensg_id.astype(str)
    expr_genes = pd.read_csv(RESULTS / "expression_gene_overlap.csv")
    gene_to_column = dict(zip(expr_genes.gene_id, expr_genes.expression_column_index))
    output = np.full((len(expression), len(vocab)), -10.0, dtype=np.float32)
    present_tokens, source_columns = [], []
    for token, gene in enumerate(vocab):
        if gene in gene_to_column:
            present_tokens.append(token)
            source_columns.append(gene_to_column[gene])
    output[:, present_tokens] = expression[:, source_columns]
    return output, 1.0 - len(present_tokens) / len(vocab)


def load_bridge(device: torch.device):
    sys.path.insert(0, str(ASSET_ROOT / "src"))
    from fm_embed.model import load_expression_performer
    model, _ = load_expression_performer(
        Path(CONFIG["bridge_checkpoint"]), Path(CONFIG["bridge_config"]),
        num_genes=15165, device=str(device),
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.eval()


def load_bulkformer(device: torch.device):
    python_rc_compatibility()
    sys.path.insert(0, str(ASSET_ROOT / "benchmarks/tcga_imputation/pipeline"))
    sys.path.insert(0, str(ASSET_ROOT / "model/BulkFormer"))
    import model_adapters
    # Supply the module globals expected by the repository-local adapter.
    model_adapters.REPO_ROOT = ASSET_ROOT
    model = model_adapters.load_bulkformer("bulkformer_147m", device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.eval()


@torch.inference_mode()
def encode_bulkformer(model, values: torch.Tensor) -> torch.Tensor:
    """Return the final normalized 640-D contextual trunk state.

    The public model forward concatenates three sample summary covariates to
    this tensor. They are excluded so the evaluated object is the contextual
    gene state itself, parallel to Bridge's final contextual layer.
    """
    gene_identity = model.gene_emb_proj(model.gene_emb_onehot_layer.weight)
    hidden = model.expr_emb(values) + gene_identity + model.global_expr_proj(values).unsqueeze(1)
    hidden = model.x_proj(hidden)
    for layer in model.gb_formers:
        hidden = layer(hidden, model.graph)
    return model.layernorm(hidden)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("representation", choices=["bridge", "bulkformer"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Dry-run sample limit; does not write the canonical cache.")
    parser.add_argument("--heartbeat-seconds", type=int, default=60)
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    expression = np.load(WORK / "expression_fm_natural_log1p_tpm.float32.npy", mmap_mode="r")
    targets = pd.read_csv(RESULTS / "gene_overlap.csv")
    if args.representation == "bridge":
        selected = targets[targets.bridge_representable].copy()
        token_indices = selected.bridge_token_index.astype(int).to_numpy()
        model_input = build_bridge_input(expression)
        model = load_bridge(device)
        dimension = 512
        missing_rate = float(np.mean(model_input[0] == -10))
    else:
        selected = targets[targets.bulkformer_representable].copy()
        token_indices = selected.bulkformer_token_index.astype(int).to_numpy()
        model_input, missing_rate = build_bulkformer_input(expression)
        model = load_bulkformer(device)
        dimension = 640
    if args.limit is not None:
        total = min(args.limit, len(model_input))
        write_cache = False
    else:
        total = len(model_input)
        write_cache = True
    say(f"startup representation={args.representation} samples={total} tokens={len(token_indices)} "
        f"dimension={dimension} device={device} batch={args.batch_size} missing_rate={missing_rate:.6f}")

    cache_dir = WORK / f"{args.representation}_contextual"
    if write_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        token_path = cache_dir / "tokens.float16.npy"
        done_path = cache_dir / "completed.npy"
        if token_path.exists():
            output = np.load(token_path, mmap_mode="r+")
            if output.shape != (total, len(token_indices), dimension):
                raise RuntimeError(f"Unexpected existing cache shape {output.shape}")
        else:
            output = np.lib.format.open_memmap(
                token_path, mode="w+", dtype=np.float16,
                shape=(total, len(token_indices), dimension),
            )
        done = np.load(done_path) if done_path.exists() else np.zeros(total, dtype=bool)
        pending = np.flatnonzero(~done)
    else:
        output = None
        done = np.zeros(total, dtype=bool)
        pending = np.arange(total)
    started = last = time.monotonic()
    for offset in range(0, len(pending), args.batch_size):
        rows = pending[offset:offset + args.batch_size]
        values = torch.as_tensor(model_input[rows], dtype=torch.float32, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            hidden = model._encode_hidden(values) if args.representation == "bridge" else encode_bulkformer(model, values)
            chosen = hidden[:, token_indices]
        if output is not None:
            output[rows] = chosen.to(torch.float16).cpu().numpy()
        done[rows] = True
        del values, hidden, chosen
        now = time.monotonic()
        completed = min(offset + len(rows), len(pending))
        if now - last >= args.heartbeat_seconds or completed == len(pending):
            elapsed = now - started
            rate = completed / max(elapsed, 1e-9)
            eta = (len(pending) - completed) / max(rate, 1e-9)
            if output is not None:
                output.flush(); np.save(done_path, done)
            say(f"heartbeat new={completed}/{len(pending)} total={done.sum()}/{total} "
                f"rate={rate:.3f}/s elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m")
            last = now
    if write_cache:
        output.flush(); np.save(done_path, done)
        selected.to_csv(cache_dir / "target_manifest.csv", index=False)
        provenance = {
            "representation": args.representation,
            "shape": [total, len(token_indices), dimension], "dtype": "float16",
            "complete": bool(done.all()), "input": "natural log1p TPM",
            "released_to_inference_transform": CONFIG["fm_inference_transform"],
            "missing_gene_rate": missing_rate, "encoder_frozen": True,
            "layer": "final encoder layer after LayerNorm",
            "bulkformer_auxiliary_summary_features_excluded": args.representation == "bulkformer",
        }
        (cache_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        say(f"complete cache={token_path} size={token_path.stat().st_size/1e9:.2f}GB")
    del model
    gc.collect()


if __name__ == "__main__":
    main()
