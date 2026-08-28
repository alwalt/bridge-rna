"""Frozen reconstruction adapters using each repository's existing model code."""

from __future__ import annotations

import gc
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from common import CONFIG, REPO_ROOT


def _python_rc_compatibility() -> None:
    # This environment is Python 3.11.0rc1; current torch._dynamo expects two
    # sys APIs introduced in the final 3.11 release.
    if not hasattr(sys, "get_int_max_str_digits"):
        def get_int_max_str_digits() -> int:
            return 4300
        sys.get_int_max_str_digits = get_int_max_str_digits  # type: ignore[attr-defined]
    if not hasattr(sys, "set_int_max_str_digits"):
        def set_int_max_str_digits(maxdigits: int) -> None:
            del maxdigits
        sys.set_int_max_str_digits = set_int_max_str_digits  # type: ignore[attr-defined]


def resolved_device(requested: str | None = None) -> torch.device:
    value = requested or str(CONFIG["device"])
    return torch.device(value if value.startswith("cuda") and torch.cuda.is_available() else "cpu")


def load_ours(device: torch.device):
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from fm_embed.model import load_expression_performer
    return load_expression_performer(
        REPO_ROOT / "model/r7hnr92k/best_model.pt",
        REPO_ROOT / "model/r7hnr92k/config.json",
        num_genes=15165, device=str(device),
    )[0]


def load_bulkformer(variant: str, device: torch.device):
    _python_rc_compatibility()
    bulk_root = REPO_ROOT / "model/BulkFormer"
    sys.path.insert(0, str(bulk_root))
    from utils.BulkFormer import BulkFormer

    settings = {
        "bulkformer_50m": (256, 2, bulk_root / "BulkFormer_50M.pt"),
        "bulkformer_147m": (640, 12, bulk_root / "model/BulkFormer_147M.pt"),
    }
    if variant not in settings:
        raise ValueError(f"Unknown BulkFormer variant: {variant}")
    dim, repeats, checkpoint = settings[variant]
    edge = torch.load(bulk_root / "data/G_tcga.pt", map_location="cpu", weights_only=False)
    weights = torch.load(
        bulk_root / "data/G_tcga_weight.pt", map_location="cpu", weights_only=False
    )
    # The official notebook constructs
    # SparseTensor(row=edge[1], col=edge[0], value=weights).t().  GCNConv treats
    # that sparse object as adj_t (row=target, col=source), so its edge-index
    # equivalent is [source=edge[1], target=edge[0]], not the stored ordering.
    # Keeping this reversal is essential: the top-k correlation graph is directed.
    graph = (edge.flip(0).to(device), weights.to(device))
    model = BulkFormer(
        dim=dim, graph=graph, gene_emb=torch.empty(0), gene_length=20010,
        bin_head=12, full_head=8, bins=0, gb_repeat=1, p_repeat=repeats,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = OrderedDict((key.removeprefix("module."), value) for key, value in state.items())
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


@torch.inference_mode()
def predict(model_name: str, matrix: np.ndarray, mask_ratio: float,
            device: torch.device, batch_size: int) -> np.ndarray:
    """Return reconstructed expression in native gene order."""
    total = len(matrix)
    started = time.monotonic()
    last_heartbeat = started
    print(
        f"[inference] model={model_name} samples={total:,} "
        f"batch_size={batch_size} device={device}",
        flush=True,
    )
    model = load_ours(device) if model_name == "ours_45.6m" else load_bulkformer(model_name, device)
    outputs = []
    autocast = device.type == "cuda"
    for start in range(0, len(matrix), batch_size):
        batch = torch.as_tensor(matrix[start:start + batch_size], dtype=torch.float32, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast):
            if model_name == "ours_45.6m":
                prediction = model(batch)
            else:
                prediction = model(batch, mask_prob=float(mask_ratio), output_expr=True)
        outputs.append(prediction.float().cpu().numpy())
        del batch, prediction
        now = time.monotonic()
        completed = min(start + batch_size, total)
        if now - last_heartbeat >= 60 or completed == total:
            elapsed = now - started
            print(
                f"[heartbeat] model={model_name} samples={completed:,}/{total:,} "
                f"elapsed={elapsed / 60:.1f}m rate={completed / elapsed:.2f}/s",
                flush=True,
            )
            last_heartbeat = now
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(outputs, axis=0)
