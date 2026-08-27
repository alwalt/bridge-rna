"""Load the pretrained ExpressionPerformer checkpoint for inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import torch

from .inference_model import ExpressionPerformer, strip_module_prefix


def load_expression_performer(
    checkpoint_path: Path,
    config_path: Path,
    num_genes: int,
    device: str = "cuda",
) -> Tuple[ExpressionPerformer, torch.device]:
    """Load the frozen ExpressionPerformer model, ready for `.encode(...)`.

    `num_genes` must equal the length of the canonical gene vocab used to
    build the input matrix (see `fm_embed.vocab.load_canonical_genes`).
    """
    checkpoint_path = Path(checkpoint_path)
    config_path = Path(config_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")

    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    cfg = json.loads(config_path.read_text())

    model = ExpressionPerformer(
        num_genes=num_genes,
        hidden_dim=int(cfg.get("hidden_dim", 512)),
        n_heads=int(cfg.get("num_heads", 8)),
        n_layers=int(cfg.get("num_layers", 12)),
        ffn_dim=int(cfg.get("ffn_dim", 2048)),
        ree_base=float(cfg.get("ree_base", 100.0)),
        mask_token_id=float(cfg.get("mask_token", -10.0)),
        feature_type=str(cfg.get("feature_type", "flash")),
        compute_type=str(cfg.get("compute_type", "iter")),
        include_species_embedding=bool(cfg.get("include_species_embedding", False)),
        num_species=2,
    )
    expected_genes = cfg.get("num_genes")
    if expected_genes is not None and int(expected_genes) != num_genes:
        raise ValueError(f"Checkpoint expects {expected_genes} genes, received {num_genes}")
    missing, unexpected = model.load_state_dict(
        strip_module_prefix(ckpt["model_state_dict"]), strict=False
    )
    if missing or unexpected:
        raise ValueError(
            f"Checkpoint mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    resolved_device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    model.to(resolved_device)
    model.eval()
    return model, resolved_device
