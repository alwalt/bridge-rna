"""Load the pretrained ExpressionPerformer checkpoint for inference."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generate_archs4_embeddings import ExpressionPerformer, _strip_module_prefix  # noqa: E402


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
    model.load_state_dict(_strip_module_prefix(ckpt["model_state_dict"]), strict=False)

    resolved_device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    model.to(resolved_device)
    model.eval()
    return model, resolved_device
