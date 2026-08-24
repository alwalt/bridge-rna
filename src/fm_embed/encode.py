"""Batched GPU-safe inference through a frozen ExpressionPerformer."""

from __future__ import annotations

import gc
import time
from typing import List, Optional

import numpy as np
import torch


def heartbeat(label: str, current: int, total: int, every: int = 1, start_time: Optional[float] = None) -> None:
    if current == 0 or current == total or (current % every == 0):
        pct = 100.0 * current / total if total else 100.0
        prefix = f"[{label}] {current}/{total} ({pct:.1f}%)"
        if start_time is not None:
            print(f"{prefix} | elapsed={time.time() - start_time:.1f}s")
        else:
            print(prefix)


def encode_matrix(
    model,
    device: torch.device,
    X: np.ndarray,
    batch_size: int = 32,
    normalize: bool = False,
    label: str = "encoding",
) -> np.ndarray:
    """Encode a preprocessed [n_samples, n_genes] matrix into [n_samples, hidden_dim]."""
    chunks: List[np.ndarray] = []
    total = X.shape[0]
    start_time = time.time()
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        heartbeat(label, end, total, every=max(1, batch_size // 4), start_time=start_time)
        chunk = torch.from_numpy(X[start:end].astype(np.float32)).to(device)
        with torch.no_grad():
            emb = model.encode(chunk, normalize=normalize).detach().cpu().numpy()
        chunks.append(emb)
        del chunk, emb
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not chunks:
        return np.empty((0, 512), dtype=np.float32)
    return np.concatenate(chunks, axis=0)
