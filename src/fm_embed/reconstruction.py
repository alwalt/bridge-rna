"""Reusable masking, reconstruction, and scoring helpers for frozen-model benchmarks."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterable

import numpy as np
import torch
from scipy.stats import rankdata


MASK_TOKEN = -10.0


def deterministic_panel(universe: np.ndarray, size: int, seed: int, label: str) -> np.ndarray:
    """Select one fixed, sorted gene panel reproducibly."""
    universe = np.asarray(universe, dtype=np.int64)
    if size <= 0 or size > len(universe):
        raise ValueError(f"Panel size {size} is outside 1..{len(universe)}")
    digest = hashlib.sha256(f"{seed}|{label}|{size}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    return np.sort(rng.choice(universe, size=size, replace=False))


def mask_except_panel(matrix: np.ndarray, visible_indices: Iterable[int],
                      mask_token: float = MASK_TOKEN) -> np.ndarray:
    """Keep one fixed gene panel visible for every sample and mask all other positions."""
    visible = np.asarray(list(visible_indices), dtype=np.int64)
    masked = np.full(np.asarray(matrix).shape, mask_token, dtype=np.float32)
    masked[:, visible] = np.asarray(matrix)[:, visible]
    return masked


def score_masked_rows(truth: np.ndarray, prediction: np.ndarray,
                      score_indices: Iterable[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pearson, Spearman, and MSE over fixed masked positions for each sample."""
    indices = np.asarray(list(score_indices), dtype=np.int64)
    left = np.asarray(truth)[:, indices]
    right = np.asarray(prediction)[:, indices]
    lc = left - left.mean(axis=1, keepdims=True)
    rc = right - right.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(lc, axis=1) * np.linalg.norm(rc, axis=1)
    pearson = np.divide((lc * rc).sum(axis=1), denom,
                        out=np.full(len(left), np.nan), where=denom > 0)
    spearman = np.empty(len(left), dtype=np.float64)
    for row in range(len(left)):
        a, b = rankdata(left[row]), rankdata(right[row])
        ac, bc = a - a.mean(), b - b.mean()
        d = np.linalg.norm(ac) * np.linalg.norm(bc)
        spearman[row] = np.dot(ac, bc) / d if d else np.nan
    mse = np.mean((left - right) ** 2, axis=1)
    return pearson, spearman, mse


@torch.inference_mode()
def reconstruct(model: torch.nn.Module, matrix: np.ndarray, device: torch.device,
                batch_size: int = 4, label: str = "condition") -> np.ndarray:
    """Run a previously loaded frozen model with heartbeat output."""
    outputs, started, last = [], time.monotonic(), time.monotonic()
    total = len(matrix)
    print(f"[inference] {label} samples={total:,} batch_size={batch_size} device={device}", flush=True)
    for start in range(0, total, batch_size):
        batch = torch.as_tensor(matrix[start:start + batch_size], dtype=torch.float32, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            output = model(batch)
        outputs.append(output.float().cpu().numpy())
        now, completed = time.monotonic(), min(start + batch_size, total)
        if now - last >= 60 or completed == total:
            elapsed = now - started
            print(f"[heartbeat] {label} samples={completed:,}/{total:,} "
                  f"elapsed={elapsed/60:.1f}m rate={completed/elapsed:.2f}/s", flush=True)
            last = now
    return np.concatenate(outputs)
