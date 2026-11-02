#!/usr/bin/env python3
"""Train the prespecified common MLP on frozen contextual tokens."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

if not hasattr(sys, "get_int_max_str_digits"):
    def get_int_max_str_digits() -> int:
        return 4300
    sys.get_int_max_str_digits = get_int_max_str_digits  # type: ignore[attr-defined]
if not hasattr(sys, "set_int_max_str_digits"):
    def set_int_max_str_digits(maxdigits: int) -> None:
        del maxdigits
    sys.set_int_max_str_digits = set_int_max_str_digits  # type: ignore[attr-defined]

import torch
from torch import nn


HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
CONFIG = json.loads((BENCH / "config.json").read_text())
WORK = Path(CONFIG["shared_work_root"])
RESULTS = BENCH / "results"
HEAD = CONFIG["mlp"]


class CommonMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        h1, h2 = map(int, HEAD["hidden_dims"])
        self.network = nn.Sequential(
            nn.Linear(input_dim, h1), nn.GELU(), nn.Dropout(float(HEAD["dropout"])),
            nn.Linear(h1, h2), nn.GELU(), nn.Dropout(float(HEAD["dropout"])),
            nn.Linear(h2, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


def split_rows(folds: pd.DataFrame, fold: int, split: str) -> np.ndarray:
    return folds[(folds.fold == fold) & (folds.split == split)].row_index.to_numpy(dtype=int)


def random_pairs(rows: np.ndarray, target_columns: np.ndarray, dependency: np.ndarray,
                 count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    sample_rows, gene_positions = [], []
    remaining = count
    while remaining:
        draw = max(remaining, min(count, 100_000))
        candidate_rows = rng.choice(rows, draw, replace=True)
        candidate_genes = rng.integers(0, len(target_columns), draw)
        observed = np.isfinite(dependency[candidate_rows, target_columns[candidate_genes]])
        take = min(remaining, int(observed.sum()))
        if take:
            sample_rows.append(candidate_rows[observed][:take])
            gene_positions.append(candidate_genes[observed][:take])
            remaining -= take
    return np.concatenate(sample_rows), np.concatenate(gene_positions)


def pair_batches(rows: np.ndarray, genes: np.ndarray, batch_size: int,
                 permutation_seed: int | None = None):
    order = np.arange(len(rows))
    if permutation_seed is not None:
        np.random.default_rng(permutation_seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        chosen = order[start:start + batch_size]
        yield rows[chosen], genes[chosen]


def loss_on_pairs(model, tokens, token_positions, dependency, target_columns, rows, genes, device) -> float:
    model.eval(); total_loss = total_n = 0
    with torch.inference_mode():
        for batch_rows, batch_genes in pair_batches(rows, genes, int(HEAD["batch_size"])):
            features = torch.as_tensor(np.asarray(tokens[batch_rows, token_positions[batch_genes]], dtype=np.float32), device=device)
            labels = torch.as_tensor(dependency[batch_rows, target_columns[batch_genes]], dtype=torch.float32, device=device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = nn.functional.mse_loss(model(features).float(), labels)
            total_loss += float(loss) * len(labels); total_n += len(labels)
    return total_loss / total_n


def predict(model, tokens, token_positions, test_rows, device) -> np.ndarray:
    model.eval()
    output = np.empty((len(test_rows), len(token_positions)), dtype=np.float32)
    batch = int(HEAD["batch_size"])
    with torch.inference_mode():
        for out_row, source_row in enumerate(test_rows):
            for start in range(0, len(token_positions), batch):
                stop = min(start + batch, len(token_positions))
                features = torch.as_tensor(np.asarray(tokens[source_row, token_positions[start:stop]], dtype=np.float32), device=device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    output[out_row, start:stop] = model(features).float().cpu().numpy()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("representation", choices=["bridge", "bulkformer", "target_expression"])
    parser.add_argument("--gene-set", choices=["matched", "native"], default="matched")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fold", type=int, default=None)
    args = parser.parse_args()
    if args.representation in {"bridge", "target_expression"} and args.gene_set == "native":
        raise SystemExit("Bridge native equals its representable set; use matched for the primary comparison")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dependency = np.load(WORK / "dependency.float32.npy", mmap_mode="r")
    folds = pd.read_csv(RESULTS / "cv_folds.csv")
    cache_dir = WORK / f"{args.representation}_contextual"
    tokens_all = np.load(cache_dir / "tokens.float16.npy", mmap_mode="r")
    manifest = pd.read_csv(cache_dir / "target_manifest.csv")
    current_universe = pd.read_csv(RESULTS / "gene_overlap.csv")[["gene_id", "all_method_common"]]
    manifest = manifest.drop(columns=["all_method_common"], errors="ignore").merge(
        current_universe, on="gene_id", how="left", validate="one_to_one"
    )
    if args.gene_set == "matched":
        keep = manifest.all_method_common.to_numpy(dtype=bool)
    else:
        keep = np.ones(len(manifest), dtype=bool)
    cache_gene_positions = np.flatnonzero(keep)
    manifest = manifest.loc[keep].reset_index(drop=True)
    target_columns = manifest.dependency_column_index.to_numpy(dtype=int)
    output_dir = WORK / "predictions" / f"{args.representation}_{args.gene_set}"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_dir / "target_manifest.csv", index=False)
    fold_values = [args.fold] if args.fold is not None else list(range(int(CONFIG["cv_folds"])))
    for fold in fold_values:
        seed = int(CONFIG["seed"]) + fold
        torch.manual_seed(seed); np.random.seed(seed)
        train_rows = split_rows(folds, fold, "train")
        validation_rows = split_rows(folds, fold, "validation")
        test_rows = split_rows(folds, fold, "test")
        validation_pair_rows, validation_pair_genes = random_pairs(
            validation_rows, target_columns, dependency, int(HEAD["validation_pairs"]), seed + 20_000
        )
        model = CommonMLP(tokens_all.shape[2]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(HEAD["learning_rate"]),
                                      weight_decay=float(HEAD["weight_decay"]))
        best_loss = math.inf; best_state = None; stale = 0; history = []
        started = time.monotonic()
        print(f"[startup] {args.representation}/{args.gene_set} fold={fold} train={len(train_rows)} "
              f"validation={len(validation_rows)} test={len(test_rows)} genes={len(target_columns)} device={device}", flush=True)
        for epoch in range(int(HEAD["max_epochs"])):
            pair_rows, pair_genes = random_pairs(
                train_rows, target_columns, dependency, int(HEAD["pairs_per_training_epoch"]),
                seed + epoch * 100_003,
            )
            model.train(); running = seen = 0
            for batch_rows, batch_genes in pair_batches(pair_rows, pair_genes, int(HEAD["batch_size"]), seed + epoch):
                features = torch.as_tensor(np.asarray(tokens_all[batch_rows, cache_gene_positions[batch_genes]], dtype=np.float32), device=device)
                labels = torch.as_tensor(dependency[batch_rows, target_columns[batch_genes]], dtype=torch.float32, device=device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    predictions = model(features)
                    loss = nn.functional.mse_loss(predictions.float(), labels)
                loss.backward(); optimizer.step()
                running += float(loss.detach()) * len(labels); seen += len(labels)
            validation_loss = loss_on_pairs(model, tokens_all, cache_gene_positions, dependency, target_columns,
                                            validation_pair_rows, validation_pair_genes, device)
            train_loss = running / seen
            history.append({"fold": fold, "epoch": epoch + 1, "train_mse": train_loss,
                            "validation_mse": validation_loss})
            improved = validation_loss < best_loss - float(HEAD["minimum_delta"])
            if improved:
                best_loss = validation_loss; best_state = copy.deepcopy(model.state_dict()); stale = 0
            else:
                stale += 1
            elapsed = time.monotonic() - started
            print(f"[heartbeat] fold={fold} epoch={epoch+1} train_mse={train_loss:.7f} "
                  f"validation_mse={validation_loss:.7f} best={best_loss:.7f} elapsed={elapsed/60:.1f}m", flush=True)
            if stale >= int(HEAD["patience"]):
                break
        if best_state is None:
            raise RuntimeError("No valid model state")
        model.load_state_dict(best_state)
        predictions = predict(model, tokens_all, cache_gene_positions, test_rows, device)
        np.save(output_dir / f"fold_{fold:02d}_predictions.float32.npy", predictions)
        pd.DataFrame({"prediction_row": np.arange(len(test_rows)), "source_row": test_rows,
                      "cell_line_id": pd.read_csv(RESULTS / "cell_lines.csv").iloc[test_rows].cell_line_id}).to_csv(
            output_dir / f"fold_{fold:02d}_test_rows.csv", index=False
        )
        pd.DataFrame(history).to_csv(output_dir / f"fold_{fold:02d}_history.csv", index=False)
        torch.save(best_state, output_dir / f"fold_{fold:02d}_head.pt")
        print(f"[complete] fold={fold} best_validation_mse={best_loss:.7f}", flush=True)


if __name__ == "__main__":
    main()
