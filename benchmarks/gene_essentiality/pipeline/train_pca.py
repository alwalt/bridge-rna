#!/usr/bin/env python3
"""Leakage-safe target-conditioned PCA and Bridge+PCA benchmarks."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.decomposition import PCA

from train_contextual import CommonMLP, pair_batches, random_pairs, split_rows


HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
CONFIG = json.loads((BENCH / "config.json").read_text())
WORK = Path(CONFIG["shared_work_root"])
RESULTS = BENCH / "results"
HEAD = CONFIG["mlp"]


def pair_features(rows, genes, scores, loadings, target_expression, bridge_tokens,
                  bridge_positions) -> np.ndarray:
    contribution = scores[rows] * loadings[:, genes].T
    expression_value = target_expression[rows, genes, None]
    conventional = np.concatenate([contribution, expression_value], axis=1).astype(np.float32)
    if bridge_tokens is None:
        return conventional
    bridge = np.asarray(bridge_tokens[rows, bridge_positions[genes]], dtype=np.float32)
    return np.concatenate([bridge, conventional], axis=1)


def evaluate_loss(model, rows, genes, scores, loadings, target_expression, bridge_tokens,
                  bridge_positions, dependency, target_columns, device) -> float:
    model.eval(); total = count = 0
    with torch.inference_mode():
        for batch_rows, batch_genes in pair_batches(rows, genes, int(HEAD["batch_size"])):
            features = torch.as_tensor(pair_features(batch_rows, batch_genes, scores, loadings,
                target_expression, bridge_tokens, bridge_positions), device=device)
            labels = torch.as_tensor(dependency[batch_rows, target_columns[batch_genes]], dtype=torch.float32, device=device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = torch.nn.functional.mse_loss(model(features).float(), labels)
            total += float(loss) * len(labels); count += len(labels)
    return total / count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["pca", "bridge_pca"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fold", type=int, default=None)
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    expression = np.load(WORK / "expression_released_log2_tpm_plus_1.float32.npy", mmap_mode="r")
    dependency = np.load(WORK / "dependency.float32.npy", mmap_mode="r")
    folds = pd.read_csv(RESULTS / "cv_folds.csv")
    targets = pd.read_csv(RESULTS / "gene_overlap.csv")
    manifest = targets[targets.all_method_common].reset_index(drop=True)
    target_columns = manifest.dependency_column_index.to_numpy(dtype=int)
    expression_columns = manifest.expression_column_index.to_numpy(dtype=int)
    target_expression = expression[:, expression_columns]
    bridge_tokens = bridge_positions = None
    if args.mode == "bridge_pca":
        bridge_dir = WORK / "bridge_contextual"
        bridge_tokens = np.load(bridge_dir / "tokens.float16.npy", mmap_mode="r")
        bridge_manifest = pd.read_csv(bridge_dir / "target_manifest.csv")
        position = dict(zip(bridge_manifest.gene_id, np.arange(len(bridge_manifest))))
        bridge_positions = np.array([position[g] for g in manifest.gene_id], dtype=int)
    output_dir = WORK / "predictions" / f"{args.mode}_matched"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_dir / "target_manifest.csv", index=False)
    fold_values = [args.fold] if args.fold is not None else range(int(CONFIG["cv_folds"]))
    for fold in fold_values:
        seed = int(CONFIG["seed"]) + fold
        torch.manual_seed(seed); np.random.seed(seed)
        train_rows = split_rows(folds, fold, "train")
        validation_rows = split_rows(folds, fold, "validation")
        test_rows = split_rows(folds, fold, "test")
        started = time.monotonic()
        pca = PCA(n_components=int(CONFIG["pca_components"]), svd_solver="randomized",
                  random_state=seed, iterated_power=4)
        pca.fit(np.asarray(expression[train_rows], dtype=np.float32))
        scores = pca.transform(np.asarray(expression, dtype=np.float32)).astype(np.float32)
        loadings = pca.components_[:, expression_columns].astype(np.float32)
        np.savez_compressed(output_dir / f"fold_{fold:02d}_pca_provenance.npz",
            explained_variance_ratio=pca.explained_variance_ratio_, mean=pca.mean_,
            singular_values=pca.singular_values_)
        validation_pair_rows, validation_pair_genes = random_pairs(
            validation_rows, target_columns, dependency, int(HEAD["validation_pairs"]), seed + 20_000)
        input_dim = 513 + (512 if bridge_tokens is not None else 0)
        model = CommonMLP(input_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(HEAD["learning_rate"]),
                                      weight_decay=float(HEAD["weight_decay"]))
        best_loss = math.inf; best_state = None; stale = 0; history = []
        print(f"[startup] {args.mode} fold={fold} genes={len(manifest)} components={pca.n_components_} "
              f"explained={pca.explained_variance_ratio_.sum():.6f} device={device}", flush=True)
        for epoch in range(int(HEAD["max_epochs"])):
            pair_rows, pair_genes = random_pairs(train_rows, target_columns, dependency,
                int(HEAD["pairs_per_training_epoch"]), seed + epoch * 100_003)
            model.train(); running = seen = 0
            for batch_rows, batch_genes in pair_batches(pair_rows, pair_genes, int(HEAD["batch_size"]), seed + epoch):
                features = torch.as_tensor(pair_features(batch_rows, batch_genes, scores, loadings,
                    target_expression, bridge_tokens, bridge_positions), device=device)
                labels = torch.as_tensor(dependency[batch_rows, target_columns[batch_genes]], dtype=torch.float32, device=device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    predictions = model(features)
                    loss = torch.nn.functional.mse_loss(predictions.float(), labels)
                loss.backward(); optimizer.step()
                running += float(loss.detach()) * len(labels); seen += len(labels)
            validation_loss = evaluate_loss(model, validation_pair_rows, validation_pair_genes,
                scores, loadings, target_expression, bridge_tokens, bridge_positions,
                dependency, target_columns, device)
            train_loss = running / seen
            history.append({"fold": fold, "epoch": epoch + 1, "train_mse": train_loss,
                            "validation_mse": validation_loss})
            if validation_loss < best_loss - float(HEAD["minimum_delta"]):
                best_loss = validation_loss; best_state = copy.deepcopy(model.state_dict()); stale = 0
            else:
                stale += 1
            print(f"[heartbeat] {args.mode} fold={fold} epoch={epoch+1} train_mse={train_loss:.7f} "
                  f"validation_mse={validation_loss:.7f} best={best_loss:.7f} "
                  f"elapsed={(time.monotonic()-started)/60:.1f}m", flush=True)
            if stale >= int(HEAD["patience"]):
                break
        model.load_state_dict(best_state)
        output = np.empty((len(test_rows), len(manifest)), dtype=np.float32)
        model.eval()
        with torch.inference_mode():
            for out_row, source_row in enumerate(test_rows):
                for start in range(0, len(manifest), int(HEAD["batch_size"])):
                    stop = min(start + int(HEAD["batch_size"]), len(manifest))
                    genes = np.arange(start, stop)
                    rows = np.full(len(genes), source_row)
                    features = torch.as_tensor(pair_features(rows, genes, scores, loadings,
                        target_expression, bridge_tokens, bridge_positions), device=device)
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                        output[out_row, start:stop] = model(features).float().cpu().numpy()
        np.save(output_dir / f"fold_{fold:02d}_predictions.float32.npy", output)
        pd.DataFrame({"prediction_row": np.arange(len(test_rows)), "source_row": test_rows,
                      "cell_line_id": pd.read_csv(RESULTS / "cell_lines.csv").iloc[test_rows].cell_line_id}).to_csv(
            output_dir / f"fold_{fold:02d}_test_rows.csv", index=False)
        pd.DataFrame(history).to_csv(output_dir / f"fold_{fold:02d}_history.csv", index=False)
        torch.save(best_state, output_dir / f"fold_{fold:02d}_head.pt")
        print(f"[complete] {args.mode} fold={fold} best_validation_mse={best_loss:.7f}", flush=True)


if __name__ == "__main__":
    main()
