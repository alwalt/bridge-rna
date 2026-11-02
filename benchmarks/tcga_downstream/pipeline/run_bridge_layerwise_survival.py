#!/usr/bin/env python3
"""Evaluate frozen Bridge L1-L12 means with the fixed five-fold OOF Cox design."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO_ROOT, RESULTS, WORK  # noqa: E402
from run_attention_pooling import load_frozen_encoder, verify_input_contract  # noqa: E402
from run_survival_5fold_oof import (OOF_SEED, fold_aware_cindex, load_cohort,
                                    make_folds, train_mlp)  # noqa: E402

LAYERS = tuple(range(1, 13))
PERMUTATIONS = 100_000
TEST_SEED = 20260929
CACHE = WORK / "bridge_all_layer_mean_embeddings.float32.npy"
COMPLETED = WORK / "bridge_all_layer_mean_embeddings_completed.npy"
EXPRESSION = WORK / "ours_log1p_tpm.npy"
FINAL_EMBEDDINGS = WORK / "ours_45.6m_embeddings.npy"
FIGURES = RESULTS / "figures"


def say(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def initialize_cache() -> tuple[int, int]:
    expression = np.load(EXPRESSION, mmap_mode="r")
    samples = len(expression)
    if CACHE.is_file():
        cache = np.load(CACHE, mmap_mode="r")
        assert cache.shape == (samples, len(LAYERS), 512) and cache.dtype == np.float32
    else:
        cache = np.lib.format.open_memmap(
            CACHE, mode="w+", dtype=np.float32, shape=(samples, len(LAYERS), 512))
        cache.flush()
    return samples, expression.shape[1]


def completion_path(worker: int, workers: int) -> Path:
    return WORK / f"bridge_all_layer_mean_embeddings_completed_{worker}of{workers}.npy"


def extract_worker(worker: int, devices: list[str], batch_size: int, heartbeat: int) -> None:
    device = torch.device(devices[worker])
    expression = np.load(EXPRESSION, mmap_mode="r")
    output = np.load(CACHE, mmap_mode="r+")
    done_path = completion_path(worker, len(devices))
    completed = np.load(done_path) if done_path.is_file() else np.zeros(len(expression), dtype=bool)
    assigned = np.arange(worker, len(expression), len(devices))
    pending = assigned[~completed[assigned]]
    if not len(pending):
        say(f"layer cache worker={worker} device={device}: nothing pending")
        return
    model = load_frozen_encoder(device)
    ids = torch.arange(expression.shape[1], device=device)
    started = last = time.monotonic()
    for offset in range(0, len(pending), batch_size):
        rows = pending[offset:offset + batch_size]
        values = torch.as_tensor(np.asarray(expression[rows]), dtype=torch.float32, device=device)
        means = []
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16,
                                            enabled=device.type == "cuda"):
            hidden = model.gene_embedding(ids).unsqueeze(0) + model.ree(values)
            for layer in model.layers:
                hidden = layer(hidden)
                means.append(hidden.mean(dim=1).float().cpu().numpy())
        output[rows] = np.stack(means, axis=1)
        completed[rows] = True
        now = time.monotonic()
        if now - last >= heartbeat or offset + len(rows) == len(pending):
            output.flush(); np.save(done_path, completed)
            done = min(offset + len(rows), len(pending)); rate = done / max(now - started, 1e-9)
            eta = (len(pending) - done) / max(rate, 1e-9)
            say(f"layer cache worker={worker} device={device} {done:,}/{len(pending):,} "
                f"elapsed={(now-started)/60:.1f}m eta={eta/60:.1f}m")
            last = now
    output.flush(); np.save(done_path, completed)


def extract(devices: list[str], batch_size: int, heartbeat: int) -> None:
    samples, genes = initialize_cache()
    say(f"layer cache startup samples={samples:,} genes={genes:,} layers=12 "
        f"devices={devices}; expected ~{20/max(len(devices), 1):.0f} minutes")
    if len(devices) == 1:
        extract_worker(0, devices, batch_size, heartbeat)
    else:
        context = mp.get_context("spawn")
        processes = [context.Process(target=extract_worker,
                                     args=(worker, devices, batch_size, heartbeat))
                     for worker in range(len(devices))]
        for process in processes: process.start()
        for process in processes: process.join()
        assert all(process.exitcode == 0 for process in processes)
    completed = np.zeros(samples, dtype=bool)
    for worker in range(len(devices)):
        completed |= np.load(completion_path(worker, len(devices)))
    np.save(COMPLETED, completed)
    assert completed.all(), f"Incomplete layer cache: {completed.sum()}/{len(completed)}"
    cache = np.load(CACHE, mmap_mode="r")
    final = np.load(FINAL_EMBEDDINGS, mmap_mode="r")
    difference = np.abs(np.asarray(cache[:, -1], dtype=np.float32) - np.asarray(final, dtype=np.float32))
    qa = {"max_abs_l12_difference": float(difference.max()),
          "mean_abs_l12_difference": float(difference.mean()),
          "correlation_l12_flat": float(np.corrcoef(cache[:, -1].ravel(), final.ravel())[0, 1])}
    # Both paths are the same frozen forward pass under autocast. A small
    # floating reduction difference is allowed, but gross mismatch is not.
    assert qa["correlation_l12_flat"] > .999999
    (RESULTS / "bridge_layerwise_embedding_qa.json").write_text(json.dumps(qa, indent=2) + "\n")
    say(f"L12 cache QA: {qa}")


def train_predictions(device: torch.device) -> pd.DataFrame:
    output = RESULTS / "bridge_layerwise_survival_5fold_oof_predictions.csv"
    if output.is_file():
        say(f"reusing {output.name}")
        return pd.read_csv(output)
    cohort = load_cohort(); folds = make_folds(cohort)
    cohort = cohort.copy()
    cohort["fold"] = cohort.patient_id.map(folds.set_index("patient_id").fold).astype(int)
    cache = np.load(CACHE, mmap_mode="r")
    existing_l12 = pd.read_csv(RESULTS / "survival_5fold_oof_predictions.csv")
    existing_l12 = existing_l12[existing_l12.representation.eq("Bridge 45.6M")].copy()
    pieces = []
    for layer in LAYERS:
        if layer == 12:
            frame = existing_l12.rename(columns={"risk_score": "risk"})
            frame["layer"] = layer
            pieces.append(frame[["patient_id", "cancer", "survival_time", "event", "fold", "layer", "risk"]])
            continue
        for fold in range(5):
            part = WORK / f"bridge_layerwise_survival_l{layer:02d}_fold{fold}.csv"
            if part.is_file():
                frame = pd.read_csv(part)
            else:
                test = np.flatnonzero(cohort.fold.to_numpy() == fold)
                train = np.flatnonzero(cohort.fold.to_numpy() != fold)
                started = time.monotonic()
                risk, epochs = train_mlp(cache[:, layer - 1, :], cohort, train, test, fold, device)
                held = cohort.iloc[test]
                frame = pd.DataFrame({"patient_id": held.patient_id, "cancer": held.cohort,
                    "survival_time": held.time_days, "event": held.event.astype(int),
                    "fold": fold, "layer": layer, "risk": risk})
                frame.to_csv(part, index=False)
                say(f"Cox layer=L{layer} fold={fold} n={len(test)} epochs={epochs} "
                    f"elapsed={(time.monotonic()-started)/60:.1f}m")
            pieces.append(frame)
    predictions = pd.concat(pieces, ignore_index=True)
    expected = len(cohort)
    assert predictions.groupby("layer").size().eq(expected).all()
    assert predictions.groupby(["layer", "patient_id"]).size().eq(1).all()
    reference = predictions[predictions.layer.eq(12)].sort_values("patient_id")
    for layer in LAYERS[:-1]:
        other = predictions[predictions.layer.eq(layer)].sort_values("patient_id")
        assert np.array_equal(reference.patient_id.to_numpy(), other.patient_id.to_numpy())
        assert np.array_equal(reference.fold.to_numpy(), other.fold.to_numpy())
        assert np.array_equal(reference.event.to_numpy(), other.event.to_numpy())
    predictions.to_csv(output, index=False)
    return predictions


def metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall_rows, cancer_rows = [], []
    for layer in LAYERS:
        layer_frame = predictions[predictions.layer.eq(layer)].rename(columns={"risk": "risk_score"})
        value, pairs = fold_aware_cindex(layer_frame)
        overall_rows.append({"layer": layer, "oof_c_index": value, "comparable_pairs": pairs})
        for cancer, frame in layer_frame.groupby("cancer"):
            value, pairs = fold_aware_cindex(frame)
            cancer_rows.append({"layer": layer, "cancer": cancer, "n_patients": len(frame),
                                "n_events": int(frame.event.sum()), "comparable_pairs": pairs,
                                "c_index": value})
    overall = pd.DataFrame(overall_rows)
    per_cancer = pd.DataFrame(cancer_rows)
    overall.to_csv(RESULTS / "bridge_layerwise_survival_5fold_oof_metrics.csv", index=False)
    per_cancer.to_csv(RESULTS / "bridge_layerwise_survival_5fold_oof_per_cancer.csv", index=False)
    wide = per_cancer.pivot(index="cancer", columns="layer", values="c_index")
    support = per_cancer[per_cancer.layer.eq(12)].set_index("cancer")
    rng = np.random.default_rng(TEST_SEED)
    summary = []
    l12 = wide[12]
    for layer in LAYERS:
        difference = (wide[layer] - l12).dropna().to_numpy()
        if layer == 12:
            signflip = wilcoxon_p = 1.0
        else:
            null = (rng.choice([-1, 1], size=(PERMUTATIONS, len(difference))) * difference).mean(axis=1)
            signflip = (1 + (np.abs(null) >= abs(difference.mean())).sum()) / (PERMUTATIONS + 1)
            wilcoxon_p = float(wilcoxon(difference).pvalue)
        summary.append({"layer": layer,
            "overall_oof_c_index": overall.set_index("layer").loc[layer, "oof_c_index"],
            "median_within_cancer_c_index": wide[layer].median(),
            "patient_weighted_within_cancer_c_index": np.average(wide[layer], weights=support.n_patients),
            "cancers_better_than_l12": int((wide[layer] > l12).sum()),
            "median_delta_vs_l12": float(np.median(difference)),
            "mean_delta_vs_l12": float(np.mean(difference)),
            "signflip_mean_p_vs_l12": signflip, "wilcoxon_p_vs_l12": wilcoxon_p})
    summary = pd.DataFrame(summary)
    for column in ["signflip_mean_p_vs_l12", "wilcoxon_p_vs_l12"]:
        indices = summary.layer.ne(12)
        values = summary.loc[indices, column].to_numpy()
        order = np.argsort(values); adjusted = np.empty_like(values)
        ranked = np.minimum.accumulate((values[order] * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
        adjusted[order] = np.minimum(ranked, 1.0)
        summary.loc[indices, column.replace("_p_", "_bh_p_")] = adjusted
        summary.loc[~indices, column.replace("_p_", "_bh_p_")] = 1.0
    summary.to_csv(RESULTS / "bridge_layerwise_survival_summary.csv", index=False)
    best_layer = wide.idxmax(axis=1)
    best = support[["n_patients", "n_events", "comparable_pairs"]].copy()
    best["best_layer"] = best_layer
    best["best_c_index"] = wide.max(axis=1)
    best["l12_c_index"] = wide[12]
    best["best_delta_vs_l12"] = best.best_c_index - best.l12_c_index
    best.reset_index().to_csv(RESULTS / "bridge_layerwise_survival_best_layer_per_cancer.csv", index=False)
    return overall, per_cancer, summary


def plots(overall: pd.DataFrame, per_cancer: pd.DataFrame, summary: pd.DataFrame) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(summary.layer, summary.overall_oof_c_index, marker="o", label="Overall fold-aware OOF")
    ax.plot(summary.layer, summary.median_within_cancer_c_index, marker="s", label="Median within-cancer")
    ax.axvline(12, color="#777", ls="--", lw=1); ax.set_xticks(LAYERS)
    ax.set(xlabel="Bridge transformer layer", ylabel="C-index",
           title="Cox survival discrimination across Bridge depth")
    ax.grid(alpha=.25); ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(FIGURES / "bridge_layerwise_survival_cindex.png", dpi=300)
    fig.savefig(FIGURES / "bridge_layerwise_survival_cindex.pdf"); plt.close(fig)

    matrix = per_cancer.pivot(index="cancer", columns="layer", values="c_index")
    order = matrix[12].sort_values(ascending=False).index
    matrix = matrix.loc[order]
    fig, ax = plt.subplots(figsize=(12.5, 11.5))
    image = ax.imshow(matrix, aspect="auto", cmap="RdYlBu", vmin=.3, vmax=.9)
    ax.set_xticks(np.arange(12), [f"L{x}" for x in LAYERS]); ax.set_yticks(np.arange(len(matrix)), matrix.index)
    ax.set(xlabel="Bridge transformer layer", ylabel="TCGA cancer type",
           title="Five-fold OOF within-cancer Cox C-index across Bridge layers\n(cancers ranked by L12)")
    for i in range(len(matrix)):
        for j in range(12):
            value = matrix.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=5.8,
                    color="white" if value < .4 or value > .8 else "black")
    fig.colorbar(image, ax=ax, pad=.015, label="Fold-aware C-index")
    fig.tight_layout(); fig.savefig(FIGURES / "bridge_layerwise_per_cancer_survival_heatmap.png", dpi=300)
    fig.savefig(FIGURES / "bridge_layerwise_per_cancer_survival_heatmap.pdf"); plt.close(fig)

    delta = matrix.subtract(matrix[12], axis=0).drop(columns=12)
    bound = max(.1, float(np.nanmax(np.abs(delta.to_numpy()))))
    fig, ax = plt.subplots(figsize=(11.8, 11.5))
    image = ax.imshow(delta, aspect="auto", cmap="RdBu_r", vmin=-bound, vmax=bound)
    ax.set_xticks(np.arange(11), [f"L{x} − L12" for x in LAYERS[:-1]], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(delta)), delta.index)
    ax.set(xlabel="Layer contrast", ylabel="TCGA cancer type",
           title="Within-cancer C-index difference from Bridge L12")
    fig.colorbar(image, ax=ax, pad=.015, label="Δ C-index")
    fig.tight_layout(); fig.savefig(FIGURES / "bridge_layerwise_vs_l12_delta_heatmap.png", dpi=300)
    fig.savefig(FIGURES / "bridge_layerwise_vs_l12_delta_heatmap.pdf"); plt.close(fig)


def provenance(devices: list[str]) -> None:
    record = {"analysis": "Bridge L1-L12 mean-pooled frozen representation Cox survival ablation",
        "checkpoint": "model/r7hnr92k/config.json", "layers": list(LAYERS),
        "pooling": "mean over 15,165 contextual gene tokens after each transformer layer",
        "encoder_frozen": True, "new_encoder_inference": "one layerwise frozen pass",
        "cohort": "same 9,668-patient TCGA survival cohort",
        "folds": "same saved true five-fold OOF manifest",
        "head": "same MLP [512,256], Cox partial likelihood, fixed hyperparameters and inner validation",
        "metric": "fold-aware C-index; pairs never cross independently fitted heads",
        "comparison": "exploratory paired cancer-level layer-versus-L12 tests",
        "test_seed": TEST_SEED, "permutations": PERMUTATIONS, "devices": devices,
        "selection_warning": "No layer should replace the prespecified L12 primary representation based solely on this test."}
    (RESULTS / "bridge_layerwise_survival_provenance.json").write_text(json.dumps(record, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--heartbeat-seconds", type=int, default=60)
    args = parser.parse_args(); devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    if not torch.cuda.is_available(): devices = ["cpu"]
    verify_input_contract(); provenance(devices); extract(devices, args.batch_size, args.heartbeat_seconds)
    predictions = train_predictions(torch.device(devices[0])); overall, per_cancer, summary = metrics(predictions)
    plots(overall, per_cancer, summary)
    say("layerwise summary\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
