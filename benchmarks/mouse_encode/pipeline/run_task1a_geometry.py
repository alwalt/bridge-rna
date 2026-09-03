#!/usr/bin/env python3
"""Representation-geometry diagnostics for the frozen Task 1A outputs."""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

# This workspace uses Python 3.11.0rc1; current torch._dynamo expects APIs
# introduced in the final 3.11 release (same compatibility shim used elsewhere).
if not hasattr(sys, "get_int_max_str_digits"):
    sys.get_int_max_str_digits = lambda: 4300  # type: ignore[attr-defined]
if not hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits = lambda maxdigits: None  # type: ignore[attr-defined]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parents[1]
RESULTS, WORK = HERE / "results", HERE / "work"
FIGURES = RESULTS / "figures" / "task1a_geometry"
TISSUES_11 = ["adrenal", "subcutaneous adipose", "cerebellum", "heart", "liver", "lung",
              "mammary gland", "ovary", "spleen", "stomach", "testis"]
TISSUES_5 = ["heart", "liver", "lung", "spleen", "testis"]
REPRESENTATIONS = ["raw_expression", "joint_pca", "bridgerna"]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return normalize_rows(a) @ normalize_rows(b).T


def centroids(x: np.ndarray, labels: np.ndarray, tissues: list[str]) -> np.ndarray:
    return np.stack([x[labels == tissue].mean(axis=0) for tissue in tissues])


def score_predictions(
    truth: np.ndarray, predicted: np.ndarray, tissues: list[str], scores: np.ndarray | None = None,
) -> tuple[dict, pd.DataFrame]:
    accuracy = float(np.mean(truth == predicted))
    macro_f1 = float(f1_score(truth, predicted, labels=tissues, average="macro", zero_division=0))
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=tissues, zero_division=0,
    )
    per_tissue = pd.DataFrame({"tissue": tissues, "samples": support, "accuracy_recall": recall,
                               "precision": precision, "f1": f1})
    result = {"top1_accuracy": accuracy, "macro_f1": macro_f1}
    if scores is not None:
        true_index = np.array([tissues.index(x) for x in truth])
        correct = scores[np.arange(len(scores)), true_index]
        incorrect = scores.copy(); incorrect[np.arange(len(scores)), true_index] = -np.inf
        result["mean_correct_tissue_margin"] = float(np.mean(correct - incorrect.max(axis=1)))
    else:
        result["mean_correct_tissue_margin"] = np.nan
    return result, per_tissue


def centroid_readout(
    query_x: np.ndarray, query_y: np.ndarray, ref_x: np.ndarray, ref_y: np.ndarray,
    tissues: list[str], center: bool,
) -> tuple[dict, np.ndarray, np.ndarray]:
    # Source/query-only mean: no target labels or target distribution enter centering.
    mean = query_x.mean(axis=0, keepdims=True) if center else np.zeros((1, query_x.shape[1]), dtype=query_x.dtype)
    query = query_x - mean
    reference = ref_x - mean
    scores = cosine(query, centroids(reference, ref_y, tissues))
    order = np.argsort(-scores, axis=1)
    true_i = np.array([tissues.index(x) for x in query_y])
    predicted = np.array([tissues[i] for i in order[:, 0]])
    ranks = np.array([np.flatnonzero(order[i] == true_i[i])[0] + 1 for i in range(len(order))])
    result, per_tissue = score_predictions(query_y, predicted, tissues, scores)
    result["mrr"] = float(np.mean(1 / ranks))
    per_tissue["mrr"] = [float(np.mean(1 / ranks[query_y == tissue])) for tissue in tissues]
    per_tissue["mean_correct_tissue_margin"] = [
        score_predictions(query_y[query_y == tissue], predicted[query_y == tissue], tissues,
                          scores[query_y == tissue])[0]["mean_correct_tissue_margin"] for tissue in tissues
    ]
    return result, predicted, scores


def knn_readout(
    query_x: np.ndarray, query_y: np.ndarray, ref_x: np.ndarray, ref_y: np.ndarray,
    tissues: list[str], k: int,
) -> tuple[dict, np.ndarray, np.ndarray]:
    similarities = cosine(query_x, ref_x)
    neighbors = np.argpartition(-similarities, kth=min(k - 1, len(ref_x) - 1), axis=1)[:, :k]
    class_scores = np.zeros((len(query_x), len(tissues)), dtype=np.float64)
    for i in range(len(query_x)):
        labels = ref_y[neighbors[i]]
        for j, tissue in enumerate(tissues):
            votes = labels == tissue
            # Vote count dominates; mean cosine resolves ties deterministically.
            class_scores[i, j] = votes.sum() + (similarities[i, neighbors[i][votes]].mean() if votes.any() else 0) * 1e-3
    predicted = np.array([tissues[i] for i in class_scores.argmax(axis=1)])
    result, _ = score_predictions(query_y, predicted, tissues, class_scores)
    result["mrr"] = np.nan
    return result, predicted, class_scores


def probe_readout(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray,
    tissues: list[str], nonlinear: bool, device: torch.device,
) -> tuple[dict, np.ndarray, np.ndarray]:
    # Fixed a priori architecture and schedule; no target data or labels select them.
    torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    model = (nn.Sequential(nn.Linear(train_x.shape[1], 64), nn.ReLU(), nn.Linear(64, len(tissues)))
             if nonlinear else nn.Linear(train_x.shape[1], len(tissues))).to(device)
    y_index = np.array([tissues.index(x) for x in train_y], dtype=np.int64)
    counts = np.bincount(y_index, minlength=len(tissues)).astype(np.float32)
    weights = len(y_index) / np.maximum(counts, 1) / len(tissues)
    loss_fn = nn.CrossEntropyLoss(weight=torch.as_tensor(weights, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = np.random.default_rng(42)
    model.train()
    for _ in range(50):
        order = generator.permutation(len(train_x))
        for start in range(0, len(order), 256):
            index = order[start:start + 256]
            batch_x = torch.as_tensor(train_x[index], dtype=torch.float32, device=device)
            batch_y = torch.as_tensor(y_index[index], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward(); optimizer.step()
    model.eval(); chunks = []
    with torch.inference_mode():
        for start in range(0, len(test_x), 512):
            chunks.append(model(torch.as_tensor(test_x[start:start + 512], dtype=torch.float32, device=device)).cpu().numpy())
    scores = np.concatenate(chunks)
    predicted = np.array([tissues[i] for i in scores.argmax(axis=1)])
    result, _ = score_predictions(test_y, predicted, tissues, scores)
    result["mrr"] = np.nan
    return result, predicted, scores


def append_result(
    summaries: list[dict], per_tissue_rows: list[pd.DataFrame], confusions: dict,
    cohort: str, representation: str, readout: str, direction: str, truth: np.ndarray,
    predicted: np.ndarray, scores: np.ndarray, tissues: list[str], result: dict,
) -> None:
    summaries.append({"cohort": cohort, "representation": representation, "readout": readout,
                      "direction": direction, "queries": len(truth), **result})
    _, per = score_predictions(truth, predicted, tissues, scores)
    if readout.startswith("centroid_cosine"):
        order = np.argsort(-scores, axis=1)
        true_index = np.array([tissues.index(x) for x in truth])
        ranks = np.array([np.flatnonzero(order[i] == true_index[i])[0] + 1 for i in range(len(order))])
        per["mrr"] = [float(np.mean(1 / ranks[truth == tissue])) for tissue in tissues]
    else:
        per["mrr"] = np.nan
    true_index = np.array([tissues.index(x) for x in truth])
    correct_score = scores[np.arange(len(scores)), true_index]
    incorrect = scores.copy(); incorrect[np.arange(len(scores)), true_index] = -np.inf
    margins = correct_score - incorrect.max(axis=1)
    per["mean_correct_tissue_margin"] = [float(np.mean(margins[truth == tissue])) for tissue in tissues]
    per.insert(0, "direction", direction); per.insert(0, "readout", readout)
    per.insert(0, "representation", representation); per.insert(0, "cohort", cohort)
    per_tissue_rows.append(per)
    confusions[(cohort, representation, readout, direction)] = confusion_matrix(truth, predicted, labels=tissues)


def geometry_stats(
    representation: str, cohort: str, human: np.ndarray, mouse: np.ndarray,
    human_y: np.ndarray, mouse_y: np.ndarray, tissues: list[str], rng: np.random.Generator,
) -> tuple[list[dict], pd.DataFrame]:
    stats = []
    norm_rows = []
    for species, x in [("human", human), ("mouse", mouse)]:
        norms = np.linalg.norm(x, axis=1)
        stats.append({"cohort": cohort, "representation": representation, "space": "original",
                      "geometry_measure": "norm", "group": species, "n": len(norms),
                      "mean": norms.mean(), "sd": norms.std(ddof=1) if len(norms) > 1 else 0,
                      "minimum": norms.min(), "maximum": norms.max(),
                      "p01": np.quantile(norms, .01), "p99": np.quantile(norms, .99)})
        norm_rows.extend({"cohort": cohort, "representation": representation, "species": species,
                          "norm": value} for value in norms)
    cross = cosine(human, mouse)
    same = human_y[:, None] == mouse_y[None, :]
    for group, values in [("same_tissue_cross_species", cross[same]), ("different_tissue_cross_species", cross[~same]),
                          ("all_cross_species", cross.ravel())]:
        stats.append({"cohort": cohort, "representation": representation, "space": "original",
                      "geometry_measure": "cosine", "group": group, "n": len(values),
                      "mean": values.mean(), "sd": values.std(ddof=1), "minimum": values.min(),
                      "maximum": values.max(), "p01": np.quantile(values, .01), "p99": np.quantile(values, .99)})
    combined = np.vstack([human, mouse]); n = len(combined); pair_n = min(20000, n * (n - 1) // 2)
    left = rng.integers(0, n, pair_n); right = rng.integers(0, n, pair_n)
    valid = left != right; left, right = left[valid], right[valid]
    pair_cos = np.empty(len(left), dtype=np.float32)
    for start in range(0, len(left), 512):
        stop = min(start + 512, len(left))
        pair_cos[start:stop] = np.sum(
            normalize_rows(combined[left[start:stop]]) * normalize_rows(combined[right[start:stop]]), axis=1,
        )
    stats.append({"cohort": cohort, "representation": representation, "space": "original",
                  "geometry_measure": "cosine", "group": "sampled_all_pairs", "n": len(pair_cos),
                  "mean": pair_cos.mean(), "sd": pair_cos.std(ddof=1), "minimum": pair_cos.min(),
                  "maximum": pair_cos.max(), "p01": np.quantile(pair_cos, .01), "p99": np.quantile(pair_cos, .99)})
    distributions = pd.DataFrame({"same_tissue": cross[same]})
    different = cross[~same]
    if len(different) > len(distributions): different = rng.choice(different, len(distributions), replace=False)
    distributions["different_tissue"] = different[:len(distributions)]
    distributions["cohort"] = cohort; distributions["representation"] = representation
    return stats, distributions


def plot_confusion_panels(confusions: dict, cohort: str, readout: str, tissues: list[str]) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(15, 20))
    for row, rep in enumerate(REPRESENTATIONS):
        for col, direction in enumerate(["human_to_mouse", "mouse_to_human"]):
            matrix = confusions[(cohort, rep, readout, direction)]
            values = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
            ax = axes[row, col]; image = ax.imshow(values, vmin=0, vmax=1, cmap="Blues", aspect="auto")
            ax.set_xticks(range(len(tissues)), tissues, rotation=45, ha="right"); ax.set_yticks(range(len(tissues)), tissues)
            ax.set(title=f"{rep} · {direction}", xlabel="Predicted", ylabel="True")
        fig.colorbar(image, ax=axes[row].tolist(), shrink=.7)
    fig.suptitle(f"{cohort} · {readout}", y=.995); fig.subplots_adjust(hspace=.35, wspace=.3, top=.96)
    fig.savefig(FIGURES / f"confusions_{cohort}_{readout}.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def plot_geometry(distributions: pd.DataFrame, stats: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for col, rep in enumerate(REPRESENTATIONS):
        subset = distributions[(distributions.cohort == "complete_11_tissue") & (distributions.representation == rep)]
        axes[0, col].hist(subset.same_tissue.dropna(), bins=50, density=True, alpha=.55, label="Same tissue")
        axes[0, col].hist(subset.different_tissue.dropna(), bins=50, density=True, alpha=.55, label="Different tissue")
        axes[0, col].set(title=rep, xlabel="Cross-species cosine", ylabel="Density"); axes[0, col].legend()
        norm = stats[(stats.cohort == "complete_11_tissue") & (stats.representation == rep) & (stats.geometry_measure == "norm")]
        axes[1, col].bar(norm.group, norm["mean"], yerr=norm.sd, capsize=4)
        axes[1, col].set(xlabel="Species", ylabel="Norm mean ± SD")
    fig.tight_layout(); fig.savefig(FIGURES / "cosine_and_norm_distributions.png", dpi=220); plt.close(fig)


def main(args: argparse.Namespace) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_parquet(RESULTS / "expanded_task1a_sample_manifest.parquet")
    nh = int(manifest.species.eq("human").sum())
    reps = {
        "raw_expression": (np.load(WORK / "expanded_gtex_log1p_tpm.npy"), np.load(WORK / "expanded_encode_log1p_tpm.npy")),
        "joint_pca": tuple(np.split(np.load(WORK / "expanded_joint_pca_scores.npy"), [nh])),
        "bridgerna": (np.load(WORK / "expanded_gtex_bridgerna_embeddings.npy"), np.load(WORK / "expanded_encode_bridgerna_embeddings.npy")),
    }
    human_meta = manifest[manifest.species.eq("human")].reset_index(drop=True)
    mouse_meta = manifest[manifest.species.eq("mouse")].reset_index(drop=True)
    if not mouse_meta.exposure_class.eq("fully_unseen").all(): raise ValueError("Mouse cohort is not fully unseen")
    summaries, per_tissue_rows, geometry_rows, distribution_rows, confusions = [], [], [], [], {}
    rng = np.random.default_rng(42)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    for cohort, tissues in [("complete_11_tissue", TISSUES_11), ("replicated_5_tissue", TISSUES_5)]:
        hm = human_meta.tissue.isin(tissues).to_numpy(); mm = mouse_meta.tissue.isin(tissues).to_numpy()
        hy, my = human_meta.loc[hm, "tissue"].to_numpy(), mouse_meta.loc[mm, "tissue"].to_numpy()
        for rep, (human_all, mouse_all) in reps.items():
            log(f"{cohort} · {rep}")
            human, mouse = human_all[hm], mouse_all[mm]
            stats, distributions = geometry_stats(rep, cohort, human, mouse, hy, my, tissues, rng)
            geometry_rows.extend(stats); distribution_rows.append(distributions)
            for direction, qx, qy, rx, ry in [
                ("human_to_mouse", human, hy, mouse, my), ("mouse_to_human", mouse, my, human, hy),
            ]:
                for centered in [False, True]:
                    readout = "centroid_cosine_centered" if centered else "centroid_cosine"
                    result, predicted, scores = centroid_readout(qx, qy, rx, ry, tissues, centered)
                    append_result(summaries, per_tissue_rows, confusions, cohort, rep, readout, direction,
                                  qy, predicted, scores, tissues, result)
                    if centered:
                        vals = cosine(qx - qx.mean(axis=0, keepdims=True), rx - qx.mean(axis=0, keepdims=True)).ravel()
                        geometry_rows.append({"cohort": cohort, "representation": rep, "space": f"centered_{direction}",
                                              "geometry_measure": "cosine", "group": "all_cross_species", "n": len(vals),
                                              "mean": vals.mean(), "sd": vals.std(ddof=1), "minimum": vals.min(),
                                              "maximum": vals.max(), "p01": np.quantile(vals,.01), "p99": np.quantile(vals,.99)})
                for k in [1, 3, 5]:
                    result, predicted, scores = knn_readout(qx, qy, rx, ry, tissues, k)
                    append_result(summaries, per_tissue_rows, confusions, cohort, rep, f"knn_cosine_k{k}", direction,
                                  qy, predicted, scores, tissues, result)
                for nonlinear, readout in [(False, "linear_softmax_probe"), (True, "shallow_mlp_probe")]:
                    result, predicted, scores = probe_readout(qx, qy, rx, ry, tissues, nonlinear, device)
                    append_result(summaries, per_tissue_rows, confusions, cohort, rep, readout, direction,
                                  ry, predicted, scores, tissues, result)
    summary = pd.DataFrame(summaries)
    per_tissue = pd.concat(per_tissue_rows, ignore_index=True)
    geometry = pd.DataFrame(geometry_rows)
    distributions = pd.concat(distribution_rows, ignore_index=True)
    summary.to_csv(RESULTS / "task1a_geometry_summary.csv", index=False)
    per_tissue.to_csv(RESULTS / "task1a_geometry_per_tissue.csv", index=False)
    geometry.to_csv(RESULTS / "task1a_geometry_statistics.csv", index=False)
    distributions.to_parquet(RESULTS / "task1a_cross_species_cosine_distributions.parquet", index=False)
    for (cohort, rep, readout, direction), matrix in confusions.items():
        tissues = TISSUES_11 if cohort == "complete_11_tissue" else TISSUES_5
        pd.DataFrame(matrix, index=tissues, columns=tissues).to_csv(
            RESULTS / f"geometry_confusion_{cohort}_{rep}_{readout}_{direction}.csv")
    for cohort, tissues in [("complete_11_tissue", TISSUES_11), ("replicated_5_tissue", TISSUES_5)]:
        for readout in summary.readout.unique(): plot_confusion_panels(confusions, cohort, readout, tissues)
    plot_geometry(distributions, geometry)
    provenance = {"center_mean": "source/training species mean only, applied unchanged to both species",
                  "knn": "cosine; fixed k=1,3,5; individual cross-species profiles; no within-mouse neighbors",
                  "linear_probe": "linear softmax, AdamW, 50 fixed epochs, balanced loss, no feature scaling",
                  "nonlinear_probe": "one 64-unit ReLU hidden layer, AdamW, 50 fixed epochs, balanced loss, no feature scaling",
                  "hyperparameter_selection": "fixed a priori; no target data/labels", "seed": 42,
                  "pairwise_cosine_sampling": "deterministic up to 20,000 non-self pairs"}
    (RESULTS / "task1a_geometry_provenance.json").write_text(json.dumps(provenance, indent=2))
    log("Geometry diagnostic complete\n" + summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__": main(parse_args())
