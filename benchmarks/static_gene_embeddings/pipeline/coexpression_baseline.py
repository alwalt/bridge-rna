#!/usr/bin/env python3
"""Matched Pearson/Spearman coexpression baseline for contextual gene neighborhoods."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

from contextual_depth_followup import GMTS, KS, LAYERS, bits_for, gmt
from contextual_gene_modules import VOCAB, WORK as MODULE_WORK
from matched_gtex_tcga import GTEX_X, TCGA_X, HERE

MATCH = HERE / "results/matched_gtex_tcga"
OUT = HERE / "results/coexpression_baseline"
SEED = 20260920


class Log:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")

    def __call__(self, message: str) -> None:
        text = f"[{pd.Timestamp.utcnow().isoformat()}] {message}"
        print(text, flush=True)
        with self.path.open("a") as handle:
            handle.write(text + "\n")


def standardized_rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = values - values.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(centered, axis=1)
    valid = np.isfinite(norm) & (norm > 1e-12)
    centered[valid] /= norm[valid, None]
    centered[~valid] = 0
    return centered.astype("float32"), valid


def exact_neighbors(values: np.ndarray, eligible: np.ndarray, device: torch.device,
                    k: int = 50, batch: int = 512) -> np.ndarray:
    """Cosine neighbors restricted to the shared eligible gene set."""
    global_indices = np.flatnonzero(eligible)
    matrix = torch.as_tensor(values[eligible], device=device, dtype=torch.float32)
    matrix = torch.nn.functional.normalize(matrix, dim=1)
    result = np.empty((len(global_indices), k), np.int32)
    for start in range(0, len(matrix), batch):
        scores = matrix[start:start + batch] @ matrix.T
        local_rows = torch.arange(len(scores), device=device)
        scores[local_rows, torch.arange(start, start + len(scores), device=device)] = -torch.inf
        result[start:start + len(scores)] = global_indices[
            torch.topk(scores, k).indices.cpu().numpy()
        ]
    return result


def hit_counts(bits: list[int], query_indices: np.ndarray, neighbors: np.ndarray,
               k: int) -> tuple[np.ndarray, np.ndarray]:
    hits = np.zeros(len(query_indices), np.int32)
    denominators = np.zeros(len(query_indices), np.int32)
    for row_index, gene_index in enumerate(query_indices):
        if not bits[gene_index]:
            continue
        annotated_neighbors = [j for j in neighbors[row_index, :k] if bits[j]]
        denominators[row_index] = len(annotated_neighbors)
        hits[row_index] = sum(bool(bits[gene_index] & bits[j]) for j in annotated_neighbors)
    return hits, denominators


def ratio_interval(hits: np.ndarray, den: np.ndarray, null_rates: np.ndarray,
                   rng: np.random.Generator, reps: int) -> tuple[float, float, float, float]:
    keep = den > 0
    hits, den = hits[keep], den[keep]
    observed = hits.sum() / den.sum()
    fold = observed / null_rates.mean()
    draws = np.empty(reps, float)
    for rep in range(reps):
        selected = rng.integers(0, len(hits), len(hits))
        sampled_rate = hits[selected].sum() / den[selected].sum()
        draws[rep] = sampled_rate / null_rates[rng.integers(0, len(null_rates))]
    low, high = np.quantile(draws, [0.025, 0.975])
    return observed, fold, low, high


def null_distribution(bits: list[int], eligible: np.ndarray, k: int,
                      reps: int, rng: np.random.Generator) -> np.ndarray:
    query = np.flatnonzero(eligible)
    annotated_candidates = np.array([i for i in query if bits[i]], np.int32)
    rates = np.empty(reps, float)
    for rep in range(reps):
        random_neighbors = rng.choice(annotated_candidates, size=(len(query), k), replace=True)
        hits, den = hit_counts(bits, query, random_neighbors, k)
        rates[rep] = hits.sum() / den.sum()
    return rates


def plot_summary(summary: pd.DataFrame) -> None:
    order = ["Pearson", "Spearman", "Bridge L0", "Bridge L1", "Bridge L6", "Bridge L12"]
    colors = {10: "#4C78A8", 25: "#F58518", 50: "#54A24B"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True, sharex=True)
    for row, dataset in enumerate(("GTEx", "TCGA")):
        for column, library in enumerate(("GO", "KEGG")):
            ax = axes[row, column]
            for k in KS:
                frame = summary[
                    summary.dataset.eq(dataset) & summary.library.eq(library) & summary.k.eq(k)
                ].set_index("method").loc[order]
                positions = np.arange(len(order))
                ax.errorbar(
                    positions,
                    frame.fold_over_null,
                    yerr=np.vstack([
                        frame.fold_over_null - frame.ci_low,
                        frame.ci_high - frame.fold_over_null,
                    ]),
                    marker="o",
                    capsize=3,
                    color=colors[k],
                    label=f"k={k}",
                )
            ax.axhline(1, color="black", linestyle=":", linewidth=1)
            ax.set_title(f"{dataset} — {library}")
            ax.set_ylabel("Shared-annotation rate / random null")
            ax.set_xticks(np.arange(len(order)), order, rotation=35, ha="right")
            ax.legend(fontsize=8)
    for extension in ("png", "pdf"):
        fig.savefig(OUT / f"matched_coexpression_vs_bridge.{extension}", dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--null-reps", type=int, default=100)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log = Log(OUT / "run.log")
    started = time.time()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    genes = pd.read_csv(VOCAB).sort_values("token_id").gene_symbol.astype(str).str.upper().tolist()
    manifest = pd.read_csv(MATCH / "cohort_manifest.csv")
    libraries = {name: gmt(path, genes) for name, path in GMTS.items()}
    bitsets = {name: bits_for(genes, terms) for name, terms in libraries.items()}
    expression_paths = {"GTEx": GTEX_X, "TCGA": TCGA_X}
    method_layers = {
        "Bridge L0": "L0_static",
        "Bridge L1": "L1_early",
        "Bridge L6": "L6_middle",
        "Bridge L12": "L12_final",
    }
    summary_rows, null_rows, eligibility_rows = [], [], []

    for dataset_index, dataset in enumerate(("GTEx", "TCGA")):
        subset = manifest[manifest.dataset.eq(dataset)]
        expression = np.asarray(
            np.load(expression_paths[dataset], mmap_mode="r")[subset.matrix_row.astype(int)],
            dtype=np.float32,
        ).T
        pearson, pearson_valid = standardized_rows(expression)
        ranked = rankdata(expression, axis=1, method="average").astype(np.float32)
        spearman, spearman_valid = standardized_rows(ranked)
        eligible = pearson_valid & spearman_valid
        eligibility_rows.append({
            "dataset": dataset,
            "canonical_genes": len(genes),
            "eligible_variable_genes": int(eligible.sum()),
            "excluded_constant_genes": int((~eligible).sum()),
        })
        log(f"{dataset}: loaded {expression.shape[1]} samples; eligible genes={eligible.sum()}/{len(genes)}")

        method_values = {"Pearson": pearson, "Spearman": spearman}
        for method, layer in method_layers.items():
            bridge = np.load(MODULE_WORK / f"{dataset}_{layer}_mean_tokens.float32.npy")
            method_values[method] = (bridge - bridge.mean(axis=0, keepdims=True)).astype(np.float32)

        rng_null = np.random.default_rng(SEED + dataset_index)
        nulls = {}
        for library, bits in bitsets.items():
            for k in KS:
                rates = null_distribution(bits, eligible, k, args.null_reps, rng_null)
                nulls[(library, k)] = rates
                for rep, rate in enumerate(rates):
                    null_rows.append({
                        "dataset": dataset, "library": library, "k": k,
                        "replicate": rep, "null_rate": rate,
                    })

        query_indices = np.flatnonzero(eligible)
        for method_index, (method, values) in enumerate(method_values.items()):
            neighbor_indices = exact_neighbors(values, eligible, device)
            log(f"{dataset}: neighbors complete for {method}")
            for library, bits in bitsets.items():
                for k in KS:
                    hits, den = hit_counts(bits, query_indices, neighbor_indices, k)
                    observed, fold, low, high = ratio_interval(
                        hits, den, nulls[(library, k)],
                        np.random.default_rng(SEED + dataset_index * 100 + method_index * 10 + k),
                        args.bootstrap_reps,
                    )
                    summary_rows.append({
                        "dataset": dataset,
                        "method": method,
                        "library": library,
                        "k": k,
                        "observed_rate": observed,
                        "null_mean": nulls[(library, k)].mean(),
                        "null_sd": nulls[(library, k)].std(ddof=1),
                        "fold_over_null": fold,
                        "ci_low": low,
                        "ci_high": high,
                        "eligible_genes": int(eligible.sum()),
                        "annotated_query_genes": int((den > 0).sum()),
                    })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "coexpression_vs_bridge_summary.csv", index=False)
    pd.DataFrame(null_rows).to_csv(OUT / "null_distributions.csv", index=False)
    pd.DataFrame(eligibility_rows).to_csv(OUT / "gene_eligibility.csv", index=False)
    plot_summary(summary)

    primary = summary[summary.k.eq(25)].pivot_table(
        index=["dataset", "library"], columns="method", values="fold_over_null"
    ).reset_index()
    primary["L12_minus_best_coexpression"] = primary["Bridge L12"] - primary[["Pearson", "Spearman"]].max(axis=1)
    primary["L12_over_best_coexpression"] = primary["Bridge L12"] / primary[["Pearson", "Spearman"]].max(axis=1)
    primary.to_csv(OUT / "primary_k25_comparison.csv", index=False)

    provenance = {
        "status": "complete",
        "cohorts": "Exact frozen matched manifest: 200 GTEx and 200 TCGA samples",
        "expression": "Natural log1p(TPM), canonical 15165-gene ordering",
        "coexpression": "Pearson or Spearman correlation across the 200 samples in each cohort",
        "bridge": "Mean-centered cosine on cohort-averaged 512-D contextual gene tokens cached from the same 200 samples",
        "layers": method_layers,
        "ks": KS,
        "annotation_libraries": {name: str(path) for name, path in GMTS.items()},
        "metric": "Aggregate fraction of annotation-eligible query-neighbor pairs sharing at least one term, divided by mean random-neighbor rate",
        "null": f"{args.null_reps} random annotation-eligible neighbor matrices, sampled with replacement; shared across methods within cohort/library/k",
        "confidence_intervals": f"95% percentile interval from {args.bootstrap_reps} gene bootstraps with a randomly drawn null replicate per bootstrap",
        "eligibility": "Canonical genes invariant across samples have undefined correlation and are excluded from queries/candidates for every method within that cohort; losses are recorded",
        "important_comparability_note": "The baseline uses one cohort-level network per method. Bridge vectors are averaged across the same cohort before neighborhood retrieval; this differs from the earlier per-sample contextual-occurrence endpoint but gives a matched unit against cohort coexpression.",
        "elapsed_seconds": time.time() - started,
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    log(f"Complete elapsed={(time.time() - started) / 60:.1f}m")


if __name__ == "__main__":
    main()
