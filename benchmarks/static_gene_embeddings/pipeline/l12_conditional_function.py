#!/usr/bin/env python3
"""Test L12 functional prediction conditional on coexpression and mean expression."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from contextual_depth_followup import GMTS, bits_for, gmt
from contextual_gene_modules import VOCAB, WORK as MODULE_WORK
from matched_gtex_tcga import GTEX_X, TCGA_X, HERE

MATCH = HERE / "results/matched_gtex_tcga"
OUT = HERE / "results/l12_conditional_function"
SEED = 20260921
BASE_COLUMNS = ["pearson", "spearman", "pair_mean_expression", "expression_difference"]
FULL_COLUMNS = BASE_COLUMNS + ["l12_cosine"]


def normalized_centered_rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = values - values.mean(axis=1, keepdims=True)
    lengths = np.linalg.norm(centered, axis=1)
    valid = np.isfinite(lengths) & (lengths > 1e-12)
    centered[valid] /= lengths[valid, None]
    centered[~valid] = 0
    return centered.astype(np.float32), valid


def unit_rows(values: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(values, axis=1, keepdims=True)
    return (values / np.maximum(lengths, 1e-12)).astype(np.float32)


def fit_cluster_robust(frame: pd.DataFrame) -> dict[str, float]:
    standardized = (frame[FULL_COLUMNS] - frame[FULL_COLUMNS].mean()) / frame[FULL_COLUMNS].std(ddof=0)
    design = sm.add_constant(standardized, has_constant="add")
    fit = sm.GLM(frame.outcome, design, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": frame.anchor_gene_index}
    )
    beta = float(fit.params["l12_cosine"])
    se = float(fit.bse["l12_cosine"])
    z = beta / se
    return {
        "l12_log_odds_per_sd": beta,
        "l12_cluster_robust_se": se,
        "l12_odds_ratio_per_sd": float(np.exp(beta)),
        "l12_or_ci_low": float(np.exp(beta - 1.96 * se)),
        "l12_or_ci_high": float(np.exp(beta + 1.96 * se)),
        "l12_z": z,
        "l12_p_value": float(2 * norm.sf(abs(z))),
        "l12_log10_p": float(np.log10(2) + norm.logsf(abs(z)) / np.log(10)),
    }


def grouped_predictions(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    base_prediction = np.empty(len(frame), float)
    full_prediction = np.empty(len(frame), float)
    splitter = GroupKFold(5)
    groups = frame.anchor_gene_index.to_numpy()
    for train, test in splitter.split(frame, frame.outcome, groups):
        for columns, destination in (
            (BASE_COLUMNS, base_prediction), (FULL_COLUMNS, full_prediction)
        ):
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1e4, max_iter=500, solver="lbfgs"),
            )
            model.fit(frame.iloc[train][columns], frame.outcome.iloc[train])
            destination[test] = model.predict_proba(frame.iloc[test][columns])[:, 1]
    return base_prediction, full_prediction


def bootstrap_metrics(frame: pd.DataFrame, base_prediction: np.ndarray,
                      full_prediction: np.ndarray, reps: int,
                      rng: np.random.Generator) -> dict[str, float]:
    outcome = frame.outcome.to_numpy()
    unique_groups = frame.anchor_gene_index.unique()
    group_rows = {
        group: np.flatnonzero(frame.anchor_gene_index.to_numpy() == group)
        for group in unique_groups
    }

    def metrics(indices: np.ndarray) -> tuple[float, float, float, float]:
        y = outcome[indices]
        base_auc = roc_auc_score(y, base_prediction[indices])
        full_auc = roc_auc_score(y, full_prediction[indices])
        base_ap = average_precision_score(y, base_prediction[indices])
        full_ap = average_precision_score(y, full_prediction[indices])
        return base_auc, full_auc, base_ap, full_ap

    observed = metrics(np.arange(len(frame)))
    draws = np.empty((reps, 4), float)
    for rep in range(reps):
        selected_groups = rng.choice(unique_groups, len(unique_groups), replace=True)
        indices = np.concatenate([group_rows[group] for group in selected_groups])
        draws[rep] = metrics(indices)
    auc_delta = draws[:, 1] - draws[:, 0]
    ap_delta = draws[:, 3] - draws[:, 2]
    result = {
        "base_auroc": observed[0], "full_auroc": observed[1],
        "delta_auroc": observed[1] - observed[0],
        "base_auprc": observed[2], "full_auprc": observed[3],
        "delta_auprc": observed[3] - observed[2],
    }
    for name, values in (
        ("base_auroc", draws[:, 0]), ("full_auroc", draws[:, 1]),
        ("delta_auroc", auc_delta), ("base_auprc", draws[:, 2]),
        ("full_auprc", draws[:, 3]), ("delta_auprc", ap_delta),
    ):
        result[f"{name}_ci_low"], result[f"{name}_ci_high"] = np.quantile(values, [0.025, 0.975])
    result["delta_auroc_bootstrap_p"] = (1 + np.sum(auc_delta <= 0)) / (reps + 1)
    result["delta_auprc_bootstrap_p"] = (1 + np.sum(ap_delta <= 0)) / (reps + 1)
    return result


def plot_results(results: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    labels, positions = [], []
    for dataset in ("GTEx", "TCGA"):
        for library in ("GO", "KEGG"):
            labels.append(f"{dataset}\n{library}")
            positions.append(len(positions))
    ordered = results.set_index(["dataset", "library"]).loc[
        [(d, l) for d in ("GTEx", "TCGA") for l in ("GO", "KEGG")]
    ]
    axes[0].errorbar(
        positions, ordered.l12_odds_ratio_per_sd,
        yerr=np.vstack([
            ordered.l12_odds_ratio_per_sd - ordered.l12_or_ci_low,
            ordered.l12_or_ci_high - ordered.l12_odds_ratio_per_sd,
        ]), fmt="o", capsize=4, color="#4C78A8",
    )
    axes[0].axhline(1, color="black", linestyle=":", linewidth=1)
    axes[0].set(title="Independent L12 association", ylabel="Odds ratio per 1 SD L12 cosine")
    axes[1].errorbar(
        positions, ordered.delta_auroc,
        yerr=np.vstack([
            ordered.delta_auroc - ordered.delta_auroc_ci_low,
            ordered.delta_auroc_ci_high - ordered.delta_auroc,
        ]), fmt="o", capsize=4, color="#E45756",
    )
    axes[1].axhline(0, color="black", linestyle=":", linewidth=1)
    axes[1].set(title="Held-out incremental prediction", ylabel="AUROC(full) − AUROC(base)")
    for axis in axes:
        axis.set_xticks(positions, labels)
    for extension in ("png", "pdf"):
        fig.savefig(OUT / f"l12_conditional_function.{extension}", dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchors", type=int, default=3000)
    parser.add_argument("--partners-per-anchor", type=int, default=100)
    parser.add_argument("--bootstrap-reps", type=int, default=500)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    genes = pd.read_csv(VOCAB).sort_values("token_id").gene_symbol.astype(str).str.upper().tolist()
    manifest = pd.read_csv(MATCH / "cohort_manifest.csv")
    expression_paths = {"GTEx": GTEX_X, "TCGA": TCGA_X}
    expression, pearson, spearman, valid = {}, {}, {}, {}

    for dataset in ("GTEx", "TCGA"):
        rows = manifest.loc[manifest.dataset.eq(dataset), "matrix_row"].astype(int)
        expression[dataset] = np.asarray(np.load(expression_paths[dataset], mmap_mode="r")[rows], dtype=np.float32).T
        pearson[dataset], pearson_valid = normalized_centered_rows(expression[dataset])
        ranks = rankdata(expression[dataset], axis=1, method="average").astype(np.float32)
        spearman[dataset], spearman_valid = normalized_centered_rows(ranks)
        valid[dataset] = pearson_valid & spearman_valid

    shared_valid = valid["GTEx"] & valid["TCGA"]
    eligible = np.flatnonzero(shared_valid)
    rng = np.random.default_rng(SEED)
    anchors = np.sort(rng.choice(eligible, args.anchors, replace=False))
    anchor_indices = np.repeat(anchors, args.partners_per_anchor)
    partner_indices = rng.choice(eligible, len(anchor_indices), replace=True)
    same = partner_indices == anchor_indices
    while same.any():
        partner_indices[same] = rng.choice(eligible, same.sum(), replace=True)
        same = partner_indices == anchor_indices

    libraries = {name: gmt(path, genes) for name, path in GMTS.items()}
    bitsets = {name: bits_for(genes, terms) for name, terms in libraries.items()}
    result_rows = []
    prediction_rows = []

    for dataset_index, dataset in enumerate(("GTEx", "TCGA")):
        l12 = np.load(MODULE_WORK / f"{dataset}_L12_final_mean_tokens.float32.npy")
        l12 = unit_rows(l12 - l12.mean(axis=0, keepdims=True))
        mean_expression = expression[dataset].mean(axis=1)
        pair_frame = pd.DataFrame({
            "anchor_gene_index": anchor_indices,
            "partner_gene_index": partner_indices,
            "pearson": np.sum(pearson[dataset][anchor_indices] * pearson[dataset][partner_indices], axis=1),
            "spearman": np.sum(spearman[dataset][anchor_indices] * spearman[dataset][partner_indices], axis=1),
            "l12_cosine": np.sum(l12[anchor_indices] * l12[partner_indices], axis=1),
            "pair_mean_expression": (mean_expression[anchor_indices] + mean_expression[partner_indices]) / 2,
            "expression_difference": np.abs(mean_expression[anchor_indices] - mean_expression[partner_indices]),
        })

        for library_index, (library, bits) in enumerate(bitsets.items()):
            annotated = np.array([bool(value) for value in bits])
            keep = annotated[anchor_indices] & annotated[partner_indices]
            frame = pair_frame.loc[keep].copy().reset_index(drop=True)
            ai = frame.anchor_gene_index.to_numpy()
            bi = frame.partner_gene_index.to_numpy()
            frame["outcome"] = np.fromiter(
                (bool(bits[a] & bits[b]) for a, b in zip(ai, bi)), bool, len(frame)
            ).astype(int)
            robust = fit_cluster_robust(frame)
            base_prediction, full_prediction = grouped_predictions(frame)
            predictive = bootstrap_metrics(
                frame, base_prediction, full_prediction, args.bootstrap_reps,
                np.random.default_rng(SEED + dataset_index * 10 + library_index),
            )
            result_rows.append({
                "dataset": dataset, "library": library,
                "pairs": len(frame), "anchor_genes": frame.anchor_gene_index.nunique(),
                "positive_pairs": int(frame.outcome.sum()), "positive_fraction": frame.outcome.mean(),
                **robust, **predictive,
            })
            prediction_rows.append(pd.DataFrame({
                "dataset": dataset, "library": library,
                "anchor_gene_index": frame.anchor_gene_index,
                "partner_gene_index": frame.partner_gene_index,
                "outcome": frame.outcome,
                "base_oof_prediction": base_prediction,
                "full_oof_prediction": full_prediction,
            }))
            print(f"{dataset} {library}: pairs={len(frame):,}", flush=True)

    results = pd.DataFrame(result_rows)
    results.to_csv(OUT / "l12_conditional_results.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_parquet(
        OUT / "out_of_fold_predictions.parquet", index=False
    )
    plot_results(results)
    provenance = {
        "status": "complete",
        "cohorts": "Frozen balanced 200-sample GTEx and 200-sample TCGA cohorts",
        "gene_universe": len(genes),
        "shared_variable_gene_universe": int(shared_valid.sum()),
        "pair_sampling": {"anchors": args.anchors, "partners_per_anchor": args.partners_per_anchor, "seed": SEED, "same_pairs_across_cohorts": True},
        "outcome": "Both genes annotated and sharing at least one GO Biological Process term or KEGG pathway",
        "base_model": BASE_COLUMNS,
        "full_model": FULL_COLUMNS,
        "expression_controls": "Pair-average cohort mean log1p(TPM) and absolute difference in cohort mean log1p(TPM)",
        "inference": "Standardized multivariable logistic regression; sandwich SE clustered by anchor gene",
        "prediction": "Five-fold anchor-gene GroupKFold; unregularized-approximation logistic regression; base versus base plus L12",
        "confidence_intervals": f"95% percentile intervals from {args.bootstrap_reps} anchor-gene bootstrap replicates",
        "limitations": [
            "Shared annotation is not an independent experimental phenotype",
            "The conditional model controls specified linear terms and cannot exclude every nonlinear expression-derived explanation",
            "Cohort-mean L12 vectors test shared cohort-level functional geometry, not sample-specific token occurrences",
        ],
        "elapsed_seconds": time.time() - started,
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
