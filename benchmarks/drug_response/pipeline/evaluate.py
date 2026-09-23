#!/usr/bin/env python3
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from common import CONFIG, RESULTS, WORK, write_json


def matrices() -> tuple[pd.DataFrame, list[str], np.ndarray]:
    observations = pd.read_parquet(RESULTS / "observation_manifest.parquet")
    samples = pd.read_csv(RESULTS / "cell_line_splits.csv")
    screens = sorted(observations.screen_id.unique())
    sample_index = {value: index for index, value in enumerate(samples.depmap_id)}
    screen_index = {value: index for index, value in enumerate(screens)}
    target = np.full((len(samples), len(screens)), np.nan, dtype=np.float32)
    for row in observations.itertuples(index=False):
        i, j = sample_index[row.ModelID], screen_index[row.screen_id]
        assert np.isnan(target[i, j]), (row.ModelID, row.screen_id)
        target[i, j] = row.IC50
    return samples, screens, target


def base_representation(name: str) -> np.ndarray:
    paths = {
        "raw_expression": WORK / "raw_expression.npy",
        "bridge_45.6m": WORK / "bridge_45.6m_embeddings.npy",
        "bulkformer_50m": WORK / "bulkformer_50m_embeddings.npy",
        "bulkformer_147m": WORK / "bulkformer_147m_embeddings.npy",
    }
    return np.load(paths[name], mmap_mode="r")


def fold_features(name: str, train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    if name.startswith("pca_") or name == "pca128_bridge":
        raw = base_representation("raw_expression")
        count = 128 if name in {"pca_128", "pca128_bridge"} else 512
        count = min(count, len(train) - 1, raw.shape[1])
        raw_scale = StandardScaler()
        train_raw = raw_scale.fit_transform(np.asarray(raw[train]))
        test_raw = raw_scale.transform(np.asarray(raw[test]))
        pca = PCA(n_components=count, svd_solver="randomized", random_state=CONFIG["split"]["seed"])
        xtrain, xtest = pca.fit_transform(train_raw), pca.transform(test_raw)
        metadata = {"pca_components": count,
                    "pca_cumulative_explained_variance": float(pca.explained_variance_ratio_.sum())}
        if name == "pca128_bridge":
            bridge = base_representation("bridge_45.6m")
            xtrain = np.concatenate([xtrain, np.asarray(bridge[train])], axis=1)
            xtest = np.concatenate([xtest, np.asarray(bridge[test])], axis=1)
        return xtrain, xtest, metadata
    matrix = base_representation(name)
    return np.asarray(matrix[train]), np.asarray(matrix[test]), {}


def main() -> None:
    samples, screens, targets = matrices()
    representations = CONFIG["representations"]
    predictions = {name: np.full_like(targets, np.nan) for name in representations}
    fold_rows, started = [], time.monotonic()
    for fold in sorted(samples.fold.unique()):
        test = np.flatnonzero(samples.fold.to_numpy() == fold)
        train = np.flatnonzero(samples.fold.to_numpy() != fold)
        for name in representations:
            xtrain, xtest, metadata = fold_features(name, train, test)
            scale = StandardScaler()
            xtrain = scale.fit_transform(xtrain)
            xtest = scale.transform(xtest)
            train_targets = targets[train].copy()
            counts = np.isfinite(train_targets).sum(axis=0)
            eligible = counts >= CONFIG["readout"]["minimum_train_observations"]
            observed = np.isfinite(train_targets[:, eligible])
            drug_means = np.nanmean(train_targets[:, eligible], axis=0)
            completed = np.where(observed, train_targets[:, eligible], drug_means)
            # EM-style completion permits one exact multi-output ridge solve per
            # iteration. Each output remains an independent linear drug head;
            # missing responses never contribute observed values to the fit.
            for _ in range(CONFIG["readout"]["missing_target_em_iterations"]):
                alpha = (CONFIG["readout"]["alpha_per_128_standardized_features"] *
                         xtrain.shape[1] / 128.0)
                model = Ridge(alpha=alpha, solver="cholesky")
                model.fit(xtrain, completed)
                fitted_train = model.predict(xtrain)
                completed[~observed] = fitted_train[~observed]
            estimate = model.predict(xtest)
            eligible_indices = np.flatnonzero(eligible)
            for local_j, j in enumerate(eligible_indices):
                keep_test = np.isfinite(targets[test, j])
                predictions[name][test[keep_test], j] = estimate[keep_test, local_j]
            fitted = int(eligible.sum())
            fold_rows.append({"fold": int(fold), "representation": name,
                              "train_cell_lines": len(train), "test_cell_lines": len(test),
                              "screens_fitted": fitted, **metadata})
            elapsed = time.monotonic() - started
            print(f"heartbeat fold={fold} representation={name} screens={fitted}/{len(screens)} "
                  f"elapsed={elapsed/60:.1f}m", flush=True)
    pd.DataFrame(fold_rows).to_csv(RESULTS / "fold_diagnostics.csv", index=False)
    metric_rows = []
    for name, prediction in predictions.items():
        np.save(WORK / f"{name}_oof_predictions.npy", prediction)
        for j, screen in enumerate(screens):
            keep = np.isfinite(targets[:, j]) & np.isfinite(prediction[:, j])
            if keep.sum() < CONFIG["readout"]["minimum_oof_observations"]:
                continue
            truth, estimate = targets[keep, j], prediction[keep, j]
            metric_rows.append({"representation": name, "screen_id": screen,
                "dataset_version": screen.split(":", 1)[0], "drug_id": int(screen.split(":", 1)[1]),
                "n": int(keep.sum()), "pcc": float(pearsonr(truth, estimate).statistic),
                "scc": float(spearmanr(truth, estimate).statistic),
                "rmse": float(mean_squared_error(truth, estimate) ** .5),
                "response_variance": float(np.var(truth, ddof=1))})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(RESULTS / "per_drug_metrics.csv", index=False)
    summary = metrics.groupby("representation", as_index=False).agg(
        mean_pcc=("pcc", "mean"), median_pcc=("pcc", "median"), mean_scc=("scc", "mean"),
        rmse=("rmse", "mean"), evaluable_drugs=("screen_id", "nunique"))
    summary.to_csv(RESULTS / "primary_results.csv", index=False)
    assert np.isfinite(summary[["mean_pcc", "median_pcc", "mean_scc", "rmse"]]).all().all()
    assert summary.rmse.max() < 10.0, "QA failure: implausibly large RMSE suggests unstable readout"
    comparisons = []
    for left, right in [("bridge_45.6m", "pca_512"), ("bridge_45.6m", "bulkformer_50m"),
                        ("bridge_45.6m", "bulkformer_147m"), ("pca128_bridge", "pca_128")]:
        a = metrics.loc[metrics.representation.eq(left), ["screen_id", "pcc"]].rename(columns={"pcc": "left_pcc"})
        b = metrics.loc[metrics.representation.eq(right), ["screen_id", "pcc"]].rename(columns={"pcc": "right_pcc"})
        paired = a.merge(b, on="screen_id"); delta = paired.left_pcc - paired.right_pcc
        rng = np.random.default_rng(CONFIG["split"]["seed"])
        boot = np.asarray([rng.choice(delta, len(delta), replace=True).mean() for _ in range(10000)])
        comparisons.append({"left": left, "right": right, "n_drugs": len(delta),
            "mean_delta_pcc": delta.mean(), "median_delta_pcc": delta.median(),
            "ci95_low": np.quantile(boot, .025), "ci95_high": np.quantile(boot, .975),
            "fraction_left_better": (delta > 0).mean()})
    pd.DataFrame(comparisons).to_csv(RESULTS / "paired_comparisons.csv", index=False)
    bridge = metrics.loc[metrics.representation.eq("bridge_45.6m")]
    controls = []
    for comparator in ["raw_expression", "pca_128", "pca_512", "bulkformer_50m", "bulkformer_147m"]:
        other = metrics.loc[metrics.representation.eq(comparator)]
        paired = bridge.merge(other, on="screen_id", suffixes=("_bridge", "_other"))
        delta = paired.pcc_bridge - paired.pcc_other
        controls.append({"comparison": f"bridge_45.6m-{comparator}",
            "spearman_delta_vs_response_variance": float(spearmanr(delta, paired.response_variance_bridge).statistic),
            "spearman_delta_vs_observation_count": float(spearmanr(delta, paired.n_bridge).statistic)})
    pd.DataFrame(controls).to_csv(RESULTS / "variance_controls.csv", index=False)
    write_json(RESULTS / "evaluation_provenance.json", {"config": CONFIG,
        "cell_lines": len(samples), "screens": len(screens), "observations": int(np.isfinite(targets).sum()),
        "metric_definition": "PCC/SCC/RMSE per dataset-version-specific drug screen over pooled held-out-cell-line OOF predictions; unweighted mean across screens"})
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
