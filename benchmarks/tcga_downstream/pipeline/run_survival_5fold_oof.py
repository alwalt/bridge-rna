#!/usr/bin/env python3
"""True five-fold OOF Cox evaluation for the complete TCGA survival cohort."""

from __future__ import annotations

import argparse
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lifelines import CoxPHFitter
from lifelines.utils.concordance import _concordance_summary_statistics
from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split

from common import CONFIG, RESULTS, WORK
from run_benchmark import (SurvivalHead, best_state, build_cohorts, cox_loss,
                           reduced, representation, standardize_native)

FOLDS = 5
OOF_SEED = 20260922
BOOTSTRAPS = 1000
BOOTSTRAP_SEED = 20260923
PERMUTATIONS = 100_000
FIGURES = RESULTS / "figures"
LABELS = ("Bridge 45.6M", "PCA-128", "Full raw expression")
SOURCES = {"Bridge 45.6M": "ours_45.6m", "PCA-128": "raw_expression",
           "Full raw expression": "full_expression_25150"}
COLORS = {"Bridge 45.6M": "#1f5a94", "PCA-128": "#d17a22",
          "Full raw expression": "#4f8b3a"}


def say(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def load_cohort() -> pd.DataFrame:
    manifest = RESULTS / "cohort_manifest.parquet"
    samples = pd.read_parquet(manifest) if manifest.is_file() else build_cohorts()
    cohort = samples.loc[samples.survival_usable].copy().reset_index(drop=True)
    required = ["patient_id", "cohort", "time_days", "event", "matrix_row"]
    assert not cohort[required].isna().any().any()
    assert cohort.patient_id.is_unique
    assert cohort.event.isin([0, 1]).all() and cohort.time_days.gt(0).all()
    rows = cohort.matrix_row.to_numpy(int)
    arrays = {
        "Bridge 45.6M": representation("ours_45.6m"),
        "PCA-128": representation("raw_expression"),
        "Full raw expression": np.load(WORK / "tcga_full_log1p_cpm.npy", mmap_mode="r"),
    }
    for name, array in arrays.items():
        assert rows.min() >= 0 and rows.max() < len(array)
        assert np.isfinite(np.asarray(array[rows])).all(), name
    return cohort


def make_folds(cohort: pd.DataFrame) -> pd.DataFrame:
    output = RESULTS / "survival_5fold_oof_fold_manifest.csv"
    if output.is_file():
        saved = pd.read_csv(output)
        assert set(saved.patient_id) == set(cohort.patient_id)
        assert saved.patient_id.is_unique and saved.fold.between(0, FOLDS - 1).all()
        return saved
    rng = np.random.default_rng(OOF_SEED)
    assigned = np.full(len(cohort), -1, dtype=int)
    # Deterministic sparse-stratum fallback: independently shuffle each
    # cancer×event stratum and round-robin it over the currently least-populated
    # folds for that cancer. This distributes even strata with fewer than 5 rows.
    for cancer in sorted(cohort.cohort.unique()):
        cancer_counts = np.zeros(FOLDS, dtype=int)
        subset = cohort.index[cohort.cohort.eq(cancer)]
        for event in (1, 0):
            indices = subset[cohort.loc[subset, "event"].eq(event).to_numpy()].to_numpy()
            rng.shuffle(indices)
            for index in indices:
                minimum = cancer_counts.min()
                choices = np.flatnonzero(cancer_counts == minimum)
                fold = int(choices[rng.integers(len(choices))])
                assigned[index] = fold
                cancer_counts[fold] += 1
    assert (assigned >= 0).all()
    result = cohort[["patient_id", "cohort", "time_days", "event", "matrix_row"]].copy()
    result.insert(1, "fold", assigned)
    assert result.patient_id.is_unique and len(result) == len(cohort)
    result.to_csv(output, index=False)
    totals = result.groupby("fold", as_index=False).agg(
        n_patients=("patient_id", "size"), n_events=("event", "sum"))
    totals["event_rate"] = totals.n_events / totals.n_patients
    cancer = result.groupby(["fold", "cohort"], as_index=False).agg(
        n_patients=("patient_id", "size"), n_events=("event", "sum"))
    cancer["event_rate"] = cancer.n_events / cancer.n_patients
    pd.concat([totals.assign(cancer="ALL"), cancer.rename(columns={"cohort": "cancer"})],
              ignore_index=True).to_csv(RESULTS / "survival_5fold_oof_fold_qc.csv", index=False)
    return result


def train_mlp(matrix: np.ndarray, cohort: pd.DataFrame, train_all: np.ndarray,
              test: np.ndarray, fold: int, device: torch.device) -> tuple[np.ndarray, int]:
    strata = cohort.cohort.astype(str)
    train, validation = train_test_split(
        train_all, test_size=.125, random_state=OOF_SEED + 1000 + fold,
        stratify=strata.iloc[train_all])
    rows = cohort.matrix_row.to_numpy(int)
    xtrain, xvalidation, xtest = standardize_native(
        matrix, rows[train], rows[validation], rows[test])
    torch.manual_seed(OOF_SEED + fold)
    if device.type == "cuda": torch.cuda.manual_seed_all(OOF_SEED + fold)
    model = SurvivalHead(xtrain.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(CONFIG["head_learning_rate"]),
                                  weight_decay=float(CONFIG["head_weight_decay"]))
    values = [torch.as_tensor(value, dtype=torch.float32, device=device) for value in
              (xtrain, cohort.time_days.to_numpy(np.float32)[train],
               cohort.event.to_numpy(np.float32)[train], xvalidation,
               cohort.time_days.to_numpy(np.float32)[validation],
               cohort.event.to_numpy(np.float32)[validation])]
    train_x, train_t, train_e, val_x, val_t, val_e = values
    optimum, state, stale = np.inf, None, 0
    for epoch in range(int(CONFIG["head_max_epochs"])):
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss = cox_loss(model(train_x), train_t, train_e)
        loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): value = float(cox_loss(model(val_x), val_t, val_e))
        if value < optimum - 1e-5:
            optimum, state, stale = value, best_state(model), 0
        else:
            stale += 1
        if stale >= int(CONFIG["head_patience"]): break
    assert state is not None
    model.load_state_dict(state); model.eval()
    with torch.no_grad():
        risk = model(torch.as_tensor(xtest, dtype=torch.float32, device=device)).cpu().numpy()
    del model, optimizer, values, train_x, val_x, xtrain, xvalidation, xtest
    if device.type == "cuda": torch.cuda.empty_cache()
    return risk, epoch + 1


def train_pca(cohort: pd.DataFrame, train: np.ndarray, test: np.ndarray, fold: int) -> np.ndarray:
    matrix = representation("raw_expression")
    rows = cohort.matrix_row.to_numpy(int)
    xtrain, xtest = reduced(np.asarray(matrix[rows[train]]), np.asarray(matrix[rows[test]]),
                            OOF_SEED + fold)
    columns = [f"x{i}" for i in range(xtrain.shape[1])]
    fit = pd.DataFrame(xtrain, columns=columns)
    fit["time_days"] = cohort.time_days.to_numpy(float)[train]
    fit["event"] = cohort.event.to_numpy(int)[train]
    model = CoxPHFitter(penalizer=float(CONFIG["survival_penalizer"]))
    model.fit(fit, duration_col="time_days", event_col="event", show_progress=False)
    return model.predict_partial_hazard(pd.DataFrame(xtest, columns=columns)).to_numpy()


def generate_predictions(cohort: pd.DataFrame, folds: pd.DataFrame,
                         device: torch.device) -> pd.DataFrame:
    output = RESULTS / "survival_5fold_oof_predictions.csv"
    if output.is_file():
        say(f"reusing {output.name}")
        return pd.read_csv(output)
    fold_lookup = folds.set_index("patient_id").fold
    cohort = cohort.copy()
    cohort["fold"] = cohort.patient_id.map(fold_lookup).astype(int)
    bridge = representation("ours_45.6m")
    raw = np.load(WORK / "tcga_full_log1p_cpm.npy", mmap_mode="r")
    pieces = []
    for fold in range(FOLDS):
        test = np.flatnonzero(cohort.fold.to_numpy() == fold)
        train = np.flatnonzero(cohort.fold.to_numpy() != fold)
        for label in LABELS:
            part = WORK / f"survival_5fold_oof_{label.lower().replace(' ', '_')}_fold{fold}.csv"
            if part.is_file():
                frame = pd.read_csv(part)
                say(f"reusing {part.name}")
            else:
                started = time.monotonic()
                if label == "Bridge 45.6M": risk, epochs = train_mlp(bridge, cohort, train, test, fold, device)
                elif label == "PCA-128": risk, epochs = train_pca(cohort, train, test, fold), None
                else: risk, epochs = train_mlp(raw, cohort, train, test, fold, device)
                held = cohort.iloc[test]
                frame = pd.DataFrame({"patient_id": held.patient_id, "cancer": held.cohort,
                                      "survival_time": held.time_days, "event": held.event.astype(int),
                                      "fold": fold, "representation": label, "risk_score": risk})
                frame.to_csv(part, index=False)
                say(f"completed fold={fold} representation={label} n={len(test)} "
                    f"epochs={epochs} elapsed={(time.monotonic()-started)/60:.1f}m")
            pieces.append(frame)
    result = pd.concat(pieces, ignore_index=True)
    result.to_csv(output, index=False)
    return result


def qa_predictions(predictions: pd.DataFrame, cohort: pd.DataFrame, folds: pd.DataFrame) -> None:
    expected = set(cohort.patient_id)
    for label in LABELS:
        frame = predictions[predictions.representation == label]
        assert len(frame) == len(cohort) and frame.patient_id.is_unique
        assert set(frame.patient_id) == expected
    keys = ["patient_id", "cancer", "survival_time", "event", "fold"]
    reference = predictions[predictions.representation == LABELS[0]][keys].sort_values("patient_id").reset_index(drop=True)
    for label in LABELS[1:]:
        other = predictions[predictions.representation == label][keys].sort_values("patient_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(reference, other, check_dtype=False)
    assert folds.patient_id.is_unique and set(folds.patient_id) == expected
    assert reference.groupby("patient_id").fold.nunique().eq(1).all()


def concordance_parts(frame: pd.DataFrame) -> tuple[float, float, int]:
    if len(frame) < 2 or frame.event.sum() == 0: return 0.0, 0.0, 0
    correct, tied, pairs = _concordance_summary_statistics(
        frame.survival_time.to_numpy(float), -frame.risk_score.to_numpy(float),
        frame.event.to_numpy(bool))
    return float(correct), float(tied), int(pairs)


def fold_aware_cindex(frame: pd.DataFrame) -> tuple[float, int]:
    correct = tied = 0.0; pairs = 0
    for _, group in frame.groupby("fold"):
        c, t, p = concordance_parts(group); correct += c; tied += t; pairs += p
    return ((correct + .5 * tied) / pairs if pairs else np.nan), pairs


def stability(events: int, valid: int, width: float) -> str:
    flags = []
    if valid < 900 or not np.isfinite(width): flags.append("unreliable_bootstrap")
    if events < 10: flags.append("very_low_event_support")
    elif events < 20: flags.append("low_event_support")
    if np.isfinite(width) and width > .25: flags.append("wide_ci")
    return ";".join(flags) if flags else "stable"


def metrics_and_bootstrap(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_path = RESULTS / "survival_5fold_oof_metrics.csv"
    cancer_path = RESULTS / "survival_5fold_oof_per_cancer.csv"
    if metrics_path.is_file() and cancer_path.is_file():
        say("reusing OOF metrics and bootstrap summary")
        return pd.read_csv(metrics_path), pd.read_csv(cancer_path)
    overall = []
    for label in LABELS:
        value, pairs = fold_aware_cindex(predictions[predictions.representation == label])
        overall.append({"representation": label, "oof_survival_c_index": value,
                        "comparable_pairs": pairs, "aggregation": "fold-aware pair-weighted"})
    pd.DataFrame(overall).to_csv(metrics_path, index=False)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    cancers = sorted(predictions.cancer.unique())
    for number, cancer in enumerate(cancers, 1):
        data = predictions[predictions.cancer == cancer]
        support = data[data.representation == LABELS[0]].sort_values("patient_id")
        patients = support.patient_id.to_numpy(str)
        draws = rng.integers(0, len(patients), size=(BOOTSTRAPS, len(patients)))
        counts = np.apply_along_axis(np.bincount, 1, draws, minlength=len(patients))
        for label in LABELS:
            frame = data[data.representation == label].set_index("patient_id").loc[patients].reset_index()
            estimate, pairs = fold_aware_cindex(frame)
            boot = []
            for multiplicity in counts:
                sampled = frame.loc[frame.index.repeat(multiplicity)]
                value, _ = fold_aware_cindex(sampled)
                if np.isfinite(value): boot.append(value)
            low, high = np.quantile(boot, [.025, .975]) if boot else (np.nan, np.nan)
            events = int(support.event.sum()); width = float(high - low) if np.isfinite(low) else np.nan
            rows.append({"representation": label, "cancer": cancer, "n_patients": len(support),
                         "n_events": events, "n_censored": len(support) - events,
                         "censoring_fraction": 1 - events / len(support),
                         "comparable_pairs": pairs, "c_index": estimate,
                         "ci_low": low, "ci_high": high, "ci_width": width,
                         "bootstrap_valid": len(boot),
                         "stability_flag": stability(events, len(boot), width)})
        say(f"bootstrap {number}/{len(cancers)} cancer={cancer}")
    per_cancer = pd.DataFrame(rows)
    per_cancer.to_csv(cancer_path, index=False)
    return pd.DataFrame(overall), per_cancer


def comparison_table(per_cancer: pd.DataFrame) -> pd.DataFrame:
    index = ["cancer", "n_patients", "n_events"]
    fields = ["c_index", "ci_low", "ci_high"]
    wide = per_cancer.pivot(index=index, columns="representation", values=fields)
    wide.columns = [f"{label}_{field}" for field, label in wide.columns]
    wide = wide.reset_index().rename(columns={"n_patients": "N", "n_events": "events"})
    rename = {}
    for label, slug in [("Bridge 45.6M", "Bridge"), ("PCA-128", "PCA"),
                        ("Full raw expression", "raw")]:
        for field, suffix in [("c_index", "C-index"), ("ci_low", "CI low"), ("ci_high", "CI high")]:
            rename[f"{label}_{field}"] = f"{slug} {suffix}"
    wide = wide.rename(columns=rename)
    wide["Bridge − PCA"] = wide["Bridge C-index"] - wide["PCA C-index"]
    wide["Bridge − raw"] = wide["Bridge C-index"] - wide["raw C-index"]
    wide.to_csv(RESULTS / "survival_5fold_oof_representation_comparison.csv", index=False)
    return wide


def limits(per_cancer: pd.DataFrame) -> tuple[float, float]:
    values = pd.concat([per_cancer.ci_low, per_cancer.ci_high]).dropna()
    return max(0., float(values.min()) - .04), min(1., float(values.max()) + .04)


def decorate(ax, order: list[str], support: pd.DataFrame, xlim, title: str, annotations=True) -> None:
    y = np.arange(len(order)); ax.axvline(.5, color="#555", ls="--", lw=1)
    ax.set(yticks=y, yticklabels=order, xlim=xlim, xlabel="Held-out C-index",
           ylabel="TCGA cancer type", title=title)
    ax.invert_yaxis(); ax.grid(axis="x", color="#ddd", lw=.7)
    if annotations:
        transform = ax.get_yaxis_transform(); x = 1.02
        ax.text(x, -1.05, "Events / N", transform=transform, clip_on=False,
                ha="left", va="bottom", weight="bold", fontsize=9)
        for yy, cancer in enumerate(order):
            row = support.loc[cancer]
            ax.text(x, yy, f"{int(row.n_events)} / {int(row.n_patients)}", transform=transform,
                    clip_on=False, ha="left", va="center", fontsize=8)


def individual_plot(per_cancer: pd.DataFrame, label: str, order: list[str], xlim,
                    stem: str, title: str, pdf=True) -> None:
    data = per_cancer[per_cancer.representation == label].set_index("cancer").reindex(order)
    fig, ax = plt.subplots(figsize=(8.3, max(8.5, .31 * len(order) + 1.8)))
    y = np.arange(len(order)); valid = data.c_index.notna() & data.ci_low.notna()
    ax.errorbar(data.loc[valid, "c_index"], y[valid],
                xerr=np.vstack([data.loc[valid, "c_index"] - data.loc[valid, "ci_low"],
                                data.loc[valid, "ci_high"] - data.loc[valid, "c_index"]]),
                fmt="o", color=COLORS[label], capsize=2.5, ms=5, lw=1.3)
    decorate(ax, order, data, xlim, title)
    fig.tight_layout(rect=(0, 0, .88, 1)); fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")
    if pdf: fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def combined_plot(per_cancer: pd.DataFrame, order: list[str], xlim) -> None:
    support = per_cancer[per_cancer.representation == LABELS[0]].set_index("cancer").reindex(order)
    fig, ax = plt.subplots(figsize=(8.5, max(8.5, .34 * len(order) + 1.8)))
    y = np.arange(len(order)); offsets = [-.18, 0, .18]
    for label, offset in zip(LABELS, offsets):
        data = per_cancer[per_cancer.representation == label].set_index("cancer").reindex(order)
        valid = data.c_index.notna() & data.ci_low.notna(); yy = y + offset
        ax.errorbar(data.loc[valid, "c_index"], yy[valid],
                    xerr=np.vstack([data.loc[valid, "c_index"] - data.loc[valid, "ci_low"],
                                    data.loc[valid, "ci_high"] - data.loc[valid, "c_index"]]),
                    fmt="o", color=COLORS[label], label=label, capsize=2, ms=4, lw=1.05)
    decorate(ax, order, support, xlim, "Within-cancer survival discrimination by transcriptomic representation")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.035), ncol=3)
    fig.tight_layout(rect=(0, .035, .88, 1))
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"bridge_pca_raw_5fold_oof_survival_forest.{ext}",
                    dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def three_panel(per_cancer: pd.DataFrame, order: list[str], xlim) -> None:
    support = per_cancer[per_cancer.representation == LABELS[0]].set_index("cancer").reindex(order)
    fig, axes = plt.subplots(1, 3, figsize=(18, max(9, .32 * len(order) + 1.7)), sharey=True)
    for panel, (ax, label) in enumerate(zip(axes, LABELS)):
        data = per_cancer[per_cancer.representation == label].set_index("cancer").reindex(order)
        y = np.arange(len(order)); valid = data.c_index.notna() & data.ci_low.notna()
        ax.errorbar(data.loc[valid, "c_index"], y[valid],
                    xerr=np.vstack([data.loc[valid, "c_index"] - data.loc[valid, "ci_low"],
                                    data.loc[valid, "ci_high"] - data.loc[valid, "c_index"]]),
                    fmt="o", color=COLORS[label], capsize=2, ms=4, lw=1.05)
        decorate(ax, order, support, xlim, f"{chr(65 + panel)}. {label}", annotations=panel == 2)
        if panel: ax.set_ylabel("")
    fig.suptitle("Complete 5-fold OOF within-cancer survival discrimination", y=.995, fontsize=15)
    fig.tight_layout(rect=(0, 0, .96, .98))
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"bridge_pca_raw_5fold_oof_survival_three_panel.{ext}",
                    dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def delta_plot(table: pd.DataFrame, order: list[str]) -> None:
    data = table.set_index("cancer").reindex(order); y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7.7, max(8.5, .31 * len(order) + 1.5)))
    ax.scatter(data["Bridge − PCA"], y - .1, label="Bridge − PCA-128", s=24)
    ax.scatter(data["Bridge − raw"], y + .1, label="Bridge − full raw expression", s=24)
    ax.axvline(0, color="#555", ls="--", lw=1)
    ax.set(yticks=y, yticklabels=order, xlabel="Paired difference in held-out C-index",
           ylabel="TCGA cancer type", title="Bridge versus conventional representations (5-fold OOF)")
    ax.invert_yaxis(); ax.grid(axis="x", color="#ddd", lw=.7); ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"bridge_vs_conventional_5fold_oof_delta_cindex.{ext}",
                    dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def plotting_frame(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in table.itertuples(index=False):
        values = row._asdict()
        # itertuples sanitizes punctuation, so positional access is clearer here.
        source = table.loc[table.cancer.eq(values["cancer"])].iloc[0]
        for label, prefix in [(LABELS[0], "Bridge"), (LABELS[1], "PCA"),
                              (LABELS[2], "raw")]:
            rows.append({"cancer": source.cancer, "representation": label,
                         "n_patients": source.N, "n_events": source.events,
                         "c_index": source[f"{prefix} C-index"],
                         "ci_low": source[f"{prefix} CI low"],
                         "ci_high": source[f"{prefix} CI high"]})
    return pd.DataFrame(rows)


def plots(table: pd.DataFrame) -> None:
    # The saved representation-comparison table is the sole plotting source.
    per_cancer = plotting_frame(table)
    FIGURES.mkdir(exist_ok=True); order = sorted(per_cancer.cancer.unique()); xlim = limits(per_cancer)
    specifications = [
        (LABELS[0], "bridge_5fold_oof_per_cancer_survival_forest", "Bridge survival discrimination across TCGA cancer types"),
        (LABELS[1], "pca128_5fold_oof_per_cancer_survival_forest", "PCA survival discrimination across TCGA cancer types"),
        (LABELS[2], "raw_expression_5fold_oof_per_cancer_survival_forest", "Full-expression survival discrimination across TCGA cancer types"),
    ]
    for label, stem, title in specifications: individual_plot(per_cancer, label, order, xlim, stem, title)
    bridge_order = per_cancer[per_cancer.representation == LABELS[0]].sort_values("c_index", ascending=False).cancer.tolist()
    sorted_stems = ["bridge_5fold_oof_per_cancer_survival_forest_sorted",
                    "pca128_5fold_oof_per_cancer_survival_forest_bridge_sorted",
                    "raw_expression_5fold_oof_per_cancer_survival_forest_bridge_sorted"]
    for (label, _, title), stem in zip(specifications, sorted_stems):
        individual_plot(per_cancer, label, bridge_order, xlim, stem, title, pdf=False)
    combined_plot(per_cancer, order, xlim); three_panel(per_cancer, order, xlim); delta_plot(table, order)


def summaries(per_cancer: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    bridge = per_cancer[per_cancer.representation == LABELS[0]].set_index("cancer")
    rows = [
        ("median_bridge_c_index", bridge.c_index.median()),
        ("patient_weighted_mean_bridge_c_index", np.average(bridge.c_index, weights=bridge.n_patients)),
        ("median_pca128_c_index", table["PCA C-index"].median()),
        ("median_full_raw_c_index", table["raw C-index"].median()),
        ("bridge_cancers_gt_0.5", (bridge.c_index > .5).sum()),
        ("bridge_ci_above_0.5", (bridge.ci_low > .5).sum()),
    ]
    for label, slug in [(LABELS[1], "pca128"), (LABELS[2], "full_raw")]:
        frame = per_cancer[per_cancer.representation == label]
        rows.append((f"{slug}_ci_above_0.5", (frame.ci_low > .5).sum()))
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    for column, slug in [("Bridge − PCA", "pca128"), ("Bridge − raw", "full_raw")]:
        difference = table[column].dropna().to_numpy(); observed = difference.mean()
        null = (rng.choice([-1, 1], size=(PERMUTATIONS, len(difference))) * difference).mean(axis=1)
        p = (1 + (np.abs(null) >= abs(observed)).sum()) / (PERMUTATIONS + 1)
        rows.extend([(f"cancers_bridge_gt_{slug}", (difference > 0).sum()),
                     (f"median_bridge_minus_{slug}", np.median(difference)),
                     (f"paired_signflip_mean_p_{slug}", p),
                     (f"paired_wilcoxon_p_{slug}", wilcoxon(difference).pvalue)])
    result = pd.DataFrame(rows, columns=["summary", "value"])
    result.to_csv(RESULTS / "survival_5fold_oof_statistical_summary.csv", index=False)
    return result


def repeated_comparison(per_cancer: pd.DataFrame) -> pd.DataFrame:
    repeated = pd.read_csv(RESULTS / "per_cancer_survival_metrics.csv")
    merged = per_cancer.merge(repeated, on=["representation", "cancer"], suffixes=("_oof", "_repeated"))
    merged["c_index_change"] = merged.c_index_oof - merged.c_index_repeated
    merged["ci_width_change"] = merged.ci_width_oof - merged.ci_width_repeated
    merged.to_csv(RESULTS / "survival_5fold_oof_comparison_to_repeated_splits.csv", index=False)
    return merged


def provenance(cohort: pd.DataFrame) -> None:
    record = {
        "analysis": "Complete true 5-fold OOF TCGA Cox survival evaluation",
        "eligible_patients": len(cohort), "cancers": int(cohort.cohort.nunique()),
        "events": int(cohort.event.sum()), "outer_folds": FOLDS, "fold_seed": OOF_SEED,
        "source_primary_tumor_patients": 9816,
        "excluded_patients": 148,
        "exclusion_reasons": {"invalid_binary_event": 21,
                              "valid_event_missing_time": 34,
                              "valid_event_nonpositive_time": 93},
        "stratification": "Deterministic within-cancer×event shuffling and balanced round-robin sparse-stratum fallback",
        "endpoint": "PanCanAtlas OS / OS.time; event in {0,1}; time_days > 0",
        "bridge": "Cached frozen Bridge 45.6M sample embeddings; downstream MLP [512,256] Cox head only",
        "pca": "StandardScaler + randomized PCA-128 fitted independently on each training fold; CoxPHFitter penalizer=0.1",
        "raw": "Existing 25,150-feature log1p(CPM), training-fold standardization, downstream MLP [512,256] Cox head",
        "cross_fold_scale_handling": "C-index numerator and comparable-pair counts are computed within held-out folds and summed; no pairs cross independently fitted heads",
        "bootstrap_replicates": BOOTSTRAPS, "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap": "Patient-level within cancer; fold membership retained; concordance remains fold-aware",
        "stability_rule": "unreliable if <900 valid bootstraps; very low events <10; low events 10-19; wide CI >0.25",
        "inputs": ["results/cohort_manifest.parquet", "work/ours_45.6m_embeddings.npy",
                   "work/ours_log1p_tpm.npy", "work/tcga_full_log1p_cpm.npy"],
    }
    (RESULTS / "survival_5fold_oof_provenance.json").write_text(json.dumps(record, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(); device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cohort = load_cohort(); folds = make_folds(cohort); provenance(cohort)
    predictions = generate_predictions(cohort, folds, device); qa_predictions(predictions, cohort, folds)
    overall, per_cancer = metrics_and_bootstrap(predictions)
    table = comparison_table(per_cancer); plots(table)
    summary = summaries(per_cancer, table); repeated_comparison(per_cancer)
    say("overall\n" + overall.to_string(index=False)); say("summary\n" + summary.to_string(index=False))


if __name__ == "__main__": main()
