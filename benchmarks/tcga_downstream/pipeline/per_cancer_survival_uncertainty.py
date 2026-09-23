#!/usr/bin/env python3
"""Bootstrap within-cancer C-indices from existing held-out Cox predictions."""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils.concordance import _concordance_summary_statistics
from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split

from common import CONFIG, RESULTS, WORK
from run_benchmark import build_cohorts, reduced, representation, safe_cindex

B = 1000
BOOTSTRAP_SEED = 20260922
PERMUTATIONS = 100_000
FIGURES = RESULTS / "figures"
NAMES = {
    "Bridge 45.6M": "ours_45.6m",
    "PCA-128": "pca128",
    "Full raw expression": "full_expression_25150",
}


def say(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def cohort_and_splits() -> tuple[pd.DataFrame, dict[int, tuple[np.ndarray, np.ndarray]]]:
    cohort = build_cohorts().loc[lambda x: x.survival_usable].copy().reset_index(drop=True)
    strata = cohort.cohort.astype(str)
    splits = {}
    for seed in CONFIG["split_seeds"]:
        train, test = train_test_split(
            np.arange(len(cohort)), test_size=CONFIG["test_fraction"],
            random_state=seed, stratify=strata)
        splits[int(seed)] = (train, test)
    return cohort, splits


def audit_native_predictions(cohort: pd.DataFrame, splits: dict) -> pd.DataFrame:
    path = RESULTS / "cox_mlp_risk_predictions.csv"
    data = pd.read_csv(path)
    required = {"seed", "patient_id", "cancer_label", "representation", "time_days", "event", "risk"}
    assert required <= set(data.columns)
    wanted = data[data.representation.isin([NAMES["Bridge 45.6M"], NAMES["Full raw expression"]])].copy()
    for seed, (_, test) in splits.items():
        expected = set(cohort.iloc[test].patient_id.astype(str))
        for representation in wanted.representation.unique():
            observed = set(wanted.loc[(wanted.seed == seed) & (wanted.representation == representation),
                                      "patient_id"].astype(str))
            assert observed == expected, (seed, representation, len(observed), len(expected))
    assert not wanted.duplicated(["seed", "patient_id", "representation"]).any()
    return wanted


def pca_predictions(cohort: pd.DataFrame, splits: dict) -> pd.DataFrame:
    output = RESULTS / "cox_pca128_risk_predictions.csv"
    if output.is_file():
        say(f"reusing {output.name}")
        return pd.read_csv(output)
    say("PCA-128 patient risks were not saved; refitting only the original PCA-128 Cox control")
    matrix = representation("raw_expression")
    rows = []
    source_rows = cohort.matrix_row.to_numpy(int)
    reference = pd.read_parquet(RESULTS / "survival_per_split.parquet")
    reference = reference[(reference.representation == "raw_expression") &
                          (reference.probe == "pca128_cox")].set_index("seed")
    for seed, (train, test) in splits.items():
        xtrain, xtest = reduced(np.asarray(matrix[source_rows[train]]),
                                np.asarray(matrix[source_rows[test]]), seed)
        columns = [f"x{i}" for i in range(xtrain.shape[1])]
        fit = pd.DataFrame(xtrain, columns=columns)
        fit["time_days"] = cohort.time_days.to_numpy(float)[train]
        fit["event"] = cohort.event.to_numpy(int)[train]
        model = CoxPHFitter(penalizer=float(CONFIG["survival_penalizer"]))
        model.fit(fit, duration_col="time_days", event_col="event", show_progress=False)
        risk = model.predict_partial_hazard(pd.DataFrame(xtest, columns=columns)).to_numpy()
        held = cohort.iloc[test].reset_index(drop=True)
        observed = safe_cindex(held, risk)
        assert np.isclose(observed, float(reference.loc[seed, "c_index"]), atol=1e-10)
        rows.extend({"seed": seed, "patient_id": row.patient_id, "cancer_label": row.cohort,
                     "representation": NAMES["PCA-128"], "time_days": float(row.time_days),
                     "event": int(row.event), "risk": float(value)}
                    for row, value in zip(held.itertuples(), risk))
        say(f"PCA-128 seed={seed} reproduced pooled C-index={observed:.6f}")
    result = pd.DataFrame(rows)
    result.to_csv(output, index=False)
    return result


def concordance_parts(frame: pd.DataFrame) -> tuple[float, float, int]:
    if len(frame) < 2 or frame.event.sum() == 0:
        return 0.0, 0.0, 0
    correct, tied, pairs = _concordance_summary_statistics(
        frame.time_days.to_numpy(float), -frame.risk.to_numpy(float), frame.event.to_numpy(bool))
    return float(correct), float(tied), int(pairs)


def aggregate_cindex(frame: pd.DataFrame) -> tuple[float, int]:
    correct = tied = 0.0
    pairs = 0
    for _, seed_frame in frame.groupby("seed"):
        c, t, p = concordance_parts(seed_frame)
        correct += c; tied += t; pairs += p
    return ((correct + 0.5 * tied) / pairs if pairs else np.nan), pairs


def stability(events: int, valid: int, width: float) -> str:
    flags = []
    if valid < 900 or not np.isfinite(width): flags.append("unreliable_bootstrap")
    if events < 10: flags.append("very_low_event_support")
    elif events < 20: flags.append("low_event_support")
    if np.isfinite(width) and width > 0.25: flags.append("wide_ci")
    return ";".join(flags) if flags else "stable"


def calculate(combined: pd.DataFrame) -> pd.DataFrame:
    output = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    cancers = sorted(combined.cancer_label.unique())
    for number, cancer in enumerate(cancers, 1):
        cancer_data = combined[combined.cancer_label == cancer]
        support = cancer_data.drop_duplicates("patient_id")
        patients = np.sort(support.patient_id.astype(str).unique())
        draws = rng.integers(0, len(patients), size=(B, len(patients)))
        multiplicities = np.apply_along_axis(np.bincount, 1, draws, minlength=len(patients))
        support = support.set_index("patient_id").loc[patients]
        for label, source in NAMES.items():
            frame = cancer_data[cancer_data.representation == source].copy()
            estimate, pairs = aggregate_cindex(frame)
            by_seed = {seed: group.set_index("patient_id").reindex(patients)
                       for seed, group in frame.groupby("seed")}
            boot = []
            for counts in multiplicities:
                pieces = []
                for seed, group in by_seed.items():
                    present = group.risk.notna().to_numpy()
                    repeats = counts * present
                    if repeats.sum():
                        piece = group.loc[present].reset_index()
                        pieces.append(piece.loc[piece.index.repeat(repeats[present])].assign(seed=seed))
                value, _ = aggregate_cindex(pd.concat(pieces, ignore_index=True)) if pieces else (np.nan, 0)
                if np.isfinite(value): boot.append(value)
            low, high = (np.quantile(boot, [0.025, 0.975]) if boot else (np.nan, np.nan))
            events = int(support.event.sum())
            width = float(high - low) if np.isfinite(low) else np.nan
            output.append({"representation": label, "cancer": cancer,
                           "n_patients": len(patients), "n_events": events,
                           "n_censored": len(patients) - events,
                           "censoring_fraction": 1 - events / len(patients),
                           "comparable_pairs": pairs, "c_index": estimate,
                           "ci_low": low, "ci_high": high, "ci_width": width,
                           "bootstrap_valid": len(boot),
                           "stability_flag": stability(events, len(boot), width)})
        say(f"bootstrap {number}/{len(cancers)} cancer={cancer}")
    result = pd.DataFrame(output)
    result.to_csv(RESULTS / "per_cancer_survival_metrics.csv", index=False)
    return result


def axis_limits(metrics: pd.DataFrame) -> tuple[float, float]:
    values = pd.concat([metrics.ci_low, metrics.ci_high]).dropna()
    return max(0.0, float(values.min()) - .04), min(1.0, float(values.max()) + .12)


def forest_bridge(metrics: pd.DataFrame, order: list[str], limits: tuple[float, float], stem: str) -> None:
    data = metrics[(metrics.representation == "Bridge 45.6M")].set_index("cancer").reindex(order)
    fig, ax = plt.subplots(figsize=(8.3, max(8.5, .31 * len(data) + 1.8)))
    y = np.arange(len(data))
    valid = data.c_index.notna() & data.ci_low.notna()
    ax.errorbar(data.loc[valid, "c_index"], y[valid],
                xerr=np.vstack([data.loc[valid, "c_index"] - data.loc[valid, "ci_low"],
                                data.loc[valid, "ci_high"] - data.loc[valid, "c_index"]]),
                fmt="o", color="#1f5a94", ecolor="#7296b8", capsize=2.5, ms=5, lw=1.3)
    ax.axvline(.5, color="#555555", ls="--", lw=1)
    ax.set(yticks=y, yticklabels=data.index, xlim=limits, xlabel="Held-out C-index",
           ylabel="TCGA cancer type", title="Bridge survival discrimination across TCGA cancer types")
    ax.invert_yaxis(); ax.grid(axis="x", color="#dddddd", lw=.7)
    annotation_x = 1.02
    transform = ax.get_yaxis_transform()
    ax.text(annotation_x, -1.05, "Events / N", transform=transform, clip_on=False,
            ha="left", va="bottom", weight="bold", fontsize=9)
    for yy, row in enumerate(data.itertuples()):
        ax.text(annotation_x, yy, f"{row.n_events} / {row.n_patients}", transform=transform,
                clip_on=False, ha="left", va="center", fontsize=8)
    fig.tight_layout(rect=(0, 0, .88, 1))
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def forest_baselines(metrics: pd.DataFrame, order: list[str], limits: tuple[float, float]) -> None:
    fig, ax = plt.subplots(figsize=(8.3, max(8.5, .31 * len(order) + 1.8)))
    colors = {"PCA-128": "#d17a22", "Full raw expression": "#4f8b3a"}
    offsets = {"PCA-128": -.11, "Full raw expression": .11}
    y = np.arange(len(order))
    for label in offsets:
        data = metrics[metrics.representation == label].set_index("cancer").reindex(order)
        valid = data.c_index.notna() & data.ci_low.notna()
        yy = y + offsets[label]
        ax.errorbar(data.loc[valid, "c_index"], yy[valid],
                    xerr=np.vstack([data.loc[valid, "c_index"] - data.loc[valid, "ci_low"],
                                    data.loc[valid, "ci_high"] - data.loc[valid, "c_index"]]),
                    fmt="o", label=label, color=colors[label], capsize=2, ms=4.5, lw=1.15)
    bridge = metrics[metrics.representation == "Bridge 45.6M"].set_index("cancer").reindex(order)
    ax.axvline(.5, color="#555555", ls="--", lw=1)
    ax.set(yticks=y, yticklabels=order, xlim=limits, xlabel="Held-out C-index",
           ylabel="TCGA cancer type", title="Conventional expression baselines across TCGA cancer types")
    ax.invert_yaxis(); ax.grid(axis="x", color="#dddddd", lw=.7); ax.legend(frameon=False)
    annotation_x = 1.02
    transform = ax.get_yaxis_transform()
    ax.text(annotation_x, -1.05, "Events / N", transform=transform, clip_on=False,
            ha="left", va="bottom", weight="bold", fontsize=9)
    for yy, row in enumerate(bridge.itertuples()):
        ax.text(annotation_x, yy, f"{row.n_events} / {row.n_patients}", transform=transform,
                clip_on=False, ha="left", va="center", fontsize=8)
    fig.tight_layout(rect=(0, 0, .88, 1))
    fig.savefig(FIGURES / "conventional_per_cancer_survival_forest.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "conventional_per_cancer_survival_forest.pdf", bbox_inches="tight")
    plt.close(fig)


def delta_plot(metrics: pd.DataFrame, order: list[str]) -> None:
    wide = metrics.pivot(index="cancer", columns="representation", values="c_index").reindex(order)
    fig, ax = plt.subplots(figsize=(7.7, max(8.5, .31 * len(order) + 1.5)))
    y = np.arange(len(order))
    ax.scatter(wide["Bridge 45.6M"] - wide["PCA-128"], y - .1, label="Bridge − PCA-128", s=24)
    ax.scatter(wide["Bridge 45.6M"] - wide["Full raw expression"], y + .1,
               label="Bridge − full raw expression", s=24)
    ax.axvline(0, color="#555555", ls="--", lw=1)
    ax.set(yticks=y, yticklabels=order, xlabel="Paired difference in held-out C-index",
           ylabel="TCGA cancer type", title="Bridge versus conventional survival representations")
    ax.invert_yaxis(); ax.grid(axis="x", color="#dddddd", lw=.7); ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "bridge_vs_conventional_delta_cindex.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "bridge_vs_conventional_delta_cindex.pdf", bbox_inches="tight")
    plt.close(fig)


def statistical_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    wide = metrics.pivot(index="cancer", columns="representation", values="c_index")
    bridge = metrics[metrics.representation == "Bridge 45.6M"].set_index("cancer")
    rows = [
        {"summary": "median_bridge_c_index", "value": bridge.c_index.median()},
        {"summary": "patient_weighted_mean_bridge_c_index", "value": np.average(bridge.c_index, weights=bridge.n_patients)},
        {"summary": "bridge_cancers_gt_0.5", "value": int((bridge.c_index > .5).sum())},
        {"summary": "bridge_cancers_le_0.5", "value": int((bridge.c_index <= .5).sum())},
        {"summary": "bridge_ci_excludes_0.5", "value": int(((bridge.ci_low > .5) | (bridge.ci_high < .5)).sum())},
        {"summary": "median_pca128_c_index", "value": wide["PCA-128"].median()},
        {"summary": "median_full_raw_c_index", "value": wide["Full raw expression"].median()},
    ]
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    for comparator in ["PCA-128", "Full raw expression"]:
        difference = (wide["Bridge 45.6M"] - wide[comparator]).dropna().to_numpy()
        observed = float(difference.mean())
        permuted = (rng.choice([-1, 1], size=(PERMUTATIONS, len(difference))) * difference).mean(axis=1)
        p_perm = (1 + np.sum(np.abs(permuted) >= abs(observed))) / (PERMUTATIONS + 1)
        p_wilcoxon = float(wilcoxon(difference, alternative="two-sided").pvalue)
        slug = "pca128" if comparator == "PCA-128" else "full_raw"
        rows.extend([
            {"summary": f"median_bridge_minus_{slug}", "value": np.median(difference)},
            {"summary": f"cancers_bridge_gt_{slug}", "value": int((difference > 0).sum())},
            {"summary": f"paired_signflip_mean_p_{slug}", "value": p_perm},
            {"summary": f"paired_wilcoxon_p_{slug}", "value": p_wilcoxon},
        ])
    result = pd.DataFrame(rows)
    result.to_csv(RESULTS / "per_cancer_survival_statistical_summary.csv", index=False)
    return result


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    cohort, splits = cohort_and_splits()
    native = audit_native_predictions(cohort, splits)
    pca = pca_predictions(cohort, splits)
    combined = pd.concat([native, pca], ignore_index=True)
    keys = ["seed", "patient_id"]
    counts = combined.groupby("representation").apply(lambda x: set(map(tuple, x[keys].to_numpy())), include_groups=False)
    assert all(value == counts.iloc[0] for value in counts.iloc[1:])
    outcome_counts = combined.groupby(keys).agg(
        cancers=("cancer_label", "nunique"), times=("time_days", "nunique"),
        events=("event", "nunique"), representations=("representation", "nunique"))
    assert outcome_counts[["cancers", "times", "events"]].eq(1).all().all()
    assert outcome_counts.representations.eq(len(NAMES)).all()
    metrics_path = RESULTS / "per_cancer_survival_metrics.csv"
    if metrics_path.is_file():
        say(f"reusing {metrics_path.name}")
        metrics = pd.read_csv(metrics_path)
    else:
        metrics = calculate(combined)
    order = sorted(metrics.cancer.unique())
    limits = axis_limits(metrics)
    forest_bridge(metrics, order, limits, "bridge_per_cancer_survival_forest")
    bridge_order = metrics[metrics.representation == "Bridge 45.6M"].sort_values("c_index", ascending=False).cancer.tolist()
    forest_bridge(metrics, bridge_order, limits, "bridge_per_cancer_survival_forest_sorted")
    forest_baselines(metrics, order, limits)
    delta_plot(metrics, order)
    summary = statistical_summary(metrics)
    provenance = {
        "analysis": "Per-cancer survival discrimination from held-out Cox risks",
        "sources": {"bridge_and_full_raw": "results/cox_mlp_risk_predictions.csv",
                    "pca128": "results/cox_pca128_risk_predictions.csv",
                    "cohort": "results/cohort_manifest.parquet",
                    "original_pca_summary": "results/survival_per_split.parquet"},
        "split_design": "Five fixed repeated stratified 80/20 train/test splits; concordance pairs are formed only within a seed's held-out test set.",
        "patient_coverage": int(combined.patient_id.nunique()),
        "eligible_survival_patients": int(len(cohort)),
        "bootstrap_replicates": B, "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_unit": "Unique patient within cancer, clustered across all held-out seed appearances",
        "c_index_implementation": "lifelines concordance_index internals; predicted event time = -risk",
        "stability_rule": "unreliable_bootstrap if <900 valid replicates; very_low_event_support if events<10; low_event_support if events<20; wide_ci if width>0.25",
        "permutation_replicates": PERMUTATIONS,
        "limitations": ["Repeated holdout is not exhaustive K-fold OOF; some eligible patients never enter a test split.",
                        "Cancer-level C-indices aggregate comparable-pair counts across held-out seeds; risk scales are never compared across seeds."],
    }
    (RESULTS / "per_cancer_survival_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    say("summary\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
