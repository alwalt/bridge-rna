#!/usr/bin/env python3
"""Build the cached publication package without model inference or training."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, f1_score


PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[1]
DEFAULT_RESEARCH = REPO / "benchmarks" / "tissue_disease_annotation"
REPRESENTATIONS = ["bridge", "pca", "raw"]
COLORS = {"bridge": "#4C78A8", "pca": "#F58518", "raw": "#54A24B"}
DISPLAY = {"bridge": "Bridge", "pca": "PCA", "raw": "Raw expression"}
TASK_DISPLAY = {
    "gtex_broad": "GTEx broad tissue",
    "gtex_detailed": "GTEx detailed tissue",
    "tcga_cancer": "TCGA cancer type",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()


def recompute_fold_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, fold, representation), frame in predictions.groupby(
        ["task", "fold", "representation"], sort=True
    ):
        rows.append(
            {
                "task": task,
                "fold": int(fold),
                "representation": representation,
                "test_n": len(frame),
                "accuracy": accuracy_score(frame["true_label"], frame["predicted_label"]),
                "macro_f1": f1_score(
                    frame["true_label"], frame["predicted_label"], average="macro"
                ),
                "weighted_f1": f1_score(
                    frame["true_label"], frame["predicted_label"], average="weighted"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["task", "fold", "representation"])


def verify_predictions(results: Path) -> tuple[pd.DataFrame, dict]:
    gtex = pd.read_csv(results / "gtex_predictions.csv")
    tcga = pd.read_csv(results / "tcga_predictions.csv")
    predictions = pd.concat([gtex, tcga], ignore_index=True)
    recomputed = recompute_fold_metrics(predictions)
    saved = pd.concat(
        [
            pd.read_csv(results / "gtex_fold_metrics.csv"),
            pd.read_csv(results / "tcga_fold_metrics.csv"),
        ],
        ignore_index=True,
    )
    keys = ["task", "fold", "representation"]
    merged = recomputed.merge(saved, on=keys, suffixes=("_recomputed", "_saved"), validate="one_to_one")
    metric_differences = {}
    for metric in ["accuracy", "macro_f1", "weighted_f1"]:
        difference = np.abs(merged[f"{metric}_recomputed"] - merged[f"{metric}_saved"])
        metric_differences[metric] = float(difference.max())
        if not np.all(difference <= 1e-12):
            raise AssertionError(f"Saved {metric} differs from predictions")

    duplicated = predictions.duplicated(["task", "representation", "sample_id"]).sum()
    if duplicated:
        raise AssertionError(f"Found {duplicated} duplicated held-out predictions")
    alignment = []
    for (task, fold), frame in predictions.groupby(["task", "fold"]):
        sample_sets = [
            frozenset(frame.loc[frame["representation"] == rep, "sample_id"])
            for rep in REPRESENTATIONS
        ]
        alignment.append(len(set(sample_sets)) == 1)
    if not all(alignment):
        raise AssertionError("Held-out samples differ among representations")

    return recomputed, {
        "prediction_rows_checked": int(len(predictions)),
        "fold_representation_groups_checked": int(len(recomputed)),
        "maximum_absolute_metric_difference": metric_differences,
        "metric_tolerance": 1e-12,
        "held_out_sample_sets_identical": True,
        "duplicate_held_out_predictions": 0,
    }


def save_performance_figure(folds: pd.DataFrame, output: Path) -> None:
    tasks = ["gtex_detailed", "tcga_cancer"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharey=True)
    rng = np.random.default_rng(20260923)
    for panel, (axis, task) in enumerate(zip(axes, tasks)):
        subset = folds[folds["task"] == task]
        for index, rep in enumerate(REPRESENTATIONS):
            values = subset.loc[subset["representation"] == rep, "macro_f1"].to_numpy()
            jitter = rng.uniform(-0.055, 0.055, len(values))
            axis.scatter(
                np.full(len(values), index) + jitter,
                values,
                s=32,
                color=COLORS[rep],
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
            axis.errorbar(
                index,
                values.mean(),
                yerr=values.std(ddof=1),
                fmt="D",
                ms=6,
                color="black",
                capsize=4,
                lw=1.2,
                zorder=4,
            )
        classes = 51 if task == "gtex_detailed" else 33
        axis.set_xticks(range(3), [DISPLAY[r] for r in REPRESENTATIONS])
        axis.set_ylim(0.84, 0.97)
        axis.set_title(f"{'AB'[panel]}. {TASK_DISPLAY[task]}", loc="left", fontweight="bold")
        axis.text(
            0.02,
            0.03,
            f"Uniform-class reference: 1/{classes} = {1/classes:.3f} (below axis)",
            transform=axis.transAxes,
            fontsize=8,
            color="#555555",
        )
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Macro F1")
    fig.suptitle("Linear annotation from frozen Bridge and transcriptomic baselines", y=1.02)
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_broad_figure(folds: pd.DataFrame, output: Path) -> None:
    subset = folds[folds["task"] == "gtex_broad"]
    fig, axis = plt.subplots(figsize=(5.8, 4.3))
    rng = np.random.default_rng(20260923)
    for index, rep in enumerate(REPRESENTATIONS):
        values = subset.loc[subset["representation"] == rep, "macro_f1"].to_numpy()
        axis.scatter(
            np.full(len(values), index) + rng.uniform(-0.055, 0.055, len(values)),
            values,
            s=34,
            color=COLORS[rep],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        axis.errorbar(index, values.mean(), yerr=values.std(ddof=1), fmt="D", color="black", capsize=4)
    axis.set_xticks(range(3), [DISPLAY[r] for r in REPRESENTATIONS])
    axis.set_ylim(0.89, 1.0)
    axis.set_ylabel("Macro F1")
    axis.set_title("Supplementary GTEx broad-tissue annotation")
    axis.text(0.02, 0.03, "Uniform-class reference: 1/30 = 0.033 (below axis)", transform=axis.transAxes, fontsize=8, color="#555555")
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_secondary_metrics_figure(folds: pd.DataFrame, output: Path) -> None:
    tasks = ["gtex_detailed", "tcga_cancer"]
    metrics = [("accuracy", "Accuracy"), ("weighted_f1", "Weighted F1")]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), sharey="row")
    rng = np.random.default_rng(20260923)
    for row, (metric, metric_label) in enumerate(metrics):
        for column, task in enumerate(tasks):
            axis = axes[row, column]
            subset = folds[folds["task"] == task]
            for index, rep in enumerate(REPRESENTATIONS):
                values = subset.loc[subset["representation"] == rep, metric].to_numpy()
                axis.scatter(
                    np.full(len(values), index) + rng.uniform(-0.055, 0.055, len(values)),
                    values,
                    s=30,
                    color=COLORS[rep],
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=3,
                )
                axis.errorbar(index, values.mean(), yerr=values.std(ddof=1), fmt="D", color="black", capsize=4)
            classes = 51 if task == "gtex_detailed" else 33
            axis.set_xticks(range(3), [DISPLAY[r] for r in REPRESENTATIONS])
            axis.set_ylim(0.90, 0.98)
            axis.set_title(f"{TASK_DISPLAY[task]} — {metric_label}")
            axis.text(0.02, 0.03, f"1/{classes} chance reference below axis", transform=axis.transAxes, fontsize=8, color="#555555")
            axis.grid(axis="y", color="#DDDDDD", linewidth=0.7)
            axis.spines[["top", "right"]].set_visible(False)
            if column == 0:
                axis.set_ylabel(metric_label)
    fig.suptitle("Supplementary annotation metrics", y=1.01)
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_confusion_figure(csv_path: Path, title: str, output: Path, size: tuple[float, float]) -> None:
    matrix = pd.read_csv(csv_path, index_col=0)
    fig, axis = plt.subplots(figsize=size)
    image = axis.imshow(matrix.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=90, fontsize=5.5)
    axis.set_yticks(np.arange(len(matrix.index)), matrix.index, fontsize=5.5)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(title)
    colorbar = fig.colorbar(image, ax=axis, fraction=0.03, pad=0.02)
    colorbar.set_label("Row-normalized fraction")
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_per_class_figure(per_class: pd.DataFrame, task: str, output: Path) -> None:
    frame = per_class[(per_class["task"] == task) & (per_class["representation"] == "bridge")].copy()
    frame = frame.sort_values(["f1_mean", "class"], ascending=[True, True])
    height = max(7.0, 0.23 * len(frame))
    fig, axis = plt.subplots(figsize=(7.5, height))
    axis.barh(frame["class"], frame["f1_mean"], xerr=frame["f1_sd"], color=COLORS["bridge"], alpha=0.9, capsize=2)
    axis.set_xlim(0, 1.03)
    axis.set_xlabel("Bridge per-class F1 (mean ± SD across five folds)")
    axis.set_title(TASK_DISPLAY[task])
    axis.grid(axis="x", color="#E1E1E1", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--research-dir", type=Path, default=DEFAULT_RESEARCH)
    args = parser.parse_args()
    research = args.research_dir.resolve()
    results = research / "results"
    data = PACKAGE / "data"
    figures = PACKAGE / "figures"
    data.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    selected = [
        "metrics_summary.csv",
        "gtex_fold_metrics.csv",
        "tcga_fold_metrics.csv",
        "paired_macro_f1_differences.csv",
        "per_class_summary.csv",
        "gtex_detailed_class_counts.csv",
        "gtex_broad_class_counts.csv",
        "tcga_class_counts.csv",
        "gtex_detailed_bridge_confusion_normalized.csv",
        "gtex_broad_bridge_confusion_normalized.csv",
        "tcga_cancer_bridge_confusion_normalized.csv",
        "final_qc.json",
        "runtime_summary.json",
    ]
    for name in selected:
        shutil.copy2(results / name, data / name)

    recomputed, verification = verify_predictions(results)
    recomputed.to_csv(data / "recomputed_fold_metrics.csv", index=False)
    fold_metrics = pd.concat(
        [pd.read_csv(data / "gtex_fold_metrics.csv"), pd.read_csv(data / "tcga_fold_metrics.csv")],
        ignore_index=True,
    )
    figure_data = fold_metrics[["task", "fold", "representation", "accuracy", "macro_f1", "weighted_f1"]].copy()
    figure_data.to_csv(data / "annotation_figure_data.csv", index=False)
    pd.DataFrame(
        [
            {"task": "gtex_detailed", "samples": 8883, "biological_units": 549, "unit": "donors", "classes": 51, "chance_reference": 1 / 51},
            {"task": "tcga_cancer", "samples": 9942, "biological_units": 9942, "unit": "patients", "classes": 33, "chance_reference": 1 / 33},
            {"task": "gtex_broad", "samples": 8883, "biological_units": 549, "unit": "donors", "classes": 30, "chance_reference": 1 / 30},
        ]
    ).to_csv(data / "cohort_summary.csv", index=False)

    save_performance_figure(fold_metrics, figures / "main_annotation_macro_f1")
    save_broad_figure(fold_metrics, figures / "supplementary_gtex_broad_macro_f1")
    save_secondary_metrics_figure(fold_metrics, figures / "supplementary_accuracy_weighted_f1")
    save_confusion_figure(
        data / "gtex_detailed_bridge_confusion_normalized.csv",
        "GTEx detailed tissue: Bridge normalized confusion matrix",
        figures / "gtex_detailed_bridge_confusion_normalized",
        (15, 14),
    )
    save_confusion_figure(
        data / "tcga_cancer_bridge_confusion_normalized.csv",
        "TCGA cancer type: Bridge normalized confusion matrix",
        figures / "tcga_cancer_bridge_confusion_normalized",
        (11, 10),
    )
    per_class = pd.read_csv(data / "per_class_summary.csv")
    save_per_class_figure(per_class, "gtex_detailed", figures / "gtex_detailed_bridge_per_class_f1")
    save_per_class_figure(per_class, "tcga_cancer", figures / "tcga_cancer_bridge_per_class_f1")

    provenance_source = json.loads((results / "provenance.json").read_text())
    portable_assets = {}
    for source_path, record in provenance_source["assets"].items():
        marker = "/bridge-rna/"
        portable_name = (
            f"source_repository/{source_path.split(marker, 1)[1]}"
            if marker in source_path
            else f"external_asset/{Path(source_path).name}"
        )
        portable_assets[portable_name] = record
    source_files = selected + ["gtex_predictions.csv", "tcga_predictions.csv", "provenance.json"]
    source_hashes = {name: sha256(results / name) for name in source_files}
    manifest_hashes = {
        "gtex_manifest.csv": sha256(results / "gtex_manifest.csv"),
        "tcga_manifest.csv": sha256(results / "tcga_manifest.csv"),
    }
    provenance = {
        "package_created_from_source_commit": git_commit(),
        "research_benchmark": "benchmarks/tissue_disease_annotation",
        "research_pipeline": "benchmarks/tissue_disease_annotation/pipeline/run_benchmark.py",
        "approval_record": "benchmarks/tissue_disease_annotation/PUBLICATION_REVIEW.md",
        "approval_event": "Explicit benchmark-notebook skill invocation in the user conversation on 2026-10-30",
        "approval_scope": "Supplementary material with a concise main-text reference",
        "source_artifact_sha256": source_hashes,
        "frozen_manifest_sha256": manifest_hashes,
        "external_asset_hashes_reused_from_research_provenance": portable_assets,
        "checkpoint_run_id": provenance_source["config"]["checkpoint_run_id"],
        "parameters": provenance_source["config"],
        "input_contract": provenance_source["input_contract"],
        "pretraining": provenance_source["pretraining"],
        "metric_definitions": {
            "primary": "macro F1",
            "secondary": ["accuracy", "weighted F1"],
            "uncertainty": "sample SD across five fixed held-out folds",
            "chance_reference": "1 / number of classes; descriptive uniform-class reference",
        },
        "verification": verification,
        "dependencies": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": plt.matplotlib.__version__,
        },
        "limitations": [
            "ARCHS4 pretraining may contain public GTEx and/or TCGA profiles.",
            "This benchmark measures frozen-representation linear separability, not guaranteed sample-unseen external generalization.",
            "Fold SD reflects one fixed five-fold partition, not independent model-training replicates.",
            "Rare GTEx classes yield unstable class-level F1 estimates.",
            "Fold-level PCA cumulative explained variance was not retained in the frozen research outputs.",
            "GTEx-versus-TCGA normal/tumor classification is out of scope because cohort effects confound disease state.",
        ],
    }
    (PACKAGE / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (PACKAGE / "validation.json").write_text(json.dumps(verification, indent=2) + "\n")
    (PACKAGE / "validation.log").write_text(
        "PASS: prediction-level metrics match saved fold metrics within 1e-12.\n"
        "PASS: held-out sample sets are identical across Bridge, PCA, and raw expression.\n"
        "PASS: no duplicate held-out predictions were detected.\n"
        "PASS: publication figures regenerated from packaged fold-level and aggregate data.\n"
    )
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
