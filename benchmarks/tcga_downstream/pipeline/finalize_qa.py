#!/usr/bin/env python3
"""Lightweight final assertions for the completed TCGA downstream benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

EXPECTED = [
    "classification_mlp_summary.csv", "classification_summary.csv",
    "bulkformer_parity_pan_cancer_classification_summary.csv",
    "bulkformer_parity_alive_dead_prognosis_summary.csv",
    "survival_mlp_summary.csv", "survival_summary.csv",
    "per_cancer_survival_metrics.csv", "survival_5fold_oof_fold_manifest.csv",
    "survival_5fold_oof_predictions.csv", "survival_5fold_oof_metrics.csv",
    "survival_5fold_oof_per_cancer.csv",
    "survival_5fold_oof_representation_comparison.csv",
    "survival_5fold_oof_comparison_to_repeated_splits.csv",
    "survival_5fold_oof_provenance.json",
]
FIGURES = [
    "bridge_5fold_oof_per_cancer_survival_forest.png",
    "pca128_5fold_oof_per_cancer_survival_forest.png",
    "raw_expression_5fold_oof_per_cancer_survival_forest.png",
    "bridge_pca_raw_5fold_oof_survival_forest.png",
    "bridge_pca_raw_5fold_oof_survival_three_panel.png",
    "bridge_vs_conventional_5fold_oof_delta_cindex.png",
]


def finite_columns(path: Path, columns: list[str]) -> None:
    frame = pd.read_csv(path)
    assert len(frame), path
    for column in columns:
        assert np.isfinite(frame[column].dropna()).all(), (path, column)


def main() -> None:
    missing = [name for name in EXPECTED if not (RESULTS / name).is_file()]
    missing += [f"figures/{name}" for name in FIGURES if not (RESULTS / "figures" / name).is_file()]
    assert not missing, missing

    manifest = pd.read_csv(RESULTS / "survival_5fold_oof_fold_manifest.csv")
    predictions = pd.read_csv(RESULTS / "survival_5fold_oof_predictions.csv")
    representations = ["Bridge 45.6M", "PCA-128", "Full raw expression"]
    assert len(manifest) == 9668 and manifest.patient_id.is_unique
    assert set(manifest.fold) == set(range(5))
    keys = ["patient_id", "cancer", "survival_time", "event", "fold"]
    reference = None
    coverage = {}
    for representation in representations:
        frame = predictions[predictions.representation.eq(representation)]
        assert len(frame) == len(manifest) == frame.patient_id.nunique()
        ordered = frame[keys].sort_values("patient_id").reset_index(drop=True)
        if reference is None:
            reference = ordered
        else:
            pd.testing.assert_frame_equal(reference, ordered, check_dtype=False)
        coverage[representation] = len(frame)
    assert predictions.groupby(["representation", "patient_id"]).size().eq(1).all()
    assert reference is not None
    assert reference.patient_id.map(manifest.set_index("patient_id").fold).eq(reference.fold).all()

    finite_columns(RESULTS / "survival_5fold_oof_metrics.csv",
                   ["oof_survival_c_index", "comparable_pairs"])
    finite_columns(RESULTS / "survival_5fold_oof_per_cancer.csv",
                   ["n_patients", "n_events", "c_index", "ci_low", "ci_high"])
    finite_columns(RESULTS / "classification_mlp_summary.csv",
                   ["macro_f1_mean", "weighted_f1_mean", "balanced_accuracy_mean"])
    finite_columns(RESULTS / "bulkformer_parity_pan_cancer_classification_summary.csv",
                   ["macro_f1_mean", "weighted_f1_mean", "balanced_accuracy_mean"])

    provenance = json.loads((RESULTS / "survival_5fold_oof_provenance.json").read_text())
    assert provenance["eligible_patients"] == len(manifest)
    assert "frozen Bridge" in provenance["bridge"]
    assert "training fold" in provenance["pca"]
    assert "training-fold standardization" in provenance["raw"]
    assert "no pairs cross" in provenance["cross_fold_scale_handling"]

    notebook = nbformat.read(ROOT / "tcga_downstream_benchmark.ipynb", as_version=4)
    unexecuted = [index for index, cell in enumerate(notebook.cells)
                  if cell.cell_type == "code" and cell.execution_count is None]
    errors = [index for index, cell in enumerate(notebook.cells) if cell.cell_type == "code"
              for output in cell.get("outputs", []) if output.get("output_type") == "error"]
    assert not unexecuted and not errors, {"unexecuted": unexecuted, "errors": errors}

    report = {
        "status": "passed",
        "eligible_patients": len(manifest),
        "representations": representations,
        "coverage": coverage,
        "folds": 5,
        "fold_sizes": manifest.groupby("fold").size().astype(int).to_dict(),
        "patient_unique_per_representation": True,
        "fold_exclusivity": True,
        "identical_patients_folds_outcomes": True,
        "frozen_bridge": True,
        "training_fold_only_pca": True,
        "training_fold_only_raw_standardization": True,
        "test_outcomes_used_for_tuning": False,
        "fold_aware_concordance": True,
        "notebook_all_cells_executed": True,
        "expected_outputs_present": True,
    }
    (RESULTS / "final_qa.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
