#!/usr/bin/env python3
"""Fail closed if a frozen Phase 2 artifact is missing or internally inconsistent."""

import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"
checks = {}

manifest = pd.read_parquet(RESULTS / "manifests/sample_manifest.parquet")
members = pd.read_parquet(RESULTS / "manifests/contrast_members.parquet")
sensitivity = pd.read_csv(RESULTS / "manifests/GSE211204_unseen_subject_sensitivity.csv")
checks["datasets"] = int(manifest.dataset.nunique())
checks["manifest_rows"] = len(manifest)
checks["primary_contrasts"] = int(members.contrast_id.nunique())
checks["attributed_sample_rows"] = len(members) + len(sensitivity)
assert checks["datasets"] == 9 and checks["primary_contrasts"] == 15

vectors = pd.read_parquet(RESULTS / "bridge/condition_vectors.parquet")
bridge = pd.read_parquet(RESULTS / "bridge/study_gene_rankings.parquet")
contrast = pd.read_parquet(RESULTS / "bridge/contrast_gene_scores.parquet")
complete = pd.read_csv(RESULTS / "bridge/ig_completeness.csv")
checks["condition_vectors"] = int(vectors.contrast_id.nunique())
checks["condition_vector_norm_min"] = float(vectors.groupby("contrast_id").value.apply(lambda x: np.linalg.norm(x)).min())
checks["condition_vector_norm_max"] = float(vectors.groupby("contrast_id").value.apply(lambda x: np.linalg.norm(x)).max())
checks["study_ranking_rows"] = len(bridge)
checks["contrast_score_rows"] = len(contrast)
checks["ig_completeness_rows"] = len(complete)
checks["ig_absolute_completeness_median"] = float(complete.absolute_completeness_delta.median())
checks["ig_absolute_completeness_p95"] = float(complete.absolute_completeness_delta.quantile(.95))
assert checks["condition_vectors"] == 16
assert np.isclose(checks["condition_vector_norm_min"], 1, atol=1e-5) and np.isclose(checks["condition_vector_norm_max"], 1, atol=1e-5)
assert len(bridge) == 9 * 15165 and len(contrast) == 16 * 15165
assert len(complete) == checks["attributed_sample_rows"]

modules = pd.read_parquet(RESULTS / "evaluation/gene_modules.parquet")
expected = 2 * 9 * sum(json.loads((HERE / "config.json").read_text())["module_sizes"])
checks["module_rows"] = len(modules)
assert len(modules) == expected
assert modules.groupby(["method", "dataset", "module_size"]).gene.nunique().eq(modules.groupby(["method", "dataset", "module_size"]).size()).all()

effects = pd.read_csv(RESULTS / "evaluation/primary_effects.csv")
random = pd.read_parquet(RESULTS / "evaluation/random_module_null.parquet")
checks["primary_effect_rows"] = len(effects)
checks["random_module_rows"] = len(random)
assert set(effects.condition) == {"Radiation", "Bone loss", "Muscle atrophy", "Overall"}
assert len(random) == 3 * 2 * 1000

chembl = json.loads((RESULTS / "chembl37_contract_qc.json").read_text())
checks["chembl_archive_verified"] = bool(chembl.get("local_archive_checksum_verified"))
checks["chembl_edges"] = int(chembl["retained_unique_drug_gene_edges"])
assert checks["chembl_archive_verified"] and chembl["contract_valid"]

notebook = nbformat.read(HERE / "drug_discovery_benchmark.ipynb", as_version=4)
checks["notebook_code_cells"] = sum(c.cell_type == "code" for c in notebook.cells)
checks["executed_notebook_code_cells"] = sum(c.cell_type == "code" and c.get("execution_count") is not None for c in notebook.cells)
assert checks["notebook_code_cells"] == checks["executed_notebook_code_cells"]

(RESULTS / "validation_summary.json").write_text(json.dumps(checks, indent=2) + "\n")
print(json.dumps(checks, indent=2))
