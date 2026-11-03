#!/usr/bin/env python3
"""Validate the executed cached-evidence drug-discovery notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import nbformat
import pandas as pd


REPO = Path(__file__).resolve().parents[3]
PACKAGE = REPO / "notebooks" / "drug_discovery"
CACHE = PACKAGE / "cached_evidence"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    provenance_path = CACHE / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    checks = []
    for artifact in provenance["artifacts"]:
        packaged = REPO / artifact["packaged_path"]
        source = REPO / artifact["source_path"]
        checks.append(
            {
                "artifact": artifact["packaged_path"],
                "packaged_exists": packaged.exists(),
                "source_exists": source.exists(),
                "packaged_hash_matches": packaged.exists() and sha256(packaged) == artifact["sha256"],
                "source_hash_matches": source.exists() and sha256(source) == artifact["sha256"],
            }
        )
    check_df = pd.DataFrame(checks)
    if not check_df.drop(columns="artifact").all().all():
        raise AssertionError(check_df.loc[~check_df.drop(columns="artifact").all(axis=1)].to_string(index=False))

    notebook_path = PACKAGE / "drug_discovery.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if not code_cells or any(cell.execution_count is None for cell in code_cells):
        raise AssertionError("Notebook contains unexecuted code cells")
    errors = [out for cell in code_cells for out in cell.get("outputs", []) if out.output_type == "error"]
    if errors:
        raise AssertionError(f"Notebook has {len(errors)} error outputs")
    joined = "\n".join(cell.source for cell in notebook.cells)
    forbidden = ["public-facing", "public notebook", "publication-ready", "publication package"]
    found = [term for term in forbidden if term in joined.lower()]
    if found:
        raise AssertionError(f"Forbidden audience/packaging language: {found}")

    expected_figures = [
        "task1_disease_gene_recovery.png", "task1_disease_gene_recovery.pdf", "task1_disease_gene_recovery.csv",
        "task2_cross_study_generalization.png", "task2_cross_study_generalization.pdf", "task2_cross_study_generalization.csv",
        "radiation_context_results.png", "radiation_context_results.pdf", "radiation_core_counts.csv",
        "radiation_fibroblast_jaccard.csv",
    ]
    missing = [name for name in expected_figures if not (PACKAGE / "figures" / name).exists()]
    if missing:
        raise AssertionError(f"Missing figures or figure data: {missing}")

    helpers = list(PACKAGE.rglob("*.py")) + list(PACKAGE.rglob("*.sh"))
    if helpers:
        raise AssertionError(f"Executable helpers found under notebook package: {helpers}")

    provenance["verification"].update(
        {
            "status": "passed",
            "artifact_hashes_verified": len(check_df),
            "executed_code_cells": len(code_cells),
            "notebook_error_outputs": 0,
            "prediction_level_verification": "not performed; compact aggregate source tables were verified",
        }
    )
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    validation = {
        "status": "passed",
        "artifact_hashes_verified": len(check_df),
        "executed_code_cells": len(code_cells),
        "figures_verified": len(expected_figures),
        "checks": checks,
    }
    (CACHE / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps({k: validation[k] for k in ("status", "artifact_hashes_verified", "executed_code_cells", "figures_verified")}, indent=2))


if __name__ == "__main__":
    main()
