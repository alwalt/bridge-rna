#!/usr/bin/env python3
"""Fail-fast validation for the de Weerd replication deliverables."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"
EXPECTED = {"GSE138614": (16, 25), "GSE101794": (61, 13), "GSE72509": (99, 18)}


def main():
    checks = []
    samples = pd.read_csv(RESULTS / "manifests/locked_samples.csv")
    for accession, (cases, controls) in EXPECTED.items():
        group = samples[samples.accession.eq(accession)]
        assert group.sample_id.is_unique
        assert int(group.role.eq("case").sum()) == cases
        assert int(group.role.eq("control").sum()) == controls
        for method, path in [
            ("Bridge", RESULTS / f"bridge/{accession}_gene_ranking.parquet"),
            ("DE", RESULTS / f"differential_expression/{accession}_gene_ranking.csv.gz"),
        ]:
            ranking = pd.read_parquet(path) if method == "Bridge" else pd.read_csv(path)
            assert len(ranking) == 15165 and ranking.gene.is_unique
            assert ranking["rank"].tolist() == list(range(1, 15166))
        completeness_path = RESULTS / f"bridge/{accession}_ig_completeness.csv"
        completeness = pd.read_csv(completeness_path)
        relative = completeness.completeness_delta.abs() / completeness.score_difference.abs().clip(lower=1e-8)
        checks.append({"accession": accession, "samples": len(group),
                       "ig_median_absolute_delta": float(completeness.completeness_delta.abs().median()),
                       "ig_max_absolute_delta": float(completeness.completeness_delta.abs().max()),
                       "ig_median_relative_delta": float(relative.median())})
    recovery = pd.read_csv(RESULTS / "evaluation/disease_gene_recovery.csv")
    assert len(recovery) == 3 * 2 * 10 * 2
    primary = recovery[recovery.module_size.eq(500)]
    assert primary.groupby(["reference_definition", "accession", "method"]).size().eq(1).all()
    audit = pd.read_csv(RESULTS / "evaluation/published_drug_recovery.csv")
    assert len(audit) == 15 * 2
    summary = {"valid": True, "cohort_checks": checks,
               "disease_gene_rows": len(recovery), "published_drug_audit_rows": len(audit),
               "phase2_rerun": False}
    (RESULTS / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
