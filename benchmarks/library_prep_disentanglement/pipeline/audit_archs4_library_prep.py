#!/usr/bin/env python3
"""Audit ARCHS4 metadata for explicit PolyA/rRNA-depletion evidence.

This script is deliberately conservative: ``total RNA`` is not considered
proof of rRNA depletion. Source files are read-only; compact audit tables are
written under the benchmark results directory.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BENCH = Path(__file__).resolve().parents[1]
POLYA = re.compile(r"poly[\s()_-]*a|polyadenylated|mrna enrichment|mrna selection", re.I)
RIBO = re.compile(r"ribo[\s_-]*(?:zero|minus)|rrna[\s_-]*(?:deplet|remov)|ribosomal rna depletion|globin[\s_-]*zero", re.I)
TEXT_COLUMNS = ["title", "source_name_ch1", "characteristics_ch1", "molecule_ch1"]


def audit(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    metadata = pd.read_parquet(path)
    required = {"gsm", "series_id", *TEXT_COLUMNS}
    missing = required - set(metadata.columns)
    if missing: raise ValueError(f"ARCHS4 metadata missing required columns: {sorted(missing)}")
    polya_masks = {c: metadata[c].fillna("").astype(str).str.contains(POLYA) for c in TEXT_COLUMNS}
    ribo_masks = {c: metadata[c].fillna("").astype(str).str.contains(RIBO) for c in TEXT_COLUMNS}
    has_polya = pd.concat(polya_masks, axis=1).any(axis=1)
    has_ribo = pd.concat(ribo_masks, axis=1).any(axis=1)
    labeled = metadata.copy()
    labeled["library_label"] = "unlabeled"
    labeled.loc[has_polya & ~has_ribo, "library_label"] = "polyA"
    labeled.loc[has_ribo & ~has_polya, "library_label"] = "ribo"
    labeled.loc[has_polya & has_ribo, "library_label"] = "conflict"
    labeled["polya_evidence_columns"] = [";".join(c for c in TEXT_COLUMNS if polya_masks[c].iat[i]) for i in range(len(metadata))]
    labeled["ribo_evidence_columns"] = [";".join(c for c in TEXT_COLUMNS if ribo_masks[c].iat[i]) for i in range(len(metadata))]
    compact = labeled[labeled.library_label != "unlabeled"].copy()
    compact["species"] = compact.organism_ch1.astype(str).str.lower().map(
        lambda x: "human" if "homo sapiens" in x else ("mouse" if "mus musculus" in x else "other")
    )
    by_label = (compact.groupby(["library_label", "species"], dropna=False)
                .agg(samples=("gsm", "nunique"), studies=("series_id", "nunique"))
                .reset_index())
    study_labels = compact.groupby("series_id").library_label.agg(set)
    dual_studies = set(study_labels[study_labels.map(lambda x: {"polyA", "ribo"}.issubset(x))].index)
    potential = compact[compact.series_id.isin(dual_studies)].copy()
    summary = {
        "metadata_rows": int(len(metadata)),
        "explicitly_labeled_samples": int(compact.gsm.nunique()),
        "explicit_polya_samples": int(compact.loc[compact.library_label == "polyA", "gsm"].nunique()),
        "explicit_ribo_samples": int(compact.loc[compact.library_label == "ribo", "gsm"].nunique()),
        "conflicting_samples": int(compact.loc[compact.library_label == "conflict", "gsm"].nunique()),
        "studies_with_both_labels": int(len(dual_studies)),
        "verified_same_rna_pairs": 0,
        "classification": "OBSERVATIONAL",
        "reason": "Explicit labels exist, but ARCHS4 metadata do not encode authoritative same-RNA pair identifiers; total RNA alone was not called rRNA-depleted.",
    }
    return by_label, potential, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=ROOT / "data/manifests/archs4_sample_metadata_v2.5.parquet")
    parser.add_argument("--output", type=Path, default=BENCH / "results/task4a_data_audit")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    by_label, potential, summary = audit(args.metadata)
    by_label.to_csv(args.output / "archs4_explicit_label_summary.csv", index=False)
    potential.to_parquet(args.output / "archs4_potential_dual_protocol_studies.parquet", index=False)
    (args.output / "archs4_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n=== ARCHS4 library-preparation audit ===")
    print(by_label.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
