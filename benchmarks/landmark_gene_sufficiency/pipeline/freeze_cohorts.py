#!/usr/bin/env python3
"""Freeze study-disjoint human discovery and strict-unseen evaluation cohorts."""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from common import CONFIG, REPO_ROOT, RESULTS


MANIFEST = REPO_ROOT / "data/manifests/sample_manifest.parquet"
REFERENCE_EVAL = (REPO_ROOT / "benchmarks/tcga_imputation/results/"
                  "archs4_holdout_selected_samples.parquet")


def id_hash(values: pd.Series) -> str:
    return hashlib.sha256("".join(f"{value}\n" for value in values.astype(str)).encode()).hexdigest()


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_parquet(MANIFEST)
    strict = manifest.loc[
        manifest["split"].eq("unseen")
        & manifest.study_exposure.eq("unseen_study")
        & manifest.mapping_status.eq("mapped_single")
    ].copy()
    evaluation = pd.read_parquet(REFERENCE_EVAL).copy()
    required = strict[["sample_id", "gse_candidates_str", "species"]]
    check = evaluation.merge(required, on=["sample_id", "gse_candidates_str", "species"],
                             how="left", indicator=True, validate="one_to_one")
    if not check._merge.eq("both").all():
        raise ValueError("Reference evaluation cohort no longer satisfies strict-unseen criteria")
    evaluation["role"] = "evaluation"

    evaluation_gses = set(evaluation.gse_candidates_str.dropna())
    discovery_pool = strict.loc[
        strict.species.eq("human") & ~strict.gse_candidates_str.isin(evaluation_gses)
    ].copy()
    cap = int(CONFIG["discovery_max_samples_per_gse"])
    seed = int(CONFIG["benchmark_seed"])
    capped = pd.concat(
        [group.sample(min(len(group), cap), random_state=seed)
         for _, group in discovery_pool.groupby("gse_candidates_str", sort=True)],
        ignore_index=True,
    )
    discovery = capped.sample(n=int(CONFIG["discovery_samples"]), random_state=seed).copy()
    discovery["role"] = "discovery"
    if set(discovery.gse_candidates_str) & evaluation_gses:
        raise AssertionError("Discovery and evaluation GSEs overlap")
    if set(discovery.sample_id) & set(evaluation.sample_id):
        raise AssertionError("Discovery and evaluation samples overlap")

    columns = sorted(set(discovery.columns) | set(evaluation.columns))
    cohorts = pd.concat([discovery.reindex(columns=columns), evaluation.reindex(columns=columns)],
                        ignore_index=True)
    cohorts.to_parquet(RESULTS / "cohort_manifest.parquet", index=False)
    summary = {
        "source_manifest": str(MANIFEST), "benchmark_seed": seed,
        "criteria": {"split": "unseen", "study_exposure": "unseen_study",
                     "mapping_status": "mapped_single"},
        "discovery": {"species": "human", "samples": len(discovery),
                      "studies": discovery.gse_candidates_str.nunique(),
                      "max_samples_per_gse": cap,
                      "ordered_sample_sha256": id_hash(discovery.sample_id)},
        "evaluation": {
            species: {"samples": len(frame), "studies": frame.gse_candidates_str.nunique(),
                      "ordered_sample_sha256": id_hash(frame.sample_id)}
            for species, frame in evaluation.groupby("species")
        },
        "discovery_evaluation_sample_overlap": 0,
        "discovery_evaluation_gse_overlap": 0,
    }
    (RESULTS / "cohort_provenance.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
