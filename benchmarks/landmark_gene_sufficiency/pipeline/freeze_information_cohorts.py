#!/usr/bin/env python3
"""Freeze species-specific discovery cohorts disjoint from final evaluation."""

from __future__ import annotations

import hashlib
import json
import pandas as pd

from common import CONFIG, REPO_ROOT, RESULTS

MANIFEST = REPO_ROOT / "data/manifests/sample_manifest.parquet"


def digest(values: pd.Series) -> str:
    return hashlib.sha256("".join(f"{x}\n" for x in values.astype(str)).encode()).hexdigest()


def main() -> None:
    existing = pd.read_parquet(RESULTS / "cohort_manifest.parquet")
    evaluation = existing.loc[existing.role.eq("evaluation")].copy()
    manifest = pd.read_parquet(MANIFEST)
    strict = manifest.loc[manifest["split"].eq("unseen")
        & manifest.study_exposure.eq("unseen_study")
        & manifest.mapping_status.eq("mapped_single")].copy()
    eval_gses = set(evaluation.gse_candidates_str.dropna())
    count = int(CONFIG["information_density_final"]["discovery_samples_per_species"])
    cap, seed = int(CONFIG["discovery_max_samples_per_gse"]), int(CONFIG["benchmark_seed"])
    discoveries = []
    for offset, species in enumerate(("human", "mouse")):
        pool = strict.loc[strict.species.eq(species)
                          & ~strict.gse_candidates_str.isin(eval_gses)].copy()
        capped = pd.concat([group.sample(min(len(group), cap), random_state=seed + offset)
                            for _, group in pool.groupby("gse_candidates_str", sort=True)], ignore_index=True)
        if len(capped) < count: raise ValueError(f"Only {len(capped):,} eligible {species} samples")
        chosen = capped.sample(count, random_state=seed + offset).sort_values("sample_id").copy()
        chosen["role"] = "information_discovery"
        discoveries.append(chosen)
    discovery = pd.concat(discoveries, ignore_index=True)
    if set(discovery.sample_id) & set(evaluation.sample_id): raise AssertionError("Sample leakage")
    if set(discovery.gse_candidates_str) & eval_gses: raise AssertionError("Study leakage")
    discovery.to_parquet(RESULTS / "information_discovery_cohort.parquet", index=False)
    provenance = {
        "criteria": {"split": "unseen", "study_exposure": "unseen_study", "mapping_status": "mapped_single"},
        "evaluation_samples": len(evaluation), "sample_overlap": 0, "study_overlap": 0,
        "discovery": {species: {"samples": len(frame), "studies": frame.gse_candidates_str.nunique(),
            "ordered_sample_sha256": digest(frame.sample_id)} for species, frame in discovery.groupby("species")},
    }
    (RESULTS / "information_discovery_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__": main()
