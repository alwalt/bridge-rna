#!/usr/bin/env python3
"""Freeze the eligible matched FLT/GC OSDR cohort before any model inference."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import RESULTS, save_json, sha256

REPO = Path("/home/walt/bridge-rna")
SOURCE = REPO / "data/osdr/metadata/selected_sample_metadata.tsv"
OUT = RESULTS / "osdr_atlas"
GROUP_COLUMNS = [
    "study", "organism", "tissue", "strain", "sex", "age_at_launch", "genotype"
]


def normalized(value: object) -> str:
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def norm_text(series: pd.Series, missing: str = "unspecified") -> pd.Series:
    return series.fillna(missing).astype(str).str.strip().str.lower().replace("", missing)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(SOURCE, sep="\t")
    raw["source_row"] = np.arange(len(raw))
    audits = []
    raw["exclusion_reason"] = ""
    raw.loc[raw.has_single_cell_rna_sequencing.ne(0), "exclusion_reason"] = "single_cell_rna_seq"
    raw = raw.loc[raw.has_single_cell_rna_sequencing.eq(0)].copy()
    raw = raw.drop_duplicates(["counts_path", "id.sample name"], keep="first")

    matched = []
    for counts_path, group in raw.groupby("counts_path", sort=True):
        path = REPO / counts_path
        if not path.exists():
            audits.append({"counts_path": counts_path, "metadata_rows": len(group), "matched_rows": 0,
                           "status": "counts_file_missing"})
            continue
        header = pd.read_csv(path, nrows=0).columns[1:]
        by_norm: dict[str, list[str]] = {}
        for column in header:
            by_norm.setdefault(normalized(column), []).append(str(column))
        group = group.copy()
        group["counts_column"] = group["id.sample name"].map(
            lambda value: by_norm.get(normalized(value), [None])[0]
            if len(by_norm.get(normalized(value), [])) == 1 else None)
        audits.append({"counts_path": counts_path, "metadata_rows": len(group),
                       "matched_rows": int(group.counts_column.notna().sum()), "status": "available"})
        matched.append(group)
    candidates = pd.concat(matched, ignore_index=True)
    candidates["condition_original"] = candidates["study.factor value.spaceflight"]
    condition = norm_text(candidates["condition_original"], "missing")
    candidates["condition"] = np.select(
        [condition.eq("space flight"), condition.isin(["ground control", "ground control rerun"])],
        ["FLT", "GC"], default="OTHER")
    candidates["study"] = candidates["id.accession"].astype(str).str.upper()
    # The local OSDR export has no populated mission/project field. Preserve the
    # accession as the authoritative experiment stratum rather than guessing a mission.
    candidates["mission_or_experiment"] = candidates["study"]
    candidates["organism"] = norm_text(candidates["study.characteristics.organism"])
    candidates["tissue"] = norm_text(candidates["study.characteristics.material type"])
    candidates["strain"] = norm_text(candidates["study.characteristics.strain"])
    candidates["sex"] = norm_text(candidates["study.characteristics.sex"])
    candidates["age_at_launch"] = norm_text(candidates["study.characteristics.age at launch"])
    candidates["genotype"] = norm_text(candidates["study.factor value.genotype"])
    candidates["sample_id"] = candidates["id.sample name"].astype(str)
    candidates["counts_file_exists"] = candidates.counts_path.map(lambda p: (REPO / p).exists())
    candidates["unique_count_column_match"] = candidates.counts_column.notna()

    pairable = candidates.loc[candidates.condition.isin(["FLT", "GC"]) & candidates.unique_count_column_match].copy()
    counts = pairable.groupby(GROUP_COLUMNS + ["condition"]).size().unstack(fill_value=0)
    for value in ("FLT", "GC"):
        if value not in counts:
            counts[value] = 0
    eligible_groups = counts.loc[(counts.FLT > 0) & (counts.GC > 0)].reset_index()
    eligible_groups["mission_or_experiment"] = eligible_groups["study"]
    eligible_groups["group_id"] = [f"OSDRG{i:03d}" for i in range(1, len(eligible_groups) + 1)]
    eligible_groups["replicate_consensus_eligible"] = (eligible_groups.FLT >= 2) & (eligible_groups.GC >= 2)
    eligible = pairable.merge(eligible_groups[GROUP_COLUMNS + ["group_id", "replicate_consensus_eligible"]],
                              on=GROUP_COLUMNS, how="inner", validate="many_to_one")
    eligible = eligible.sort_values(["group_id", "condition", "sample_id"]).reset_index(drop=True)
    eligible["mission_or_experiment"] = eligible["study"]
    eligible["matrix_row"] = np.arange(len(eligible))
    eligible["biological_replicate"] = eligible.groupby(["group_id", "condition"]).cumcount() + 1

    candidates["eligibility"] = "excluded_non_flt_gc"
    candidates.loc[candidates.condition.isin(["FLT", "GC"]), "eligibility"] = "excluded_no_matched_opposite_condition"
    candidates.loc[~candidates.unique_count_column_match, "eligibility"] = "excluded_no_unique_count_column_match"
    key = set(zip(eligible.counts_path, eligible.counts_column))
    candidates.loc[[tuple(x) in key for x in zip(candidates.counts_path, candidates.counts_column)], "eligibility"] = "eligible"

    eligible.to_parquet(OUT / "eligible_sample_manifest.parquet", index=False)
    eligible.to_csv(OUT / "eligible_sample_manifest.csv", index=False)
    eligible_groups.to_csv(OUT / "eligible_group_manifest.csv", index=False)
    candidates[["source_row", "id.accession", "id.sample name", "counts_path", "counts_column",
                "condition_original", "condition", "study.characteristics.material type", "eligibility"]].to_csv(
                    OUT / "sample_eligibility_audit.csv", index=False)
    pd.DataFrame(audits).to_csv(OUT / "counts_file_compatibility_audit.csv", index=False)

    old_manifest = pd.read_csv(REPO / "benchmarks/osdr_batch_effect_representation/results/sample_manifest.csv")
    old_embeddings = np.load(REPO / "benchmarks/osdr_batch_effect_representation/work/bridgerna_embeddings.npy", mmap_mode="r")
    old_keys = set(zip(old_manifest.OSD.astype(str).str.upper(), old_manifest.sample_id.astype(str)))
    reusable = sum((study, sample) in old_keys for study, sample in zip(eligible.study, eligible.sample_id))
    summary = {
        "source_metadata_rows": len(pd.read_csv(SOURCE, sep="\t")),
        "bulk_rna_candidate_rows": len(candidates),
        "eligible_studies": int(eligible.study.nunique()),
        "eligible_samples": len(eligible),
        "eligible_organisms": int(eligible.organism.nunique()),
        "organisms": sorted(eligible.organism.unique().tolist()),
        "eligible_tissues": int(eligible.tissue.nunique()),
        "eligible_matched_groups": int(eligible.group_id.nunique()),
        "replicate_consensus_groups": int(eligible_groups.replicate_consensus_eligible.sum()),
        "flt_samples": int(eligible.condition.eq("FLT").sum()),
        "gc_samples": int(eligible.condition.eq("GC").sum()),
        "cached_compatible_bridge_embeddings": int(reusable),
        "bridge_embeddings_requiring_inference": int(len(eligible) - reusable),
        "cached_embedding_shape": list(old_embeddings.shape),
        "cached_embedding_checkpoint_sha256": "f3491beaa697a0408dc1daeb6f8d648566bd46bafc6e44938d723e7ffc325541",
        "source_metadata_sha256": sha256(SOURCE),
        "eligibility_rule": "bulk RNA-seq; unique one-to-one metadata/count-column match; FLT and GC both present within exact study/organism/tissue/strain/sex/age/genotype stratum",
        "group_hierarchy": "study/experiment -> organism -> tissue -> strain -> sex -> age -> genotype -> FLT/GC -> replicate",
        "mission_field_policy": "Project Identifier is empty in the local export; OSD accession is preserved as experiment and no mission identity is guessed.",
        "note": "The prior 1000-sample landmark benchmark was randomly capped and is not used as the all-eligible cohort."
    }
    save_json(OUT / "pre_inference_census.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
