#!/usr/bin/env python3
"""Audit whether Task 3C BridgeRNA geometry clusters track available metadata."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
RESULTS = HERE / "results"
RAW_METADATA = ROOT / "data/osdr/metadata/selected_sample_metadata.tsv"
N_PERMUTATIONS = 9999

RAW_VARIABLES = {
    "study.characteristics.animal source": "animal_source",
    "study.characteristics.age at launch": "age_at_launch_raw",
    "study.characteristics.sex": "sex_raw",
    "study.characteristics.material type": "material_raw",
    "study.factor value.genotype": "genotype",
    "study.parameter value.habitat": "habitat",
    "study.parameter value.duration": "flight_duration_raw",
    "study.parameter value.light cycle": "light_cycle",
    "study.parameter value.diet": "diet",
    "study.parameter value.feeding schedule": "feeding_schedule",
    "study.parameter value.enrichment material": "enrichment_material",
    "study.parameter value.euthanasia method": "euthanasia_method",
    "study.parameter value.age at euthanasia": "age_at_euthanasia",
    "study.parameter value.carcass preservation method": "carcass_preservation_raw",
    "study.parameter value.sample weight": "sample_weight",
    "study.parameter value.sample preservation method": "sample_preservation_raw",
    "study.parameter value.sample storage temperature": "sample_storage_temperature",
    "study.comment.euthanasia date": "euthanasia_date",
    "study.comment.bsp dissection date": "dissection_date",
    "study.comment.source description": "source_description",
}
CORE_VARIABLES = [
    "OSD", "mission", "rr_experiment", "strain", "flight_duration", "sex", "age_at_launch",
    "material", "library_preparation", "sequencing_parameters", "sequencing_facility", "habitat",
    "animal_source", "light_cycle", "diet", "feeding_schedule", "enrichment_material",
    "euthanasia_method", "age_at_euthanasia", "preservation", "carcass_preservation", "sample_preservation",
    "sample_storage_temperature", "sample_weight", "euthanasia_date", "dissection_date",
    "genotype", "source_description",
]


def clean(value: object) -> str:
    if pd.isna(value) or str(value).strip().lower() in {"", "nan", "{not available}", "not_reported"}:
        return "not_reported"
    return str(value).strip()


def signature(frame: pd.DataFrame, column: str) -> str:
    """Preserve condition-specific values instead of hiding within-contrast variation."""
    condition_values = {}
    for condition in ("GC", "FLT"):
        values = sorted({clean(x) for x in frame.loc[frame.condition == condition, column]})
        condition_values[condition] = values or ["not_reported"]
    if condition_values["GC"] == condition_values["FLT"] and len(condition_values["GC"]) == 1:
        return condition_values["GC"][0]
    return "GC[" + "; ".join(condition_values["GC"]) + "] | FLT[" + "; ".join(condition_values["FLT"]) + "]"


def build_contrast_metadata() -> pd.DataFrame:
    clusters = pd.read_csv(RESULTS / "task3c_cluster_assignments.csv")
    contrasts = pd.read_csv(RESULTS / "task3b_contrast_summary.csv")
    members = pd.read_csv(RESULTS / "task3b_contrast_sample_membership.csv")
    raw = pd.read_csv(RAW_METADATA, sep="\t", low_memory=False).rename(
        columns={"id.accession": "OSD", "id.sample name": "sample_id", **RAW_VARIABLES})
    wanted = ["OSD", "sample_id", *RAW_VARIABLES.values()]
    samples = members.merge(raw[wanted], on=["OSD", "sample_id"], how="left", validate="one_to_one")
    if samples["animal_source"].isna().all():
        raise RuntimeError("Raw OSDR metadata failed to join to Task 3B samples")
    records = []
    for contrast_id, frame in samples.groupby("contrast_id", sort=False):
        record = {"contrast_id": contrast_id}
        for output in RAW_VARIABLES.values():
            record[output] = signature(frame, output)
        records.append(record)
    raw_contrast = pd.DataFrame(records)
    out = contrasts.merge(raw_contrast, on="contrast_id", validate="one_to_one")
    out = out.merge(clusters[["contrast_id", "geometry_cluster", "heatmap_order"]],
                    on="contrast_id", validate="one_to_one")
    out["sex"] = out.sex.astype(str).str.strip().str.lower().replace("nan", "not_reported")
    out["rr_experiment"] = out.mission.str.replace("_NASA", "", regex=False).str.replace("_CASIS", "", regex=False)
    # The compact Task 3B preservation fields are authoritative technical labels.
    out["carcass_preservation"] = out.carcass_preservation.map(clean)
    out["sample_preservation"] = out.sample_preservation.map(clean)
    return out.sort_values(["geometry_cluster", "heatmap_order"]).reset_index(drop=True)


def cramers_v(table: pd.DataFrame) -> float:
    if min(table.shape) < 2:
        return 0.0
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    return float(np.sqrt(chi2 / (n * min(table.shape[0] - 1, table.shape[1] - 1))))


def association(values: pd.Series, clusters: pd.Series, rng: np.random.Generator) -> dict[str, object]:
    values = values.map(clean)
    table = pd.crosstab(values, clusters)
    observed = cramers_v(table)
    null = np.empty(N_PERMUTATIONS)
    value_codes, categories = pd.factorize(values, sort=True)
    cluster_codes = pd.Categorical(clusters).codes

    def fast_v(permuted: np.ndarray) -> float:
        contingency = np.zeros((len(categories), 2), dtype=float)
        np.add.at(contingency, (value_codes, permuted), 1)
        expected = contingency.sum(axis=1, keepdims=True) @ (contingency.sum(axis=0, keepdims=True) / len(values))
        valid = expected > 0
        chi2 = np.sum(((contingency - expected) ** 2)[valid] / expected[valid])
        return float(np.sqrt(chi2 / len(values)))

    for index in range(N_PERMUTATIONS):
        null[index] = fast_v(rng.permutation(cluster_codes))
    pvalue = (np.count_nonzero(null >= observed - 1e-12) + 1) / (N_PERMUTATIONS + 1)
    counts = values.value_counts()
    purity = sum(pd.crosstab(values, clusters).max(axis=1)) / len(values)
    missing = values.eq("not_reported").sum()
    singleton_fraction = counts.eq(1).sum() / len(counts)
    sparse = (missing / len(values) >= .35 or len(counts) > len(values) / 2 or
              singleton_fraction >= .5 or counts.min() < 2)
    perfect_partition = (table.astype(bool).sum(axis=0).eq(1).all() and
                         table.astype(bool).sum(axis=1).eq(1).all())
    if table.shape[0] == 1:
        interpretation = "constant; not evaluable"
    elif sparse:
        interpretation = "sparse/confounded; unreliable"
    elif perfect_partition:
        interpretation = "perfect partition"
    elif pvalue < .05:
        interpretation = "partial association"
    elif pvalue < .10 or observed >= .5:
        interpretation = "possible partial alignment"
    else:
        interpretation = "no clear alignment"
    return {"n_levels": len(counts), "missing_contrasts": int(missing),
            "smallest_level_n": int(counts.min()), "singleton_level_fraction": singleton_fraction,
            "cramers_v": observed, "permutation_pvalue": pvalue,
            "category_to_cluster_purity": purity, "perfect_partition": perfect_partition,
            "reliability_flag": "sparse_or_confounded" if sparse else "usable_exploratory",
            "interpretation": interpretation}


def main() -> None:
    metadata = build_contrast_metadata()
    missing = [x for x in CORE_VARIABLES if x not in metadata]
    if missing:
        raise RuntimeError(f"Requested audit variables are absent: {missing}")
    rng = np.random.default_rng(3407)
    rows = []
    for variable in CORE_VARIABLES:
        rows.append({"variable": variable, **association(metadata[variable], metadata.geometry_cluster, rng)})
    audit = pd.DataFrame(rows).sort_values(["reliability_flag", "permutation_pvalue", "cramers_v"],
                                           ascending=[False, True, False])
    compact = metadata[["geometry_cluster", "contrast_id", *CORE_VARIABLES, "n_FLT", "n_GC"]]
    cluster_summary = []
    for variable in CORE_VARIABLES:
        row = {"variable": variable}
        for cluster in (1, 2):
            counts = metadata.loc[metadata.geometry_cluster == cluster, variable].map(clean).value_counts()
            row[f"cluster_{cluster}_values_n"] = "; ".join(f"{value} (n={count})" for value, count in counts.items())
        cluster_summary.append(row)
    compact.to_csv(RESULTS / "task3c_cluster_metadata_table.csv", index=False)
    pd.DataFrame(cluster_summary).to_csv(RESULTS / "task3c_cluster_metadata_compact.csv", index=False)
    audit.to_csv(RESULTS / "task3c_cluster_metadata_associations.csv", index=False)
    print("Task 3C metadata/confounding audit complete")
    print(f"Contrasts: {len(metadata)}; permutation replicates per variable: {N_PERMUTATIONS}")
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
