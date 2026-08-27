#!/usr/bin/env python3
"""Read-only QA summary for the downstream sample manifest.

The script never modifies the manifest. It reports schema problems, split and
exposure counts, identifier overlap, mapping consistency, and the strict external
evaluation pool.

Examples
--------
    python scripts/data_audit/inspect_manifest.py
    python scripts/data_audit/inspect_manifest.py --examples 10
    python scripts/data_audit/inspect_manifest.py --manifest path/to/manifest.parquet
"""

from __future__ import annotations

import argparse
import re
from itertools import combinations
from pathlib import Path
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "data" / "manifests" / "sample_manifest.parquet"
EXPECTED_COLUMNS = {
    "sample_id", "gsm", "split", "study_exposure", "mapping_status",
}
EXPECTED_SPLITS = {"train", "val", "unseen"}
MAPPED_STATUSES = {"mapped_single", "mapped_multiple"}
GSE_RE = re.compile(r"GSE\d+", re.IGNORECASE)


def heading(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def load_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {path}")
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        raise RuntimeError(f"Could not read manifest {path}: {exc}") from exc


def validate_schema(manifest: pd.DataFrame, path: Path) -> set[str]:
    heading("MANIFEST SCHEMA")
    print(f"Path:    {path}")
    print(f"Shape:   {len(manifest):,} rows x {len(manifest.columns):,} columns")
    print(f"Columns: {list(manifest.columns)}")
    print("\nData types:")
    print(manifest.dtypes.to_string())

    missing = EXPECTED_COLUMNS - set(manifest.columns)
    if missing:
        print(f"\nERROR: missing required manifest columns: {sorted(missing)}")
        print("The strict external-evaluation QA cannot run until these fields exist.")
    else:
        print("\nRequired manifest columns are present.")
    return missing


def print_counts(manifest: pd.DataFrame, column: str) -> None:
    heading(f"COUNTS BY {column.upper()}")
    if column not in manifest.columns:
        print(f"Unavailable: column '{column}' is missing.")
        return
    counts = manifest[column].astype("string").fillna("<missing>").value_counts(dropna=False)
    print(counts.to_string())


def _gse_candidates(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        text = ";".join(str(item) for item in value)
    elif hasattr(value, "tolist") and not isinstance(value, str):
        items = value.tolist()
        text = ";".join(str(item) for item in items)
    elif pd.isna(value):
        return []
    else:
        text = str(value)
    return sorted({match.upper() for match in GSE_RE.findall(text)})


def gse_sets_by_row(manifest: pd.DataFrame) -> pd.Series:
    if "gse_candidates" in manifest.columns:
        return manifest["gse_candidates"].map(_gse_candidates)
    if "gse_candidates_str" in manifest.columns:
        return manifest["gse_candidates_str"].map(_gse_candidates)
    if "gse" in manifest.columns:
        return manifest["gse"].map(_gse_candidates)
    return pd.Series([[] for _ in range(len(manifest))], index=manifest.index)


def summarize_identifiers(manifest: pd.DataFrame, row_gses: pd.Series) -> None:
    heading("IDENTIFIER AND MAPPING SUMMARY")
    gsm = manifest["gsm"].astype("string").str.upper() if "gsm" in manifest else None
    unique_gsms = gsm.dropna().nunique() if gsm is not None else 0
    unique_gses = len({gse for candidates in row_gses for gse in candidates})
    print(f"Rows:                          {len(manifest):,}")
    print(f"Unique GSMs:                   {unique_gsms:,}")
    print(f"Unique GSEs:                   {unique_gses:,}")
    if gsm is not None:
        print(f"Rows without GSM:              {gsm.isna().sum():,}")
        print(f"Duplicate GSM rows:            {gsm.duplicated(keep=False).sum():,}")
    print(f"Rows without a GSE candidate:  {row_gses.map(len).eq(0).sum():,}")

    if "mapping_status" in manifest:
        statuses = manifest["mapping_status"].astype("string")
        print(f"Unresolved mappings:           {statuses.eq('unresolved').sum():,}")
        print(f"Multiple-GSE mappings:         {statuses.eq('mapped_multiple').sum():,}")
    else:
        print("Mapping-status summaries unavailable: 'mapping_status' is missing.")


def report_species_breakdown(manifest: pd.DataFrame) -> None:
    heading("HUMAN / MOUSE COMPOSITION")
    if "species" not in manifest.columns:
        print("Unavailable: column 'species' is missing.")
        return
    species = manifest["species"].astype("string").fillna("<missing>")
    print("Overall:")
    print(species.value_counts().to_string())
    if "split" in manifest.columns:
        print("\nBy split:")
        print(pd.crosstab(manifest["split"], species, margins=True).to_string())
    if "study_exposure" in manifest.columns:
        print("\nBy study exposure:")
        print(pd.crosstab(manifest["study_exposure"], species, margins=True).to_string())


def report_benchmark_cohorts(manifest: pd.DataFrame, row_gses: pd.Series) -> None:
    """Summarize benchmark pools by exposure, species, and mapping ambiguity."""
    heading("BENCHMARK COHORTS")
    required = {"split", "study_exposure", "mapping_status"}
    missing = required - set(manifest.columns)
    if missing:
        print(f"Unavailable: missing columns {sorted(missing)}")
        return

    split = manifest["split"].astype("string")
    exposure = manifest["study_exposure"].astype("string")
    status = manifest["mapping_status"].astype("string")
    mapped = status.isin(MAPPED_STATUSES)
    masks = {
        "train_reference": split.eq("train"),
        "validation_seen_study": split.eq("val") & exposure.eq("seen_study"),
        "validation_unseen_study": split.eq("val") & exposure.eq("unseen_study"),
        "unseen_sample_seen_study": split.eq("unseen") & exposure.eq("seen_study"),
        "unseen_sample_unseen_study": (
            split.eq("unseen") & exposure.eq("unseen_study") & mapped
        ),
        "strict_unseen_single_gse": (
            split.eq("unseen") & exposure.eq("unseen_study")
            & status.eq("mapped_single")
        ),
        "strict_unseen_multiple_gse": (
            split.eq("unseen") & exposure.eq("unseen_study")
            & status.eq("mapped_multiple")
        ),
    }

    species = (
        manifest["species"].astype("string").str.lower()
        if "species" in manifest.columns
        else pd.Series(pd.NA, index=manifest.index, dtype="string")
    )
    rows = []
    for name, mask in masks.items():
        cohort_gses = {gse for values in row_gses.loc[mask] for gse in values}
        rows.append({
            "cohort": name,
            "samples": int(mask.sum()),
            "human": int((mask & species.eq("human")).sum()),
            "mouse": int((mask & species.eq("mouse")).sum()),
            "other/missing": int((mask & ~species.isin(["human", "mouse"])).sum()),
            "candidate_GSEs": len(cohort_gses),
            "mapped_single": int((mask & status.eq("mapped_single")).sum()),
            "mapped_multiple": int((mask & status.eq("mapped_multiple")).sum()),
            "unresolved": int((mask & status.eq("unresolved")).sum()),
        })
    summary = pd.DataFrame(rows).set_index("cohort")
    print(summary.to_string())
    print("\nRecommended primary benchmark: strict_unseen_single_gse")
    print("Comparison cohort: unseen_sample_seen_study")


def values_by_split(
    manifest: pd.DataFrame, value_column: str, row_gses: pd.Series
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    if "split" not in manifest:
        return result
    split_values = manifest["split"].astype("string")
    for split in ("train", "val", "unseen"):
        mask = split_values.eq(split)
        if value_column == "gse":
            result[split] = {
                gse for candidates in row_gses.loc[mask] for gse in candidates
            }
        elif value_column in manifest:
            values = manifest.loc[mask, value_column].astype("string").dropna()
            result[split] = set(values.str.upper())
        else:
            result[split] = set()
    return result


def summarize_overlap(manifest: pd.DataFrame, row_gses: pd.Series) -> None:
    heading("TRAIN / VAL / UNSEEN OVERLAP")
    if "split" not in manifest:
        print("Unavailable: column 'split' is missing.")
        return
    observed = set(manifest["split"].dropna().astype(str))
    unexpected = observed - EXPECTED_SPLITS
    if unexpected:
        print(f"WARNING: unexpected split labels: {sorted(unexpected)}")

    for identifier in ("gsm", "gse"):
        groups = values_by_split(manifest, identifier, row_gses)
        print(f"\n{identifier.upper()} overlap:")
        for left, right in combinations(("train", "val", "unseen"), 2):
            overlap = groups[left] & groups[right]
            print(f"  {left} / {right}: {len(overlap):,}")
            if overlap:
                print(f"    examples: {sorted(overlap)[:5]}")


def strict_external_pool(manifest: pd.DataFrame) -> pd.DataFrame:
    required = {"split", "study_exposure", "mapping_status"}
    if not required.issubset(manifest.columns):
        return manifest.iloc[0:0]
    mask = (
        manifest["split"].astype("string").eq("unseen")
        & manifest["study_exposure"].astype("string").eq("unseen_study")
        & manifest["mapping_status"].astype("string").isin(MAPPED_STATUSES)
    )
    return manifest.loc[mask]


def report_strict_pool(manifest: pd.DataFrame, examples: int) -> None:
    heading("STRICT EXTERNAL EVALUATION POOL")
    required = {"split", "study_exposure", "mapping_status"}
    missing = required - set(manifest.columns)
    if missing:
        print(f"Unavailable: missing columns {sorted(missing)}")
        return
    unseen_count = manifest["split"].astype("string").eq("unseen").sum()
    pool = strict_external_pool(manifest)
    print(f"All unseen samples:                  {unseen_count:,}")
    print(f"Mapped samples from unseen studies: {len(pool):,}")
    percentage = 100.0 * len(pool) / unseen_count if unseen_count else 0.0
    print(f"Share of unseen samples:             {percentage:.2f}%")
    if not pool.empty:
        print(f"\nFirst {min(examples, len(pool))} eligible rows:")
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(pool.head(examples).to_string(index=False))


def suspicious_rows(manifest: pd.DataFrame, row_gses: pd.Series) -> dict[str, pd.DataFrame]:
    issues: dict[str, pd.DataFrame] = {}
    empty = manifest.iloc[0:0]
    if "sample_id" in manifest:
        sample_ids = manifest["sample_id"].astype("string").str.strip()
        issues["missing sample_id"] = manifest[sample_ids.isna() | sample_ids.eq("")]
        issues["duplicate sample_id"] = manifest[sample_ids.notna() & sample_ids.duplicated(False)]
    if "gsm" in manifest:
        gsm = manifest["gsm"].astype("string").str.strip()
        issues["missing GSM"] = manifest[gsm.isna() | gsm.eq("")]
    if "mapping_status" in manifest:
        status = manifest["mapping_status"].astype("string")
        has_gse = row_gses.map(bool)
        issues["mapped status without GSE"] = manifest[status.isin(MAPPED_STATUSES) & ~has_gse]
        issues["unresolved status with GSE"] = manifest[status.eq("unresolved") & has_gse]
        issues["mapped_single with non-single GSE count"] = manifest[
            status.eq("mapped_single") & row_gses.map(len).ne(1)
        ]
        issues["mapped_multiple with fewer than two GSEs"] = manifest[
            status.eq("mapped_multiple") & row_gses.map(len).lt(2)
        ]
    if {"split", "study_exposure"}.issubset(manifest.columns):
        split = manifest["split"].astype("string")
        exposure = manifest["study_exposure"].astype("string")
        train_gses = {
            gse for candidates in row_gses.loc[split.eq("train")] for gse in candidates
        }
        intersects_train = row_gses.map(lambda values: bool(set(values) & train_gses))
        issues["unseen_study intersects training GSEs"] = manifest[
            exposure.eq("unseen_study") & intersects_train
        ]
        issues["seen_study has no training GSE match"] = manifest[
            exposure.eq("seen_study") & ~intersects_train
        ]
    return issues or {"no checks available": empty}


def report_suspicious_rows(
    manifest: pd.DataFrame, row_gses: pd.Series, examples: int
) -> None:
    heading("SUSPICIOUS ROWS")
    issues = suspicious_rows(manifest, row_gses)
    total_flagged_indices: set[object] = set()
    for label, rows in issues.items():
        total_flagged_indices.update(rows.index)
        print(f"{label}: {len(rows):,}")
        if not rows.empty:
            with pd.option_context("display.max_columns", None, "display.width", 200):
                print(rows.head(examples).to_string(index=False))
    print(f"\nDistinct rows flagged by any check: {len(total_flagged_indices):,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help=f"Manifest parquet (default: {DEFAULT_MANIFEST})")
    parser.add_argument("--examples", type=int, default=5,
                        help="Example rows per section (default: 5).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = args.manifest.expanduser().resolve()
    try:
        manifest = load_manifest(path)
    except (FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(f"Error: {exc}") from exc

    validate_schema(manifest, path)
    row_gses = gse_sets_by_row(manifest)
    for column in ("split", "study_exposure", "mapping_status"):
        print_counts(manifest, column)
    summarize_identifiers(manifest, row_gses)
    report_species_breakdown(manifest)
    report_benchmark_cohorts(manifest, row_gses)
    summarize_overlap(manifest, row_gses)
    report_strict_pool(manifest, args.examples)
    report_suspicious_rows(manifest, row_gses, args.examples)


if __name__ == "__main__":
    main()
