#!/usr/bin/env python3
"""Inspect foundation-model train/validation parquet sample metadata.

The default input directory is ``data/pretraining``. Use ``--data-dir`` or the
individual path options when the split files live elsewhere. This is a read-only
audit; GSM -> GSE mapping belongs in ``map_gsm_to_gse.py``.

Examples
--------
    python scripts/data_audit/inspect_training_samples.py
    python scripts/data_audit/inspect_training_samples.py \
        --data-dir training/pretraining/sample_split
    python scripts/data_audit/inspect_training_samples.py \
        --train path/to/train.parquet --validation path/to/validation.parquet
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "pretraining"
GSM_SEARCH_PATTERN = r"(?i)GSM\d+"
GSM_EXTRACT_PATTERN = r"(?i)(GSM\d+)"
ID_NAME_HINTS = (
    "sample_id", "gsm_accession", "geo_accession", "accession", "sample",
    "sample_name", "id", "__index_level_0__",
)
METADATA_HINTS = (
    "species", "organism", "tissue", "source", "platform", "dataset",
    "study", "gse", "split", "batch", "condition", "cell_type", "celltype",
)


def heading(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def resolve_split_path(
    explicit_path: Path | None, data_dir: Path, split: str
) -> Path:
    """Resolve a split path from common names, without silently guessing."""
    if explicit_path is not None:
        path = explicit_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{split} parquet does not exist: {path}")
        return path

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Input directory does not exist: {data_dir}\n"
            "Pass --data-dir or explicit --train/--validation paths."
        )

    aliases = ("train",) if split == "train" else ("validation", "valid", "val")
    candidates = [
        path for path in data_dir.rglob("*.parquet")
        if any(re.search(rf"(^|[_-]){alias}([_.-]|$)", path.name.lower())
               for alias in aliases)
    ]
    if len(candidates) == 1:
        return candidates[0].resolve()
    if not candidates:
        raise FileNotFoundError(
            f"No {split} parquet found under {data_dir}. "
            f"Pass --{split} explicitly."
        )
    choices = "\n  ".join(str(path) for path in sorted(candidates))
    raise RuntimeError(
        f"Multiple possible {split} parquets found; select one with --{split}:\n  {choices}"
    )


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a parquet without altering it or its index."""
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        raise RuntimeError(f"Unable to read parquet {path}: {exc}") from exc


def print_structure(name: str, path: Path, frame: pd.DataFrame, examples: int) -> None:
    heading(f"{name.upper()} PARQUET STRUCTURE")
    print(f"Path:    {path}")
    print(f"Shape:   {frame.shape[0]:,} rows x {frame.shape[1]:,} columns")
    print(f"Columns: {list(frame.columns)}")
    print("\nData types:")
    print(frame.dtypes.to_string())
    print(f"\nFirst {min(examples, len(frame))} rows:")
    if frame.empty:
        print("  [empty parquet]")
    else:
        with pd.option_context("display.max_columns", None, "display.width", 180):
            print(frame.head(examples).to_string(index=False))


def _string_values(series: pd.Series, limit: int = 10_000) -> pd.Series:
    values = series.dropna()
    if len(values) > limit:
        values = values.sample(limit, random_state=0)
    return values.astype("string")


def identify_id_columns(frame: pd.DataFrame) -> list[str]:
    """Rank plausible sample/accession columns using names and observed values."""
    scored: list[tuple[float, str]] = []
    for column in frame.columns:
        name = str(column)
        lower = name.lower()
        values = _string_values(frame[column])
        if values.empty:
            gsm_rate = 0.0
            uniqueness = 0.0
        else:
            gsm_rate = float(
                values.str.contains(GSM_SEARCH_PATTERN, regex=True, na=False).mean()
            )
            uniqueness = float(values.nunique(dropna=True) / len(values))

        if lower in ID_NAME_HINTS:
            name_score = 4.0
        elif any(hint in lower for hint in ("accession", "sample", "gsm")):
            name_score = 2.0
        elif lower == "id" or lower.endswith("_id"):
            name_score = 1.0
        else:
            name_score = 0.0

        score = name_score + 5.0 * gsm_rate + 0.25 * uniqueness
        if name_score > 0 or gsm_rate > 0:
            scored.append((score, name))
    return [name for _, name in sorted(scored, key=lambda item: (-item[0], item[1]))]


def normalized_ids(series: pd.Series) -> pd.Series:
    """Return trimmed IDs while retaining missing values."""
    result = series.astype("string").str.strip()
    return result.mask(result.eq(""))


def extract_gsm_ids(frame: pd.DataFrame, id_columns: Iterable[str]) -> pd.Series:
    """Extract one normalized GSM per row, trying ranked ID columns in order."""
    gsm = pd.Series(pd.NA, index=frame.index, dtype="string")
    for column in id_columns:
        extracted = frame[column].astype("string").str.extract(
            GSM_EXTRACT_PATTERN, expand=False
        )
        gsm = gsm.fillna(extracted.str.upper())
    return gsm


def audit_ids(name: str, frame: pd.DataFrame) -> dict[str, object]:
    heading(f"{name.upper()} SAMPLE ID AUDIT")
    candidates = identify_id_columns(frame)
    if not candidates:
        print("No plausible sample/accession ID columns were detected.")
        print("GSM extraction and ID-level counts are unavailable for this split.")
        return {"sample_ids": set(), "gsm_ids": set(), "id_column": None}

    primary = candidates[0]
    ids = normalized_ids(frame[primary])
    present = ids.dropna()
    duplicate_rows = int(present.duplicated(keep=False).sum())
    duplicate_values = int(present[present.duplicated(keep=False)].nunique())
    gsm = extract_gsm_ids(frame, candidates)
    valid_gsm = gsm.dropna()

    print(f"Detected ID column(s), ranked: {candidates}")
    print(f"Primary sample ID column:       {primary}")
    print(f"Total samples (rows):           {len(frame):,}")
    print(f"Unique non-missing sample IDs:  {present.nunique():,}")
    print(f"Duplicate-ID rows:              {duplicate_rows:,}")
    print(f"Distinct duplicated IDs:        {duplicate_values:,}")
    print(f"Missing/blank sample IDs:       {ids.isna().sum():,}")
    print(f"Rows with a valid GSM ID:       {gsm.notna().sum():,}")
    print(f"Rows without a valid GSM ID:    {gsm.isna().sum():,}")
    print(f"Unique valid GSM IDs:           {valid_gsm.nunique():,}")
    examples = valid_gsm.drop_duplicates().head(10).tolist()
    print(f"Example GSM IDs:                {examples if examples else '[none]'}")
    return {
        "sample_ids": set(present.tolist()),
        "gsm_ids": set(valid_gsm.tolist()),
        "id_column": primary,
    }


def summarize_metadata(name: str, frame: pd.DataFrame, id_columns: Iterable[str]) -> None:
    heading(f"{name.upper()} USEFUL METADATA")
    id_set = set(id_columns)
    columns = [
        str(column) for column in frame.columns
        if str(column) not in id_set
        and any(hint in str(column).lower() for hint in METADATA_HINTS)
    ]
    if not columns:
        print("No common metadata columns (species, tissue, source, etc.) detected.")
        return

    for column in columns:
        series = frame[column]
        print(f"\n{column}: {series.nunique(dropna=True):,} unique; {series.isna().sum():,} missing")
        counts = series.fillna("<missing>").astype("string").value_counts().head(15)
        print(counts.to_string())
        if series.nunique(dropna=True) > 15:
            print("  ... top 15 values shown")


def compare_splits(train: dict[str, object], validation: dict[str, object]) -> None:
    heading("TRAIN / VALIDATION OVERLAP")
    sample_overlap = train["sample_ids"] & validation["sample_ids"]
    gsm_overlap = train["gsm_ids"] & validation["gsm_ids"]
    print(f"Sample IDs present in both splits: {len(sample_overlap):,}")
    if sample_overlap:
        print(f"Examples: {sorted(sample_overlap)[:10]}")
    print(f"GSM IDs present in both splits:    {len(gsm_overlap):,}")
    if gsm_overlap:
        print(f"Examples: {sorted(gsm_overlap)[:10]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help=f"Directory containing split parquets (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--train", type=Path, help="Explicit train parquet path.")
    parser.add_argument("--validation", type=Path, help="Explicit validation parquet path.")
    parser.add_argument("--examples", type=int, default=5,
                        help="Number of example rows to print (default: 5).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    try:
        train_path = resolve_split_path(args.train, data_dir, "train")
        validation_path = resolve_split_path(args.validation, data_dir, "validation")
        train_frame = load_parquet(train_path)
        validation_frame = load_parquet(validation_path)
    except (FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(f"Input error: {exc}") from exc

    print_structure("train", train_path, train_frame, args.examples)
    print_structure("validation", validation_path, validation_frame, args.examples)
    train_audit = audit_ids("train", train_frame)
    validation_audit = audit_ids("validation", validation_frame)
    compare_splits(train_audit, validation_audit)
    summarize_metadata("train", train_frame, identify_id_columns(train_frame))
    summarize_metadata("validation", validation_frame, identify_id_columns(validation_frame))


if __name__ == "__main__":
    main()
