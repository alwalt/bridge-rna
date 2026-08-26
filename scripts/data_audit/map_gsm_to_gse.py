#!/usr/bin/env python3
"""Build the authoritative GSM -> GSE mapping manifest from local ARCHS4 metadata.

No GEO/NCBI network fallback is used. Only the aligned ``geo_accession`` and
``series_id`` metadata arrays are read from the ARCHS4 H5 files; expression data
is never accessed. The resulting ``data/manifests/sample_manifest.parquet`` is the
persistent reference artifact for downstream sample-exposure checks and analysis.

Examples
--------
    python scripts/data_audit/map_gsm_to_gse.py
    python scripts/data_audit/map_gsm_to_gse.py --train data/archs4/training/sample_split/train_samples.parquet --validation data/archs4/training/sample_split/validation_samples.parquet
    python scripts/data_audit/map_gsm_to_gse.py --unused unused_samples.parquet
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from archs4py.meta import get_meta_sample_field


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRETRAINING_DIR = REPO_ROOT / "data" / "pretraining"
DEFAULT_ARCHS4_DIR = REPO_ROOT / "data" / "archs4"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "manifests" / "sample_manifest.parquet"
GSM_RE = re.compile(r"GSM\d+", re.IGNORECASE)
GSE_RE = re.compile(r"GSE\d+", re.IGNORECASE)
OUTPUT_COLUMNS = [
    "gsm", "gse_candidates", "gse_candidates_str", "gse_count",
    "mapping_status", "has_gse_conflict", "mapping_source", "species",
    "archs4_file", "has_source_conflict",
]


def heading(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def normalize_gsm(value: object) -> str | None:
    match = GSM_RE.search(str(value)) if pd.notna(value) else None
    return match.group(0).upper() if match else None


def parse_gse_candidates(value: object) -> list[str]:
    """Return unique, naturally ordered GSE accessions from an ARCHS4 value."""
    if value is None or (not isinstance(value, (list, tuple, set)) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        text = ";".join(str(item) for item in value)
    else:
        text = str(value)
    candidates = {match.upper() for match in GSE_RE.findall(text)}
    return sorted(candidates, key=lambda item: int(item[3:]))


def _candidate_id_columns(frame: pd.DataFrame) -> list[str]:
    hints = ("gsm", "geo_accession", "accession", "sample", "_id", "id")
    columns = [str(column) for column in frame.columns]
    likely = [column for column in columns if any(h in column.lower() for h in hints)]
    if likely:
        return likely
    # Graceful fallback for unusually named metadata tables; do not stringify
    # numeric expression columns.
    return [
        str(column) for column in frame.select_dtypes(include=["object", "string"]).columns
    ]


def extract_gsms_from_file(path: Path) -> set[str]:
    """Extract GSMs from a parquet/CSV/TSV/TXT input without changing it."""
    if not path.is_file():
        raise FileNotFoundError(f"GSM input does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    elif suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix in {".tsv", ".tab"}:
        frame = pd.read_csv(path, sep="\t")
    else:
        gsms: set[str] = set()
        for match in GSM_RE.findall(path.read_text(errors="replace")):
            gsms.add(match.upper())
        return gsms

    gsms = set()
    for column in _candidate_id_columns(frame):
        values = frame[column].dropna().astype("string")
        extracted = values.str.extract(r"(?i)(GSM\d+)", expand=False).dropna()
        gsms.update(extracted.str.upper().tolist())
    if not isinstance(frame.index, pd.RangeIndex):
        extracted = frame.index.to_series().astype("string").str.extract(
            r"(?i)(GSM\d+)", expand=False
        ).dropna()
        gsms.update(extracted.str.upper().tolist())
    return gsms


def resolve_split_path(explicit: Path | None, split: str, data_dir: Path) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{split} parquet does not exist: {path}")
        return path
    names = ("train_samples.parquet", "train.parquet") if split == "train" else (
        "validation_samples.parquet", "validation.parquet", "val_samples.parquet",
        "val.parquet",
    )
    found = [data_dir / name for name in names if (data_dir / name).is_file()]
    if len(found) == 1:
        return found[0].resolve()
    if not found:
        raise FileNotFoundError(
            f"No {split} parquet found in {data_dir}. Pass --{split} explicitly."
        )
    raise RuntimeError(f"Multiple {split} parquets found; pass --{split} explicitly: {found}")


def discover_unused_paths(explicit: Sequence[Path], data_dir: Path) -> list[Path]:
    """Resolve explicit unused inputs or discover clearly named unused parquets."""
    if explicit:
        paths = [path.expanduser().resolve() for path in explicit]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Unused GSM input(s) do not exist: {missing}")
        return paths

    names = ("unused_samples.parquet", "unused.parquet")
    discovered = [data_dir / name for name in names if (data_dir / name).is_file()]
    return [path.resolve() for path in discovered]


def extract_required_gsms(
    train_path: Path, validation_path: Path, unused_paths: Sequence[Path] = ()
) -> set[str]:
    """Build ``train_gsms | validation_gsms | unused_gsms`` explicitly."""
    heading("REQUIRED GSM INPUTS")
    train_gsms = extract_gsms_from_file(train_path)
    validation_gsms = extract_gsms_from_file(validation_path)
    unused_gsms: set[str] = set()
    print(f"Train ({train_path}): {len(train_gsms):,} unique GSMs")
    print(f"Validation ({validation_path}): {len(validation_gsms):,} unique GSMs")
    if unused_paths:
        for path in unused_paths:
            gsms = extract_gsms_from_file(path)
            unused_gsms.update(gsms)
            print(f"Unused ({path}): {len(gsms):,} unique GSMs")
        print(f"Unused union: {len(unused_gsms):,} unique GSMs")
    else:
        print("Unused: no input supplied or discovered; continuing with train + validation")

    required = train_gsms | validation_gsms | unused_gsms
    print(f"Union: {len(required):,} unique GSMs")
    return required


def _list_value(value: object) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return parse_gse_candidates(value)
    # PyArrow list columns commonly return numpy arrays after read_parquet.
    if hasattr(value, "tolist") and not isinstance(value, str):
        return parse_gse_candidates(value.tolist())
    return parse_gse_candidates(value)


def load_existing_mapping(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    frame = pd.read_parquet(path)
    if "gsm" not in frame.columns:
        raise ValueError(f"Existing mapping lacks required 'gsm' column: {path}")
    frame = frame.copy()
    frame["gsm"] = frame["gsm"].map(normalize_gsm).astype("string")
    if frame["gsm"].isna().any():
        raise ValueError("Existing mapping contains invalid/missing GSM values.")
    duplicates = int(frame["gsm"].duplicated(keep=False).sum())
    if duplicates:
        raise ValueError(
            f"Existing mapping contains {duplicates:,} duplicate GSM rows; "
            "repair the cache rather than silently selecting one."
        )
    if "gse_candidates" in frame.columns:
        frame["gse_candidates"] = frame["gse_candidates"].map(_list_value)
    elif "gse_candidates_str" in frame.columns:
        frame["gse_candidates"] = frame["gse_candidates_str"].map(parse_gse_candidates)
    else:
        frame["gse_candidates"] = [[] for _ in range(len(frame))]
    for column in OUTPUT_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[OUTPUT_COLUMNS]


def inspect_archs4_metadata(path: Path) -> tuple[str, str]:
    """Confirm required fields exist and infer species without reading expression."""
    import h5py

    with h5py.File(path, "r") as handle:
        if "meta/samples" not in handle:
            raise ValueError(f"Missing meta/samples group in {path}")
        fields = set(handle["meta/samples"].keys())
        missing = {"geo_accession", "series_id"} - fields
        if missing:
            raise ValueError(f"Missing ARCHS4 metadata field(s) {sorted(missing)} in {path}")
    lower = str(path).lower()
    species = "human" if "human" in lower else "mouse" if "mouse" in lower else "unknown"
    return species, path.name


def discover_archs4_files(directory: Path, explicit: Sequence[Path]) -> list[Path]:
    files = [path.expanduser().resolve() for path in explicit]
    if not files:
        files = sorted(path.resolve() for path in directory.rglob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No ARCHS4 H5 files found under {directory}")
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"ARCHS4 H5 file(s) not found: {missing}")
    return files


def load_archs4_metadata(path: Path) -> tuple[list[str], list[object], str]:
    """Read only aligned sample metadata through archs4py."""
    species, _ = inspect_archs4_metadata(path)
    accessions = get_meta_sample_field(str(path), "geo_accession")
    series_ids = get_meta_sample_field(str(path), "series_id")
    if len(accessions) != len(series_ids):
        raise ValueError(
            f"Unaligned ARCHS4 metadata in {path}: {len(accessions):,} GSM values "
            f"versus {len(series_ids):,} series values"
        )
    return accessions, series_ids, species


def build_archs4_mapping(gsms: set[str], archs4_files: Sequence[Path]) -> pd.DataFrame:
    """Map requested GSMs, retaining every GSE and every matching source."""
    candidates_by_gsm: dict[str, set[str]] = defaultdict(set)
    source_candidate_sets: dict[str, list[tuple[str, frozenset[str]]]] = defaultdict(list)
    species_by_gsm: dict[str, set[str]] = defaultdict(set)
    files_by_gsm: dict[str, set[str]] = defaultdict(set)

    remaining = set(gsms)
    heading("ARCHS4 METADATA MAPPING")
    for path in archs4_files:
        accessions, series_ids, species = load_archs4_metadata(path)
        matched = 0
        for raw_gsm, raw_series in zip(accessions, series_ids):
            gsm = normalize_gsm(raw_gsm)
            if gsm is None or gsm not in gsms:
                continue
            candidates = frozenset(parse_gse_candidates(raw_series))
            candidates_by_gsm[gsm].update(candidates)
            source_candidate_sets[gsm].append((path.name, candidates))
            species_by_gsm[gsm].add(species)
            files_by_gsm[gsm].add(path.name)
            remaining.discard(gsm)
            matched += 1
        print(
            f"{path.name}: scanned {len(accessions):,} aligned metadata rows; "
            f"matched {matched:,} requested rows"
        )

    rows = []
    for gsm in sorted(gsms, key=lambda item: int(item[3:])):
        candidates = sorted(candidates_by_gsm[gsm], key=lambda item: int(item[3:]))
        nonempty_source_sets = {
            values for _, values in source_candidate_sets[gsm] if values
        }
        source_conflict = len(nonempty_source_sets) > 1
        if not candidates:
            status = "unresolved"
        elif len(candidates) == 1:
            status = "mapped_single"
        else:
            status = "mapped_multiple"
        rows.append({
            "gsm": gsm,
            "gse_candidates": candidates,
            "gse_candidates_str": ";".join(candidates),
            "gse_count": len(candidates),
            "mapping_status": status,
            "has_gse_conflict": len(candidates) > 1,
            "mapping_source": "archs4",
            "species": ";".join(sorted(species_by_gsm[gsm])) or pd.NA,
            "archs4_file": ";".join(sorted(files_by_gsm[gsm])) or pd.NA,
            "has_source_conflict": source_conflict,
        })
    print(f"Not found in any ARCHS4 file: {len(remaining):,}")
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def update_mapping(
    required: set[str], existing: pd.DataFrame, archs4_files: Sequence[Path],
    retry_unresolved: bool = False,
) -> tuple[pd.DataFrame, int, int]:
    cached = set(existing["gsm"].dropna().astype(str))
    retry = set()
    if retry_unresolved and not existing.empty:
        retry = set(existing.loc[existing.mapping_status == "unresolved", "gsm"].astype(str))
    to_process = (required - cached) | (required & retry)
    retained = existing.loc[~existing["gsm"].isin(to_process)].copy()
    new = build_archs4_mapping(to_process, archs4_files) if to_process else existing.iloc[0:0]
    combined = pd.concat([retained, new], ignore_index=True)[OUTPUT_COLUMNS]
    combined = combined.sort_values("gsm", key=lambda s: s.str[3:].astype(int)).reset_index(drop=True)
    return combined, len(required & cached), len(to_process)


def validate_mapping(mapping: pd.DataFrame, required: set[str], cached: int, new: int) -> None:
    requested = mapping[mapping["gsm"].isin(required)].copy()
    duplicate_rows = int(mapping["gsm"].duplicated(keep=False).sum())
    candidate_union = {
        gse for candidates in requested["gse_candidates"] for gse in _list_value(candidates)
    }
    status_counts = requested["mapping_status"].value_counts()
    mapped = int(pd.to_numeric(requested["gse_count"], errors="coerce").gt(0).sum())
    source_conflicts = int(
        requested["has_source_conflict"].astype("boolean").fillna(False).sum()
    )

    heading("AUTHORITATIVE MAPPING MANIFEST VALIDATION")
    print(f"Total unique GSMs requested:             {len(required):,}")
    print(f"GSMs already present in mapping cache:   {cached:,}")
    print(f"Newly processed GSMs:                    {new:,}")
    print(f"mapped_single:                           {status_counts.get('mapped_single', 0):,}")
    print(f"mapped_multiple:                         {status_counts.get('mapped_multiple', 0):,}")
    print(f"unresolved:                              {status_counts.get('unresolved', 0):,}")
    percentage = 100.0 * mapped / len(required) if required else 0.0
    print(f"Overall mapping percentage:              {percentage:.2f}%")
    print(f"Unique GSEs across all candidates:       {len(candidate_union):,}")
    print(f"GSMs with multiple GSE candidates:       {(requested.gse_count > 1).sum():,}")
    print(f"Duplicate GSM rows in final mapping:     {duplicate_rows:,}")
    print(f"Conflicting mappings across H5 sources:  {source_conflicts:,}")
    print("\nMapping counts by species:")
    print(requested["species"].fillna("<unresolved>").value_counts().to_string())

    examples = requested[requested.mapping_status == "mapped_multiple"].head(10)
    print("\nExamples with multiple GSE candidates:")
    if examples.empty:
        print("  [none]")
    else:
        print(examples[["gsm", "gse_candidates_str", "species", "archs4_file"]].to_string(index=False))

    if duplicate_rows:
        raise ValueError("Validation failed: final mapping contains duplicate GSM rows.")
    missing = required - set(requested["gsm"])
    if missing:
        raise ValueError(f"Validation failed: {len(missing):,} requested GSMs are absent.")


def save_mapping(mapping: pd.DataFrame, output: Path) -> None:
    """Atomically replace the derived cache, leaving all source data untouched."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    mapping.to_parquet(temporary, index=False)
    temporary.replace(output)
    print(f"\nSaved authoritative manifest with {len(mapping):,} unique GSM mappings to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_PRETRAINING_DIR,
                        help=f"Train/validation directory (default: {DEFAULT_PRETRAINING_DIR})")
    parser.add_argument("--train", type=Path, help="Explicit train parquet path.")
    parser.add_argument("--validation", type=Path, help="Explicit validation parquet path.")
    parser.add_argument("--unused", type=Path, action="append", default=[],
                        help="Unused/evaluation parquet, CSV, TSV, or TXT; repeat as needed. "
                             "If omitted, a clearly named unused parquet is discovered in "
                             "--data-dir when available.")
    parser.add_argument("--archs4-dir", type=Path, default=DEFAULT_ARCHS4_DIR)
    parser.add_argument("--archs4-file", type=Path, action="append", default=[],
                        help="Explicit ARCHS4 H5 file; repeat for human/mouse.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--retry-unresolved", action="store_true",
                        help="Retry cached unresolved GSMs against current ARCHS4 files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        data_dir = args.data_dir.expanduser().resolve()
        train = resolve_split_path(args.train, "train", data_dir)
        validation = resolve_split_path(args.validation, "validation", data_dir)
        unused = discover_unused_paths(args.unused, data_dir)
        required = extract_required_gsms(train, validation, unused)
        output = args.output.expanduser().resolve()
        existing = load_existing_mapping(output)
        archs4_files = discover_archs4_files(
            args.archs4_dir.expanduser().resolve(), args.archs4_file
        )
        mapping, cached, newly_processed = update_mapping(
            required, existing, archs4_files, args.retry_unresolved
        )
        validate_mapping(mapping, required, cached, newly_processed)
        save_mapping(mapping, output)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
