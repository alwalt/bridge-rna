#!/usr/bin/env python3
"""Build a versioned, metadata-only ARCHS4 snapshot for manifest samples.

Only fields under ``meta/samples`` are accessed; the H5 expression matrix is
never read. The installed ``archs4py`` metadata interface is used to retrieve
aligned fields. Output contains one row per GSM in ``sample_manifest.parquet``.

Examples
--------
    python scripts/data_audit/build_archs4_sample_metadata.py
    python scripts/data_audit/build_archs4_sample_metadata.py --fields title source_name_ch1 platform_id
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import h5py
import pandas as pd
from archs4py.meta import get_meta_sample_field


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "data" / "manifests" / "sample_manifest.parquet"
DEFAULT_ARCHS4_DIR = REPO_ROOT / "data" / "archs4"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "manifests"
DEFAULT_FIELDS = [
    "series_id",
    "organism_ch1",
    "title",
    "source_name_ch1",
    "characteristics_ch1",
    "platform_id",
    "library_strategy",
    "library_source",
    "instrument_model",
    "molecule_ch1",
    "submission_date",
    "singlecellprobability",
]
VERSION_RE = re.compile(r"v(\d+(?:\.\d+)*)", re.IGNORECASE)


def discover_archs4_files(directory: Path, explicit: Sequence[Path]) -> list[Path]:
    paths = [path.expanduser().resolve() for path in explicit]
    if not paths:
        paths = sorted(path.resolve() for path in directory.rglob("*.h5"))
    if not paths:
        raise FileNotFoundError(f"No ARCHS4 H5 files found under {directory}")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"ARCHS4 H5 files do not exist: {missing}")
    return paths


def inspect_sample_fields(path: Path) -> set[str]:
    """List available sample metadata fields without touching expression data."""
    with h5py.File(path, "r") as handle:
        if "meta/samples" not in handle:
            raise ValueError(f"Missing meta/samples in {path}")
        return set(handle["meta/samples"].keys())


def archs4_version(paths: Sequence[Path]) -> str:
    versions = {
        match.group(1)
        for path in paths
        if (match := VERSION_RE.search(path.name)) is not None
    }
    if not versions:
        return "unknown"
    return "_".join(sorted(versions))


def default_output_path(paths: Sequence[Path], output_dir: Path) -> Path:
    return output_dir / f"archs4_sample_metadata_v{archs4_version(paths)}.parquet"


def load_required_gsms(manifest_path: Path) -> set[str]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Sample manifest does not exist: {manifest_path}")
    frame = pd.read_parquet(manifest_path, columns=["gsm"])
    gsm = frame["gsm"].astype("string").str.strip().str.upper()
    if gsm.isna().any() or gsm.eq("").any() or gsm.duplicated().any():
        raise ValueError("Sample manifest GSMs must be non-missing and unique")
    return set(gsm)


def extract_file_metadata(
    path: Path, required_gsms: set[str], fields: Sequence[str], version: str
) -> pd.DataFrame:
    """Extract aligned metadata for required GSMs through archs4py."""
    available = inspect_sample_fields(path)
    if "geo_accession" not in available:
        raise ValueError(f"ARCHS4 file lacks geo_accession metadata: {path}")
    missing_fields = set(fields) - available
    if missing_fields:
        print(f"  warning: {path.name} lacks fields {sorted(missing_fields)}")

    accessions = get_meta_sample_field(str(path), "geo_accession")
    normalized = [str(value).strip().upper() for value in accessions]
    selected_indices = [
        index for index, gsm in enumerate(normalized) if gsm in required_gsms
    ]
    selected_gsms = [normalized[index] for index in selected_indices]
    result: dict[str, object] = {"gsm": selected_gsms}

    print(
        f"  {path.name}: {len(accessions):,} metadata rows; "
        f"{len(selected_indices):,} manifest GSMs"
    )
    for field in fields:
        if field not in available:
            result[field] = [pd.NA] * len(selected_indices)
            continue
        values = get_meta_sample_field(str(path), field)
        if len(values) != len(accessions):
            raise ValueError(
                f"Unaligned field {field!r} in {path}: {len(values):,} values "
                f"for {len(accessions):,} accessions"
            )
        result[field] = [values[index] for index in selected_indices]
        print(f"    extracted {field}")

    result["metadata_archs4_file"] = [path.name] * len(selected_indices)
    result["archs4_version"] = [version] * len(selected_indices)
    return pd.DataFrame(result)


def build_metadata_snapshot(
    manifest_path: Path, archs4_files: Sequence[Path], fields: Sequence[str]
) -> pd.DataFrame:
    required = load_required_gsms(manifest_path)
    version = archs4_version(archs4_files)
    frames = [
        extract_file_metadata(path, required, fields, version)
        for path in archs4_files
    ]
    metadata = pd.concat(frames, ignore_index=True)
    duplicate_rows = metadata["gsm"].duplicated(keep=False)
    if duplicate_rows.any():
        examples = metadata.loc[duplicate_rows, "gsm"].drop_duplicates().head(10).tolist()
        raise ValueError(
            f"{duplicate_rows.sum():,} metadata rows have GSMs present in multiple "
            f"ARCHS4 sources; examples: {examples}"
        )
    found = set(metadata["gsm"])
    missing = required - found
    extra = found - required
    if missing or extra:
        raise ValueError(
            f"Metadata coverage mismatch: {len(missing):,} missing and {len(extra):,} extra GSMs; "
            f"missing examples: {sorted(missing)[:10]}"
        )
    return metadata.sort_values("gsm", key=lambda values: values.str[3:].astype(int)).reset_index(drop=True)


def save_snapshot(
    metadata: pd.DataFrame, output: Path, manifest_path: Path,
    archs4_files: Sequence[Path], fields: Sequence[str]
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    metadata.to_parquet(temporary, index=False)
    temporary.replace(output)
    provenance_path = output.with_suffix(".json")
    provenance_path.write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "archs4_version": archs4_version(archs4_files),
        "archs4_files": [str(path) for path in archs4_files],
        "sample_manifest": str(manifest_path),
        "num_samples": len(metadata),
        "fields": list(fields),
        "metadata_only": True,
        "expression_matrix_accessed": False,
    }, indent=2))
    print(f"Saved {len(metadata):,} metadata rows to {output}")
    print(f"Saved provenance to {provenance_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--archs4-dir", type=Path, default=DEFAULT_ARCHS4_DIR)
    parser.add_argument("--archs4-file", type=Path, action="append", default=[])
    parser.add_argument("--fields", nargs="+", default=DEFAULT_FIELDS,
                        help="meta/samples fields to extract.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output parquet; default includes detected ARCHS4 version.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        manifest_path = args.manifest.expanduser().resolve()
        archs4_files = discover_archs4_files(
            args.archs4_dir.expanduser().resolve(), args.archs4_file
        )
        output = (
            args.output.expanduser().resolve()
            if args.output is not None
            else default_output_path(archs4_files, DEFAULT_OUTPUT_DIR).resolve()
        )
        print(f"ARCHS4 version: {archs4_version(archs4_files)}")
        print(f"Fields: {args.fields}")
        metadata = build_metadata_snapshot(manifest_path, archs4_files, args.fields)
        save_snapshot(metadata, output, manifest_path, archs4_files, args.fields)
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
