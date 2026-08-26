#!/usr/bin/env python3
"""Quickly check GSM sample or GSE study exposure in the foundation-model manifest.

Examples
--------
    python scripts/data_audit/check_exposure.py GSM123456
    python scripts/data_audit/check_exposure.py GSM123456 GSM789012
    python scripts/data_audit/check_exposure.py GSE12345 GSE67890
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "data" / "manifests" / "sample_manifest.parquet"
ACCESSION_RE = re.compile(r"^(GSM|GSE)\d+$", re.IGNORECASE)
REQUIRED_COLUMNS = [
    "gsm", "split", "gse_candidates_str", "study_exposure",
    "mapping_status", "species",
]


def normalize_accession(value: str) -> tuple[str, str] | None:
    """Return (kind, normalized accession), or None for invalid input."""
    accession = value.strip().upper()
    match = ACCESSION_RE.fullmatch(accession)
    return (match.group(1).upper(), accession) if match else None


def load_manifest(path: Path) -> pd.DataFrame:
    """Read only the columns needed for exposure lookups."""
    if not path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {path}")
    try:
        manifest = pd.read_parquet(path, columns=REQUIRED_COLUMNS)
    except Exception as exc:
        raise RuntimeError(f"Could not read manifest {path}: {exc}") from exc
    manifest["gsm"] = manifest["gsm"].astype("string").str.upper()
    manifest["split"] = manifest["split"].astype("string")
    manifest["gse_candidates_str"] = manifest["gse_candidates_str"].astype("string")
    return manifest


def yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def print_gsm_result(accession: str, manifest: pd.DataFrame) -> None:
    print(accession)
    rows = manifest.loc[manifest["gsm"].eq(accession)]
    if rows.empty:
        print("Found: No")
        return

    row = rows.iloc[0]
    candidates = str(row["gse_candidates_str"]) if pd.notna(row["gse_candidates_str"]) else ""
    candidates = candidates or "None"
    split = str(row["split"])
    study_seen = str(row["study_exposure"]) == "seen_study"
    print("Found: Yes")
    print(f"Split: {split}")
    print(f"Sample used in training: {yes_no(split == 'train')}")
    print(f"GSE candidate(s): {candidates}")
    print(f"Study seen in training: {yes_no(study_seen)}")
    print(f"Study exposure: {row['study_exposure']}")
    print(f"Mapping status: {row['mapping_status']}")
    print(f"Species: {row['species'] if pd.notna(row['species']) else 'Unknown'}")
    if len(rows) > 1:
        print(f"Warning: {len(rows)} manifest rows share this GSM")


def rows_for_gse(accession: str, manifest: pd.DataFrame) -> pd.DataFrame:
    """Match a GSE exactly within the semicolon-delimited candidate field."""
    pattern = rf"(?:^|;){re.escape(accession)}(?:;|$)"
    mask = manifest["gse_candidates_str"].str.contains(pattern, regex=True, na=False)
    return manifest.loc[mask]


def print_gse_result(accession: str, manifest: pd.DataFrame) -> None:
    print(accession)
    rows = rows_for_gse(accession, manifest)
    if rows.empty:
        print("Found: No")
        return

    counts = rows["split"].value_counts()
    train = int(counts.get("train", 0))
    validation = int(counts.get("val", 0))
    unseen = int(counts.get("unseen", 0))
    print("Found: Yes")
    print(f"Study seen in training: {yes_no(train > 0)}")
    print(f"Samples: {len(rows):,}")
    print(f"Train: {train:,}")
    print(f"Validation: {validation:,}")
    print(f"Unseen: {unseen:,}")


def check_accessions(accessions: list[str], manifest: pd.DataFrame) -> None:
    for index, raw in enumerate(accessions):
        if index:
            print()
        normalized = normalize_accession(raw)
        if normalized is None:
            print(raw)
            print("Found: No")
            print("Error: expected an accession matching GSM<digits> or GSE<digits>")
            continue
        kind, accession = normalized
        if kind == "GSM":
            print_gsm_result(accession, manifest)
        else:
            print_gse_result(accession, manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("accessions", nargs="+", help="One or more GSM/GSE accessions.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help=f"Manifest parquet (default: {DEFAULT_MANIFEST})")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        manifest = load_manifest(args.manifest.expanduser().resolve())
    except (FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    check_accessions(args.accessions, manifest)


if __name__ == "__main__":
    main()
