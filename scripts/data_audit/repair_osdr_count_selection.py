#!/usr/bin/env python3
"""Replace supplementary STAR OSDR counts with primary RSEM counts.

The original selector scored STAR filenames above RSEM filenames without using
the OSDR ``Data_Type`` field. This utility audits all selected studies against
the saved OSDR API file manifest, downloads primary files into a staging area,
validates sample columns, archives replaced files, and updates both manifests.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def identify_replacements(selected: pd.DataFrame, available: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, current in selected.iterrows():
        candidates = available[available["OSD"].eq(current["id.accession"])].copy()
        candidates = candidates[candidates["Filename"].str.contains("unnormalized", case=False, na=False)]
        candidates = candidates[~candidates["Filename"].str.contains(r"differential|_vst_|_mrna_", case=False, na=False, regex=True)]
        primary = candidates[candidates["Data_Type"].fillna("").str.strip().str.lower().eq("unnormalized counts")]
        if "_STAR_" not in current["counts_file"] or primary.empty:
            continue
        rsem = primary[primary["Filename"].str.contains("_RSEM_", case=False, na=False)]
        replacement = (rsem if not rsem.empty else primary).iloc[0]
        if replacement["Filename"] != current["counts_file"]:
            rows.append({
                "id.accession": current["id.accession"],
                "old_file": current["counts_file"],
                "new_file": replacement["Filename"],
                "download_url": replacement["Download_URL"],
                "api_data_type": replacement["Data_Type"],
                "old_source_path": current["source_path"],
            })
    return pd.DataFrame(rows)


def download(row: dict, staging: Path) -> tuple[str, Path, str]:
    target = staging / row["new_file"]
    acquisition = "downloaded_from_osdr_api_url"
    try:
        with requests.get(row["download_url"], stream=True, timeout=300) as response:
            response.raise_for_status()
            with target.open("wb") as handle:
                for chunk in response.iter_content(1 << 20):
                    if chunk:
                        handle.write(chunk)
    except requests.RequestException:
        # Some URLs in the saved API manifest now return 404 although the file
        # was successfully downloaded during the original API crawl. Reuse that
        # byte-identical local API artifact rather than silently skipping it.
        fallback = Path("/home/walt/BioFM") / Path(row["old_source_path"]).with_name(row["new_file"])
        if not fallback.exists():
            raise
        shutil.copy2(fallback, target)
        acquisition = "reused_original_osdr_api_download_after_current_url_404"
    if target.stat().st_size < 1024:
        raise RuntimeError(f"Suspiciously small download: {target}")
    header = pd.read_csv(target, nrows=0)
    if len(header.columns) < 2:
        raise RuntimeError(f"No sample columns in {target}")
    return row["id.accession"], target, acquisition


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-file-manifest", type=Path, default=Path("/home/walt/BioFM/02_mouse_counts_file_manifest.tsv"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    selected_path = ROOT / "data/osdr/metadata/selected_counts_manifest.tsv"
    samples_path = ROOT / "data/osdr/metadata/selected_sample_metadata.tsv"
    raw = ROOT / "data/osdr/raw"
    selected = pd.read_csv(selected_path, sep="\t", dtype=str)
    samples = pd.read_csv(samples_path, sep="\t", dtype=str)
    available = pd.read_csv(args.api_file_manifest, sep="\t", dtype=str)
    replacements = identify_replacements(selected, available)
    print(f"Selected studies: {len(selected)}; replacements required: {len(replacements)}", flush=True)
    if replacements.empty or args.dry_run:
        if not replacements.empty:
            print(replacements[["id.accession", "old_file", "new_file"]].to_string(index=False))
        return

    staging = Path(tempfile.mkdtemp(prefix="osdr_rsem_repair_", dir=raw))
    archive = raw / "replaced_star_supplementary"
    archive.mkdir(exist_ok=True)
    try:
        downloaded, acquisition = {}, {}
        records = replacements.to_dict("records")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(download, row, staging): row for row in records}
            for i, future in enumerate(as_completed(futures), 1):
                accession, path, mode = future.result()
                downloaded[accession] = path
                acquisition[accession] = mode
                print(f"[download {i}/{len(futures)}] {accession}: {path.name} ({mode})", flush=True)

        # Validate sample columns against the old selected file before any move.
        for row in records:
            old = raw / row["old_file"]
            new = downloaded[row["id.accession"]]
            old_samples = set(pd.read_csv(old, nrows=0).columns[1:])
            new_samples = set(pd.read_csv(new, nrows=0).columns[1:])
            if old_samples != new_samples:
                raise RuntimeError(f"Sample-column mismatch for {row['id.accession']}: old={len(old_samples)}, new={len(new_samples)}, old-only={sorted(old_samples-new_samples)[:5]}, new-only={sorted(new_samples-old_samples)[:5]}")

        audit_rows = []
        for row in records:
            accession = row["id.accession"]; old = raw / row["old_file"]; new = downloaded[accession]
            archived = archive / row["old_file"]
            if archived.exists() and sha256(archived) != sha256(old):
                raise RuntimeError(f"Conflicting archive file: {archived}")
            if not archived.exists(): shutil.move(old, archived)
            else: old.unlink()
            shutil.move(new, raw / row["new_file"])
            mask = selected["id.accession"].eq(accession)
            selected.loc[mask, "counts_file"] = row["new_file"]
            selected.loc[mask, "source_path"] = str(Path(row["old_source_path"]).with_name(row["new_file"]))
            selected.loc[mask, "is_star"] = "0"; selected.loc[mask, "is_rsem"] = "1"
            smask = samples["id.accession"].eq(accession)
            samples.loc[smask, "counts_file"] = row["new_file"]
            samples.loc[smask, "counts_path"] = f"data/osdr/raw/{row['new_file']}"
            audit_rows.append({"id.accession": accession, "old_file": row["old_file"], "new_file": row["new_file"],
                               "new_sha256": sha256(raw / row["new_file"]), "sample_columns": len(pd.read_csv(raw / row["new_file"], nrows=0).columns)-1,
                               "acquisition": acquisition[accession], "old_archived_at": str(archived.relative_to(ROOT))})
        selected.to_csv(selected_path, sep="\t", index=False)
        samples.to_csv(samples_path, sep="\t", index=False)
        pd.DataFrame(audit_rows).to_csv(ROOT / "data/osdr/metadata/rsem_replacement_audit.tsv", sep="\t", index=False)
        print(f"Replaced and validated {len(audit_rows)} files; STAR originals archived at {archive}")
    finally:
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
