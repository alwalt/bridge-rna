#!/usr/bin/env python3
"""Lock and prepare the three de Weerd case-study cohorts.

This script only acquires, maps, and freezes expression/sample manifests. It
does not run Bridge, DE, disease-gene tests, or drug enrichment.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import shutil
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parents[1]
WORK = HERE / "work"
RAW = WORK / "raw"
PREP = WORK / "prepared"
RESULTS = HERE / "results"
MANIFESTS = RESULTS / "manifests"
BRIDGE_ROOT = Path("/home/walt/bridge-rna")
CANONICAL = BRIDGE_ROOT / "data/ensembl/canonical_genes.csv"
HGNC = BRIDGE_ROOT / "data/annotations/hgnc/hgnc_complete_set_2026-08-27.tsv"
LENGTHS = BRIDGE_ROOT / "data/gencode/gencode_v49_gene_exon_lengths.csv"
SPLITS = BRIDGE_ROOT / "data/archs4/training/sample_split"

SOURCES = {
    "GSE138614_countMatrix.txt.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE138nnn/GSE138614/suppl/GSE138614_countMatrix.txt.gz",
    "GSE138614_family.soft.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE138nnn/GSE138614/soft/GSE138614_family.soft.gz",
    "GSE101794_RAW.tar": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE101nnn/GSE101794/suppl/GSE101794_RAW.tar",
    "GSE101794_family.soft.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE101nnn/GSE101794/soft/GSE101794_family.soft.gz",
    "GSE72509_SLE_RPKMs.txt.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE72nnn/GSE72509/suppl/GSE72509_SLE_RPKMs.txt.gz",
    "GSE72509_family.soft.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE72nnn/GSE72509/soft/GSE72509_family.soft.gz",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, url: str) -> Path:
    path = RAW / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        return path
    cached = Path("/tmp") / name
    if cached.exists() and cached.stat().st_size:
        shutil.copy2(cached, path)
    else:
        urllib.request.urlretrieve(url, path)
    return path


def parse_soft(gse: str) -> pd.DataFrame:
    records, current = [], None
    with gzip.open(RAW / f"{gse}_family.soft.gz", "rt", errors="replace") as handle:
        for line in handle:
            if line.startswith("^SAMPLE = "):
                if current:
                    records.append(current)
                current = {"sample_id": line.strip().split(" = ", 1)[1], "accession": gse}
            elif current is not None and line.startswith("!Sample_"):
                key, value = line.strip().split(" = ", 1)
                key = key.removeprefix("!Sample_")
                if key == "characteristics_ch1" and ": " in value:
                    label, val = value.split(": ", 1)
                    current[re.sub(r"\W+", "_", label.lower()).strip("_")] = val
                elif key in {"title", "source_name_ch1"}:
                    current[key] = value
        if current:
            records.append(current)
    return pd.DataFrame(records)


def split_lookup() -> dict[str, str]:
    lookup = {}
    for split in ("train", "validation", "unused"):
        frame = pd.read_parquet(SPLITS / f"{split}_samples.parquet", columns=["gsm_accession"])
        lookup.update(dict.fromkeys(frame.gsm_accession.dropna().astype(str), split))
    return lookup


def annotations():
    canonical = pd.read_csv(CANONICAL).gene_symbol.astype(str).tolist()
    hgnc = pd.read_csv(HGNC, sep="\t", dtype=str)
    ens = (hgnc.dropna(subset=["ensembl_gene_id"]).drop_duplicates("ensembl_gene_id")
           .set_index("ensembl_gene_id").symbol.to_dict())
    lengths = (pd.read_csv(LENGTHS).drop_duplicates("gene_symbol")
               .set_index("gene_symbol").exon_length.to_dict())
    return canonical, ens, lengths


def counts_to_tpm(counts: pd.DataFrame, lengths: dict[str, float]) -> pd.DataFrame:
    length = pd.Series(counts.index.map(lengths), index=counts.index, dtype=float)
    valid = length.notna() & length.gt(0)
    rate = counts.loc[valid].div(length.loc[valid] / 1000.0, axis=0)
    return rate.div(rate.sum(axis=0), axis=1) * 1e6


def align_and_save(accession: str, tpm: pd.DataFrame, manifest: pd.DataFrame,
                   canonical: list[str], counts: pd.DataFrame | None = None) -> dict:
    ordered = manifest.sample_id.tolist()
    tpm = tpm.loc[~tpm.index.duplicated()].reindex(columns=ordered)
    aligned = tpm.reindex(canonical).fillna(0).T.to_numpy(np.float32)
    log1p = np.log1p(aligned)
    npy_path = PREP / f"{accession}_log1p_tpm.npy"
    if npy_path.exists():
        existing = np.load(npy_path, mmap_mode="r")
        if existing.shape != log1p.shape or not np.allclose(existing, log1p, rtol=0, atol=1e-6):
            raise RuntimeError(f"existing prepared input differs for {accession}; refusing overwrite")
    else:
        np.save(npy_path, log1p)
    expression_frame = pd.DataFrame(log1p, columns=canonical)
    expression_frame.insert(0, "sample_id", ordered)
    expression_frame.to_csv(PREP / f"{accession}_log1p_tpm.csv.gz", index=False, compression="gzip")
    pd.DataFrame({"sample_id": ordered, "matrix_row": np.arange(len(ordered))}).to_csv(
        PREP / f"{accession}_matrix_rows.csv", index=False)
    if counts is not None:
        counts.reindex(columns=ordered).to_parquet(PREP / f"{accession}_counts.parquet")
    mapped = set(tpm.index) & set(canonical)
    expressed = int((aligned.sum(axis=0) > 0).sum())
    return {"source_gene_rows": int(tpm.shape[0]), "mapped_canonical_genes": len(mapped),
            "mapped_canonical_fraction": len(mapped) / len(canonical),
            "expressed_canonical_genes": expressed,
            "missing_canonical_genes": len(canonical) - len(mapped)}


def prepare_ms(canonical, ens, lengths, exposure):
    meta = parse_soft("GSE138614")
    selected = meta[meta.lesion_type.isin(["Active (AL)", "White matter (WM)"])].copy()
    selected["role"] = np.where(selected.lesion_type.eq("Active (AL)"), "case", "control")
    selected["donor"] = selected.individual
    selected["tissue"] = "brain white matter"
    selected["matrix_col"] = "G58-" + selected.title.str.extract(r"Sample_(\d+)$")[0]
    frame = pd.read_csv(RAW / "GSE138614_countMatrix.txt.gz", sep="\t", index_col=0)
    symbols = frame.index.astype(str).str.replace(r"\.\d+$", "", regex=True).map(ens)
    values = frame.apply(pd.to_numeric, errors="coerce").fillna(0)
    values.insert(0, "gene", symbols)
    counts = values.dropna(subset=["gene"]).groupby("gene", sort=False).sum()
    tpm = counts_to_tpm(counts, lengths)
    selected["sample_id"] = selected.sample_id.astype(str)
    rename = dict(zip(selected.matrix_col, selected.sample_id))
    tpm = tpm.rename(columns=rename); counts = counts.rename(columns=rename)
    selected["pretraining_split"] = selected.sample_id.map(exposure).fillna("absent")
    coverage = align_and_save("GSE138614", tpm, selected, canonical, counts)
    return selected, coverage


def prepare_crohn(canonical, exposure):
    meta = parse_soft("GSE101794")
    selected = meta[meta.paris_age.eq("A1a")].copy()
    selected["role"] = np.where(selected.diagnosis.eq("CD"), "case", "control")
    selected["donor"] = selected.sample_id
    selected["tissue"] = "terminal ileum biopsy"
    selected["pretraining_split"] = selected.sample_id.map(exposure).fillna("absent")
    pieces = []
    with tarfile.open(RAW / "GSE101794_RAW.tar") as archive:
        members = {Path(m.name).name.split("_", 1)[0]: m for m in archive.getmembers()}
        for sample in selected.sample_id:
            handle = archive.extractfile(members[sample])
            if handle is None:
                raise RuntimeError(f"missing TPM member for {sample}")
            with gzip.open(handle, "rt") as stream:
                frame = pd.read_csv(stream, sep="\t")
            series = frame.groupby("Gene").TPM.sum().rename(sample)
            pieces.append(series)
    tpm = pd.concat(pieces, axis=1).fillna(0)
    coverage = align_and_save("GSE101794", tpm, selected, canonical)
    return selected, coverage


def prepare_sle(canonical, exposure):
    meta = parse_soft("GSE72509")
    selected = meta.copy()
    selected["role"] = np.where(selected.disease_status.str.startswith("systemic"), "case", "control")
    selected["donor"] = selected.sample_id
    selected["tissue"] = "whole blood"
    selected["matrix_col"] = selected.title
    selected["pretraining_split"] = selected.sample_id.map(exposure).fillna("absent")
    frame = pd.read_csv(RAW / "GSE72509_SLE_RPKMs.txt.gz", sep="\t")
    sample_cols = selected.matrix_col.tolist()
    values = frame[sample_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    values.insert(0, "gene", frame.SYMBOL)
    rpkm = values.dropna(subset=["gene"]).groupby("gene", sort=False).sum()
    # RPKM -> TPM by within-sample rescaling; no additional length correction.
    tpm = rpkm.div(rpkm.sum(axis=0), axis=1) * 1e6
    tpm = tpm.rename(columns=dict(zip(selected.matrix_col, selected.sample_id)))
    coverage = align_and_save("GSE72509", tpm, selected, canonical)
    return selected, coverage


def main():
    for directory in (RAW, PREP, RESULTS, MANIFESTS):
        directory.mkdir(parents=True, exist_ok=True)
    source_records = {}
    for name, url in SOURCES.items():
        path = fetch(name, url)
        source_records[name] = {"url": url, "bytes": path.stat().st_size, "sha256": sha256(path)}
    canonical, ens, lengths = annotations()
    exposure = split_lookup()
    manifests, coverage = {}, {}
    for accession, function, args in [
        ("GSE138614", prepare_ms, (canonical, ens, lengths, exposure)),
        ("GSE101794", prepare_crohn, (canonical, exposure)),
        ("GSE72509", prepare_sle, (canonical, exposure)),
    ]:
        manifest, stats = function(*args)
        manifests[accession] = manifest
        coverage[accession] = stats
    combined = pd.concat(manifests.values(), ignore_index=True, sort=False)
    keep = ["accession", "sample_id", "title", "tissue", "role", "donor", "sex",
            "age_at_diagnosis_in_years", "paris_age", "lesion_type", "anti_ro", "ism",
            "pretraining_split"]
    for column in keep:
        if column not in combined:
            combined[column] = np.nan
    combined[keep].to_csv(MANIFESTS / "locked_samples.csv", index=False)
    combined[keep].to_parquet(MANIFESTS / "locked_samples.parquet", index=False)

    cohort_rows = []
    definitions = {
        "GSE138614": ("Multiple sclerosis — active lesion", "Active (AL) white-matter lesion", "non-neurological control white matter"),
        "GSE101794": ("Crohn's disease — ileum", "Crohn disease, Paris age A1a ileal biopsy", "non-IBD, Paris age A1a ileal biopsy"),
        "GSE72509": ("Systemic lupus erythematosus — blood", "SLE whole blood", "healthy whole blood"),
    }
    for accession, group in combined.groupby("accession", sort=False):
        label, case_def, control_def = definitions[accession]
        splits = group.pretraining_split.value_counts().to_dict()
        cohort_rows.append({"disease_contrast": label, "accession": accession,
                            "tissue": group.tissue.iloc[0], "case_definition": case_def,
                            "control_definition": control_def,
                            "case_samples": int(group.role.eq("case").sum()),
                            "control_samples": int(group.role.eq("control").sum()),
                            "case_donors": int(group[group.role.eq("case")].donor.nunique()),
                            "control_donors": int(group[group.role.eq("control")].donor.nunique()),
                            "archs4_train": int(splits.get("train", 0)),
                            "archs4_validation": int(splits.get("validation", 0)),
                            "archs4_unused": int(splits.get("unused", 0)),
                            "archs4_absent": int(splits.get("absent", 0)), **coverage[accession]})
    cohorts = pd.DataFrame(cohort_rows)
    cohorts.to_csv(RESULTS / "locked_cohorts.csv", index=False)
    provenance = {
        "lock_status": "frozen",
        "deweerd_labels": {
            "GSE138614": "ms_modalyser_al_vs_control / MS Active Lesion",
            "GSE101794": "E-GEOD-101794_chron_a1a_vs_control / Crohn Ileum 1",
            "GSE72509": "E-GEOD-72509_sle_vs_control / Systemic Lupus Erythematosus Blood",
        },
        "expression_contract": "natural log1p(TPM), canonical 15,165 genes",
        "ms_independence": "donor is the biological unit; no lesion-control pairing is invented",
        "crohn_subset": "Paris age A1a only, matching de Weerd supplemental contrast label",
        "sle_subset": "all SLE and healthy whole-blood samples",
        "sources": source_records,
        "coverage": coverage,
    }
    (RESULTS / "cohort_lock_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(cohorts.to_string(index=False))


if __name__ == "__main__":
    main()
