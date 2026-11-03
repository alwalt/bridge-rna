#!/usr/bin/env python3
"""Freeze sample manifests and prepare canonical Bridge/DE matrices."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
WORK = HERE / "work"
RAW = WORK / "raw"
PREP = WORK / "prepared"
RESULTS = HERE / "results"
MANIFESTS = RESULTS / "manifests"
CONFIG = json.loads((HERE / "config.json").read_text())

GEO_FILES = {
    "GSE297090": "GSE297090_bulkRNAseq_featureCounts-genes.csv.gz",
    "GSE184119": "GSE184119_RAW.tar",
    "GSE189524": "GSE189524_Differential_Expression_Transcripts_Cassidy_et_al_2021_Op.xlsx",
    "GSE276529": "GSE276529_rsem_count.tsv.gz",
    "GSE211204": "GSE211204_RawCountFile_RSEM_genes.txt.gz",
    "GSE113165": "GSE113165_counts.txt.gz",
}
OSDR_FILES = {
    "GSE297560": ("OSD-993", "GLDS-805_rna_seq_STAR_Unnormalized_Counts_GLbulkRNAseq.csv"),
    "GSE234465": ("OSD-684", "GLDS-615_rna_seq_STAR_Unnormalized_Counts_GLbulkRNAseq.csv"),
    "GSE273868": ("OSD-867", "GLDS-719_rna_seq_STAR_Unnormalized_Counts_GLbulkRNAseq.csv"),
}


def say(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, path: Path, cached: Path | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size:
        return
    if cached is not None and cached.is_file():
        shutil.copy2(cached, path)
        return
    say(f"download {url}")
    urllib.request.urlretrieve(url, path)


def geo_prefix(gse: str) -> str:
    return gse[:-3] + "nnn"


def acquire() -> dict[str, dict[str, str | int]]:
    records = {}
    cache_counts = Path("/tmp/bridge_phase1_counts")
    cache_meta = Path("/tmp/bridge_phase1_geo")
    for gse, name in GEO_FILES.items():
        url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{geo_prefix(gse)}/{gse}/suppl/{name}"
        cached_name = "GSE189524.xlsx" if gse == "GSE189524" else name
        path = RAW / name
        fetch(url, path, cache_counts / cached_name)
        soft_name = f"{gse}_family.soft.gz"
        soft_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{geo_prefix(gse)}/{gse}/soft/{soft_name}"
        fetch(soft_url, RAW / soft_name, cache_meta / soft_name)
        records[name] = {"url": url, "size": path.stat().st_size, "sha256": sha256(path)}
        records[soft_name] = {"url": soft_url, "size": (RAW / soft_name).stat().st_size, "sha256": sha256(RAW / soft_name)}
    for gse, (osd, name) in OSDR_FILES.items():
        osd_num = osd.split("-")[1]
        url = f"https://osdr.nasa.gov/geode-py/ws/studies/{osd}/download?source=datamanager&file={name}"
        cached = {
            "GSE297560": cache_counts / name,
            "GSE234465": cache_counts / name,
            "GSE273868": Path("/tmp/osd867-counts.csv"),
        }[gse]
        path = RAW / name
        fetch(url, path, cached)
        soft_name = f"{gse}_family.soft.gz"
        soft_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{geo_prefix(gse)}/{gse}/soft/{soft_name}"
        fetch(soft_url, RAW / soft_name, cache_meta / soft_name)
        records[name] = {"url": url, "size": path.stat().st_size, "sha256": sha256(path), "osd": osd, "osd_numeric": osd_num}
        records[soft_name] = {"url": soft_url, "size": (RAW / soft_name).stat().st_size, "sha256": sha256(RAW / soft_name)}
    (MANIFESTS / "source_files.json").write_text(json.dumps(records, indent=2) + "\n")
    return records


def parse_soft(gse: str) -> pd.DataFrame:
    records, current = [], None
    with gzip.open(RAW / f"{gse}_family.soft.gz", "rt", errors="replace") as handle:
        for line in handle:
            if line.startswith("^SAMPLE = "):
                if current:
                    records.append(current)
                current = {"gsm": line.strip().split(" = ", 1)[1], "gse": gse}
            elif current is not None and line.startswith("!Sample_"):
                key, value = line.strip().split(" = ", 1)
                key = key.removeprefix("!Sample_")
                if key == "characteristics_ch1" and ": " in value:
                    name, val = value.split(": ", 1)
                    current[re.sub(r"\W+", "_", name.lower()).strip("_")] = val
                elif key in {"title", "source_name_ch1"}:
                    current[key] = value
        if current:
            records.append(current)
    frame = pd.DataFrame(records)
    if frame.gsm.duplicated().any():
        raise AssertionError(f"duplicate GSM in {gse}")
    return frame


def annotations():
    h = pd.read_csv(CONFIG["hgnc"], sep="\t", dtype=str)
    ens = h.dropna(subset=["ensembl_gene_id"]).drop_duplicates("ensembl_gene_id").set_index("ensembl_gene_id").symbol.to_dict()
    ent = h.dropna(subset=["entrez_id"]).drop_duplicates("entrez_id").set_index("entrez_id").symbol.to_dict()
    ucsc = h.dropna(subset=["ucsc_id"]).drop_duplicates("ucsc_id").set_index("ucsc_id").symbol.to_dict()
    genes = pd.read_csv(CONFIG["canonical_genes"]).gene_symbol.astype(str).tolist()
    lengths = pd.read_csv(CONFIG["gene_lengths"]).drop_duplicates("gene_symbol").set_index("gene_symbol").exon_length.to_dict()
    return h, ens, ent, ucsc, genes, lengths


def aggregate_ids(frame: pd.DataFrame, id_col: str, sample_cols: list[str], mapping: dict[str, str], strip_version: bool = True) -> pd.DataFrame:
    ids = frame[id_col].astype(str)
    if strip_version:
        ids = ids.str.replace(r"\.\d+$", "", regex=True)
    symbols = ids.map(mapping)
    values = frame[sample_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    values.insert(0, "gene", symbols)
    values = values.dropna(subset=["gene"]).groupby("gene", sort=False).sum()
    return values


def counts_to_tpm(counts: pd.DataFrame, lengths: dict[str, float]) -> pd.DataFrame:
    length = pd.Series(counts.index.map(lengths), index=counts.index, dtype=float)
    valid = length.notna() & length.gt(0)
    rate = counts.loc[valid].div(length.loc[valid] / 1000, axis=0)
    return rate.div(rate.sum(axis=0), axis=1) * 1e6


def align(tpm: pd.DataFrame, genes: list[str]) -> np.ndarray:
    return tpm.reindex(genes).fillna(0).T.to_numpy(np.float32)


def split_lookup() -> dict[str, str]:
    answer = {}
    base = Path(CONFIG["archs4_splits"])
    for split in ("train", "validation", "unused"):
        frame = pd.read_parquet(base / f"{split}_samples.parquet", columns=["gsm_accession"])
        answer.update(dict.fromkeys(frame.gsm_accession.dropna().astype(str), split))
    return answer


def exposure_label(gsms: str, lookup: dict[str, str]) -> str:
    labels = sorted({lookup[g] for g in gsms.split(";") if g in lookup})
    return ";".join(labels) if labels else "absent"


def prepare_gse297(ens, genes, lengths):
    gse = "GSE297090"; meta = parse_soft(gse)
    frame = pd.read_csv(RAW / GEO_FILES[gse]); cols = frame.columns[3:].tolist()
    counts = aggregate_ids(frame, "GeneID", cols, ens)
    tpm = counts_to_tpm(counts, lengths)
    def key(title):
        donor = re.match(r"(J\d+)", title).group(1); state = "crypt" if "crypt" in title else "vill"
        suffix = "RadP" if "proton irradiated" in title else "RadG" if "gamma irradiated" in title else "CPro" if "proton mock" in title else "CG"
        return donor + state + suffix
    meta["matrix_col"] = meta.title.map(key); meta["sample_id"] = meta.gsm
    meta["donor"] = meta.title.str.extract(r"^(J\d+)"); meta["state"] = np.where(meta.title.str.contains("crypt"), "crypt", "villus")
    meta["arm"] = meta.title.map(lambda x: "proton_case" if "proton irradiated" in x else "gamma_case" if "gamma irradiated" in x else "proton_control" if "proton mock" in x else "gamma_control")
    return meta, tpm[meta.matrix_col], counts[meta.matrix_col]


def prepare_gse184(ent, genes):
    gse = "GSE184119"; meta = parse_soft(gse); extracted = RAW / "gse184"
    if not extracted.exists():
        extracted.mkdir()
        with tarfile.open(RAW / GEO_FILES[gse]) as tar:
            tar.extractall(extracted, filter="data")
    pieces = []
    lengths = None
    for path in sorted(extracted.glob("*.txt.gz")):
        gsm = path.name.split("_")[0]; x = pd.read_csv(path, sep="\t", dtype={"entrez_gene_id": str})
        x["gene"] = x.entrez_gene_id.map(ent); x = x.dropna(subset=["gene"])
        pieces.append(x.groupby("gene").read_count.sum().rename(gsm))
        if lengths is None: lengths = x.groupby("gene").gene_length.max()
    counts = pd.concat(pieces, axis=1).fillna(0)
    rate = counts.div(lengths.reindex(counts.index) / 1000, axis=0); tpm = rate.div(rate.sum(axis=0), axis=1) * 1e6
    meta["matrix_col"] = meta.gsm; meta["sample_id"] = meta.gsm; meta["cell_type"] = meta.cell_line
    meta["arm"] = meta.treatment.str.lower()
    return meta, tpm[meta.matrix_col], counts[meta.matrix_col]


def prepare_gse189(ucsc, genes):
    gse = "GSE189524"; meta = parse_soft(gse); path = RAW / GEO_FILES[gse]
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True); ws = wb["O v NO"]
    iterator = ws.iter_rows(values_only=True); header = list(next(iterator)); sample_cols = header[17:31]
    by_gene = {s: {} for s in sample_cols}; mapped_rows = 0; total_rows = 0
    for row in iterator:
        total_rows += 1; gene = ucsc.get(str(row[6] or row[9] or ""))
        if gene is None: continue
        mapped_rows += 1
        for idx, sample in enumerate(sample_cols, 17):
            value = float(row[idx] or 0); by_gene[sample][gene] = by_gene[sample].get(gene, 0.0) + value
    fpkm = pd.DataFrame(by_gene).fillna(0); tpm_lib = fpkm.div(fpkm.sum(axis=0), axis=1) * 1e6
    meta["matrix_col"] = "FPKM." + meta.title; meta["donor"] = meta.title.str.extract(r"^([FC][OF]\d+)")
    meta["condition"] = np.where(meta.condition.str.startswith("With"), "case", "control")
    correlations = []
    donor_tpm = {}
    rows = []
    for donor, group in meta.groupby("donor", sort=True):
        cols = group.matrix_col.tolist(); donor_tpm[donor] = tpm_lib[cols].mean(axis=1)
        correlations.append({"donor": donor, "library_1": cols[0], "library_2": cols[1], "pearson_log1p_tpm": np.corrcoef(np.log1p(tpm_lib[cols[0]]), np.log1p(tpm_lib[cols[1]]))[0, 1]})
        rows.append({"gse": gse, "gsm": ";".join(group.gsm), "sample_id": f"{gse}_{donor}", "matrix_col": donor, "donor": donor, "condition": group.condition.iloc[0], "title": ";".join(group.title)})
    pd.DataFrame(correlations).to_csv(RESULTS / "gse189524_duplicate_qc.csv", index=False)
    qc = {"total_transcript_rows": total_rows, "mapped_transcript_rows": mapped_rows, "mapped_fraction": mapped_rows / total_rows, "mapped_hgnc_genes": int(fpkm.shape[0]), "aggregation": "sum FPKM across UCSC transcripts mapped to one HGNC symbol; renormalize each library to TPM; arithmetic-mean paired libraries within donor", "donors": 7}
    (RESULTS / "gse189524_aggregation_qc.json").write_text(json.dumps(qc, indent=2) + "\n")
    return pd.DataFrame(rows), pd.DataFrame(donor_tpm), None


def prepare_simple_ensembl(gse, path, ens, lengths):
    meta = parse_soft(gse); frame = pd.read_csv(path, sep=None, engine="python", comment="#")
    id_col = frame.columns[0]; sample_cols = frame.columns[1:].tolist()
    if gse == "GSE211204":
        ids = frame[id_col].str.split("|").str[0]; frame = frame.assign(_id=ids); counts = aggregate_ids(frame, "_id", sample_cols, ens)
        meta["matrix_col"] = meta.title.str.extract(r"\[(S\d+)\]")[0]
        meta["arm"] = meta.treatment.str.lower(); meta["donor"] = meta.subject
    elif gse == "GSE113165":
        counts = aggregate_ids(frame, id_col, sample_cols, ens)
        meta["matrix_col"] = meta.hci_id; meta["arm"] = np.where(meta.time.str.startswith("post"), "case", "control"); meta["donor"] = meta.subject
    elif gse == "GSE276529":
        counts = aggregate_ids(frame, id_col, sample_cols, ens)
        meta["matrix_col"] = sample_cols
        meta["arm"] = np.where(meta.title.str.contains("Non-GIOP"), "control", "case"); meta["donor"] = meta.gsm
        meta["age_numeric"] = pd.to_numeric(meta.age)
    else: raise ValueError(gse)
    meta["sample_id"] = meta.gsm
    tpm = counts_to_tpm(counts, lengths)
    return meta, tpm[meta.matrix_col], counts[meta.matrix_col]


def prepare_osdr(gse, path, ens, lengths):
    meta = parse_soft(gse); frame = pd.read_csv(path); frame = frame.rename(columns={frame.columns[0]: "gene_id"})
    cols = frame.columns[1:].tolist(); counts = aggregate_ids(frame, "gene_id", cols, ens); tpm = counts_to_tpm(counts, lengths)
    meta["matrix_col"] = meta.gsm; meta["sample_id"] = meta.gsm
    if gse == "GSE297560":
        tr = meta.treatment.str.lower()
        meta["arm"] = np.select([tr.str.contains("silicon"), tr.str.contains("iron"), tr.str.contains("hydrogen|proton"), tr.str.contains("4gy.*gamma|gamma irradiation$") , tr.str.contains("gamma irradiation control"), tr.str.contains("irradiation control shipped")], ["silicon_case", "iron_case", "proton_case", "gamma_case", "gamma_control", "particle_control"], default="exclude")
    elif gse == "GSE234465":
        meta["age_group"] = np.where(meta.title.str.contains("YA"), "young", "old")
        meta["arm"] = np.where(meta.title.str.startswith("GC-"), "control", "case")
    elif gse == "GSE273868":
        number = meta.gsm.str.extract(r"(\d+)$")[0].astype(int)
        meta["time_week"] = np.where(number <= 8438335, 1, 2)
        meta["arm"] = np.where(number % 2 == 1, "case", "control")
    return meta, tpm[meta.matrix_col], counts[meta.matrix_col]


def add_contrast(rows, meta, dataset, condition, contrast_id, selector, case_value="case", block_cols=()):
    group = meta.loc[selector].copy()
    for _, r in group.iterrows():
        item = {"dataset": dataset, "condition_group": condition, "contrast_id": contrast_id, "sample_id": r.sample_id, "gsm": r.gsm, "role": "case" if r.arm == case_value else "control"}
        for col in block_cols: item[col] = r.get(col, "")
        rows.append(item)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--skip-download", action="store_true"); args = parser.parse_args()
    for directory in (RAW, PREP, RESULTS, MANIFESTS): directory.mkdir(parents=True, exist_ok=True)
    if not args.skip_download: acquire()
    _, ens, ent, ucsc, genes, lengths = annotations(); split = split_lookup()
    prepared = {}
    prepared["GSE297090"] = prepare_gse297(ens, genes, lengths)
    prepared["GSE184119"] = prepare_gse184(ent, genes)
    prepared["GSE189524"] = prepare_gse189(ucsc, genes)
    for gse in ("GSE276529", "GSE211204", "GSE113165"):
        prepared[gse] = prepare_simple_ensembl(gse, RAW / GEO_FILES[gse], ens, lengths)
    for gse, (_, name) in OSDR_FILES.items():
        prepared[gse] = prepare_osdr(gse, RAW / name, ens, lengths)

    all_meta = []; coverage = []
    for gse, (meta, tpm, counts) in prepared.items():
        meta = meta.copy(); meta["pretraining_status"] = meta.gsm.map(lambda x: exposure_label(str(x), split)); prepared[gse] = (meta, tpm, counts)
        matrix = align(tpm, genes); np.save(PREP / f"{gse}_log1p_tpm.npy", np.log1p(matrix).astype(np.float32))
        pd.DataFrame({"sample_id": meta.sample_id, "matrix_row": np.arange(len(meta))}).to_csv(PREP / f"{gse}_matrix_rows.csv", index=False)
        if counts is not None:
            counts.reindex(genes).fillna(0).to_parquet(PREP / f"{gse}_counts.parquet")
            counts.reindex(genes).fillna(0).to_csv(PREP / f"{gse}_counts.csv.gz", compression="gzip")
        else:
            pd.DataFrame(matrix.T, index=genes, columns=meta.sample_id).to_csv(PREP / f"{gse}_tpm.csv.gz", compression="gzip")
        all_meta.append(meta.assign(dataset=gse))
        coverage.append({"dataset": gse, "samples": len(meta), "mapped_bridge_genes": len(set(tpm.index) & set(genes)), "bridge_genes": len(genes)})
    sample_manifest = pd.concat(all_meta, ignore_index=True, sort=False)
    sample_manifest.to_csv(MANIFESTS / "sample_manifest.csv", index=False); sample_manifest.to_parquet(MANIFESTS / "sample_manifest.parquet", index=False)
    pd.DataFrame(coverage).to_csv(RESULTS / "expression_coverage.csv", index=False)

    contrast_rows = []
    m = prepared["GSE297090"][0]
    add_contrast(contrast_rows,m,"GSE297090","Radiation","GSE297090_gamma",m.arm.isin(["gamma_case","gamma_control"]),"gamma_case",("donor","state"))
    add_contrast(contrast_rows,m,"GSE297090","Radiation","GSE297090_proton",m.arm.isin(["proton_case","proton_control"]),"proton_case",("donor","state"))
    m=prepared["GSE184119"][0]; add_contrast(contrast_rows,m,"GSE184119","Radiation","GSE184119_10Gy",m.arm.isin(["0gy","10gy"]),"10gy",("cell_type",))
    m=prepared["GSE297560"][0]
    for particle in ("silicon","iron","proton"):
        add_contrast(contrast_rows,m,"GSE297560","Radiation",f"GSE297560_{particle}",m.arm.isin([f"{particle}_case","particle_control"]),f"{particle}_case")
    add_contrast(contrast_rows,m,"GSE297560","Radiation","GSE297560_gamma",m.arm.isin(["gamma_case","gamma_control"]),"gamma_case")
    m=prepared["GSE189524"][0]; m=m.assign(arm=m.condition); add_contrast(contrast_rows,m,"GSE189524","Bone loss","GSE189524_OP",m.index==m.index,"case",("donor",))
    m=prepared["GSE276529"][0]; add_contrast(contrast_rows,m,"GSE276529","Bone loss","GSE276529_GIOP",m.index==m.index,"case",("age_numeric",))
    m=prepared["GSE273868"][0]
    for week in (1,2): add_contrast(contrast_rows,m,"GSE273868","Bone loss",f"GSE273868_week{week}",m.time_week.eq(week),"case",("time_week",))
    m=prepared["GSE211204"][0]; add_contrast(contrast_rows,m,"GSE211204","Muscle atrophy","GSE211204_ULLS",m.arm.isin(["baseline","ulls"]),"ulls",("donor",))
    m=prepared["GSE113165"][0]; add_contrast(contrast_rows,m,"GSE113165","Muscle atrophy","GSE113165_bedrest",m.index==m.index,"case",("donor","age","sex","susceptibility"))
    m=prepared["GSE234465"][0]
    for age in ("young","old"): add_contrast(contrast_rows,m,"GSE234465","Muscle atrophy",f"GSE234465_{age}",m.age_group.eq(age),"case",("age_group",))
    contrasts = pd.DataFrame(contrast_rows)
    contrast_sizes = contrasts.groupby(["contrast_id","role"]).size().unstack(fill_value=0)
    say("contrast sizes\n" + contrast_sizes.to_string())
    if contrast_sizes.min().min() < 2: raise AssertionError("contrast with fewer than 2 samples in an arm")
    contrasts.to_csv(MANIFESTS / "contrast_members.csv", index=False); contrasts.to_parquet(MANIFESTS / "contrast_members.parquet", index=False)

    # GSE211204 strict sensitivity removes a subject if either endpoint was in train/validation.
    g = contrasts.loc[contrasts.dataset.eq("GSE211204")].copy(); exposed_gsm=set(sample_manifest.loc[(sample_manifest.dataset.eq("GSE211204")) & sample_manifest.pretraining_status.isin(["train","validation"]),"gsm"])
    exposed_subjects=set(g.loc[g.gsm.isin(exposed_gsm),"donor"]); sensitivity=g.loc[~g.donor.isin(exposed_subjects)].copy()
    sensitivity.to_csv(MANIFESTS / "GSE211204_unseen_subject_sensitivity.csv", index=False)
    if sensitivity.donor.nunique() != 7: say(f"WARNING expected 7 endpoint-unseen subjects, found {sensitivity.donor.nunique()}")
    provenance={"config":CONFIG,"canonical_gene_count":len(genes),"dataset_count":len(prepared),"contrast_count":contrasts.contrast_id.nunique(),"sample_manifest_rows":len(sample_manifest),"gse211204_exposed_subjects":sorted(exposed_subjects),"gse211204_unseen_subjects":sorted(set(sensitivity.donor)),"no_embeddings_or_statistics_run":True}
    (RESULTS / "preexecution_qc_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    say(f"prepared {len(prepared)} datasets, {len(sample_manifest)} samples/donors, {contrasts.contrast_id.nunique()} contrasts")


if __name__ == "__main__": main()
