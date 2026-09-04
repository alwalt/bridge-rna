#!/usr/bin/env python3
"""Prepare controlled same-RNA library-preparation datasets for Task 4.

The Chen 2020 T-cell dataset is downloaded from its official Figshare record.
It is converted from true counts to natural log1p(TPM), aligned to the exact
BridgeRNA vocabulary, and encoded with the frozen checkpoint. Other datasets
must be supplied as the same manifest/array schema; no pair is inferred from a
sample name alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.fm_embed.encode import encode_matrix
from src.fm_embed.model import load_expression_performer
from src.fm_embed.vocab import load_canonical_genes

URLS = {
    "counts": "https://ndownloader.figshare.com/files/23820140",
    "metadata": "https://ndownloader.figshare.com/files/24668162",
    "readme": "https://ndownloader.figshare.com/files/24668165",
}


def download(url: str, path: Path) -> None:
    if path.exists(): return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url} -> {path}", flush=True)
    urllib.request.urlretrieve(url, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def gtf_id_to_symbol(path: Path) -> dict[str, str]:
    import gzip
    opener = gzip.open if path.suffix == ".gz" else open
    result: dict[str, str] = {}
    with opener(path, "rt") as handle:
        for line in handle:
            if line.startswith("#"): continue
            fields = line.rstrip().split("\t")
            if len(fields) < 9 or fields[2] != "gene": continue
            gid = re.search(r'gene_id "([^"]+)', fields[8]); symbol = re.search(r'gene_name "([^"]+)', fields[8])
            if gid and symbol: result[gid.group(1).split(".")[0]] = symbol.group(1).upper()
    return result


def counts_to_log1p_tpm(counts: pd.DataFrame, lengths: pd.Series) -> pd.DataFrame:
    length_kb = lengths.reindex(counts.index).astype(float) / 1000.0
    missing = length_kb.isna()
    nonzero_missing = counts.loc[missing].sum(axis=1).gt(0) if missing.any() else pd.Series(dtype=bool)
    if nonzero_missing.any():
        raise ValueError(f"Observed counts but no gene length for: {nonzero_missing[nonzero_missing].index.tolist()}")
    # A vocabulary gene absent from the source and annotation remains exactly
    # zero regardless of its placeholder length.
    length_kb = length_kb.fillna(1.0)
    rates = counts.div(length_kb, axis=0)
    denom = rates.sum(axis=0)
    if (denom <= 0).any(): raise ValueError("Zero TPM denominator")
    return np.log1p(rates.div(denom, axis=1) * 1e6).T


def prepare_chen(work: Path, device: str, batch_size: int) -> tuple[Path, Path]:
    dataset_dir = work / "datasets/chen_2020_tcells"
    if (dataset_dir / "manifest.parquet").exists() and (dataset_dir / "bridgerna_embeddings.npy").exists():
        print("[cache] Chen 2020 prepared dataset", flush=True)
        return dataset_dir / "manifest.parquet", dataset_dir / "bridgerna_embeddings.npy"
    source = work / "sources/chen_2020_tcells"
    for key, url in URLS.items():
        ext = {"counts": ".txt", "metadata": ".xlsx", "readme": ".txt"}[key]
        download(url, source / f"{key}{ext}")
    meta = pd.read_excel(source / "metadata.xlsx")
    raw = pd.read_csv(source / "counts.txt", sep="\t", index_col=0)
    raw.index = raw.index.astype(str).str.extract(r"^(ENSG\d+)", expand=False)
    mapping = gtf_id_to_symbol(ROOT / "data/gencode/gencode.v36.annotation.gtf.gz")
    symbols = raw.index.to_series().map(mapping)
    raw = raw.loc[symbols.notna()].copy(); raw.index = symbols[symbols.notna()].values
    raw = raw.groupby(level=0).sum()
    genes = load_canonical_genes(ROOT / "data/ensembl/canonical_genes.csv")
    lengths = pd.read_csv(ROOT / "data/gencode/gencode_v49_gene_exon_lengths.csv").set_index("gene_symbol").exon_length
    available = [g for g in genes if g in raw.index and g in lengths.index]
    aligned_counts = raw.reindex(genes, fill_value=0)
    # Missing-but-valid vocabulary genes have zero counts; all vocabulary genes
    # have canonical lengths. This mirrors standard zero-fill alignment.
    x = counts_to_log1p_tpm(aligned_counts, lengths)
    x = x.reindex(meta.NAME).astype(np.float32)
    out_meta = pd.DataFrame({
        "sample_id": meta.NAME.astype(str), "dataset": "chen_2020_tcells",
        "study": "doi:10.1038/s41597-020-00719-4", "pair_id": meta.DONOR_ID.astype(str),
        "library_prep": meta["LIBRARY PROTOCOL"].map({"polyA-selected": "polyA", "rRNA-depleted": "ribo"}),
        "species": "human", "tissue": "naive CD4+ T cell", "donor_id": meta.DONOR_ID.astype(str),
        "role": "train", "same_rna_verified": True,
    })
    assert len(out_meta) == 80 and out_meta.pair_id.nunique() == 40
    assert (out_meta.groupby(["pair_id", "library_prep"]).size() == 1).all()
    model, torch_device = load_expression_performer(ROOT / "model/r7hnr92k/best_model.pt", ROOT / "model/r7hnr92k/config.json", len(genes), device)
    z = encode_matrix(model, torch_device, x.to_numpy(), batch_size=batch_size, label="chen BridgeRNA")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    np.save(dataset_dir / "log1p_tpm.npy", x.to_numpy())
    np.save(dataset_dir / "bridgerna_embeddings.npy", z.astype(np.float32))
    out_meta.to_parquet(dataset_dir / "manifest.parquet", index=False)
    provenance = {"source_urls": URLS, "source_sha256": {p.name: sha256(p) for p in source.iterdir()},
                  "samples": 80, "pairs": 40, "model_genes": len(genes), "genes_observed": len(available),
                  "preprocessing": "counts -> gene-length TPM -> natural log1p; absent vocabulary genes zero-filled",
                  "checkpoint": "model/r7hnr92k/best_model.pt"}
    (dataset_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return dataset_dir / "manifest.parquet", dataset_dir / "bridgerna_embeddings.npy"


def prepare_srp127360(work: Path, device: str, batch_size: int) -> tuple[Path, Path]:
    from scipy.io import mmread
    source = work / "srp127360"
    required = [source / x for x in ["counts.mtx", "genes.txt", "samples.txt"]]
    if not all(p.exists() for p in required):
        raise FileNotFoundError("Run pipeline/download_srp127360_recount3.R before --include-srp127360")
    genes_in = pd.Series((source / "genes.txt").read_text().splitlines()).str.split(".").str[0]
    samples = (source / "samples.txt").read_text().splitlines()
    counts = np.asarray(mmread(source / "counts.mtx").todense(), dtype=np.float64)
    raw = pd.DataFrame(counts, index=genes_in, columns=samples).groupby(level=0).sum()
    mapping = gtf_id_to_symbol(ROOT / "data/gencode/gencode.v36.annotation.gtf.gz")
    symbols = raw.index.to_series().map(mapping); raw = raw.loc[symbols.notna()]; raw.index = symbols[symbols.notna()].values; raw = raw.groupby(level=0).sum()
    model_genes = load_canonical_genes(ROOT / "data/ensembl/canonical_genes.csv")
    lengths = pd.read_csv(ROOT / "data/gencode/gencode_v49_gene_exon_lengths.csv").set_index("gene_symbol").exon_length
    x = counts_to_log1p_tpm(raw.reindex(model_genes, fill_value=0), lengths)
    run_to_name = {
        "SRR6410613":"B_M_1","SRR6410614":"B_M_2","SRR6410611":"B_M_3","SRR6410612":"B_M_4",
        "SRR6410617":"B_T_1","SRR6410618":"B_T_2","SRR6410615":"B_T_3","SRR6410616":"B_T_4",
        "SRR6410605":"C_M_1","SRR6410606":"C_M_2","SRR6410603":"C_M_3","SRR6410604":"C_M_4",
        "SRR6410609":"C_T_1","SRR6410610":"C_T_2","SRR6410607":"C_T_3","SRR6410608":"C_T_4"}
    out_meta=pd.DataFrame({"sample_id":samples}); out_meta["library_name"]=out_meta.sample_id.map(run_to_name)
    if out_meta.library_name.isna().any(): raise ValueError("Unexpected SRP127360 run")
    out_meta["dataset"]="zhao_2018_srp127360"; out_meta["study"]="SRP127360"; out_meta["pair_id"]=out_meta.library_name.str[0].map({"B":"pooled_blood","C":"colon"}); out_meta["library_prep"]=out_meta.library_name.str.split("_").str[1].map({"M":"polyA","T":"ribo"}); out_meta["species"]="human";out_meta["tissue"]=out_meta.pair_id;out_meta["donor_id"]=out_meta.pair_id;out_meta["role"]="test";out_meta["same_rna_verified"]=True
    x=x.reindex(samples).astype(np.float32)
    model, torch_device=load_expression_performer(ROOT/"model/r7hnr92k/best_model.pt",ROOT/"model/r7hnr92k/config.json",len(model_genes),device)
    z=encode_matrix(model,torch_device,x.to_numpy(),batch_size=batch_size,label="SRP127360 BridgeRNA")
    dataset_dir=work/"datasets/zhao_2018_srp127360";dataset_dir.mkdir(parents=True,exist_ok=True)
    np.save(dataset_dir/"log1p_tpm.npy",x.to_numpy());np.save(dataset_dir/"bridgerna_embeddings.npy",z.astype(np.float32));out_meta.to_parquet(dataset_dir/"manifest.parquet",index=False)
    (dataset_dir/"provenance.json").write_text(json.dumps({"source":"recount3 SRP127360","paper":"doi:10.1038/s41598-018-23226-4","samples":16,"biological_source_RNAs":2,"warning":"Four libraries per protocol are technical replicates; pair metrics use protocol centroids, not arbitrary replicate matching."},indent=2)+"\n")
    return dataset_dir/"manifest.parquet",dataset_dir/"bridgerna_embeddings.npy"


def write_audit(results: Path) -> None:
    rows = [
        {"dataset": "Chen 2020 T cells", "accession": "doi:10.1038/s41597-020-00719-4; syn22250947; figshare 12646238.v5", "species": "human", "tissue": "naive CD4+ T cells", "n_pairs": 40, "same_rna": "yes", "polyA_protocol": "TruSeq RNA Library Prep v2", "ribo_protocol": "TruSeq Stranded Total RNA + Ribo-Zero Gold", "platform": "Illumina paired-end", "role": "TRAIN"},
        {"dataset": "GSE150097 freeze-thaw", "accession": "GSE150097 / SRP257988", "species": "human", "tissue": "blood leukocytes", "n_pairs": "pending authoritative pair map", "same_rna": "candidate; verify pair IDs", "polyA_protocol": "TruSeq Stranded mRNA", "ribo_protocol": "TruSeq Stranded Total RNA + Ribo-Zero Gold", "platform": "HiSeq 4000 (SE50 vs PE100 confounded)", "role": "VALIDATION candidate"},
        {"dataset": "Zhao 2018 blood/colon", "accession": "SRP127360", "species": "human", "tissue": "pooled blood; colon", "n_pairs": 2, "same_rna": "yes; four technical libraries/arm", "polyA_protocol": "TruSeq Stranded mRNA", "ribo_protocol": "Globin-Zero blood; Ribo-Zero Gold colon", "platform": "NextSeq 500 PE", "role": "TEST"},
    ]
    pd.DataFrame(rows).to_csv(results / "controlled_dataset_audit.csv", index=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0"); p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--include-srp127360", action="store_true")
    p.add_argument("--work", type=Path, default=HERE / "work"); p.add_argument("--results", type=Path, default=HERE / "results/task4a_data_audit")
    args = p.parse_args(); args.results.mkdir(parents=True, exist_ok=True)
    write_audit(args.results)
    manifest, embedding = prepare_chen(args.work, args.device, args.batch_size)
    print(f"[complete] manifest={manifest}\n[complete] embeddings={embedding}", flush=True)
    if args.include_srp127360:
        manifest, embedding = prepare_srp127360(args.work, args.device, args.batch_size)
        print(f"[complete] manifest={manifest}\n[complete] embeddings={embedding}", flush=True)


if __name__ == "__main__": main()
