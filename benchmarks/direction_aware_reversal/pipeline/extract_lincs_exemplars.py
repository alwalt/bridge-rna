#!/usr/bin/env python3
"""Verify and extract exemplar compound signatures from LINCS GSE92742 Level 5."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
WORK = HERE / "work/lincs/GSE92742"
META = HERE / "work/lincs_metadata/GSE92742"
OUT = HERE / "results"
GZ = WORK / "GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz"
GCTX = WORK / "GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx"
EXPECTED_SHA512 = "6a3115cf3aaa402bb1bc098678b52b9c03632fc21087050e3e1a56c24a60196c4809bbac3797978dab63a47d955ff7a2c6110025a6e6d3cef4c5f6e010208a2a"


def digest(path, algorithm="sha512"):
    h=hashlib.new(algorithm)
    with path.open("rb") as f:
        for b in iter(lambda:f.read(16<<20),b""):h.update(b)
    return h.hexdigest()


def decode(values):
    return np.asarray([x.decode() if isinstance(x,bytes) else str(x) for x in values])


def one(pattern):
    paths=list(META.glob(pattern))
    if len(paths)!=1:raise RuntimeError((pattern,paths))
    return paths[0]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    actual=digest(GZ)
    if actual!=EXPECTED_SHA512:raise RuntimeError(f"SHA512 mismatch: {actual}")
    if not GCTX.exists():raise RuntimeError(f"Decompress {GZ} before extraction")
    canonical=set(pd.read_csv("/home/walt/bridge-rna/data/ensembl/canonical_genes.csv").gene_symbol.astype(str))
    gene_info=pd.read_csv(one("*gene_info*.gz"),sep="\t",low_memory=False)
    gene_info["pr_gene_id_str"]=gene_info.pr_gene_id.astype(str)
    gene_lookup=gene_info.set_index("pr_gene_id_str")
    sig=pd.read_csv(one("*sig_info*.gz"),sep="\t",low_memory=False)
    metrics=pd.read_csv(one("*sig_metrics*.gz"),sep="\t",low_memory=False)
    selected_meta=sig.merge(metrics[["sig_id","is_exemplar","distil_nsample","distil_cc_q75","distil_ss","tas"]],on="sig_id",how="left")
    selected_ids=set(selected_meta.loc[selected_meta.pert_type.eq("trt_cp")&selected_meta.is_exemplar.eq(1),"sig_id"])
    with h5py.File(GCTX,"r") as h:
        matrix=h["0/DATA/0/matrix"]
        row_ids=decode(h["0/META/ROW/id"][:]); col_ids=decode(h["0/META/COL/id"][:])
        if matrix.shape not in [(len(row_ids),len(col_ids)),(len(col_ids),len(row_ids))]:
            raise RuntimeError((matrix.shape,len(row_ids),len(col_ids)))
        # GEO's GCTX stores the logical gene x signature matrix transposed in
        # HDF5 (signature rows, gene columns).  Keep the public row/column IDs
        # authoritative and handle either physical orientation explicitly.
        transposed=matrix.shape==(len(col_ids),len(row_ids))
        row_frame=pd.DataFrame({"matrix_row":np.arange(len(row_ids)),"pr_gene_id_str":row_ids}).join(gene_lookup,on="pr_gene_id_str")
        row_frame=row_frame[row_frame.pr_is_bing.eq(1)&row_frame.pr_gene_symbol.astype(str).isin(canonical)].copy()
        row_frame=row_frame.drop_duplicates("pr_gene_symbol").sort_values("matrix_row")
        row_idx=row_frame.matrix_row.to_numpy(dtype=int)
        keep_col=np.fromiter((x in selected_ids for x in col_ids),dtype=bool,count=len(col_ids))
        selected_col_idx=np.flatnonzero(keep_col)
        output=np.lib.format.open_memmap(WORK/"lincs_gse92742_exemplar_bing.npy",mode="w+",dtype="float32",
                                         shape=(len(selected_col_idx),len(row_idx)))
        cursor=0;chunk=512
        for start in range(0,len(selected_col_idx),chunk):
            sig_idx=selected_col_idx[start:start+chunk]
            if transposed:
                # h5py only supports one fancy-indexed axis at a time.
                block=np.asarray(matrix[sig_idx,:],dtype=np.float32)[:,row_idx]
            else:
                block=np.asarray(matrix[:,sig_idx],dtype=np.float32)[row_idx,:].T
            output[cursor:cursor+len(sig_idx)]=block
            cursor+=len(sig_idx)
        output.flush()
    if cursor!=len(selected_col_idx):raise RuntimeError((cursor,len(selected_col_idx)))
    aligned=pd.DataFrame({"matrix_col":selected_col_idx,"sig_id":col_ids[selected_col_idx]})
    aligned=aligned.merge(selected_meta,on="sig_id",how="left",validate="one_to_one")
    if aligned.pert_id.isna().any():raise RuntimeError("Missing signature metadata")
    aligned.to_parquet(OUT/"lincs_exemplar_contexts.parquet",index=False)
    row_frame[["matrix_row","pr_gene_id_str","pr_gene_symbol","pr_is_lm","pr_is_bing"]].rename(
        columns={"pr_gene_symbol":"gene"}).to_csv(OUT/"lincs_exemplar_genes.csv",index=False)
    provenance={"source_accession":"GSE92742","source_file":GZ.name,"source_sha512":actual,
                "source_checksum_verified":True,"source_logical_matrix_shape":[len(row_ids),len(col_ids)],
                "source_hdf5_matrix_shape":list(matrix.shape),"source_hdf5_transposed":transposed,
                "selected_contexts":len(selected_col_idx),"selected_genes":len(row_idx),
                "filters":{"pert_type":"trt_cp","is_exemplar":1,"genes":"BING intersect canonical"},
                "extracted_matrix":str((WORK/"lincs_gse92742_exemplar_bing.npy").relative_to(ROOT)),
                "extracted_matrix_sha256":digest(WORK/"lincs_gse92742_exemplar_bing.npy","sha256")}
    (OUT/"lincs_extraction_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    print(json.dumps(provenance,indent=2))


if __name__=="__main__":main()
