#!/usr/bin/env python3
"""Freeze signed Bridge/DE condition signatures without recomputing upstream analyses."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
OUT = HERE / "results"
DRUG = ROOT / "benchmarks/drug_discovery/results"
DEWEERD = ROOT / "benchmarks/deweerd_replication/results"

DISEASE = {"GSE138614": "Multiple sclerosis", "GSE101794": "Crohn's disease", "GSE72509": "Systemic lupus erythematosus"}
SPACE = {
    "GSE297090_gamma": "Radiation injury", "GSE297090_proton": "Radiation injury", "GSE184119_10Gy": "Radiation injury",
    "GSE297560_gamma": "Radiation injury", "GSE297560_proton": "Radiation injury",
    "GSE297560_iron": "Radiation injury", "GSE297560_silicon": "Radiation injury",
    "GSE189524_OP": "Bone loss", "GSE276529_GIOP": "Bone loss",
    "GSE273868_week1": "Bone loss", "GSE273868_week2": "Bone loss",
    "GSE211204_ULLS": "Muscle atrophy", "GSE113165_bedrest": "Muscle atrophy",
    "GSE234465_old": "Muscle atrophy", "GSE234465_young": "Muscle atrophy",
}
ROLES = {"GSE297090":"terrestrial_discovery", "GSE184119":"terrestrial_replication", "GSE297560":"osdr",
         "GSE189524":"terrestrial_discovery", "GSE276529":"terrestrial_replication", "GSE273868":"osdr",
         "GSE211204":"terrestrial_discovery", "GSE113165":"terrestrial_replication", "GSE234465":"osdr"}


def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(4<<20),b""):h.update(b)
    return h.hexdigest()


def dataset_from_contrast(contrast):
    return contrast.split("_")[0]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows=[]; hashes={}
    for accession, condition in DISEASE.items():
        bp=DEWEERD/f"bridge/{accession}_gene_ranking.parquet"
        dp=DEWEERD/f"differential_expression/{accession}_gene_ranking.csv.gz"
        b=pd.read_parquet(bp); d=pd.read_csv(dp)
        for method,frame,score in [("Bridge",b,"signed_bridge_score"),("DE",d,"t")]:
            f=frame[["gene",score]].rename(columns={score:"signed_score"}).copy()
            f["eligible"]=True; f["condition"]=condition;f["contrast_id"]=accession;f["dataset"]=accession
            f["role"]="published_disease";f["method"]=method;rows.append(f)
        hashes[str(bp.relative_to(ROOT))]=sha(bp);hashes[str(dp.relative_to(ROOT))]=sha(dp)
    bp=DRUG/"bridge/contrast_gene_scores.parquet"; bridge=pd.read_parquet(bp);hashes[str(bp.relative_to(ROOT))]=sha(bp)
    for contrast,condition in SPACE.items():
        dataset=dataset_from_contrast(contrast)
        b=bridge[bridge.contrast_id.eq(contrast)][["gene","signed_bridge_score"]].rename(columns={"signed_bridge_score":"signed_score"})
        b["eligible"]=True;b["condition"]=condition;b["contrast_id"]=contrast;b["dataset"]=dataset;b["role"]=ROLES[dataset];b["method"]="Bridge";rows.append(b)
        dp=DRUG/f"differential_expression/{contrast}.csv.gz";d=pd.read_csv(dp)
        d=d[["gene","t","eligible"]].rename(columns={"t":"signed_score"})
        d["condition"]=condition;d["contrast_id"]=contrast;d["dataset"]=dataset;d["role"]=ROLES[dataset];d["method"]="DE";rows.append(d)
        hashes[str(dp.relative_to(ROOT))]=sha(dp)
    signatures=pd.concat(rows,ignore_index=True)
    signatures["absolute_score"]=signatures.signed_score.abs()
    signatures["rank"]=signatures.groupby(["condition","contrast_id","method"])["absolute_score"].rank(method="first",ascending=False)
    signatures["top500"]=signatures["eligible"] & signatures["rank"].le(500)
    signatures.to_parquet(OUT/"frozen_condition_signatures.parquet",index=False)

    # Exact contrast-level complementarity from the already-frozen rankings.
    partition_rows=[]
    for (condition,contrast),group in signatures.groupby(["condition","contrast_id"]):
        b=set(group[(group.method.eq("Bridge"))&group.top500].gene);d=set(group[(group.method.eq("DE"))&group.top500].gene)
        for partition,genes in [("Bridge-only",b-d),("shared",b&d),("DE-only",d-b)]:
            for gene in genes: partition_rows.append({"condition":condition,"contrast_id":contrast,"partition":partition,"gene":gene})
    pd.DataFrame(partition_rows).to_csv(OUT/"frozen_condition_partitions.csv.gz",index=False,compression="gzip")

    # Recurrence and direction stability are descriptive annotations, not new modules.
    top=signatures[signatures.top500].copy(); top["sign"]=np.sign(top.signed_score)
    recurrence=[]
    for (condition,method,gene),group in top.groupby(["condition","method","gene"]):
        datasets=sorted(group.dataset.unique()); signs=group.sign[group.sign.ne(0)]
        recurrence.append({"condition":condition,"method":method,"gene":gene,"contrast_count":group.contrast_id.nunique(),
                           "dataset_count":len(datasets),"contrasts":";".join(sorted(group.contrast_id.unique())),
                           "datasets":";".join(datasets),"includes_osdr":group.role.eq("osdr").any(),
                           "direction_class":"stable" if signs.nunique()<=1 else "context_changing",
                           "majority_sign":int(np.sign(signs.sum())) if len(signs) else 0,
                           "majority_sign_fraction":float(max((signs>0).mean(),(signs<0).mean())) if len(signs) else np.nan})
    pd.DataFrame(recurrence).to_csv(OUT/"condition_gene_recurrence_direction.csv.gz",index=False,compression="gzip")
    (OUT/"condition_signature_provenance.json").write_text(json.dumps({
        "space_primary_contrasts":SPACE,"disease_contrasts":DISEASE,
        "top_module_definition":"frozen absolute-score top 500 per contrast and method",
        "direction_rule":"signed_bridge_score for Bridge and moderated t statistic for DE; no directions inferred from drug data",
        "recurrent_direction_rule":"stable only when every contributing frozen top-500 contrast has the same nonzero sign",
        "excluded_primary_duplicate":"GSE211204_ULLS_unseen is a leakage sensitivity subset, not an independent condition contrast",
        "input_hashes":hashes,"upstream_steps_rerun":False},indent=2)+"\n")


if __name__=="__main__":main()
