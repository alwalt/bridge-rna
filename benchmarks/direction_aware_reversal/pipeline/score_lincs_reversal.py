#!/usr/bin/env python3
"""Score frozen condition signatures against LINCS exemplar perturbations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
OUT = HERE / "results"
WORK = HERE / "work/lincs/GSE92742"
MATRIX = WORK / "lincs_gse92742_exemplar_bing.npy"


def digest(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(16<<20),b""):h.update(b)
    return h.hexdigest()


def add_definition(definitions, members, key, condition, contrast, dataset, method, analysis, frame, weight_mode="linear"):
    frame=frame.dropna(subset=["signed_score"]).copy().sort_values("rank")
    frame=frame[frame.signed_score.ne(0)]
    if weight_mode=="linear":
        n=max(len(frame),1); magnitude=1-(frame["rank"].rank(method="first").to_numpy()-1)/n
    elif weight_mode=="inverse_sqrt":
        magnitude=1/np.sqrt(frame["rank"].to_numpy())
    else:
        magnitude=np.ones(len(frame))
    for row,mag in zip(frame.itertuples(),magnitude):
        members.append({"definition_id":key,"gene":row.gene,"weight":float(np.sign(row.signed_score)*mag),
                        "condition_sign":int(np.sign(row.signed_score)),"source_rank":float(row.rank)})
    definitions.append({"definition_id":key,"condition":condition,"contrast_id":contrast,"dataset":dataset,
                        "method":method,"analysis":analysis,"source_gene_count":len(frame)})


def main():
    genes=pd.read_csv(OUT/"lincs_exemplar_genes.csv");gene_index={g:i for i,g in enumerate(genes.gene)}
    sig=pd.read_parquet(OUT/"frozen_condition_signatures.parquet")
    partitions=pd.read_csv(OUT/"frozen_condition_partitions.csv.gz")
    recurrence=pd.read_csv(OUT/"condition_gene_recurrence_direction.csv.gz")
    definitions=[];members=[]
    for (condition,contrast,method),group in sig.groupby(["condition","contrast_id","method"]):
        dataset=group.dataset.iloc[0]
        top=group[group.top500 & group.gene.isin(gene_index)].copy()
        add_definition(definitions,members,f"top500|{method}|{contrast}",condition,contrast,dataset,method,"top500",top,"linear")
        full=group[group.eligible & group.gene.isin(gene_index)].copy()
        add_definition(definitions,members,f"full|{method}|{contrast}",condition,contrast,dataset,method,"full_ranking",full,"inverse_sqrt")
        for partition in (["Bridge-only","shared"] if method=="Bridge" else ["DE-only"]):
            pg=set(partitions[(partitions.contrast_id.eq(contrast))&partitions.partition.eq(partition)].gene)
            part=top[top.gene.isin(pg)].copy()
            add_definition(definitions,members,f"partition|{partition}|{contrast}",condition,contrast,dataset,method,partition,part,"linear")

    # Stable recurrent programs use only genes whose sign is invariant across contributing frozen contrasts.
    for condition in ["Radiation injury","Bone loss","Muscle atrophy"]:
        for method in ["Bridge","DE"]:
            r=recurrence[(recurrence.condition.eq(condition))&(recurrence.method.eq(method))&
                         recurrence.dataset_count.ge(2)&recurrence.direction_class.eq("stable")].copy()
            if not len(r):continue
            r=r.sort_values(["dataset_count","contrast_count","gene"],ascending=[False,False,True])
            r["signed_score"]=r.majority_sign;r["rank"]=np.arange(1,len(r)+1)
            r=r[r.gene.isin(gene_index)]
            add_definition(definitions,members,f"stable_recurrent|{method}|{condition}",condition,"cross_study_stable",condition,method,"stable_recurrent",r,"linear")
        r=recurrence[(recurrence.condition.eq(condition))&(recurrence.method.eq("Bridge"))&
                     recurrence.dataset_count.ge(2)&recurrence.direction_class.eq("stable")].copy()
        partition_condition={"Radiation injury":"Radiation","Bone loss":"Bone loss","Muscle atrophy":"Muscle atrophy"}[condition]
        partition_frame=pd.read_csv(ROOT/"benchmarks/drug_discovery/results/expanded_chembl_sensitivity/recurrent_partition_genes.csv")
        bo=set(partition_frame[(partition_frame.condition.eq(partition_condition))&
                               partition_frame.partition.eq("Bridge-only")].gene)
        r=r[r.gene.isin(bo)];r["signed_score"]=r.majority_sign
        r=r.sort_values(["dataset_count","contrast_count","gene"],ascending=[False,False,True]);r["rank"]=np.arange(1,len(r)+1)
        r=r[r.gene.isin(gene_index)]
        if len(r):add_definition(definitions,members,f"stable_recurrent_bridge_only|{condition}",condition,"cross_study_stable",condition,"Bridge","stable_recurrent_bridge_only",r,"linear")

    definitions=pd.DataFrame(definitions);members=pd.DataFrame(members)
    counts=members.groupby("definition_id").gene.nunique()
    definitions["lincs_overlap_gene_count"]=definitions.definition_id.map(counts).fillna(0).astype(int)
    definitions.to_csv(OUT/"reversal_signature_definitions.csv",index=False)
    members.to_csv(OUT/"reversal_signature_members.csv.gz",index=False,compression="gzip")

    n_genes=len(genes);n_defs=len(definitions)
    W=np.zeros((n_genes,n_defs),dtype=np.float32);M=np.zeros((n_genes,n_defs),dtype=np.float32)
    def_index={d:i for i,d in enumerate(definitions.definition_id)}
    for row in members.itertuples():
        if row.gene in gene_index:
            i=gene_index[row.gene];j=def_index[row.definition_id];W[i,j]=row.weight;M[i,j]=1
    norms=np.sqrt((W*W).sum(0));
    X=np.load(MATRIX,mmap_mode="r");
    if X.shape[1]!=n_genes:raise RuntimeError((X.shape,n_genes))
    scores=np.lib.format.open_memmap(WORK/"reversal_context_scores.npy",mode="w+",dtype="float32",shape=(X.shape[0],n_defs))
    chunk=2048
    for start in range(0,X.shape[0],chunk):
        end=min(start+chunk,X.shape[0]);block=np.asarray(X[start:end],dtype=np.float32)
        numerator=block@W;drug_norm_sq=(block*block)@M
        denominator=np.sqrt(drug_norm_sq)*norms[None,:]
        scores[start:end]=np.divide(-numerator,denominator,out=np.full_like(numerator,np.nan),where=denominator>0)
    scores.flush()
    provenance={"matrix_sha256":digest(MATRIX),"score_matrix_sha256":digest(WORK/"reversal_context_scores.npy"),
                "contexts":X.shape[0],"genes":X.shape[1],"definitions":n_defs,
                "formula":"negative weighted cosine on definition genes; positive is reversal",
                "definition_counts":definitions.analysis.value_counts().to_dict(),"upstream_steps_rerun":False}
    (OUT/"reversal_score_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    print(json.dumps(provenance,indent=2))


if __name__=="__main__":main()
