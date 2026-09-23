#!/usr/bin/env python3
"""Aggregate context-preserving LINCS reversal scores to drugs and studies."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

HERE=Path(__file__).resolve().parents[1];OUT=HERE/"results";WORK=HERE/"work/lincs/GSE92742"
RELEVANT={"Multiple sclerosis":{"U937","THP1","JURKAT","NOMO1","CD34"},
          "Systemic lupus erythematosus":{"U937","THP1","JURKAT","NOMO1","CD34"},
          "Crohn's disease":{"HT29"},"Radiation injury":{"FIBRNPC","HT29"},
          "Bone loss":{"CD34","ASC"},"Muscle atrophy":{"SKB"}}
PUBLISHED={"Multiple sclerosis":["muromonab","ibrutinib","daclizumab","zanubrutinib","alemtuzumab"],
           "Crohn's disease":["zinc","zinc acetate","dilmapimod","glucosamine","vx-702"],
           "Systemic lupus erythematosus":["gemcitabine","enzastaurin","sunitinib","fostamatinib","cladribine"]}


def norm(x):return re.sub(r"[^a-z0-9]+","",str(x).lower())


def add_competitive(frame,group_cols,score="median_reversal"):
    parts=[]
    for _,g in frame.groupby(group_cols,dropna=False):
        g=g.copy();n=len(g);g["competitive_p"]=(g[score].rank(method="max",ascending=False)+0)/(n+1)
        g["competitive_q"]=multipletests(g.competitive_p,method="fdr_bh")[1]
        parts.append(g)
    return pd.concat(parts,ignore_index=True) if parts else frame


def summarize_drugs(meta,scores,definition):
    frame=meta[["pert_id","pert_iname","cell_id","pert_idose","pert_itime","distil_nsample","distil_cc_q75","tas"]].copy()
    frame["reversal_score"]=scores
    frame["positive_reversal"]=frame.reversal_score.gt(0).astype(float)
    out=(frame.groupby(["pert_id","pert_iname"],as_index=False,sort=False)
         .agg(median_reversal=("reversal_score","median"),mean_reversal=("reversal_score","mean"),
              sign_consistency=("positive_reversal","mean"),context_count=("reversal_score","size"),
              cell_count=("cell_id","nunique"),dose_count=("pert_idose","nunique"),
              timepoint_count=("pert_itime","nunique"),best_reversal=("reversal_score","max"),
              worst_reversal=("reversal_score","min"))
         .rename(columns={"pert_iname":"drug_name"}))
    out["eligible_consensus"]=out.context_count.ge(3)&out.cell_count.ge(2)
    out=add_competitive(out[out.eligible_consensus].copy(),[],"median_reversal") if False else out
    eligible=out[out.eligible_consensus].copy();n=len(eligible)
    if n:
        eligible["competitive_p"]=eligible.median_reversal.rank(method="max",ascending=False)/(n+1)
        eligible["competitive_q"]=multipletests(eligible.competitive_p,method="fdr_bh")[1]
    else:
        eligible["competitive_p"]=np.nan;eligible["competitive_q"]=np.nan
    out=out.merge(eligible[["pert_id","competitive_p","competitive_q"]],on="pert_id",how="left")
    out["candidate"]=out.eligible_consensus&out.competitive_p.le(.01)&out.median_reversal.gt(0)&out.sign_consistency.ge(.60)
    for col in ["definition_id","condition","contrast_id","dataset","method","analysis"]:out[col]=definition[col]
    return out


def main():
    meta=pd.read_parquet(OUT/"lincs_exemplar_contexts.parquet")
    definitions=pd.read_csv(OUT/"reversal_signature_definitions.csv")
    score=np.load(WORK/"reversal_context_scores.npy",mmap_mode="r")
    if score.shape!=(len(meta),len(definitions)):raise RuntimeError((score.shape,len(meta),len(definitions)))
    primary=[];partition=[];full=[];relevant=[]
    for j,d in definitions.iterrows():
        if d.analysis not in ["top500","full_ranking","Bridge-only","DE-only","shared","stable_recurrent","stable_recurrent_bridge_only"]:continue
        result=summarize_drugs(meta,np.asarray(score[:,j]),d)
        (primary if d.analysis=="top500" else full if d.analysis=="full_ranking" else partition).append(result)
        rel_cells=RELEVANT.get(d.condition,set());mask=meta.cell_id.isin(rel_cells)
        if mask.sum():
            rr=summarize_drugs(meta[mask].reset_index(drop=True),np.asarray(score[mask,j]),d)
            rr["relevant_cells"]=";".join(sorted(rel_cells));relevant.append(rr)
    primary=pd.concat(primary,ignore_index=True);parts=pd.concat(partition,ignore_index=True)
    primary.to_parquet(OUT/"drug_reversal_by_contrast.parquet",index=False)
    parts.to_parquet(OUT/"partition_and_stable_program_reversal.parquet",index=False)
    pd.concat(full,ignore_index=True).to_parquet(OUT/"full_ranking_reversal_sensitivity.parquet",index=False)
    pd.concat(relevant,ignore_index=True).to_parquet(OUT/"relevant_cell_reversal.parquet",index=False)

    # Dataset summaries preserve distinct study contexts before condition consensus.
    space=primary[primary.condition.isin(["Radiation injury","Bone loss","Muscle atrophy"])&primary.eligible_consensus]
    dataset=(space.groupby(["condition","method","dataset","pert_id","drug_name"],as_index=False)
             .agg(median_reversal=("median_reversal","median"),contrast_sign_consistency=("median_reversal",lambda x:(x>0).mean()),
                  context_count=("context_count","sum"),cell_count=("cell_count","max"),contrast_count=("contrast_id","nunique")))
    dataset=add_competitive(dataset,["condition","method","dataset"])
    dataset.to_csv(OUT/"space_drug_reversal_by_dataset.csv.gz",index=False,compression="gzip")
    condition=(dataset.groupby(["condition","method","pert_id","drug_name"],as_index=False)
               .agg(consensus_reversal=("median_reversal","median"),dataset_sign_consistency=("median_reversal",lambda x:(x>0).mean()),
                    dataset_count=("dataset","nunique"),max_dataset_p=("competitive_p","max"),
                    contexts=("context_count","sum"),cell_count=("cell_count","max")))
    condition=condition[condition.dataset_count.eq(3)].copy()
    condition=add_competitive(condition,["condition","method"],score="consensus_reversal")
    condition["cross_study_reproducible"]=condition.dataset_sign_consistency.eq(1)&condition.max_dataset_p.le(.10)
    condition["candidate"]=condition.consensus_reversal.gt(0)&condition.dataset_sign_consistency.ge(.60)&condition.competitive_p.le(.01)
    condition.to_csv(OUT/"space_condition_consensus.csv.gz",index=False,compression="gzip")

    # Published-drug audit uses the exact same context scores for Bridge and DE.
    audit=[]
    for disease,names in PUBLISHED.items():
        subset=primary[primary.condition.eq(disease)&primary.analysis.eq("top500")]
        available={norm(x):x for x in subset.drug_name.unique()}
        for requested in names:
            actual=available.get(norm(requested))
            for method in ["Bridge","DE"]:
                q=subset[(subset.method.eq(method))&subset.drug_name.eq(actual)] if actual else subset.iloc[0:0]
                if not len(q) or not bool(q.iloc[0].eligible_consensus):classification="insufficient perturbation data"
                else:
                    row=q.iloc[0]
                    if row.median_reversal>0 and row.sign_consistency>=.60:classification="reverses"
                    elif row.median_reversal<0 and row.sign_consistency<=.40:classification="reinforces"
                    else:classification="mixed/context-dependent"
                audit.append({"condition":disease,"requested_drug":requested,"represented":actual is not None,
                              "lincs_name":actual,"method":method,"classification":classification,
                              "median_reversal":q.iloc[0].median_reversal if len(q) else np.nan,
                              "sign_consistency":q.iloc[0].sign_consistency if len(q) else np.nan,
                              "competitive_p":q.iloc[0].competitive_p if len(q) else np.nan,
                              "competitive_q":q.iloc[0].competitive_q if len(q) else np.nan,
                              "contexts":q.iloc[0].context_count if len(q) else 0,"cells":q.iloc[0].cell_count if len(q) else 0})
    pd.DataFrame(audit).to_csv(OUT/"published_drug_direction_audit.csv",index=False)

    # Bridge-vs-DE candidate overlap, including partition candidates.
    rows=[]
    for condition_name in sorted(primary.condition.unique()):
        if condition_name in PUBLISHED:
            x=primary[primary.condition.eq(condition_name)]
            b=set(x[(x.method.eq("Bridge"))&x.candidate].pert_id);d=set(x[(x.method.eq("DE"))&x.candidate].pert_id)
            cross_b=set();
        else:
            x=condition[condition.condition.eq(condition_name)]
            b=set(x[(x.method.eq("Bridge"))&x.candidate].pert_id);d=set(x[(x.method.eq("DE"))&x.candidate].pert_id)
            cross_b=set(x[(x.method.eq("Bridge"))&x.cross_study_reproducible].pert_id)
        rows.append({"condition":condition_name,"bridge_candidates":len(b),"de_candidates":len(d),
                     "bridge_only_candidates":len(b-d),"shared_candidates":len(b&d),"de_only_candidates":len(d-b),
                     "cross_study_reproducible_bridge":len(cross_b)})
    pd.DataFrame(rows).to_csv(OUT/"six_condition_candidate_counts.csv",index=False)
    (OUT/"aggregation_provenance.json").write_text(json.dumps({
        "candidate_rule":"competitive p<=0.01, median>0, >=60% reversing, >=3 contexts, >=2 cell lines",
        "cross_study_rule":"positive in all three datasets and dataset competitive p<=0.10 in each",
        "competitive_null":"other eligible drug consensuses within the same frozen signature",
        "cell_line_dependence":"context rows retained; relevant-cell subset saved separately"},indent=2)+"\n")


if __name__=="__main__":main()
