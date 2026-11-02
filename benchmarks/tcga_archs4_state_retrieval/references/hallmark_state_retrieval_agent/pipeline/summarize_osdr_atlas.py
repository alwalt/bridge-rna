#!/usr/bin/env python3
"""Summarize OSDR atlas retrieval, controls, and scientific interpretation tables."""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import RESULTS, save_json

OUT = RESULTS / "osdr_atlas"
CATEGORIES = ["cancer","infection_inflammation","oxidative_stress","mitochondrial","hypoxia",
              "fibrosis_remodeling","metabolic_disease","radiation","exercise_disuse","drug_perturbation"]


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else np.nan


def main() -> None:
    hits = pd.read_parquet(OUT / "individual_and_centroid_top100.parquet")
    groups = pd.read_csv(OUT / "eligible_group_manifest.csv")
    individual = hits[hits.query_type.eq("individual")]
    rows=[]
    for (gid,condition,method), block in individual.groupby(["group_id","condition","method"]):
        samples=[g.sort_values("rank") for _,g in block.groupby("osdr_sample_id")]
        for k in (10,50,100):
            agreements=[]
            for a,b in combinations(samples,2):
                ca={c for c in CATEGORIES if a.head(k)[c].mean()>=.2}; cb={c for c in CATEGORIES if b.head(k)[c].mean()>=.2}
                agreements.append(jaccard(ca,cb))
            rows.append({"group_id":gid,"condition":condition,"method":method,"k":k,
                         "metadata_category_jaccard":np.nanmean(agreements)})
    category_agreement=pd.DataFrame(rows); category_agreement.to_csv(OUT/"replicate_metadata_category_agreement.csv",index=False)
    consistency=pd.read_csv(OUT/"replicate_consistency.csv").merge(category_agreement,on=["group_id","condition","method","k"],validate="one_to_one")
    consistency.to_csv(OUT/"replicate_consistency_complete.csv",index=False)

    cent=hits[hits.query_type.eq("centroid") & hits["rank"].le(50)].copy()
    cat=cent.groupby(["group_id","condition","method"])[CATEGORIES+['same_tissue','signal_coverage_top10']].mean().reset_index()
    wide=cat.pivot(index=["group_id","method"],columns="condition",values=CATEGORIES+['same_tissue','signal_coverage_top10']).reset_index()
    wide.columns=['_'.join(x).strip('_') if isinstance(x,tuple) else x for x in wide.columns]
    for col in CATEGORIES+['same_tissue','signal_coverage_top10']:
        wide[col+'_delta_flt_minus_gc']=wide.get(col+'_FLT',0)-wide.get(col+'_GC',0)
    wide.to_csv(OUT/"centroid_flt_gc_category_shifts.csv",index=False)

    recurring=[]
    for method,catname in ((m,c) for m in wide.method.unique() for c in CATEGORIES):
        x=wide[wide.method.eq(method)][["group_id",catname+'_delta_flt_minus_gc']].merge(groups[["group_id","tissue","study"]],on="group_id")
        for tissue,g in x.groupby("tissue"):
            recurring.append({"method":method,"category":catname,"tissue":tissue,"groups":len(g),"studies":g.study.nunique(),
                              "mean_delta":g.iloc[:,1].mean(),"flt_shift_groups":int((g.iloc[:,1]>.05).sum()),"gc_shift_groups":int((g.iloc[:,1]<-.05).sum())})
    pd.DataFrame(recurring).to_csv(OUT/"cross_study_category_recurrence.csv",index=False)

    # Global-versus-functional overlap and high-Hallmark/lower-Bridge cases.
    comparison=[]
    cent100=hits[hits.query_type.eq("centroid")]
    for (gid,condition),b in cent100.groupby(["group_id","condition"]):
        by={m:set(g.reference_row) for m,g in b.groupby("method")}
        bridge=by["Bridge-512 cosine"]
        for method in ("Hallmark-50 cosine","Hallmark signed top10 overlap"):
            hs=by[method]; comparison.append({"group_id":gid,"condition":condition,"hallmark_method":method,
                "top100_overlap":len(bridge&hs),"top100_jaccard":jaccard(bridge,hs),"hallmark_specific":len(hs-bridge)})
    pd.DataFrame(comparison).to_csv(OUT/"global_vs_functional_overlap.csv",index=False)

    loho=pd.read_csv(OUT/"hallmark_leave_one_dominant_out.csv")
    audit=pd.read_csv(OUT/"centroid_top5_biological_audit.csv")
    audit["any_metadata_relationship"]=audit.same_tissue | audit[CATEGORIES].any(axis=1)
    audit["plausible_shared_functional_state"]=audit.signal_coverage_top10.ge(.6)
    audit["no_obvious_relationship"]=~audit.any_metadata_relationship & ~audit.plausible_shared_functional_state
    audit=audit.merge(loho[["group_id","condition","gsm","loho_rank","survives_top100"]],on=["group_id","condition","gsm"],how="left")
    audit.to_csv(OUT/"centroid_top5_biological_audit_scored.csv",index=False)

    # Particularly interesting FLT top-five hits absent from the corresponding GC top-100.
    interesting=[]
    for (gid,method),b in cent100.groupby(["group_id","method"]):
        gc=set(b[b.condition.eq("GC")].gsm); flt=b[(b.condition.eq("FLT"))&b["rank"].le(5)]
        for r in flt.itertuples():
            if r.gsm not in gc and (r.same_tissue or r.signal_coverage_top10>=.6):
                interesting.append(r._asdict())
    pd.DataFrame(interesting).to_csv(OUT/"flt_centroid_interesting_top5_absent_gc_top100.csv",index=False)

    # Figures.
    c10=consistency[consistency.k.eq(10)].groupby(["method","condition"],as_index=False).agg(
        neighbor_jaccard=("mean_neighbor_jaccard","mean"),gse_jaccard=("mean_gse_jaccard","mean"),category_jaccard=("metadata_category_jaccard","mean"))
    plot=c10.pivot(index="method",columns="condition",values=["neighbor_jaccard","gse_jaccard","category_jaccard"])
    ax=plot.plot.bar(figsize=(13,6));ax.set(ylabel="Mean replicate agreement",ylim=(0,1),xlabel="",title="OSDR replicate-consensus retrieval across 88 matched groups");ax.legend(frameon=False,ncol=3,fontsize=8);plt.xticks(rotation=18,ha="right");plt.tight_layout();plt.savefig(OUT/"figures/replicate_consistency.png",dpi=240);plt.savefig(OUT/"figures/replicate_consistency.pdf");plt.close()
    delta_cols=[c+'_delta_flt_minus_gc' for c in CATEGORIES]
    hm=wide.groupby("method")[delta_cols].mean(); fig,ax=plt.subplots(figsize=(13,4.5)); vmax=max(abs(hm.to_numpy().min()),abs(hm.to_numpy().max()),.01); im=ax.imshow(hm,vmin=-vmax,vmax=vmax,cmap="coolwarm",aspect="auto");ax.set_yticks(range(len(hm)),hm.index);ax.set_xticks(range(len(delta_cols)),[c.replace('_',' ') for c in CATEGORIES],rotation=30,ha="right");fig.colorbar(im,ax=ax,label="FLT − GC top-50 metadata fraction");ax.set_title("Cross-group retrieval-category shifts");fig.tight_layout();fig.savefig(OUT/"figures/flt_gc_category_shifts.png",dpi=240);fig.savefig(OUT/"figures/flt_gc_category_shifts.pdf");plt.close(fig)

    diff=pd.read_parquet(OUT/"flt_vs_gc_differential_retrieval.parquet"); robust=diff[(diff.bh_q<=.1)&(diff.rrf_delta_flt_minus_gc>0)&(diff.flt_replicate_prevalence>=.75)&(diff.gc_replicate_prevalence<=.25)]
    summary={"eligible_groups":88,"individual_samples":1471,"robust_flt_enriched_hits":len(robust),"groups_with_robust_flt_enrichment":robust.group_id.nunique(),
             "robust_by_method":robust.groupby("method").size().to_dict(),"loho_top100_survival":loho.groupby("condition").survives_top100.mean().to_dict(),
             "macro_replicate_consistency_top10":c10.to_dict("records")}
    save_json(OUT/"scientific_summary.json",summary)


if __name__ == "__main__": main()
