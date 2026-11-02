#!/usr/bin/env python3
"""Frozen-output-only audit of scaling/metric choices in 50-D Hallmark retrieval."""
from __future__ import annotations
import json, time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from common import CONFIG, RESULTS, WORK, l2_rows, save_json, sha256

SOURCE=Path(CONFIG["source_hallmark_benchmark"])/"results"
OUT=RESULTS/"similarity_scaling_audit"
METHODS=["Raw cosine","GTEx-target-centered cosine","GTEx-target-z-scored cosine","Pearson correlation"]
TCGA_TERMS={"BRCA":["breast","mammary","carcinoma"],"COAD":["colon","colorectal"],"GBM":["glioblastoma","brain","glioma"],"KIRC":["kidney","renal","clear cell"],"KIRP":["kidney","renal","papillary"],"LIHC":["liver","hepatic","hepatocellular"],"LUAD":["lung","adenocarcinoma"],"LUSC":["lung","squamous"],"PRAD":["prostate"],"THCA":["thyroid"],"UCEC":["uterus","uterine","endometrial"]}

def transform(x,method,mean,sd):
    x=np.asarray(x,np.float32)
    if method=="Raw cosine": y=x
    elif method=="GTEx-target-centered cosine": y=x-mean
    elif method=="GTEx-target-z-scored cosine": y=(x-mean)/sd
    elif method=="Pearson correlation": y=x-x.mean(1,keepdims=True)
    else: raise KeyError(method)
    return l2_rows(y)

def frozen_queries(names):
    # Reconstruct exact earlier query identities; centroids retain the exact frozen group memberships.
    tprof=pd.read_parquet(RESULTS/"tcga/hallmark_signatures.parquet"); tm=pd.read_parquet(CONFIG["tcga_manifest"]).sort_values("matrix_row").reset_index(drop=True)
    assert tprof.sample_id.astype(str).tolist()==tm.sample_id.astype(str).tolist()
    oldt=pd.read_csv(RESULTS/"tcga/retrieval_results.csv"); tq=oldt[["query_id","query_type","cohort","sample_id"]].drop_duplicates("query_id").sort_values("query_id").reset_index(drop=True)
    th=tprof[names].to_numpy(np.float32); tv=[]
    for r in tq.itertuples():
        if r.query_type=="cohort_centroid": tv.append(th[tm.cohort.eq(r.cohort)].mean(0))
        else: tv.append(th[tm.sample_id.astype(str).eq(str(r.sample_id))][0])
    oprof=pd.read_parquet(RESULTS/"osdr/hallmark_signatures.parquet"); om=pd.read_csv(CONFIG["osdr_manifest"])
    assert oprof.sample_id.astype(str).tolist()==om.sample_id.astype(str).tolist()
    oldo=pd.read_csv(RESULTS/"osdr/retrieval_results.csv"); oq=oldo[["query_id","query_type","mission","condition","sample_id","OSD"]].drop_duplicates("query_id").sort_values("query_id").reset_index(drop=True)
    oh=oprof[names].to_numpy(np.float32); ov=[]
    for r in oq.itertuples():
        if r.query_type=="group_centroid": ov.append(oh[(om.mission.eq(r.mission)&om.condition.eq(r.condition)).to_numpy()].mean(0))
        else: ov.append(oh[om.sample_id.astype(str).eq(str(r.sample_id))][0])
    meta=pd.concat([tq.assign(dataset="TCGA"),oq.assign(dataset="OSDR")],ignore_index=True,sort=False)
    return meta,np.vstack([tv,ov]).astype(np.float32)

def retrieve(q,ref,k=10,chunk=32768):
    nq=len(q); best_s=np.full((nq,k),-np.inf,np.float32); best_i=np.full((nq,k),-1,np.int64)
    for left in range(0,len(ref),chunk):
        right=min(left+chunk,len(ref)); s=ref[left:right]@q.T
        for j in range(nq):
            cand_s=np.r_[best_s[j],s[:,j]]; cand_i=np.r_[best_i[j],np.arange(left,right)]
            take=np.argpartition(cand_s,-k)[-k:]; take=take[np.argsort(cand_s[take])[::-1]]; best_s[j],best_i[j]=cand_s[take],cand_i[take]
    return best_i,best_s

def main():
    started=time.time(); OUT.mkdir(parents=True,exist_ok=True); figdir=OUT/"figures"; figdir.mkdir(exist_ok=True)
    probe=np.load(RESULTS/"frozen/frozen_gtex_hallmark_probe.npz",allow_pickle=False); names=probe["hallmarks"].astype(str).tolist()
    targets=pd.read_parquet(SOURCE/"gtex_expression_hallmark_scores.parquet"); pred=pd.read_parquet(SOURCE/"gtex_frozen_probe_hallmark_profiles.parquet")
    assert [c for c in targets if c!="sample_id"]==names and targets.sample_id.astype(str).tolist()==pred.sample_id.astype(str).tolist()
    Y=targets[names].to_numpy(np.float64); P=pred[names].to_numpy(np.float64); mean=Y.mean(0).astype(np.float32); sd=Y.std(0,ddof=0).astype(np.float32)
    assert np.all(sd>0) and np.allclose(P.mean(0),mean,atol=2e-6) and np.allclose(probe["bias"],mean,atol=2e-5)
    stats=pd.DataFrame({"hallmark":names,"gtex_target_mean":mean,"gtex_target_sd":sd,"gtex_probe_prediction_mean":P.mean(0),"gtex_probe_prediction_sd":P.std(0,ddof=0)})
    stats.to_csv(OUT/"gtex_target_statistics.csv",index=False)
    ref_raw=np.load(WORK/"archs4_hallmark_signatures.float32.npy",mmap_mode="r"); refmeta=pd.read_parquet(RESULTS/"archs4/sample_metadata.parquet"); qmeta,qraw=frozen_queries(names)
    all_hits=[]; contribution=[]
    for method in METHODS:
        print(f"retrieving {method}",flush=True); q=transform(qraw,method,mean,sd); ref=np.empty(ref_raw.shape,np.float32)
        for left in range(0,len(ref),32768): ref[left:left+32768]=transform(ref_raw[left:left+32768],method,mean,sd)
        ii,ss=retrieve(q,ref)
        for qi in range(len(qmeta)):
            for rank,(ri,sim) in enumerate(zip(ii[qi],ss[qi]),1):
                r=refmeta.iloc[int(ri)]; rec={**qmeta.iloc[qi].to_dict(),"method":method,"rank":rank,"archs4_sample_id":r.sample_id,"series_id":r.series_id,"similarity":float(sim),"title":r.title,"source_name":r.source_name_ch1,"characteristics":r.characteristics_ch1}; all_hits.append(rec)
                c=q[qi]*ref[int(ri)];
                for j,v in enumerate(c): contribution.append({"query_id":qmeta.iloc[qi].query_id,"dataset":qmeta.iloc[qi].dataset,"method":method,"rank":rank,"archs4_sample_id":r.sample_id,"hallmark":names[j],"cosine_contribution":float(v)})
    hits=pd.DataFrame(all_hits); contrib=pd.DataFrame(contribution); hits.to_parquet(OUT/"all_retrieval_results.parquet",index=False); hits.to_csv(OUT/"all_retrieval_results.csv",index=False)
    text=hits[["title","source_name","characteristics"]].fillna("").agg(" ".join,axis=1).str.lower(); hits["liver_match"]=text.str.contains(r"liver|hepatic|hepatocyte",regex=True); hits["adipose_match"]=text.str.contains(r"adipose|fat tissue|adipocyte",regex=True)
    hits["tcga_keyword_match"]=[any(t in s for t in TCGA_TERMS.get(str(c),[])) if d=="TCGA" else np.nan for c,s,d in zip(hits.cohort,text,hits.dataset)]
    tcga=hits[hits.dataset.eq("TCGA")].groupby("method",as_index=False).agg(metadata_precision_at_10=("tcga_keyword_match","mean"),mean_similarity=("similarity","mean"),unique_studies=("series_id","nunique"))
    osdr=hits[hits.dataset.eq("OSDR")].groupby("method",as_index=False).agg(liver_precision_at_10=("liver_match","mean"),adipose_fraction_at_10=("adipose_match","mean"),mean_similarity=("similarity","mean"),unique_studies=("series_id","nunique"))
    # Append the unchanged, already-evaluated 512-D baseline.
    oldt=pd.read_csv(RESULTS/"tcga/hallmark50_vs_bridge512.csv"); oldo=pd.read_csv(RESULTS/"osdr/retrieval_results.csv"); otxt=oldo[["title","source_name","characteristics"]].fillna("").agg(" ".join,axis=1).str.lower()
    tcga=pd.concat([tcga,pd.DataFrame([{"method":"Bridge-512 cosine","metadata_precision_at_10":oldt[oldt.method.eq("Bridge-512 cosine")].top10_keyword_precision.mean(),"mean_similarity":oldt[oldt.method.eq("Bridge-512 cosine")].mean_top10_similarity.mean(),"unique_studies":pd.read_csv(RESULTS/"tcga/retrieval_results.csv").query("method == 'Bridge-512 cosine'").series_id.nunique()}])],ignore_index=True)
    ob=oldo.method.eq("Bridge-512 cosine"); osdr=pd.concat([osdr,pd.DataFrame([{"method":"Bridge-512 cosine","liver_precision_at_10":otxt[ob].str.contains(r"liver|hepatic|hepatocyte",regex=True).mean(),"adipose_fraction_at_10":otxt[ob].str.contains(r"adipose|fat tissue|adipocyte",regex=True).mean(),"mean_similarity":oldo.loc[ob,"similarity"].mean(),"unique_studies":oldo.loc[ob,"series_id"].nunique()}])],ignore_index=True)
    comp=tcga.merge(osdr,on="method",how="outer",suffixes=("_tcga","_osdr")); comp.to_csv(OUT/"similarity_method_comparison.csv",index=False)
    # Raw-space dominance: second moment controls expected same-coordinate dot product; variance and mean are separated.
    count=0; sumx=np.zeros(50); sumsq=np.zeros(50)
    for left in range(0,len(ref_raw),32768): x=np.asarray(ref_raw[left:left+32768],np.float64); count+=len(x); sumx+=x.sum(0); sumsq+=(x*x).sum(0)
    rmean=sumx/count; rvar=sumsq/count-rmean*rmean; second=sumsq/count
    dom=pd.DataFrame({"hallmark":names,"archs4_mean":rmean,"archs4_variance":rvar,"archs4_second_moment":second,"raw_squared_norm_share":second/second.sum()})
    rawc=contrib[(contrib.method.eq("Raw cosine"))&(contrib["rank"].le(10))].groupby("hallmark",as_index=False).agg(mean_signed_top_hit_contribution=("cosine_contribution","mean"),mean_abs_top_hit_contribution=("cosine_contribution",lambda x:np.abs(x).mean()))
    dom=dom.merge(rawc,on="hallmark").sort_values("raw_squared_norm_share",ascending=False); dom.to_csv(OUT/"raw_cosine_hallmark_dominance.csv",index=False)
    # Drivers of OSDR adipose-vs-liver hits under each Hallmark metric.
    c=contrib.merge(hits[["query_id","method","rank","archs4_sample_id","liver_match","adipose_match"]],on=["query_id","method","rank","archs4_sample_id"],how="left",validate="many_to_one"); c=c[c.dataset.eq("OSDR")]
    drivers=c.groupby(["method","hallmark"],as_index=False).agg(adipose_mean_contribution=("cosine_contribution",lambda x:x[c.loc[x.index,"adipose_match"]].mean()),liver_mean_contribution=("cosine_contribution",lambda x:x[c.loc[x.index,"liver_match"]].mean()),adipose_hits=("adipose_match","sum"),liver_hits=("liver_match","sum")); drivers["adipose_minus_liver_contribution"]=drivers.adipose_mean_contribution-drivers.liver_mean_contribution; drivers.to_csv(OUT/"osdr_liver_vs_adipose_hallmark_drivers.csv",index=False)
    order=METHODS+["Bridge-512 cosine"]; fig,ax=plt.subplots(1,2,figsize=(13,5)); a=comp.set_index("method").reindex(order); ax[0].bar(np.arange(len(a)),a.metadata_precision_at_10,color="#7a5195"); ax[0].set_xticks(range(len(a)),a.index,rotation=30,ha="right"); ax[0].set(ylim=(0,1),ylabel="Precision@10",title="TCGA metadata coherence"); ax[1].bar(np.arange(len(a)),a.liver_precision_at_10,color="#ef5675",label="Liver"); ax[1].bar(np.arange(len(a)),a.adipose_fraction_at_10,bottom=a.liver_precision_at_10,color="#ffa600",label="Adipose"); ax[1].set_xticks(range(len(a)),a.index,rotation=30,ha="right"); ax[1].set(ylim=(0,1),ylabel="Fraction of top-10 hits",title="OSDR metadata composition"); ax[1].legend(frameon=False); fig.tight_layout(); fig.savefig(figdir/"similarity_scaling_comparison.png",dpi=240); fig.savefig(figdir/"similarity_scaling_comparison.pdf"); plt.close(fig)
    save_json(OUT/"provenance.json",{"frozen_outputs_only":True,"bridge_inference":False,"head_retrained":False,"query_sets":"identical saved representative samples and group memberships","gtex_target_statistics_source":str(SOURCE/"gtex_expression_hallmark_scores.parquet"),"target_statistics_definition":"population mean and SD (ddof=0) over all 4,260 GTEx head-training targets","probe_mean_scale_definition":"512-D Bridge input-feature StandardScaler; not Hallmark-target statistics","cosine_implementation":"each transformed vector row-wise L2 normalized; centered/z-scored L2 variants are therefore already the evaluated cosine methods","pearson_implementation":"per-sample centering across 50 Hallmarks followed by row-wise L2 normalization","runtime_seconds":time.time()-started,"input_hashes":{"probe":sha256(RESULTS/"frozen/frozen_gtex_hallmark_probe.npz"),"gtex_targets":sha256(SOURCE/"gtex_expression_hallmark_scores.parquet")}})
if __name__=="__main__": main()
