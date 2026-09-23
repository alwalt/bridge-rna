#!/usr/bin/env python3
"""Build matched modules, run ChEMBL enrichment, and evaluate convergence."""

from __future__ import annotations
import itertools,json,time
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import hypergeom,spearmanr
from statsmodels.stats.multitest import multipletests

HERE=Path(__file__).resolve().parents[1]; CONFIG=json.loads((HERE/"config.json").read_text()); PREP=HERE/"work/prepared"; RESULTS=HERE/"results"; OUT=RESULTS/"evaluation"
ORDER={"Radiation":["GSE297090","GSE184119","GSE297560"],"Bone loss":["GSE189524","GSE276529","GSE273868"],"Muscle atrophy":["GSE211204","GSE113165","GSE234465"]}
STATUS={"GSE297090":"unseen","GSE184119":"seen","GSE297560":"unseen","GSE189524":"unseen","GSE276529":"unseen","GSE273868":"unseen","GSE211204":"partial","GSE113165":"seen","GSE234465":"unused-held-out"}
def say(x): print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {x}",flush=True)
def rbo(a,b,p=.9):
    sa=set(); sb=set(); total=0.; depth=max(len(a),len(b))
    for d in range(1,depth+1):
        if d<=len(a): sa.add(a[d-1])
        if d<=len(b): sb.add(b[d-1])
        total += (len(sa&sb)/d)*(p**(d-1))
    return (1-p)*total
def ranking_similarity(a,b,topn):
    aa=a[:topn]; bb=b[:topn]; A=set(aa); B=set(bb); inter=len(A&B)
    all_ids=sorted(set(a)|set(b)); ra={x:i+1 for i,x in enumerate(a)}; rb={x:i+1 for i,x in enumerate(b)}
    n=max(len(a),len(b))+1; xa=np.array([ra.get(x,n) for x in all_ids]); xb=np.array([rb.get(x,n) for x in all_ids]); rho=spearmanr(xa,xb).statistic if len(all_ids)>2 and xa.std()>0 and xb.std()>0 else np.nan
    return inter,inter/max(len(A|B),1),inter/max(min(len(A),len(B)),1),rbo(a,b),rho
def top_jaccard(a,b,topn):
    A=set(a[:topn]); B=set(b[:topn]); return len(A&B)/max(len(A|B),1)
def enrich(module,universe,edges,min_targets):
    universe=set(universe); module=set(module)&universe; e=edges[edges.target_gene.isin(universe)]
    targets=e.groupby(["drug_id","drug_name"]).target_gene.agg(lambda x:set(x)); targets=targets[targets.map(len)>=min_targets]
    rows=[]; M=len(universe); n=len(module)
    for (did,name),t in targets.items():
        K=len(t); k=len(module&t); p=float(hypergeom.sf(k-1,M,K,n)); odds=(k*(M-K-n+k))/max((n-k)*(K-k),1e-12)
        rows.append({"drug_id":did,"drug_name":name,"target_count":K,"overlap_count":k,"overlap_genes":";".join(sorted(module&t)),"odds_ratio":odds,"pvalue":p})
    d=pd.DataFrame(rows)
    if d.empty:return d
    d["fdr"]=multipletests(d.pvalue,method="fdr_bh")[1]; return d.sort_values(["fdr","pvalue","odds_ratio","drug_id"],ascending=[True,True,False,True]).reset_index(drop=True).assign(drug_rank=lambda x:np.arange(1,len(x)+1))
def convergence(rankings,condition,method,size,threshold):
    rows=[]; studies=ORDER[condition]
    for a,b in itertools.combinations(studies,2):
        ra=rankings[(method,a,size,threshold)]; rb=rankings[(method,b,size,threshold)]
        for n in (10,25,50):
            overlap,jacc,coef,rv,rho=ranking_similarity(ra,rb,n); rows.append({"condition":condition,"method":method,"module_size":size,"min_targets":threshold,"dataset_1":a,"dataset_2":b,"status_1":STATUS[a],"status_2":STATUS[b],"top_n":n,"overlap":overlap,"jaccard":jacc,"overlap_coefficient":coef,"rbo":rv,"censored_spearman":rho})
    for n in (10,25,50):
        sets=[set(rankings[(method,s,size,threshold)][:n]) for s in studies]; rows.append({"condition":condition,"method":method,"module_size":size,"min_targets":threshold,"dataset_1":"three_way","dataset_2":"three_way","status_1":"mixed","status_2":"mixed","top_n":n,"overlap":len(set.intersection(*sets)),"jaccard":len(set.intersection(*sets))/max(len(set.union(*sets)),1),"overlap_coefficient":len(set.intersection(*sets))/max(min(map(len,sets)),1),"rbo":np.nan,"censored_spearman":np.nan})
    return rows

def main():
    OUT.mkdir(parents=True,exist_ok=True); genes=pd.read_csv(CONFIG["canonical_genes"]).gene_symbol.tolist(); edges=pd.read_csv(RESULTS/"chembl37_direct_human_mechanism_edges.csv.gz")
    bridge=pd.read_parquet(RESULTS/"bridge/study_gene_rankings.parquet")
    # Construct matched study-level DE consensus from exactly the same frozen contrasts.
    cm=pd.read_csv(RESULTS/"manifests/contrast_members.csv"); de_parts=[]
    for cid,row in cm.drop_duplicates("contrast_id").set_index("contrast_id").iterrows():
        d=pd.read_csv(RESULTS/"differential_expression"/f"{cid}.csv.gz"); scale=max(d.t.abs().sum(),1e-12); d["score"]=d.t/scale; d["dataset"]=row.dataset; de_parts.append(d[["dataset","gene","score"]])
    de=pd.concat(de_parts).groupby(["dataset","gene"],as_index=False).score.mean(); de["absolute_score"]=de.score.abs(); de["rank"]=de.groupby("dataset").absolute_score.rank(method="first",ascending=False).astype(int)
    de.to_parquet(RESULTS/"differential_expression/study_gene_rankings.parquet",index=False)
    scores={"Bridge":bridge.rename(columns={"signed_bridge_score":"score","absolute_bridge_score":"absolute_score"}),"DE":de}
    universes={}
    for gse in sorted(bridge.dataset.unique()):
        x=np.load(PREP/f"{gse}_log1p_tpm.npy",mmap_mode="r"); universes[gse]=[g for g,v in zip(genes,np.asarray(x).sum(0)) if v>0]
    module_rows=[]; enrichment_rows=[]; rankings={}; modules={}
    thresholds=sorted(set([CONFIG["primary_min_drug_targets"]]+CONFIG["drug_target_threshold_sensitivity"]))
    for method,table in scores.items():
        for gse,g in table.groupby("dataset"):
            universe=set(universes[gse]); ranked=g[g.gene.isin(universe)].sort_values(["absolute_score","gene"],ascending=[False,True])
            for size in CONFIG["module_sizes"]:
                module=ranked.head(size).gene.tolist(); modules[(method,gse,size)]=module
                for rank,gene in enumerate(module,1): module_rows.append({"method":method,"dataset":gse,"module_size":size,"rank":rank,"gene":gene,"signed_score":float(ranked.set_index("gene").loc[gene,"score"])})
                for threshold in thresholds:
                    e=enrich(module,universe,edges,threshold); e["method"]=method;e["dataset"]=gse;e["module_size"]=size;e["min_targets"]=threshold; enrichment_rows.append(e)
                    # Exclude zero-overlap p=1 ties from rank convergence; their
                    # deterministic ChEMBL-ID ordering contains no biological signal.
                    rankings[(method,gse,size,threshold)]=e.loc[e.overlap_count.gt(0),"drug_id"].tolist()
    modules_df=pd.DataFrame(module_rows); modules_df.to_parquet(OUT/"gene_modules.parquet",index=False)
    enr=pd.concat(enrichment_rows,ignore_index=True); enr.to_parquet(OUT/"drug_enrichment.parquet",index=False); enr.to_csv(OUT/"drug_enrichment.csv.gz",index=False,compression="gzip")
    conv=[]
    for condition in ORDER:
      for method in scores:
       for size in CONFIG["module_sizes"]:
        for threshold in thresholds: conv.extend(convergence(rankings,condition,method,size,threshold))
    conv=pd.DataFrame(conv); conv.to_csv(OUT/"convergence_metrics.csv",index=False)

    # Prespecified interpretation split: comparisons composed only of cohorts
    # absent/unused in ARCHS4 versus comparisons involving any exposed cohort.
    primary_pairs=conv[(conv.module_size.eq(CONFIG["primary_module_size"])) & (conv.min_targets.eq(CONFIG["primary_min_drug_targets"])) & (conv.top_n.eq(25)) & conv.dataset_1.ne("three_way")].copy()
    heldout={"unseen","unused-held-out"}
    primary_pairs["pretraining_pair"]=["strict_unseen_pair" if a in heldout and b in heldout else "exposure_involved" for a,b in zip(primary_pairs.status_1,primary_pairs.status_2)]
    pair_wide=primary_pairs.pivot_table(index=["condition","dataset_1","dataset_2","status_1","status_2","pretraining_pair"],columns="method",values="jaccard").reset_index()
    pair_wide["bridge_minus_de"]=pair_wide["Bridge"]-pair_wide["DE"]
    pair_wide.to_csv(OUT/"pretraining_pair_effects.csv",index=False)
    pretraining_summary=pair_wide.groupby("pretraining_pair",as_index=False).agg(n_pairs=("bridge_minus_de","size"),bridge_mean=("Bridge","mean"),de_mean=("DE","mean"),bridge_minus_de=("bridge_minus_de","mean"))
    pretraining_summary.to_csv(OUT/"pretraining_status_summary.csv",index=False)

    # Primary paired method-label randomization: swap Bridge/DE labels within studies.
    size=CONFIG["primary_module_size"]; threshold=CONFIG["primary_min_drug_targets"]; n=25
    def score(assign,condition):
        vals=[]
        for a,b in itertools.combinations(ORDER[condition],2):
            ra=rankings[(assign[a],a,size,threshold)]; rb=rankings[(assign[b],b,size,threshold)]; vals.append(top_jaccard(ra,rb,n))
        return float(np.mean(vals))
    perm_rows=[]; effect_rows=[]
    for condition,studies in ORDER.items():
        bscore=score({s:"Bridge" for s in studies},condition); dscore=score({s:"DE" for s in studies},condition); observed=bscore-dscore; null=[]
        for mask in itertools.product([0,1],repeat=3):
            amap={s:("DE" if bit else "Bridge") for s,bit in zip(studies,mask)}; bmap={s:("Bridge" if bit else "DE") for s,bit in zip(studies,mask)}; value=score(amap,condition)-score(bmap,condition); null.append(value); perm_rows.append({"condition":condition,"mask":"".join(map(str,mask)),"difference":value})
        p=(sum(abs(x)>=abs(observed)-1e-15 for x in null))/len(null); effect_rows.append({"condition":condition,"bridge_mean_pairwise_top25_jaccard":bscore,"de_mean_pairwise_top25_jaccard":dscore,"bridge_minus_de":observed,"exact_p":p,"permutations":len(null)})
    all_studies=[s for studies in ORDER.values() for s in studies]
    observed_overall=float(np.mean([r["bridge_minus_de"] for r in effect_rows])); overall_null=[]
    for mask in itertools.product([0,1],repeat=len(all_studies)):
        bit=dict(zip(all_studies,mask)); diffs=[]
        for condition,studies in ORDER.items():
            amap={s:("DE" if bit[s] else "Bridge") for s in studies}; bmap={s:("Bridge" if bit[s] else "DE") for s in studies}; diffs.append(score(amap,condition)-score(bmap,condition))
        overall_null.append(float(np.mean(diffs)))
    overall_p=sum(abs(x)>=abs(observed_overall)-1e-15 for x in overall_null)/len(overall_null)
    effect_rows.append({"condition":"Overall","bridge_mean_pairwise_top25_jaccard":float(np.mean([r["bridge_mean_pairwise_top25_jaccard"] for r in effect_rows])),"de_mean_pairwise_top25_jaccard":float(np.mean([r["de_mean_pairwise_top25_jaccard"] for r in effect_rows])),"bridge_minus_de":observed_overall,"exact_p":overall_p,"permutations":len(overall_null)})
    pd.DataFrame({"condition":"Overall","mask_index":np.arange(len(overall_null)),"difference":overall_null}).to_csv(OUT/"overall_method_label_permutation_null.csv",index=False)
    pd.DataFrame(perm_rows).to_csv(OUT/"method_label_permutation_null.csv",index=False); effects=pd.DataFrame(effect_rows); effects.to_csv(OUT/"primary_effects.csv",index=False)

    # Expression-decile-matched random modules, preserving each observed module's detectability profile.
    rng=np.random.default_rng(CONFIG["seed"]); reps=CONFIG["random_module_replicates"]; random_rows=[]
    expr_bins={}
    for gse in universes:
        x=np.asarray(np.load(PREP/f"{gse}_log1p_tpm.npy",mmap_mode="r")); mean=pd.Series(x.mean(0),index=genes); u=pd.Index(universes[gse]); bins=pd.qcut(mean.loc[u].rank(method="first"),10,labels=False); expr_bins[gse]=bins
    for condition,studies in ORDER.items():
      for method in scores:
        observed=float(effects.loc[effects.condition.eq(condition),"bridge_mean_pairwise_top25_jaccard" if method=="Bridge" else "de_mean_pairwise_top25_jaccard"].iloc[0]); values=[]
        for rep in range(reps):
            rr={}
            for gse in studies:
                bins=expr_bins[gse]; observed_module=modules[(method,gse,size)]; want=bins.loc[observed_module].value_counts(); sampled=[]
                for b,count in want.items(): sampled.extend(rng.choice(bins.index[bins.eq(b)],int(count),replace=False))
                ee=enrich(sampled,universes[gse],edges,threshold); rr[gse]=ee.loc[ee.overlap_count.gt(0),"drug_id"].tolist()
            vals=[top_jaccard(rr[a],rr[b],n) for a,b in itertools.combinations(studies,2)]; values.append(float(np.mean(vals)))
            random_rows.append({"condition":condition,"method":method,"replicate":rep,"mean_pairwise_top25_jaccard":values[-1]})
        p=(1+sum(v>=observed for v in values))/(1+len(values)); effect_rows.append({"condition":condition,"method":method,"observed":observed,"random_mean":float(np.mean(values)),"random_sd":float(np.std(values,ddof=1)),"empirical_p":p})
        say(f"random null {condition} {method}: observed={observed:.4g} p={p:.4g}")
    pd.DataFrame(random_rows).to_parquet(OUT/"random_module_null.parquet",index=False); pd.DataFrame(effect_rows[4:]).to_csv(OUT/"random_module_summary.csv",index=False)

    # Strict endpoint-unseen GSE211204 sensitivity, replacing only the muscle discovery ranking.
    bs=pd.read_parquet(RESULTS/"bridge/GSE211204_unseen_subject_gene_ranking.parquet").rename(columns={"signed_bridge_score":"score","absolute_bridge_score":"absolute_score"})
    ds=pd.read_csv(RESULTS/"differential_expression/GSE211204_ULLS_unseen.csv.gz"); ds["score"]=ds.t/max(ds.t.abs().sum(),1e-12); ds["absolute_score"]=ds.score.abs()
    sens_rows=[]; sens_rank={}
    for method,table in {"Bridge":bs,"DE":ds}.items():
      ranked=table[table.gene.isin(universes["GSE211204"])].sort_values(["absolute_score","gene"],ascending=[False,True])
      for msize in CONFIG["module_sizes"]:
       module=ranked.head(msize).gene.tolist()
       for tmin in thresholds:
        ee=enrich(module,universes["GSE211204"],edges,tmin); ee["method"]=method;ee["module_size"]=msize;ee["min_targets"]=tmin;sens_rows.append(ee)
        sens_rank[(method,msize,tmin)]=ee.loc[ee.overlap_count.gt(0),"drug_id"].tolist()
    pd.concat(sens_rows,ignore_index=True).to_parquet(OUT/"GSE211204_unseen_subject_drug_enrichment.parquet",index=False)
    sensitivity_summary=[]
    for method in scores:
      rr={"GSE211204":sens_rank[(method,size,threshold)],"GSE113165":rankings[(method,"GSE113165",size,threshold)],"GSE234465":rankings[(method,"GSE234465",size,threshold)]}
      vals=[top_jaccard(rr[a],rr[b],25) for a,b in itertools.combinations(ORDER["Muscle atrophy"],2)]
      sensitivity_summary.append({"method":method,"mean_pairwise_top25_jaccard":float(np.mean(vals)),"pairwise_jaccards":";".join(map(str,vals))})
    pd.DataFrame(sensitivity_summary).to_csv(OUT/"GSE211204_unseen_subject_convergence.csv",index=False)
    prov={"module_ranking":"absolute signed score; identical dataset-specific expressed Bridge universe","primary_module_size":size,"primary_min_targets":threshold,"drug_test":"right-sided hypergeometric/Fisher equivalent with BH correction","primary_convergence":"mean of three pairwise top-25 nonzero-overlap drug-list Jaccards","zero_overlap_ties":"excluded from ranking convergence because p=1 ChEMBL-ID ordering is non-biological","method_permutation":"exact within-study Bridge/DE label swaps","random_modules":"1000 expression-decile-matched modules per dataset and method","leakage_sensitivity":"replace GSE211204 discovery with ranking from seven subjects whose baseline and ULLS endpoints are absent from ARCHS4 train/validation","pretraining_analysis":"primary pairwise Bridge-minus-DE effects split into strict-unseen pairs versus comparisons involving any train/validation-exposed cohort","direction_aware_drug_analysis":False,"pretraining_status":STATUS}
    (OUT/"provenance.json").write_text(json.dumps(prov,indent=2)+"\n"); say("enrichment and convergence complete")
if __name__=="__main__": main()
