#!/usr/bin/env python3
"""Conventional DE→preranked GSEA under whole and Bridge-vocabulary universes."""
from __future__ import annotations
import json
import h5py,numpy as np,pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import gseapy as gp
from common import CONFIG,RESULTS,WORK,resolve,read_vocab,read_gmt,bh_fdr,save_json

def norm(x):return str(x).upper().split(".")[0]

def de_table(X,genes,a,b):
    ma,mb=np.asarray(X[a]).mean(0),np.asarray(X[b]).mean(0);va,vb=np.asarray(X[a]).var(0,ddof=1),np.asarray(X[b]).var(0,ddof=1);na,nb=a.sum(),b.sum()
    se=np.sqrt(va/na+vb/nb);t=np.divide(ma-mb,se,out=np.zeros_like(ma),where=se>0);df=(va/na+vb/nb)**2/np.maximum((va/na)**2/(na-1)+(vb/nb)**2/(nb-1),1e-20);p=2*stats.t.sf(np.abs(t),df)
    out=pd.DataFrame({"gene":genes,"mean_a":ma,"mean_b":mb,"log_expression_difference":ma-mb,"statistic":t,"p_value":p});out["fdr"]=bh_fdr(p);return out.sort_values("statistic",ascending=False)

def gsea(de,sets,universe,contrast):
    ranking=de.drop_duplicates("gene").set_index("gene")["statistic"].sort_values(ascending=False)
    eligible={k:sorted(set(v)&set(ranking.index)) for k,v in sets.items()};eligible={k:v for k,v in eligible.items() if len(v)>=5}
    pre=gp.prerank(rnk=ranking,gene_sets=eligible,min_size=5,max_size=5000,permutation_num=CONFIG["gsea_permutations"],weight=1.0,ascending=False,seed=CONFIG["seed"],threads=4,verbose=False,outdir=None)
    r=pre.res2d.copy().rename(columns={"Term":"hallmark","NES":"nes","NOM p-val":"nominal_p","FDR q-val":"gsea_fdr","Lead_genes":"leading_edge","Rank at Max":"rank_at_max"})
    cols=[c for c in ["hallmark","nes","nominal_p","gsea_fdr","leading_edge","rank_at_max"] if c in r];r=r[cols];r["contrast"]=contrast;r["universe"]=universe;r["pathway_direction"]=np.where(r.nes>=0,"group_a","group_b");r["pathway_rank"]=r.nes.abs().rank(ascending=False,method="min").astype(int)
    return r

def run():
    meta=pd.read_parquet(RESULTS/"tcga_sample_manifest.parquet");X=np.load(resolve(CONFIG["tcga_full_expression"]),mmap_mode="r")
    with h5py.File(resolve(CONFIG["tcga_full_h5"]),"r") as h: genes=[norm(x.decode() if isinstance(x,bytes) else x) for x in h["meta/genes"][:]]
    if X.shape!=(len(meta),len(genes)):raise AssertionError((X.shape,len(meta),len(genes)))
    bridge=set(read_vocab());sets=read_gmt();all_gsea=[];contrast_records=[]
    for contrast,(ga,gb) in CONFIG["tcga_contrasts"].items():
        a=meta.cohort.eq(ga).to_numpy();b=meta.cohort.eq(gb).to_numpy();de=de_table(X,genes,a,b);de.to_parquet(RESULTS/f"de_{contrast}_whole.parquet",index=False)
        reduced=de[de.gene.isin(bridge)].copy();reduced.to_parquet(RESULTS/f"de_{contrast}_bridge_vocab.parquet",index=False)
        whole=gsea(de,sets,"whole",contrast);small=gsea(reduced,sets,"bridge_15165",contrast);all_gsea.extend([whole,small]);contrast_records.append({"contrast":contrast,"group_a":ga,"group_b":gb,"n_a":int(a.sum()),"n_b":int(b.sum()),"de_method":"Welch t statistic on log1p CPM","whole_genes":len(de),"bridge_intersection_genes":len(reduced)})
    g=pd.concat(all_gsea,ignore_index=True);g.to_csv(RESULTS/"conventional_gsea_results.csv",index=False);pd.DataFrame(contrast_records).to_csv(RESULTS/"contrast_manifest.csv",index=False)
    profiles=pd.read_parquet(RESULTS/"tcga_zero_shot_hallmark_profiles.parquet");pm=meta.merge(profiles,on="sample_id",validate="one_to_one");names=[c for c in profiles if c!="sample_id"];comparisons=[]
    for rec in contrast_records:
        c,ga,gb=rec["contrast"],rec["group_a"],rec["group_b"];delta=pm.loc[pm.cohort.eq(ga),names].mean()-pm.loc[pm.cohort.eq(gb),names].mean();w=g[(g.contrast==c)&(g.universe=="whole")].set_index("hallmark");v=g[(g.contrast==c)&(g.universe=="bridge_15165")].set_index("hallmark")
        for name in names:
            comparisons.append({"contrast":c,"hallmark":name,"bridge_delta":delta[name],"bridge_direction":"group_a" if delta[name]>=0 else "group_b","whole_gsea_nes":w.loc[name,"nes"],"whole_gsea_fdr":w.loc[name,"gsea_fdr"],"bridge_vocab_gsea_nes":v.loc[name,"nes"],"bridge_vocab_gsea_fdr":v.loc[name,"gsea_fdr"],"bridge_vs_whole_direction":np.sign(delta[name])==np.sign(w.loc[name,"nes"]),"bridge_vs_vocab_direction":np.sign(delta[name])==np.sign(v.loc[name,"nes"]),"whole_vs_vocab_direction":np.sign(w.loc[name,"nes"] )==np.sign(v.loc[name,"nes"]),"whole_leading_edge":w.loc[name,"leading_edge"],"vocab_leading_edge":v.loc[name,"leading_edge"]})
    comp=pd.DataFrame(comparisons);comp.to_csv(RESULTS/"bridge_vs_gsea_pathway_comparison.csv",index=False)
    agreements=[]
    for c,d in comp.groupby("contrast"):
        wsig=set(d.loc[d.whole_gsea_fdr<.05,"hallmark"]);vsig=set(d.loc[d.bridge_vocab_gsea_fdr<.05,"hallmark"])
        lead=[]
        for x,y in zip(d.whole_leading_edge.fillna(""),d.vocab_leading_edge.fillna("")):
            a=set(str(x).split(";"))-{""};b=set(str(y).split(";"))-{""};lead.append(len(a&b)/max(len(a|b),1))
        agreements.append({"contrast":c,"nes_pearson":stats.pearsonr(d.whole_gsea_nes,d.bridge_vocab_gsea_nes).statistic,"rank_spearman":stats.spearmanr(d.whole_gsea_nes.abs(),d.bridge_vocab_gsea_nes.abs()).statistic,"direction_agreement":d.whole_vs_vocab_direction.mean(),"significant_both":len(wsig&vsig),"significant_whole_only":len(wsig-vsig),"significant_vocab_only":len(vsig-wsig),"significant_jaccard":len(wsig&vsig)/max(len(wsig|vsig),1),"mean_leading_edge_jaccard":np.mean(lead),"bridge_whole_rank_spearman":stats.spearmanr(d.bridge_delta,d.whole_gsea_nes).statistic,"bridge_vocab_rank_spearman":stats.spearmanr(d.bridge_delta,d.bridge_vocab_gsea_nes).statistic})
    pd.DataFrame(agreements).to_csv(RESULTS/"gsea_universe_and_bridge_agreement.csv",index=False)
    plots(comp)

def plots(comp):
    for c,d in comp.groupby("contrast"):
      for x,y,title,name in [("whole_gsea_nes","bridge_vocab_gsea_nes","Whole vs Bridge-vocabulary GSEA",f"{c}_gsea_whole_vs_vocab.png"),("whole_gsea_nes","bridge_delta","Bridge delta vs whole GSEA",f"{c}_bridge_vs_whole_gsea.png"),("bridge_vocab_gsea_nes","bridge_delta","Bridge delta vs vocabulary-matched GSEA",f"{c}_bridge_vs_vocab_gsea.png")]:
       fig,ax=plt.subplots(figsize=(7,6));ax.scatter(d[x],d[y]);ax.axhline(0,color="grey",lw=.5);ax.axvline(0,color="grey",lw=.5);discord=d[np.sign(d[x])!=np.sign(d[y])].assign(mag=lambda z:abs(z[x])+abs(z[y])).nlargest(6,"mag")
       for row in discord.itertuples():ax.annotate(row.hallmark,(getattr(row,x),getattr(row,y)),fontsize=6)
       ax.set(xlabel=x,ylabel=y,title=f"{c}: {title}");fig.tight_layout();fig.savefig(RESULTS/name,dpi=180);plt.close(fig)

if __name__=="__main__":run()
