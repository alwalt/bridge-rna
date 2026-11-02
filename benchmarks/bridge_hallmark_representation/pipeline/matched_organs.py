#!/usr/bin/env python3
"""Exploratory GTEx–TCGA matched-organ Bridge and conventional comparisons."""
from __future__ import annotations
import h5py,numpy as np,pandas as pd
from scipy import stats
from common import CONFIG,RESULTS,resolve,read_vocab,read_gmt,bh_fdr
from de_gsea import gsea,norm

PAIRS={"lung_LUAD":("lung","LUAD"),"breast_BRCA":("mammary gland","BRCA")}

def summary_de(a,b,genes):
 ma,mb=a.mean(0),b.mean(0);va,vb=a.var(0,ddof=1),b.var(0,ddof=1);na,nb=len(a),len(b);se=np.sqrt(va/na+vb/nb);t=(ma-mb)/np.maximum(se,1e-12);df=(va/na+vb/nb)**2/np.maximum((va/na)**2/(na-1)+(vb/nb)**2/(nb-1),1e-20);p=2*stats.t.sf(abs(t),df);o=pd.DataFrame({"gene":genes,"mean_a":ma,"mean_b":mb,"log_expression_difference":ma-mb,"statistic":t,"p_value":p});o["fdr"]=bh_fdr(p);return o.sort_values("statistic",ascending=False)

def run():
 gm=pd.read_parquet(RESULTS/"gtex_sample_manifest.parquet");tm=pd.read_parquet(RESULTS/"tcga_sample_manifest.parquet");gp=pd.read_parquet(RESULTS/"gtex_frozen_probe_hallmark_profiles.parquet");tp=pd.read_parquet(RESULTS/"tcga_zero_shot_hallmark_profiles.parquet");names=[c for c in gp if c!="sample_id"]
 graw_path=resolve(CONFIG["gtex_expression"]); del graw_path # canonical matrix is not used for whole-genome baseline
 gtex_parquet=pd.read_parquet("/home/walt/Attention/data/gtex/GTEx_Analysis_2026-05-19_v11_RNASeQCv2.4.3_gene_reads.parquet",columns=["Description"]+gm.sample_id.tolist())
 gtex_parquet["gene"]=gtex_parquet.Description.map(norm);gtex_parquet=gtex_parquet.drop_duplicates("gene").set_index("gene").drop(columns="Description")
 # natural log1p(CPM), matching the full-expression normalization used in TCGA downstream controls
 gc=gtex_parquet.to_numpy(dtype=np.float32).T;gc=np.log1p(gc/np.maximum(gc.sum(1,keepdims=True),1)*1e6);ggenes=gtex_parquet.index.tolist();gindex={g:i for i,g in enumerate(ggenes)}
 TX=np.load(resolve(CONFIG["tcga_full_expression"]),mmap_mode="r");
 with h5py.File(resolve(CONFIG["tcga_full_h5"]),"r") as h:tgenes=[norm(x.decode() if isinstance(x,bytes) else x) for x in h["meta/genes"][:]]
 tindex={g:i for i,g in enumerate(tgenes)};common=sorted(set(gindex)&set(tindex));gi=np.array([gindex[g] for g in common]);ti=np.array([tindex[g] for g in common]);bridge=set(read_vocab());sets=read_gmt();rows=[];gsea_rows=[]
 for label,(tissue,cancer) in PAIRS.items():
  ga=gm.tissue.eq(tissue).to_numpy();tb=tm.cohort.eq(cancer).to_numpy();de=summary_de(np.asarray(TX[tb])[:,ti],gc[ga][:,gi],common);de.to_parquet(RESULTS/f"exploratory_de_{label}_whole.parquet",index=False);small=de[de.gene.isin(bridge)];small.to_parquet(RESULTS/f"exploratory_de_{label}_bridge_vocab.parquet",index=False)
  wg=gsea(de,sets,"whole",label);vg=gsea(small,sets,"bridge_15165",label);gsea_rows.extend([wg,vg]);delta=tp.merge(tm[["sample_id","cohort"]],on="sample_id").loc[lambda d:d.cohort.eq(cancer),names].mean()-gp.merge(gm[["sample_id","tissue"]],on="sample_id").loc[lambda d:d.tissue.eq(tissue),names].mean();w=wg.set_index("hallmark");v=vg.set_index("hallmark")
  for n in names:rows.append({"comparison":label,"hallmark":n,"bridge_delta_tcga_minus_gtex":delta[n],"whole_gsea_nes_tcga_minus_gtex":w.loc[n,"nes"],"whole_gsea_fdr":w.loc[n,"gsea_fdr"],"bridge_vocab_gsea_nes_tcga_minus_gtex":v.loc[n,"nes"],"bridge_vocab_gsea_fdr":v.loc[n,"gsea_fdr"],"n_tcga":int(tb.sum()),"n_gtex":int(ga.sum())})
 pd.concat(gsea_rows).to_csv(RESULTS/"exploratory_matched_organ_gsea.csv",index=False);out=pd.DataFrame(rows);out.to_csv(RESULTS/"exploratory_matched_organ_pathways.csv",index=False);plot(out)

def plot(out):
 import matplotlib.pyplot as plt
 p=out.pivot(index="hallmark",columns="comparison",values="bridge_delta_tcga_minus_gtex");fig,ax=plt.subplots(figsize=(7,13));im=ax.imshow(p,cmap="coolwarm",aspect="auto");ax.set_xticks(range(len(p.columns)),p.columns,rotation=45,ha="right");ax.set_yticks(range(len(p)),p.index,fontsize=6);fig.colorbar(im,ax=ax,label="TCGA − GTEx frozen-probe score");fig.tight_layout();fig.savefig(RESULTS/"exploratory_gtex_tcga_matched_organ_heatmap.png",dpi=180);plt.close(fig)
if __name__=="__main__":run()
