#!/usr/bin/env python3
"""Pathway-output Integrated Gradients and comparison with decoder/DE/GSEA."""
from __future__ import annotations
import argparse,sys,time
import numpy as np,pandas as pd,torch
from scipy import stats
from scipy.special import expit
from common import CONFIG,REPO,RESULTS,WORK,resolve,read_vocab,read_gmt,save_json
sys.path.insert(0,str(REPO/"src"))
from fm_embed.model import load_expression_performer

def integrated_gradients(model,x,w,b,mean,scale,device,steps=24,batch=4):
    base=torch.zeros_like(torch.as_tensor(x,dtype=torch.float32,device=device));total=torch.zeros_like(base)
    for start in range(0,steps,batch):
        alphas=torch.linspace((start+1)/steps,min(start+batch,steps)/steps,min(batch,steps-start),device=device)
        xb=(base[None]+alphas[:,None]*(torch.as_tensor(x,device=device)-base)[None]).requires_grad_(True)
        hidden=model._encode_hidden(xb).mean(1);score=((hidden-torch.as_tensor(mean,device=device))/torch.as_tensor(scale,device=device))@torch.as_tensor(w,device=device)+b
        grad=torch.autograd.grad(score.sum(),xb)[0];total+=grad.sum(0)
    return ((torch.as_tensor(x,device=device)-base)*total/steps).detach().cpu().numpy()

def top_overlap(a,b,k):return len(set(np.argsort(-a)[:k])&set(np.argsort(-b)[:k]))

def main():
 p=argparse.ArgumentParser();p.add_argument("--device",default="cuda:0");p.add_argument("--steps",type=int,default=24);p.add_argument("--pathways",type=int,default=4);a=p.parse_args()
 meta=pd.read_parquet(RESULTS/"tcga_sample_manifest.parquet");X=np.load(resolve(CONFIG["tcga_bridge_expression"]),mmap_mode="r");probe=np.load(WORK/"frozen_gtex_hallmark_probe.npz");decoder=np.load(WORK/"hallmark_linear_decoder.npz")
 genes=read_vocab();sets=read_gmt();names=probe["hallmarks"].astype(str).tolist();transfer=pd.read_csv(RESULTS/"gtex_tcga_probe_transfer_summary.csv");chosen=transfer.nlargest(a.pathways,"pearson").hallmark.tolist()
 contrasts=CONFIG["tcga_contrasts"]; sample_rows=[]
 for contrast,(ga,gb) in contrasts.items():
  for group in [ga,gb]:
   ids=meta.index[meta.cohort.eq(group)].to_numpy();sample_rows.extend((contrast,int(i)) for i in ids[:1])
 model,device=load_expression_performer(resolve(CONFIG["checkpoint"]),resolve(CONFIG["model_config"]),15165,a.device);model.requires_grad_(False)
 gsea=pd.read_csv(RESULTS/"conventional_gsea_results.csv");rows=[];metrics=[];t0=time.time()
 for si,(contrast,row) in enumerate(sample_rows):
  x=np.array(X[row],copy=True);sample=meta.iloc[row];
  with torch.inference_mode(): hidden=model._encode_hidden(torch.as_tensor(x[None],device=device))[0].float().cpu().numpy()
  dprob=expit(((hidden-decoder["mean"])/decoder["scale"])@decoder["weight"].T+decoder["bias"])
  de=pd.read_parquet(RESULTS/f"de_{contrast}_bridge_vocab.parquet").set_index("gene");
  for name in chosen:
   k=names.index(name);ig=integrated_gradients(model,x,probe["weight"][k],float(probe["bias"][k]),probe["mean"],probe["scale"],device,a.steps)
   member=np.isin(genes,list(sets[name])); leadrow=gsea[(gsea.contrast==contrast)&(gsea.universe=="bridge_15165")&gsea.hallmark.eq(name)];lead=set(str(leadrow.iloc[0].leading_edge).split(";")) if len(leadrow) else set()
   decoder_score=dprob[:,k];absig=np.abs(ig);rho=stats.spearmanr(decoder_score,absig).statistic
   high_ig=set(np.argsort(-absig)[:50]);odds,pv=stats.fisher_exact([[member[list(high_ig)].sum(),50-member[list(high_ig)].sum()],[member.sum()-member[list(high_ig)].sum(),len(member)-member.sum()-(50-member[list(high_ig)].sum())]],alternative="greater")
   metrics.append({"sample_id":sample.sample_id,"contrast":contrast,"cohort":sample.cohort,"hallmark":name,"decoder_abs_ig_spearman":rho,"top10_overlap":top_overlap(decoder_score,absig,10),"top25_overlap":top_overlap(decoder_score,absig,25),"top50_overlap":top_overlap(decoder_score,absig,50),"canonical_high_ig_odds_ratio":odds,"canonical_high_ig_p":pv,"leading_edge_top50_overlap":len({genes[i] for i in high_ig}&lead)})
   for gi,gene in enumerate(genes):
    d=de.loc[gene] if gene in de.index else None
    rows.append({"sample_id":sample.sample_id,"patient_id":sample.patient_id,"cohort":sample.cohort,"contrast":contrast,"hallmark":name,"gene":gene,"de_log_expression_difference":d.log_expression_difference if d is not None else np.nan,"de_statistic":d.statistic if d is not None else np.nan,"de_fdr":d.fdr if d is not None else np.nan,"gsea_leading_edge":gene in lead,"decoder_score":decoder_score[gi],"ig_signed":ig[gi],"ig_absolute":absig[gi],"canonical_member":member[gi]})
  print(f"[IG] {si+1}/{len(sample_rows)} elapsed={time.time()-t0:.1f}s",flush=True)
 pd.DataFrame(metrics).to_csv(RESULTS/"decoder_ig_comparison_metrics.csv",index=False);pd.DataFrame(rows).to_parquet(RESULTS/"ig_de_gene_level_comparison.parquet",index=False)
 save_json(RESULTS/"integrated_gradients_provenance.json",{"target":"frozen GTEx-trained sample-level Hallmark probe output","baseline":"all-zero log1p(TPM)","steps":a.steps,"selected_pathways":chosen,"selected_samples":"first deterministic patient-sorted sample per contrast group","interpretation":"sensitivity/contribution, not membership"})
 plot(pd.DataFrame(metrics),pd.DataFrame(rows))

def plot(metrics,rows):
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(7,6));ax.scatter(metrics.decoder_abs_ig_spearman,metrics.canonical_high_ig_odds_ratio);ax.set(xlabel="Spearman(decoder, |IG|)",ylabel="Canonical-member enrichment odds ratio");fig.tight_layout();fig.savefig(RESULTS/"decoder_vs_ig_gene_ranking.png",dpi=180);plt.close(fig)
 subset=rows.groupby(["gene"],as_index=False).agg(ig_absolute=("ig_absolute","mean"),de_statistic=("de_statistic",lambda x:np.nanmean(abs(x))));fig,ax=plt.subplots(figsize=(7,6));ax.scatter(subset.de_statistic,subset.ig_absolute,s=4,alpha=.4);ax.set(xlabel="Mean absolute DE statistic",ylabel="Mean absolute IG");fig.tight_layout();fig.savefig(RESULTS/"ig_vs_de.png",dpi=180);plt.close(fig)
 first=rows[(rows.hallmark==rows.hallmark.iloc[0])].nlargest(50,"ig_absolute").sort_values("ig_absolute");fig,ax=plt.subplots(figsize=(10,12));colors=np.where(first.canonical_member,"tab:red","tab:blue");ax.barh(first.gene,first.ig_signed,color=colors);ax.set(xlabel="Signed IG",title=f"{first.hallmark.iloc[0]}: top |IG| genes (red=canonical)");fig.tight_layout();fig.savefig(RESULTS/"selected_pathway_gene_level_comparison.png",dpi=180);plt.close(fig)
if __name__=="__main__":main()
