#!/usr/bin/env python3
"""BridgeRNA full-response IG versus conventional expression change."""
from __future__ import annotations
import argparse,json,sys,time
from itertools import combinations
from pathlib import Path
import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import average_precision_score,roc_auc_score,r2_score
from sklearn.model_selection import KFold,cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parents[1];REPO=HERE.parents[1];OUT=HERE/'results/task4_attribution_vs_expression';FIG=OUT/'figures';WORK=HERE/'work/task4_attribution_vs_expression'
GD=HERE/'results/task4_gene_attribution_diagnostic';T3=REPO/'benchmarks/osdr_batch_effect_representation';R3=T3/'results';W3=T3/'work';GMTROOT=REPO/'benchmarks/cross_species_exercise_response/results/per_study_ranked_gsea';GMTS={'GO:BP':'GO_Biological_Process_2026.gmt','KEGG':'KEGG_2026.gmt','Reactome':'Reactome_Pathways_2024.gmt'};SEED=20260909
GENES=np.asarray(pd.read_csv(REPO/'data/ensembl/canonical_genes.csv').gene_symbol.astype(str));G=len(GENES)
sys.path[:0]=[str(REPO/'benchmarks/tcga_downstream/pipeline'),str(REPO)];from run_attention_pooling import load_frozen_encoder

CORE={'controlled_tcell':'controlled','RR1 OSD48':'RR1_OSD48_original_matched','RR1 OSD168':'RR1_OSD168_no-ERCC','RR3-39 OSD137':'C01_OSD137_original_matched','RR3-39 OSD168':'C01_OSD168_all_ERCC','RR3-40 OSD137':'C02_OSD137_original_matched','RR3-40 OSD168':'C02_OSD168_all_ERCC'}
PAIRS=[('RR1','RR1 OSD48','RR1 OSD168'),('RR3-39','RR3-39 OSD137','RR3-39 OSD168'),('RR3-40','RR3-40 OSD137','RR3-40 OSD168')]
def unit(x):return x/max(np.linalg.norm(x),1e-12)
def cos(a,b):
 d=np.linalg.norm(a)*np.linalg.norm(b);return float(a@b/d) if d else np.nan
def score(model,x,direction):return (model._encode_hidden(x).mean(1)*direction).sum(1)
def ig(model,values,direction,device,steps=16,path_batch=4):
 base=torch.zeros((1,len(values)),device=device);obs=torch.as_tensor(np.array(values,dtype=np.float32,copy=True),device=device)[None];target=torch.as_tensor(direction.astype(np.float32),device=device)[None];total=torch.zeros_like(base);alphas=(np.arange(steps,dtype=np.float32)+.5)/steps
 for s in range(0,steps,path_batch):
  a=torch.as_tensor(alphas[s:s+path_batch],device=device)[:,None];x=(base+a*(obs-base)).requires_grad_(True);v=score(model,x,target.expand(len(x),-1));total+=torch.autograd.grad(v.sum(),x)[0].detach().sum(0,keepdim=True)
 return ((obs-base)*total/steps)[0].cpu().numpy()

def four_state_inputs():
 q=pd.read_csv(HERE/'results/task4_rr3_39_40_four_state/animal_metadata_audit.csv');m=pd.read_csv(R3/'sample_manifest.csv');ix=dict(zip(m.sample_id,range(len(m))));x=np.load(W3/'bridgerna_log1p_tpm_inputs.npy',mmap_mode='r');z=np.load(W3/'bridgerna_embeddings.npy');groups={'GC39':['G1','G2'],'FLT39':['F1','F2'],'GC40':['G3','G5'],'FLT40':['F3','F4','F5']};profiles={};emb={};samples={}
 for g,aa in groups.items():
  ids=q[q.animal_id.isin(aa)].sample_id.map(ix).tolist();samples[g]=np.asarray(x[ids],float);profiles[g]=samples[g].mean(0).astype(np.float32);emb[g]=z[ids].mean(0)
 return profiles,emb,samples
def generate_four_state(device_name):
 WORK.mkdir(parents=True,exist_ok=True);cache=WORK/'four_state_full_response_ig.npz'
 if cache.exists():return
 profiles,emb,_=four_state_inputs();device=torch.device(device_name if torch.cuda.is_available() else 'cpu');model=load_frozen_encoder(device);out={}
 for name,a,b in [('RR3 GC39→GC40','GC40','GC39'),('RR3 FLT39→FLT40','FLT40','FLT39')]:
  direction=unit(emb[a]-emb[b]).astype(np.float32);out[name]=ig(model,profiles[a],direction,device)-ig(model,profiles[b],direction,device);print('[heartbeat]',name,flush=True)
 np.savez_compressed(cache,**out)

def all_inputs():
 aa=np.load(GD/'signed_gene_attributions.npz');xx=np.load(GD/'expression_response_vectors.npz');attrs={n:aa[k].astype(float) for n,k in CORE.items()};expr={n:xx[k].astype(float) for n,k in CORE.items() if n!='controlled_tcell'};means={};variances={}
 # Controlled T cells.
 m=pd.read_parquet(HERE/'work/datasets/chen_2020_tcells/manifest.parquet');x=np.load(HERE/'work/datasets/chen_2020_tcells/log1p_tpm.npy',mmap_mode='r');means['controlled_tcell']=np.asarray(x).mean(0);variances['controlled_tcell']=np.asarray(x).var(0);expr['controlled_tcell']=np.asarray(x[m.library_prep.eq('ribo')]).mean(0)-np.asarray(x[m.library_prep.eq('polyA')]).mean(0)
 # Technical contrasts: available expression delta is primary; pooled abundance/variance reconstructed from design.
 from analyze_technical_replication_gene_attributions import design
 specs,X,_=design()
 for n,k in CORE.items():
  if n=='controlled_tcell':continue
  ids=specs[k]['indices'];q=np.asarray(X[ids],float);means[n]=q.mean(0);variances[n]=q.var(0)
 fs=np.load(WORK/'four_state_full_response_ig.npz');profiles,emb,samples=four_state_inputs()
 for n,a,b in [('RR3 GC39→GC40','GC40','GC39'),('RR3 FLT39→FLT40','FLT40','FLT39')]:attrs[n]=fs[n].astype(float);expr[n]=profiles[a].astype(float)-profiles[b].astype(float);q=np.r_[samples[a],samples[b]];means[n]=q.mean(0);variances[n]=q.var(0)
 return attrs,expr,means,variances

def build_tables(attrs,expr,means,variances):
 tables=[];summary=[];kf=KFold(5,shuffle=True,random_state=SEED)
 for name in attrs:
  e=expr[name];a=attrs[name];X=np.c_[e,np.abs(e),means[name],np.log1p(variances[name])];model=make_pipeline(StandardScaler(),LinearRegression());pred=cross_val_predict(model,X,a,cv=kf,n_jobs=1);model.fit(X,a);ins=model.predict(X);res=a-pred
  magmodel=make_pipeline(StandardScaler(),LinearRegression());mpred=cross_val_predict(magmodel,X,np.abs(a),cv=kf,n_jobs=1)
  erank=pd.Series(np.abs(e)).rank(ascending=False,method='min').astype(int).to_numpy();arank=pd.Series(np.abs(a)).rank(ascending=False,method='min').astype(int).to_numpy();rrank=pd.Series(np.abs(res)).rank(ascending=False,method='min').astype(int).to_numpy();et=np.quantile(np.abs(e),.9);at=np.quantile(np.abs(a),.9);em=np.median(np.abs(e));am=np.median(np.abs(a));cat=np.full(G,'other',object);cat[(np.abs(e)>=et)&(np.abs(a)>=at)]='A_high_expression_high_attribution';cat[(np.abs(e)>=et)&(np.abs(a)<=am)]='B_high_expression_low_attribution';cat[(np.abs(e)<=em)&(np.abs(a)>=at)]='C_low_moderate_expression_high_attribution';cat[(np.abs(e)<=em)&(np.abs(a)<=am)]='D_low_expression_low_attribution'
  t=pd.DataFrame({'contrast':name,'gene_symbol':GENES,'expression_change':e,'absolute_expression_change':np.abs(e),'mean_expression':means[name],'expression_variance':variances[name],'attribution':a,'absolute_attribution':np.abs(a),'predicted_attribution_oof':pred,'predicted_absolute_attribution_oof':mpred,'residual_attribution':res,'standardized_residual':(res-res.mean())/(res.std(ddof=1) or 1),'attribution_rank':arank,'expression_rank':erank,'residual_rank':rrank,'sign_agreement':np.sign(e)==np.sign(a),'category':cat});tables.append(t)
  se=set(np.argsort(np.abs(e))[-100:]);sa=set(np.argsort(np.abs(a))[-100:]);ov=list(se&sa);summary.append({'contrast':name,'pearson_signed':np.corrcoef(e,a)[0,1],'spearman_signed':spearmanr(e,a).statistic,'pearson_magnitude':np.corrcoef(np.abs(e),np.abs(a))[0,1],'spearman_magnitude':spearmanr(np.abs(e),np.abs(a)).statistic,'linear_R2_in_sample':r2_score(a,ins),'linear_R2_5fold_CV':r2_score(a,pred),'magnitude_R2_5fold_CV':r2_score(np.abs(a),mpred),'predicted_magnitude_rank_spearman':spearmanr(np.abs(a),mpred).statistic,'top100_overlap':len(ov),'top100_sign_agreement':np.mean(np.sign(e[ov])==np.sign(a[ov])) if ov else np.nan})
 out=pd.concat(tables,ignore_index=True);out.to_parquet(OUT/'contrast_gene_tables.parquet',index=False);pd.DataFrame(summary).to_csv(OUT/'expression_attribution_correspondence.csv',index=False);out[out.category.ne('other')].to_csv(OUT/'discordant_gene_categories.csv',index=False);out.sort_values(['contrast','residual_rank']).groupby('contrast').head(500).to_csv(OUT/'top500_residual_attribution.csv',index=False);return out,pd.DataFrame(summary)

def pathways(tables):
 terms=[]
 for source,file in GMTS.items():
  for term,members in gp.parser.read_gmt(path=str(GMTROOT/file)).items():
   ii=np.flatnonzero(np.isin(GENES,list(set(members))))
   if 10<=len(ii)<=500:terms.append((source,term,ii))
 rows=[]
 for name,q in tables.groupby('contrast',sort=False):
  q=q.set_index('gene_symbol').loc[GENES]
  for signal,col in [('expression','expression_change'),('attribution','attribution'),('residual','residual_attribution')]:
   x=q[col].to_numpy();z=(x-x.mean())/(x.std() or 1)
   for source,term,ii in terms:rows.append({'contrast':name,'signal':signal,'source':source,'pathway':term,'profile_score':z[ii].mean()*np.sqrt(len(ii))})
 d=pd.DataFrame(rows);d.to_parquet(OUT/'expression_attribution_residual_pathways.parquet',index=False);return d

def residual_gsea(tables):
 cache=OUT/'residual_attribution_gsea.parquet'
 if cache.exists():return pd.read_parquet(cache)
 rows=[]
 for name,q in tables.groupby('contrast',sort=False):
  rank=q[['gene_symbol','residual_attribution']].sort_values('residual_attribution',ascending=False)
  for source,file in GMTS.items():
   print('[GSEA]',name,source,flush=True);pre=gp.prerank(rnk=rank,gene_sets=str(GMTROOT/file),min_size=10,max_size=500,permutation_num=250,threads=8,seed=SEED,outdir=None,verbose=False)
   z=pre.res2d.rename(columns={'Term':'pathway','NES':'nes','FDR q-val':'fdr','NOM p-val':'nominal_p','Lead_genes':'leading_edge'});z['contrast']=name;z['source']=source;rows.append(z[['contrast','source','pathway','nes','fdr','nominal_p','leading_edge']])
 out=pd.concat(rows,ignore_index=True);out.to_parquet(cache,index=False);out.to_csv(OUT/'residual_attribution_gsea.csv',index=False);return out

def category_reproducibility(tables):
 rows=[]
 for pair,a,b in PAIRS:
  qa=tables[tables.contrast.eq(a)];qb=tables[tables.contrast.eq(b)]
  for category in ['A_high_expression_high_attribution','B_high_expression_low_attribution','C_low_moderate_expression_high_attribution','D_low_expression_low_attribution']:
   sa=set(qa[qa.category.eq(category)].gene_symbol);sb=set(qb[qb.category.eq(category)].gene_symbol);rows.append({'comparison':pair,'category':category,'n_original':len(sa),'n_remeasurement':len(sb),'overlap':len(sa&sb),'jaccard':len(sa&sb)/len(sa|sb) if sa|sb else np.nan,'genes':';'.join(sorted(sa&sb))})
 out=pd.DataFrame(rows);out.to_csv(OUT/'discordant_category_reproducibility.csv',index=False);return out

def reproducibility(tables,path):
 vectors={(c,s):g.set_index('gene_symbol').loc[GENES,s].to_numpy() for c,g in tables.groupby('contrast') for s in ['expression_change','attribution','residual_attribution']};w=path.pivot(index=['source','pathway'],columns=['contrast','signal'],values='profile_score');rows=[]
 for pair,a,b in PAIRS:
  r={'comparison':pair}
  for s in ['expression_change','attribution','residual_attribution']:
   x,y=vectors[(a,s)],vectors[(b,s)];r[f'{s}_cosine']=cos(x,y);r[f'{s}_spearman']=spearmanr(x,y).statistic
   for n in [100,500]:sa=set(np.argsort(np.abs(x))[-n:]);sb=set(np.argsort(np.abs(y))[-n:]);ov=list(sa&sb);r[f'{s}_top{n}_overlap']=len(ov);r[f'{s}_top{n}_sign_agreement']=np.mean(np.sign(x[ov])==np.sign(y[ov])) if ov else np.nan
   r[f'{s}_pathway_pearson']=w[(a,'residual' if s=='residual_attribution' else 'attribution' if s=='attribution' else 'expression')].corr(w[(b,'residual' if s=='residual_attribution' else 'attribution' if s=='attribution' else 'expression')])
  rows.append(r)
 out=pd.DataFrame(rows);out.to_csv(OUT/'technical_replication_residual_metrics.csv',index=False);return out,vectors,w

def false_friend(vectors,w):
 rows=[]
 for s in ['expression_change','attribution','residual_attribution']:
  x,y=vectors[('RR1 OSD48',s)],vectors[('RR3-39 OSD137',s)];rows.append({'signal':s,'cosine':cos(x,y),'spearman':spearmanr(x,y).statistic,'pathway_pearson':w[('RR1 OSD48','residual' if s=='residual_attribution' else 'attribution' if s=='attribution' else 'expression')].corr(w[('RR3-39 OSD137','residual' if s=='residual_attribution' else 'attribution' if s=='attribution' else 'expression')])})
 d=pd.DataFrame(rows);d.to_csv(OUT/'geometric_false_friend_expression_adjustment.csv',index=False);return d

def aucs(tables):
 names=['RR1 OSD48','RR1 OSD168','RR3-39 OSD137','RR3-39 OSD168','RR3-40 OSD137','RR3-40 OSD168'];true={frozenset([a,b]) for _,a,b in PAIRS};rows=[];pair=[]
 for a,b in combinations(names,2):
  qa=tables[tables.contrast.eq(a)].set_index('gene_symbol').loc[GENES];qb=tables[tables.contrast.eq(b)].set_index('gene_symbol').loc[GENES];pair.append({'a':a,'b':b,'y':int(frozenset([a,b]) in true),'expression_similarity':cos(qa.expression_change, qb.expression_change),'attribution_similarity':cos(qa.attribution,qb.attribution),'residual_similarity':cos(qa.residual_attribution,qb.residual_attribution)})
 d=pd.DataFrame(pair);y=d.y
 scores={'expression_similarity':d.expression_similarity,'attribution_similarity':d.attribution_similarity,'residual_similarity':d.residual_similarity,'expression_plus_residual':(d.expression_similarity.rank(pct=True)+d.residual_similarity.rank(pct=True))/2}
 rng=np.random.default_rng(SEED);pos=np.flatnonzero(y);neg=np.flatnonzero(~y)
 for n,s in scores.items():
  bs=[]
  for _ in range(5000):j=np.r_[rng.choice(pos,len(pos),True),rng.choice(neg,len(neg),True)];bs.append(roc_auc_score(y.iloc[j],s.iloc[j]))
  rows.append({'model':n,'ROC_AUC':roc_auc_score(y,s),'PR_AUC':average_precision_score(y,s),'bootstrap_low':np.quantile(bs,.025),'bootstrap_high':np.quantile(bs,.975)})
 out=pd.DataFrame(rows);out.to_csv(OUT/'added_information_auc.csv',index=False);d.to_csv(OUT/'reproducibility_pair_scores.csv',index=False);return out

def figures(tables,summary,repro,auc):
 plt.style.use('seaborn-v0_8-whitegrid');FIG.mkdir(exist_ok=True);sel=['controlled_tcell','RR1 OSD48','RR1 OSD168','RR3-40 OSD137','RR3-40 OSD168'];fig,axes=plt.subplots(len(sel),2,figsize=(13,4*len(sel)))
 for i,n in enumerate(sel):
  q=tables[tables.contrast.eq(n)];axes[i,0].scatter(q.expression_change,q.attribution,s=2,alpha=.15);axes[i,0].set(xlabel='Expression change',ylabel='IG attribution',title=n);axes[i,1].scatter(q.absolute_expression_change,q.absolute_attribution,s=2,alpha=.15);axes[i,1].set(xlabel='|Expression change|',ylabel='|IG attribution|')
 fig.tight_layout();
 for e in ['png','pdf']:fig.savefig(FIG/f'expression_vs_attribution.{e}',dpi=400);plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(17,5));axes[0].bar(summary.contrast,summary.spearman_signed);axes[0].tick_params(axis='x',rotation=75);axes[0].set(title='Signed correspondence',ylabel='Spearman');axes[1].bar(repro.comparison,repro.residual_attribution_cosine);axes[1].set(title='Residual attribution replication',ylabel='Cosine');axes[2].barh(auc.model,auc.ROC_AUC);axes[2].set(xlim=(0,1),title='Descriptive replication AUC');fig.tight_layout()
 for e in ['png','pdf']:fig.savefig(FIG/f'summary.{e}',dpi=400,bbox_inches='tight');plt.close(fig)

def main(device):
 OUT.mkdir(parents=True,exist_ok=True);WORK.mkdir(exist_ok=True);generate_four_state(device);attrs,expr,means,var=all_inputs();tables,summary=build_tables(attrs,expr,means,var);path=pathways(tables);gsea=residual_gsea(tables);cats=category_reproducibility(tables);repro,vectors,w=reproducibility(tables,path);ff=false_friend(vectors,w);auc=aucs(tables);figures(tables,summary,repro,auc)
 decision={'contrasts':len(attrs),'genes_per_contrast':G,'median_signed_spearman':float(summary.spearman_signed.median()),'median_magnitude_spearman':float(summary.spearman_magnitude.median()),'median_cv_R2':float(summary.linear_R2_5fold_CV.median()),'residual_cosines':dict(zip(repro.comparison,repro.residual_attribution_cosine)),'decision':'MIXED / INSUFFICIENT EVIDENCE','note':'Residual attribution is out-of-fold linear residual against signed/absolute expression change, mean expression, and log1p variance; it is not automatically biology.'};(OUT/'summary.json').write_text(json.dumps(decision,indent=2)+'\n');(OUT/'provenance.json').write_text(json.dumps({'BridgeRNA_retrained':False,'existing_IG_reused':7,'new_IG_only':['RR3 GC39→GC40','RR3 FLT39→FLT40'],'IG_baseline':'zero log1p(TPM)','IG_steps':16,'prediction':'5-fold gene-level OOF linear regression; features signed delta expression, absolute delta, mean, log1p variance','pathway_profile':'standardized mean signed signal times sqrt(set size); descriptive','claim_scope':'three technical remeasurements, not independent biological replication'},indent=2)+'\n');print(summary.to_string(index=False));print('\nResidual replication\n',repro.to_string(index=False));print('\nFalse friend\n',ff.to_string(index=False));print('\nAUC\n',auc.to_string(index=False));print('\n',json.dumps(decision,indent=2));print('[complete]',OUT)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');main(p.parse_args().device)
