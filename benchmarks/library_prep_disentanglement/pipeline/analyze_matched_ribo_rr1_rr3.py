#!/usr/bin/env python3
"""Matched-ribodepletion diagnostic for RR1/RR3 response relationships."""
from __future__ import annotations
import json,sys,time
from pathlib import Path
import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import hypergeom,pearsonr,spearmanr

HERE=Path(__file__).resolve().parents[1];REPO=HERE.parents[1];OUT=HERE/'results/task4_matched_ribo_rr1_rr3';FIG=OUT/'figures';WORK=HERE/'work/task4_matched_ribo_rr1_rr3'
T3=REPO/'benchmarks/osdr_batch_effect_representation';R3=T3/'results';W3=T3/'work';GMTROOT=REPO/'benchmarks/cross_species_exercise_response/results/per_study_ranked_gsea'
GMTS={'GO:BP':'GO_Biological_Process_2026.gmt','KEGG':'KEGG_2026.gmt','Reactome':'Reactome_Pathways_2024.gmt'};GENES=np.asarray(pd.read_csv(REPO/'data/ensembl/canonical_genes.csv').gene_symbol);G=len(GENES);SEED=90731
sys.path[:0]=[str(REPO/'benchmarks/tcga_downstream/pipeline'),str(REPO)];from run_attention_pooling import load_frozen_encoder

def cos(a,b):
 d=np.linalg.norm(a)*np.linalg.norm(b);return float(a@b/d) if d else np.nan
def fixed_basis():
 m=pd.read_parquet(HERE/'work/datasets/chen_2020_tcells/manifest.parquet').reset_index(drop=True);z=np.load(HERE/'work/datasets/chen_2020_tcells/bridgerna_embeddings.npy').astype(float);d=[]
 for _,q in m.groupby('pair_id',sort=True):d.append(z[q.index[q.library_prep.eq('ribo')]].mean(0)-z[q.index[q.library_prep.eq('polyA')]].mean(0))
 return np.linalg.svd(np.stack(d),full_matrices=False)[2][:2]

def groups():
 m=pd.read_csv(R3/'sample_manifest.csv');membership=pd.read_csv(R3/'task3b_contrast_sample_membership.csv');result={}
 for name,cid in {'RR1 OSD-48 carcass':'C14__OSD-48__RR1-NASA__37-day','RR1 OSD-48 upon-euthanasia':'C13__OSD-48__RR1-NASA__37-day','RR3-39 OSD-137':'C01__OSD-137__RR3__39-day','RR3-40 OSD-137':'C02__OSD-137__RR3__40-day'}.items():
  q=membership[membership.contrast_id.eq(cid)];result[name]={c:q[q.condition.eq(c)].sample_id.tolist() for c in ['FLT','GC']}
 q=m[(m.OSD.astype(str).str.contains('168'))&m.sample_id.str.contains('RR1')];result['RR1 OSD-168 Ribo']={c:q[q.condition.eq(c)].sample_id.tolist() for c in ['FLT','GC']}
 design=pd.read_csv(R3/'task3_osd168_technical_replication/technical_response_design.csv').set_index('representation')
 samples=design.loc['C02_OSD168_all_ERCC','samples'].split(' | ');result['RR3-40 OSD-168']={c:[s for s in samples if f'_{c}_' in s] for c in ['FLT','GC']}
 return m,result

def audit(m,gr):
 rows=[]
 for cohort,cs in gr.items():
  for condition,ids in cs.items():
   q=m.set_index('sample_id').loc[ids].reset_index();animals=q.sample_id.str.extract(r'_(M\d+|F\d+|G\d+)$')[0]
   rows.append({'cohort':cohort,'OSD':'OSD-168' if 'OSD-168' in cohort else 'OSD-48' if 'OSD-48' in cohort else 'OSD-137','mission':q.mission.mode().iloc[0],
    'FLT_GC':condition,'library_preparation':q.library_preparation.mode().iloc[0],'sample_state':'carcass' if 'carcass' in cohort else 'upon-euthanasia' if 'euthanasia' in cohort else 'not equivalent/unspecified',
    'profiles_n':len(ids),'independent_animals_n':animals.nunique(),'sample_ids':' | '.join(ids)})
 d=pd.DataFrame(rows);d.to_csv(OUT/'comparison_matrix_sample_audit.csv',index=False);return d

def response_vectors(m,gr):
 ix=dict(zip(m.sample_id,range(len(m))));z=np.load(W3/'bridgerna_embeddings.npy');res={};individual={}
 for name,cs in gr.items():
  collapsed={}
  for c,ids in cs.items():
   q=pd.DataFrame({'sample_id':ids});q['animal']=q.sample_id.str.extract(r'_(M\d+|F\d+|G\d+)$')[0]
   collapsed[c]=np.stack([z[[ix[s] for s in g.sample_id]].mean(0) for _,g in q.groupby('animal',sort=True)])
  means={c:v.mean(0) for c,v in collapsed.items()};res[name]=means['FLT']-means['GC'];individual[name]=collapsed
 return res,individual

def context_response(m,gr,device='cuda:0'):
 # Reuse five cached responses; only full 10/10 OSD-168 RR1 requires aggregation.
 old=np.memmap(HERE/'work/task4_rna_processing_gene_analysis/matched_contextual_responses.float16.dat',dtype='float16',mode='r',shape=(6,G,512));full=np.memmap(HERE/'work/task4_selective_tcell_signature_filter/full_biological_contextual_responses.float16.dat',dtype='float16',mode='r',shape=(3,G,512))
 result={'RR1 OSD-48 carcass':np.asarray(full[0],np.float32),'RR1 OSD-48 upon-euthanasia':np.asarray(full[1],np.float32),'RR3-39 OSD-137':np.asarray(old[2],np.float32),'RR3-40 OSD-137':np.asarray(full[2],np.float32),'RR3-40 OSD-168':np.asarray(old[5],np.float32)}
 cache=WORK/'rr1_osd168_full_10x10_context_response.float16.dat';shape=(G,512);WORK.mkdir(parents=True,exist_ok=True)
 if not(cache.exists() and cache.stat().st_size==np.prod(shape)*2):
  ix=dict(zip(m.sample_id,range(len(m))));x=np.load(W3/'bridgerna_log1p_tpm_inputs.npy',mmap_mode='r');dev=torch.device(device if torch.cuda.is_available() else 'cpu');model=load_frozen_encoder(dev);mm=np.memmap(cache,dtype='float16',mode='w+',shape=shape);means={};started=time.time()
  for c,ids in gr['RR1 OSD-168 Ribo'].items():
   acc=np.zeros(shape,np.float32)
   for j,s in enumerate(ids):
    with torch.no_grad(),torch.autocast(device_type=dev.type,dtype=torch.float16,enabled=dev.type=='cuda'):h=model._encode_hidden(torch.as_tensor(np.array(x[ix[s]],copy=True)[None],dtype=torch.float32,device=dev))[0].float().cpu().numpy()
    acc+=h;print(f'[heartbeat] OSD-168 context {c} {j+1}/{len(ids)} elapsed={(time.time()-started)/60:.1f}m',flush=True)
   means[c]=acc/len(ids)
  mm[:]=(means['FLT']-means['GC']).astype(np.float16);mm.flush();del mm
 result['RR1 OSD-168 Ribo']=np.asarray(np.memmap(cache,dtype='float16',mode='r',shape=shape),np.float32);return result

def attribution(ctx,B):
 latent={};attr={};signed={};ref=None
 for n,h in ctx.items():
  c=h@B.T/G;latent[n]=c.sum(0);attr[n]=c.reshape(-1)
 return latent,attr

def pathway_profiles(ctx,B):
 # Orient all scores to the independently controlled T-cell direction.
 tc=np.memmap(HERE/'work/task4_controlled_gene_context/controlled_context_displacements.float16.dat',dtype='float16',mode='r',shape=(40,G,512));mean=np.empty((G,512),np.float32)
 for s in range(0,G,256):mean[s:s+256]=np.asarray(tc[:,s:s+256],np.float32).mean(0)
 orient=(mean@B.T).sum(0);orient/=np.linalg.norm(orient)
 terms=[]
 for source,file in GMTS.items():
  for term,members in gp.parser.read_gmt(path=str(GMTROOT/file)).items():
   ii=np.flatnonzero(np.isin(GENES,list(set(members))))
   if 10<=len(ii)<=500:terms.append((source,term,ii))
 rows=[]
 for n,h in ctx.items():
  score=(h@B.T/G)@orient;z=(score-score.mean())/(score.std() or 1)
  for source,term,ii in terms:rows.append({'cohort':n,'source':source,'pathway':term,'score':z[ii].mean()*np.sqrt(len(ii))})
 d=pd.DataFrame(rows);d.to_parquet(OUT/'pathway_attribution_profiles.parquet',index=False);return d.pivot(index=['source','pathway'],columns='cohort',values='score')

def bootstrap(a,b,B,reps=1000):
 rng=np.random.default_rng(SEED);vals=[]
 for _ in range(reps):
  ra=a['FLT'][rng.integers(len(a['FLT']),size=len(a['FLT']))].mean(0)-a['GC'][rng.integers(len(a['GC']),size=len(a['GC']))].mean(0)
  rb=b['FLT'][rng.integers(len(b['FLT']),size=len(b['FLT']))].mean(0)-b['GC'][rng.integers(len(b['GC']),size=len(b['GC']))].mean(0)
  vals.append(cos(ra,rb))
 return np.quantile(vals,[.025,.5,.975])

def comparisons(res,individual,latent,attr,path,B):
 specs=[('cross-library','RR1 OSD-48 carcass','RR3-39 OSD-137'),('cross-library','RR1 OSD-48 carcass','RR3-40 OSD-137'),('cross-library','RR1 OSD-48 upon-euthanasia','RR3-39 OSD-137'),('cross-library','RR1 OSD-48 upon-euthanasia','RR3-40 OSD-137'),('matched-Ribo','RR1 OSD-168 Ribo','RR3-39 OSD-137'),('matched-Ribo','RR1 OSD-168 Ribo','RR3-40 OSD-137'),('matched-Ribo same OSD','RR1 OSD-168 Ribo','RR3-40 OSD-168'),('technical control','RR1 OSD-48 carcass','RR1 OSD-168 Ribo'),('technical control','RR3-40 OSD-137','RR3-40 OSD-168')]
 rows=[]
 for kind,a,b in specs:
  x,y=res[a],res[b];px=(x@B.T)@B;py=(y@B.T)@B;ox=x-px;oy=y-py;av,bv=attr[a],attr[b];q=path[[a,b]].dropna();lo,med,hi=bootstrap(individual[a],individual[b],B)
  row={'comparison_type':kind,'response_A':a,'response_B':b,'full_latent_cosine':cos(x,y),'PC1_2_cosine':cos(px,py),'outside_PC1_2_cosine':cos(ox,oy),'euclidean_distance':np.linalg.norm(x-y),'bootstrap_cosine_low':lo,'bootstrap_cosine_median':med,'bootstrap_cosine_high':hi,
   'attribution_cosine':cos(av,bv),'attribution_pearson':pearsonr(av,bv).statistic,'attribution_spearman':spearmanr(av,bv).statistic,'pathway_pearson':q[a].corr(q[b]),'pathway_spearman':q[a].corr(q[b],method='spearman')}
  ma=np.linalg.norm(av.reshape(G,2),axis=1);mb=np.linalg.norm(bv.reshape(G,2),axis=1)
  for n in [100,250,500,1000]:
   sa=set(np.argpartition(ma,-n)[-n:]);sb=set(np.argpartition(mb,-n)[-n:]);ov=list(sa&sb);row.update({f'top{n}_overlap':len(ov),f'top{n}_jaccard':len(ov)/(2*n-len(ov)),f'top{n}_p':hypergeom.sf(len(ov)-1,G,n,n)})
  rows.append(row)
 d=pd.DataFrame(rows);d.to_csv(OUT/'matched_ribo_comparison_summary.csv',index=False);return d

def figures(d):
 plt.style.use('seaborn-v0_8-whitegrid');FIG.mkdir(exist_ok=True);q=d[d.comparison_type.isin(['cross-library','matched-Ribo'])].copy();q['RR3']=q.response_B.str.extract(r'(RR3-\d+)')[0];state=np.where(q.response_A.str.contains('carcass'),'carcass','euthanasia');q['RR1 source']=np.where(q.response_A.str.contains('168'),'OSD-168 Ribo',pd.Series(state,index=q.index).map(lambda x:f'OSD-48 PolyA {x}'))
 metrics=[('full_latent_cosine','Full latent'),('PC1_2_cosine','PC1–2'),('attribution_cosine','Attribution'),('pathway_pearson','Pathway')];fig,axes=plt.subplots(2,2,figsize=(13,9),sharey=True)
 for ax,(col,title) in zip(axes.flat,metrics):
  p=q.pivot(index='RR3',columns='RR1 source',values=col);p.plot.bar(ax=ax);ax.axhline(0,color='k',lw=.8);ax.set_title(title);ax.set_ylabel('Similarity');ax.tick_params(axis='x',rotation=0);ax.legend(fontsize=7)
 fig.suptitle('RR1→RR3 response agreement: cross-library versus matched ribodepletion',fontweight='bold');fig.tight_layout()
 for e in ['png','pdf']:fig.savefig(FIG/f'central_matched_ribo_comparison.{e}',dpi=400)
 plt.close(fig)

def main():
 OUT.mkdir(parents=True,exist_ok=True);m,gr=groups();audit(m,gr);B=fixed_basis();res,individual=response_vectors(m,gr);ctx=context_response(m,gr);latent,attr=attribution(ctx,B);path=pathway_profiles(ctx,B);d=comparisons(res,individual,latent,attr,path,B);figures(d)
 np.savez_compressed(OUT/'response_vectors.npz',**{k.replace(' ','_').replace('→','to'):v for k,v in res.items()});(OUT/'provenance.json').write_text(json.dumps({'basis':'unchanged controlled T-cell PC1-2','OSD168_RR1':'all 10 FLT and 10 GC profiles; 5+5 animals each represented as no-ERCC and ERCC technical profiles','response_aggregation':'technical profiles are averaged within animal before FLT/GC means; with two profiles per animal this equals the all-profile point estimate','bootstrap':'1000 animal-level resamples; confidence intervals are descriptive for small cohorts','correction':False},indent=2)+'\n');print(d.to_string(index=False));print('[complete]',OUT)
if __name__=='__main__':main()
