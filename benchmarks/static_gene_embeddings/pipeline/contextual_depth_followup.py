#!/usr/bin/env python3
"""Lightweight per-gene contextual-depth functional organization analysis."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np,pandas as pd,torch
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parents[1]
OUT=HERE/'results/contextual_depth';WORK=HERE/'work/contextual_depth'
EXPR=ROOT/'benchmarks/tcga_downstream/work/ours_log1p_tpm.npy'
META=ROOT/'benchmarks/tcga_downstream/results/cohort_manifest.parquet'
VOCAB=ROOT/'data/ensembl/canonical_genes.csv'
GMTS={'GO':ROOT/'benchmarks/cross_species_exercise_response/results/per_study_ranked_gsea/GO_Biological_Process_2026.gmt','KEGG':ROOT/'benchmarks/cross_species_exercise_response/results/per_study_ranked_gsea/KEGG_2026.gmt'}
LAYERS={'L0_static':0,'L1_early':1,'L6_middle':6,'L12_final':12};KS=[10,25,50];GEOMS=['raw_cosine','mean_centered_cosine']

class Log:
 def __init__(self,p):self.p=p;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('')
 def __call__(self,s):
  z=f'[{pd.Timestamp.utcnow().isoformat()}] {s}';print(z,flush=True)
  with self.p.open('a') as f:f.write(z+'\n')

def gmt(path,genes):
 u=set(genes);d={}
 for line in path.read_text().splitlines():
  x=line.split('\t');m=set(x[1:])&u
  if 5<=len(m)<=2000:d[x[0]]=m
 return d

def bits_for(genes,terms):
 ix={g:i for i,g in enumerate(genes)};b=[0]*len(genes)
 for j,s in enumerate(terms.values()):
  q=1<<j
  for g in s:b[ix[g]]|=q
 return b

def choose_samples(meta,n):
 preferred=['GBM','BRCA','LUAD','KIRC','LIHC','PRAD','COAD','SKCM','THCA','OV','LUSC','STAD']
 q=meta[meta.sample_type.eq('Primary Tumor')].copy();rows=[]
 for c in preferred:
  z=q[q.cohort.eq(c)].sort_values('sample_id')
  if len(z):rows.append(z.iloc[len(z)//2])
  if len(rows)==n:break
 if len(rows)<n:raise ValueError('Insufficient diverse cohorts')
 return pd.DataFrame(rows).reset_index(drop=True)

def load_model(device):
 sys.path.insert(0,str(ROOT/'benchmarks/tcga_downstream/pipeline'))
 from run_attention_pooling import load_frozen_encoder
 return load_frozen_encoder(device).eval()

def extract(samples,device,log):
 x=np.load(EXPR,mmap_mode='r');model=load_model(device);ids=torch.arange(15165,device=device);cache={k:[] for k in LAYERS};started=time.time()
 with torch.no_grad():
  for si,row in samples.iterrows():
   v=torch.as_tensor(np.array(x[int(row.matrix_row)],copy=True)[None],device=device)
   cache['L0_static'].append(model.gene_embedding(ids).detach().cpu().half().numpy())
   with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
    h=model.gene_embedding(ids).unsqueeze(0)+model.ree(v)
    for li,layer in enumerate(model.layers,1):
     h=layer(h)
     key=next((k for k,n in LAYERS.items() if n==li),None)
     if key:cache[key].append(h[0].detach().cpu().half().numpy())
   log(f"Extracted sample {si+1}/{len(samples)} cohort={row.cohort} elapsed={(time.time()-started)/60:.1f}m")
 for key,a in cache.items():np.save(WORK/f'{key}.float16.npy',np.stack(a))

def neighbors(z,k=50,batch=512):
 z=torch.nn.functional.normalize(z.float(),dim=1);out=np.empty((len(z),k),np.int32)
 for a in range(0,len(z),batch):
  s=z[a:a+batch]@z.T;r=torch.arange(len(s),device=z.device);s[r,torch.arange(a,a+len(s),device=z.device)]=-torch.inf;out[a:a+len(s)]=torch.topk(s,k).indices.cpu().numpy()
 return out

def overlap(bits,nbr,k):
 hit=den=0;per=[]
 for i,row in enumerate(nbr[:,:k]):
  if not bits[i]:continue
  v=[j for j in row if bits[j]]
  if v:
   h=sum(bool(bits[i]&bits[j]) for j in v);hit+=h;den+=len(v);per.append(h/len(v))
 return hit/den,float(np.mean(per)),den

def matched_pairs(bits,n=5000,seed=42):
 rng=np.random.default_rng(seed);ann=np.array([i for i,b in enumerate(bits) if b]);deg=np.array([b.bit_count() for b in bits]);edges=[]
 while len(edges)<n:
  a,b=rng.choice(ann,2,replace=False)
  if bits[a]&bits[b]:edges.append((a,b))
 edges=np.array(edges);neg=[]
 bins=np.quantile(deg[ann],[0,.25,.5,.75,1]);groups=[ann[(deg[ann]>=bins[i])&(deg[ann]<=bins[i+1])] for i in range(4)]
 for a,b in edges:
  ia=min(3,np.searchsorted(bins[1:],deg[a],side='right'));ib=min(3,np.searchsorted(bins[1:],deg[b],side='right'))
  for _ in range(100):
   c=rng.choice(groups[ia]);d=rng.choice(groups[ib])
   if c!=d and not(bits[c]&bits[d]):neg.append((c,d));break
 if len(neg)!=n:raise RuntimeError('Could not construct matched negatives')
 return edges,np.array(neg)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--samples',type=int,default=8);ap.add_argument('--device',default='cuda:0');ap.add_argument('--null-reps',type=int,default=10);a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True);log=Log(OUT/'run.log');started=time.time();dev=torch.device(a.device if torch.cuda.is_available() else 'cpu')
 genes=pd.read_csv(VOCAB).sort_values('token_id').gene_symbol.astype(str).str.upper().tolist();meta=pd.read_parquet(META);samples=choose_samples(meta,a.samples);samples.to_csv(OUT/'sample_manifest.csv',index=False);extract(samples,dev,log)
 libraries={n:(gmt(p,genes)) for n,p in GMTS.items()};bitsets={n:bits_for(genes,t) for n,t in libraries.items()};pairs={n:matched_pairs(bitsets[n],seed=42+i) for i,n in enumerate(libraries)};expr=np.load(EXPR,mmap_mode='r');rng=np.random.default_rng(77);metrics=[];pairrows=[];example=[];anchor_names=['TP53','ESR1','PAX8','ALB','CD3D','MKI67'];gix={g:i for i,g in enumerate(genes)};anchors=[g for g in anchor_names if g in gix]
 nulls={}
 for lib,bits in bitsets.items():
  ann=np.array([i for i,b in enumerate(bits) if b]);nulls[lib]={}
  for k in KS:
   vals=[]
   for rep in range(a.null_reps):vals.append(overlap(bits,rng.choice(ann,size=(len(genes),k),replace=True),k)[0])
   nulls[lib][k]=np.array(vals)
 for layer in LAYERS:
  arr=np.load(WORK/f'{layer}.float16.npy',mmap_mode='r');sample_range=range(1) if layer=='L0_static' else range(len(samples))
  for si in sample_range:
   base=torch.as_tensor(np.asarray(arr[si],dtype=np.float32),device=dev)
   for geom in GEOMS:
    z=base if geom=='raw_cosine' else base-base.mean(0,keepdim=True);nbr=neighbors(z)
    for lib,bits in bitsets.items():
     for k in KS:
      rate,per,den=overlap(bits,nbr,k);nul=nulls[lib][k];metrics.append({'layer':layer,'sample_index':si,'cohort':samples.iloc[si].cohort if layer!='L0_static' else 'sample_invariant','geometry':geom,'library':lib,'k':k,'shared_pair_fraction':rate,'mean_gene_fraction':per,'eligible_pairs':den,'null_mean':nul.mean(),'fold_over_null':rate/nul.mean(),'empirical_p':(1+(nul>=rate).sum())/(len(nul)+1)})
     pos,neg=pairs[lib];sim=torch.nn.functional.normalize(z.float(),dim=1);ps=(sim[pos[:,0]]*sim[pos[:,1]]).sum(1).cpu().numpy();ns=(sim[neg[:,0]]*sim[neg[:,1]]).sum(1).cpu().numpy();y=np.r_[np.ones(len(ps)),np.zeros(len(ns))];score=np.r_[ps,ns];pairrows.append({'layer':layer,'sample_index':si,'cohort':samples.iloc[si].cohort if layer!='L0_static' else 'sample_invariant','geometry':geom,'library':lib,'auroc':roc_auc_score(y,score),'mean_positive_similarity':ps.mean(),'mean_negative_similarity':ns.mean(),'standardized_mean_difference':(ps.mean()-ns.mean())/np.sqrt((ps.var()+ns.var())/2),'pairs_per_class':len(ps)})
    # examples use centered cosine, top 10
    z=base-base.mean(0,keepdim=True);nbr=neighbors(z,25)
    for gene in anchors:
     gi=gix[gene]
     for rank,j in enumerate(nbr[gi,:10],1):example.append({'layer':layer,'sample_index':si,'cohort':samples.iloc[si].cohort if layer!='L0_static' else 'sample_invariant','anchor_gene':gene,'rank':rank,'neighbor_gene':genes[j],'anchor_log1p_tpm':float(expr[int(samples.iloc[si].matrix_row),gi]) if layer!='L0_static' else np.nan,'shares_go':bool(bitsets['GO'][gi]&bitsets['GO'][j]),'shares_kegg':bool(bitsets['KEGG'][gi]&bitsets['KEGG'][j])})
   log(f"Metrics complete layer={layer} sample={si+1}/{len(list(sample_range))}")
 pd.DataFrame(metrics).to_csv(OUT/'neighbor_enrichment_per_sample.csv',index=False);pd.DataFrame(pairrows).to_csv(OUT/'related_pair_retrieval_per_sample.csv',index=False);ex=pd.DataFrame(example);ex.to_csv(OUT/'selected_gene_context_neighborhoods.csv',index=False)
 m=pd.DataFrame(metrics);p=pd.DataFrame(pairrows);summary=m.groupby(['layer','geometry','library','k'],as_index=False).agg(neighbor_fold=('fold_over_null','mean'),neighbor_fold_sd=('fold_over_null','std'),samples=('sample_index','nunique'),null_mean=('null_mean','first')).merge(p.groupby(['layer','geometry','library'],as_index=False).agg(related_pair_auroc=('auroc','mean'),auroc_sd=('auroc','std'),pair_smd=('standardized_mean_difference','mean')),on=['layer','geometry','library']);summary.to_csv(OUT/'layerwise_summary.csv',index=False)
 order=list(LAYERS);fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
 q=summary[(summary.geometry=='mean_centered_cosine')&(summary.k==25)]
 for lib,color in [('GO','#2878B5'),('KEGG','#E07A1F')]:
  z=q[q.library==lib].set_index('layer').loc[order];axes[0].errorbar(range(4),z.neighbor_fold,yerr=z.neighbor_fold_sd.fillna(0),marker='o',label=lib,color=color);axes[1].errorbar(range(4),z.related_pair_auroc,yerr=z.auroc_sd.fillna(0),marker='o',label=lib,color=color)
 axes[0].axhline(1,color='black',ls='--',lw=1);axes[0].set(ylabel='Neighbor overlap / random',title='Functional nearest-neighbor enrichment');axes[1].axhline(.5,color='black',ls='--',lw=1);axes[1].set(ylabel='Related-pair AUROC',title='Known relationship retrieval')
 for ax in axes:ax.set_xticks(range(4),['L0 static','L1 early','L6 middle','L12 final']);ax.set_xlabel('Representation depth');ax.legend()
 for ext in ['png','pdf']:fig.savefig(OUT/f'contextual_depth_summary.{ext}',dpi=300,bbox_inches='tight')
 plt.close(fig)
 ctx=[]
 for (layer,anchor),q in ex.groupby(['layer','anchor_gene']):
  sets={c:set(z.neighbor_gene) for c,z in q.groupby('cohort')};vals=[len(sets[x]&sets[y])/len(sets[x]|sets[y]) for i,x in enumerate(sets) for y in list(sets)[i+1:]];ctx.append({'layer':layer,'anchor_gene':anchor,'mean_cross_context_neighbor_jaccard':np.mean(vals) if vals else np.nan,'contexts':len(sets),'go_coherent_fraction':q.shares_go.mean(),'kegg_coherent_fraction':q.shares_kegg.mean()})
 pd.DataFrame(ctx).to_csv(OUT/'selected_gene_context_dependence.csv',index=False)
 prov={'status':'complete','profile':'lightweight_exploratory','samples':len(samples),'sample_selection':'one deterministic primary tumor from each of eight distinct TCGA cohorts','layers':LAYERS,'L0_definition':'pure gene_embedding.weight; sample invariant; excludes expression rotary embedding','individual_gene_tokens':True,'neighbor_candidates':'other genes within the same sample; same-gene cross-sample identity matches excluded by design','geometries':GEOMS,'pc_removal':'not run in lightweight screen','ks':KS,'annotation_background_genes':15165,'matched_pairs_per_class':5000,'null_reps':a.null_reps,'limitations':['TCGA tumors are biologically diverse but not healthy tissue controls','small exploratory sample count','random controls are degree-quartile matched for pair retrieval and annotation-eligible for neighborhoods','empirical p-values are coarse with 10 null replicates'],'elapsed_seconds':time.time()-started};(OUT/'provenance.json').write_text(json.dumps(prov,indent=2)+'\n');log(f"Complete elapsed={(time.time()-started)/60:.1f}m")
if __name__=='__main__':main()
