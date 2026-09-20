#!/usr/bin/env python3
"""Matched exploratory contextual geometry in balanced GTEx and TCGA cohorts."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np,pandas as pd,torch,umap
from scipy import sparse
from scipy.stats import hypergeom
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score,silhouette_score
from statsmodels.stats.multitest import multipletests
from contextual_depth_followup import ROOT,HERE,VOCAB,GMTS,LAYERS,KS,GEOMS,Log,gmt,bits_for,neighbors,overlap,load_model

OUT=HERE/'results/matched_gtex_tcga';WORK=HERE/'work/matched_gtex_tcga';N=20;SEED=20260920
GTEX_X=ROOT/'benchmarks/landmark_gene_sufficiency/work/gtex_model_log1p_tpm.npy';GTEX_ROWS=ROOT/'benchmarks/landmark_gene_sufficiency/results/gtex_model_samples.parquet';GTEX_META=ROOT/'benchmarks/mouse_encode/results/task1a_balanced_geometry/gtex_human_sample_manifest.parquet'
TCGA_X=ROOT/'benchmarks/tcga_downstream/work/ours_log1p_tpm.npy';TCGA_META=ROOT/'benchmarks/tcga_downstream/results/cohort_manifest.parquet'
GTEX_TISSUES=['heart','skeletal muscle','lung','liver','cortex','cerebellum','pancreas','spleen','testis','subcutaneous adipose']
TCGA_COHORTS=['BRCA','COAD','GBM','KIRC','LIHC','LUAD','PRAD','SKCM','STAD','THCA']

def cohorts():
 gm=pd.read_parquet(GTEX_META);rows=pd.read_parquet(GTEX_ROWS);gm=gm.merge(rows,on='sample_id');parts=[]
 for t in GTEX_TISSUES:parts.append(gm[gm.tissue.eq(t)].sample(N,random_state=SEED).assign(dataset='GTEx',context=t,matrix_row=lambda x:x.row_index))
 tm=pd.read_parquet(TCGA_META);tm=tm[tm.sample_type.eq('Primary Tumor')];
 for t in TCGA_COHORTS:parts.append(tm[tm.cohort.eq(t)].sample(N,random_state=SEED).assign(dataset='TCGA',context=t))
 q=pd.concat(parts,ignore_index=True);q['analysis_index']=np.arange(len(q));return q[['analysis_index','dataset','context','sample_id','matrix_row']]

def ann_matrix(genes,terms):
 gi={g:i for i,g in enumerate(genes)};rr=[];cc=[]
 for j,s in enumerate(terms.values()):
  for g in s:rr.append(gi[g]);cc.append(j)
 return sparse.csr_matrix((np.ones(len(rr),np.int8),(rr,cc)),shape=(len(genes),len(terms)))

def spherical(z,k=10,seed=42,iters=8):
 z=torch.nn.functional.normalize(z.float(),dim=1);rng=np.random.default_rng(seed);cent=z[torch.as_tensor(rng.choice(len(z),k,replace=False),device=z.device)]
 for _ in range(iters):
  lab=(z@cent.T).argmax(1);new=[]
  for c in range(k):
   q=z[lab==c];new.append(q.mean(0) if len(q) else cent[c])
  cent=torch.nn.functional.normalize(torch.stack(new),dim=1)
 return lab.cpu().numpy(),z

def cluster_geometry(z):
 a,zn=spherical(z,seed=42);b,_=spherical(z,seed=43);pick=np.random.default_rng(42).choice(len(a),1000,replace=False);return a,silhouette_score(zn[pick].cpu().numpy(),a[pick],metric='cosine'),adjusted_rand_score(a,b)

def cluster_enrichment(a,mat,term_sizes,ann_count):
 sig=[];annotated=np.asarray(mat.sum(1)).ravel()>0
 for c in range(10):
  q=a==c;n=(q&annotated).sum();hits=np.asarray(mat[q].sum(0)).ravel();p=hypergeom.sf(hits-1,ann_count,term_sizes,n);sig.append(int((multipletests(p,method='fdr_bh')[1]<.05).sum()))
 return sum(sig)

def visualize(means,genes,out):
 rng=np.random.default_rng(42);chosen=np.sort(rng.choice(len(genes),2000,replace=False));rows=[]
 for layer in LAYERS:
  pair=np.vstack([means[('GTEx',layer)][chosen],means[('TCGA',layer)][chosen]]);p=PCA(50,random_state=42).fit_transform(pair);ts=TSNE(2,perplexity=30,init='pca',learning_rate='auto',max_iter=1000,random_state=42).fit_transform(p);um=umap.UMAP(n_neighbors=30,min_dist=.1,metric='cosine',random_state=42).fit_transform(p);base=(means[('GTEx',layer)][chosen]+means[('TCGA',layer)][chosen])/2;cl=KMeans(10,random_state=42,n_init=10).fit_predict(base)+1
  for di,d in enumerate(['GTEx','TCGA']):
   sl=slice(di*len(chosen),(di+1)*len(chosen));rows.append(pd.DataFrame({'dataset':d,'layer':layer,'gene':np.array(genes)[chosen],'cluster':cl,'tsne_1':ts[sl,0],'tsne_2':ts[sl,1],'umap_1':um[sl,0],'umap_2':um[sl,1]}))
 q=pd.concat(rows);q.to_csv(out/'matched_projections.csv',index=False);fig,ax=plt.subplots(4,4,figsize=(15,14),constrained_layout=True)
 for i,l in enumerate(LAYERS):
  for j,(d,x,y,title) in enumerate([('GTEx','tsne_1','tsne_2','t-SNE'),('TCGA','tsne_1','tsne_2','t-SNE'),('GTEx','umap_1','umap_2','UMAP'),('TCGA','umap_1','umap_2','UMAP')]):
   z=q[(q.layer==l)&(q.dataset==d)];ax[i,j].scatter(z[x],z[y],c=z.cluster,s=3,cmap='tab10',vmin=.5,vmax=10.5,rasterized=True);ax[i,j].set(title=f'{d} {l} {title}',xticks=[],yticks=[])
 for ext in ['png','pdf']:fig.savefig(out/f'matched_contextual_projections.{ext}',dpi=180,bbox_inches='tight')
 plt.close(fig)

def plot_summary(summary,dep,out):
 order=list(LAYERS);fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
 q=summary[(summary.geometry=='mean_centered_cosine')&(summary.k==25)]
 for dataset,ls in [('GTEx','-'),('TCGA','--')]:
  for lib,color in [('GO','#2878B5'),('KEGG','#E07A1F')]:
   z=q[(q.dataset==dataset)&(q.library==lib)].set_index('layer').loc[order];axes[0].plot(range(4),z.neighbor_fold,marker='o',ls=ls,color=color,label=f'{dataset} {lib}')
  z=q[q.dataset==dataset].drop_duplicates('layer').set_index('layer').loc[order];axes[1].plot(range(4),z.silhouette,marker='o',ls=ls,label=f'{dataset} silhouette');axes[1].plot(range(4),z.stability_ari,marker='s',ls=ls,label=f'{dataset} ARI')
  z=dep[dep.dataset==dataset].set_index('layer').loc[order];axes[2].plot(range(4),z.within_minus_across,marker='o',ls=ls,label=dataset)
 axes[0].axhline(1,color='black',lw=1,ls=':');axes[0].set(title='Functional neighborhoods (k=25)',ylabel='Fold over random')
 axes[1].set(title='Within-sample gene clustering',ylabel='Score');axes[2].axhline(0,color='black',lw=1,ls=':');axes[2].set(title='Same-gene context dependence',ylabel='Within − across context cosine')
 for ax in axes:ax.set_xticks(range(4),['L0','L1','L6','L12']);ax.set_xlabel('Layer');ax.legend(fontsize=7)
 for ext in ['png','pdf']:fig.savefig(out/f'gtex_tcga_matched_summary.{ext}',dpi=250,bbox_inches='tight')
 plt.close(fig)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda:0');ap.add_argument('--null-reps',type=int,default=10);a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True);log=Log(OUT/'run.log');start=time.time();genes=pd.read_csv(VOCAB).sort_values('token_id').gene_symbol.str.upper().tolist();manifest=cohorts();manifest.to_csv(OUT/'cohort_manifest.csv',index=False);libs={n:gmt(p,genes) for n,p in GMTS.items()};bits={n:bits_for(genes,t) for n,t in libs.items()};mats={n:ann_matrix(genes,t) for n,t in libs.items()};rng=np.random.default_rng(77);null={}
 for lib,b in bits.items():
  ann=np.array([i for i,x in enumerate(b) if x]);
  for k in KS:null[(lib,k)]=np.array([overlap(b,rng.choice(ann,size=(len(genes),k),replace=True),k)[0] for _ in range(a.null_reps)])
 dev=torch.device(a.device if torch.cuda.is_available() else 'cpu');model=load_model(dev);ids=torch.arange(len(genes),device=dev);arrays={'GTEx':np.load(GTEX_X,mmap_mode='r'),'TCGA':np.load(TCGA_X,mmap_mode='r')};metric=[];clrows=[];sums={};unit_sums={};counts={}
 for dataset in ['GTEx','TCGA']:
  contexts=GTEX_TISSUES if dataset=='GTEx' else TCGA_COHORTS
  for l in LAYERS:sums[(dataset,l)]=np.zeros((len(genes),512),np.float32)
  for l in LAYERS:
   for c in contexts:unit_sums[(dataset,l,c)]=np.zeros((len(genes),512),np.float32);counts[(dataset,l,c)]=0
  sub=manifest[manifest.dataset.eq(dataset)].reset_index(drop=True)
  for si,row in sub.iterrows():
   v=torch.as_tensor(np.array(arrays[dataset][int(row.matrix_row)],copy=True)[None],device=dev);states={'L0_static':model.gene_embedding(ids)}
   with torch.no_grad(),torch.autocast(device_type=dev.type,dtype=torch.float16,enabled=dev.type=='cuda'):
    h=model.gene_embedding(ids).unsqueeze(0)+model.ree(v)
    for li,layer in enumerate(model.layers,1):
     h=layer(h)
     for key,num in LAYERS.items():
      if num==li:states[key]=h[0]
   for layer,z0 in states.items():
    z=z0.float();sums[(dataset,layer)]+=z.cpu().numpy();u=torch.nn.functional.normalize(z,dim=1);unit_sums[(dataset,layer,row.context)]+=u.cpu().numpy();counts[(dataset,layer,row.context)]+=1
    # L0 is identical: compute expensive endpoints once per dataset.
    if layer=='L0_static' and si>0:continue
    for geom in GEOMS:
     z=z0.float() if geom=='raw_cosine' else z0.float()-z0.float().mean(0,keepdim=True);nbr=neighbors(z)
     for lib,b in bits.items():
      for k in KS:
       rate,per,den=overlap(b,nbr,k);nul=null[(lib,k)];metric.append({'dataset':dataset,'context':row.context,'sample_id':row.sample_id,'layer':layer,'geometry':geom,'library':lib,'k':k,'neighbor_fold':rate/nul.mean(),'neighbor_rate':rate,'null_mean':nul.mean()})
    z=z0.float()-z0.float().mean(0,keepdim=True);cluster_labels,sil,ari=cluster_geometry(z)
    for lib,mat in mats.items():
     size=np.asarray(mat.sum(0)).ravel();ann=np.asarray(mat.sum(1)).ravel()>0;sig=cluster_enrichment(cluster_labels,mat,size,ann.sum());clrows.append({'dataset':dataset,'context':row.context,'sample_id':row.sample_id,'layer':layer,'library':lib,'silhouette':sil,'stability_ari':ari,'significant_terms':sig})
   if (si+1)%20==0:log(f'{dataset} {si+1}/{len(sub)} elapsed={(time.time()-start)/60:.1f}m')
 metric=pd.DataFrame(metric);metric.to_csv(OUT/'neighbor_enrichment_per_sample.csv',index=False);clusters=pd.DataFrame(clrows);clusters.to_csv(OUT/'contextual_clustering_per_sample.csv',index=False);means={(d,l):sums[(d,l)]/N/10 for d in ['GTEx','TCGA'] for l in LAYERS};visualize(means,genes,OUT)
 dep=[]
 for d in ['GTEx','TCGA']:
  contexts=GTEX_TISSUES if d=='GTEx' else TCGA_COHORTS
  for l in LAYERS:
   within=[];across=[]
   for c in contexts:
    s=unit_sums[(d,l,c)].astype(np.float64);n=counts[(d,l,c)];within.append((np.square(s).sum(1)-n)/(n*(n-1)))
   for i,c in enumerate(contexts):
    for e in contexts[i+1:]:across.append((unit_sums[(d,l,c)].astype(np.float64)*unit_sums[(d,l,e)]).sum(1)/(counts[(d,l,c)]*counts[(d,l,e)]))
   wi=np.mean(within,0);be=np.mean(across,0);pd.DataFrame({'gene':genes,'dataset':d,'layer':l,'within_context_cosine':wi,'across_context_cosine':be,'within_minus_across':wi-be}).to_csv(OUT/f'context_dependence_{d}_{l}.csv',index=False);dep.append({'dataset':d,'layer':l,'within_context_cosine':wi.mean(),'across_context_cosine':be.mean(),'within_minus_across':(wi-be).mean(),'genes_positive_difference':(wi>be).mean()})
 dep=pd.DataFrame(dep);dep.to_csv(OUT/'context_dependence_summary.csv',index=False);ns=metric.groupby(['dataset','layer','geometry','library','k'],as_index=False).agg(neighbor_fold=('neighbor_fold','mean'),neighbor_fold_sd=('neighbor_fold','std'),samples=('sample_id','nunique'));cs=clusters.groupby(['dataset','layer','library'],as_index=False).agg(silhouette=('silhouette','mean'),stability_ari=('stability_ari','mean'),significant_terms=('significant_terms','mean'),samples=('sample_id','nunique'));summary=ns.merge(cs,on=['dataset','layer','library'],how='left');summary=summary.merge(dep,on=['dataset','layer']);summary.to_csv(OUT/'gtex_tcga_summary.csv',index=False);plot_summary(summary,dep,OUT)
 prov={'status':'complete','profile':'exploratory','samples_per_context':N,'contexts_per_dataset':10,'total_samples':len(manifest),'preprocessing':{'GTEx':'raw counts -> GENCODE v49 exon-length TPM -> natural log1p, canonical 15165 order','TCGA':'source counts -> GENCODE v49 exon-length TPM -> natural log1p, canonical 15165 order'},'shared_model':'frozen r7hnr92k','layers':LAYERS,'neighbor_parameters':{'k':KS,'geometries':GEOMS,'null_reps':a.null_reps},'clustering':'within-sample centered-cosine spherical k-means k=10, seeds 42/43, 8 iterations; each assignment reused for GO and KEGG; L0 computed once because sample invariant','context_dependence':'genome-wide exact mean same-gene cosine within versus across contexts from unit-vector sufficient statistics','visualization':'fixed 2000-gene sample; joint PCA50 and joint t-SNE/UMAP per layer; identical parameters and seed','limitations':['GTEx normal tissues and TCGA tumor cohorts are not paired biological conditions','token-heavy neighbor/clustering endpoints are exploratory','L0 repeated samples are collapsed because identical'],'elapsed_seconds':time.time()-start};(OUT/'provenance.json').write_text(json.dumps(prov,indent=2)+'\n');log(f'Complete elapsed={(time.time()-start)/60:.1f}m')
if __name__=='__main__':main()
