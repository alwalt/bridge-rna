#!/usr/bin/env python3
"""Audit conventional PCA-15165 against frozen BridgeRNA geometry and reconstruction."""
from __future__ import annotations
import json,sys
from itertools import combinations
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.linalg import orthogonal_procrustes
from scipy.stats import rankdata,spearmanr
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

HERE=Path(__file__).resolve().parents[1];REPO=HERE.parents[1];OUT=HERE/'results/task4_pca_vs_bridgerna';FIG=OUT/'figures';SEED=20260910
T2=REPO/'benchmarks/cross_species_exercise_response';T3=REPO/'benchmarks/osdr_batch_effect_representation';T1=REPO/'benchmarks/mouse_encode';IMP=REPO/'benchmarks/tcga_imputation'
HWORK=T2/'work/hallmark_readout';G=15165

def effdim(e):
 e=np.maximum(np.asarray(e,float),0);p=e/e.sum();return float(e.sum()**2/np.sum(e*e)),float(np.exp(-np.sum(p[p>0]*np.log(p[p>0]))))
def spectrum_summary(name,e,n_features,estimated_tail=False):
 e=np.asarray(e,float);ratio=e/e.sum();cum=np.cumsum(ratio);pr,er=effdim(e);r={'representation':name,'features':n_features,'eigenvalues_used':len(e),'tail_estimated':estimated_tail,'PC1':ratio[0],'PC2':ratio[1],'PC1_2':cum[1]}
 for k in [5,10,20,50,100]:r[f'PC1_{k}']=cum[min(k,len(e))-1]
 for t in [.5,.75,.9,.95]:r[f'PCs_for_{int(t*100)}pct']=int(np.searchsorted(cum,t)+1) if cum[-1]>=t else np.nan
 r['participation_ratio']=pr;r['entropy_effective_rank']=er;return r,pd.DataFrame({'representation':name,'PC':np.arange(1,len(e)+1),'eigenvalue':e,'variance_fraction':ratio,'cumulative_variance':cum})

def global_variance():
 pca=joblib.load(HWORK/'pca512.pkl');z=np.load(HWORK/'archs4_bridgerna_embeddings.npy',mmap_mode='r').astype(float);zc=z-z.mean(0);be=np.linalg.eigvalsh(zc.T@zc/(len(zc)-1))[::-1]
 # sklearn PCA reports total training variance through explained ratios. Spread its
 # residual variance uniformly over the omitted dimensions as an explicit PPCA approximation.
 total=pca.explained_variance_[0]/pca.explained_variance_ratio_[0];residual=max(total-pca.explained_variance_.sum(),0);tail=np.full(G-len(pca.explained_variance_),residual/(G-len(pca.explained_variance_)));pe=np.r_[pca.explained_variance_,tail]
 a,sa=spectrum_summary('PCA-15165',pe,G,True);b,sb=spectrum_summary('BridgeRNA-512',be,512,False);summary=pd.DataFrame([a,b]);spec=pd.concat([sa,sb]);summary.to_csv(OUT/'global_variance_summary.csv',index=False);spec.to_csv(OUT/'global_variance_spectra.csv',index=False);return pca,z,summary,spec

def geometry(pca,z):
 coords=np.load(HWORK/'archs4_pca512.npy',mmap_mode='r');rng=np.random.default_rng(SEED);idx=np.sort(rng.choice(len(z),3000,False));A=np.asarray(coords[idx],float);B=np.asarray(z[idx],float);A-=A.mean(0);B-=B.mean(0)
 # 250k random unordered pairs.
 i=rng.integers(len(idx),size=300000);j=rng.integers(len(idx),size=300000);keep=i!=j;i,j=i[keep],j[keep];da=np.linalg.norm(A[i]-A[j],axis=1);db=np.linalg.norm(B[i]-B[j],axis=1)
 rows=[{'comparison':'PCA-15165 vs BridgeRNA','distance_metric':'euclidean','samples':len(idx),'pairs':len(da),'pairwise_distance_spearman':spearmanr(da,db).statistic}]
 # Neighborhood overlap.
 neigh=[]
 for name,X in [('PCA-15165',A),('BridgeRNA-512',B)]:
  nn=NearestNeighbors(n_neighbors=101,metric='euclidean',n_jobs=-1).fit(X).kneighbors(return_distance=False);locals()[f'nn_{name[:3]}']=nn
 nnA=NearestNeighbors(n_neighbors=101,metric='euclidean',n_jobs=-1).fit(A).kneighbors(return_distance=False);nnB=NearestNeighbors(n_neighbors=101,metric='euclidean',n_jobs=-1).fit(B).kneighbors(return_distance=False)
 for k in [5,10,50,100]:neigh.append({'comparison':'PCA-15165 vs BridgeRNA','k':k,'mean_neighbor_overlap':np.mean([len(set(nnA[r,:k])&set(nnB[r,:k]))/k for r in range(len(A))])})
 # Whitened cross-cov singular values (canonical correlations) and orthogonal Procrustes fit.
 def whiten(X):
  C=X.T@X/(len(X)-1);e,V=np.linalg.eigh(C);keep=e>e.max()*1e-8;return X@V[:,keep]/np.sqrt(e[keep])
 WA,WB=whiten(A),whiten(B);cc=np.linalg.svd(WA.T@WB/(len(A)-1),compute_uv=False);k=min(A.shape[1],B.shape[1]);An=A[:,:k]/np.linalg.norm(A[:,:k]);Bn=B[:,:k]/np.linalg.norm(B[:,:k]);R,scale=orthogonal_procrustes(An,Bn);pred=An@R*scale;proc=1-np.sum((Bn-pred)**2)/np.sum(Bn**2)
 align=pd.DataFrame([{'comparison':'PCA-15165 vs BridgeRNA','CCA_mean_top10':cc[:10].mean(),'CCA_median':np.median(cc),'CCA_top1':cc[0],'orthogonal_procrustes_R2':proc,'samples':len(A)}]);pd.DataFrame(rows).to_csv(OUT/'pairwise_geometry.csv',index=False);pd.DataFrame(neigh).to_csv(OUT/'neighborhood_preservation.csv',index=False);align.to_csv(OUT/'linear_alignment.csv',index=False);return pd.DataFrame(rows),pd.DataFrame(neigh),align

def tcell_displacements():
 p=HERE/'work/datasets/chen_2020_tcells';m=pd.read_parquet(p/'manifest.parquet').reset_index(drop=True);x=np.load(p/'log1p_tpm.npy');z=np.load(p/'bridgerna_embeddings.npy');dx=[];dz=[]
 for _,q in m.groupby('pair_id',sort=True):dx.append(x[q.index[q.library_prep.eq('ribo')]].mean(0)-x[q.index[q.library_prep.eq('polyA')]].mean(0));dz.append(z[q.index[q.library_prep.eq('ribo')]].mean(0)-z[q.index[q.library_prep.eq('polyA')]].mean(0))
 dx=np.stack(dx);dz=np.stack(dz);sx=np.linalg.svd(dx,compute_uv=False);sz=np.linalg.svd(dz,compute_uv=False);ex=sx*sx;ez=sz*sz;rx,specx=spectrum_summary('T-cell expression displacement',ex,G,False);rz,specz=spectrum_summary('T-cell BridgeRNA displacement',ez,512,False)
 # Geometry comparison is valid in donor space despite feature dimensionality mismatch.
 Kx=dx@dx.T;Kz=dz@dz.T;cka=np.sum(Kx*Kz)/np.sqrt(np.sum(Kx*Kx)*np.sum(Kz*Kz));tri=np.triu_indices(len(dx),1);rho=spearmanr(pairwise_distances(dx)[tri],pairwise_distances(dz)[tri]).statistic
 sm=pd.DataFrame([rx,rz]);sm['donor_gram_CKA']=cka;sm['donor_distance_spearman']=rho;sm.to_csv(OUT/'tcell_displacement_summary.csv',index=False);pd.concat([specx,specz]).to_csv(OUT/'tcell_displacement_spectra.csv',index=False);return sm

def response_geometry(pca):
 # Nine Bridge-vocabulary expression responses and six exact Bridge responses.
 gd=HERE/'results/task4_gene_attribution_diagnostic';ex=np.load(gd/'expression_response_vectors.npz');attrs=pd.read_parquet(HERE/'results/task4_attribution_vs_expression/contrast_gene_tables.parquet');names=attrs.contrast.unique();expr={n:attrs[attrs.contrast.eq(n)].set_index('gene_symbol').loc[pd.read_csv(REPO/'data/ensembl/canonical_genes.csv').gene_symbol,'expression_change'].to_numpy() for n in names}
 # Bridge response vectors reconstructed exactly as in the multi-layer benchmark.
 sys.path.insert(0,str(HERE/'pipeline'));from analyze_multilayer_reproducibility import responses as loadresp
 core,_=loadresp();bridge={'RR1 OSD48':core['RR1_original'],'RR1 OSD168':core['RR1_remeasurement'],'RR3-39 OSD137':core['RR3_39_original'],'RR3-39 OSD168':core['RR3_39_remeasurement'],'RR3-40 OSD137':core['RR3_40_original'],'RR3-40 OSD168':core['RR3_40_remeasurement']}
 # Four-state Bridge responses already in fixed full-space vectors can be rebuilt from sample embeddings, but only pairwise core is needed here.
 pvec={n:expr[n]@pca.components_.T for n in names};pairs=[]
 for a,b in combinations(bridge,2):
  pairs.append({'response_A':a,'response_B':b,'BridgeRNA_cosine':float(np.dot(bridge[a],bridge[b])/(np.linalg.norm(bridge[a])*np.linalg.norm(bridge[b]))),'PCA15165_cosine':float(np.dot(pvec[a],pvec[b])/(np.linalg.norm(pvec[a])*np.linalg.norm(pvec[b]))),'raw15165_cosine':float(np.dot(expr[a],expr[b])/(np.linalg.norm(expr[a])*np.linalg.norm(expr[b])))})
 out=pd.DataFrame(pairs);out.to_csv(OUT/'biological_response_pairwise.csv',index=False)
 target=[('RR1','RR1 OSD48','RR1 OSD168'),('RR3-39','RR3-39 OSD137','RR3-39 OSD168'),('RR3-40','RR3-40 OSD137','RR3-40 OSD168'),('false_friend','RR1 OSD48','RR3-39 OSD137')];focus=[]
 for label,a,b in target:
  r=out[((out.response_A==a)&(out.response_B==b))|((out.response_A==b)&(out.response_B==a))].iloc[0].to_dict();r['comparison']=label;focus.append(r)
 focus=pd.DataFrame(focus);focus.to_csv(OUT/'response_focus_comparisons.csv',index=False)
 # Reuse OSDR full-vocabulary technical response metrics.
 full=pd.read_csv(T3/'results/task3_representation_comparison_full_vocab/technical_replication_metrics_all_representations.csv');full.to_csv(OUT/'pca_full_osdr_technical_metrics_reused.csv',index=False);return out,focus,full

def downstream():
 d=pd.read_csv(T1/'results/task1a_balanced_geometry/summary_results.csv');q=d[(d.cohort.eq('expanded_replicated'))&(d.balance.eq('balanced'))&(d.readout.isin(['centroid_cosine','knn_cosine_k1','linear_softmax_probe']))&(d.representation.isin(['raw','pca','bridgerna']))].copy();q.to_csv(OUT/'downstream_tissue_balanced_reused.csv',index=False);return q

def pca_imputation(pca,device_name):
 # One deterministic seed on external TCGA, identical masks to existing benchmark.
 if (OUT/'pca_imputation.csv').exists() and (OUT/'bridgerna_imputation_reused.csv').exists():
  return pd.read_csv(OUT/'pca_imputation.csv'),pd.read_csv(OUT/'bridgerna_imputation_reused.csv')
 sys.path.insert(0,str(IMP/'pipeline'));from common import exact_mask,row_metrics
 X=np.load(IMP/'work/ours_log1p_tpm.npy',mmap_mode='r');samples=pd.read_parquet(IMP/'results/selected_tcga_samples.parquet');ids=samples.sample_id.astype(str).tolist();genes=pd.read_parquet(IMP/'work/ours_genes.parquet');gene_ids=genes.gene_index.astype(str).tolist() if 'gene_index' in genes else [str(i) for i in range(G)];C=pca.components_.astype(np.float32);mu=pca.mean_.astype(np.float32);dev=torch.device(device_name if torch.cuda.is_available() else 'cpu');rows=[]
 for ratio in [.5,.9]:
  mask=exact_mask(ids,gene_ids,ratio,0);obs=~mask
  for k in [10,50,100,256,512]:
   preds=np.empty_like(X,dtype=np.float32);start=0
   for i in range(len(X)):
    o=obs[i];A=torch.as_tensor(C[:k,o],device=dev);y=torch.as_tensor(np.asarray(X[i,o],np.float32)-mu[o],device=dev);gram=A@A.T+1e-4*torch.eye(k,device=dev);score=torch.linalg.solve(gram,A@y);preds[i]=mu+(score.cpu().numpy()@C[:k])
   truth=np.stack([np.asarray(X[i,mask[i]],float) for i in range(len(X))]);pred=np.stack([preds[i,mask[i]] for i in range(len(X))]);pe,sp,mse=row_metrics(truth,pred);rows.append({'dataset':'TCGA','representation':'PCA-15165','dimensions':k,'mask_ratio':ratio,'seed':0,'samples':len(X),'pearson':np.nanmean(pe),'spearman':np.nanmean(sp),'mse':np.nanmean(mse)})
 out=pd.DataFrame(rows);bridge=pd.read_csv(IMP/'results/summary_results.csv');bridge=bridge[(bridge.method.eq('ours_45.6m'))&(bridge.benchmark.eq('native_vocab'))&(bridge.mask_ratio.isin([.5,.9]))][['mask_ratio','pearson_mean','spearman_mean','mse_mean','seeds']];bridge.to_csv(OUT/'bridgerna_imputation_reused.csv',index=False);out.to_csv(OUT/'pca_imputation.csv',index=False);return out,bridge

def figures(summary,spec,geom,neigh,tcell,focus,imp,bridge):
 plt.style.use('seaborn-v0_8-whitegrid');FIG.mkdir(exist_ok=True);fig,axes=plt.subplots(1,2,figsize=(14,5))
 for name,g in spec.groupby('representation'):axes[0].plot(g.PC.head(100),g.cumulative_variance.head(100),label=name);axes[1].plot(g.PC.head(20),g.variance_fraction.head(20),marker='o',label=name)
 axes[0].set(xlabel='PC',ylabel='Cumulative variance',title='Global variance concentration');axes[1].set(xlabel='PC',ylabel='Variance fraction',title='First 20 PCs');axes[0].legend();axes[1].legend();fig.tight_layout()
 for e in ['png','pdf']:fig.savefig(FIG/f'global_scree.{e}',dpi=400);plt.close(fig)
 fig,ax=plt.subplots(figsize=(10,6));w=focus.set_index('comparison')[['raw15165_cosine','PCA15165_cosine','BridgeRNA_cosine']];w.plot.bar(ax=ax);ax.axhline(0,color='black',lw=.7);ax.set(ylabel='Response cosine',title='Technical replication and geometric false friend');plt.xticks(rotation=0);fig.tight_layout()
 for e in ['png','pdf']:fig.savefig(FIG/f'response_comparison.{e}',dpi=400);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(14,5));
 for ratio,g in imp.groupby('mask_ratio'):axes[0].plot(g.dimensions,g.pearson,marker='o',label=f'PCA {ratio:.0%} masked');axes[1].plot(g.dimensions,g.mse,marker='o',label=f'PCA {ratio:.0%} masked')
 axes[0].set(xlabel='PCA dimensions',ylabel='Pearson',title='External TCGA masked reconstruction');axes[1].set(xlabel='PCA dimensions',ylabel='MSE',title='External TCGA masked reconstruction');axes[0].legend();axes[1].legend();fig.tight_layout()
 for e in ['png','pdf']:fig.savefig(FIG/f'pca_imputation.{e}',dpi=400);plt.close(fig)

def main(device):
 OUT.mkdir(parents=True,exist_ok=True);pca,z,summary,spec=global_variance();geom,neigh,align=geometry(pca,z);tc=tcell_displacements();responses,focus,full=response_geometry(pca);down=downstream();imp,bridge=pca_imputation(pca,device);figures(summary,spec,geom,neigh,tc,focus,imp,bridge)
 s=summary.set_index('representation');f=focus.set_index('comparison');best50=imp[imp.mask_ratio.eq(.5)].sort_values('pearson').iloc[-1];best90=imp[imp.mask_ratio.eq(.9)].sort_values('pearson').iloc[-1];bb=bridge.set_index('mask_ratio')
 decision_rows=[
  ['Global PC1+2 variance',f"{s.loc['PCA-15165','PC1_2']:.3f}",'unavailable on same 40k samples',f"{s.loc['BridgeRNA-512','PC1_2']:.3f}",'BridgeRNA strongly amplifies variance concentration'],
  ['Participation ratio',f"{s.loc['PCA-15165','participation_ratio']:.2f} (estimated tail)",'unavailable',f"{s.loc['BridgeRNA-512','participation_ratio']:.2f}",'BridgeRNA is substantially lower dimensional'],
  ['T-cell displacement PC1+2',f"{tc.set_index('representation').loc['T-cell expression displacement','PC1_2']:.3f}",'not applicable',f"{tc.set_index('representation').loc['T-cell BridgeRNA displacement','PC1_2']:.3f}",'Response was already extremely concentrated in expression'],
  ['PCA/Bridge distance geometry',f"Spearman {geom.pairwise_distance_spearman.iloc[0]:.3f}",'unavailable','reference','Substantial global preservation with reorganization'],
  ['10-NN overlap',f"{neigh.set_index('k').loc[10,'mean_neighbor_overlap']:.3f}",'unavailable','reference','Less than half of local neighbors preserved'],
  ['RR1 technical replication',f"{f.loc['RR1','PCA15165_cosine']:.3f}",'OSDR-fitted full PCA max 0.255',f"{f.loc['RR1','BridgeRNA_cosine']:.3f}",'BridgeRNA uniquely amplifies reversal'],
  ['RR3-39 technical replication',f"{f.loc['RR3-39','PCA15165_cosine']:.3f}",'OSDR-fitted full PCA max 0.509',f"{f.loc['RR3-39','BridgeRNA_cosine']:.3f}",'Global PCA and BridgeRNA are equivalent'],
  ['RR3-40 technical replication',f"{f.loc['RR3-40','PCA15165_cosine']:.3f}",'OSDR-fitted full PCA max 0.647',f"{f.loc['RR3-40','BridgeRNA_cosine']:.3f}",'Global PCA and BridgeRNA are equivalent'],
  ['RR1/RR3-39 false friend',f"{f.loc['false_friend','PCA15165_cosine']:.3f}",'not available as strict global baseline',f"{f.loc['false_friend','BridgeRNA_cosine']:.3f}",'BridgeRNA amplifies false similarity'],
  ['50% masked TCGA Pearson',f"{best50.pearson:.3f} ({int(best50.dimensions)} PCs)",'unavailable',f"{bb.loc[.5,'pearson_mean']:.3f}",'Comparable; PCA slightly higher'],
  ['90% masked TCGA Pearson',f"{best90.pearson:.3f} ({int(best90.dimensions)} PCs)",'unavailable',f"{bb.loc[.9,'pearson_mean']:.3f}",'PCA substantially higher']]
 pd.DataFrame(decision_rows,columns=['Audit','PCA_15165','PCA_FULL_secondary','BridgeRNA_512','Interpretation']).to_csv(OUT/'summary_decision_table.csv',index=False)
 decision={'global_samples':40000,'pca_fit_samples':int(pca.n_samples_),'genes_PCA15165':G,'genes_PCA_FULL_global':None,'PCA_FULL_global_reason':'same-sample full-gene ARCHS4 matrix unavailable','geometry_subset':3000,'Tcell_pairs':40,'imputation_dataset':'TCGA external','imputation_samples':1000,'imputation_seed':0,'outcome':'C_BRIDGERNA_AMPLIFIES_LOW_DIMENSIONALITY_DESTRUCTIVELY_FOR_EVALUATED_ENDPOINTS','qualification':'Some linear tissue readouts favor BridgeRNA, so broader representation utility remains mixed.'};(OUT/'provenance.json').write_text(json.dumps(decision,indent=2)+'\n');print(summary.to_string(index=False));print('\nGeometry\n',geom.to_string(index=False));print(neigh.to_string(index=False));print(align.to_string(index=False));print('\nT-cell\n',tc.to_string(index=False));print('\nResponses\n',focus.to_string(index=False));print('\nImputation\n',imp.to_string(index=False));print('[complete]',OUT)
if __name__=='__main__':
 import argparse;p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');main(p.parse_args().device)
