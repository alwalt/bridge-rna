#!/usr/bin/env python3
"""Select improved sample readouts from frozen cached BridgeRNA layer summaries."""
from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parents[1];PREV=HERE/'results/task4_pca_vs_bridgerna_generalization';COL=HERE/'results/task4_information_collapse';CACHE=HERE/'work/task4_information_collapse';OUT=HERE/'results/task4_frozen_readout_selection';FIG=OUT/'figures';SEED=20260906
LAYERS=[4,5,6,7,8,9,12];READOUTS=['mean','std','mean_plus_std','mean_plus_variance','mean_plus_max','expression_weighted_mean','expression_weighted_mean_plus_std','mean_plus_std_plus_max']
def say(x):print(f'[{time.strftime("%F %T")}] {x}',flush=True)
def base(layer,name):return np.load(CACHE/f'layer_{layer}__{name}.float32.npy',mmap_mode='r')
def readout(layer,name):
 m=base(layer,'mean');s=base(layer,'std')
 if name=='mean':return m
 if name=='std':return s
 if name=='mean_plus_std':return np.concatenate([m,s],1)
 if name=='mean_plus_variance':return np.concatenate([m,np.asarray(s)**2],1)
 if name=='mean_plus_max':return np.concatenate([m,base(layer,'max')],1)
 if name=='expression_weighted_mean':return base(layer,'expression_weighted_mean')
 if name=='expression_weighted_mean_plus_std':return np.concatenate([base(layer,'expression_weighted_mean'),s],1)
 if name=='mean_plus_std_plus_max':return np.concatenate([m,s,base(layer,'max')],1)
 raise KeyError(name)
def metric(y,p):return {'accuracy':accuracy_score(y,p),'balanced_accuracy':balanced_accuracy_score(y,p),'macro_f1':f1_score(y,p,average='macro',zero_division=0)}
def fit(A,y,B,yt,alpha):
 sc=StandardScaler().fit(A);mod=RidgeClassifier(alpha=alpha,solver='lsqr',tol=1e-3,class_weight='balanced').fit(sc.transform(A),y);return metric(yt,mod.predict(sc.transform(B)))
def split_indices(m,fold):
 s=pd.read_csv(PREV/'exact_splits.csv');q=s[(s.scheme=='group5')&(s.fold==fold)];ix=pd.Series(np.arange(len(m)),index=m.matrix_row);return ix.loc[q[q.partition=='train'].matrix_row].to_numpy(),ix.loc[q[q.partition=='test'].matrix_row].to_numpy()
def effective(A):
 A=np.asarray(A,float);A-=A.mean(0);e=np.linalg.eigvalsh(A.T@A/(len(A)-1));e=np.maximum(e,0);r=e/e.sum();return r[-2:].sum(),e.sum()**2/(e@e),float(np.exp(-np.sum(r[r>0]*np.log(r[r>0]))))
def fixed_grid():
 m=pd.read_parquet(PREV/'sample_manifest.parquet').sort_values('matrix_row').reset_index(drop=True);y=m.tissue.astype(str).to_numpy();g=m.gse.astype(str).to_numpy();rows=[];geom=[];started=time.time()
 for layer in LAYERS:
  for name in READOUTS:
   A=readout(layer,name);pc12,pr,er=effective(A);nn=NearestNeighbors(n_neighbors=11,metric='cosine',n_jobs=-1).fit(A).kneighbors(return_distance=False)[:,:10];tp=np.mean([np.mean(y[nn[i]]==y[i]) for i in range(len(y))]);sp=np.mean([np.mean(g[nn[i]]==g[i]) for i in range(len(y))]);geom.append({'layer':layer,'readout':name,'dimensions':A.shape[1],'PC1_2':pc12,'participation_ratio':pr,'entropy_effective_rank':er,'tissue_10nn_purity':tp,'study_10nn_purity':sp})
   for fold in range(5):
    tr,te=split_indices(m,fold);ia,iv=next(GroupShuffleSplit(1,test_size=.2,random_state=SEED+fold).split(A[tr],y[tr],g[tr]));cand=[]
    for alpha in [.1,1.,10.]:cand.append((fit(A[tr][ia],y[tr][ia],A[tr][iv],y[tr][iv],alpha)['macro_f1'],alpha))
    alpha=max(cand)[1];rows.append({'layer':layer,'readout':name,'fold':fold,'dimensions':A.shape[1],'alpha':alpha,**fit(A[[ *tr ]],y[tr],A[te],y[te],alpha)})
   say(f'fixed layer={layer} readout={name} elapsed={(time.time()-started)/60:.1f}m')
 f=pd.DataFrame(rows);gdf=pd.DataFrame(geom);f.to_csv(OUT/'fixed_readout_fold_metrics.csv',index=False);gdf.to_csv(OUT/'fixed_readout_geometry.csv',index=False);s=f.groupby(['layer','readout','dimensions']).agg(macro_f1=('macro_f1','mean'),macro_f1_sd=('macro_f1','std'),balanced_accuracy=('balanced_accuracy','mean'),accuracy=('accuracy','mean')).reset_index().merge(gdf,on=['layer','readout','dimensions']);s.to_csv(OUT/'fixed_readout_summary.csv',index=False);return m,y,g,s
def projections(m,y,g,s):
 # Unsupervised PCA compression is fitted inside every outer training fold.
 candidates=s.sort_values(['macro_f1','tissue_10nn_purity'],ascending=False).head(3)[['layer','readout']].itertuples(index=False);rows=[]
 for layer,name in candidates:
  A=readout(layer,name)
  for fold in range(5):
   tr,te=split_indices(m,fold);ia,iv=next(GroupShuffleSplit(1,test_size=.2,random_state=SEED+fold).split(A[tr],y[tr],g[tr]))
   for dim in [128,256,512]:
    dim=min(dim,len(tr)-1,A.shape[1]);p=PCA(dim,svd_solver='randomized',random_state=SEED+fold).fit(A[tr][ia]);U=p.transform(A[tr][ia]);V=p.transform(A[tr][iv]);cand=[(fit(U,y[tr][ia],V,y[tr][iv],a)['macro_f1'],a) for a in [.1,1.,10.]];alpha=max(cand)[1];p=PCA(dim,svd_solver='randomized',random_state=SEED+fold).fit(A[tr]);z=fit(p.transform(A[tr]),y[tr],p.transform(A[te]),y[te],alpha);rows.append({'source_layer':layer,'source_readout':name,'projection':'unsupervised_pca','target_dimensions':dim,'fold':fold,'alpha':alpha,**z})
  say(f'projection layer={layer} readout={name}')
 out=pd.DataFrame(rows);out.to_csv(OUT/'projection_fold_metrics.csv',index=False);out.groupby(['source_layer','source_readout','projection','target_dimensions']).agg(macro_f1=('macro_f1','mean'),macro_f1_sd=('macro_f1','std'),balanced_accuracy=('balanced_accuracy','mean'),accuracy=('accuracy','mean')).reset_index().to_csv(OUT/'projection_summary.csv',index=False)
def main():
 OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(exist_ok=True);m,y,g,s=fixed_grid();projections(m,y,g,s);best=s.sort_values(['macro_f1','tissue_10nn_purity'],ascending=False);best.to_csv(OUT/'final_readout_ranking.csv',index=False);(OUT/'provenance.json').write_text(json.dumps({'backbone_frozen':True,'backbone_inference_recomputed':False,'layers':LAYERS,'readouts':READOUTS,'folds':'exact prior five GSE-disjoint folds','selection':'training-only alpha; final table exploratory multi-metric ranking','supervised_projection':'not run in fixed-readout phase; would be task-specific rather than a universal frozen embedding'},indent=2)+'\n');say('complete')
if __name__=='__main__':main()
