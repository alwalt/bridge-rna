#!/usr/bin/env python3
"""Localize dimensionality and tissue-information changes through frozen BridgeRNA."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import joblib,numpy as np,pandas as pd,torch
from numpy.lib.format import open_memmap
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parents[1];REPO=HERE.parents[1]
PREV=HERE/'results/task4_pca_vs_bridgerna_generalization';OUT=HERE/'results/task4_information_collapse';CACHE=HERE/'work/task4_information_collapse'
HWORK=REPO/'benchmarks/cross_species_exercise_response/work/hallmark_readout';G=15165;SEED=20260906
STAGES=['input_embedding']+[f'layer_{i}' for i in range(1,13)]
POOLS={'mean':512,'median':512,'max':512,'std':512,'expression_weighted_mean':512,'mean_plus_std':1024}

def say(x):print(f"[{time.strftime('%F %T')}] {x}",flush=True)
def architecture():
 cfg=json.loads((REPO/'model/r7hnr92k/config.json').read_text());d={'input_shape':['batch',15165],'gene_embedding':[15165,512],'rotary_expression_embedding':[15165,512],'species_embedding_enabled':cfg['include_species_embedding'],'layers':12,'hidden_dim':512,'attention_heads':8,'head_dim':64,'ffn_dim':2048,'block':'pre-LayerNorm self-attention + residual; pre-LayerNorm GELU FFN + residual','contextual_shape':['batch',15165,512],'pooling':'unweighted mean over 15,165 genes','post_pool_projection':False,'sample_embedding':['batch',512],'mask_token':-10,'standard_unmasked_inference':'no masking','normalization':'log1p_tpm'};(OUT/'architecture.json').write_text(json.dumps(d,indent=2)+'\n')

def load_data():
 m=pd.read_parquet(PREV/'sample_manifest.parquet').sort_values('matrix_row').reset_index(drop=True);Xall=np.memmap(HWORK/'archs4_log1p_tpm.float32.mmap',mode='r',dtype='float32',shape=(40000,G));return m,np.asarray(Xall[m.matrix_row.to_numpy(int)])

def paths(stage,pool):return CACHE/f'{stage}__{pool}.float32.npy'
def pool(hidden,x):
 h=hidden.float();mean=h.mean(1);std=h.std(1,unbiased=False);w=torch.clamp(x.float(),min=0);w=w/(w.sum(1,keepdim=True)+1e-8)
 return {'mean':mean,'median':h.median(1).values,'max':h.max(1).values,'std':std,'expression_weighted_mean':(h*w[:,:,None]).sum(1),'mean_plus_std':torch.cat([mean,std],1)}

def extract(device,batch_size):
 CACHE.mkdir(parents=True,exist_ok=True);m,X=load_data();complete=OUT/'extraction_complete.json'
 if complete.exists() and all(paths(s,p).exists() for s in STAGES for p in POOLS):say('reusing complete layer cache');return
 sys.path.insert(0,str(REPO/'benchmarks/tcga_downstream/pipeline'));from run_attention_pooling import load_frozen_encoder
 dev=torch.device(device);model=load_frozen_encoder(dev);N=len(m);maps={(s,p):open_memmap(paths(s,p),mode='w+',dtype='float32',shape=(N,d)) for s in STAGES for p,d in POOLS.items()};started=time.time()
 with torch.no_grad():
  for a in range(0,N,batch_size):
   b=min(a+batch_size,N);x=torch.as_tensor(X[a:b],device=dev);ids=torch.arange(G,device=dev)
   with torch.autocast(device_type=dev.type,dtype=torch.float16,enabled=dev.type=='cuda'):
    h=model.gene_embedding(ids).unsqueeze(0)+model.ree(x)
    for stage in STAGES:
     if stage!='input_embedding':h=model.layers[int(stage.split('_')[1])-1](h)
     for name,v in pool(h,x).items():maps[(stage,name)][a:b]=v.cpu().numpy()
   if b==N or time.time()-started>0 and (a//batch_size)%100==0:say(f'extract {b:,}/{N:,} elapsed={(time.time()-started)/60:.1f}m')
 for v in maps.values():v.flush()
 complete.write_text(json.dumps({'samples':N,'stages':STAGES,'pools':POOLS,'device':str(dev),'batch_size':batch_size},indent=2)+'\n');say('extraction complete')

def effdim(A):
 A=np.asarray(A,float);A-=A.mean(0);e=np.linalg.eigvalsh(A.T@A/(len(A)-1));e=np.maximum(e,0);e=e[::-1];r=e/e.sum();pr=e.sum()**2/(e@e);er=np.exp(-np.sum(r[r>0]*np.log(r[r>0])));return e,r,pr,er
def metrics(y,p):return {'accuracy':accuracy_score(y,p),'balanced_accuracy':balanced_accuracy_score(y,p),'macro_f1':f1_score(y,p,average='macro',zero_division=0)}
def fit(A,y,B,yt,alpha):
 # A deterministic LSQR ridge probe retains a linear decision function while
 # avoiding the repeatedly non-convergent and hours-long logistic fits.
 sc=StandardScaler().fit(A)
 mod=RidgeClassifier(alpha=alpha,solver='lsqr',tol=1e-3,
                     class_weight='balanced').fit(sc.transform(A),y)
 p=mod.predict(sc.transform(B));return metrics(yt,p)

def analyze():
 m,_=load_data();y=m.tissue.to_numpy(str);g=m.gse.to_numpy(str);spl=pd.read_csv(PREV/'exact_splits.csv');variance=[];cls=[];nnrows=[];started=time.time()
 for si,stage in enumerate(STAGES):
  for poolname,dim in POOLS.items():
   A=np.load(paths(stage,poolname),mmap_mode='r');e,r,pr,er=effdim(A);variance.append({'stage':stage,'stage_index':si,'pool':poolname,'dimensions':dim,'PC1':r[0],'PC2':r[1],'PC1_2':r[:2].sum(),'PC1_5':r[:5].sum(),'PC1_10':r[:10].sum(),'participation_ratio':pr,'entropy_effective_rank':er})
   nn=NearestNeighbors(n_neighbors=11,metric='cosine',n_jobs=-1).fit(A).kneighbors(return_distance=False)[:,:10];nnrows.append({'stage':stage,'stage_index':si,'pool':poolname,'tissue_10nn_purity':np.mean([np.mean(y[nn[i]]==y[i]) for i in range(len(y))]),'study_10nn_purity':np.mean([np.mean(g[nn[i]]==g[i]) for i in range(len(y))])})
   for fold in range(5):
    q=spl[(spl.fold==fold)&(spl.scheme=='group5')];tr=q[q.partition=='train'].matrix_row;te=q[q.partition=='test'].matrix_row;ix=pd.Series(np.arange(len(m)),index=m.matrix_row);tr=ix.loc[tr].to_numpy();te=ix.loc[te].to_numpy();ia,iv=next(GroupShuffleSplit(1,test_size=.2,random_state=SEED).split(A[tr],y[tr],g[tr]));cand=[]
    for alpha in [.1,1.,10.]:cand.append((fit(A[tr][ia],y[tr][ia],A[tr][iv],y[tr][iv],alpha)['macro_f1'],alpha))
    alpha=max(cand)[1];z=fit(A[tr],y[tr],A[te],y[te],alpha);cls.append({'stage':stage,'stage_index':si,'pool':poolname,'fold':fold,'dimensions':dim,'alpha':alpha,**z})
   say(f'analyze stage={stage} pool={poolname} {si*len(POOLS)+list(POOLS).index(poolname)+1}/{len(STAGES)*len(POOLS)} elapsed={(time.time()-started)/60:.1f}m')
 pd.DataFrame(variance).to_csv(OUT/'layer_pooling_dimensionality.csv',index=False);pd.DataFrame(nnrows).to_csv(OUT/'layer_pooling_neighborhoods.csv',index=False);pd.DataFrame(cls).to_csv(OUT/'layer_pooling_classification_folds.csv',index=False)
 s=pd.DataFrame(cls).groupby(['stage','stage_index','pool','dimensions']).agg(study_disjoint_macro_f1=('macro_f1','mean'),macro_f1_sd=('macro_f1','std'),balanced_accuracy=('balanced_accuracy','mean'),accuracy=('accuracy','mean')).reset_index();s.to_csv(OUT/'layer_pooling_classification_summary.csv',index=False)
 v=pd.DataFrame(variance);n=pd.DataFrame(nnrows);loss=v.merge(s,on=['stage','stage_index','pool','dimensions']).merge(n,on=['stage','stage_index','pool']);loss.to_csv(OUT/'information_loss_map.csv',index=False);figures(loss)
 (OUT/'probe_provenance.json').write_text(json.dumps({'classifier':'class-balanced RidgeClassifier, LSQR solver','alpha_grid':[.1,1.,10.],'selection':'one training-only GSE-disjoint validation split per outer fold','outer_splits':'exact five GroupKFold splits from prior generalization benchmark','reason':'LBFGS logistic first pass stopped after repeated non-convergence and projected multi-hour runtime; ridge is applied identically to every representation','cached_representations_reused':True,'comparison_note':'Prior raw/PCA/Bridge logistic results remain fixed external references; all layer-localization comparisons use this identical ridge probe.'},indent=2)+'\n');say('analysis complete')

def figures(d):
 import matplotlib.pyplot as plt
 F=OUT/'figures';F.mkdir(exist_ok=True);plt.style.use('seaborn-v0_8-whitegrid')
 for col,title,file in [('participation_ratio','Effective dimensionality through BridgeRNA','participation_ratio'),('PC1_2','PC1+PC2 variance through BridgeRNA','pc1_pc2'),('study_disjoint_macro_f1','Study-disjoint tissue information through BridgeRNA','tissue_f1')]:
  fig,ax=plt.subplots(figsize=(12,6))
  for p,q in d.groupby('pool'):ax.plot(q.stage_index,q[col],marker='o',label=p)
  ax.set(xticks=range(len(STAGES)),xticklabels=STAGES,xlabel='Stage',ylabel=col,title=title);ax.tick_params(axis='x',rotation=40);ax.legend(ncol=2,fontsize=8);fig.tight_layout()
  for ext in ['png','pdf']:fig.savefig(F/f'{file}.{ext}',dpi=400)
  plt.close(fig)

def main():
 p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');p.add_argument('--batch-size',type=int,default=1);p.add_argument('--phase',choices=['all','extract','analyze'],default='all');a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True);architecture()
 if a.phase in ['all','extract']:extract(a.device,a.batch_size)
 if a.phase in ['all','analyze']:analyze()
if __name__=='__main__':main()
