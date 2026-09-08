#!/usr/bin/env python3
"""Train and evaluate small readouts on frozen BridgeRNA layers 11 and 12."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torch import nn

HERE=Path(__file__).resolve().parents[1];REPO=HERE.parents[1]
OLD=REPO/'benchmarks/library_prep_disentanglement';PREV=OLD/'results/task4_pca_vs_bridgerna_generalization';CACHE=OLD/'work/task4_information_collapse'
T3=REPO/'benchmarks/osdr_batch_effect_representation';HWORK=REPO/'benchmarks/cross_species_exercise_response/work/hallmark_readout'
OUT=HERE/'results';WORK=HERE/'work';G=15165;SEED=20260906
FIXED=['mean','std','mean_plus_std','mean_plus_std_plus_max','expression_weighted_mean']
SPECS={'RR1_original':('C14__OSD-48__RR1-NASA__37-day',None),'RR1_remeasurement':('C04__OSD-168__RR1-NASA__37-day',None),'RR3_39_original':('C01__OSD-137__RR3__39-day',None),'RR3_39_remeasurement':('C05__OSD-168__RR3__39-day',None),'RR3_40_original':('C02__OSD-137__RR3__40-day','strict'),'RR3_40_remeasurement':('C06__OSD-168__RR3__40-day',None)}
def say(s):print(f'[{time.strftime("%F %T")}] {s}',flush=True)
def cos(a,b):
 d=np.linalg.norm(a)*np.linalg.norm(b);return float(a@b/d) if d else np.nan
def arrays(layer,name):
 p=lambda n:np.load(CACHE/f'layer_{layer}__{n}.float32.npy',mmap_mode='r')
 if name=='mean_plus_std_plus_max':return np.concatenate([p('mean'),p('std'),p('max')],1)
 return p(name)
def cohort():
 m=pd.read_parquet(PREV/'sample_manifest.parquet').sort_values('matrix_row').reset_index(drop=True)
 x=np.memmap(HWORK/'archs4_log1p_tpm.float32.mmap',mode='r',dtype='float32',shape=(40000,G));return m,x
def split(m,fold):
 s=pd.read_csv(PREV/'exact_splits.csv');q=s[(s.scheme=='group5')&(s.fold==fold)];lookup=pd.Series(np.arange(len(m)),index=m.matrix_row)
 return lookup.loc[q[q.partition=='train'].matrix_row].to_numpy(),lookup.loc[q[q.partition=='test'].matrix_row].to_numpy()
def metrics(y,p):return {'accuracy':accuracy_score(y,p),'balanced_accuracy':balanced_accuracy_score(y,p),'macro_f1':f1_score(y,p,average='macro',zero_division=0)}
def ridge(A,y,B,yt,alpha):
 sc=StandardScaler().fit(A);model=RidgeClassifier(alpha=alpha,solver='lsqr',tol=1e-3,class_weight='balanced').fit(sc.transform(A),y);return metrics(yt,model.predict(sc.transform(B)))
def geometry(A,y,g):
 a=np.asarray(A,float);a-=a.mean(0);e=np.maximum(np.linalg.eigvalsh(a.T@a/(len(a)-1)),0);r=e/e.sum();nn=NearestNeighbors(n_neighbors=11,metric='cosine',n_jobs=-1).fit(A).kneighbors(return_distance=False)[:,:10]
 return {'PC1_2':float(np.sort(r)[-2:].sum()),'participation_ratio':float(e.sum()**2/(e@e)),'tissue_10nn_purity':float(np.mean([np.mean(y[nn[i]]==y[i]) for i in range(len(y))])),'study_10nn_purity':float(np.mean([np.mean(g[nn[i]]==g[i]) for i in range(len(y))]))}

class QueryPool(nn.Module):
 def __init__(self,heads=1):super().__init__();self.query=nn.Parameter(torch.randn(heads,512)/512**.5);self.gate=nn.Parameter(torch.zeros(heads));self.classifier=None
 def attach(self,n):self.classifier=nn.Linear(512,n);return self
 def pool(self,h):
  logits=torch.einsum('bgd,hd->bhg',h.float(),self.query);w=logits.softmax(-1);p=torch.einsum('bhg,bgd->bhd',w,h.float());return (p*self.gate.softmax(0)[None,:,None]).sum(1)
 def forward(self,h):return self.classifier(self.pool(h))

def fixed():
 m,_=cohort();y=m.tissue.astype(str).to_numpy();g=m.gse.astype(str).to_numpy();rows=[];geoms=[]
 for layer in [11,12]:
  for name in FIXED:
   A=arrays(layer,name);geoms.append({'layer':layer,'readout':name,'dimensions':A.shape[1],'trainable_params':0,**geometry(A,y,g)})
   for fold in range(5):
    tr,te=split(m,fold);ia,iv=next(GroupShuffleSplit(1,test_size=.2,random_state=SEED+fold).split(A[tr],y[tr],g[tr]));best=max((ridge(A[tr][ia],y[tr][ia],A[tr][iv],y[tr][iv],a)['macro_f1'],a) for a in [.1,1,10])[1];rows.append({'layer':layer,'readout':name,'fold':fold,'dimensions':A.shape[1],'trainable_params':0,'alpha':best,**ridge(A[tr],y[tr],A[te],y[te],best)})
   say(f'fixed layer={layer} {name}')
 pd.DataFrame(rows).to_csv(OUT/'fixed_fold_metrics.csv',index=False);pd.DataFrame(geoms).to_csv(OUT/'fixed_geometry.csv',index=False)

def combined():
 """Cheap cross-layer readouts, with every choice fitted inside training data."""
 target=OUT/'combined_fold_metrics.csv'
 if target.exists():return pd.read_csv(target)
 m,_=cohort();y=m.tissue.astype(str).to_numpy();g=m.gse.astype(str).to_numpy();m11=np.asarray(arrays(11,'mean'));m12=np.asarray(arrays(12,'mean'));s11=np.asarray(arrays(11,'std'));s12=np.asarray(arrays(12,'std'));rows=[]
 for fold in range(5):
  tr,te=split(m,fold);ia,iv=next(GroupShuffleSplit(1,test_size=.2,random_state=SEED+fold).split(tr,y[tr],g[tr]));fitix=tr[ia];valid=tr[iv]
  candidates=[]
  for weight in [0,.25,.5,.75,1.]:
   A=weight*m11+(1-weight)*m12
   for alpha in [.1,1,10]:candidates.append((ridge(A[fitix],y[fitix],A[valid],y[valid],alpha)['macro_f1'],weight,alpha))
  _,weight,alpha=max(candidates);A=weight*m11+(1-weight)*m12;z=ridge(A[tr],y[tr],A[te],y[te],alpha);geo=geometry(A[te],y[te],g[te]);rows.append({'layer':'11+12','readout':'scalar_mixture_mean','fold':fold,'dimensions':512,'trainable_params':1,'mixture_weight_layer11':weight,**z,**geo})
  C=np.concatenate([m11,s11,m12,s12],1);p=PCA(n_components=512,svd_solver='randomized',random_state=SEED+fold).fit(C[fitix]);U=p.transform(C[fitix]);V=p.transform(C[valid]);alpha=max((ridge(U,y[fitix],V,y[valid],a)['macro_f1'],a) for a in [.1,1,10])[1];p=PCA(n_components=512,svd_solver='randomized',random_state=SEED+fold).fit(C[tr]);Atr=p.transform(C[tr]);Ate=p.transform(C[te]);z=ridge(Atr,y[tr],Ate,y[te],alpha);geo=geometry(Ate,y[te],g[te]);rows.append({'layer':'11+12','readout':'mean_std_pca_projection','fold':fold,'dimensions':512,'trainable_params':2048*512+512,'mixture_weight_layer11':np.nan,**z,**geo});say(f'combined fold {fold}')
 out=pd.DataFrame(rows);out.to_csv(target,index=False);return out

def hidden(model,v):
 ids=torch.arange(G,device=v.device);h=model.gene_embedding(ids).unsqueeze(0)+model.ree(v);h11=None
 for i,layer in enumerate(model.layers,1):h=layer(h);h11=h if i==11 else h11
 return h11.detach(),h.detach()
def train(folds,device,epochs,batch):
 m,X=cohort();ycat=pd.Categorical(m.tissue);y=ycat.codes;g=m.gse.astype(str).to_numpy();dev=torch.device(device);sys.path.insert(0,str(REPO/'benchmarks/tcga_downstream/pipeline'));from run_attention_pooling import load_frozen_encoder
 enc=load_frozen_encoder(dev);worker=OUT/'workers';worker.mkdir(parents=True,exist_ok=True)
 for fold in folds:
  target=worker/f'fold_{fold}.csv'
  if target.exists():say(f'reuse fold {fold}');continue
  tr,te=split(m,fold);ia,iv=next(GroupShuffleSplit(1,test_size=.2,random_state=SEED+fold).split(tr,y[tr],g[tr]));trainix=tr[ia];valix=tr[iv]
  models={(layer,kind):QueryPool(1 if kind=='single_query_attention' else 4).attach(len(ycat.categories)).to(dev) for layer in [11,12] for kind in ['single_query_attention','multihead_attention']}
  opts={k:torch.optim.AdamW(v.parameters(),lr=2e-4,weight_decay=1e-3) for k,v in models.items()};lossfn=nn.CrossEntropyLoss();best={k:(-1,None) for k in models};started=time.time();rng=np.random.default_rng(SEED+fold)
  for epoch in range(epochs):
   order=rng.permutation(trainix);[v.train() for v in models.values()]
   for bi,a in enumerate(range(0,len(order),batch)):
    ix=order[a:a+batch];v=torch.as_tensor(np.array(X[m.matrix_row.to_numpy()[ix]],copy=True),device=dev)
    with torch.no_grad(),torch.autocast(device_type=dev.type,dtype=torch.float16):h11,h12=hidden(enc,v)
    targety=torch.as_tensor(y[ix],dtype=torch.long,device=dev)
    for k,mod in models.items():opts[k].zero_grad(set_to_none=True);loss=lossfn(mod(h11 if k[0]==11 else h12),targety);loss.backward();opts[k].step()
    if bi%250==0:say(f'fold={fold} epoch={epoch+1}/{epochs} samples={min(a+batch,len(order))}/{len(order)} elapsed={(time.time()-started)/60:.1f}m')
   # validation after each epoch, one shared encoder pass
   pred={k:[] for k in models};truth=[];[v.eval() for v in models.values()]
   with torch.no_grad():
    for a in range(0,len(valix),batch):
     ix=valix[a:a+batch];v=torch.as_tensor(np.array(X[m.matrix_row.to_numpy()[ix]],copy=True),device=dev)
     with torch.autocast(device_type=dev.type,dtype=torch.float16):h11,h12=hidden(enc,v)
     truth.extend(y[ix]);
     for k,mod in models.items():pred[k].extend(mod(h11 if k[0]==11 else h12).argmax(1).cpu().tolist())
   for k in models:
    score=f1_score(truth,pred[k],average='macro',zero_division=0)
    if score>best[k][0]:best[k]=(score,{n:t.detach().cpu() for n,t in models[k].state_dict().items()})
  # one test pass; save pooled test embeddings for response-independent geometry
  pred={k:[] for k in models};pooled={k:[] for k in models};truth=[]
  for k in models:models[k].load_state_dict(best[k][1]);models[k].eval()
  with torch.no_grad():
   for a in range(0,len(te),batch):
    ix=te[a:a+batch];v=torch.as_tensor(np.array(X[m.matrix_row.to_numpy()[ix]],copy=True),device=dev)
    with torch.autocast(device_type=dev.type,dtype=torch.float16):h11,h12=hidden(enc,v)
    truth.extend(y[ix])
    for k,mod in models.items():
     p=mod.pool(h11 if k[0]==11 else h12);pred[k].extend(mod.classifier(p).argmax(1).cpu().tolist());pooled[k].append(p.cpu().numpy())
  rows=[]
  for k in models:
   A=np.concatenate(pooled[k]);geo=geometry(A,np.asarray(ycat.categories)[y[te]],g[te]);rows.append({'layer':k[0],'readout':k[1],'fold':fold,'dimensions':512,'trainable_params':sum(p.numel() for p in models[k].parameters()),'validation_macro_f1':best[k][0],**metrics(y[te],pred[k]),**geo})
   torch.save(best[k][1],worker/f'fold_{fold}_layer{k[0]}_{k[1]}.pt')
  pd.DataFrame(rows).to_csv(target,index=False);say(f'fold {fold} complete')

def finalize():
 fixed=pd.read_csv(OUT/'fixed_fold_metrics.csv') if (OUT/'fixed_fold_metrics.csv').exists() else pd.DataFrame();geo=pd.read_csv(OUT/'fixed_geometry.csv') if (OUT/'fixed_geometry.csv').exists() else pd.DataFrame();att=pd.concat([pd.read_csv(p) for p in sorted((OUT/'workers').glob('fold_*.csv'))]);comb=combined()
 fs=fixed.groupby(['layer','readout','dimensions','trainable_params']).agg(study_disjoint_f1=('macro_f1','mean'),f1_sd=('macro_f1','std'),balanced_accuracy=('balanced_accuracy','mean')).reset_index().merge(geo,on=['layer','readout','dimensions','trainable_params'])
 ats=att.groupby(['layer','readout','dimensions','trainable_params']).agg(study_disjoint_f1=('macro_f1','mean'),f1_sd=('macro_f1','std'),balanced_accuracy=('balanced_accuracy','mean'),tissue_10nn_purity=('tissue_10nn_purity','mean'),study_10nn_purity=('study_10nn_purity','mean'),PC1_2=('PC1_2','mean'),participation_ratio=('participation_ratio','mean')).reset_index();cs=comb.groupby(['layer','readout','dimensions','trainable_params']).agg(study_disjoint_f1=('macro_f1','mean'),f1_sd=('macro_f1','std'),balanced_accuracy=('balanced_accuracy','mean'),tissue_10nn_purity=('tissue_10nn_purity','mean'),study_10nn_purity=('study_10nn_purity','mean'),PC1_2=('PC1_2','mean'),participation_ratio=('participation_ratio','mean')).reset_index();rank=pd.concat([fs,ats,cs],ignore_index=True).sort_values('study_disjoint_f1',ascending=False);rank.to_csv(OUT/'ranked_summary.csv',index=False)
 (OUT/'provenance.json').write_text(json.dumps({'encoder_frozen':True,'layers':[11,12],'folds':'exact existing five GSE-disjoint folds','trainable_readouts':'trained only within each downstream outer fold','epochs':8,'note':'Attention test-fold geometry is averaged within fold; fixed readout geometry uses all samples without labels.'},indent=2)+'\n');print(rank.to_string(index=False))
def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fixed','train','finalize'],required=True);p.add_argument('--folds',nargs='*',type=int,default=list(range(5)));p.add_argument('--device',default='cuda:0');p.add_argument('--epochs',type=int,default=8);p.add_argument('--batch-size',type=int,default=1);a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True);WORK.mkdir(exist_ok=True)
 if a.phase=='fixed':fixed()
 elif a.phase=='train':train(a.folds,a.device,a.epochs,a.batch_size)
 else:finalize()
if __name__=='__main__':main()
