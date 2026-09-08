#!/usr/bin/env python3
"""Task-agnostic frozen readouts from BridgeRNA layers 11 and 12."""
from __future__ import annotations
import argparse,json,math,sys,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from numpy.lib.format import open_memmap
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parents[1];REPO=HERE.parents[1];OUT=HERE/'results/task_agnostic';WORK=HERE/'work/task_agnostic'
OLD=REPO/'benchmarks/library_prep_disentanglement';PREV=OLD/'results/task4_pca_vs_bridgerna_generalization';LCACHE=OLD/'work/task4_information_collapse';HWORK=REPO/'benchmarks/cross_species_exercise_response/work/hallmark_readout';T3=REPO/'benchmarks/osdr_batch_effect_representation';G=15165;SEED=20260907
READOUTS={'ditto_sampled_incoming':512,'ditto_centroid_query':512,'hallmark_module_mean':50*512,'hallmark_module_mean_std':50*1024}
SPECS={'RR1_original':('C14__OSD-48__RR1-NASA__37-day',None),'RR1_remeasurement':('C04__OSD-168__RR1-NASA__37-day',None),'RR3_39_original':('C01__OSD-137__RR3__39-day',None),'RR3_39_remeasurement':('C05__OSD-168__RR3__39-day',None),'RR3_40_original':('C02__OSD-137__RR3__40-day','strict'),'RR3_40_remeasurement':('C06__OSD-168__RR3__40-day',None)}
def say(s):print(f'[{time.strftime("%F %T")}] {s}',flush=True)
def cos(a,b):
 d=np.linalg.norm(a)*np.linalg.norm(b);return float(a@b/d) if d else np.nan
def cohort():
 m=pd.read_parquet(PREV/'sample_manifest.parquet').sort_values('matrix_row').reset_index(drop=True);x=np.memmap(HWORK/'archs4_log1p_tpm.float32.mmap',mode='r',dtype='float32',shape=(40000,G));return m,x,m.matrix_row.to_numpy(int)
def hallmarks():
 raw=json.loads((REPO/'data/gsea/hallmark_gene_sets.json').read_text());genes=pd.read_csv(REPO/'data/ensembl/canonical_genes.csv').sort_values('token_id').gene_symbol.astype(str).str.upper().tolist();lookup={g:i for i,g in enumerate(genes)};names=[];sets=[];rows=[]
 for name,members in raw.items():
  ix=np.array(sorted({lookup[str(g).upper()] for g in members if str(g).upper() in lookup}),dtype=int);names.append(name);sets.append(ix);rows.append({'module':name,'source_genes':len(set(members)),'mapped_genes':len(ix)})
 if len(names)!=50 or min(map(len,sets))<2:raise AssertionError('Invalid Hallmark mapping')
 pd.DataFrame(rows).to_csv(OUT/'hallmark_mapping.csv',index=False);return names,sets
def paths(dataset,layer,name):return WORK/f'{dataset}__layer{layer}__{name}.float16.npy'
def attention_weights(layer,h,query_positions):
 n=layer.norm1(h);B,S,_=n.shape;H=layer.n_heads;D=layer.head_dim;q=layer.q_proj(n).view(B,S,H,D).transpose(1,2);k=layer.k_proj(n).view(B,S,H,D).transpose(1,2)
 sampled=torch.softmax(torch.matmul(q[:,:,query_positions],k.transpose(-2,-1))/math.sqrt(D),dim=-1).mean((1,2))
 centroid=torch.softmax(torch.einsum('bhd,bhsd->bhs',q.mean(2),k)/math.sqrt(D),dim=-1).mean(1)
 return sampled,centroid
def extract(dataset,device,batch_size):
 OUT.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True);_,sets=hallmarks()
 if dataset=='tissue':m,x,rows=cohort()
 else:m=pd.read_csv(T3/'results/sample_manifest.csv');x=np.load(T3/'work/bridgerna_log1p_tpm_inputs.npy',mmap_mode='r');rows=np.arange(len(m))
 if all(paths(dataset,l,n).exists() for l in [11,12] for n in READOUTS):say(f'reuse complete {dataset} cache');return
 sys.path.insert(0,str(REPO/'benchmarks/tcga_imputation/pipeline'));from model_adapters import load_ours
 dev=torch.device(device);model=load_ours(dev).eval();[p.requires_grad_(False) for p in model.parameters()];N=len(rows);maps={(l,n):open_memmap(paths(dataset,l,n),mode='w+',dtype='float16',shape=(N,d)) for l in [11,12] for n,d in READOUTS.items()};ids=torch.arange(G,device=dev);qpos=torch.linspace(0,G-1,64,device=dev).round().long();started=time.time()
 with torch.no_grad():
  for a in range(0,N,batch_size):
   b=min(a+batch_size,N);v=torch.as_tensor(np.array(x[rows[a:b]],copy=True),device=dev);h=model.gene_embedding(ids).unsqueeze(0)+model.ree(v)
   with torch.autocast('cuda',dtype=torch.float16,enabled=dev.type=='cuda'):
    for li,layer in enumerate(model.layers,1):
     weights=attention_weights(layer,h,qpos) if li in [11,12] else None;h=layer(h)
     if li in [11,12]:
      hf=h.float();sampled,centroid=weights;values={'ditto_sampled_incoming':(hf*sampled[:,:,None]).sum(1),'ditto_centroid_query':(hf*centroid[:,:,None]).sum(1)};means=[];stds=[]
      for ix in sets:
       z=hf[:,ix];means.append(z.mean(1));stds.append(z.std(1,unbiased=False))
      values['hallmark_module_mean']=torch.cat(means,1);values['hallmark_module_mean_std']=torch.cat([*means,*stds],1)
      for name,z in values.items():maps[(li,name)][a:b]=z.cpu().numpy().astype('float16')
   if b==N or b%100==0:say(f'extract {dataset} {b}/{N} elapsed={(time.time()-started)/60:.1f}m')
 for z in maps.values():z.flush()
 (WORK/f'{dataset}_complete.json').write_text(json.dumps({'samples':N,'layers':[11,12],'readouts':READOUTS,'attention_queries':64,'dtype':'float16'},indent=2)+'\n')
def split(m,fold):
 s=pd.read_csv(PREV/'exact_splits.csv');q=s[(s.scheme=='group5')&(s.fold==fold)];look=pd.Series(np.arange(len(m)),index=m.matrix_row);return look.loc[q[q.partition=='train'].matrix_row].to_numpy(),look.loc[q[q.partition=='test'].matrix_row].to_numpy()
def fit(A,y,B,yt,alpha):
 sc=StandardScaler().fit(A);model=RidgeClassifier(alpha=alpha,solver='lsqr',tol=1e-3,class_weight='balanced').fit(sc.transform(A),y);p=model.predict(sc.transform(B));return {'accuracy':accuracy_score(yt,p),'balanced_accuracy':balanced_accuracy_score(yt,p),'macro_f1':f1_score(yt,p,average='macro',zero_division=0)}
def geometry(A,y,g,device='cuda:0'):
 A=np.array(A,dtype='float32',copy=True);A-=A.mean(0);sv=TruncatedSVD(2,random_state=SEED).fit(A).singular_values_**2;total=float(np.square(A).sum());pc12=float(sv.sum()/total);gram=A@A.T;pr=float(total**2/np.square(gram).sum());norm=np.linalg.norm(A,axis=1,keepdims=True);norm[norm==0]=1;sim=(A/norm)@(A/norm).T;np.fill_diagonal(sim,-np.inf);nn=np.argpartition(sim,-10,axis=1)[:,-10:];return {'PC1_2':pc12,'participation_ratio':pr,'tissue_10nn_purity':np.mean([np.mean(y[j]==y[i]) for i,j in enumerate(nn)]),'study_10nn_purity':np.mean([np.mean(g[j]==g[i]) for i,j in enumerate(nn)])}
def fixed_arrays():
 p=lambda l,n:np.load(LCACHE/f'layer_{l}__{n}.float32.npy',mmap_mode='r');return {'L11_L12_equal_mean_std':(np.concatenate([p(11,'mean'),p(11,'std')],1)+np.concatenate([p(12,'mean'),p(12,'std')],1))/2,'L11_L12_concat_mean_std':np.concatenate([p(11,'mean'),p(11,'std'),p(12,'mean'),p(12,'std')],1),'L11_mean_std':np.concatenate([p(11,'mean'),p(11,'std')],1),'L12_mean_std':np.concatenate([p(12,'mean'),p(12,'std')],1),'L12_mean':p(12,'mean')}
def evaluate():
 m,_,_=cohort();y=m.tissue.astype(str).to_numpy();g=m.gse.astype(str).to_numpy();features=fixed_arrays()
 for l in [11,12]:
  for n in READOUTS:features[f'L{l}_{n}']=np.load(paths('tissue',l,n),mmap_mode='r')
 rows=[];geos=[]
 for name,A in features.items():
  geos.append({'readout':name,'dimensions':A.shape[1],**geometry(A,y,g)});say(f'geometry {name}')
  for fold in range(5):
   tr,te=split(m,fold);ia,iv=next(GroupShuffleSplit(1,test_size=.2,random_state=SEED+fold).split(tr,y[tr],g[tr]));fi=tr[ia];va=tr[iv];best=max((fit(A[fi],y[fi],A[va],y[va],a)['macro_f1'],a) for a in [.1,1,10])[1];rows.append({'readout':name,'fold':fold,'dimensions':A.shape[1],**fit(A[tr],y[tr],A[te],y[te],best)});say(f'classify {name} fold {fold}')
 f=pd.DataFrame(rows);geo=pd.DataFrame(geos);f.to_csv(OUT/'study_disjoint_fold_metrics.csv',index=False);geo.to_csv(OUT/'representation_geometry.csv',index=False);s=f.groupby(['readout','dimensions']).agg(study_disjoint_f1=('macro_f1','mean'),f1_sd=('macro_f1','std'),balanced_accuracy=('balanced_accuracy','mean')).reset_index().merge(geo,on=['readout','dimensions']);s.sort_values('study_disjoint_f1',ascending=False).to_csv(OUT/'task_agnostic_ranked_summary.csv',index=False)
def responses(A):
 m=pd.read_csv(T3/'results/sample_manifest.csv');mem=pd.read_csv(T3/'results/task3b_contrast_sample_membership.csv');ix=dict(zip(m.sample_id,range(len(m))));out={}
 for name,(cid,strict) in SPECS.items():
  q=mem[mem.contrast_id.eq(cid)].copy()
  if name=='RR1_original':q=q[~q.sample_id.str.endswith('_M27')]
  if name=='RR1_remeasurement':q=q[~q.sample_id.str.endswith('_M29')]
  if strict:q=q[~q.sample_id.str.endswith('_F5')]
  z={c:q[q.condition.eq(c)].sample_id.map(ix).to_numpy(int) for c in ['FLT','GC']};out[name]=np.asarray(A[z['FLT']],float).mean(0)-np.asarray(A[z['GC']],float).mean(0)
 return out
def response_check():
 features={'current_mean':np.load(T3/'work/bridgerna_embeddings.npy')}
 for l in [11,12]:
  for n in READOUTS:features[f'L{l}_{n}']=np.load(paths('task3',l,n),mmap_mode='r')
 # fixed cross-layer combinations are reconstructed from newly cached Task3 module-free summaries where possible
 features['L11_L12_equal_ditto_sampled']=(np.asarray(features['L11_ditto_sampled_incoming'])+np.asarray(features['L12_ditto_sampled_incoming']))/2
 rows=[]
 for name,A in features.items():
  v=responses(A);rows.append({'readout':name,'RR1_cosine':cos(v['RR1_original'],v['RR1_remeasurement']),'RR3_39_cosine':cos(v['RR3_39_original'],v['RR3_39_remeasurement']),'RR3_40_cosine':cos(v['RR3_40_original'],v['RR3_40_remeasurement']),'false_friend_cosine':cos(v['RR1_original'],v['RR3_39_original'])})
 pd.DataFrame(rows).to_csv(OUT/'response_geometry_summary.csv',index=False)
def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['extract','evaluate','responses','all'],default='all');p.add_argument('--device',default='cuda:0');p.add_argument('--batch-size',type=int,default=1);a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True)
 if a.phase in ['extract','all']:extract('tissue',a.device,a.batch_size);extract('task3',a.device,a.batch_size)
 if a.phase in ['evaluate','all']:evaluate()
 if a.phase in ['responses','all']:response_check()
 (OUT/'provenance.json').write_text(json.dumps({'encoder':'frozen r7hnr92k','layers':[11,12],'ditto_sampled_queries':64,'ditto_definition':'mean existing scaled-dot-product attention received by genes from 64 deterministic query tokens and 8 heads','ditto_centroid_definition':'existing Q/K projections; per-head mean query attends to all keys; weights averaged over heads','modules':'50 MSigDB Hallmarks; overlapping membership retained','module_representation':'ordered concatenation of per-module contextual-token means or means+SDs','splits':'exact existing five GSE-disjoint folds','limitations':['no CLS token exists','sampled incoming attention approximates rather than materializes the full 15165x15165 attention matrix']},indent=2)+'\n');say('complete')
if __name__=='__main__':main()
