#!/usr/bin/env python3
"""Established graph-kernel audit on frozen BridgeRNA contextual graphs."""
from __future__ import annotations
import hashlib,json,time,resource,sys
from collections import defaultdict,deque
from pathlib import Path
import joblib,numpy as np,pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.metrics.pairwise import cosine_similarity

HERE=Path(__file__).resolve().parents[1];REPO=HERE.parents[1];OUT=HERE/'results';WORK=HERE/'work';SEED=20260907;G=15165;D=2**18
OLD=REPO/'benchmarks/frozen_sample_embedding_readout';CACHE=OLD/'work/graph_fingerprint';T4=REPO/'benchmarks/library_prep_disentanglement';EX=REPO/'benchmarks/cross_species_exercise_response';T3=REPO/'benchmarks/osdr_batch_effect_representation';PREV=T4/'results/task4_pca_vs_bridgerna_generalization';HWORK=EX/'work/hallmark_readout';TAG=OLD/'work/task_agnostic';LCACHE=T4/'work/task4_information_collapse'
def say(s):print(f'[{time.strftime("%F %T")}] {s}',flush=True)
def stable(x):return int.from_bytes(hashlib.blake2b(str(x).encode(),digest_size=8).digest(),'little')
def load(name):
 p=CACHE/name;return pd.read_parquet(p/'manifest.parquet'),np.load(p/'neighbors_top20.uint16.npy',mmap_mode='r'),np.load(p/'weights_top20.float16.npy',mmap_mode='r')
def adjacency(n,w,k=10):
 src=np.repeat(np.arange(G),k);dst=np.asarray(n[:,:k],int).ravel();val=np.asarray(w[:,:k],np.float32).ravel();a=np.minimum(src,dst);b=np.maximum(src,dst);code=a.astype(np.int64)*G+b;order=np.argsort(code);code=code[order];val=val[order];u,start=np.unique(code,return_index=True);v=np.maximum.reduceat(val,start);i=u//G;j=u%G;return csr_matrix((np.r_[v,v],(np.r_[i,j],np.r_[j,i])),shape=(G,G))
def hashed(rows):
 cols=[];vals=[]
 for d in rows:
  cols.extend(d);vals.extend(d.values())
 indptr=np.cumsum([0]+[len(d) for d in rows]);return csr_matrix((vals,cols,indptr),shape=(len(rows),D),dtype=np.float32)
def wl_features(A,identity,iterations=2):
 ind=A.indptr;ix=A.indices;deg=np.diff(ind);labels=np.arange(G,dtype=np.uint64) if identity else np.minimum(7,(deg*8/max(1,deg.max()))).astype(np.uint64);feat=defaultdict(float)
 for it in range(iterations+1):
  for x in labels:feat[stable((it,int(x)))%D]+=1
  if it<iterations:
   new=np.empty(G,np.uint64)
   for g in range(G):new[g]=stable((int(labels[g]),tuple(sorted(map(int,labels[ix[ind[g]:ind[g+1]]])))))
   labels=new
 return feat
def shortest_features(A,landmarks):
 ind=A.indptr;ix=A.indices;hist=defaultdict(float)
 for root in landmarks:
  dist=np.full(G,-1,np.int16);dist[root]=0;q=deque([int(root)])
  while q:
   u=q.popleft()
   if dist[u]>=6:continue
   for v in ix[ind[u]:ind[u+1]]:
    if dist[v]<0:dist[v]=dist[u]+1;q.append(int(v))
  for d,c in zip(*np.unique(dist,return_counts=True)):hist[stable(('sp',int(d)))%D]+=float(c)
 return hist
def graphlet_features(A,triples,quads):
 feat=defaultdict(float)
 for size,sets in [(3,triples),(4,quads)]:
  for nodes in sets:
   z=A[nodes][:,nodes].toarray()>0;edges=int(z.sum()//2);degrees=tuple(sorted(z.sum(0).astype(int)));feat[stable(('graphlet',size,edges,degrees))%D]+=1
 return feat
def spectral(A,q=16):
 deg=np.asarray(A.sum(1)).ravel();inv=np.zeros(G);inv[deg>0]=1/np.sqrt(deg[deg>0]);N=A.multiply(inv[:,None]).multiply(inv[None,:]);return np.sort(eigsh(N,k=q,which='LM',return_eigenvectors=False)).astype(np.float32)
def kernel_features(name,indices,n,w):
 rng=np.random.default_rng(SEED);land=rng.choice(G,8,replace=False);trip=np.sort(rng.choice(G,(1000,3),replace=True),1);quad=np.sort(rng.choice(G,(1000,4),replace=True),1);rows={'WL identity-aware':[],'WL structural':[],'Shortest-path':[],'Graphlet':[]};spec=[];times=defaultdict(float)
 for z,i in enumerate(indices):
  A=adjacency(n[i],w[i]);
  for key,fun in [('WL identity-aware',lambda:wl_features(A,True)),('WL structural',lambda:wl_features(A,False)),('Shortest-path',lambda:shortest_features(A,land)),('Graphlet',lambda:graphlet_features(A,trip,quad))]:
   t=time.time();rows[key].append(fun());times[key]+=time.time()-t
  t=time.time();spec.append(spectral(A));times['Spectral']+=time.time()-t
  if (z+1)%10==0 or z+1==len(indices):say(f'{name} kernels {z+1}/{len(indices)}')
 out={k:hashed(v) for k,v in rows.items()};out['Spectral']=np.stack(spec);return out,times
def retrieval(S,meta,pair_col=None,label_col=None,exclude_study=False):
 ranks=[]
 for i in range(len(meta)):
  score=S[i].copy();score[i]=-np.inf
  if exclude_study:score[meta.gse.astype(str).to_numpy()==str(meta.gse.iloc[i])]=-np.inf
  order=np.argsort(-score)
  if pair_col is not None:hit=np.flatnonzero(meta[pair_col].astype(str).to_numpy()[order]==str(meta[pair_col].iloc[i]))
  else:hit=np.flatnonzero(meta[label_col].astype(str).to_numpy()[order]==str(meta[label_col].iloc[i]))
  ranks.append(int(hit[0]+1) if len(hit) else len(meta)+1)
 r=np.array(ranks);return {'queries':len(r),'R@1':np.mean(r<=1),'R@5':np.mean(r<=5),'R@10':np.mean(r<=10),'MRR':np.mean(1/r),'median_rank':np.median(r)}
def kernel_matrix(X):return cosine_similarity(X)
def run_group(name,meta,n,w,indices,task):
 feats,times=kernel_features(name,indices,n,w);m=meta.iloc[indices].reset_index(drop=True);rows=[]
 for method,X in feats.items():
  S=kernel_matrix(X);metric=retrieval(S,m,pair_col='pair_id' if task=='technical' else None,label_col='tissue' if task=='tissue' else None,exclude_study=task=='tissue');rows.append({'endpoint':task,'method':method,**metric,'runtime_seconds':times[method],'peak_RAM_MB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024});np.save(OUT/f'{task}/{method.lower().replace(" ","_")}_kernel.npy',S)
 return pd.DataFrame(rows),feats,m
def simple_pair(A,B):
 a=A.tocsr();b=B.tocsr();ea=set(zip(*a.nonzero()));eb=set(zip(*b.nonzero()));j=len(ea&eb)/len(ea|eb);x=[];y=[]
 for e in ea|eb:x.append(a[e] if e in ea else 0);y.append(b[e] if e in eb else 0)
 return j,float(np.corrcoef(x,y)[0,1]),float(np.minimum(x,y).sum()/np.maximum(x,y).sum())
def main():
 for p in ['technical','tissue','response','stress','nulls','figures','summary']:(OUT/p).mkdir(parents=True,exist_ok=True)
 say('estimated runtime 12-30 minutes; cache-only, no model inference')
 tm,tn,tw=load('tcell');tech,tf,tm2=run_group('technical',tm,tn,tw,np.arange(len(tm)),'technical');tech.to_csv(OUT/'technical/kernel_retrieval.csv',index=False)
 # Prespecified balanced, study-diverse subset: first deterministic unique GSE samples, max 10/tissue.
 m,n,w=load('tissue');sel=[]
 for tissue,q in m.sort_values(['tissue','gse','gsm']).drop_duplicates(['tissue','gse']).groupby('tissue'):sel.extend(q.head(10).index)
 sel=np.array(sel);tissue,ff,ms=run_group('tissue',m,n,w,sel,'tissue');tissue.to_csv(OUT/'tissue/kernel_retrieval.csv',index=False);m.loc[sel].to_csv(OUT/'tissue/subset_manifest.csv',index=False)
 # Non-graph tissue references use existing full-cohort neighborhood purity; clearly separate from subset kernel MRR.
 pd.read_csv(PREV/'neighborhood_results.csv').to_csv(OUT/'tissue/non_graph_full_cohort_reference.csv',index=False)
 # Existing response/stress comparisons are reused; higher-order response kernels require condition-level feature differences and are recorded as a follow-up limitation.
 stress=pd.read_csv(OLD/'results/graph_fingerprint/rr1_rr3_stress/cross_representation_comparison.csv');stress.to_csv(OUT/'stress/non_graph_and_simple_graph.csv',index=False)
 simple=pd.read_csv(OLD/'results/graph_fingerprint/rr1_rr3_stress/graph_response_stress_metrics.csv');simple.to_csv(OUT/'stress/simple_graph_stress.csv',index=False)
 pd.DataFrame([{'method':'higher-order response kernels','status':'not_estimable_from_sample-kernel features without constructing treatment/control feature-map differences','reason':'WL/graphlet feature differences can be signed but their cosine is not a positive-semidefinite graph kernel; no unvalidated response kernel invented'}]).to_csv(OUT/'response/status.csv',index=False)
 # Null: row-permute identity-aware WL features; structural kernels correctly remain invariant to gene relabeling.
 rng=np.random.default_rng(SEED);null=[]
 for method,X in tf.items():
  observed=retrieval(kernel_matrix(X),tm2,pair_col='pair_id')['MRR'];perm=rng.permutation(len(tm2));shuffled=retrieval(kernel_matrix(X[perm]),tm2,pair_col='pair_id')['MRR'];null.append({'method':method,'observed_MRR':observed,'label_shuffled_MRR':shuffled,'degradation':observed-shuffled,'null_note':'sample-feature permutation; structural graph randomization retained from prior validation'})
 pd.DataFrame(null).to_csv(OUT/'nulls/technical_null.csv',index=False)
 allm=pd.concat([tech,tissue],ignore_index=True);allm.to_csv(OUT/'summary/kernel_summary.csv',index=False)
 # Important requested table; unavailable cells remain explicit rather than invented.
 table=[]
 for method in ['Raw','PCA','Bridge mean','Hallmark mean+SD','Edge Jaccard','Neighborhood Jaccard','Weighted edge','WL identity-aware','WL structural','Shortest-path','Graphlet','Spectral','Propagation']:
  a=tech[tech.method.eq(method)];b=tissue[tissue.method.eq(method)];table.append({'Method':method,'Technical MRR':a.MRR.iloc[0] if len(a) else np.nan,'Tissue MRR':b.MRR.iloc[0] if len(b) else np.nan,'Perturbation MRR':np.nan,'RR3-39 rank':np.nan,'RR3-40 rank':np.nan,'RR1 rank':np.nan,'False-friend rank':np.nan,'Null degradation':np.nan,'Runtime seconds':a.runtime_seconds.iloc[0]+b.runtime_seconds.iloc[0] if len(a) and len(b) else np.nan})
 pd.DataFrame(table).to_csv(OUT/'summary/important_output_table.csv',index=False)
 best=tissue.sort_values('MRR',ascending=False).iloc[0];decision='B. GRAPH KERNEL IS A USEFUL SECONDARY SIMILARITY' if best.MRR>.5 else 'C. SIMPLE GRAPH METRICS ARE SUFFICIENT';summary={'decision':decision,'best_kernel':best.method,'best_subset_tissue_MRR':float(best.MRR),'primary_perturbation_endpoint':'unavailable without defining an unvalidated signed-response kernel','propagation_kernel':'unavailable','tissue_subset_samples':len(sel),'claim_limitation':'graph similarity, not GRN inference'};(OUT/'summary/decision.json').write_text(json.dumps(summary,indent=2)+'\n')
 (OUT/'summary/provenance.json').write_text(json.dumps({'graph':'frozen layer12 union k10','WL_iterations':2,'shortest_path_landmarks':8,'graphlet_samples':{'3_node':1000,'4_node':1000},'spectral_eigenvalues':16,'hash_dimensions':D,'seed':SEED,'RR1_RR3_parameter_selection':False},indent=2)+'\n');say('COMPLETE')
if __name__=='__main__':main()
