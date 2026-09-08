#!/usr/bin/env python3
"""Cache-only contextual RAG retrieval benchmark."""
from __future__ import annotations
import json,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

HERE=Path(__file__).resolve().parents[1];REPO=HERE.parents[1];OUT=HERE/'results';SEED=20260907
T4=REPO/'benchmarks/library_prep_disentanglement';FROZEN=REPO/'benchmarks/frozen_sample_embedding_readout';EX=REPO/'benchmarks/cross_species_exercise_response';T3=REPO/'benchmarks/osdr_batch_effect_representation'
PREV=T4/'results/task4_pca_vs_bridgerna_generalization';HWORK=EX/'work/hallmark_readout';LCACHE=T4/'work/task4_information_collapse';TAG=FROZEN/'work/task_agnostic'
G=15165
def say(s):print(f'[{time.strftime("%F %T")}] {s}',flush=True)
def cosmat(x):
 x=np.asarray(x,dtype=np.float32);n=np.linalg.norm(x,axis=1,keepdims=True);n[n==0]=1;return x/n
def retrieval_metrics(A,meta,label='tissue'):
 # Enough neighbors to remove even the largest same-study block.
 maxg=int(meta.groupby('gse').size().max());k=min(len(meta),maxg+51);nn=NearestNeighbors(n_neighbors=k,metric='cosine',n_jobs=-1).fit(A).kneighbors(return_distance=False);ranks=[];records=[]
 y=meta[label].astype(str).to_numpy();g=meta.gse.astype(str).to_numpy()
 for i,row in enumerate(nn):
  cand=row[g[row]!=g[i]];hits=np.flatnonzero(y[cand]==y[i]);rank=int(hits[0]+1) if len(hits) else len(meta)+1;ranks.append(rank);records.append({'query':meta.gsm.iloc[i],'gse':g[i],'label':y[i],'first_correct_rank':rank,'top1':meta.gsm.iloc[cand[0]],'top1_label':y[cand[0]]})
 r=np.array(ranks);return {'queries':len(r),'R@1':np.mean(r<=1),'R@5':np.mean(r<=5),'R@10':np.mean(r<=10),'MRR':np.mean(1/r),'median_rank':np.median(r)},pd.DataFrame(records)
def retrieval_metrics_chunked(A,meta,label='tissue',chunk=32):
 A=np.asarray(A,dtype=np.float32);norm=np.linalg.norm(A,axis=1);norm[norm==0]=1;y=meta[label].astype(str).to_numpy();g=meta.gse.astype(str).to_numpy();ranks=[];records=[]
 for start in range(0,len(A),chunk):
  q=A[start:start+chunk];sim=(q@A.T)/(np.linalg.norm(q,axis=1,keepdims=True)*norm[None,:])
  for local in range(len(q)):
   i=start+local;sim[local,g==g[i]]=-np.inf;order=np.argsort(-sim[local]);hits=np.flatnonzero(y[order]==y[i]);rank=int(hits[0]+1) if len(hits) else len(A)+1;ranks.append(rank);records.append({'query':meta.gsm.iloc[i],'gse':g[i],'label':y[i],'first_correct_rank':rank,'top1':meta.gsm.iloc[order[0]],'top1_label':y[order[0]]})
 r=np.array(ranks);return {'queries':len(r),'R@1':np.mean(r<=1),'R@5':np.mean(r<=5),'R@10':np.mean(r<=10),'MRR':np.mean(1/r),'median_rank':np.median(r)},pd.DataFrame(records)
def sample_retrieval():
 out=OUT/'sample_retrieval';out.mkdir(parents=True,exist_ok=True);m=pd.read_parquet(PREV/'sample_manifest.parquet').sort_values('matrix_row').reset_index(drop=True);rows=m.matrix_row.to_numpy(int)
 raw=np.memmap(HWORK/'archs4_log1p_tpm.float32.mmap',mode='r',dtype='float32',shape=(40000,G))[rows];pca=joblib.load(HWORK/'pca512.pkl').transform(raw);bridge=np.load(HWORK/'archs4_bridgerna_embeddings.npy',mmap_mode='r')[rows]
 meanstd=np.concatenate([np.load(LCACHE/'layer_12__mean.float32.npy',mmap_mode='r'),np.load(LCACHE/'layer_12__std.float32.npy',mmap_mode='r')],1);hall=np.load(TAG/'tissue__layer12__hallmark_module_mean_std.float16.npy',mmap_mode='r')
 reps={'Raw expression':raw,'PCA-15165':pca,'BridgeRNA global mean':bridge,'BridgeRNA layer12 mean+SD':meanstd,'BridgeRNA Hallmark mean+SD':hall};summary=[]
 for name,A in reps.items():
  say(f'sample retrieval {name}');metric,detail=(retrieval_metrics_chunked(A,m) if 'Hallmark' in name else retrieval_metrics(A,m));summary.append({'representation':name,**metric});detail.to_parquet(out/(name.lower().replace(' ','_').replace('+','plus')+'.parquet'),index=False)
 pd.DataFrame(summary).to_csv(out/'sample_retrieval_summary.csv',index=False);m[['gsm','gse','tissue']].to_csv(out/'sample_manifest.csv',index=False)
def response_assets():
 # Existing response vectors; all share canonical ortholog order where applicable.
 em=pd.read_parquet(EX/'results/contrast_members.parquet');manifest=pd.read_parquet(EX/'results/matched_manifest.parquet').reset_index(names='sample_index');em=em.merge(manifest[['GSM','sample_index']],on='GSM');exids=pd.read_csv(EX/'results/response_contrasts.csv').contrast_id.tolist();exx=np.load(EX/'work/matched_log1p_tpm_corrected.npy',mmap_mode='r');exz=np.load(EX/'work/response_effects_bridgerna.npy')
 def response(matrix,q,idcol='contrast_id'):
  ans=[]
  for cid in exids:
   d=q[q[idcol].eq(cid)];role=d.role.str.contains('post|exercise',case=False,regex=True);a=d[role].sample_index.to_numpy(int);b=d[~role].sample_index.to_numpy(int);ans.append(np.asarray(matrix[a]).mean(0)-np.asarray(matrix[b]).mean(0))
  return np.stack(ans)
 raw=response(exx,em);pca_model=joblib.load(HWORK/'pca512.pkl');pca=raw@pca_model.components_.T
 return exids,raw,pca,exz
def ranking(A,ids,family):
 A=cosmat(A);S=A@A.T;np.fill_diagonal(S,-np.inf);rows=[];ranks=[]
 for i,cid in enumerate(ids):
  order=np.argsort(-S[i]);hits=np.flatnonzero(np.array(family)[order]==family[i]);rank=int(hits[0]+1) if len(hits) else len(ids)+1;ranks.append(rank)
  for r,j in enumerate(order[:5],1):rows.append({'query':cid,'rank':r,'retrieved':ids[j],'similarity':S[i,j],'query_family':family[i],'retrieved_family':family[j],'correct_family':family[i]==family[j]})
 r=np.array(ranks);return {'queries':len(ids),'R@1':np.mean(r<=1),'R@5':np.mean(r<=5),'R@10':np.mean(r<=10),'MRR':np.mean(1/r),'median_rank':np.median(r)},pd.DataFrame(rows)
def response_retrieval():
 out=OUT/'response_retrieval';rer=OUT/'reranking';out.mkdir(parents=True,exist_ok=True);rer.mkdir(parents=True,exist_ok=True);ids,raw,pca,bridge=response_assets();axisA={'human_GSE108643','human_GSE86931','mouse_GSE126962','mouse_GSE132520'};axisB={'human_GSE71972','human_GSE87748','mouse_GSE97718'};family=['exploratory_axis_A' if x in axisA else 'exploratory_axis_B' if x in axisB else 'intermediate' for x in ids]
 # Context fingerprint is the full per-gene graph-turnover vector, not giant token concatenation.
 d=pd.read_parquet(REPO/'benchmarks/contextual_gene_network_utility/results/final_decision/per_gene_metrics.parquet');context=np.stack([d[d.contrast_id.eq(x)].sort_values('gene').graph_neighborhood_turnover.to_numpy() for x in ids])
 # Existing predicted Hallmark delta, if complete for these studies.
 hp=EX/'results/hallmark_response_axes/predicted_hallmark_deltas.parquet';h=pd.read_parquet(hp);num=h.select_dtypes('number').columns.tolist();hall=np.stack([h.loc[x.split('_')[-1],num].to_numpy() for x in ids])
 reps={'Raw expression':raw,'PCA-15165':pca,'BridgeRNA global mean':bridge,'Hallmark response fingerprint':hall,'Contextual gene-response fingerprint':context};summary=[];details={}
 for name,A in reps.items():
  metric,detail=ranking(A,ids,family);summary.append({'representation':name,'ground_truth':'previous exploratory Axis A/B; not independent of BridgeRNA discovery',**metric});detail['representation']=name;details[name]=detail
 pd.DataFrame(summary).to_csv(out/'exercise_response_retrieval_summary.csv',index=False);pd.concat(details.values()).to_csv(out/'exercise_response_top5.csv',index=False)
 # Transparent full-corpus rerank. K values collapse because only 8 candidates exist.
 base=cosmat(hall)@cosmat(hall).T;ctx=cosmat(context)@cosmat(context).T;comp=.5*base+.5*ctx;np.fill_diagonal(comp,-np.inf);rr=[]
 for i,q in enumerate(ids):
  order=np.argsort(-comp[i])
  for r,j in enumerate(order[:5],1):rr.append({'query':q,'rank':r,'retrieved':ids[j],'composite':comp[i,j],'hallmark_similarity':base[i,j],'context_similarity':ctx[i,j],'same_exploratory_family':family[i]==family[j]})
 pd.DataFrame(rr).to_csv(rer/'transparent_contextual_reranking_top5.csv',index=False);pd.DataFrame([{'requested_K':k,'effective_candidates':len(ids)-1,'distinct_stage_test':False} for k in [25,50,100]]).to_csv(rer/'candidate_depth_audit.csv',index=False)
 ranks=[]
 for i in range(len(ids)):
  order=np.argsort(-comp[i]);hit=np.flatnonzero(np.array(family)[order]==family[i]);ranks.append(int(hit[0]+1) if len(hit) else len(ids)+1)
 ranks=np.array(ranks);summary.append({'representation':'Transparent Hallmark+context reranking','ground_truth':'previous exploratory Axis A/B; not independent of BridgeRNA discovery','queries':len(ids),'R@1':np.mean(ranks<=1),'R@5':np.mean(ranks<=5),'R@10':np.mean(ranks<=10),'MRR':np.mean(1/ranks),'median_rank':np.median(ranks)})
 pd.DataFrame(summary).to_csv(out/'exercise_response_retrieval_summary.csv',index=False)
def technical_false_friend():
 out=OUT/'false_friend';out.mkdir(parents=True,exist_ok=True);v=pd.read_csv(FROZEN/'results/graph_fingerprint/rr1_rr3_stress/cross_representation_comparison.csv');g=pd.read_csv(FROZEN/'results/graph_fingerprint/rr1_rr3_stress/graph_response_stress_metrics.csv').query("k == 10 and graph == 'union'")
 v.to_csv(out/'vector_readout_reference.csv',index=False);g.to_csv(out/'contextual_graph_stress.csv',index=False)
 rows=[]
 for rep,q in v.groupby('representation'):
  z=q.set_index('comparison').similarity;rows.append({'representation':rep,'RR3_39_above_false_friend':z.get('RR3-39',-9)>z.get('false_friend',9),'RR3_40_above_false_friend':z.get('RR3-40',-9)>z.get('false_friend',9)})
 z=g.set_index('comparison').signed_edge_cosine;rows.append({'representation':'Contextual kNN graph','RR3_39_above_false_friend':z['RR3-39']>z['RR1↔RR3-39 false friend'],'RR3_40_above_false_friend':z['RR3-40']>z['RR1↔RR3-39 false friend']});pd.DataFrame(rows).to_csv(out/'false_friend_safety.csv',index=False)
def reuse_external():
 tp=OUT/'technical_pairs';tp.mkdir(parents=True,exist_ok=True);summary=pd.read_csv(REPO/'benchmarks/paired_recount3/results/benchmark_summary.csv');summary.to_csv(tp/'paired_recount3_existing_summary.csv',index=False);pd.DataFrame([{'representation':'contextual/Hallmark reranking','status':'unavailable','reason':'paired recount3 contextual tokens and Hallmark module fingerprints were not cached; no expensive regeneration authorized'}]).to_csv(tp/'contextual_reranking_status.csv',index=False)
 cs=OUT/'cross_species';cs.mkdir(parents=True,exist_ok=True);src=REPO/'benchmarks/mouse_encode/results/task1a_balanced_geometry/summary_results.csv';pd.read_csv(src).to_csv(cs/'existing_balanced_tissue_retrieval.csv',index=False)
def finalize():
 s=OUT/'summary';figdir=OUT/'figures';s.mkdir(parents=True,exist_ok=True);figdir.mkdir(parents=True,exist_ok=True);sample=pd.read_csv(OUT/'sample_retrieval/sample_retrieval_summary.csv');resp=pd.read_csv(OUT/'response_retrieval/exercise_response_retrieval_summary.csv');safety=pd.read_csv(OUT/'false_friend/false_friend_safety.csv');best_sample=sample.sort_values('MRR',ascending=False).iloc[0].representation;best_resp=resp.sort_values('MRR',ascending=False).iloc[0].representation
 contextual=resp.set_index('representation').loc['Contextual gene-response fingerprint','MRR'];rerank=resp.set_index('representation').loc['Transparent Hallmark+context reranking','MRR'];globalm=resp.set_index('representation').loc['BridgeRNA global mean','MRR'];decision='A. CONTEXTUAL RAG ADDS CLEAR VALUE' if rerank>globalm and best_sample not in ['Raw expression','PCA-15165'] else ('B. CONTEXTUAL RAG HELPS SELECTED TASKS' if rerank>globalm else 'C. GLOBAL BASELINES ARE SUFFICIENT')
 d={'decision':decision,'best_sample_retrieval':best_sample,'best_response_retrieval':best_resp,'contextual_response_MRR':float(contextual),'transparent_reranking_MRR':float(rerank),'global_mean_response_MRR':float(globalm),'recount3_contextual_reranking':'unavailable','reranking_K_test':'not distinct because response corpus <25','claim':'retrieved transcriptomic analogue, not mechanistic equivalence'};(s/'decision.json').write_text(json.dumps(d,indent=2)+'\n');pd.DataFrame([d]).to_csv(s/'decision.csv',index=False)
 fig,axes=plt.subplots(1,2,figsize=(13,5));
 for ax,data,title in [(axes[0],sample,'Study-excluded tissue retrieval'),(axes[1],resp,'Exercise-response retrieval')]:
  q=data.sort_values('MRR');ax.barh(q.representation,q.MRR,color='#4C78A8');ax.set_xlim(0,1);ax.set_xlabel('MRR');ax.set_title(title)
  for i,v in enumerate(q.MRR):ax.text(v+.01,i,f'{v:.3f}',va='center',fontsize=8)
 fig.tight_layout();fig.savefig(figdir/'retrieval_summary.png',dpi=350,bbox_inches='tight');fig.savefig(figdir/'retrieval_summary.pdf',bbox_inches='tight');plt.close(fig)
def main():
 for p in ['sample_retrieval','response_retrieval','reranking','technical_pairs','exercise_spaceflight','cross_species','false_friend','figures','summary']:(OUT/p).mkdir(parents=True,exist_ok=True)
 say('estimated cache-only runtime 3-8 minutes');sample_retrieval();response_retrieval();technical_false_friend();reuse_external();finalize();say('COMPLETE')
if __name__=='__main__':main()
