#!/usr/bin/env python3
"""Compare simple controlled library corrections on held-out Task 3 responses."""
from __future__ import annotations
import json
from datetime import datetime,timezone
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage,fcluster
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score,balanced_accuracy_score,f1_score,roc_auc_score,silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[1];OUT=ROOT/'results/task4_simple_correction_comparison';OUT.mkdir(parents=True,exist_ok=True)
T3=REPO/'benchmarks/osdr_batch_effect_representation';R3=T3/'results';W3=T3/'work';KS=(1,2,3,5)

def cos(a,b):
 d=np.linalg.norm(a)*np.linalg.norm(b);return float(np.dot(a,b)/d) if d else np.nan
def remove(X,B):return X-(X@B.T)@B
def condition(s):return 'FLT' if '_FLT_' in s else 'GC' if '_GC_' in s else 'other'
def fit_effects(m,z,exclude=None):
 q=m if exclude is None else m[~m.pair_id.eq(exclude)];D=[]
 for _,g in q.groupby('pair_id',sort=True):D.append(z[g.index[g.library_prep.eq('ribo')]].mean(0)-z[g.index[g.library_prep.eq('polyA')]].mean(0))
 D=np.stack(D);mean=D.mean(0);direction=mean/np.linalg.norm(mean);_,_,basis=np.linalg.svd(D,full_matrices=False);return mean,direction,basis
def transform(X,method,effects,labels=None):
 mean,direction,basis=effects
 if method=='none':return X.copy()
 if method=='mean_direction_projection':return remove(X,direction[None])
 if method.startswith('svd_pc1_'):return remove(X,basis[:int(method.rsplit('_',1)[1])])
 if method=='paired_additive_residualization':
  if labels is None:raise ValueError('library labels required for additive residualization')
  sign=np.asarray(pd.Series(labels).map({'polyA':-.5,'ribo':.5,'ribodepleted':.5}),float)
  if np.isnan(sign).any():raise ValueError(f'unmapped library labels: {sorted(set(pd.Series(labels)[np.isnan(sign)]))}')
  return X-sign[:,None]*mean
 raise KeyError(method)
def pair_metrics(m,z):
 p=[];r=[]
 for pid,g in m.groupby('pair_id',sort=True):
  a=z[g.index[g.library_prep.eq('polyA')]].mean(0);b=z[g.index[g.library_prep.eq('ribo')]].mean(0);p.append((pid,a,b));r.append((cos(a,b),np.linalg.norm(a-b)))
 A=np.stack([x[1] for x in p]);B=np.stack([x[2] for x in p]);A/=np.linalg.norm(A,axis=1,keepdims=True);B/=np.linalg.norm(B,axis=1,keepdims=True);S=A@B.T;ranks=np.r_[[1+(S[i]>S[i,i]).sum() for i in range(len(p))],[1+(S[:,i]>S[i,i]).sum() for i in range(len(p))]]
 return {'paired_cosine':np.median(np.array(r)[:,0]),'paired_euclidean':np.median(np.array(r)[:,1]),'pair_r1':np.mean(ranks<=1),'pair_r5':np.mean(ranks<=5),'pair_mrr':np.mean(1/ranks),'median_rank':np.median(ranks)}
def loo_probe(m,z):
 probs=[];truth=[];pred=[]
 for pid,g in m.groupby('pair_id',sort=True):
  test=g.index.to_numpy();train=m.index[~m.pair_id.eq(pid)].to_numpy();y=m.library_prep.map({'polyA':0,'ribo':1}).to_numpy();scale=StandardScaler().fit(z[train]);clf=LogisticRegression(max_iter=3000,class_weight='balanced',random_state=41721).fit(scale.transform(z[train]),y[train]);pr=clf.predict_proba(scale.transform(z[test]))[:,1];probs.extend(pr);truth.extend(y[test]);pred.extend(pr>=.5)
 auc=roc_auc_score(truth,probs);bacc=balanced_accuracy_score(truth,pred)
 return {'library_auroc':auc,'library_balanced_accuracy':bacc,'library_macro_f1':f1_score(truth,pred,average='macro'),
         'library_orientation_free_auroc':max(auc,1-auc),'library_accuracy_chance_proximity':1-2*abs(bacc-.5)}
def controlled_oof(m,z,method):
 out=np.empty_like(z)
 for pid,g in m.groupby('pair_id',sort=True):out[g.index]=transform(z[g.index],method,fit_effects(m,z,pid),g.library_prep.tolist())
 return out
def response(ids,z,index):
 f=[index[s] for s in ids if condition(s)=='FLT'];g=[index[s] for s in ids if condition(s)=='GC'];return z[f].mean(0)-z[g].mean(0)
def technical_vectors(z,index,design):
 return {x.representation:response(str(x.samples).split(' | '),z,index) for x in design.itertuples()}
COMPS={'RR1':('RR1_OSD48_original_matched','RR1_OSD168_no-ERCC'),'RR1_ERCC':('RR1_OSD48_original_matched','RR1_OSD168_all_ERCC'),'RR3-39':('C01_OSD137_original_matched','C01_OSD168_all_ERCC'),'RR3-40':('C02_OSD137_original_matched','C02_OSD168_all_ERCC'),'ERCC/no-ERCC':('RR1_OSD168_no-ERCC','RR1_OSD168_all_ERCC')}
def all_responses(z,index,members):return {cid:response(g.sample_id.tolist(),z,index) for cid,g in members.groupby('contrast_id',sort=False)}
def matrix(X):X=X/np.linalg.norm(X,axis=1,keepdims=True);return X@X.T
def response_geometry(vectors,order,labels,M0,original_vectors=None):
 X=np.stack([vectors[x] for x in order]);M=matrix(X);same=[];diff=[]
 for i in range(len(X)):
  for j in range(i+1,len(X)):(same if labels[i]==labels[j] else diff).append(M[i,j])
 D=np.clip(1-M,0,2);np.fill_diagonal(D,0);clusters=fcluster(linkage(squareform(D,checks=False),method='average'),2,criterion='maxclust');u=np.triu_indices(len(X),1)
 result={'same_mode_mean':np.mean(same),'different_mode_mean':np.mean(diff),'same_different_separation':np.mean(same)-np.mean(diff),'mode_silhouette':silhouette_score(X,labels,metric='cosine'),'mode_ARI':adjusted_rand_score(labels,clusters),'response_matrix_preservation':spearmanr(M[u],M0[u]).statistic}
 if original_vectors is not None and X.shape[1]==next(iter(original_vectors.values())).shape[0]:result['median_response_preservation']=np.median([cos(original_vectors[x],vectors[x]) for x in order])
 else:result['median_response_preservation']=np.nan
 return result,M
def neighbor_overlap(a,b,k=10):
 def nn(x):x=x/np.linalg.norm(x,axis=1,keepdims=True);s=x@x.T;np.fill_diagonal(s,-np.inf);return np.argpartition(-s,k,axis=1)[:,:k]
 A,B=nn(a),nn(b);return np.mean([len(set(x)&set(y))/k for x,y in zip(A,B)])
def pareto(df):
 cols=['controlled_pair_improvement','library_accuracy_chance_proximity','rr1_improvement','rr3_mean_cosine','response_matrix_preservation'];V=df[cols].fillna(-np.inf).to_numpy();optimal=[]
 for i,x in enumerate(V):optimal.append(not any(np.all(y>=x) and np.any(y>x) for j,y in enumerate(V) if i!=j))
 return optimal
def figures(summary,curves,mode_matrices,order,random5):
 fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained');q=curves[curves.method.str.startswith('svd_pc')|curves.method.eq('none')].copy();q['k']=q.method.map({'none':0,'svd_pc1_1':1,'svd_pc1_2':2,'svd_pc1_3':3,'svd_pc1_5':5})
 for c,color in [('RR1','#e15759'),('RR3-39','#4e79a7'),('RR3-40','#59a14f')]:ax.plot(q.k,q[c],marker='o',label=c,color=color)
 ax.axhline(0,color='black',lw=.7);ax.set(xticks=[0,1,2,3,5],xlabel='Controlled SVD dimensions removed',ylabel='Technical-replication cosine',title='Correction versus technical-replication preservation');ax.legend()
 for ext in ['png','pdf']:fig.savefig(OUT/f'figure_a_correction_curves.{ext}',dpi=400,bbox_inches='tight')
 plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,5),layout='constrained');s=summary[summary.method.isin(['none','mean_direction_projection','svd_pc1_1','svd_pc1_2','svd_pc1_3','svd_pc1_5','paired_additive_residualization'])]
 ax.scatter(s.response_matrix_preservation,s.rr1_improvement,c=np.where(s.pareto_optimal,'#e15759','#4e79a7'),s=60)
 for x in s.itertuples():ax.annotate(x.method.replace('svd_pc1_','PC').replace('mean_direction_projection','mean direction').replace('paired_additive_residualization','paired additive'),(x.response_matrix_preservation,x.rr1_improvement),fontsize=8,xytext=(3,3),textcoords='offset points')
 ax.set(xlabel='Task 3 response-matrix preservation',ylabel='RR1 cosine improvement',title='Pareto view: correction–preservation tradeoff')
 for ext in ['png','pdf']:fig.savefig(OUT/f'figure_b_pareto_tradeoff.{ext}',dpi=400,bbox_inches='tight')
 plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(17,5.5),layout='constrained')
 for ax,(name,M) in zip(axes,mode_matrices.items()):im=ax.imshow(M,cmap='RdBu_r',vmin=-1,vmax=1);ax.set(title=name,xticks=[],yticks=range(len(order)),yticklabels=[x.split('__')[0] for x in order]);fig.colorbar(im,ax=ax,fraction=.046)
 for ext in ['png','pdf']:fig.savefig(OUT/f'figure_c_selected_response_matrices.{ext}',dpi=400,bbox_inches='tight')
 plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,4),layout='constrained');obs=summary.set_index('method').loc['svd_pc1_5','RR1'];ax.hist(random5.rr1_cosine,bins=35,color='#bab0ac');ax.axvline(obs,color='#e15759',lw=2,label=f'controlled PC1–5: {obs:.3f}');ax.set(xlabel='RR1 cosine',ylabel='Random subspaces',title='Matched random-subspace control');ax.legend()
 for ext in ['png','pdf']:fig.savefig(OUT/f'figure_d_random_control.{ext}',dpi=400,bbox_inches='tight')
 plt.close(fig)
def main():
 cp=ROOT/'work/datasets/chen_2020_tcells';cm=pd.read_parquet(cp/'manifest.parquet').reset_index(drop=True);cz=np.load(cp/'bridgerna_embeddings.npy').astype(float);effects=fit_effects(cm,cz)
 methods=['none','mean_direction_projection']+[f'svd_pc1_{k}' for k in KS]+['paired_additive_residualization']
 controlled={}
 for method in methods:
  x=controlled_oof(cm,cz,method);controlled[method]=x
 # FE is fixed from the existing fit; its controlled metrics are apparent/training-cohort, not correction-LODO.
 rep=np.load(ROOT/'results/task4_disentanglement/sample_representations.npz',allow_pickle=True);ids=list(rep['sample_id']);loc=[ids.index(s) for s in cm.sample_id];controlled['FE_existing']=rep['FE'][loc]
 crows=[];base=pair_metrics(cm,controlled['none'])
 for method,x in controlled.items():crows.append({'method':method,**pair_metrics(cm,x),**loo_probe(cm,x),'correction_validation':'LODO correction + LODO probe' if method!='FE_existing' else 'existing FE fit on this cohort; LODO probe only'})
 ctab=pd.DataFrame(crows);ctab['controlled_pair_improvement']=ctab.paired_cosine-base['paired_cosine'];ctab['controlled_distance_reduction']=base['paired_euclidean']-ctab.paired_euclidean
 # OSDR held-out transformations.
 tm=pd.read_csv(R3/'sample_manifest.csv');tz=np.load(W3/'bridgerna_embeddings.npy').astype(float);idx=dict(zip(tm.sample_id,range(len(tm))));design=pd.read_csv(R3/'task3_osd168_technical_replication/technical_response_design.csv');members=pd.read_csv(R3/'task3b_contrast_sample_membership.csv');modes=pd.read_csv(R3/'task3c_cluster_assignments.csv').sort_values('heatmap_order');order=modes.contrast_id.tolist();labels=modes.geometry_cluster.to_numpy();orig=all_responses(tz,idx,members);M0=matrix(np.stack([orig[x] for x in order]));rows=[];matrices={}
 for method in methods:
  corrected=transform(tz,method,effects,tm.library_preparation.tolist());tech=technical_vectors(corrected,idx,design);vals={name:cos(tech[a],tech[b]) for name,(a,b) in COMPS.items()};allv=all_responses(corrected,idx,members);geo,M=response_geometry(allv,order,labels,M0,orig);sample_cos=np.median([cos(a,b) for a,b in zip(tz,corrected)]);top10=neighbor_overlap(tz,corrected)
  rows.append({'method':method,**vals,**geo,'sample_cosine_preservation':sample_cos,'top10_neighbor_overlap':top10});
  if method in ['none','svd_pc1_1','svd_pc1_5']:matrices[method]=M
 # Existing FE Task 3 representation.
 f=np.load(ROOT/'results/task4g_task3_challenge/task3_fe_re_sample_embeddings.npz',allow_pickle=True);fe=f['FE'];tech=technical_vectors(fe,idx,design);vals={name:cos(tech[a],tech[b]) for name,(a,b) in COMPS.items()};allv=all_responses(fe,idx,members);geo,M=response_geometry(allv,order,labels,M0,None);rows.append({'method':'FE_existing',**vals,**geo,'sample_cosine_preservation':np.nan,'top10_neighbor_overlap':neighbor_overlap(tz,fe)});matrices['FE_existing']=M
 rtab=pd.DataFrame(rows);summary=ctab.merge(rtab,on='method');summary['rr1_improvement']=summary.RR1-summary.loc[summary.method.eq('none'),'RR1'].iloc[0];summary['rr3_39_change']=summary['RR3-39']-summary.loc[summary.method.eq('none'),'RR3-39'].iloc[0];summary['rr3_40_change']=summary['RR3-40']-summary.loc[summary.method.eq('none'),'RR3-40'].iloc[0];summary['rr3_mean_cosine']=summary[['RR3-39','RR3-40']].mean(axis=1);summary['pareto_optimal']=pareto(summary)
 # Existing matched random controls.
 random=pd.read_parquet(ROOT/'results/task4_response_robustness/random_subspace_metrics.parquet');rsummary=pd.read_csv(ROOT/'results/task4_response_robustness/random_subspace_summary.csv')
 summary.to_csv(OUT/'correction_tradeoff_summary.csv',index=False);ctab.to_csv(OUT/'controlled_pair_metrics.csv',index=False);rtab.to_csv(OUT/'osdr_response_metrics.csv',index=False);rsummary.to_csv(OUT/'random_subspace_comparison.csv',index=False)
 curves=rtab[rtab.method.isin(['none']+[f'svd_pc1_{k}' for k in KS])][['method','RR1','RR3-39','RR3-40','response_matrix_preservation','median_response_preservation','mode_ARI','mode_silhouette']];curves.to_csv(OUT/'svd_correction_curve.csv',index=False)
 figures(summary,curves,matrices,order,random[random.removed_components.eq(5)])
 prov={'created_utc':datetime.now(timezone.utc).isoformat(),'bridge_retrained':False,'FE_RE_retrained':False,'OSDR_used_for_fit_or_tuning':False,'controlled_correction_validation':'leave-one-donor-out','paired_additive_definition':'subtract (library_indicator - 0.5) times paired mean Ribo-minus-PolyA effect','important_identity':'study-constant additive residualization cancels exactly in within-study FLT-minus-GC responses','probe_interpretation':'AUROC below 0.5 is retained as signed out-of-fold performance; orientation-free AUROC and chance proximity prevent systematic inversion from being mislabeled as removal','pareto_objectives':['pair cosine improvement','library accuracy chance proximity','RR1 improvement','RR3 mean cosine','response matrix preservation']};(OUT/'provenance.json').write_text(json.dumps(prov,indent=2))
 print(summary[['method','paired_cosine','paired_euclidean','pair_r1','library_auroc','library_orientation_free_auroc','library_balanced_accuracy','library_accuracy_chance_proximity','RR1','RR3-39','RR3-40','response_matrix_preservation','median_response_preservation','mode_ARI','mode_silhouette','sample_cosine_preservation','top10_neighbor_overlap','pareto_optimal']].to_string(index=False))
if __name__=='__main__':main()
