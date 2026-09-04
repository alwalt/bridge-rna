#!/usr/bin/env python3
"""Compare full-transcriptome linear baselines with frozen BridgeRNA Task 3 geometry."""
from __future__ import annotations
import gzip,itertools,json,re
from datetime import datetime,timezone
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage,fcluster
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score,silhouette_score

HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parents[1];R=HERE/'results';W=HERE/'work';O=R/'task3_representation_comparison_full_vocab';O.mkdir(parents=True,exist_ok=True)
STUDIES=[47,48,137,168,173,242,245]
def cosine(a,b):return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
def counts_path(study):
 x=sorted((ROOT/'data/osdr/raw').glob(f'GLDS-{study}_*.csv'))
 if len(x)!=1:raise RuntimeError(f'Expected one count table for OSD-{study}: {x}')
 return x[0]

def ensembl_symbol_map():
 out={};pat=re.compile(r'(\w+) "([^"]+)"')
 with gzip.open(ROOT/'data/gencode/gencode.vM38.basic.annotation.gtf.gz','rt') as f:
  for line in f:
   if line.startswith('#'):continue
   fields=line.rstrip().split('\t')
   if len(fields)>8 and fields[2]=='gene':
    attrs=dict(pat.findall(fields[8]));gid=attrs.get('gene_id','').split('.')[0];sym=attrs.get('gene_name','')
    if gid and sym:out[gid]=sym
 return out

def build_full_log1p_tpm(manifest):
 cache=O/'full_common_log1p_tpm.npy';genes_file=O/'full_common_genes.csv'
 if cache.exists() and genes_file.exists():return np.load(cache),pd.read_csv(genes_file).gene_id.tolist()
 ens2sym=ensembl_symbol_map();lengths=pd.read_csv(ROOT/'data/gencode/gencode_v49_mouse_gene_exon_lengths.csv').drop_duplicates('gene_symbol').set_index('gene_symbol').exon_length.to_dict();blocks=[]
 for study in STUDIES:
  ids=manifest.loc[manifest.study_number.eq(study),'sample_id'].tolist();d=pd.read_csv(counts_path(study),usecols=lambda c:c=='Unnamed: 0' or c in ids);g=d.columns[0];d[g]=d[g].astype(str).str.split('.').str[0];d=d.groupby(g,sort=False).sum();lens=pd.Series({x:lengths.get(ens2sym.get(x,''),np.nan) for x in d.index});keep=lens.notna()&(lens>0);d=d.loc[keep];rate=d.div(lens[keep]/1000,axis=0);tpm=rate.div(rate.sum(axis=0),axis=1)*1e6;blocks.append(tpm.T)
 common=sorted(set.intersection(*(set(x.columns) for x in blocks)));x=pd.concat([b[common] for b in blocks]).loc[manifest.sample_id];arr=np.log1p(x.to_numpy(np.float64)).astype(np.float32);np.save(cache,arr)
 pd.DataFrame({'gene_id':common,'gene_symbol':[ens2sym[g] for g in common]}).to_csv(genes_file,index=False);return arr,common

def fit_pca(name,x):
 max_pc=min(x.shape[0]-1,x.shape[1]);p=PCA(n_components=max_pc,svd_solver='full').fit(x);scores=p.transform(x).astype(np.float32);np.save(O/f'{name}_pca_scores.npy',scores)
 v=pd.DataFrame({'PC':np.arange(1,max_pc+1),'variance_explained':p.explained_variance_ratio_,'cumulative_variance':np.cumsum(p.explained_variance_ratio_)});v.to_csv(O/f'{name}_pca_variance.csv',index=False);return scores,v
def response(matrix,ids,members,contrasts):
 pos={s:i for i,s in enumerate(ids)};out={}
 for cid in contrasts:
  d=members[members.contrast_id.eq(cid)];f=[pos[x] for x in d.loc[d.condition.eq('FLT'),'sample_id']];g=[pos[x] for x in d.loc[d.condition.eq('GC'),'sample_id']];out[cid]=matrix[f].mean(0)-matrix[g].mean(0)
 return out
def technical_vectors(matrix,ids):
 pos={s:i for i,s in enumerate(ids)};design=pd.read_csv(R/'task3_osd168_technical_replication/technical_response_design.csv');out={}
 for x in design.itertuples():
  samples=x.samples.split(' | ');f=[pos[s] for s in samples if '_FLT_' in s];g=[pos[s] for s in samples if '_GC_' in s];out[x.representation]=matrix[f].mean(0)-matrix[g].mean(0)
 return out

def geometry_metrics(name,standard,technical,meta):
 pairs={'RR1_noERCC':('RR1_OSD48_original_matched','RR1_OSD168_no-ERCC'),'RR1_ERCC':('RR1_OSD48_original_matched','RR1_OSD168_all_ERCC'),'RR1_168_no_vs_ERCC':('RR1_OSD168_no-ERCC','RR1_OSD168_all_ERCC'),
 'RR3_39':('C01_OSD137_original_matched','C01_OSD168_all_ERCC'),'RR3_40':('C02_OSD137_original_matched','C02_OSD168_all_ERCC')};rows=[]
 for label,(a,b) in pairs.items():rows.append({'representation':name,'comparison':label,'cosine':cosine(technical[a],technical[b]),'spearman':spearmanr(technical[a],technical[b]).statistic,'magnitude_a':np.linalg.norm(technical[a]),'magnitude_b':np.linalg.norm(technical[b]),'euclidean_distance':np.linalg.norm(technical[a]-technical[b]),'reversal':cosine(technical[a],technical[b])<0})
 tech=pd.DataFrame(rows)
 ribo=meta[meta.library_preparation.eq('ribodepleted')].sort_values('heatmap_order');names=ribo.contrast_id.tolist();X=np.stack([standard[n] for n in names]);M=X@X.T/np.outer(np.linalg.norm(X,axis=1),np.linalg.norm(X,axis=1));labels=ribo.geometry_cluster.to_numpy();vals=[]
 for i,j in itertools.combinations(range(len(names)),2):vals.append({'relation':'same_mode' if labels[i]==labels[j] else 'different_mode','cosine':M[i,j]})
 s=pd.DataFrame(vals).groupby('relation').cosine.agg(['count','mean','median','std']);sil=silhouette_score(X,labels,metric='cosine');cl=fcluster(linkage(squareform(1-M,checks=False),method='average'),2,criterion='maxclust');ari=adjusted_rand_score(labels,cl)
 pd.DataFrame(M,index=names,columns=names).to_csv(O/f'{name}_ribodepleted_cosine.csv')
 summary={'representation':name,'RR1_preservation':tech.set_index('comparison').loc['RR1_noERCC','cosine'],'RR1_ERCC_preservation':tech.set_index('comparison').loc['RR1_ERCC','cosine'],'OSD168_noERCC_vs_ERCC':tech.set_index('comparison').loc['RR1_168_no_vs_ERCC','cosine'],
  'RR3_39_preservation':tech.set_index('comparison').loc['RR3_39','cosine'],'RR3_40_preservation':tech.set_index('comparison').loc['RR3_40','cosine'],'same_mode_cosine':s.loc['same_mode','mean'],'same_mode_median':s.loc['same_mode','median'],'different_mode_cosine':s.loc['different_mode','mean'],'different_mode_median':s.loc['different_mode','median'],'same_minus_different':s.loc['same_mode','mean']-s.loc['different_mode','mean'],'silhouette':sil,'ARI':ari}
 return tech,summary,names,M

def plot_heatmaps(results,order):
 fig,axes=plt.subplots(2,3,figsize=(17,10),layout='constrained');titles={'full_expression':'Full-vocabulary expression','full_pca20':'Full-vocabulary PCA20','full_pca_max':'Full-vocabulary PCA111','bridge_vocab_pca20':'15,165-gene PCA20','bridge_vocab_pca_max':'15,165-gene PCA111','BridgeRNA512':'BridgeRNA512'}
 for ax,(name,(_,_,names,M)) in zip(axes.flat,results.items()):
  im=ax.imshow(M,cmap='RdBu_r',vmin=-1,vmax=1);labs=[f"M{order.set_index('contrast_id').loc[x,'geometry_cluster']} {x[:3]} {order.set_index('contrast_id').loc[x,'OSD']}" for x in names];ax.set(xticks=range(len(labs)),yticks=range(len(labs)),xticklabels=labs,yticklabels=labs,title=titles[name]);ax.tick_params(axis='x',rotation=65,labelsize=7);ax.tick_params(axis='y',labelsize=7)
 fig.colorbar(im,ax=axes.ravel().tolist(),label='FLT−GC response cosine',shrink=.75)
 for ext in ['png','pdf']:fig.savefig(O/f'all_representation_ribodepleted_heatmaps.{ext}',dpi=400,bbox_inches='tight')
 plt.close(fig)

def main():
 manifest=pd.read_csv(R/'sample_manifest.csv');ids=manifest.sample_id.tolist();members=pd.read_csv(R/'task3b_contrast_sample_membership.csv');contrasts=pd.read_csv(R/'task3b_contrast_summary.csv').contrast_id.tolist();meta=pd.read_csv(R/'task3c_cluster_assignments.csv')
 full,full_genes=build_full_log1p_tpm(manifest);bridge_expr=np.load(W/'bridgerna_log1p_tpm_inputs.npy');bridge=np.load(W/'bridgerna_embeddings.npy');assert bridge_expr.shape==(112,15165)
 full_pc,full_var=fit_pca('full_vocab',full);bridge_pc,bridge_var=fit_pca('bridge_vocab',bridge_expr)
 reps={'full_expression':full,'full_pca20':full_pc[:,:20],'full_pca_max':full_pc,'bridge_vocab_pca20':bridge_pc[:,:20],'bridge_vocab_pca_max':bridge_pc,'BridgeRNA512':bridge};results={};alltech=[];summary=[]
 for name,x in reps.items():
  st=response(x,ids,members,contrasts);tv=technical_vectors(x,ids);tech,s,names,M=geometry_metrics(name,st,tv,meta);results[name]=(tech,s,names,M);alltech.append(tech);summary.append(s)
 pd.concat(alltech).to_csv(O/'technical_replication_metrics_all_representations.csv',index=False)
 info={'full_expression':(len(full_genes),full.shape[1]),'full_pca20':(len(full_genes),20),'full_pca_max':(len(full_genes),full_pc.shape[1]),'bridge_vocab_pca20':(15165,20),'bridge_vocab_pca_max':(15165,bridge_pc.shape[1]),'BridgeRNA512':(15165,512)}
 table=pd.DataFrame(summary);table.insert(1,'Genes',[info[x][0] for x in table.representation]);table.insert(2,'Dimensions',[info[x][1] for x in table.representation]);table.to_csv(O/'main_representation_comparison.csv',index=False);plot_heatmaps(results,meta)
 # All-PC PCA must preserve centered expression response inner products because every contrast vector is in the centered sample row-span.
 checks=[]
 for prefix,a,b in [('full',results['full_expression'][3],results['full_pca_max'][3]),('bridge_vocab',None,None)]:
  if prefix=='bridge_vocab':
   raw=response(bridge_expr,ids,members,contrasts);ribo=meta[meta.library_preparation.eq('ribodepleted')].sort_values('heatmap_order');X=np.stack([raw[n] for n in ribo.contrast_id]);a=X@X.T/np.outer(np.linalg.norm(X,axis=1),np.linalg.norm(X,axis=1));b=results['bridge_vocab_pca_max'][3]
  checks.append({'space':prefix,'max_abs_cosine_difference_expression_vs_all_PC':np.max(np.abs(a-b))})
 pd.DataFrame(checks).to_csv(O/'all_pc_cosine_preservation_check.csv',index=False)
 pca_summary=pd.DataFrame([{'vocabulary':'full','genes':len(full_genes),'samples':len(ids),'maximum_nonzero_PCs':len(ids)-1,'PCA20_actual':20,'PCA20_cumulative_variance':full_var.iloc[19].cumulative_variance,'high_dim_actual':len(full_pc[0]),'high_dim_cumulative_variance':full_var.iloc[-1].cumulative_variance},
  {'vocabulary':'BridgeRNA','genes':15165,'samples':len(ids),'maximum_nonzero_PCs':len(ids)-1,'PCA20_actual':20,'PCA20_cumulative_variance':bridge_var.iloc[19].cumulative_variance,'high_dim_actual':len(bridge_pc[0]),'high_dim_cumulative_variance':bridge_var.iloc[-1].cumulative_variance}]);pca_summary.to_csv(O/'pca_dimensionality_and_variance.csv',index=False)
 # Classification follows reproducibility first, not clustering alone.
 br=table.set_index('representation').loc['BridgeRNA512'];linear=table.set_index('representation').loc['full_pca_max'];classification='D_Bridge_stronger_geometry_but_worse_RR1_technical_reproducibility' if br.same_minus_different>linear.same_minus_different and br.RR1_preservation<linear.RR1_preservation else 'A_full_vocabulary_high_dimensional_PCA_reproduces_key_geometry'
 prov={'created_utc':datetime.now(timezone.utc).isoformat(),'full_common_genes':len(full_genes),'bridge_genes':15165,'PCA_fit_samples':len(ids),'maximum_nonzero_PCs':len(ids)-1,'classification':classification,'PCA_fit':'once per sample-level expression matrix; contrasts computed after transform','normalization':'species-correct mouse exon-length TPM followed by natural log1p; no batch correction'};(O/'provenance.json').write_text(json.dumps(prov,indent=2));print(pca_summary.to_string(index=False));print('\n',table.to_string(index=False));print('\n',pd.DataFrame(checks).to_string(index=False));print('\nClassification:',classification)
if __name__=='__main__':main()
