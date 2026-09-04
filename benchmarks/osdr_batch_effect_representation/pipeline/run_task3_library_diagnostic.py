#!/usr/bin/env python3
"""Diagnose whether Task 3 response modes are explained by library/batch structure."""
from __future__ import annotations
import itertools,json,re
from datetime import datetime,timezone
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage,fcluster
from scipy.spatial.distance import squareform
from scipy.stats import fisher_exact,spearmanr
from sklearn.metrics import adjusted_rand_score

ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results';W=ROOT/'work';O=R/'task3_library_diagnostic';O.mkdir(parents=True,exist_ok=True)
def cos(a,b):return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
def aid(s):
 m=re.search(r'_(M\d+)$',str(s));return m.group(1) if m else ''
def cond(s):return 'FLT' if '_FLT_' in s else ('GC' if '_GC_' in s else '')
def num(s):
 m=re.search(r'[\d.]+',str(s));return float(m.group()) if m else np.nan

def technical_metadata():
 rows=[]
 for osd,label,subset in [('48','OSD-48 original RR-1','RR1'),('168','OSD-168 RR-1 no-ERCC','no-ERCC'),('168','OSD-168 RR-1 ERCC','ERCC'),('137','OSD-137 original RR-3','RR3'),('168','OSD-168 RR-3 ERCC','RR3')]:
  d=pd.read_csv(R/f'task3_manifest_api_validation/api_OSD-{osd}_sample_metadata.csv',dtype=str,low_memory=False).fillna('')
  d=d[d['investigation.study assays.study assay technology type'].str.contains('RNA Sequencing',case=False,na=False)].drop_duplicates('id.sample name')
  if osd=='168' and 'RR-1' in label:
   d=d[d['id.sample name'].str.contains('RR1')];d=d[d['study.factor value.spike-in quality control'].eq('Without Spike-in' if subset=='no-ERCC' else 'With Spike-in')]
  if osd=='168' and 'RR-3' in label:d=d[d['id.sample name'].str.contains('RR3')]
  depth=pd.to_numeric(d['assay.parameter value.read depth'].str.extract(r'([\d.]+)')[0],errors='coerce')
  def vals(c):return ' | '.join(sorted(set(d[c]))) if c in d else 'not reported'
  rows.append({'representation':label,'OSD':f'OSD-{osd}','library_selection':vals('assay.parameter value.library selection'),'library_kit':vals('assay.parameter value.library kit'),
   'layout':vals('assay.parameter value.library layout'),'read_length':vals('assay.parameter value.read length'),'read_depth_min':depth.min(),'read_depth_median':depth.median(),'read_depth_max':depth.max(),
   'instrument':vals('assay.parameter value.sequencing instrument'),'facility':'UC Davis','ERCC':('none declared' if osd in ['48','137'] else vals('study.factor value.spike-in quality control')),
   'ERCC_mix':('not applicable' if osd in ['48','137'] or subset=='no-ERCC' else vals('assay.parameter value.spike-in mix number')),'samples':len(d)})
 out=pd.DataFrame(rows);out.to_csv(O/'authoritative_technical_metadata.csv',index=False);return out

def rr1_shift():
 m=pd.read_csv(R/'sample_manifest.csv');z=np.load(W/'bridgerna_embeddings.npy');idx={s:i for i,s in enumerate(m.sample_id)}
 c=pd.read_csv(R/'task3_osd168_technical_replication/biological_sample_correspondence.csv');c=c[(c.RR_mission=='RR1')&c.exact_animal_match&c.group.isin(['FLT','GC'])]
 variants={'no-ERCC':c[c.ERCC_condition.eq('no-ERCC')],'ERCC':c[c.ERCC_condition.ne('no-ERCC')]}
 rows=[];shifts=[]
 for tech,d in variants.items():
  paired=d.drop_duplicates('animal_id');orig=np.stack([z[idx[s]] for s in paired.source_sample]);new=np.stack([z[idx[s]] for s in paired.OSD168_sample]);delta=new-orig
  mean_shift=delta.mean(0);response_shift=delta[np.array(paired.group)=='FLT'].mean(0)-delta[np.array(paired.group)=='GC'].mean(0)
  _,_,vt=np.linalg.svd(delta-delta.mean(0),full_matrices=False);pc1=vt[0]
  shifts.append({'technical_condition':tech,'mean_shift':mean_shift,'response_shift':response_shift,'pc1':pc1})
  rows.append({'technical_condition':tech,'matched_animals':len(paired),'FLT_animals':sum(paired.group=='FLT'),'GC_animals':sum(paired.group=='GC'),'mean_sample_shift_norm':np.linalg.norm(mean_shift),
   'response_change_norm':np.linalg.norm(response_shift),'response_change_vs_mean_technical_shift_cosine':cos(response_shift,mean_shift),'response_change_vs_paired_shift_PC1_abs_cosine':abs(cos(response_shift,pc1)),
   'note':'paired same-animal estimate; PC1 is descriptive and fitted on these pairs'})
 out=pd.DataFrame(rows);out.to_csv(O/'rr1_technical_shift_alignment.csv',index=False);return out

def response_diagnostics():
 comp=pd.read_csv(R/'task3_osd168_technical_replication/original_vs_osd168_response_similarity.csv');v=np.load(R/'task3_osd168_technical_replication/technical_response_vectors.npz',allow_pickle=True);vectors=dict(zip(v['names'],v['delta_z']))
 rows=[]
 for name in ['RR1_OSD48_original_matched','RR1_OSD168_no-ERCC','RR1_OSD168_all_ERCC']:
  x=vectors[name];r={'representation':name,'response_norm':np.linalg.norm(x)}
  dirs=np.load(R/'task3d_mode_ig/mode_response_directions.npz');r.update({'mode1_projection':cos(x,dirs['mode_1']),'mode2_projection':cos(x,dirs['mode_2'])});r['fixed_mode_assignment']=1 if r['mode1_projection']>r['mode2_projection'] else 2;rows.append(r)
 out=pd.DataFrame(rows);out.to_csv(O/'rr1_response_magnitude_and_mode.csv',index=False)
 direct=[]
 for a,b in itertools.combinations(out.representation,2):
  direct.append({'representation_a':a,'representation_b':b,'cosine':cos(vectors[a],vectors[b]),'spearman':spearmanr(vectors[a],vectors[b]).statistic})
 pd.DataFrame(direct).to_csv(O/'rr1_response_pairwise.csv',index=False);return out

def all_contrast_associations():
 meta=pd.read_csv(R/'task3c_cluster_assignments.csv');v=np.load(R/'task3b_bridgerna_response_vectors.npz',allow_pickle=True);vec=dict(zip(v['contrast_id'],v['delta_z']));dirs=np.load(R/'task3d_mode_ig/mode_response_directions.npz');sep=dirs['mode_1']-dirs['mode_2'];sep/=np.linalg.norm(sep)
 meta['mode_position']=[np.dot(vec[x],sep) for x in meta.contrast_id];meta['layout']=meta.sequencing_parameters.str.extract(r'^(SE|PE)');meta['read_length_bp']=pd.to_numeric(meta.sequencing_parameters.str.extract(r'(\d+)bp')[0]);meta['depth_M']=pd.to_numeric(meta.sequencing_parameters.str.extract(r'(\d+)M')[0]);meta.to_csv(O/'all_contrast_technical_mode_positions.csv',index=False)
 rows=[]
 for var in ['library_preparation','layout','read_length_bp','depth_M','sequencing_facility']:
  g=meta.groupby(var).agg(contrasts=('contrast_id','size'),mode1=('geometry_cluster',lambda s:int((s==1).sum())),mode2=('geometry_cluster',lambda s:int((s==2).sum())),mean_mode_position=('mode_position','mean'),sd_mode_position=('mode_position','std')).reset_index();g.insert(0,'variable',var);g=g.rename(columns={var:'level'});rows.append(g)
 assoc=pd.concat(rows,ignore_index=True);assoc.to_csv(O/'technical_variable_mode_associations.csv',index=False)
 tab=pd.crosstab(meta.library_preparation,meta.geometry_cluster);odds,p=fisher_exact(tab) if tab.shape==(2,2) else (np.nan,np.nan)
 # Exact label permutation for difference in mode position by library.
 x=meta.mode_position.to_numpy();poly=(meta.library_preparation=='polyA').to_numpy();obs=x[poly].mean()-x[~poly].mean();null=[]
 for ids in itertools.combinations(range(len(x)),poly.sum()):
  q=np.zeros(len(x),bool);q[list(ids)]=1;null.append(x[q].mean()-x[~q].mean())
 test=pd.DataFrame([{'test':'polyA vs ribodepleted fixed mode membership','effect':odds,'permutation_or_exact_p':p,'note':'Fisher exact; studies not independent'},
  {'test':'polyA vs ribodepleted mode-position mean difference','effect':obs,'permutation_or_exact_p':np.mean(np.abs(null)>=abs(obs)),'note':'exact label permutation; descriptive because library is study-nested'}]);test.to_csv(O/'technical_mode_exploratory_tests.csv',index=False)
 # Within-study decisive control: studies containing both modes under identical library class.
 within=[]
 for osd,g in meta.groupby('OSD'):
  if g.geometry_cluster.nunique()==2:within.append({'OSD':osd,'library_preparation':' | '.join(g.library_preparation.unique()),'sequencing_parameters':' | '.join(g.sequencing_parameters.unique()),'mode1_contrasts':sum(g.geometry_cluster==1),'mode2_contrasts':sum(g.geometry_cluster==2)})
 pd.DataFrame(within).to_csv(O/'within_study_opposite_modes_same_protocol.csv',index=False)
 return meta,test

def ribo_geometry(meta):
 v=np.load(R/'task3b_bridgerna_response_vectors.npz',allow_pickle=True);vec=dict(zip(v['contrast_id'],v['delta_z']));d=meta[meta.library_preparation.eq('ribodepleted')].sort_values(['geometry_cluster','heatmap_order']);names=d.contrast_id.tolist();X=np.stack([vec[n] for n in names]);M=X@X.T/np.outer(np.linalg.norm(X,axis=1),np.linalg.norm(X,axis=1));pd.DataFrame(M,index=names,columns=names).to_csv(O/'within_ribodepletion_cosine.csv')
 vals=[]
 for i,j in itertools.combinations(range(len(names)),2):vals.append({'relation':'same_mode' if d.iloc[i].geometry_cluster==d.iloc[j].geometry_cluster else 'different_mode','cosine':M[i,j]})
 summary=pd.DataFrame(vals).groupby('relation').cosine.agg(['count','mean','median','std']).reset_index();summary.to_csv(O/'within_ribodepletion_similarity_summary.csv',index=False)
 cluster=fcluster(linkage(squareform(1-M,checks=False),method='average'),2,criterion='maxclust');ari=adjusted_rand_score(d.geometry_cluster,cluster)
 fig,ax=plt.subplots(figsize=(10,8),layout='constrained');im=ax.imshow(M,cmap='RdBu_r',vmin=-1,vmax=1);labs=[f"M{x.geometry_cluster} {x.OSD} {x.contrast_id[:3]}" for x in d.itertuples()];ax.set(xticks=range(len(labs)),yticks=range(len(labs)),xticklabels=labs,yticklabels=labs,title=f'Ribodepleted-only BridgeRNA response cosine (ARI={ari:.2f})');ax.tick_params(axis='x',rotation=65,labelsize=8);fig.colorbar(im,ax=ax,label='Cosine')
 for ext in ['png','pdf']:fig.savefig(O/f'within_ribodepletion_response_cosine.{ext}',dpi=400,bbox_inches='tight')
 plt.close(fig);return summary,ari

def rr3_protocol_comparison(tech):
 d=tech[tech.representation.isin(['OSD-48 original RR-1','OSD-168 RR-1 no-ERCC','OSD-137 original RR-3','OSD-168 RR-3 ERCC'])].copy();d['comparison_context']=['RR1 original comparator','RR1 technical replicate','RR3 original comparator','RR3 technical replicate'];d.to_csv(O/'rr1_vs_rr3_protocol_comparison.csv',index=False)

def figures():
 pair=pd.read_csv(O/'rr1_response_pairwise.csv');fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained');labs=[f"{a.split('_',1)[1]}\nvs\n{b.split('_',1)[1]}" for a,b in zip(pair.representation_a,pair.representation_b)];ax.bar(range(len(pair)),pair.cosine,color=['#d7301f','#d7301f','#2b8cbe']);ax.set(xticks=range(len(pair)),xticklabels=labs,ylabel='Response cosine',ylim=(-1,1),title='RR-1 response change across technical representations');ax.axhline(0,color='black',lw=.8)
 for ext in ['png','pdf']:fig.savefig(O/f'rr1_response_reversal_diagnostic.{ext}',dpi=400,bbox_inches='tight')
 plt.close(fig)

def main():
 tech=technical_metadata();shift=rr1_shift();rr=response_diagnostics();meta,tests=all_contrast_associations();ribo,ari=ribo_geometry(meta);rr3_protocol_comparison(tech);figures()
 prov={'created_utc':datetime.now(timezone.utc).isoformat(),'classification':'B_technical_processing_influences_some_vectors_not_overall_modes','ribodepleted_contrasts':int((meta.library_preparation=='ribodepleted').sum()),'ribodepleted_cluster_ARI':ari,'OSD168_independent_biological_replication':False}
 (O/'provenance.json').write_text(json.dumps(prov,indent=2));print(tech.to_string(index=False));print('\nRR1\n',rr.to_string(index=False));print('\nShift alignment\n',shift.to_string(index=False));print('\nTests\n',tests.to_string(index=False));print('\nRibo\n',ribo.to_string(index=False));print(f'\nClassification B; ribo ARI={ari:.3f}')
if __name__=='__main__':main()
