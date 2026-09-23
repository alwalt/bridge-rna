#!/usr/bin/env python3
"""Score observed GSE264130 RNA-seq deltas against frozen condition modules."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from statsmodels.stats.multitest import multipletests
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parents[1];OUT=HERE/'results';WORK=HERE/'work';WORK.mkdir(exist_ok=True)
SRC=Path('/home/walt/bridge-rna.worktrees/attachment-pasted-text-1-30bae6d6/benchmarks/drug_perturbation_prediction')
FROZEN=ROOT/'benchmarks/direction_aware_reversal/results/frozen_condition_signatures.parquet'
SPACE={'Radiation injury':['GSE297090','GSE184119','GSE297560'],'Bone loss':['GSE189524','GSE276529','GSE273868'],'Muscle atrophy':['GSE211204','GSE113165','GSE234465']}
def empirical(frame,group,score='reversal_score'):
 out=[]
 for _,g in frame.groupby(group,sort=False):
  g=g.copy();n=len(g);g['empirical_p']=g[score].rank(method='max',ascending=False)/(n+1);g['fdr']=multipletests(g.empirical_p,method='fdr_bh')[1];out.append(g)
 return pd.concat(out,ignore_index=True)
def main():
 manifest=pd.read_parquet(SRC/'results/perturbation_vectors/observed_expression_delta_manifest.parquet').sort_values('vector_row')
 delta=np.load(SRC/'work/prepared/observed_condition_delta.npy',mmap_mode='r')
 genes=pd.read_parquet(SRC/'results/perturbation_vectors/gene_order.parquet').bridge_symbol.astype(str).tolist();gidx={g:i for i,g in enumerate(genes)}
 sig=pd.read_parquet(FROZEN);defs=[];members=[]
 for (cond,cid,ds,role,method),g in sig[sig.top500].groupby(['condition','contrast_id','dataset','role','method'],sort=True):
  g=g[g.gene.isin(gidx)].sort_values('rank');n=len(g);did=f'{method}|{cid}'
  w=np.sign(g.signed_score.to_numpy())*(n-np.arange(n))/n;w=w/np.linalg.norm(w)
  defs.append({'definition_id':did,'condition':cond,'contrast_id':cid,'dataset':ds,'role':role,'method':method,'usable_genes':n})
  members.extend({'definition_id':did,'gene':gene,'weight':weight,'condition_score':score,'condition_rank':rank} for gene,weight,score,rank in zip(g.gene,w,g.signed_score,g['rank']))
 defs=pd.DataFrame(defs);members=pd.DataFrame(members);defs.to_csv(OUT/'expression_signature_definitions.csv',index=False);members.to_csv(OUT/'expression_signature_members.csv.gz',index=False,compression='gzip')
 W=np.zeros((len(genes),len(defs)),np.float32)
 for j,d in defs.iterrows():
  m=members[members.definition_id==d.definition_id];W[[gidx[x] for x in m.gene],j]=m.weight
 dnorm=np.linalg.norm(np.asarray(delta),axis=1);scores=-(np.asarray(delta)@W)/(dnorm[:,None]+1e-12)
 np.save(WORK/'expression_reversal_scores.npy',scores.astype(np.float32))
 rows=[]
 for j,d in defs.iterrows():
  q=manifest[['vector_row','drug','dose_uM','time_hours','cell_line','n_plate_replicates']].copy();q['reversal_score']=scores[:,j]
  for c in defs.columns:q[c]=d[c]
  rows.append(q)
 result=pd.concat(rows,ignore_index=True);result=empirical(result,['definition_id','cell_line'])
 result['direction']=np.where(result.reversal_score>0,'opposes','reinforces')
 result.to_parquet(OUT/'expression_reversal_by_contrast_cell.parquet',index=False)
 # Dataset summaries retain separate cells and independent studies.
 ds=(result.groupby(['condition','method','dataset','role','drug','cell_line'],as_index=False)
     .agg(reversal_score=('reversal_score','median'),contrast_count=('contrast_id','nunique'),usable_genes=('usable_genes','min')))
 ds=empirical(ds,['condition','method','dataset','cell_line']);ds.to_csv(OUT/'expression_reversal_by_dataset_cell.csv.gz',index=False,compression='gzip')
 # Cell classification at condition level; for space use median of independent dataset scores.
 cc=(ds.groupby(['condition','method','drug','cell_line'],as_index=False)
     .agg(reversal_score=('reversal_score','median'),dataset_count=('dataset','nunique'),max_dataset_p=('empirical_p','max'),min_fdr=('fdr','min')))
 cc=empirical(cc,['condition','method','cell_line']);wide=cc.pivot(index=['condition','method','drug'],columns='cell_line',values=['reversal_score','empirical_p']).reset_index()
 wide.columns=['_'.join(x).rstrip('_') if isinstance(x,tuple) else x for x in wide.columns]
 def classify(r):
  a,b=r.reversal_score_DIPG6,r.reversal_score_SF8628;pa,pb=r.empirical_p_DIPG6,r.empirical_p_SF8628
  if a>0 and b>0 and pa<=.05 and pb<=.05:return 'reversal in both contexts'
  if a<0 and b<0:return 'reinforcement in both'
  if a*b<0:return 'mixed/opposing contexts'
  if (a>0 and pa<=.05) or (b>0 and pb<=.05):return 'reversal in one context only'
  return 'same direction, insufficient null support'
 wide['cell_classification']=wide.apply(classify,axis=1);wide['candidate_both_cells']=wide.cell_classification.eq('reversal in both contexts')
 wide.to_csv(OUT/'expression_drug_condition_consensus.csv.gz',index=False,compression='gzip')
 # Strict cross-study result: every study, both cells, top decile and positive.
 cross=[]
 for cond,dsets in SPACE.items():
  for method in ['Bridge','DE']:
   q=ds[(ds.condition==cond)&(ds.method==method)&ds.dataset.isin(dsets)]
   for drug,g in q.groupby('drug'):
    ok=(g.dataset.nunique()==3 and g.cell_line.nunique()==2 and len(g)==6 and (g.reversal_score>0).all() and (g.empirical_p<=.10).all())
    cross.append({'condition':cond,'method':method,'drug':drug,'cross_study_consistent':ok,'dataset_cells':len(g),
                  'median_reversal':g.reversal_score.median(),'max_empirical_p':g.empirical_p.max(),
                  'terrestrial_osdr_sign_agreement':g.groupby('dataset').reversal_score.median().gt(0).all()})
 pd.DataFrame(cross).to_csv(OUT/'expression_cross_study_consensus.csv',index=False)
 prov={'source':'observed_condition_delta.npy only','drug_contexts':len(manifest),'definitions':len(defs),'score':'negative weighted cosine',
       'null':'competitive empirical drug rank within frozen signature and cell line','prediction_assets_used':False,'upstream_rerun':False}
 (OUT/'expression_reversal_provenance.json').write_text(json.dumps(prov,indent=2)+'\n');print(json.dumps(prov,indent=2))
if __name__=='__main__':main()
