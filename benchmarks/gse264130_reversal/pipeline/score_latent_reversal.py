#!/usr/bin/env python3
"""Construct observed GSE264130 latent deltas and compare to frozen condition vectors."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from statsmodels.stats.multitest import multipletests
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parents[1];OUT=HERE/'results';WORK=HERE/'work'
SRC=Path('/home/walt/bridge-rna.worktrees/attachment-pasted-text-1-30bae6d6/benchmarks/drug_perturbation_prediction')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def empirical(frame,group,score='latent_reversal'):
 out=[]
 for _,g in frame.groupby(group,sort=False):
  g=g.copy();n=len(g);g['empirical_p']=g[score].rank(method='max',ascending=False)/(n+1);g['fdr']=multipletests(g.empirical_p,method='fdr_bh')[1];out.append(g)
 return pd.concat(out,ignore_index=True)
def main():
 manifest=pd.read_parquet(SRC/'work/prepared/sample_manifest.parquet').reset_index(drop=True)
 emb=np.empty((len(manifest),512),np.float32)
 for s in [0,1]:
  idx=np.load(WORK/f'observed_sample_indices_shard{s}.npy');val=np.load(WORK/f'observed_sample_embeddings_shard{s}.npy');assert len(idx)==len(val);emb[idx]=val
 np.save(WORK/'observed_sample_embeddings.npy',emb)
 # Mean DMSO sample embeddings within each independently plated control block.
 controls={}
 for (cell,plate),g in manifest[manifest.drug.eq('DMSO')].groupby(['cell_line','plate']):controls[(cell,plate)]=emb[g.index].mean(0)
 treated=manifest[manifest.is_treated].copy();deltas=[];rows=[]
 for condition,g in treated.groupby('condition_group',sort=True):
  values=[]
  for i,r in g.iterrows():values.append(emb[i]-controls[(r.cell_line,r.plate)])
  assert len(values)==2 and g.plate.nunique()==2
  z=np.mean(values,axis=0);r=g.iloc[0];rows.append({'vector_row':len(rows),'condition_group':condition,'drug':r.drug,'dose_uM':r.dose_uM,'time_hours':r.time_hours,'cell_line':r.cell_line,'n_plate_replicates':2});deltas.append(z)
 zdelta=np.asarray(deltas,np.float32);zmanifest=pd.DataFrame(rows);np.save(WORK/'observed_latent_delta.npy',zdelta);zmanifest.to_parquet(OUT/'observed_latent_delta_manifest.parquet',index=False)
 # Frozen disease and space condition vectors.
 defs=[];vecs=[]
 for acc,cond in [('GSE138614','Multiple sclerosis'),('GSE101794',"Crohn's disease"),('GSE72509','Systemic lupus erythematosus')]:
  v=np.load(ROOT/f'benchmarks/deweerd_replication/work/bridge/{acc}_condition_direction.npy');defs.append({'condition':cond,'contrast_id':acc,'dataset':acc,'role':'published_disease'});vecs.append(v)
 cv=pd.read_parquet(ROOT/'benchmarks/drug_discovery/results/bridge/condition_vectors.parquet')
 for (cid,ds),g in cv[~cv.contrast_id.str.endswith('_unseen')].sort_values('dimension').groupby(['contrast_id','dataset'],sort=True):
  cond='Radiation injury' if ds in ['GSE297090','GSE184119','GSE297560'] else 'Bone loss' if ds in ['GSE189524','GSE276529','GSE273868'] else 'Muscle atrophy'
  role={'GSE297090':'terrestrial_discovery','GSE184119':'terrestrial_replication','GSE297560':'osdr','GSE189524':'terrestrial_discovery','GSE276529':'terrestrial_replication','GSE273868':'osdr','GSE211204':'terrestrial_discovery','GSE113165':'terrestrial_replication','GSE234465':'osdr'}[ds]
  defs.append({'condition':cond,'contrast_id':cid,'dataset':ds,'role':role});vecs.append(g.value.to_numpy())
 defs=pd.DataFrame(defs);C=np.asarray(vecs,float);C=C/(np.linalg.norm(C,axis=1,keepdims=True)+1e-12);Z=zdelta/(np.linalg.norm(zdelta,axis=1,keepdims=True)+1e-12);score=-(Z@C.T)
 result=[]
 for j,d in defs.iterrows():
  q=zmanifest.copy();q['latent_reversal']=score[:,j];q['latent_cosine']=-score[:,j]
  for c in defs.columns:q[c]=d[c]
  result.append(q)
 result=empirical(pd.concat(result,ignore_index=True),['contrast_id','cell_line']);result['direction']=np.where(result.latent_reversal>0,'opposes','reinforces')
 result.to_parquet(OUT/'latent_reversal_by_contrast_cell.parquet',index=False)
 ds=(result.groupby(['condition','dataset','role','drug','cell_line'],as_index=False).agg(latent_reversal=('latent_reversal','median'),latent_cosine=('latent_cosine','median'),contrast_count=('contrast_id','nunique')))
 ds=empirical(ds,['condition','dataset','cell_line']);ds.to_csv(OUT/'latent_reversal_by_dataset_cell.csv.gz',index=False,compression='gzip')
 cc=(ds.groupby(['condition','drug','cell_line'],as_index=False).agg(latent_reversal=('latent_reversal','median'),dataset_count=('dataset','nunique'),max_dataset_p=('empirical_p','max')))
 cc=empirical(cc,['condition','cell_line']);wide=cc.pivot(index=['condition','drug'],columns='cell_line',values=['latent_reversal','empirical_p']).reset_index();wide.columns=['_'.join(x).rstrip('_') if isinstance(x,tuple) else x for x in wide.columns]
 def cls(r):
  a,b=r.latent_reversal_DIPG6,r.latent_reversal_SF8628;pa,pb=r.empirical_p_DIPG6,r.empirical_p_SF8628
  if a>0 and b>0 and pa<=.05 and pb<=.05:return 'reversal in both contexts'
  if a<0 and b<0:return 'reinforcement in both'
  if a*b<0:return 'mixed/opposing contexts'
  if (a>0 and pa<=.05) or (b>0 and pb<=.05):return 'reversal in one context only'
  return 'same direction, insufficient null support'
 wide['cell_classification']=wide.apply(cls,axis=1);wide['candidate_both_cells']=wide.cell_classification.eq('reversal in both contexts');wide.to_csv(OUT/'latent_drug_condition_consensus.csv.gz',index=False,compression='gzip')
 cross=[]
 for cond in ['Radiation injury','Bone loss','Muscle atrophy']:
  for drug,g in ds[ds.condition==cond].groupby('drug'):
   ok=g.dataset.nunique()==3 and g.cell_line.nunique()==2 and len(g)==6 and (g.latent_reversal>0).all() and (g.empirical_p<=.10).all()
   cross.append({'condition':cond,'drug':drug,'cross_study_consistent':ok,'dataset_cells':len(g),'median_latent_reversal':g.latent_reversal.median(),'max_empirical_p':g.empirical_p.max()})
 pd.DataFrame(cross).to_csv(OUT/'latent_cross_study_consensus.csv',index=False)
 prov={'checkpoint':'/home/walt/bridge-rna/model/r7hnr92k/best_model.pt','samples_encoded':len(manifest),'input':'observed GSE264130 natural log1p(TPM), canonical zero-fill 137 genes',
       'drug_delta':'two plate-level treated embeddings minus six-DMSO mean embedding per matched plate, then averaged','latent_vectors':len(zmanifest),'dimension':512,
       'prediction_assets_used':False,'embedding_sha256':sha(WORK/'observed_sample_embeddings.npy'),'delta_sha256':sha(WORK/'observed_latent_delta.npy')}
 (OUT/'latent_reversal_provenance.json').write_text(json.dumps(prov,indent=2)+'\n');print(json.dumps(prov,indent=2))
if __name__=='__main__':main()
