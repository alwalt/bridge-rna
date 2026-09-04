#!/usr/bin/env python3
"""Technical-replication audit of OSD-168 versus its OSD-48/137 source material."""
from __future__ import annotations
import json,re
from datetime import datetime,timezone
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'results'; W=ROOT/'work'; O=R/'task3_osd168_technical_replication';O.mkdir(parents=True,exist_ok=True)
def cos(a,b): return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
def animal(s):
 m=re.search(r'_(M|F|G|B)(\d+)$',re.sub(r'_?(?:wERCC|noERCC)(?=_)','',str(s)));return ''.join(m.groups()) if m else ''
def condition(s):
 s=str(s);return 'FLT' if '_FLT_' in s else ('GC' if '_GC_' in s else ('BSL' if '_BSL_' in s else 'other'))
def mission(s): return 'RR3' if '_RR3_' in str(s) else 'RR1'

def metadata_correspondence():
 a=pd.read_csv(R/'task3_manifest_api_validation/api_OSD-168_sample_metadata.csv',dtype=str,low_memory=False).fillna('')
 a=a[a['investigation.study assays.study assay technology type'].str.contains('RNA Sequencing',case=False,na=False)].drop_duplicates('id.sample name')
 originals=[]
 for osd in ['48','137']:
  d=pd.read_csv(R/f'task3_manifest_api_validation/api_OSD-{osd}_sample_metadata.csv',dtype=str,low_memory=False).fillna('')
  d=d[d['investigation.study assays.study assay technology type'].str.contains('RNA Sequencing',case=False,na=False)].drop_duplicates('id.sample name');d['source_OSD']=f'OSD-{osd}';originals.append(d)
 orig=pd.concat(originals,ignore_index=True);orig['animal_id']=orig['id.sample name'].map(animal);orig['group']=orig['id.sample name'].map(condition);orig['RR_mission']=orig.source_OSD.map({'OSD-48':'RR1','OSD-137':'RR3'})
 rows=[]
 for _,x in a.iterrows():
  sid=x['id.sample name'];ani=animal(sid);rr=mission(sid);grp=condition(sid);hit=orig[(orig.animal_id==ani)&(orig.group==grp)&(orig.RR_mission==rr)]
  spike='no-ERCC' if 'Without' in x.get('study.factor value.spike-in quality control','') else x.get('assay.parameter value.spike-in mix number','')
  source='OSD-48' if rr=='RR1' else 'OSD-137'; match=hit[hit.source_OSD.eq(source)]
  rows.append({'source_OSD':source,'source_sample':match['id.sample name'].iloc[0] if len(match) else '', 'OSD168_sample':sid,'animal_id':ani,'RR_mission':rr,'group':grp,'ERCC_condition':spike,
   'library_condition':f"{x.get('assay.parameter value.library selection','')}; {x.get('assay.parameter value.library kit','')}",
   'sequencing_condition':f"{x.get('assay.parameter value.library layout','')}; {x.get('assay.parameter value.read length','')}; {x.get('assay.parameter value.sequencing instrument','')}",
   'exact_animal_match':bool(len(match)),'biological_material_status':'same source animal/liver material' if len(match) else 'no original assay match',
   'identical_RNA_status':('supported same RR3 RNA material; OSD-137 is the previously sequenced no-ERCC comparator' if rr=='RR3' and len(match) else ('same animal/liver; identical RNA aliquot not established across OSD-48 and OSD-168' if len(match) else 'not applicable')),
   'authority':'OSD-168 dataset protocol explicitly identifies GLDS-48/GLDS-137 source material; animal, mission, and FLT/GC agree in assay metadata'})
 out=pd.DataFrame(rows);out.to_csv(O/'biological_sample_correspondence.csv',index=False);return out

def response(name, ids, emb, index):
 f=[index[x] for x in ids if condition(x)=='FLT'];g=[index[x] for x in ids if condition(x)=='GC'];return emb[f].mean(0)-emb[g].mean(0)
def build_responses(corr):
 manifest=pd.read_csv(R/'sample_manifest.csv');emb=np.load(W/'bridgerna_embeddings.npy');idx={s:i for i,s in enumerate(manifest.sample_id)}
 c=pd.read_csv(R/'task3b_contrast_sample_membership.csv');clusters=pd.read_csv(R/'task3c_cluster_assignments.csv').set_index('contrast_id').geometry_cluster
 specs=[]
 # RR1 carcass: restrict all representations to animals observed in both OSD-48 and OSD-168.
 rr1=corr[(corr.RR_mission=='RR1')&corr.exact_animal_match&corr.group.isin(['FLT','GC'])]; common=sorted(rr1.animal_id.unique())
 orig=rr1.drop_duplicates('animal_id').source_sample.tolist()
 for ercc in ['no-ERCC','Mix 1','Mix 2']:
  z=rr1[rr1.ERCC_condition.eq(ercc)]; ids=z.OSD168_sample.tolist()
  if set(z.group)>={'FLT','GC'}: specs.append((f'RR1_OSD168_{ercc}',ids,'C14__OSD-48__RR1-NASA__37-day',1,ercc))
 z=rr1[rr1.ERCC_condition.ne('no-ERCC')]
 specs.append(('RR1_OSD168_all_ERCC',z.OSD168_sample.tolist(),'C14__OSD-48__RR1-NASA__37-day',1,'all ERCC mixes'))
 specs.append(('RR1_OSD48_original_matched',orig,'C14__OSD-48__RR1-NASA__37-day',1,'original polyA/no declared ERCC'))
 # RR3 contrasts C01/C02; use exact animals for each original/OSD168 comparison.
 for cid in ['C01__OSD-137__RR3__39-day','C02__OSD-137__RR3__40-day']:
  original_members=c[c.contrast_id.eq(cid)].sample_id.tolist(); animals={animal(x) for x in original_members}
  z=corr[(corr.RR_mission=='RR3')&corr.animal_id.isin(animals)&corr.exact_animal_match&corr.group.isin(['FLT','GC'])]
  original=z.drop_duplicates('animal_id').source_sample.tolist(); mode=int(clusters.loc[cid])
  specs.append((f'{cid[:3]}_OSD137_original_matched',original,cid,mode,'original no-ERCC'))
  specs.append((f'{cid[:3]}_OSD168_all_ERCC',z.OSD168_sample.tolist(),cid,mode,'all ERCC mixes'))
  for mix in ['Mix 1','Mix 2']:
   q=z[z.ERCC_condition.eq(mix)]
   if set(q.group)>={'FLT','GC'}:
    # Exact matched original subset counterpart for a fair Mix-specific comparison.
    specs.append((f'{cid[:3]}_OSD137_original_{mix}',q.source_sample.tolist(),cid,mode,f'original animals assigned {mix} in OSD-168'))
    specs.append((f'{cid[:3]}_OSD168_{mix}',q.OSD168_sample.tolist(),cid,mode,mix))
 rows=[];vectors={}
 for name,ids,cid,mode,technical in specs:
  if not all(x in idx for x in ids): raise KeyError(f'missing embedding: {set(ids)-set(idx)}')
  v=response(name,ids,emb,idx);vectors[name]=v
  rows.append({'representation':name,'source_contrast':cid,'assigned_mode':mode,'technical_condition':technical,'n_FLT':sum(condition(x)=='FLT' for x in ids),'n_GC':sum(condition(x)=='GC' for x in ids),'samples':' | '.join(ids)})
 design=pd.DataFrame(rows);design.to_csv(O/'technical_response_design.csv',index=False)
 np.savez_compressed(O/'technical_response_vectors.npz',names=np.array(list(vectors),dtype=object),delta_z=np.stack(list(vectors.values())))
 return vectors,design

def comparisons(vectors,design):
 directions=np.load(R/'task3d_mode_ig/mode_response_directions.npz');rows=[]
 for _,x in design[design.representation.str.contains('OSD168')].iterrows():
  n=x.representation; key='RR1_OSD48_original_matched' if n.startswith('RR1') else n.replace('OSD168','OSD137_original')
  if n.endswith('all_ERCC') and not n.startswith('RR1'): key=n[:3]+'_OSD137_original_matched'
  elif 'Mix' in n and n.startswith('C'): key=n.replace('OSD168','OSD137_original')
  a,b=vectors[key],vectors[n]; sims={m:cos(b,directions[f'mode_{m}']) for m in [1,2]};pred=max(sims,key=sims.get)
  rows.append({'source_contrast':x.source_contrast,'original_representation':key,'OSD168_representation':n,'technical_condition':x.technical_condition,'assigned_mode':x.assigned_mode,
   'cosine':cos(a,b),'spearman':spearmanr(a,b).statistic,'euclidean_distance':np.linalg.norm(a-b),'relative_euclidean_distance':np.linalg.norm(a-b)/np.linalg.norm(a),
   'mode1_projection':sims[1],'mode2_projection':sims[2],'predicted_fixed_mode':pred,'mode_preserved':pred==x.assigned_mode})
 out=pd.DataFrame(rows);out.to_csv(O/'original_vs_osd168_response_similarity.csv',index=False)
 # Full response geometry, retaining fixed original Task 3 responses.
 z=np.load(R/'task3b_bridgerna_response_vectors.npz',allow_pickle=True);base=dict(zip(z['contrast_id'],z['delta_z']));allv={**base,**vectors};names=list(allv);X=np.stack([allv[n] for n in names]);M=X@X.T/np.outer(np.linalg.norm(X,axis=1),np.linalg.norm(X,axis=1))
 pd.DataFrame(M,index=names,columns=names).to_csv(O/'augmented_response_cosine.csv')
 # Similarity-profile preservation versus other independent biological contrasts.
 bio=[k for k in base if 'OSD-168' not in k]
 prof=[]
 for x in out.itertuples():
  so=[cos(vectors[x.original_representation],base[k]) for k in bio];st=[cos(vectors[x.OSD168_representation],base[k]) for k in bio]
  prof.append({'OSD168_representation':x.OSD168_representation,'other_contrasts':len(bio),'similarity_profile_spearman':spearmanr(so,st).statistic})
 pd.DataFrame(prof).to_csv(O/'similarity_relationship_preservation.csv',index=False)
 # Direct technical comparisons inside OSD-168. The all-ERCC/no-ERCC RR1
 # comparison uses exactly the same animals; Mix-specific contrasts use their
 # documented subsets and are retained as secondary diagnostics.
 tech=[]
 pairs=[('RR1_OSD168_no-ERCC','RR1_OSD168_all_ERCC','same animals, spike-in differs'),('RR1_OSD168_Mix 1','RR1_OSD168_Mix 2','different documented animal subsets')]
 for cid in ['C01','C02']: pairs.append((f'{cid}_OSD168_Mix 1',f'{cid}_OSD168_Mix 2','different documented animal subsets'))
 for a,b,identity in pairs:
  tech.append({'representation_a':a,'representation_b':b,'identity_control':identity,'cosine':cos(vectors[a],vectors[b]),'spearman':spearmanr(vectors[a],vectors[b]).statistic,'euclidean_distance':np.linalg.norm(vectors[a]-vectors[b])})
 pd.DataFrame(tech).to_csv(O/'within_osd168_technical_similarity.csv',index=False)
 return out,names,M

def plots(comp,names,M):
 fig,ax=plt.subplots(figsize=(9,5.5),layout='constrained');y=np.arange(len(comp));ax.barh(y,comp.cosine,color=np.where(comp.mode_preserved,'#2b8cbe','#d7301f'));ax.set(yticks=y,yticklabels=comp.OSD168_representation,xlim=(-1,1),xlabel='Cosine(original response, OSD-168 response)',title='Exact-material BridgeRNA technical response preservation');ax.axvline(0,color='black',lw=.8)
 for ext in ['png','pdf']:fig.savefig(O/f'original_vs_osd168_response_similarity.{ext}',dpi=400,bbox_inches='tight')
 plt.close(fig)
 sel=[i for i,n in enumerate(names) if 'OSD168' in n or n in set(comp.original_representation)];m=M[np.ix_(sel,sel)];labs=[names[i] for i in sel]
 fig,ax=plt.subplots(figsize=(11,9),layout='constrained');im=ax.imshow(m,cmap='RdBu_r',vmin=-1,vmax=1);ax.set(xticks=range(len(labs)),yticks=range(len(labs)),xticklabels=labs,yticklabels=labs,title='Original and technical-replicate response cosine');ax.tick_params(axis='x',rotation=70,labelsize=7);ax.tick_params(axis='y',labelsize=7);fig.colorbar(im,ax=ax,label='Cosine')
 for ext in ['png','pdf']:fig.savefig(O/f'technical_response_cosine_heatmap.{ext}',dpi=400,bbox_inches='tight')
 plt.close(fig)

def main():
 corr=metadata_correspondence();vectors,design=build_responses(corr);comp,names,M=comparisons(vectors,design);plots(comp,names,M)
 summary=(comp.groupby('source_contrast').agg(technical_versions=('cosine','size'),cosine_mean=('cosine','mean'),cosine_min=('cosine','min'),spearman_mean=('spearman','mean'),mode_preservation_rate=('mode_preserved','mean')).reset_index());summary.to_csv(O/'mode_preservation_summary.csv',index=False)
 overall={'created_utc':datetime.now(timezone.utc).isoformat(),'classification':'B_partial_mixed_preservation','OSD168_independent_biological_replication':False,'exact_animal_matches':int(corr.exact_animal_match.sum()),'OSD168_samples_audited':len(corr),'technical_comparisons':len(comp),'mean_cosine':comp.cosine.mean(),'median_cosine':comp.cosine.median(),'mode_preservation_rate':comp.mode_preserved.mean()}
 (O/'provenance.json').write_text(json.dumps(overall,indent=2));print(summary.to_string(index=False));print('\n',comp.to_string(index=False));print('\nClassification: B — partial/mixed preservation')
if __name__=='__main__':main()
