#!/usr/bin/env python3
"""Audit the OSD-137 RR3 39/40/41-day cohorts without rerunning BridgeRNA."""
from __future__ import annotations
import json, re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

ROOT=Path(__file__).resolve().parents[1]; REPO=ROOT.parents[1]
OUT=ROOT/'results/task4_confounding_profiler/rr3_cohort_audit'; FIG=OUT/'figures'
T3=REPO/'benchmarks/osdr_batch_effect_representation'; RES=T3/'results'; WORK=T3/'work'
API=RES/'task3_manifest_api_validation/api_OSD-137_sample_metadata.csv'

def cos(a,b):
 d=np.linalg.norm(a)*np.linalg.norm(b); return float(a@b/d) if d else np.nan

def basis():
 m=pd.read_parquet(ROOT/'work/datasets/chen_2020_tcells/manifest.parquet').reset_index(drop=True)
 z=np.load(ROOT/'work/datasets/chen_2020_tcells/bridgerna_embeddings.npy').astype(float); ds=[]
 for _,g in m.groupby('pair_id',sort=True): ds.append(z[g.index[g.library_prep.eq('ribo')][0]]-z[g.index[g.library_prep.eq('polyA')][0]])
 d=np.stack(ds); _,_,vt=np.linalg.svd(d,full_matrices=False)
 for j in range(2):
  if np.mean(d@vt[j])<0: vt[j]*=-1
 direction=d.mean(0); direction=(vt[:2].T@(vt[:2]@direction)); direction/=np.linalg.norm(direction)
 return vt[:2],direction

def duration(sample):
 x=re.search(r'_([FG])(\d+)$',sample); animal=x.group(1)+x.group(2)
 return {'F1':39,'F2':39,'G1':39,'G2':39,'F3':40,'F4':40,'F5':40,'G3':40,'G5':40,'F6':41,'G6':41,'G7':41}[animal],animal

def metadata_table(membership):
 api=pd.read_csv(API); api=api[api['id.assay name'].str.contains('transcription-profiling',case=False,na=False)].copy()
 api=api.drop_duplicates('id.sample name')
 q=membership[membership.OSD.eq('OSD-137')].merge(api,left_on='sample_id',right_on='id.sample name',how='left',validate='one_to_one')
 q['duration_day']=[duration(x)[0] for x in q.sample_id]; q['animal_id']=[duration(x)[1] for x in q.sample_id]
 q['technical_replication_subset']=q.animal_id.isin(['F1','F2','G1','G2','F3','F4','G3','G5'])
 rename={'study.parameter value.duration':'api_duration','study.parameter value.exposure duration':'exposure_duration','study.parameter value.carcass weight':'carcass_weight','study.parameter value.absorbed radiation dose':'radiation_dose','study.parameter value.absorbed radiation dose rate':'radiation_dose_rate','study.parameter value.sample preservation method':'sample_preservation','study.parameter value.sample storage temperature':'storage_temperature','study.parameter value.habitat':'habitat','study.parameter value.euthanasia method':'euthanasia_method','study.characteristics.strain':'strain','study.characteristics.sex':'sex','study.characteristics.age at launch':'age_at_launch','assay.parameter value.library selection':'library_selection','assay.parameter value.library layout':'library_layout','assay.parameter value.sequencing instrument':'sequencing_instrument','assay.parameter value.read length':'read_length','assay.parameter value.read depth':'read_depth','assay.parameter value.rrna contamination':'rrna_contamination','assay.parameter value.qa score':'RIN','assay.parameter value.mid':'library_index','assay.parameter value.stranded':'stranded'}
 q=q.rename(columns=rename)
 cols=['contrast_id','sample_id','animal_id','condition','duration_day','technical_replication_subset']+list(rename.values())
 for c in cols:
  if c not in q:q[c]=np.nan
 return q[cols]

def summarize_metadata(q):
 fields=[c for c in q if c not in ['contrast_id','sample_id','animal_id','condition','technical_replication_subset']]
 rows=[]
 for c in fields:
  vals={d:sorted(q.loc[q.duration_day.eq(d),c].dropna().astype(str).unique()) for d in [39,40,41]}
  sets=[set(vals[d]) for d in [39,40] if vals[d]]
  if not sets: assoc='E. missing / unavailable'
  elif len(sets)==2 and sets[0]==sets[1] and len(sets[0])==1: assoc='A. identical across RR3-39 and RR3-40'
  elif len(sets)==2 and sets[0].isdisjoint(sets[1]): assoc='D. perfectly associated with cohort'
  else: assoc='B/C. varies within and/or partially associated'
  expl='Duration/collection cohort are inseparable in this design.' if c in ['duration_day','api_duration','exposure_duration','radiation_dose'] else ''
  rows.append({'variable':c,'rr3_39_values':' | '.join(vals[39]),'rr3_40_values':' | '.join(vals[40]),'rr3_41_values_if_available':' | '.join(vals[41]),'association_with_cohort':assoc,'possible_explanation':expl,'confidence':'high' if vals[39] and vals[40] else 'low','notes':'API field; missing means unavailable, not identical.'})
 for c in ['euthanasia_date','euthanasia_time','euthanasia_order','tissue_collection_time','dissection_order','RNA_extraction_batch','library_preparation_date','sequencing_run_lane','cage_position','food_consumption','water_consumption','health_observations','tissue_weight']:
  rows.append({'variable':c,'rr3_39_values':'','rr3_40_values':'','rr3_41_values_if_available':'','association_with_cohort':'E. missing / unavailable','possible_explanation':'','confidence':'low','notes':'Not present in locally cached authoritative API sample metadata.'})
 return pd.DataFrame(rows)

def geometry(q):
 manifest=pd.read_csv(RES/'sample_manifest.csv'); z=np.load(WORK/'bridgerna_embeddings.npy').astype(float); x=np.load(WORK/'bridgerna_log1p_tpm_inputs.npy').astype(float)
 idx=dict(zip(manifest.sample_id,range(len(manifest)))); B,direction=basis(); rows=[]
 for r in q.itertuples():
  v=z[idx[r.sample_id]]; p=B@v
  rows.append({'sample_id':r.sample_id,'animal_id':r.animal_id,'condition':r.condition,'duration_day':r.duration_day,'contrast_id':r.contrast_id,'pc1_coordinate':p[0],'pc2_coordinate':p[1],'direction_coordinate':v@direction,'embedding_norm':np.linalg.norm(v)})
 coords=pd.DataFrame(rows)
 decomp=[]; loo=[]; responses={}
 for day,g in q.groupby('duration_day'):
  fi=[idx[s] for s in g.loc[g.condition.eq('FLT'),'sample_id']]; gi=[idx[s] for s in g.loc[g.condition.eq('GC'),'sample_id']]
  f=z[fi].mean(0); c=z[gi].mean(0); r=f-c; responses[day]=r
  decomp.append({'duration_day':day,'n_FLT':len(fi),'n_GC':len(gi),'FLT_direction_coordinate':f@direction,'GC_direction_coordinate':c@direction,'response_directional_cosine':cos(B.T@(B@r),direction),'response_pc12_occupancy':np.sum((B@r)**2)/(r@r),'response_norm':np.linalg.norm(r),'FLT39_or_40_similarity':np.nan,'GC39_or_40_similarity':np.nan})
  for condition,ids in [('FLT',fi),('GC',gi)]:
   for drop in ids:
    ff=[i for i in fi if i!=drop]; gg=[i for i in gi if i!=drop]
    if not ff or not gg: continue
    rr=z[ff].mean(0)-z[gg].mean(0); proj=B.T@(B@rr)
    loo.append({'duration_day':day,'removed_sample':manifest.iloc[drop].sample_id,'removed_condition':condition,'n_FLT':len(ff),'n_GC':len(gg),'directional_cosine':cos(proj,direction),'pc12_occupancy':np.sum((B@rr)**2)/(rr@rr),'response_norm':np.linalg.norm(rr)})
 # Direct FLT and GC cohort shifts.
 ddf=pd.DataFrame(decomp)
 for cond in ['FLT','GC']:
  a=q[(q.duration_day==39)&(q.condition==cond)].sample_id.map(idx).tolist(); b=q[(q.duration_day==40)&(q.condition==cond)].sample_id.map(idx).tolist()
  shift=z[b].mean(0)-z[a].mean(0)
  ddf.loc[:,'FLT39_or_40_similarity' if cond=='FLT' else 'GC39_or_40_similarity']=cos(z[a].mean(0),z[b].mean(0))
  ddf[f'{cond}_39_to_40_shift_norm']=np.linalg.norm(shift); ddf[f'{cond}_39_to_40_shift_pc12_fraction']=np.sum((B@shift)**2)/(shift@shift)
 # Conventional expression comparisons and joint PCA.
 rrids=[idx[s] for s in q.sample_id]; pc=PCA(n_components=min(10,len(rrids)-1),random_state=1).fit_transform(x[rrids]); expr=[]
 for day,g in q.groupby('duration_day'):
  fi=g[g.condition.eq('FLT')].sample_id.map(idx).tolist();gi=g[g.condition.eq('GC')].sample_id.map(idx).tolist(); rv=x[fi].mean(0)-x[gi].mean(0)
  expr.append({'duration_day':day,'expression_response_norm':np.linalg.norm(rv),'expression_response_rms':np.sqrt(np.mean(rv**2))})
 expr=pd.DataFrame(expr); expr['cosine_with_39d']=[cos(x[q[(q.duration_day==39)&(q.condition=='FLT')].sample_id.map(idx)].mean(0)-x[q[(q.duration_day==39)&(q.condition=='GC')].sample_id.map(idx)].mean(0),x[q[(q.duration_day==d)&(q.condition=='FLT')].sample_id.map(idx)].mean(0)-x[q[(q.duration_day==d)&(q.condition=='GC')].sample_id.map(idx)].mean(0)) for d in expr.duration_day]
 return coords,pd.DataFrame(loo),ddf.merge(expr,on='duration_day'),pc,responses

def gene_programs():
 p=ROOT/'results/task4_confounding_profiler/rna_processing_gene_analysis/rna_processing_gene_profiles.csv'; d=pd.read_csv(p)
 # Existing shared top-10% definition, recovered from saved rank columns.
 cols=d.columns.tolist(); r39=[c for c in cols if 'rr3_39' in c.lower() and 'rank' in c.lower()]; r40=[c for c in cols if 'rr3_40' in c.lower() and 'rank' in c.lower()]
 if r39 and r40:
  n=len(d); d['shared_top10_reproducible']=(d[r39[0]]<=np.ceil(.1*n))&(d[r40[0]]<=np.ceil(.1*n))
 else: d['shared_top10_reproducible']=False
 d['specifically_requested_gene']=d.gene.isin(['CNOT3','RBM14','ZFC3H1'])
 keep=['gene','shared_top10_reproducible','specifically_requested_gene','rr3_39_context_reproducibility','rr3_40_context_reproducibility','rr3_39_reproducible_rna_rank','rr3_40_reproducible_rna_rank','rr3_39_vs_40_original_context_cosine','rr3_39_logFC','rr3_40_logFC']
 return d.loc[d.shared_top10_reproducible|d.specifically_requested_gene,keep].sort_values(['shared_top10_reproducible','rr3_39_reproducible_rna_rank'],ascending=[False,True])

def figures(q,coords,loo,decomp,pc):
 plt.style.use('seaborn-v0_8-whitegrid'); colors={39:'#377eb8',40:'#e41a1c',41:'#4daf4a'}
 fig,ax=plt.subplots(figsize=(8,6))
 for day,g in coords.groupby('duration_day'):
  for cond,marker in [('FLT','o'),('GC','s')]:
   h=g[g.condition.eq(cond)];ax.scatter(h.pc1_coordinate,h.pc2_coordinate,label=f'{day}d {cond}',marker=marker,s=80,color=colors[day])
   for r in h.itertuples():ax.annotate(r.animal_id,(r.pc1_coordinate,r.pc2_coordinate),fontsize=8)
 ax.set(xlabel='Controlled T-cell difference PC1 coordinate',ylabel='PC2 coordinate',title='Individual OSD-137 RR3 samples in the fixed PC1–2 reference');ax.legend(ncol=2,fontsize=8);fig.tight_layout();
 for e in ['png','pdf']:fig.savefig(FIG/f'rr3_individual_pc12.{e}',dpi=400)
 plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,6))
 for day,g in coords.groupby('duration_day'):
  f=g[g.condition.eq('FLT')][['pc1_coordinate','pc2_coordinate']].mean().to_numpy();c=g[g.condition.eq('GC')][['pc1_coordinate','pc2_coordinate']].mean().to_numpy();ax.scatter(*c,marker='s',s=100,color=colors[day]);ax.arrow(c[0],c[1],*(f-c),head_width=.015,length_includes_head=True,color=colors[day]);ax.text(f[0],f[1],f'{day}d FLT')
 ax.set(xlabel='PC1 coordinate',ylabel='PC2 coordinate',title='FLT−GC response arrows by collection cohort');fig.tight_layout()
 for e in ['png','pdf']:fig.savefig(FIG/f'rr3_flt_gc_centroids.{e}',dpi=400)
 plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,5));
 for day,g in loo.groupby('duration_day'):ax.scatter([day]*len(g),g.directional_cosine,s=55,color=colors[day],label=f'{day}d')
 ax.axhline(0,color='k',ls='--');ax.set(xticks=[39,40,41],xlabel='Collection/duration cohort',ylabel='PolyA→Ribo directional cosine',title='Leave-one-animal-out response orientation');fig.tight_layout()
 for e in ['png','pdf']:fig.savefig(FIG/f'rr3_leave_one_out.{e}',dpi=400)
 plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,5));g=decomp.set_index('duration_day');ax.bar(np.arange(3)-.18,g.FLT_direction_coordinate,.36,label='FLT');ax.bar(np.arange(3)+.18,g.GC_direction_coordinate,.36,label='GC');ax.set(xticks=np.arange(3),xticklabels=[f'{x}d' for x in g.index],ylabel='Mean coordinate on oriented reference',title='FLT and GC contributions to response direction');ax.legend();fig.tight_layout()
 for e in ['png','pdf']:fig.savefig(FIG/f'rr3_flt_gc_decomposition.{e}',dpi=400)
 plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,5));
 for day,marker in [(39,'o'),(40,'s'),(41,'^')]:
  ids=q[q.duration_day.eq(day)].index.to_numpy();ax.scatter(pc[ids,0],pc[ids,1],label=f'{day}d',marker=marker,s=80,color=colors[day])
 ax.set(xlabel='Expression PC1',ylabel='Expression PC2',title='Conventional log1p(TPM) PCA: OSD-137 samples');ax.legend();fig.tight_layout()
 for e in ['png','pdf']:fig.savefig(FIG/f'rr3_conventional_expression_pca.{e}',dpi=400)
 plt.close(fig)

def main():
 OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(exist_ok=True)
 membership=pd.read_csv(RES/'task3b_contrast_sample_membership.csv');q=metadata_table(membership);q.to_csv(OUT/'rr3_sample_metadata_audit.csv',index=False)
 sm=summarize_metadata(q);sm.to_csv(OUT/'rr3_cohort_metadata_summary.csv',index=False)
 coords,loo,decomp,pc,responses=geometry(q);coords.to_csv(OUT/'rr3_individual_pc12_coordinates.csv',index=False);loo.to_csv(OUT/'rr3_leave_one_out_sensitivity.csv',index=False);decomp.to_csv(OUT/'rr3_flt_gc_decomposition.csv',index=False)
 genes=gene_programs();genes.to_csv(OUT/'rr3_39_vs_40_gene_programs.csv',index=False)
 figures(q,coords,loo,decomp,pc)
 signs=loo.groupby('duration_day').directional_cosine.agg(['min','max','median'])
 verdict='G. MIXED / UNRESOLVED'
 text=f'''# RR3 cohort audit summary\n\n- Samples: 39d 2 FLT/2 GC; 40d 3 FLT/2 GC; 41d 1 FLT/2 GC.\n- All cohorts share BALB/c female mouse liver, age, ribodepletion, paired 150-bp sequencing, UC Davis, liquid-nitrogen preservation, diet, habitat, euthanasia method, and stranded library processing.\n- Animal identity, duration/exposure, collection cohort, radiation dose for FLT animals, RIN, read depth, rRNA contamination, and library index differ. Exact euthanasia times/order and dissection times/order are unavailable locally.\n- Duration and collection/animal cohort are inseparable; no independent duration replication exists.\n- Leave-one-out directional ranges: {signs.to_dict('index')}.\n- Bridge response-vector cosine between 39d and 40d: {cos(responses[39],responses[40]):.4f}. The conventional expression response cosine is recorded separately in rr3_flt_gc_decomposition.csv.\n- Classification: **{verdict}**. The configurations are technically reproducible but not biologically explained as a duration transition.\n'''
 (OUT/'rr3_cohort_audit_summary.md').write_text(text)
 (OUT/'provenance.json').write_text(json.dumps({'inputs':[str(API),str(RES/'task3b_contrast_sample_membership.csv'),str(WORK/'bridgerna_embeddings.npy'),str(WORK/'bridgerna_log1p_tpm_inputs.npy')],'model_inference':False,'classification':verdict,'limitations':['Duration, animal identity, and collection cohort are confounded.','RR3-39 has n=2/2; RR3-41 has n=1/2.','Unavailable metadata are reported as unavailable rather than inferred.']},indent=2)+'\n')
 print(text);print(decomp.to_string(index=False));print('[complete]',OUT)
if __name__=='__main__':main()
