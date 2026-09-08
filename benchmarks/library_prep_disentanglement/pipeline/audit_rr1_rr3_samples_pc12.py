#!/usr/bin/env python3
"""Sample-first RR1/RR3 audit in the fixed controlled T-cell PC1-2 basis."""
from __future__ import annotations
import json,re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f_oneway,spearmanr
import statsmodels.formula.api as smf

HERE=Path(__file__).resolve().parents[1];REPO=HERE.parents[1];OUT=HERE/'results/task4_rr1_rr3_sample_pc12_audit';FIG=OUT/'figures'
T3=REPO/'benchmarks/osdr_batch_effect_representation';R3=T3/'results';W3=T3/'work';API=R3/'task3_manifest_api_validation/api_sample_metadata_combined.csv';SEED=80831

def basis_and_center():
 m=pd.read_parquet(HERE/'work/datasets/chen_2020_tcells/manifest.parquet').reset_index(drop=True);z=np.load(HERE/'work/datasets/chen_2020_tcells/bridgerna_embeddings.npy').astype(float);d=[]
 for _,q in m.groupby('pair_id',sort=True):d.append(z[q.index[q.library_prep.eq('ribo')]].mean(0)-z[q.index[q.library_prep.eq('polyA')]].mean(0))
 return np.linalg.svd(np.stack(d),full_matrices=False)[2][:2],z.mean(0)

def cohort_for(osd,s):
 if osd=='OSD-48':return 'RR1 carcass' if '_C_' in s else 'RR1 upon-euthanasia' if '_I_' in s else 'RR1 other'
 if osd=='OSD-137':
  a=re.search(r'_(F\d+|G\d+)$',s);a=a.group(1) if a else ''
  if a in {'F1','F2','G1','G2'}:return 'RR3-39'
  if a in {'F3','F4','F5','G3','G5'}:return 'RR3-40'
  if a in {'F6','G6','G7'}:return 'RR3-41'
 if osd=='OSD-168' and 'RR1' in s:return 'RR1 Ribo remeasurement'
 if osd=='OSD-168' and 'RR3' in s:
  a=re.search(r'_(F\d+|G\d+)$',s);a=a.group(1) if a else ''
  if a in {'F1','F2','G1','G2'}:return 'RR3-39 technical remeasurement'
  if a in {'F3','F4','G3','G5'}:return 'RR3-40 technical remeasurement'
 return 'unassigned'

def api_fields():
 a=pd.read_csv(API,low_memory=False);a=a[a['id.accession'].isin(['OSD-48','OSD-137','OSD-168'])].copy();key='assay.sample name';a=a.drop_duplicates(key).set_index(key)
 fields={'assay_accession':'id.assay name','api_sample_accession':'id.sample name','age_api':'study.characteristics.age at launch','strain_api':'study.characteristics.strain','sex_api':'study.characteristics.sex','tissue_api':'study.characteristics.material type','duration_api':'study.parameter value.duration','exposure_duration_api':'study.parameter value.exposure duration','euthanasia_method':'study.parameter value.euthanasia method','sample_preservation_api':'study.parameter value.sample preservation method','storage_temperature':'study.parameter value.sample storage temperature','carcass_preservation':'study.parameter value.carcass preservation method','dissection_condition':'study.factor value.dissection condition','collection_time':'study.factor value.euthanasia location','RNA_extraction_method':'assay.extract name','RIN':'assay.parameter value.qa score','library_kit':'assay.parameter value.library kit','library_layout':'assay.parameter value.library layout','library_selection_api':'assay.parameter value.library selection','sequencing_platform':'assay.parameter value.sequencing instrument','read_length':'assay.parameter value.read length','sequencing_depth':'assay.parameter value.read depth','rrna_contamination':'assay.parameter value.rrna contamination','library_batch':'assay.parameter value.spike-in mix number','ERCC_condition':'study.factor value.spike-in quality control','processing_date':'assay.data transformation name'}
 out=pd.DataFrame(index=a.index)
 for new,old in fields.items():out[new]=a[old] if old in a else pd.NA
 return out

def master():
 m=pd.read_csv(R3/'sample_manifest.csv');keep=m.OSD.eq('OSD-48')|(m.OSD.eq('OSD-137')&m.sample_id.str.contains('BAL-TAL_LVR'))|(m.OSD.eq('OSD-168')&m.sample_id.str.contains('RR1|RR3'));m=m[keep].copy();m['cohort']=[cohort_for(o,s) for o,s in zip(m.OSD,m.sample_id)];m=m[m.cohort.ne('unassigned')].copy();m['animal_id']=m.sample_id.str.extract(r'_(M\d+|F\d+|G\d+)$')[0];m['experiment']=m.mission;m['subgroup']=m.cohort;m['sample_state']=np.select([m.cohort.str.contains('carcass',case=False),m.cohort.str.contains('euthanasia',case=False)],['carcass','upon-euthanasia'],default='not equivalent/unspecified');m['embedding_id']=m.sample_id
 a=api_fields();m=m.join(a,on='sample_id');
 # Prefer authoritative API fields, retaining local values and all missingness.
 m['RIN_numeric']=pd.to_numeric(m.RIN.astype(str).str.extract(r'([0-9]+(?:\.[0-9]+)?)')[0],errors='coerce');m['read_depth_numeric']=pd.to_numeric(m.sequencing_depth.astype(str).str.extract(r'([0-9]+)')[0],errors='coerce');m['postmortem_interval']=pd.NA;m['RNA_extraction_batch']=pd.NA;m['dissection_batch']=pd.NA;m['sequencing_batch']=pd.NA;m['processing_batch']=pd.NA
 B,center=basis_and_center();allm=pd.read_csv(R3/'sample_manifest.csv');ix=dict(zip(allm.sample_id,range(len(allm))));z=np.load(W3/'bridgerna_embeddings.npy');emb=np.stack([z[ix[s]] for s in m.sample_id]);centered=emb-center;coords=emb@B.T;cc=centered@B.T;m['embedding_index']=[ix[s] for s in m.sample_id];m['embedding_norm']=np.linalg.norm(emb,axis=1);m['TCell_PC1_score']=coords[:,0];m['TCell_PC2_score']=coords[:,1];m['PC12_magnitude']=np.linalg.norm(coords,axis=1);m['PC12_angle_degrees']=np.degrees(np.arctan2(coords[:,1],coords[:,0]));m['centered_PC12_fraction']=np.sum(cc**2,axis=1)/np.maximum(np.sum(centered**2,axis=1),1e-12);m['outside_PC12_magnitude']=np.linalg.norm(centered-(cc@B),axis=1);m['extreme_PC1']=np.abs((m.TCell_PC1_score-m.TCell_PC1_score.mean())/m.TCell_PC1_score.std())>=2;m['extreme_PC2']=np.abs((m.TCell_PC2_score-m.TCell_PC2_score.mean())/m.TCell_PC2_score.std())>=2
 cent=m.groupby(['cohort','condition'])[['TCell_PC1_score','TCell_PC2_score']].transform('mean');m['within_cohort_condition_PC12_distance']=np.linalg.norm(m[['TCell_PC1_score','TCell_PC2_score']].to_numpy()-cent.to_numpy(),axis=1);threshold=m.groupby(['cohort','condition']).within_cohort_condition_PC12_distance.transform(lambda x:x.mean()+2*x.std(ddof=0));m['within_cohort_condition_extreme']=m.within_cohort_condition_PC12_distance>threshold
 m.to_parquet(OUT/'master_sample_level_audit.parquet',index=False);return m,emb,B

def bootstrap_response(flt,gc,B,reps=1000):
 rng=np.random.default_rng(SEED);rows=[]
 for _ in range(reps):
  d=flt[rng.integers(len(flt),size=len(flt))].mean(0)-gc[rng.integers(len(gc),size=len(gc))].mean(0);p=d@B.T
  rows.append([np.linalg.norm(d),p[0],p[1],np.linalg.norm(p),np.sum(p*p)/np.sum(d*d)])
 return np.quantile(rows,[.025,.5,.975],axis=0)

def collapse_animals(q,emb):
 q=q.copy();q['_i']=range(len(q));return np.stack([emb[g._i].mean(0) for _,g in q.groupby('animal_id',sort=True)])

def cohorts(m,emb,B):
 rows=[];boot=[]
 for cohort,q in m.groupby('cohort',sort=False):
  local=emb[q.index.to_numpy()-m.index.min()] if np.array_equal(m.index,np.arange(len(m))) else np.stack([emb[list(m.index).index(i)] for i in q.index])
  # Collapse ERCC/no-ERCC profiles from the same animal before condition means.
  groups={c:collapse_animals(q[q.condition.eq(c)].reset_index(drop=True),local[q.condition.eq(c).to_numpy()]) for c in ['FLT','GC']}
  means={c:v.mean(0) for c,v in groups.items()};d=means['FLT']-means['GC'];p=d@B.T;inside=np.linalg.norm(p);outside=np.linalg.norm(d-(p@B));ci=bootstrap_response(groups['FLT'],groups['GC'],B)
  def mean_col(cond,col):return pd.to_numeric(q.loc[q.condition.eq(cond),col],errors='coerce').mean()
  rows.append({'OSD':q.OSD.mode().iloc[0],'mission':q.mission.mode().iloc[0],'cohort':cohort,'sample_state':q.sample_state.mode().iloc[0],'library_prep':q.library_preparation.mode().iloc[0],
   'FLT_animals':' | '.join(sorted(q.loc[q.condition.eq('FLT'),'animal_id'].dropna().unique())),'GC_animals':' | '.join(sorted(q.loc[q.condition.eq('GC'),'animal_id'].dropna().unique())),'n_FLT':len(groups['FLT']),'n_GC':len(groups['GC']),
   'mean_FLT_RIN':mean_col('FLT','RIN_numeric'),'mean_GC_RIN':mean_col('GC','RIN_numeric'),'RIN_difference_FLT_minus_GC':mean_col('FLT','RIN_numeric')-mean_col('GC','RIN_numeric'),'preservation':' | '.join(sorted(q.sample_preservation_api.dropna().astype(str).unique())),
   'sequencing_platform':' | '.join(sorted(q.sequencing_platform.dropna().astype(str).unique())),'sequencing_facility':' | '.join(sorted(q.sequencing_facility.dropna().astype(str).unique())),'library_kit':' | '.join(sorted(q.library_kit.dropna().astype(str).unique())),'read_configuration':' | '.join(sorted(q.library_layout.dropna().astype(str).unique())),'read_length':' | '.join(sorted(q.read_length.dropna().astype(str).unique())),
   'GC_mean_PC1':means['GC']@B[0],'GC_mean_PC2':means['GC']@B[1],'FLT_mean_PC1':means['FLT']@B[0],'FLT_mean_PC2':means['FLT']@B[1],'delta_PC1':p[0],'delta_PC2':p[1],
   'full_response_norm':np.linalg.norm(d),'PC1_2_response_magnitude':inside,'response_energy_fraction_PC1_2':inside**2/np.dot(d,d),'outside_PC1_2_response_magnitude':outside,
   'response_norm_bootstrap_low':ci[0,0],'response_norm_bootstrap_median':ci[1,0],'response_norm_bootstrap_high':ci[2,0],'occupancy_bootstrap_low':ci[0,4],'occupancy_bootstrap_median':ci[1,4],'occupancy_bootstrap_high':ci[2,4]})
 d=pd.DataFrame(rows);d.to_csv(OUT/'cohort_level_audit.csv',index=False);return d

def paired_position_tables(c):
 rows=[]
 for comparison,cohort_names in [('RR1',['RR1 carcass','RR1 Ribo remeasurement']),('RR3-40',['RR3-40','RR3-40 technical remeasurement'])]:
  for cohort in cohort_names:
   r=c[c.cohort.eq(cohort)].iloc[0]
   for metric in ['GC_mean_PC1','GC_mean_PC2','FLT_mean_PC1','FLT_mean_PC2','delta_PC1','delta_PC2','PC1_2_response_magnitude','outside_PC1_2_response_magnitude','mean_FLT_RIN','mean_GC_RIN','n_FLT','n_GC','library_prep','sample_state','preservation','sequencing_platform','sequencing_facility','library_kit','read_configuration','read_length']:
    rows.append({'comparison':comparison,'cohort':cohort,'metric':metric,'value':r[metric]})
 d=pd.DataFrame(rows);d.to_csv(OUT/'technical_pair_position_audit_long.csv',index=False);d.pivot(index=['comparison','metric'],columns='cohort',values='value').reset_index().to_csv(OUT/'technical_pair_position_audit_wide.csv',index=False)

def associations(m):
 outcomes=['TCell_PC1_score','TCell_PC2_score','PC12_magnitude','outside_PC12_magnitude'];factors=['condition','OSD','mission','cohort','library_preparation','sample_state','RIN_numeric','preservation','sequencing_platform','sequencing_facility','RNA_extraction_method','library_kit','library_layout','ERCC_condition'];rows=[]
 for f in factors:
  if f not in m:continue
  for y in outcomes:
   q=m[[f,y]].dropna()
   if f=='RIN_numeric':
    if len(q)>=3:r=spearmanr(q[f],q[y]);rows.append({'factor':f,'outcome':y,'type':'continuous Spearman','effect_size':r.statistic,'p_value':r.pvalue,'levels_or_n':len(q)})
   else:
    groups=[g[y].to_numpy() for _,g in q.groupby(f) if len(g)>=2]
    if len(groups)>=2:
     grand=q[y].mean();ssb=sum(len(g)*(g.mean()-grand)**2 for g in groups);sst=((q[y]-grand)**2).sum();eta=ssb/sst if sst else np.nan;stat,p=f_oneway(*groups);rows.append({'factor':f,'outcome':y,'type':'categorical eta-squared/ANOVA','effect_size':eta,'p_value':p,'levels_or_n':q[f].nunique()})
 d=pd.DataFrame(rows);d.to_csv(OUT/'metadata_latent_associations.csv',index=False);return d

def interactions(m):
 rows=[]
 # Dataset is the estimable composite context; OSD and library are not jointly identifiable in RR1.
 q=m[m.cohort.isin(['RR1 carcass','RR1 Ribo remeasurement'])].copy();q['dataset']=q.cohort
 for y in ['TCell_PC1_score','TCell_PC2_score']:
  fit=smf.ols(f'{y} ~ C(condition)*C(dataset)',q).fit()
  for term,val in fit.params.items():rows.append({'analysis':'RR1 FLT_GC × dataset composite','outcome':y,'term':term,'coefficient':val,'SE':fit.bse[term],'p_value':fit.pvalues[term],'n_profiles':len(q),'note':'OSD/library/ERCC-sequencing context structurally confounded'})
 # RIN is observational and incomplete; include only as a diagnostic interaction.
 q=m[m.RIN_numeric.notna()].copy()
 for y in ['TCell_PC1_score','TCell_PC2_score']:
  fit=smf.ols(f'{y} ~ C(condition)*RIN_numeric',q).fit()
  for term,val in fit.params.items():rows.append({'analysis':'FLT_GC × RIN descriptive','outcome':y,'term':term,'coefficient':val,'SE':fit.bse[term],'p_value':fit.pvalues[term],'n_profiles':len(q),'note':'cohort-confounded; not causal'})
 d=pd.DataFrame(rows);d.to_csv(OUT/'condition_technical_interactions.csv',index=False);return d

def confounded_table(c):
 rows=[
 ('Library preparation','polyA','ribodepletion','ribodepletion','ribodepletion','strongly associated','no / structurally confounded','Perfectly tracks OSD in RR1; RR3 provides a same-library stability control but not causal isolation.'),
 ('OSD','OSD-48','OSD-168','OSD-137','OSD-137/168','strongly associated','no / structurally confounded','OSD changes together with protocol and, for RR1, sample representation.'),
 ('FLT/GC','both','both','both','both','possible','yes within cohort','Defines the response; interaction with dataset is directly estimable.'),
 ('Sample state','carcass/upon-euthanasia','not equivalent/unspecified','liquid-nitrogen collection','liquid-nitrogen collection','strongly associated','partially','OSD-168 has no valid carcass/euthanasia mapping.'),
 ('RIN','available','available','available','available','possible','partially','Continuous but cohort-associated and missing for some profiles.'),
 ('Preservation','carcass or immediate-state dependent','liquid nitrogen/source storage','liquid nitrogen','liquid nitrogen','possible','no / structurally confounded','RR1 preservation changes with OSD and library protocol.'),
 ('Platform/read setup','HiSeq 3000; SE50','HiSeq 4000; PE150','HiSeq 4000; PE150','HiSeq 4000; PE150','strongly associated','no / structurally confounded','Changes jointly with RR1 OSD/library; largely matched for RR3.'),
 ('Facility','UC Davis','UC Davis','UC Davis','UC Davis','unlikely','no variation','Facility is held constant in available metadata.'),
 ('Animal composition','distinct representation/subset','5 FLT + 5 GC animals duplicated by ERCC','duration-specific animals','duration-specific/matched subset','possible','partially','Animal identity differs across biological cohorts; technical matches exist only for defined subsets.'),
 ('Extraction/batch','incomplete','incomplete','incomplete','incomplete','cannot determine','no / structurally confounded','Extraction, dissection, and sequencing batch fields are incomplete or unavailable.')]
 d=pd.DataFrame(rows,columns=['Factor','RR1 OSD-48','RR1 OSD-168','RR3-39','RR3-40','Could explain latent difference?','Separable from other factors?','Evidence']);d.to_csv(OUT/'what_is_confounded.csv',index=False);return d

def figures(m,c,B):
 plt.style.use('seaborn-v0_8-whitegrid');FIG.mkdir(exist_ok=True);panels=[('OSD-48 RR1',m.OSD.eq('OSD-48')),('OSD-168 RR1',m.OSD.eq('OSD-168')&m.mission.str.contains('RR1')),('OSD-137 RR3',m.OSD.eq('OSD-137')),('OSD-168 RR3',m.OSD.eq('OSD-168')&m.mission.str.contains('RR3'))]
 fig,axes=plt.subplots(2,2,figsize=(13,11))
 for ax,(title,mask) in zip(axes.flat,panels):
  q=m[mask]
  for cond,marker,color in [('FLT','^','#E45756'),('GC','o','#4C78A8')]:
   z=q[q.condition.eq(cond)];ax.scatter(z.TCell_PC1_score,z.TCell_PC2_score,marker=marker,color=color,label=cond)
   for _,r in z.iterrows():ax.annotate(str(r.animal_id),(r.TCell_PC1_score,r.TCell_PC2_score),fontsize=6)
  for cohort,g in q.groupby('cohort'):
   means=g.groupby('condition')[['TCell_PC1_score','TCell_PC2_score']].mean()
   if {'FLT','GC'}<=set(means.index):ax.annotate('',xy=means.loc['FLT'],xytext=means.loc['GC'],arrowprops={'arrowstyle':'->','lw':2})
  ax.set(title=title,xlabel='T-cell PC1 score',ylabel='T-cell PC2 score');ax.legend()
 fig.suptitle('Individual samples in controlled T-cell PolyA/Ribo-sensitive directions');fig.tight_layout()
 for e in ['png','pdf']:fig.savefig(FIG/f'individual_samples_by_panel.{e}',dpi=400)
 plt.close(fig)
 # Separate encodings avoid an unreadable combined plot.
 for field,name in [('library_preparation','library_prep'),('RIN_numeric','RIN'),('sample_state','sample_state'),('cohort','cohort')]:
  fig,ax=plt.subplots(figsize=(10,7));q=m
  if field=='RIN_numeric':sc=ax.scatter(q.TCell_PC1_score,q.TCell_PC2_score,c=q[field],cmap='viridis',s=55);fig.colorbar(sc,ax=ax,label='RIN')
  else:
   for value,g in q.groupby(field,dropna=False):ax.scatter(g.TCell_PC1_score,g.TCell_PC2_score,label=str(value),s=50)
   ax.legend(fontsize=7,bbox_to_anchor=(1.02,1),loc='upper left')
  ax.set(xlabel='T-cell PC1 score',ylabel='T-cell PC2 score',title=f'Sample positions colored by {name.replace("_"," ")}');fig.tight_layout()
  for e in ['png','pdf']:fig.savefig(FIG/f'positions_by_{name}.{e}',dpi=400,bbox_inches='tight')
  plt.close(fig)
 for cohorts,title,file in [(['RR1 carcass','RR1 upon-euthanasia','RR1 Ribo remeasurement'],'RR1: baseline, flight, and response','rr1_response_arrows'),(['RR3-40','RR3-40 technical remeasurement'],'RR3-40 technical replication','rr3_40_response_arrows')]:
  fig,ax=plt.subplots(figsize=(9,7))
  for cohort in cohorts:
   r=c[c.cohort.eq(cohort)].iloc[0];ax.scatter([r.GC_mean_PC1,r.FLT_mean_PC1],[r.GC_mean_PC2,r.FLT_mean_PC2],s=65);ax.annotate('',xy=(r.FLT_mean_PC1,r.FLT_mean_PC2),xytext=(r.GC_mean_PC1,r.GC_mean_PC2),arrowprops={'arrowstyle':'->','lw':2},label=cohort);ax.text(r.FLT_mean_PC1,r.FLT_mean_PC2,cohort,fontsize=8)
  ax.set(xlabel='T-cell PC1 score',ylabel='T-cell PC2 score',title=title);fig.tight_layout()
  for e in ['png','pdf']:fig.savefig(FIG/f'{file}.{e}',dpi=400)
  plt.close(fig)

def main():
 OUT.mkdir(parents=True,exist_ok=True);m,emb,B=master();m=m.reset_index(drop=True);c=cohorts(m,emb,B);paired_position_tables(c);a=associations(m);interactions(m);confounded_table(c);figures(m,c,B)
 cols=['sample_id','OSD','mission','cohort','condition','library_preparation','sample_state','RIN_numeric','TCell_PC1_score','TCell_PC2_score','PC12_magnitude','extreme_PC1','extreme_PC2','within_cohort_condition_PC12_distance','within_cohort_condition_extreme'];m[cols].sort_values(['OSD','cohort','condition','sample_id']).to_csv(OUT/'sample_PC1_PC2_table.csv',index=False)
 (OUT/'summary.md').write_text('# RR1/RR3 sample-level PC1–2 audit\n\nSee cohort, interaction, association, and confounding tables. PC1–2 is a controlled T-cell PolyA/Ribo-sensitive reference, not a technical-only subspace.\n');(OUT/'provenance.json').write_text(json.dumps({'basis':'unchanged controlled T-cell uncentered PC1-2','sample_center_for_fraction':'independent mean of the 80 controlled T-cell embeddings','bootstrap_replicates':1000,'bootstrap_unit':'animal after averaging ERCC/no-ERCC technical profiles','correction':False,'missing_metadata':'retained as NA'},indent=2)+'\n');print('samples',len(m),'cohorts',len(c));print(c[['OSD','cohort','n_FLT','n_GC','delta_PC1','delta_PC2','response_energy_fraction_PC1_2']].to_string(index=False));print('[complete]',OUT)
if __name__=='__main__':main()
