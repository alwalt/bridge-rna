#!/usr/bin/env python3
"""Strict same-animal technical replication for RR1, RR3-39, and RR3-40."""
from __future__ import annotations
import json,re
from itertools import product
from pathlib import Path
import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE=Path(__file__).resolve().parents[1];REPO=HERE.parents[1];OUT=HERE/'results/task4_rr1_rr3_paired_technical_replication';FIG=OUT/'figures'
T3=REPO/'benchmarks/osdr_batch_effect_representation';R3=T3/'results';W3=T3/'work';GMTROOT=REPO/'benchmarks/cross_species_exercise_response/results/per_study_ranked_gsea';GMTS={'GO:BP':'GO_Biological_Process_2026.gmt','KEGG':'KEGG_2026.gmt','Reactome':'Reactome_Pathways_2024.gmt'}
GENES=np.asarray(pd.read_csv(REPO/'data/ensembl/canonical_genes.csv').gene_symbol);G=len(GENES);SEED=66791
DESIGN={
 'RR1':{'FLT':['M25','M26','M28','M30'],'GC':['M36','M37','M38','M39','M40'],'original':'OSD-48','remeasure':'OSD-168 no-ERCC'},
 'RR3-39':{'FLT':['F1','F2'],'GC':['G1','G2'],'original':'OSD-137','remeasure':'OSD-168 ERCC'},
 'RR3-40':{'FLT':['F3','F4'],'GC':['G3','G5'],'original':'OSD-137','remeasure':'OSD-168 ERCC'}}

def cos(a,b):
 d=np.linalg.norm(a)*np.linalg.norm(b);return float(a@b/d) if d else np.nan
def basis():
 m=pd.read_parquet(HERE/'work/datasets/chen_2020_tcells/manifest.parquet').reset_index(drop=True);z=np.load(HERE/'work/datasets/chen_2020_tcells/bridgerna_embeddings.npy').astype(float);d=[]
 for _,q in m.groupby('pair_id',sort=True):d.append(z[q.index[q.library_prep.eq('ribo')]].mean(0)-z[q.index[q.library_prep.eq('polyA')]].mean(0))
 return np.linalg.svd(np.stack(d),full_matrices=False)[2][:2]

def resolve_samples():
 m=pd.read_csv(R3/'sample_manifest.csv');rows=[];lookup={}
 for cohort,spec in DESIGN.items():
  for measurement in ['original','remeasure']:
   osd='OSD-48' if spec[measurement]=='OSD-48' else 'OSD-137' if spec[measurement]=='OSD-137' else 'OSD-168'
   for cond,animals in [(x,spec[x]) for x in ['FLT','GC']]:
    for animal in animals:
     q=m[(m.OSD.eq(osd))&m.condition.eq(cond)&m.sample_id.str.endswith('_'+animal)]
     if cohort=='RR1' and measurement=='remeasure':q=q[q.sample_id.str.contains('RR1')&q.sample_id.str.contains('noERCC')]
     if cohort.startswith('RR3') and measurement=='remeasure':q=q[q.sample_id.str.contains('RR3')&q.sample_id.str.contains('wERCC')]
     if len(q)!=1:raise ValueError(f'{cohort} {measurement} {cond} {animal}: found {len(q)}')
     r=q.iloc[0];lookup[(cohort,measurement,animal)]=r.sample_id;rows.append({'cohort':cohort,'measurement':measurement,'OSD':osd,'condition':cond,'animal_id':animal,'sample_id':r.sample_id,'mission':r.mission,'library_preparation':r.library_preparation,'sequencing_facility':r.sequencing_facility,'sequencing_parameters':r.sequencing_parameters,'preservation':r.preservation,'strain':r.strain,'sex':r.sex,'age_at_launch':r.age_at_launch})
 d=pd.DataFrame(rows);d.to_csv(OUT/'animal_mapping.csv',index=False)
 # Explicit discrepancies between instructions and authoritative/local records.
 disc=pd.DataFrame([{'item':'RR1 unmatched FLT animals','instruction':'OSD-48 M27; OSD-168 M29','source_metadata':'confirmed; neither has a counterpart','impact':'both excluded from strict comparison'},
  {'item':'RR3-40 full versus strict','instruction':'full F3/F4/F5; strict F3/F4','source_metadata':'confirmed; F5 lacks OSD-168 counterpart','impact':'F5 excluded only from strict pair'},
  {'item':'RR3-41','instruction':'F6 vs G6/G7','source_metadata':'confirmed; absent from OSD-168','impact':'no technical replication possible'},
  {'item':'OSD-137 G5 naming','instruction':'G5','source_metadata':'local/audited G5; API RNA extract field contains G7','impact':'retained audited G5 mapping and flagged ambiguity'}]);disc.to_csv(OUT/'mapping_discrepancies.csv',index=False);return m,d,lookup

def metadata_audit(mapping):
 prior=pd.read_csv(HERE/'results/task4_rr1_rr3_sample_pc12_audit/master_sample_level_audit.parquet') if False else pd.read_parquet(HERE/'results/task4_rr1_rr3_sample_pc12_audit/master_sample_level_audit.parquet')
 fields=['strain','sex','age_at_launch','duration_api','habitat' if 'habitat' in prior else None,'sample_preservation_api','RNA_extraction_method','library_selection_api','library_kit','ERCC_condition','sequencing_platform','library_layout','sequencing_facility','RIN','collection_time','dissection_condition','postmortem_interval','processing_batch','RNA_extraction_batch','sequencing_batch'];fields=[f for f in fields if f and f in prior]
 rows=[]
 for cohort in DESIGN:
  ids=mapping[mapping.cohort.eq(cohort)].sample_id;d=prior[prior.sample_id.isin(ids)]
  for field in fields:
   vals={meas:' | '.join(sorted(d[d.sample_id.isin(mapping[(mapping.cohort.eq(cohort))&(mapping.measurement.eq(meas))].sample_id)][field].dropna().astype(str).unique())) or 'NA' for meas in ['original','remeasure']}
   rows.append({'cohort':cohort,'factor':field,'original_measurement':vals['original'],'remeasurement':vals['remeasure'],'same_or_different':'same' if vals['original']==vals['remeasure'] else 'different','potentially_relevant':'yes' if vals['original']!=vals['remeasure'] else 'controlled/reported same'})
 out=pd.DataFrame(rows);out.to_csv(OUT/'metadata_audit.csv',index=False);return out

def vectors(m,mapping):
 ix=dict(zip(m.sample_id,range(len(m))));z=np.load(W3/'bridgerna_embeddings.npy');samples={};responses={}
 for cohort in DESIGN:
  samples[cohort]={}
  for meas in ['original','remeasure']:
   samples[cohort][meas]={}
   for cond in ['FLT','GC']:
    ids=mapping[(mapping.cohort.eq(cohort))&(mapping.measurement.eq(meas))&(mapping.condition.eq(cond))].sample_id;samples[cohort][meas][cond]=np.stack([z[ix[s]] for s in ids])
   responses[(cohort,meas)]=samples[cohort][meas]['FLT'].mean(0)-samples[cohort][meas]['GC'].mean(0)
 # Secondary full RR3-40 and full RR1 cohorts from established definitions.
 mem=pd.read_csv(R3/'task3b_contrast_sample_membership.csv')
 for label,cid in [('RR3-40 full','C02__OSD-137__RR3__40-day'),('RR1 carcass full','C14__OSD-48__RR1-NASA__37-day')]:
  q=mem[mem.contrast_id.eq(cid)];responses[(label,'original')]=z[q[q.condition.eq('FLT')].sample_id.map(ix)].mean(0)-z[q[q.condition.eq('GC')].sample_id.map(ix)].mean(0)
 return z,ix,samples,responses

def response_metrics(samples,responses,B):
 rows=[]
 for cohort in DESIGN:
  a,b=responses[(cohort,'original')],responses[(cohort,'remeasure')]
  for meas,v in [('original',a),('remeasure',b)]:
   p=v@B.T;rows.append({'cohort':cohort,'measurement':meas,'response_norm':np.linalg.norm(v),'PC1':p[0],'PC2':p[1],'PC1_2_energy_fraction':np.sum(p*p)/np.sum(v*v)})
  rows.append({'cohort':cohort,'measurement':'technical comparison','technical_response_cosine':cos(a,b),'technical_response_euclidean':np.linalg.norm(a-b)})
 # F5 secondary effect.
 a=responses[('RR3-40','original')];full=responses[('RR3-40 full','original')];rows.append({'cohort':'RR3-40','measurement':'strict vs full original','technical_response_cosine':cos(a,full),'technical_response_euclidean':np.linalg.norm(a-full)})
 for meas in ['original','remeasure']:
  a=responses[('RR3-39',meas)];b=responses[('RR3-40',meas)];rows.append({'cohort':'RR3-39 vs RR3-40','measurement':meas,'technical_response_cosine':cos(a,b),'technical_response_euclidean':np.linalg.norm(a-b)})
 d=pd.DataFrame(rows);d.to_csv(OUT/'response_replication_metrics.csv',index=False);return d

def displacements(mapping,z,ix,samples,responses,B):
 rows=[];summary=[]
 for cohort in DESIGN:
  ds={}
  for cond in ['FLT','GC']:
   rr=[]
   for animal in DESIGN[cohort][cond]:
    a=mapping[(mapping.cohort.eq(cohort))&(mapping.measurement.eq('original'))&(mapping.animal_id.eq(animal))].sample_id.iloc[0];b=mapping[(mapping.cohort.eq(cohort))&(mapping.measurement.eq('remeasure'))&(mapping.animal_id.eq(animal))].sample_id.iloc[0];v=z[ix[b]]-z[ix[a]];rr.append(v);rows.append({'cohort':cohort,'condition':cond,'animal_id':animal,'original_sample':a,'remeasurement_sample':b,'displacement_norm':np.linalg.norm(v),'PC1_projection':v@B[0],'PC2_projection':v@B[1]})
   ds[cond]=np.stack(rr)
  mean_all=np.concatenate([ds['FLT'],ds['GC']]).mean(0)
  for r in rows:
   if r['cohort']==cohort:
    animal=r['animal_id'];cond=r['condition'];i=DESIGN[cohort][cond].index(animal);r['cosine_with_mean_displacement']=cos(ds[cond][i],mean_all)
  df,dg=ds['FLT'].mean(0),ds['GC'].mean(0);identity=(responses[(cohort,'remeasure')]-responses[(cohort,'original')])-(df-dg)
  summary.append({'cohort':cohort,'D_FLT_D_GC_cosine':cos(df,dg),'D_FLT_norm':np.linalg.norm(df),'D_GC_norm':np.linalg.norm(dg),'D_FLT_minus_D_GC_norm':np.linalg.norm(df-dg),'identity_error':np.linalg.norm(identity),'technical_response_cosine':cos(responses[(cohort,'original')],responses[(cohort,'remeasure')]),'response_preserved':cos(responses[(cohort,'original')],responses[(cohort,'remeasure')])>=.75})
 pd.DataFrame(rows).to_csv(OUT/'paired_animal_displacements.csv',index=False);s=pd.DataFrame(summary);s.to_csv(OUT/'displacement_summary.csv',index=False);return pd.DataFrame(rows),s

def bootstrap(samples,responses,B,reps=5000):
 rng=np.random.default_rng(SEED);boot=[];loo=[]
 for cohort in DESIGN:
  for i in range(reps):
   draw={cond:rng.integers(len(samples[cohort]['original'][cond]),size=len(samples[cohort]['original'][cond])) for cond in ['FLT','GC']};rr={}
   for meas in ['original','remeasure']:
    means={cond:samples[cohort][meas][cond][draw[cond]].mean(0) for cond in ['FLT','GC']};rr[meas]=means['FLT']-means['GC']
   delta=rr['remeasure']-rr['original'];p=delta@B.T;boot.append({'cohort':cohort,'replicate':i,'technical_response_cosine':cos(rr['original'],rr['remeasure']),'discrepancy_norm':np.linalg.norm(delta),'PC1_response_original':rr['original']@B[0],'PC2_response_original':rr['original']@B[1],'PC1_response_remeasure':rr['remeasure']@B[0],'PC2_response_remeasure':rr['remeasure']@B[1]})
  for cond in ['FLT','GC']:
   for j,animal in enumerate(DESIGN[cohort][cond]):
    rr={}
    for meas in ['original','remeasure']:
     means={c:np.delete(samples[cohort][meas][c],j,axis=0).mean(0) if c==cond else samples[cohort][meas][c].mean(0) for c in ['FLT','GC']};rr[meas]=means['FLT']-means['GC']
    loo.append({'cohort':cohort,'removed_condition':cond,'removed_animal':animal,'technical_response_cosine':cos(rr['original'],rr['remeasure'])})
 b=pd.DataFrame(boot);b.to_parquet(OUT/'bootstrap_results.parquet',index=False);l=pd.DataFrame(loo);l.to_csv(OUT/'leave_one_animal_out.csv',index=False);summary=b.groupby('cohort').technical_response_cosine.agg(mean='mean',median='median',low=lambda x:x.quantile(.025),high=lambda x:x.quantile(.975)).reset_index();summary.to_csv(OUT/'bootstrap_summary.csv',index=False);return b,l

def attribution(B):
 names=['RR1 OSD48','RR1 OSD168','RR3-39 OSD137','RR3-39 OSD168','RR3-40 OSD137','RR3-40 OSD168'];mm=np.memmap(HERE/'work/task4_rna_processing_gene_analysis/matched_contextual_responses.float16.dat',dtype='float16',mode='r',shape=(6,G,512));profiles={n:(np.asarray(mm[i],np.float32)@B.T/G) for i,n in enumerate(names)};rows=[]
 for cohort,a,b in [('RR1',names[0],names[1]),('RR3-39',names[2],names[3]),('RR3-40',names[4],names[5])]:
  x,y=profiles[a].reshape(-1),profiles[b].reshape(-1);mx=np.linalg.norm(profiles[a],axis=1);my=np.linalg.norm(profiles[b],axis=1);row={'cohort':cohort,'geometry_cosine':np.nan,'attribution_cosine':cos(x,y),'attribution_spearman':spearmanr(x,y).statistic}
  for n in [100,500]:
   sa=set(np.argpartition(mx,-n)[-n:]);sb=set(np.argpartition(my,-n)[-n:]);ov=list(sa&sb);row[f'top{n}_overlap']=len(ov);row[f'top{n}_direction_agreement']=np.mean(np.sign(profiles[a][ov])==np.sign(profiles[b][ov]))
  rows.append(row)
 d=pd.DataFrame(rows);d.to_csv(OUT/'attribution_replication.csv',index=False);return profiles,d

def pathways(profiles):
 terms=[]
 for source,file in GMTS.items():
  for term,members in gp.parser.read_gmt(path=str(GMTROOT/file)).items():
   ii=np.flatnonzero(np.isin(GENES,list(set(members))))
   if 10<=len(ii)<=500:terms.append((source,term,ii))
 rows=[]
 for name,c in profiles.items():
  s=c[:,0]+c[:,1];s=(s-s.mean())/(s.std() or 1)
  for source,term,ii in terms:rows.append({'response':name,'source':source,'pathway':term,'score':s[ii].mean()*np.sqrt(len(ii))})
 d=pd.DataFrame(rows);d.to_parquet(OUT/'pathway_profiles.parquet',index=False);w=d.pivot(index=['source','pathway'],columns='response',values='score');out=[]
 for cohort,a,b in [('RR1','RR1 OSD48','RR1 OSD168'),('RR3-39','RR3-39 OSD137','RR3-39 OSD168'),('RR3-40','RR3-40 OSD137','RR3-40 OSD168')]:
  shared=w[a]*w[b];discord=(w[a]-w[b]).abs();out.append({'cohort':cohort,'pathway_pearson':w[a].corr(w[b]),'pathway_spearman':w[a].corr(w[b],method='spearman'),'leading_shared_pathways':' | '.join(shared.nlargest(5).index.get_level_values('pathway')),'leading_discordant_pathways':' | '.join(discord.nlargest(5).index.get_level_values('pathway'))})
 o=pd.DataFrame(out);o.to_csv(OUT/'pathway_replication.csv',index=False);return o

def figures(mapping,z,ix,samples,responses,disp,B):
 plt.style.use('seaborn-v0_8-whitegrid');FIG.mkdir(exist_ok=True);fig,axes=plt.subplots(1,3,figsize=(15,5))
 for ax,cohort in zip(axes,DESIGN):
  for meas,color in [('original','#4C78A8'),('remeasure','#E45756')]:
   p=responses[(cohort,meas)]@B.T;ax.arrow(0,0,p[0],p[1],color=color,width=.003,length_includes_head=True,label=meas);ax.text(p[0],p[1],meas,fontsize=8)
  ax.set(title=cohort,xlabel='ΔPC1',ylabel='ΔPC2');ax.axhline(0,color='grey',lw=.6);ax.axvline(0,color='grey',lw=.6)
 fig.suptitle('Strict matched-animal response-vector technical replication');fig.tight_layout()
 for e in ['png','pdf']:fig.savefig(FIG/f'response_replication.{e}',dpi=400)
 plt.close(fig);fig,axes=plt.subplots(1,3,figsize=(16,5))
 for ax,cohort in zip(axes,DESIGN):
  q=disp[disp.cohort.eq(cohort)]
  for cond,color in [('FLT','#E45756'),('GC','#4C78A8')]:
   g=q[q.condition.eq(cond)];ax.quiver(np.zeros(len(g)),np.zeros(len(g)),g.PC1_projection,g.PC2_projection,color=color,angles='xy',scale_units='xy',scale=1,alpha=.55,label=cond)
   for _,r in g.iterrows():ax.text(r.PC1_projection,r.PC2_projection,r.animal_id,fontsize=7)
   ax.arrow(0,0,g.PC1_projection.mean(),g.PC2_projection.mean(),color=color,width=.004,length_includes_head=True)
  ax.set(title=cohort,xlabel='technical displacement PC1',ylabel='technical displacement PC2');ax.legend()
 fig.suptitle('Paired technical displacement by condition');fig.tight_layout()
 for e in ['png','pdf']:fig.savefig(FIG/f'paired_displacements.{e}',dpi=400)
 plt.close(fig)
 # RR3 four states for both measurements.
 fig,axes=plt.subplots(1,2,figsize=(13,6))
 for ax,meas in zip(axes,['original','remeasure']):
  for cohort,color in [('RR3-39','#4C78A8'),('RR3-40','#F58518')]:
   means={c:samples[cohort][meas][c].mean(0)@B.T for c in ['FLT','GC']};ax.scatter(*means['GC'],marker='o',color=color);ax.scatter(*means['FLT'],marker='^',color=color);ax.annotate('',xy=means['FLT'],xytext=means['GC'],arrowprops={'arrowstyle':'->','color':color,'lw':2});ax.text(*means['FLT'],cohort,fontsize=8)
  ax.set(title='OSD-137' if meas=='original' else 'OSD-168',xlabel='PC1',ylabel='PC2')
 fig.suptitle('Opposing RR3 subgroup responses across technical remeasurement');fig.tight_layout()
 for e in ['png','pdf']:fig.savefig(FIG/f'rr3_four_state_replication.{e}',dpi=400)
 plt.close(fig)
 # RR1 each animal original→remeasurement.
 fig,ax=plt.subplots(figsize=(9,7));q=mapping[mapping.cohort.eq('RR1')]
 for cond,color in [('FLT','#E45756'),('GC','#4C78A8')]:
  for animal in DESIGN['RR1'][cond]:
   a=q[(q.measurement.eq('original'))&q.animal_id.eq(animal)].sample_id.iloc[0];b=q[(q.measurement.eq('remeasure'))&q.animal_id.eq(animal)].sample_id.iloc[0];pa=z[ix[a]]@B.T;pb=z[ix[b]]@B.T;ax.scatter(*pa,color=color,marker='o');ax.scatter(*pb,color=color,marker='x');ax.annotate('',xy=pb,xytext=pa,arrowprops={'arrowstyle':'->','color':color,'alpha':.65});ax.text(*pb,animal,fontsize=7)
 ax.set(xlabel='PC1',ylabel='PC2',title='RR1 matched animals: OSD-48 original → OSD-168 no-ERCC');fig.tight_layout()
 for e in ['png','pdf']:fig.savefig(FIG/f'rr1_matched_animals.{e}',dpi=400)
 plt.close(fig)

def main():
 OUT.mkdir(parents=True,exist_ok=True);B=basis();m,mapping,lookup=resolve_samples();meta=metadata_audit(mapping);z,ix,samples,responses=vectors(m,mapping);metrics=response_metrics(samples,responses,B);disp,ds=displacements(mapping,z,ix,samples,responses,B);boot,loo=bootstrap(samples,responses,B);profiles,attr=attribution(B);path=pathways(profiles);figures(mapping,z,ix,samples,responses,disp,B)
 final=ds.merge(attr.drop(columns='geometry_cosine'),on='cohort').merge(path,on='cohort');final.to_csv(OUT/'technical_replication_summary.csv',index=False)
 fi=final.set_index('cohort');bs=pd.read_csv(OUT/'bootstrap_summary.csv').set_index('cohort');lo=pd.read_csv(OUT/'leave_one_animal_out.csv').groupby('cohort').technical_response_cosine.agg(['min','max']);questions=[
  ('Same animals available across measurements?','Yes: 4 FLT/5 GC','Yes: 2/2','Yes: 2/2'),('Technical response geometry reproduced?',f"No: {fi.loc['RR1','technical_response_cosine']:.3f}",f"Yes: {fi.loc['RR3-39','technical_response_cosine']:.3f}",f"Yes: {fi.loc['RR3-40','technical_response_cosine']:.3f}"),('Attribution reproduced?',f"Weak/opposing: {fi.loc['RR1','attribution_cosine']:.3f}",f"Strong: {fi.loc['RR3-39','attribution_cosine']:.3f}",f"Strong: {fi.loc['RR3-40','attribution_cosine']:.3f}"),('Pathway profile reproduced?',f"Weak: {fi.loc['RR1','pathway_pearson']:.3f}",f"Strong: {fi.loc['RR3-39','pathway_pearson']:.3f}",f"Strong: {fi.loc['RR3-40','pathway_pearson']:.3f}"),('FLT and GC undergo similar technical displacement?',f"Same direction ({fi.loc['RR1','D_FLT_D_GC_cosine']:.3f}) but unequal magnitude",f"Broadly ({fi.loc['RR3-39','D_FLT_D_GC_cosine']:.3f})",f"Yes ({fi.loc['RR3-40','D_FLT_D_GC_cosine']:.3f})"),('Condition-dependent measurement displacement?',f"Strong magnitude difference; ||DF-DG||={fi.loc['RR1','D_FLT_minus_D_GC_norm']:.3f}",f"Smaller: {fi.loc['RR3-39','D_FLT_minus_D_GC_norm']:.3f}",f"Smallest: {fi.loc['RR3-40','D_FLT_minus_D_GC_norm']:.3f}"),('Response robust to leave-one-out?',f"All negative [{lo.loc['RR1','min']:.3f},{lo.loc['RR1','max']:.3f}]",f"Positive [{lo.loc['RR3-39','min']:.3f},{lo.loc['RR3-39','max']:.3f}]",f"Positive [{lo.loc['RR3-40','min']:.3f},{lo.loc['RR3-40','max']:.3f}]"),('Major metadata differences?','Library selection, kit/read setup, ERCC, handling and OSD','ERCC/resequencing; same animals/library class','ERCC/resequencing; same animals/library class'),('Biological interpretation trustworthy?','No: response measurement-sensitive','Technically reproduced; mechanism still unresolved','Technically reproduced; mechanism still unresolved'),('Remaining confounders?','Compound protocol transition','Tiny n; collection subgroup/duration','Tiny n; collection subgroup/duration')]
 pd.DataFrame(questions,columns=['Question','RR1','RR3-39','RR3-40']).to_csv(OUT/'decision_table.csv',index=False);(OUT/'summary.md').write_text('# Paired technical replication\n\nStrict same-animal RR1, RR3-39, and RR3-40 comparison using the unchanged controlled T-cell PC1–2 reference. See machine-readable tables for results.\n');(OUT/'provenance.json').write_text(json.dumps({'RR1_primary_remeasurement':'OSD-168 no-ERCC, preserving established -0.804 comparison','RR3_remeasurement':'OSD-168 ERCC','basis':'unchanged T-cell PC1-2','bootstrap_replicates':5000,'correction':False},indent=2)+'\n');print(final.to_string(index=False));print('[complete]',OUT)
if __name__=='__main__':main()
