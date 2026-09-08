#!/usr/bin/env python3
"""Test sample-state explanations for RR1/RR3 PC1-2 response similarity."""
from __future__ import annotations
import json,re
from pathlib import Path
import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import hypergeom, pearsonr, spearmanr

ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[1]
OUT=ROOT/'results/task4_confounding_profiler/rr1_rr3_sample_state';FIG=OUT/'figures'
T3=REPO/'benchmarks/osdr_batch_effect_representation';RES=T3/'results';WORK=T3/'work'
GMTROOT=REPO/'benchmarks/cross_species_exercise_response/results/per_study_ranked_gsea'
GMT={'GO:BP':'GO_Biological_Process_2026.gmt','KEGG':'KEGG_2026.gmt','Reactome':'Reactome_Pathways_2024.gmt'}

def cos(a,b):
 d=np.linalg.norm(a)*np.linalg.norm(b);return float(a@b/d) if d else np.nan
def metric(a,b):return {'cosine':cos(a,b),'pearson':pearsonr(a,b).statistic,'spearman':spearmanr(a,b).statistic}
def controlled_basis():
 m=pd.read_parquet(ROOT/'work/datasets/chen_2020_tcells/manifest.parquet').reset_index(drop=True);z=np.load(ROOT/'work/datasets/chen_2020_tcells/bridgerna_embeddings.npy').astype(float);d=[]
 for _,g in m.groupby('pair_id',sort=True):d.append(z[g.index[g.library_prep.eq('ribo')][0]]-z[g.index[g.library_prep.eq('polyA')][0]])
 return np.linalg.svd(np.stack(d),full_matrices=False)[2][:2]
def remove(v,b):return v-(v@b.T)@b

def responses():
 q=np.load(RES/'task3b_bridgerna_response_vectors.npz',allow_pickle=True);return dict(zip(q['contrast_id'].astype(str),q['delta_z'].astype(float)))

def similarity_table(v,b):
 groups={'RR1 carcass':'C14__OSD-48__RR1-NASA__37-day','RR1 upon-euthanasia':'C13__OSD-48__RR1-NASA__37-day'}
 rr={'RR3-39':'C01__OSD-137__RR3__39-day','RR3-40':'C02__OSD-137__RR3__40-day'};rows=[]
 for a,ai in groups.items():
  for c,ci in rr.items():
   for state,x,y in [('raw',v[ai],v[ci]),('PC1-2 removed',remove(v[ai],b),remove(v[ci],b))]:rows.append({'RR1_group':a,'RR3_group':c,'representation':state,**metric(x,y)})
 return pd.DataFrame(rows)

def metadata():
 members=pd.read_csv(RES/'task3b_contrast_sample_membership.csv');wanted=['C13__OSD-48__RR1-NASA__37-day','C14__OSD-48__RR1-NASA__37-day','C01__OSD-137__RR3__39-day','C02__OSD-137__RR3__40-day']
 members=members[members.contrast_id.isin(wanted)].copy();frames=[]
 for osd in ['48','137']:
  a=pd.read_csv(RES/f'task3_manifest_api_validation/api_OSD-{osd}_sample_metadata.csv');a=a[a['id.assay name'].str.contains('transcription-profiling',case=False,na=False)].drop_duplicates('id.sample name');frames.append(a)
 api=pd.concat(frames,ignore_index=True,sort=False);q=members.merge(api,left_on='sample_id',right_on='id.sample name',how='left',validate='one_to_one')
 mapping={'study.characteristics.age at launch':'age','study.parameter value.duration':'duration','study.parameter value.euthanasia method':'euthanasia_method','study.parameter value.carcass preservation method':'carcass_preservation','study.parameter value.sample preservation method':'sample_preservation','study.parameter value.sample storage temperature':'storage_temperature','study.characteristics.material type':'tissue','assay.parameter value.qa score':'RIN','assay.parameter value.library selection':'RNA_selection','assay.parameter value.library kit':'library_kit','assay.parameter value.library layout':'read_layout','assay.parameter value.read length':'read_length','assay.parameter value.read depth':'read_depth','assay.parameter value.sequencing instrument':'sequencing_platform','assay.parameter value.rrna contamination':'rrna_contamination','assay.parameter value.preservation':'assay_preservation'}
 q=q.rename(columns=mapping);q['collection_group']=q.contrast_id.map({wanted[0]:'upon-euthanasia',wanted[1]:'carcass',wanted[2]:'RR3 39-day cohort',wanted[3]:'RR3 40-day cohort'})
 keep=['contrast_id','collection_group','sample_id','condition']+list(mapping.values())
 for c in keep:
  if c not in q:q[c]=np.nan
 q['time_euthanasia_to_freezing']='unavailable';q['RNA_extraction_method']='unavailable';q['processing_batch']='unavailable';q['dissection_time_order']='unavailable'
 return q[keep+['time_euthanasia_to_freezing','RNA_extraction_method','processing_batch','dissection_time_order']]

def quality_analysis(meta,b):
 manifest=pd.read_csv(RES/'sample_manifest.csv');z=np.load(WORK/'bridgerna_embeddings.npy').astype(float);x=np.load(WORK/'bridgerna_log1p_tpm_inputs.npy',mmap_mode='r');idx=dict(zip(manifest.sample_id,range(len(manifest))))
 genes=pd.read_csv(REPO/'data/ensembl/canonical_genes.csv');gcol='gene_symbol' if 'gene_symbol' in genes else genes.columns[-1]; symbols=genes[gcol].astype(str).tolist()
 rna=set(pd.read_csv(ROOT/'results/task4_confounding_profiler/rna_processing_gene_analysis/rna_processing_gene_universe.csv').iloc[:,0].astype(str));ri=[i for i,g in enumerate(symbols) if g in rna]
 rows=[]
 for r in meta.itertuples():
  i=idx[r.sample_id];rin=float(re.search(r'[0-9.]+',str(r.RIN)).group()) if re.search(r'[0-9.]+',str(r.RIN)) else np.nan
  rows.append({'sample_id':r.sample_id,'collection_group':r.collection_group,'condition':r.condition,'RIN':rin,'PC1':z[i]@b[0],'PC2':z[i]@b[1],'RNA_processing_expression_proxy':float(np.mean(np.asarray(x[i,ri])))})
 d=pd.DataFrame(rows);corr=[]
 for y in ['PC1','PC2','RNA_processing_expression_proxy']:
  q=d.dropna(subset=['RIN',y]);corr.append({'outcome':y,'n':len(q),'spearman_rho':spearmanr(q.RIN,q[y]).statistic,'pearson_r':pearsonr(q.RIN,q[y]).statistic})
 return d,pd.DataFrame(corr)

def gene_overlap():
 rr1=pd.read_parquet(ROOT/'results/task4_confounding_profiler/rr1_preservation_context/component_contextual_gene_rankings.parquet');rr1=rr1[(rr1.component=='parallel')&rr1.contrast_id.isin(['C13__OSD-48__RR1-NASA__37-day','C14__OSD-48__RR1-NASA__37-day'])]
 rr3=pd.read_parquet(ROOT/'results/task4_confounding_profiler/rr3_functional_overlap/component_gene_attribution_rankings.parquet');rr3=rr3[(rr3.component=='parallel')&(rr3.measurement=='OSD-137 original')]
 ctl=pd.read_parquet(ROOT/'results/task4_confounding_profiler/controlled_gene_context/controlled_gene_sensitivity.parquet')
 rows=[];sets={}
 for label,g in [('RR1 carcass',rr1[rr1.contrast_id.str.startswith('C14')]),('RR1 upon-euthanasia',rr1[rr1.contrast_id.str.startswith('C13')]),('RR3-39',rr3[rr3.timepoint=='RR3-39']),('RR3-40',rr3[rr3.timepoint=='RR3-40']),('Controlled T-cell',ctl)]:
  rank='rank' if 'rank' in g else 'absolute_rank' if 'absolute_rank' in g else 'sensitivity_rank';sets[label]=set(g.nsmallest(500,rank).gene_symbol)
 for a,c in [('RR1 carcass','RR3-39'),('RR1 carcass','RR3-40'),('RR1 upon-euthanasia','RR3-39'),('RR1 upon-euthanasia','RR3-40')]:
  overlap=sets[a]&sets[c];t=overlap&sets['Controlled T-cell'];M=15165;N=500
  rows.append({'RR1_group':a,'RR3_group':c,'top_n':N,'shared_PC1_2_genes':len(overlap),'expected_overlap':N*N/M,'fold_enrichment':len(overlap)/(N*N/M),'hypergeom_p':hypergeom.sf(len(overlap)-1,M,N,N),'shared_also_controlled_Tcell':len(t),'shared_genes':';'.join(sorted(overlap)),'shared_controlled_genes':';'.join(sorted(t))})
 return pd.DataFrame(rows),sets

def ora(overlap):
 universe=set(pd.read_csv(REPO/'data/ensembl/canonical_genes.csv').iloc[:,-1].astype(str));M=len(universe);out=[]
 for r in overlap.itertuples():
  selected=set(str(r.shared_genes).split(';'))-{''};N=len(selected)
  for source,file in GMT.items():
   for term,members in gp.parser.read_gmt(path=str(GMTROOT/file)).items():
    members=set(members)&universe;n=len(members);k=len(selected&members)
    if N and 10<=n<=500 and k:out.append({'RR1_group':r.RR1_group,'RR3_group':r.RR3_group,'source':source,'pathway':term,'overlap':k,'set_size':N,'p_value':hypergeom.sf(k-1,M,n,N),'genes':';'.join(sorted(selected&members))})
 d=pd.DataFrame(out)
 if len(d):
  d['fdr']=d.groupby(['RR1_group','RR3_group']).p_value.transform(lambda x:pd.Series(np.minimum.accumulate((np.sort(x)*len(x)/np.arange(1,len(x)+1))[::-1])[::-1],index=x.sort_values().index).reindex(x.index).values)
 return d

def figures(sim,quality,overlap,enr):
 plt.style.use('seaborn-v0_8-whitegrid');pivot=sim.pivot_table(index=['RR1_group','RR3_group'],columns='representation',values='cosine');fig,ax=plt.subplots(figsize=(9,5));pivot.plot.bar(ax=ax,color=['#4C78A8','#F58518']);ax.axhline(0,color='k',lw=.8);ax.set(ylabel='Response cosine',title='RR1↔RR3 similarity before and after PC1–2 removal');ax.tick_params(axis='x',rotation=25);fig.tight_layout()
 for e in ['png','pdf']:fig.savefig(FIG/f'rr1_rr3_2x2_similarity.{e}',dpi=400)
 plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,5));
 for name,g in quality.groupby('collection_group'):ax.scatter(g.RIN,g.PC1,label=name,s=65)
 ax.set(xlabel='RIN',ylabel='PolyA/Ribo-associated PC1 coordinate',title='RNA quality and PC1 position');ax.legend(fontsize=8);fig.tight_layout()
 for e in ['png','pdf']:fig.savefig(FIG/f'rin_pc1.{e}',dpi=400)
 plt.close(fig)
 fig,ax=plt.subplots(figsize=(10,6));q=enr[enr.fdr<.05].sort_values('fdr').head(15) if len(enr) else enr
 if len(q): ax.barh(range(len(q)), -np.log10(q.fdr.clip(lower=1e-300)),color='#4C78A8');ax.set_yticks(range(len(q)),q.pathway);ax.invert_yaxis()
 ax.set(xlabel='−log10(FDR)',title='Shared PC1–2 high-contribution gene enrichment');fig.tight_layout()
 for e in ['png','pdf']:fig.savefig(FIG/f'shared_pc12_pathways.{e}',dpi=400,bbox_inches='tight')
 plt.close(fig)

def main():
 OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(exist_ok=True);b=controlled_basis();v=responses();sim=similarity_table(v,b);sim.to_csv(OUT/'rr1_rr3_similarity_before_after.csv',index=False)
 meta=metadata();meta.to_csv(OUT/'sample_processing_metadata.csv',index=False);quality,corr=quality_analysis(meta,b);quality.to_csv(OUT/'rna_quality_individuals.csv',index=False);corr.to_csv(OUT/'rna_quality_correlations.csv',index=False)
 overlap,sets=gene_overlap();overlap.to_csv(OUT/'shared_pc12_gene_overlap.csv',index=False);enr=ora(overlap);enr.to_csv(OUT/'shared_pc12_pathway_enrichment.csv',index=False)
 figures(sim,quality,overlap,enr)
 conclusion={'carcass_sufficient':False,'reason':'RR1 carcass matches RR3-39 but opposes RR3-40 even though both RR3 cohorts share the same reported immediate liquid-nitrogen preservation and RNA protocol. RR1 upon-euthanasia instead matches RR3-40.','interpretation':'Mixed/non-identifiable: PC1-2 contains controlled protocol-associated structure and coherent biological/sample-state organization. Removing it improves RR1 technical replication because the discrepancy lies in this subspace, but destroys RR1-RR3 similarity because the same subspace also carries the cross-experiment response correspondence.','limitations':['RR3 postmortem interval and exact dissection timing/order are unavailable.','RIN correlations are descriptive at small n and confounded by cohort.','Full RR1 contextual outputs contain component magnitude, not signed vectors; shared genes are based on high component contribution, not signed concordance.']}
 (OUT/'summary.json').write_text(json.dumps(conclusion,indent=2)+'\n');(OUT/'summary.md').write_text('# RR1/RR3 sample-state diagnostic\n\n'+conclusion['interpretation']+'\n\nCarcass status alone is insufficient: '+conclusion['reason']+'\n')
 print(sim.to_string(index=False));print(corr.to_string(index=False));print(overlap.drop(columns=['shared_genes','shared_controlled_genes']).to_string(index=False));print('[complete]',OUT)
if __name__=='__main__':main()
