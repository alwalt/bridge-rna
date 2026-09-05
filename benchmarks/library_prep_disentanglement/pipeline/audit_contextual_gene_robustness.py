#!/usr/bin/env python3
"""Audit contextual-gene instability metrics and ranked enrichment (no inference)."""
from __future__ import annotations
import json,time
from datetime import datetime,timezone
from pathlib import Path
import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[1]
BASE=ROOT/'results/task4_confounding_profiler';OUT=BASE/'contextual_robustness';FIG=OUT/'figures';OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(exist_ok=True)
T3=REPO/'benchmarks/osdr_batch_effect_representation';R3=T3/'results';W3=T3/'work'
PAIRS={'RR1':('RR1_OSD48_original_matched','RR1_OSD168_no-ERCC'),'RR3-39':('C01_OSD137_original_matched','C01_OSD168_all_ERCC'),'RR3-40':('C02_OSD137_original_matched','C02_OSD168_all_ERCC')}
GMT={'GO:BP':'GO_Biological_Process_2026.gmt','KEGG':'KEGG_2026.gmt','REAC':'Reactome_Pathways_2024.gmt'};PERM=1000;SEED=42641

def cond(s):return 'FLT' if '_FLT_' in s else 'GC' if '_GC_' in s else 'other'
def bar(df,col,title,path,ylim=(0,1)):
 order=list(PAIRS);s=df.set_index('comparison').loc[order,col];colors=['#CC3311','#0077BB','#009988'];fig,ax=plt.subplots(figsize=(7,4.4),layout='constrained');b=ax.bar(order,s,color=colors);ax.set(ylim=ylim,title=title)
 for q,v in zip(b,s):ax.text(q.get_x()+q.get_width()/2,v+(.025 if v>=0 else -.055),f'{v:.3f}',ha='center',fontweight='bold')
 ax.axhline(0,color='black',lw=.8);fig.savefig(FIG/f'{path}.png',dpi=300);fig.savefig(FIG/f'{path}.pdf');plt.close(fig)

def main():
 metrics=pd.read_parquet(BASE/'gene_context_reproducibility.parquet');assert len(metrics)==15165*3
 metrics['combined_response_magnitude']=np.sqrt(metrics.original_response_norm*metrics.remeasurement_response_norm)
 metrics['directional_instability']=-metrics.context_reproducibility
 assert np.isfinite(metrics.select_dtypes('number')).all().all()
 # Exact implementation/sample audit.
 design=pd.read_csv(R3/'task3_osd168_technical_replication/technical_response_design.csv').set_index('representation');manifest=pd.read_csv(R3/'sample_manifest.csv');index=dict(zip(manifest.sample_id,range(len(manifest))))
 audit=[];all_ids=[]
 for pair,names in PAIRS.items():
  for measurement,name in zip(['A_original','B_remeasurement'],names):
   ids=str(design.loc[name,'samples']).split(' | ');groups=[cond(s) for s in ids];all_ids+=ids
   audit.append({'comparison':pair,'measurement':measurement,'response_name':name,'n_samples':len(ids),'n_FLT':groups.count('FLT'),'n_GC':groups.count('GC'),'all_in_input_manifest':all(s in index for s in ids),'unique_within_response':len(ids)==len(set(ids)),'FLT_minus_GC':True,'sample_ids':' | '.join(ids)})
 pd.DataFrame(audit).to_csv(OUT/'sample_and_metric_audit.csv',index=False)
 # Expression abundance from exact comparison samples, no contextual inference.
 x=np.load(W3/'bridgerna_log1p_tpm_inputs.npy',mmap_mode='r');assert x.shape[1]==15165
 genes=metrics.query("comparison=='RR1'").gene_symbol.to_numpy();assert len(set(genes))==15165
 abundance=[]
 for pair,names in PAIRS.items():
  ids=[]
  for name in names:ids+=str(design.loc[name,'samples']).split(' | ')
  mean=np.asarray(x[[index[s] for s in ids]]).mean(0)
  abundance.append(pd.DataFrame({'comparison':pair,'gene_symbol':genes,'mean_log1p_tpm':mean}))
 abundance=pd.concat(abundance,ignore_index=True);metrics=metrics.merge(abundance,on=['comparison','gene_symbol'],validate='one_to_one')
 metrics.to_parquet(OUT/'gene_level_metric_audit.parquet',index=False)
 # Norm/cosine diagnostics and bottom-magnitude exclusions.
 rows=[];filters=[];norm_groups=[]
 for pair,g in metrics.groupby('comparison',sort=False):
  rho=spearmanr(g.combined_response_magnitude,g.context_reproducibility).statistic
  rho_metrics=spearmanr(g.directional_instability,g.normalized_context_discrepancy).statistic
  rows.append({'comparison':pair,'genes':len(g),'zero_norm_A':int((g.original_response_norm<=1e-12).sum()),'zero_norm_B':int((g.remeasurement_response_norm<=1e-12).sum()),'magnitude_vs_cosine_spearman':rho,'directional_vs_normalized_discrepancy_spearman':rho_metrics,'median_cosine':g.context_reproducibility.median(),'fraction_reversed':(g.context_reproducibility<0).mean(),'median_normalized_discrepancy':g.normalized_context_discrepancy.median()})
  for label,mask in [('reversed_C_lt_0',g.context_reproducibility<0),('poor_C_lt_0.5',g.context_reproducibility<.5),('reproducible_C_ge_0.5',g.context_reproducibility>=.5)]:
   q=g.loc[mask,'combined_response_magnitude'];norm_groups.append({'comparison':pair,'group':label,'genes':len(q),'mean':q.mean(),'median':q.median(),'q25':q.quantile(.25),'q75':q.quantile(.75)})
  for excluded in [0,.05,.10,.20]:
   cutoff=g.combined_response_magnitude.quantile(excluded);q=g[g.combined_response_magnitude>=cutoff]
   filters.append({'comparison':pair,'lowest_magnitude_fraction_excluded':excluded,'genes_retained':len(q),'magnitude_cutoff':cutoff,'median_context_reproducibility':q.context_reproducibility.median(),'fraction_reversed':(q.context_reproducibility<0).mean(),'median_normalized_discrepancy':q.normalized_context_discrepancy.median()})
 metric_summary=pd.DataFrame(rows);metric_summary.to_csv(OUT/'metric_robustness_summary.csv',index=False);pd.DataFrame(norm_groups).to_csv(OUT/'response_norm_by_reproducibility_group.csv',index=False);filters=pd.DataFrame(filters);filters.to_csv(OUT/'magnitude_filter_sensitivity.csv',index=False)
 # IG membership for later leading-edge characterization.
 ranks=pd.read_parquet(ROOT/'results/task4_gene_attribution_diagnostic/per_response_gene_rankings.parquet')
 igsets={pair:set(ranks[(ranks.response.isin(names))&(ranks.rank_absolute<=100)].gene_symbol_human) for pair,names in PAIRS.items()}
 # Two rankings: magnitude-aware normalized discrepancy, and negative cosine after dropping bottom 10% magnitude.
 gmtroot=REPO/'benchmarks/cross_species_exercise_response/results/per_study_ranked_gsea';started=time.time();gsea_cache=OUT/'gsea_full_results.parquet'
 if gsea_cache.exists():
  print(f'[cache] loading {gsea_cache}',flush=True);result=pd.read_parquet(gsea_cache)
 else:
  results=[];job=0
  for pair,g in metrics.groupby('comparison',sort=False):
   cutoff=g.combined_response_magnitude.quantile(.10)
   rankings={'normalized_discrepancy':g[['gene_symbol','normalized_context_discrepancy']].sort_values('normalized_context_discrepancy',ascending=False),
             'magnitude_filtered_negative_cosine':g[g.combined_response_magnitude>=cutoff][['gene_symbol','directional_instability']].sort_values('directional_instability',ascending=False)}
   for rank_name,rnk in rankings.items():
    for source,file in GMT.items():
     job+=1;print(f'[GSEA {job}/18] {pair} {rank_name} {source}',flush=True)
     pre=gp.prerank(rnk=rnk,gene_sets=str(gmtroot/file),min_size=10,max_size=500,permutation_num=PERM,threads=8,seed=SEED,outdir=None,verbose=False)
     z=pre.res2d.rename(columns={'Term':'pathway','ES':'es','NES':'nes','NOM p-val':'nominal_p','FDR q-val':'fdr','Lead_genes':'leading_edge','Tag %':'tag_fraction','Gene %':'gene_fraction'})
     z['comparison']=pair;z['ranking']=rank_name;z['source']=source;results.append(z[['comparison','ranking','source','pathway','es','nes','nominal_p','fdr','leading_edge','tag_fraction','gene_fraction']])
  result=pd.concat(results,ignore_index=True);result.to_parquet(gsea_cache,index=False)
 # Pathway-set size after exact tested-universe intersection.
 universe_by_rank={('normalized_discrepancy'):set(genes)}
 sizes=[]
 for source,file in GMT.items():
  sets=gp.parser.read_gmt(path=str(gmtroot/file))
  for term,members in sets.items():sizes.append({'source':source,'pathway':term,'represented_genes':len(set(members)&set(genes))})
 sizes=pd.DataFrame(sizes);result=result.merge(sizes,on=['source','pathway'],how='left')
 def leading_rows(z):
  out=[]
  for r in z.itertuples():
   members=[] if pd.isna(r.leading_edge) else str(r.leading_edge).replace(';',',').split(',')
   for gene in filter(None,members):out.append({'comparison':r.comparison,'ranking':r.ranking,'source':r.source,'pathway':r.pathway,'nes':r.nes,'fdr':r.fdr,'represented_genes':r.represented_genes,'gene_symbol':gene})
  return pd.DataFrame(out)
 # Robust = same pathway significant and same NES direction under both definitions.
 sig=result[result.fdr<.05];wide=sig.pivot_table(index=['comparison','source','pathway'],columns='ranking',values=['nes','fdr'],aggfunc='first').dropna().reset_index()
 wide.columns=['_'.join(c).strip('_') for c in wide.columns];wide=wide[np.sign(wide.nes_normalized_discrepancy)==np.sign(wide.nes_magnitude_filtered_negative_cosine)]
 unstable=wide[(wide.nes_normalized_discrepancy>0)&(wide.nes_magnitude_filtered_negative_cosine>0)].copy();stable=wide[(wide.nes_normalized_discrepancy<0)&(wide.nes_magnitude_filtered_negative_cosine<0)].copy()
 unstable.to_csv(OUT/'robust_unstable_pathways.csv',index=False);stable.to_csv(OUT/'robust_stable_pathways.csv',index=False)
 # Major RR1 families and suspicious RR3-39 pathways.
 rr1pat='RNA|MRNA|SPLIC|CHROMATIN|DNA REPAIR|DNA METABOL';rr1=result[(result.comparison=='RR1')&result.pathway.str.contains(rr1pat,case=False,regex=True,na=False)&(result.fdr<.05)]
 suspicious='VISUAL|LIGHT STIMULUS|PHOTOTRANSDUCTION|CORNIFIED|KERATIN|INTERMEDIATE FILAMENT';rr3=result[(result.comparison=='RR3-39')&result.pathway.str.contains(suspicious,case=False,regex=True,na=False)&(result.fdr<.05)]
 le=pd.concat([leading_rows(rr1),leading_rows(rr3)],ignore_index=True);le=le.merge(metrics[['comparison','gene_symbol','context_reproducibility','original_response_norm','remeasurement_response_norm','combined_response_magnitude','normalized_context_discrepancy','mean_log1p_tpm']],on=['comparison','gene_symbol'],how='left');le['in_IG_Top100_union']=[r.gene_symbol in igsets[r.comparison] for r in le.itertuples()]
 le.to_csv(OUT/'leading_edge_gene_audit.csv',index=False)
 rr3audit=le[le.comparison.eq('RR3-39')].groupby(['source','pathway','ranking']).agg(leading_edge_genes=('gene_symbol','nunique'),median_expression=('mean_log1p_tpm','median'),median_combined_magnitude=('combined_response_magnitude','median'),median_context_cosine=('context_reproducibility','median'),median_normalized_discrepancy=('normalized_context_discrepancy','median')).reset_index();rr3audit.to_csv(OUT/'rr3_39_suspicious_pathway_audit.csv',index=False)
 # Compact pathway families, avoiding redundant-term overstatement.
 def family(term):
  t=term.upper()
  if 'SPLIC' in t or 'RNA' in t or 'MRNA' in t:return 'RNA processing / splicing'
  if 'CHROMATIN' in t:return 'Chromatin organization / remodeling'
  if 'DNA REPAIR' in t or 'DNA METABOL' in t:return 'DNA repair / DNA metabolism'
  if 'FATTY ACID' in t or 'PEROX' in t:return 'Fatty-acid / peroxisomal metabolism'
  return 'Other'
 rr1copy=rr1.copy();rr1copy['family']=rr1copy.pathway.map(family);rr1copy.to_csv(OUT/'rr1_major_pathway_families.csv',index=False)
 # Audit metadata and p-value floor.
 audit={'ranking_statistics':{'normalized_discrepancy':'norm(A-B)/(norm(A)+norm(B)+epsilon), descending','magnitude_filtered_negative_cosine':'-cosine(A,B), descending after excluding bottom 10% by sqrt(normA*normB)'},'gene_universe_full':15165,'gene_universe_filtered':int((metrics.query("comparison=='RR1'").combined_response_magnitude>=metrics.query("comparison=='RR1'").combined_response_magnitude.quantile(.1)).sum()),'databases':list(GMT),'min_gene_set_size':10,'max_gene_set_size':500,'permutations':PERM,'seed':SEED,'p_value':'gseapy phenotype-label permutation empirical nominal p','FDR':'gseapy permutation-based FDR q-value','previous_floor':1/(250+1),'current_nominal_resolution_approx':1/(PERM+1),'background_note':'Ranked universes are restricted to valid BridgeRNA canonical genes; gene sets are intersected with that tested universe. Results describe programs represented within the BridgeRNA vocabulary, not the complete mouse transcriptome.'}
 (OUT/'gsea_audit.json').write_text(json.dumps(audit,indent=2));pd.DataFrame([audit|{'ranking_statistics':json.dumps(audit['ranking_statistics']),'databases':';'.join(audit['databases'])}]).to_csv(OUT/'gsea_audit.csv',index=False)
 # Required plots.
 bar(metric_summary,'median_cosine','Median Context Reproducibility','median_context_reproducibility',(-.3,1));bar(metric_summary,'fraction_reversed','Fraction of Contextually Reversed Genes','context_reversal_fraction',(0,1));bar(metric_summary,'median_normalized_discrepancy','Median Normalized Contextual Discrepancy','median_normalized_discrepancy',(0,1))
 mf=filters[filters.lowest_magnitude_fraction_excluded.eq(.1)];bar(mf,'median_context_reproducibility','Context Reproducibility after excluding lowest 10% magnitude','magnitude_filtered_context_reproducibility',(-.3,1))
 for frame,title,path,positive in [(unstable,'Robust RR1 contextually unstable pathways','robust_rr1_unstable_pathways',True),(stable,'Robust RR1 contextually stable pathways','robust_rr1_stable_pathways',False)]:
  q=frame[frame.comparison.eq('RR1')].copy();q['score']=q[['nes_normalized_discrepancy','nes_magnitude_filtered_negative_cosine']].mean(axis=1);q=q.reindex(q.score.abs().sort_values().tail(12).index)
  if len(q):
   fig,ax=plt.subplots(figsize=(9,6),layout='constrained');ax.barh(q.pathway,q.score,color='#CC3311' if positive else '#228833');ax.axvline(0,color='black',lw=.8);ax.set(xlabel='Mean NES across two rankings',title=title);fig.savefig(FIG/f'{path}.png',dpi=300);fig.savefig(FIG/f'{path}.pdf');plt.close(fig)
 prov={'created_utc':datetime.now(timezone.utc).isoformat(),'inference_rerun':False,'context_metrics_input':str(BASE/'gene_context_reproducibility.parquet'),'samples_verified':len(set(all_ids)),'genes':15165,'permutations':PERM,'elapsed_minutes':(time.time()-started)/60}
 (OUT/'provenance.json').write_text(json.dumps(prov,indent=2));print(metric_summary.to_string(index=False));print(filters.to_string(index=False));print(f'[complete] {OUT}',flush=True)
if __name__=='__main__':main()
