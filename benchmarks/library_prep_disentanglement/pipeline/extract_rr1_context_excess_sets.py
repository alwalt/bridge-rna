#!/usr/bin/env python3
"""Extract and compare saved RR1 DE/non-DE context-excess gene sets."""
from __future__ import annotations
import json
from pathlib import Path
import gseapy as gp
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

HERE=Path(__file__).resolve().parents[1];REPO=HERE.parents[1]
OUT=HERE/'results/task4_confounding_profiler/expression_adjusted_context'
GMT_ROOT=REPO/'benchmarks/cross_species_exercise_response/results/per_study_ranked_gsea'
GMT={'GO:BP':'GO_Biological_Process_2026.gmt','KEGG':'KEGG_2026.gmt','REAC':'Reactome_Pathways_2024.gmt'}
COLS=['gene_symbol','logFC','FDR','expression_statistic','contextual_statistic','predicted_contextual_sensitivity','residual_contextual_sensitivity','standardized_residual','residual_rank']

def family(term):
 t=term.upper()
 if 'SPLIC' in t or 'RNA PROCESS' in t or 'MRNA' in t:return 'RNA processing / splicing'
 if 'CHROMATIN' in t:return 'Chromatin organization / remodeling'
 if 'DNA REPAIR' in t or 'DNA METABOL' in t:return 'DNA repair / DNA metabolism'
 return 'Other'

def ora(name,selected,universe):
 rows=[];M=len(universe);selected=set(selected)&universe;N=len(selected)
 for source,file in GMT.items():
  source_rows=[]
  for term,members in gp.parser.read_gmt(path=str(GMT_ROOT/file)).items():
   members=set(members)&universe;n=len(members)
   if n<10 or n>500:continue
   overlap=selected&members;k=len(overlap);p=hypergeom.sf(k-1,M,n,N) if k else 1.0
   source_rows.append({'gene_set':name,'source':source,'pathway':term,'selected_genes':N,'background_genes':M,'pathway_genes_in_background':n,'overlap':k,'expected_overlap':N*n/M,'fold_enrichment':k/(N*n/M) if N*n else np.nan,'p_value':p,'overlap_genes':';'.join(sorted(overlap))})
  q=pd.DataFrame(source_rows);q['fdr']=multipletests(q.p_value,method='fdr_bh')[1];rows.append(q)
 return pd.concat(rows,ignore_index=True)

def describe(name,q):
 r={'gene_set':name,'genes':len(q)}
 for c in ['residual_contextual_sensitivity','standardized_residual','expression_statistic','contextual_statistic','logFC','FDR']:
  r[f'{c}_mean']=q[c].mean();r[f'{c}_median']=q[c].median();r[f'{c}_q25']=q[c].quantile(.25);r[f'{c}_q75']=q[c].quantile(.75)
 return r

def main():
 d=pd.read_parquet(OUT/'RR1_matched_gene_table.parquet').sort_values('standardized_residual',ascending=False).copy()
 valid=d[np.isfinite(d[COLS[1:]].to_numpy(float)).all(axis=1)].copy();universe=set(valid.gene_symbol)
 de=valid[(valid.FDR<.05)&(valid.residual_contextual_sensitivity>0)].sort_values('standardized_residual',ascending=False)
 de[COLS].to_csv(OUT/'RR1_de_associated_context_excess_genes.csv',index=False)
 nonde=valid[(valid.FDR>=.05)&(valid.residual_contextual_sensitivity>0)].sort_values('standardized_residual',ascending=False)
 cutoffs={'nonDE_top100':100,'nonDE_top250':250,'nonDE_top500':500,'nonDE_top5pct':int(np.ceil(.05*len(valid)))}
 sets={'DE_associated_positive_residual':de}
 for name,n in cutoffs.items():
  q=nonde.head(n).copy();q[COLS].to_csv(OUT/f'RR1_{name}_context_excess_genes.csv',index=False);sets[name]=q
 top=[]
 for name,q in sets.items():
  z=q[COLS].head(25).copy();z.insert(0,'gene_set',name);top.append(z)
 pd.concat(top,ignore_index=True).to_csv(OUT/'RR1_context_excess_sets_top25.csv',index=False)
 summaries=pd.DataFrame([describe(name,q) for name,q in sets.items()]);summaries.to_csv(OUT/'RR1_context_excess_set_distribution_summary.csv',index=False)
 enrich=pd.concat([ora(name,q.gene_symbol,universe) for name,q in sets.items()],ignore_index=True);enrich.to_csv(OUT/'RR1_context_excess_set_pathway_enrichment.csv',index=False)
 fam=[]
 for name in sets:
  z=enrich[enrich.gene_set.eq(name)].copy();z['family']=z.pathway.map(family)
  for f in ['RNA processing / splicing','Chromatin organization / remodeling','DNA repair / DNA metabolism']:
   q=z[z.family.eq(f)].sort_values(['fdr','p_value']);sig=q[q.fdr<.05]
   fam.append({'gene_set':name,'family':f,'tested_terms':len(q),'significant_terms':len(sig),'best_pathway':q.pathway.iloc[0] if len(q) else '',
               'best_overlap':q.overlap.iloc[0] if len(q) else 0,'best_fold_enrichment':q.fold_enrichment.iloc[0] if len(q) else np.nan,'best_fdr':q.fdr.iloc[0] if len(q) else np.nan,
               'supported':bool(len(sig))})
 family_df=pd.DataFrame(fam);family_df.to_csv(OUT/'RR1_context_excess_set_pathway_family_summary.csv',index=False)
 primary=sets['nonDE_top5pct'];comparison=pd.DataFrame([describe('DE_associated_positive_residual',de),describe('nonDE_top5pct',primary)])
 comparison['gene_overlap']=0;comparison['definition']='FDR < 0.05 and residual > 0' ;comparison.loc[comparison.gene_set.eq('nonDE_top5pct'),'definition']=f'FDR >= 0.05; top {len(primary)} positive residuals (5% of {len(valid)} valid genes)'
 comparison.to_csv(OUT/'RR1_DE_vs_nonDE_context_excess_comparison.csv',index=False)
 result={'valid_rr1_genes':len(valid),'de_associated_context_excess_genes':len(de),'nonde_positive_residual_pool':len(nonde),'nonde_thresholds':cutoffs,'primary_nonde_set':'nonDE_top5pct','primary_nonde_genes':len(primary),
         'gene_overlap':0,'nonde_primary_significant_pathways':int(((enrich.gene_set=='nonDE_top5pct')&(enrich.fdr<.05)).sum()),
         'family_support':family_df[family_df.gene_set.isin(['DE_associated_positive_residual','nonDE_top5pct'])].to_dict('records'),
         'answer':'Yes' if ((enrich.gene_set=='nonDE_top5pct')&(enrich.fdr<.05)).any() else 'No',
         'interpretation':'Non-DE genes with high positive expression-adjusted residuals show significant pathway organization.' if ((enrich.gene_set=='nonDE_top5pct')&(enrich.fdr<.05)).any() else 'The non-DE context-excess set does not show FDR-significant pathway organization.'}
 (OUT/'RR1_DE_vs_nonDE_context_excess_summary.json').write_text(json.dumps(result,indent=2)+'\n');pd.DataFrame([{k:(json.dumps(v) if isinstance(v,(dict,list)) else v) for k,v in result.items()}]).to_csv(OUT/'RR1_DE_vs_nonDE_context_excess_summary.csv',index=False)
 print(json.dumps(result,indent=2))
 print('\nTop 25 DE-associated context-excess\n',de[COLS].head(25).to_string(index=False))
 print('\nTop 25 non-DE context-excess\n',primary[COLS].head(25).to_string(index=False))
if __name__=='__main__':main()
