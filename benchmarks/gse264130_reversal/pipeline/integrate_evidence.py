#!/usr/bin/env python3
"""Integrate frozen ChEMBL/LINCS/literature layers after GSE264130 scoring."""
from pathlib import Path
import re,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parents[1];OUT=HERE/'results'
def norm(x):return re.sub('[^a-z0-9]+','',str(x).lower())
PUBLISHED={'Multiple sclerosis':['muromonab','ibrutinib','daclizumab','zanubrutinib','alemtuzumab'],"Crohn's disease":['zinc','zinc acetate','dilmapimod','glucosamine','vx-702'],'Systemic lupus erythematosus':['gemcitabine','enzastaurin','sunitinib','fostamatinib','cladribine']}
RAD=['pentoxifylline','sitagliptin','linagliptin','saxagliptin','alogliptin','omipalisib','ku-0060648','nu-7441','rg-547','thz1','afatinib','lapatinib','neratinib','jq1','molibresib']
def main():
 e=pd.read_csv(OUT/'expression_drug_condition_consensus.csv.gz');l=pd.read_csv(OUT/'latent_drug_condition_consensus.csv.gz')
 ec=pd.read_parquet(OUT/'expression_reversal_by_contrast_cell.parquet');lc=pd.read_parquet(OUT/'latent_reversal_by_contrast_cell.parquet')
 ex=pd.read_csv(OUT/'expression_cross_study_consensus.csv');lx=pd.read_csv(OUT/'latent_cross_study_consensus.csv')
 # Expression-latent agreement uses Bridge expression only, same drug/cell/contrast.
 a=ec[ec.method.eq('Bridge')].merge(lc,on=['condition','contrast_id','dataset','role','drug','cell_line'],suffixes=('_expr','_latent'))
 agree=[]
 for (cond,cell),g in a.groupby(['condition','cell_line']):
  agree.append({'condition':cond,'cell_line':cell,'pairs':len(g),'spearman_r':spearmanr(g.reversal_score,g.latent_reversal).statistic,
                'sign_agreement':(np.sign(g.reversal_score)==np.sign(g.latent_reversal)).mean()})
 pd.DataFrame(agree).to_csv(OUT/'expression_latent_agreement.csv',index=False)

 # Frozen LINCS condition consensuses.
 lp=ROOT/'benchmarks/direction_aware_reversal/results';ld=pd.read_parquet(lp/'drug_reversal_by_contrast.parquet');ls=pd.read_csv(lp/'space_condition_consensus.csv.gz')
 disease=ld[ld.condition.isin(PUBLISHED)&ld.analysis.eq('top500')][['condition','method','drug_name','median_reversal']]
 space=ls[['condition','method','drug_name','consensus_reversal']].rename(columns={'consensus_reversal':'median_reversal'})
 lin=pd.concat([disease,space],ignore_index=True);lin['key']=lin.drug_name.map(norm);lin=lin.groupby(['condition','method','key'],as_index=False).median_reversal.median()
 gd=ec.groupby(['condition','method','drug','cell_line'],as_index=False).reversal_score.median();gd['key']=gd.drug.map(norm)
 ca=gd.merge(lin,on=['condition','method','key'],suffixes=('_rnaseq','_lincs'));ca['sign_agreement']=np.sign(ca.reversal_score)==np.sign(ca.median_reversal)
 ca['agreement_class']=np.select([(ca.reversal_score>0)&(ca.median_reversal>0),(ca.reversal_score<0)&(ca.median_reversal<0)],['RNA-seq + LINCS reversal','reinforcement in both'],'opposing conclusions')
 ca.to_csv(OUT/'rnaseq_lincs_drug_agreement.csv.gz',index=False,compression='gzip')
 summary=[]
 for (cond,method,cell),g in ca.groupby(['condition','method','cell_line']):
  top_r=set(g.nlargest(min(20,len(g)),'reversal_score').key);top_l=set(g.nlargest(min(20,len(g)),'median_reversal').key)
  summary.append({'condition':cond,'method':method,'cell_line':cell,'shared_drugs':len(g),'spearman_r':spearmanr(g.reversal_score,g.median_reversal).statistic,
                  'direction_agreement':g.sign_agreement.mean(),'top20_overlap':len(top_r&top_l),'top20_jaccard':len(top_r&top_l)/len(top_r|top_l)})
 pd.DataFrame(summary).to_csv(OUT/'rnaseq_lincs_agreement_summary.csv',index=False)

 # Frozen expanded-ChEMBL target-overlap support.
 sp=pd.read_csv(ROOT/'benchmarks/drug_discovery/results/expanded_chembl_sensitivity/expanded_drug_enrichment.csv.gz')
 dp=pd.read_csv(ROOT/'benchmarks/deweerd_replication/results/evaluation2_expanded/drug_enrichment_gt10.csv.gz')
 def disease_cond(x):
  x=str(x).lower();return 'Multiple sclerosis' if 'multiple sclerosis' in x else "Crohn's disease" if 'crohn' in x else 'Systemic lupus erythematosus'
 dp['condition']=dp.disease.map(disease_cond);ch=pd.concat([sp,dp],ignore_index=True,sort=False);ch['key']=ch.drug_name.map(norm)
 target=(ch.groupby(['condition','method','key'],as_index=False).agg(chembl_support=('overlap_count',lambda x:bool((x.fillna(0)>0).any())),
          chembl_genes=('overlap_genes',lambda x:';'.join(sorted(set(';'.join(x.dropna().astype(str)).split(';'))-{'','nan'}))),chembl_best_fdr=('fdr','min')))
 # Candidate union: Bridge expression or latent both-cell screens.
 eb=e[(e.method=='Bridge')&e.candidate_both_cells].copy();lb=l[l.candidate_both_cells].copy();keys=set(zip(eb.condition,eb.drug))|set(zip(lb.condition,lb.drug));rows=[]
 lin_lookup=lin[lin.method.eq('Bridge')].set_index(['condition','key']).median_reversal.to_dict();tar=target[target.method.eq('Bridge')].set_index(['condition','key'])
 lit_rad=pd.read_csv(ROOT/'benchmarks/drug_discovery/results/radiation_pharmacology/literature_evidence_audit.csv');lit_keys=set(lit_rad[~lit_rad.evidence_class.str.startswith('6 ',na=False)].drug_name.map(norm))
 for cond,drug in sorted(keys):
  er=eb[(eb.condition==cond)&(eb.drug==drug)];lr=lb[(lb.condition==cond)&(lb.drug==drug)];de=e[(e.condition==cond)&(e.method=='DE')&(e.drug==drug)]
  key=norm(drug);t=tar.loc[(cond,key)] if (cond,key) in tar.index else None
  literature=(key in {norm(x) for x in PUBLISHED.get(cond,[])}) or (cond=='Radiation injury' and key in lit_keys)
  cross_expr=bool(ex[(ex.condition==cond)&(ex.method=='Bridge')&(ex.drug==drug)].cross_study_consistent.any())
  cross_lat=bool(lx[(lx.condition==cond)&(lx.drug==drug)].cross_study_consistent.any())
  rows.append({'condition':cond,'drug':drug,'bridge_expression_candidate':len(er)>0,'bridge_expression_DIPG6':er.reversal_score_DIPG6.iloc[0] if len(er) else np.nan,
   'bridge_expression_SF8628':er.reversal_score_SF8628.iloc[0] if len(er) else np.nan,'de_candidate':bool(len(de) and de.candidate_both_cells.iloc[0]),
   'de_expression_DIPG6':de.reversal_score_DIPG6.iloc[0] if len(de) else np.nan,'de_expression_SF8628':de.reversal_score_SF8628.iloc[0] if len(de) else np.nan,
   'latent_candidate':len(lr)>0,'latent_DIPG6':lr.latent_reversal_DIPG6.iloc[0] if len(lr) else np.nan,'latent_SF8628':lr.latent_reversal_SF8628.iloc[0] if len(lr) else np.nan,
   'expression_latent_direction_agreement':bool(len(er) and len(lr)),'cross_study_expression':cross_expr,'cross_study_latent':cross_lat,
   'gse264130_cell_consistency':'both' if len(er) or len(lr) else 'none','chembl_support':bool(t.chembl_support) if t is not None else False,
   'chembl_genes':t.chembl_genes if t is not None else '', 'chembl_best_fdr':t.chembl_best_fdr if t is not None else np.nan,
   'lincs_bridge_reversal':lin_lookup.get((cond,key),np.nan),'lincs_agreement':('same reversal direction' if lin_lookup.get((cond,key),-1)>0 else 'opposing/not reversing') if (cond,key) in lin_lookup else 'not assayed/matched',
   'literature_support':literature})
 integrated=pd.DataFrame(rows);integrated.to_csv(OUT/'integrated_candidate_evidence.csv',index=False)

 # Prespecified previously highlighted audit, with absence explicit.
 all_drugs={norm(x):x for x in e.drug.unique()};audit=[]
 for cond,terms in {**PUBLISHED,'Radiation injury':RAD}.items():
  for requested in terms:
   actual=all_drugs.get(norm(requested))
   if actual is None:audit.append({'condition':cond,'requested_drug':requested,'gse264130_status':'not assayed / insufficient GSE264130 data'})
   else:
    for method in ['Bridge','DE']:
     q=e[(e.condition==cond)&(e.method==method)&(e.drug==actual)];z=l[(l.condition==cond)&(l.drug==actual)]
     audit.append({'condition':cond,'requested_drug':requested,'gse264130_status':'assayed','drug':actual,'method':method,
       'expression_classification':q.cell_classification.iloc[0],'expression_DIPG6':q.reversal_score_DIPG6.iloc[0],'expression_SF8628':q.reversal_score_SF8628.iloc[0],
       'latent_classification':z.cell_classification.iloc[0] if len(z) else np.nan,'latent_DIPG6':z.latent_reversal_DIPG6.iloc[0] if len(z) else np.nan,'latent_SF8628':z.latent_reversal_SF8628.iloc[0] if len(z) else np.nan})
 pd.DataFrame(audit).to_csv(OUT/'highlighted_drug_audit.csv',index=False)
 (OUT/'integration_provenance.json').write_text(json.dumps({'prior_analyses_rerun':False,'LINCS_role':'secondary cross-assay only','composite_score_created':False,'drug_matching':'normalized exact name','layers_kept_separate':True},indent=2)+'\n')
if __name__=='__main__':main()
