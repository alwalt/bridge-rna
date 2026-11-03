#!/usr/bin/env python3
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib.pyplot as plt,seaborn as sns
HERE=Path(__file__).resolve().parents[1];OUT=HERE/'results';FIG=HERE/'figures';FIG.mkdir(exist_ok=True)
CONDS=['Multiple sclerosis',"Crohn's disease",'Systemic lupus erythematosus','Radiation injury','Bone loss','Muscle atrophy'];SHORT=['MS','Crohn','SLE','Radiation','Bone','Muscle']
def main():
 e=pd.read_csv(OUT/'expression_drug_condition_consensus.csv.gz');l=pd.read_csv(OUT/'latent_drug_condition_consensus.csv.gz');integ=pd.read_csv(OUT/'integrated_candidate_evidence.csv')
 ex=pd.read_csv(OUT/'expression_cross_study_consensus.csv');lx=pd.read_csv(OUT/'latent_cross_study_consensus.csv');el=pd.read_csv(OUT/'expression_latent_agreement.csv');rl=pd.read_csv(OUT/'rnaseq_lincs_agreement_summary.csv')
 rows=[]
 for cond in CONDS:
  b=set(e[(e.condition==cond)&(e.method=='Bridge')&e.candidate_both_cells].drug);d=set(e[(e.condition==cond)&(e.method=='DE')&e.candidate_both_cells].drug);z=set(l[(l.condition==cond)&l.candidate_both_cells].drug)
  q=integ[integ.condition==cond]
  rows.append({'Condition':cond,'Bridge expression candidates':len(b),'DE expression candidates':len(d),'Shared Bridge/DE':len(b&d),'Bridge-only':len(b-d),'DE-only':len(d-b),
   'Latent candidates':len(z),'Expression + latent':len(b&z),'Strict cross-study expression':int(ex[(ex.condition==cond)&(ex.method=='Bridge')].cross_study_consistent.sum()),
   'Strict cross-study latent':int(lx[lx.condition==cond].cross_study_consistent.sum()),'ChEMBL-supported candidate union':int(q.chembl_support.sum()),
   'LINCS same-direction candidate union':int((q.lincs_agreement=='same reversal direction').sum()),'Literature-supported candidate union':int(q.literature_support.sum())})
 pd.DataFrame(rows).to_csv(OUT/'unified_primary_summary.csv',index=False)
 # Candidate list by method and latent.
 out=[]
 for cond in CONDS:
  for method in ['Bridge','DE']:
   q=e[(e.condition==cond)&(e.method==method)&e.candidate_both_cells]
   for _,r in q.iterrows():out.append({'condition':cond,'analysis':method+' expression','drug':r.drug,'DIPG6':r.reversal_score_DIPG6,'SF8628':r.reversal_score_SF8628,'DIPG6_p':r.empirical_p_DIPG6,'SF8628_p':r.empirical_p_SF8628})
  q=l[(l.condition==cond)&l.candidate_both_cells]
  for _,r in q.iterrows():out.append({'condition':cond,'analysis':'Bridge latent','drug':r.drug,'DIPG6':r.latent_reversal_DIPG6,'SF8628':r.latent_reversal_SF8628,'DIPG6_p':r.empirical_p_DIPG6,'SF8628_p':r.empirical_p_SF8628})
 pd.DataFrame(out).to_csv(OUT/'all_primary_candidates.csv',index=False)

 sns.set_theme(style='whitegrid',context='talk');fig,ax=plt.subplots(2,2,figsize=(16,11));u=pd.DataFrame(rows);x=np.arange(6);w=.25
 ax[0,0].bar(x-w,u['Bridge expression candidates'],w,label='Bridge expression');ax[0,0].bar(x,u['DE expression candidates'],w,label='DE expression');ax[0,0].bar(x+w,u['Latent candidates'],w,label='Bridge latent');ax[0,0].set_xticks(x,SHORT,rotation=25);ax[0,0].set_ylabel('Both-cell candidates');ax[0,0].legend(fontsize=10);ax[0,0].set_title('Observed GSE264130 screens')
 mix=[]
 for cond in CONDS:
  q=e[(e.condition==cond)&(e.method=='Bridge')];mix.append((q.cell_classification=='mixed/opposing contexts').mean())
 ax[0,1].bar(x,np.array(mix)*100,color='#d69e2e');ax[0,1].set_xticks(x,SHORT,rotation=25);ax[0,1].set_ylabel('Bridge drugs with opposing cell signs (%)');ax[0,1].set_title('DIPG6 versus SF8628 dependence')
 piv=el.pivot(index='condition',columns='cell_line',values='spearman_r').reindex(CONDS);ax[1,0].bar(x-w/2,piv.DIPG6,w,label='DIPG6');ax[1,0].bar(x+w/2,piv.SF8628,w,label='SF8628');ax[1,0].axhline(0,color='black',lw=1);ax[1,0].set_xticks(x,SHORT,rotation=25);ax[1,0].set_ylabel('Spearman r');ax[1,0].set_title('Bridge expression versus latent reversal');ax[1,0].legend(fontsize=10)
 q=rl[rl.method=='Bridge'];p=q.pivot(index='condition',columns='cell_line',values='spearman_r').reindex(CONDS);ax[1,1].bar(x-w/2,p.DIPG6,w,label='DIPG6');ax[1,1].bar(x+w/2,p.SF8628,w,label='SF8628');ax[1,1].axhline(0,color='black',lw=1);ax[1,1].set_xticks(x,SHORT,rotation=25);ax[1,1].set_ylabel('Spearman r');ax[1,1].set_title('GSE264130 versus frozen LINCS');ax[1,1].legend(fontsize=10)
 fig.suptitle('Primary observed RNA-seq direction-aware reversal benchmark',fontweight='bold');fig.tight_layout();fig.savefig(FIG/'gse264130_primary_reversal_summary.png',dpi=300,bbox_inches='tight');fig.savefig(FIG/'gse264130_primary_reversal_summary.pdf',bbox_inches='tight');plt.close(fig)
 print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
