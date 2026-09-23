#!/usr/bin/env python3
"""Build compact benchmark tables and a publication-quality summary figure."""
from pathlib import Path
import re, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns

HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parents[1];OUT=HERE/'results';FIG=HERE/'figures';FIG.mkdir(exist_ok=True)
CONDITIONS=['Multiple sclerosis',"Crohn's disease",'Systemic lupus erythematosus','Radiation injury','Bone loss','Muscle atrophy']

def norm(x):return re.sub('[^a-z0-9]+','',str(x).lower())
def main():
    primary=pd.read_parquet(OUT/'drug_reversal_by_contrast.parquet')
    space=pd.read_csv(OUT/'space_condition_consensus.csv.gz')
    parts=pd.read_parquet(OUT/'partition_and_stable_program_reversal.parquet')
    target=pd.read_parquet(OUT/'target_reversal_integration.parquet')
    nulls=pd.read_csv(OUT/'prioritized_candidate_nulls.csv');audit=pd.read_csv(OUT/'published_drug_direction_audit.csv')
    rows=[];method_rows=[]
    disease=set(CONDITIONS[:3])
    for cond in CONDITIONS:
        x=primary[primary.condition==cond] if cond in disease else space[space.condition==cond]
        b=set(x[(x.method=='Bridge')&x.candidate].pert_id);d=set(x[(x.method=='DE')&x.candidate].pert_id)
        cross=set() if cond in disease else set(x[(x.method=='Bridge')&x.cross_study_reproducible].pert_id)
        ti=target[(target.condition==cond)&target.candidate.fillna(False)&(target.method=='Bridge')]
        rows.append({'Condition':cond,'Bridge reversal candidates':len(b),'DE reversal candidates':len(d),
                     'Bridge-only candidates':len(b-d),'Cross-study reproducible':np.nan if cond in disease else len(cross),
                     'Target-supported':int(ti.target_supported.sum()),'Literature-supported':np.nan})
        # Identical compounds, paired method comparison.
        key='median_reversal' if cond in disease else 'consensus_reversal'
        a=x[x.method=='Bridge'][['pert_id',key]].merge(x[x.method=='DE'][['pert_id',key]],on='pert_id',suffixes=('_bridge','_de'))
        stat,p=wilcoxon(a[key+'_bridge'],a[key+'_de'])
        method_rows.append({'condition':cond,'eligible_shared_drugs':len(a),
                            'median_bridge_minus_de':float(np.median(a[key+'_bridge']-a[key+'_de'])),
                            'spearman_r':spearmanr(a[key+'_bridge'],a[key+'_de']).statistic,
                            'paired_wilcoxon_p':p,'bridge_only_top1':len(b-d),'de_only_top1':len(d-b),'shared_top1':len(b&d)})
    unified=pd.DataFrame(rows)
    # Frozen literature evidence: de Weerd highlighted drugs for disease; prior radiation audit for radiation.
    for cond in disease:
        rev=set(audit[(audit.condition==cond)&(audit.method=='Bridge')&audit.classification.eq('reverses')].lincs_name.dropna().map(norm))
        unified.loc[unified.Condition==cond,'Literature-supported']=len(rev)
    lit=pd.read_csv(ROOT/'benchmarks/drug_discovery/results/radiation_pharmacology/literature_evidence_audit.csv')
    supported=set(lit[~lit.evidence_class.str.startswith('6 ',na=False)].drug_name.map(norm))
    rad=space[(space.condition=='Radiation injury')&(space.method=='Bridge')&space.candidate]
    unified.loc[unified.Condition=='Radiation injury','Literature-supported']=rad.drug_name.map(norm).isin(supported).sum()
    unified.to_csv(OUT/'unified_six_condition_table.csv',index=False)
    pd.DataFrame(method_rows).to_csv(OUT/'bridge_vs_de_paired_comparison.csv',index=False)

    # Partition-level candidates quantify complementary signatures directly.
    ps=[]
    for cond in CONDITIONS:
        q=parts[parts.condition==cond]
        for analysis in ['Bridge-only','shared','DE-only','stable_recurrent_bridge_only']:
            z=q[(q.analysis==analysis)&q.candidate]
            ps.append({'condition':cond,'partition':analysis,'candidate_count':z.pert_id.nunique(),
                       'named_candidate_count':z.loc[~z.drug_name.str.startswith('BRD-',na=False),'pert_id'].nunique()})
    pd.DataFrame(ps).to_csv(OUT/'partition_reversal_summary.csv',index=False)

    # Radiation hypotheses, preserving separate exact perturbagen IDs.
    terms=['pentoxifylline','sitagliptin','linagliptin','saxagliptin','alogliptin','omipalisib','KU-0060648','NU-7441',
           'RG-547','THZ1','afatinib','lapatinib','neratinib','JQ1','molibresib']
    r=parts[(parts.condition=='Radiation injury')&(parts.analysis=='stable_recurrent')].copy();out=[]
    for term in terms:
        q=r[r.drug_name.map(norm)==norm(term)]
        if q.empty:out.append({'requested_hypothesis':term,'represented':False})
        else:
            for _,z in q.iterrows():out.append({'requested_hypothesis':term,'represented':True,'pert_id':z.pert_id,'method':z.method,
              'median_reversal':z.median_reversal,'sign_consistency':z.sign_consistency,'competitive_p':z.competitive_p,'competitive_q':z.competitive_q,
              'contexts':z.context_count,'cells':z.cell_count,
              'classification':'reverses' if z.median_reversal>0 and z.sign_consistency>=.6 else 'reinforces' if z.median_reversal<0 and z.sign_consistency<=.4 else 'mixed/context-dependent'})
    pd.DataFrame(out).to_csv(OUT/'radiation_hypothesis_direction_audit.csv',index=False)

    # Figure emphasizes effect sizes and uncertainty rather than the arbitrary top-1% counts.
    sns.set_theme(style='whitegrid',context='talk');fig,ax=plt.subplots(2,2,figsize=(15,11))
    pc=pd.DataFrame(method_rows);x=np.arange(6);w=.36
    cross=[]
    for cond in CONDITIONS[3:]:
        q=space[space.condition==cond];cross.append([q[(q.method==m)&q.cross_study_reproducible].pert_id.nunique() for m in ['Bridge','DE']])
    cross=np.array(cross);ax[0,0].bar(np.arange(3)-w/2,cross[:,0],w,label='Bridge');ax[0,0].bar(np.arange(3)+w/2,cross[:,1],w,label='DE')
    ax[0,0].set_xticks(range(3),['Radiation','Bone','Muscle']);ax[0,0].set_ylabel('Cross-study reproducible drugs');ax[0,0].legend();ax[0,0].set_title('Positive reversal in all three datasets')
    colors=['#2b6cb0' if v>0 else '#c53030' for v in pc.median_bridge_minus_de]
    ax[0,1].barh(range(6),pc.median_bridge_minus_de,color=colors);ax[0,1].axvline(0,color='black',lw=1);ax[0,1].set_yticks(range(6),['MS','Crohn','SLE','Radiation','Bone','Muscle']);ax[0,1].set_xlabel('Median paired score: Bridge − DE');ax[0,1].set_title('Identical-drug paired comparison')
    aa=audit[audit.represented].copy();cats=['reverses','mixed/context-dependent','reinforces'];tab=aa.groupby(['method','classification']).size().unstack(fill_value=0).reindex(columns=cats,fill_value=0)
    bottom=np.zeros(2)
    for c,col in zip(cats,['#2f855a','#d69e2e','#c53030']):ax[1,0].bar(tab.index,tab[c],bottom=bottom,label=c,color=col);bottom+=tab[c].to_numpy()
    ax[1,0].set_ylabel('Available highlighted drugs');ax[1,0].set_title('de Weerd drug direction audit');ax[1,0].legend(fontsize=10)
    ns=(nulls.groupby(['condition','method']).agg(tested=('drug_name','size'),significant=('sign_shuffle_q',lambda x:(x<.05).sum())).reset_index())
    piv=ns.pivot(index='condition',columns='method',values='significant').reindex(CONDITIONS)
    ax[1,1].bar(np.arange(6)-w/2,piv.Bridge,w,label='Bridge');ax[1,1].bar(np.arange(6)+w/2,piv.DE,w,label='DE')
    ax[1,1].set_xticks(range(6),['MS','Crohn','SLE','Rad','Bone','Muscle'],rotation=30);ax[1,1].set_ylabel('Prioritized drugs, sign-null q<0.05');ax[1,1].set_title('Prespecified prioritized null tests');ax[1,1].legend()
    fig.suptitle('Direction-aware LINCS reversal benchmark',fontweight='bold');fig.tight_layout();
    fig.savefig(FIG/'direction_aware_reversal_summary.png',dpi=300,bbox_inches='tight');fig.savefig(FIG/'direction_aware_reversal_summary.pdf',bbox_inches='tight');plt.close(fig)
    print(unified.to_string(index=False))
if __name__=='__main__':main()
