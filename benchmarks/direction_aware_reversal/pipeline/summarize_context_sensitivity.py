#!/usr/bin/env python3
"""Retain and summarize cell/dose/time dependence for prioritized compounds."""
from pathlib import Path
import json
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parents[1];OUT=HERE/'results';WORK=HERE/'work/lincs/GSE92742'
def main():
    n=pd.read_csv(OUT/'prioritized_candidate_nulls.csv');defs=pd.read_csv(OUT/'reversal_signature_definitions.csv')
    meta=pd.read_parquet(OUT/'lincs_exemplar_contexts.parquet');scores=np.load(WORK/'reversal_context_scores.npy',mmap_mode='r')
    didx=dict(zip(defs.definition_id,range(len(defs))));rows=[]
    for _,r in n.iterrows():
        mask=meta.pert_id.eq(r.pert_id).to_numpy();q=meta.loc[mask,['cell_id','pert_idose','pert_itime','distil_nsample']].copy()
        q['score']=np.asarray(scores[mask,didx[r.definition_id]])
        q['condition']=r.condition;q['method']=r.method;q['definition_id']=r.definition_id;q['pert_id']=r.pert_id;q['drug_name']=r.drug_name
        rows.append(q)
    raw=pd.concat(rows,ignore_index=True);raw.to_csv(OUT/'prioritized_context_scores.csv.gz',index=False,compression='gzip')
    grp=(raw.groupby(['condition','method','definition_id','pert_id','drug_name','cell_id','pert_idose','pert_itime'],as_index=False)
         .agg(median_reversal=('score','median'),sign_consistency=('score',lambda x:(x>0).mean()),signatures=('score','size'),distil_replicates=('distil_nsample','sum')))
    grp.to_csv(OUT/'prioritized_cell_dose_time_sensitivity.csv.gz',index=False,compression='gzip')
    summary=(grp.groupby(['condition','method','pert_id','drug_name'],as_index=False)
             .agg(context_groups=('median_reversal','size'),cell_lines=('cell_id','nunique'),
                  min_context_reversal=('median_reversal','min'),max_context_reversal=('median_reversal','max'),
                  context_sign_consistency=('median_reversal',lambda x:(x>0).mean())))
    summary.to_csv(OUT/'prioritized_context_sensitivity_summary.csv',index=False)
    (OUT/'context_sensitivity_provenance.json').write_text(json.dumps({'unit':'exact pert_id x cell x dose x time','aggregation':'median Level-5 signature reversal','heterogeneity_not_pooled':True},indent=2)+'\n')
if __name__=='__main__':main()
