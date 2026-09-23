#!/usr/bin/env python3
"""Prespecified sign and expression-matched nulls for prioritized drugs."""
from pathlib import Path
import json
import numpy as np, pandas as pd
from statsmodels.stats.multitest import multipletests

HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parents[1];OUT=HERE/'results';WORK=HERE/'work/lincs/GSE92742'
CAN=Path('/home/walt/bridge-rna/data/ensembl/canonical_genes.csv')
DISEASE={'Multiple sclerosis':'GSE138614','Crohn\'s disease':'GSE101794','Systemic lupus erythematosus':'GSE72509'}
SPACE={'Radiation injury':['GSE297090','GSE184119','GSE297560'],
       'Bone loss':['GSE189524','GSE276529','GSE273868'],
       'Muscle atrophy':['GSE211204','GSE113165','GSE234465']}

def expr_path(ds):
    base='deweerd_replication' if ds in DISEASE.values() else 'drug_discovery'
    return ROOT/f'benchmarks/{base}/work/prepared/{ds}_log1p_tpm.npy'

def main():
    rng=np.random.default_rng(20260922)
    genes=pd.read_csv(OUT/'lincs_exemplar_genes.csv');glist=genes.gene.astype(str).tolist();gidx={g:i for i,g in enumerate(glist)}
    canon=pd.read_csv(CAN).gene_symbol.astype(str).tolist();cidx={g:i for i,g in enumerate(canon)}
    meta=pd.read_parquet(OUT/'lincs_exemplar_contexts.parquet');X=np.load(WORK/'lincs_gse92742_exemplar_bing.npy',mmap_mode='r')
    defs=pd.read_csv(OUT/'reversal_signature_definitions.csv');members=pd.read_csv(OUT/'reversal_signature_members.csv.gz')
    primary=pd.read_parquet(OUT/'drug_reversal_by_contrast.parquet');space=pd.read_csv(OUT/'space_condition_consensus.csv.gz')
    audit=pd.read_csv(OUT/'published_drug_direction_audit.csv')
    # Three top named candidates per method/condition, plus all available published drugs.
    selected=[]
    for cond in DISEASE:
        q=primary[(primary.condition==cond)&primary.eligible_consensus&~primary.drug_name.str.startswith('BRD-',na=False)]
        for method in ['Bridge','DE']:
            selected += [(cond,method,x) for x in q[q.method==method].nlargest(3,'median_reversal').pert_id]
    for cond in SPACE:
        q=space[(space.condition==cond)&~space.drug_name.str.startswith('BRD-',na=False)]
        for method in ['Bridge','DE']:
            selected += [(cond,method,x) for x in q[q.method==method].nlargest(3,'consensus_reversal').pert_id]
    name_pid=meta[['pert_id','pert_iname']].drop_duplicates().set_index(meta[['pert_id','pert_iname']].drop_duplicates().pert_iname.str.lower()).pert_id.to_dict()
    for _,r in audit[audit.represented].iterrows():
        pid=name_pid.get(str(r.lincs_name).lower())
        if pid:selected.append((r.condition,r.method,pid))
    selected=list(dict.fromkeys(selected))

    # Mean-expression percentiles are averaged across each condition's frozen source datasets.
    expr_bin={}
    for cond in list(DISEASE)+list(SPACE):
        dsets=[DISEASE[cond]] if cond in DISEASE else SPACE[cond]; percentiles=[]
        for ds in dsets:
            m=np.asarray(np.load(expr_path(ds),mmap_mode='r')).mean(axis=0)
            s=pd.Series(m,index=canon).rank(pct=True);percentiles.append(s)
        avg=pd.concat(percentiles,axis=1).mean(axis=1)
        expr_bin[cond]={g:min(9,int(avg.get(g,0)*10)) for g in glist}
    lm=dict(zip(glist,genes.pr_is_lm.astype(int)))
    pools={}
    for cond in expr_bin:
        for g in glist:pools.setdefault((cond,expr_bin[cond][g],lm[g]),[]).append(g)

    rows=[]
    for cond,method,pid in selected:
        if cond in DISEASE:
            did=defs[(defs.condition==cond)&(defs.method==method)&(defs.analysis=='top500')].definition_id.iloc[0]
        else:
            q=defs[(defs.condition==cond)&(defs.method==method)&(defs.analysis=='stable_recurrent')]
            if q.empty:continue
            did=q.definition_id.iloc[0]
        mem=members[members.definition_id==did].copy();mem=mem[mem.gene.isin(gidx)]
        gi=np.array([gidx[g] for g in mem.gene]);w=mem.weight.to_numpy(float);w=w/np.linalg.norm(w)
        drug_rows=np.flatnonzero(meta.pert_id.to_numpy()==pid)
        dvec=np.median(np.asarray(X[drug_rows,:][:,gi],float),axis=0);dn=np.linalg.norm(dvec)
        obs=float(-dvec.dot(w)/dn) if dn else np.nan
        sign_null=np.empty(2000)
        for b in range(2000):sign_null[b]=-dvec.dot(rng.permutation(w))/dn
        matched=np.empty(200)
        for b in range(200):
            picked=[]
            for g in mem.gene:
                pool=pools[(cond,expr_bin[cond][g],lm[g])];picked.append(pool[rng.integers(len(pool))])
            xi=np.array([gidx[g] for g in picked]);rv=np.median(np.asarray(X[drug_rows,:][:,xi],float),axis=0)
            matched[b]=-rv.dot(w)/(np.linalg.norm(rv) or np.nan)
        rows.append({'condition':cond,'method':method,'definition_id':did,'pert_id':pid,
                     'drug_name':meta.loc[meta.pert_id==pid,'pert_iname'].iloc[0],
                     'consensus_signature_reversal':obs,
                     'sign_shuffle_p':(1+(sign_null>=obs).sum())/2001,
                     'expression_matched_p':(1+(matched>=obs).sum())/201,
                     'sign_null_mean':sign_null.mean(),'expression_matched_null_mean':np.nanmean(matched),
                     'contexts':len(drug_rows),'genes':len(mem)})
    out=pd.DataFrame(rows)
    for family in ['sign_shuffle','expression_matched']:
        out[family+'_q']=out.groupby(['condition','method'])[family+'_p'].transform(lambda x:multipletests(x,method='fdr_bh')[1])
    out.to_csv(OUT/'prioritized_candidate_nulls.csv',index=False)
    (OUT/'prioritized_null_provenance.json').write_text(json.dumps({
      'seed':20260922,'sign_permutations':2000,'expression_matched_modules':200,
      'matching':'condition source-expression mean decile plus L1000 landmark/inferred status',
      'statistic':'negative cosine to the median compound signature (auxiliary null statistic)',
      'scope':'top three named candidates per method/condition plus available de Weerd-highlighted drugs'},indent=2)+'\n')
    print(out.shape)
if __name__=='__main__':main()
