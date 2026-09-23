#!/usr/bin/env python3
"""Join frozen reversal results to the frozen expanded-ChEMBL graph."""
from pathlib import Path
import re, sqlite3, json
import numpy as np, pandas as pd

HERE=Path(__file__).resolve().parents[1]; ROOT=HERE.parents[1]; OUT=HERE/'results'
META=HERE/'work/lincs_metadata/GSE92742'
DB=ROOT/'benchmarks/deweerd_replication/work/chembl37_sqlite/chembl_37/chembl_37_sqlite/chembl_37.db'
EDGES=ROOT/'benchmarks/deweerd_replication/results/evaluation2_expanded/chembl37_broad_drug_target_edges.csv.gz'
SPACE=ROOT/'benchmarks/drug_discovery/results/expanded_chembl_sensitivity/expanded_drug_enrichment.csv.gz'
DISEASE=ROOT/'benchmarks/deweerd_replication/results/evaluation2_expanded/drug_enrichment_gt10.csv.gz'

def norm(x): return re.sub('[^a-z0-9]+','',str(x).lower())
def one(pattern):
    x=list(META.glob(pattern)); assert len(x)==1,(pattern,x); return x[0]

def main():
    primary=pd.read_parquet(OUT/'drug_reversal_by_contrast.parquet')
    consensus=pd.read_csv(OUT/'space_condition_consensus.csv.gz')
    pert=pd.read_csv(one('*pert_info*.gz'),sep='\t',low_memory=False)
    edges=pd.read_csv(EDGES)
    ids=edges.drug_id.drop_duplicates().tolist()
    con=sqlite3.connect(DB)
    q='''select md.chembl_id, md.pref_name, cs.standard_inchi_key
         from molecule_dictionary md left join compound_structures cs on md.molregno=cs.molregno'''
    chem=pd.read_sql_query(q,con); con.close(); chem=chem[chem.chembl_id.isin(ids)].copy()
    key_to_id=(chem.dropna(subset=['standard_inchi_key']).drop_duplicates('standard_inchi_key')
               .set_index('standard_inchi_key').chembl_id.to_dict())
    name_to_ids={}
    for _,r in chem.dropna(subset=['pref_name']).iterrows(): name_to_ids.setdefault(norm(r.pref_name),set()).add(r.chembl_id)
    edge_names=edges[['drug_id','drug_name']].drop_duplicates()
    for _,r in edge_names.iterrows(): name_to_ids.setdefault(norm(r.drug_name),set()).add(r.drug_id)
    pmap=pert[['pert_id','pert_iname','inchi_key']].drop_duplicates('pert_id').copy()
    def resolve(r):
        k=str(r.inchi_key)
        if k not in {'nan','-666',''} and k in key_to_id:return key_to_id[k],'inchi_key'
        candidates=name_to_ids.get(norm(r.pert_iname),set())
        return (next(iter(candidates)),'exact_name') if len(candidates)==1 else (None,'unmapped')
    resolved=pmap.apply(resolve,axis=1,result_type='expand');pmap[['drug_id','mapping_method']]=resolved
    pmap.to_csv(OUT/'lincs_to_expanded_chembl_mapping.csv.gz',index=False,compression='gzip')

    space=pd.read_csv(SPACE); disease=pd.read_csv(DISEASE)
    def disease_condition(x):
        x=str(x).lower()
        if 'multiple sclerosis' in x:return 'Multiple sclerosis'
        if 'crohn' in x:return "Crohn's disease"
        if 'lupus' in x:return 'Systemic lupus erythematosus'
        return None
    disease['condition']=disease['disease'].map(disease_condition) if 'disease' in disease else disease.condition
    if 'dataset' not in disease:disease['dataset']=disease['accession']
    if 'drug_rank' not in disease:disease['drug_rank']=disease['rank']
    enrich=pd.concat([space,disease],ignore_index=True,sort=False)
    enrich=enrich[['condition','method','dataset','drug_id','drug_name','target_count','overlap_count','overlap_genes','pvalue','fdr','drug_rank']].drop_duplicates()

    rev=pd.concat([primary[primary.condition.isin(['Multiple sclerosis',"Crohn's disease",'Systemic lupus erythematosus'])],
                   consensus.rename(columns={'consensus_reversal':'median_reversal','dataset_sign_consistency':'sign_consistency'})],ignore_index=True,sort=False)
    rev=rev.merge(pmap[['pert_id','drug_id','mapping_method']],on='pert_id',how='left')
    # For space consensus, a target is supported if it overlaps in any frozen dataset module.
    eagg=(enrich.groupby(['condition','method','drug_id'],as_index=False)
          .agg(target_supported=('overlap_count',lambda x:bool((x.fillna(0)>0).any())),
               target_overlap_genes=('overlap_genes',lambda x:';'.join(sorted(set(';'.join(x.dropna().astype(str)).split(';'))-{'','nan'}))),
               best_target_p=('pvalue','min'),best_target_fdr=('fdr','min')))
    rev=rev.merge(eagg,on=['condition','method','drug_id'],how='left')
    rev['target_supported']=rev.target_supported.fillna(False)
    def cls(r):
        if pd.isna(r.drug_id):return 'transcriptional reversal without mapped ChEMBL identity' if r.median_reversal>0 else 'unmapped ChEMBL identity'
        if not r.target_supported:return 'transcriptional reversal without direct ChEMBL target overlap' if r.median_reversal>0 else 'no direct module-target overlap'
        if r.median_reversal>0 and r.sign_consistency>=.6:return 'target overlap + transcriptional reversal'
        if r.median_reversal<0 and r.sign_consistency<=.4:return 'target overlap + transcriptional reinforcement'
        return 'target overlap + context-dependent/mixed reversal'
    rev['integration_class']=rev.apply(cls,axis=1)
    rev.to_parquet(OUT/'target_reversal_integration.parquet',index=False)

    candidates=rev[rev.candidate.fillna(False)].copy()
    counts=(candidates.groupby(['condition','method'],as_index=False)
            .agg(candidates=('pert_id','nunique'),target_supported=('target_supported','sum')))
    counts.to_csv(OUT/'target_supported_candidate_counts.csv',index=False)
    # Interpretable mapped shortlist; unknown BRD identifiers remain in the complete parquet.
    short=candidates[~candidates.drug_name.str.startswith('BRD-',na=False)].copy()
    short=short.sort_values(['condition','method','median_reversal'],ascending=[True,True,False])
    short.groupby(['condition','method'],group_keys=False).head(25).to_csv(OUT/'top_named_reversal_candidates.csv',index=False)
    prov={'mapping_priority':['exact full InChIKey','unique exact normalized preferred name'],
          'expanded_chembl_edge_file':str(EDGES.relative_to(ROOT)),'eligible_chembl_drugs':int(edges.drug_id.nunique()),
          'lincs_compounds':int(pmap.pert_id.nunique()),'mapped_lincs_compounds':int(pmap.drug_id.notna().sum()),
          'target_support_definition':'at least one module-overlapping target in the frozen expanded-ChEMBL enrichment'}
    (OUT/'target_integration_provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
    print(json.dumps(prov,indent=2))
if __name__=='__main__':main()
