#!/usr/bin/env python3
"""GO/KEGG/Reactome enrichment of frozen latent-axis IG gene sets."""
from __future__ import annotations
import json, sys
from datetime import datetime, timezone
from pathlib import Path
import matplotlib.pyplot as plt, numpy as np, pandas as pd, requests

HERE=Path(__file__).resolve().parents[1]; ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from src.fm_embed.vocab import load_canonical_genes
OUT=HERE/'results/latent_axis_attribution/enrichment'; OUT.mkdir(parents=True,exist_ok=True)
URL='https://biit.cs.ut.ee/gprofiler/api/gost/profile/'

def main():
    consensus=pd.read_csv(HERE/'results/latent_axis_attribution/axis_consensus_attributed_genes.csv')
    conserved=pd.read_csv(HERE/'results/full_transcriptome_de/conserved_high_ig_genes.csv')
    queries={
      'axis_a_top100':consensus.query("axis == 'Axis A' and rank <= 100").gene.tolist(),
      'axis_b_top100':consensus.query("axis == 'Axis B' and rank <= 100").gene.tolist(),
      'axis_a_conserved_human_mouse':conserved.query("axis == 'Axis A'").gene.tolist(),
      'axis_b_conserved_human_mouse':conserved.query("axis == 'Axis B'").gene.tolist(),
    }
    background=load_canonical_genes(ROOT/'data/ensembl/canonical_genes.csv')
    if len(background)!=15165 or len(set(background))!=15165: raise AssertionError('Background must be 15,165 unique genes')
    for name,genes in queries.items():
      if not genes or len(genes)!=len(set(genes)) or not set(genes)<=set(background): raise AssertionError(f'Invalid query {name}')
    payload={'organism':'hsapiens','query':queries,'sources':['GO:BP','KEGG','REAC'],'user_threshold':0.05,'domain_scope':'custom','background':background,'no_evidences':False}
    response=requests.post(URL,json=payload,timeout=300); response.raise_for_status(); raw=response.json()
    (OUT/'ig_pathway_enrichment_raw.json').write_text(json.dumps(raw,indent=2)+'\n')
    result=pd.DataFrame(raw.get('result',[]))
    keep=['query','source','native','name','p_value','significant','term_size','query_size','intersection_size','effective_domain_size','precision','recall','intersection']
    if not result.empty: result=result[[c for c in keep if c in result]].sort_values(['query','source','p_value'])
    result.to_parquet(OUT/'ig_pathway_enrichment.parquet',index=False); result.to_csv(OUT/'ig_pathway_enrichment.csv',index=False)
    meta=raw.get('meta',{}).get('result_metadata',{})
    summary=pd.DataFrame([{'gene_set':q,'source':s,'query_genes':len(queries[q]),'background_genes':len(background),'tested_terms':int(meta.get(s,{}).get('number_of_terms',0)),'significant_terms':int(((result['query']==q)&(result['source']==s)).sum()) if not result.empty else 0} for q in queries for s in ['GO:BP','KEGG','REAC']])
    summary.to_csv(OUT/'ig_pathway_enrichment_summary.csv',index=False)
    # Quantify whether A and B share enriched terms and genes.
    comparisons=[]
    for scope,a,b in [('top100','axis_a_top100','axis_b_top100'),('conserved','axis_a_conserved_human_mouse','axis_b_conserved_human_mouse')]:
      ga,gb=set(queries[a]),set(queries[b]); rec={'scope':scope,'axis_a_genes':len(ga),'axis_b_genes':len(gb),'shared_genes':len(ga&gb),'gene_jaccard':len(ga&gb)/len(ga|gb)}
      for source in ['GO:BP','KEGG','REAC']:
        ta=set(result.query('query == @a and source == @source').native); tb=set(result.query('query == @b and source == @source').native); rec[f'{source}_axis_a_terms']=len(ta); rec[f'{source}_axis_b_terms']=len(tb); rec[f'{source}_shared_terms']=len(ta&tb); rec[f'{source}_term_jaccard']=len(ta&tb)/len(ta|tb) if ta|tb else np.nan
      comparisons.append(rec)
    pd.DataFrame(comparisons).to_csv(OUT/'axis_biology_comparison.csv',index=False)
    # Compact top-term dot plot; facet labels remain explicit about query/source.
    top=result.groupby(['query','source'],group_keys=False).head(5).copy()
    top['label']=top['query'].str.replace('_',' ',regex=False)+' | '+top.source+' | '+top.name.str.slice(0,58)
    top=top.sort_values('p_value',ascending=False); fig,ax=plt.subplots(figsize=(12,max(6,.24*len(top))))
    ax.scatter(-np.log10(top.p_value.clip(lower=np.finfo(float).tiny)),np.arange(len(top)),s=25+18*top.intersection_size,c=top.source.map({'GO:BP':'#4C72B0','KEGG':'#DD8452','REAC':'#55A868'}))
    ax.set(yticks=np.arange(len(top)),yticklabels=top.label,xlabel='−log10 adjusted p-value',title='Frozen IG gene-set enrichment'); fig.tight_layout(); fig.savefig(OUT/'ig_pathway_enrichment_top_terms.png',dpi=320,bbox_inches='tight'); fig.savefig(OUT/'ig_pathway_enrichment_top_terms.pdf',bbox_inches='tight'); plt.close(fig)
    provenance={'service':'g:Profiler g:GOSt','endpoint':URL,'retrieved_utc':datetime.now(timezone.utc).isoformat(),'organism':'hsapiens','sources':['GO:BP','KEGG','REAC'],'multiple_testing':'g:SCS','threshold':0.05,'domain_scope':'custom','background_genes':len(background),'queries':{k:len(v) for k,v in queries.items()},'definitions':{'axis_top100':'Top 100 by absolute mean IG attribution change across fixed axis studies','conserved_human_mouse':'intersection of human and mouse within-axis consensus Top 100 IG rankings'},'service_meta':raw.get('meta',{})}
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(summary.to_string(index=False)); print('\nTop terms\n',top[['query','source','native','name','p_value','intersection_size']].to_string(index=False))
if __name__=='__main__': main()
