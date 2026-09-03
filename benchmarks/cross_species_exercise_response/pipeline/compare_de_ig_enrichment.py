#!/usr/bin/env python3
"""Compare pathway enrichment of axis-consensus edgeR and frozen IG rankings."""
from __future__ import annotations
import json, sys
from datetime import datetime, timezone
from pathlib import Path
import matplotlib.pyplot as plt, numpy as np, pandas as pd, requests
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist

HERE=Path(__file__).resolve().parents[1]; ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from src.fm_embed.vocab import load_canonical_genes
OUT=HERE/'results/latent_axis_attribution/de_ig_enrichment'; OUT.mkdir(parents=True,exist_ok=True)
URL='https://biit.cs.ut.ee/gprofiler/api/gost/profile/'
AXES={'Axis A':['human_GSE108643','human_GSE86931','mouse_GSE126962','mouse_GSE132520'],
      'Axis B':['human_GSE71972','human_GSE87748','mouse_GSE97718']}
LABELS={'de_top100':'DE Top-100','ig_top100':'IG Top-100','shared':'Shared IG+DE','high_ig_low_de':'High-IG/Low-DE'}

def build_sets(background):
    genes=pd.read_parquet(HERE/'results/full_transcriptome_de/within_vocabulary_gene_comparisons.parquet')
    ig=pd.read_csv(HERE/'results/latent_axis_attribution/axis_consensus_attributed_genes.csv')
    gene_sets={}; rank_rows=[]
    for axis,cids in AXES.items():
      parts=[]
      for cid in cids:
        x=genes[genes.contrast_id.eq(cid)].set_index('gene')
        # Untested genes receive the worst percentile, preventing sparse detection
        # from improving the axis consensus rank.
        pct=pd.Series(1.0,index=background); common=pct.index.intersection(x.index)
        pct.loc[common]=x.loc[common,'de_rank']/len(x)
        parts.append(pct.rename(cid))
      matrix=pd.concat(parts,axis=1); mean_pct=matrix.mean(axis=1); de_order=mean_pct.sort_values()
      de_top=set(de_order.head(100).index); de_top500=set(de_order.head(500).index)
      ig_top=set(ig.query('axis == @axis and rank <= 100').gene)
      sets={'de_top100':de_top,'ig_top100':ig_top,'shared':de_top&ig_top,'high_ig_low_de':ig_top-de_top500}
      for kind,values in sets.items(): gene_sets[f"{axis.lower().replace(' ','_')}__{kind}"]=sorted(values)
      for rank,(gene,score) in enumerate(de_order.items(),1): rank_rows.append({'axis':axis,'gene':gene,'consensus_de_rank':rank,'mean_de_rank_percentile':score,'in_de_top100':gene in de_top,'in_de_top500':gene in de_top500,'in_ig_top100':gene in ig_top})
    pd.DataFrame(rank_rows).to_parquet(OUT/'axis_consensus_de_rankings.parquet',index=False)
    rows=[]
    for query,values in gene_sets.items():
      axis,kind=query.split('__'); rows.extend({'axis':axis.replace('_',' ').title(),'gene_set':kind,'gene':g} for g in values)
    pd.DataFrame(rows).to_csv(OUT/'de_ig_gene_sets.csv',index=False)
    return gene_sets

def enrichment(gene_sets,background):
    payload={'organism':'hsapiens','query':gene_sets,'sources':['GO:BP','KEGG','REAC'],'user_threshold':0.05,'domain_scope':'custom','background':background,'no_evidences':False}
    response=requests.post(URL,json=payload,timeout=300); response.raise_for_status(); raw=response.json()
    (OUT/'de_ig_enrichment_raw.json').write_text(json.dumps(raw,indent=2)+'\n')
    result=pd.DataFrame(raw.get('result',[])); keep=['query','source','native','name','p_value','significant','term_size','query_size','intersection_size','effective_domain_size','precision','recall']
    if not result.empty: result=result[[c for c in keep if c in result]].sort_values(['query','source','p_value'])
    result.to_parquet(OUT/'de_ig_enrichment.parquet',index=False); result.to_csv(OUT/'de_ig_enrichment.csv',index=False)
    meta=raw.get('meta',{}).get('result_metadata',{})
    summary=pd.DataFrame([{'query':q,'axis':q.split('__')[0].replace('_',' ').title(),'gene_set':q.split('__')[1],'query_genes':len(gs),'source':s,'tested_terms':int(meta.get(s,{}).get('number_of_terms',0)),'significant_terms':int(((result['query']==q)&(result['source']==s)).sum()) if not result.empty else 0} for q,gs in gene_sets.items() for s in ['GO:BP','KEGG','REAC']])
    summary.to_csv(OUT/'de_ig_enrichment_summary.csv',index=False)
    return result,raw

def figures_and_comparisons(result):
    compact=[]
    programs={'stress':r'stress|heat shock|unfolded protein|hypoxi|oxidative',
      'inflammatory':r'inflamm|cytokine|interferon|immune|leukocyte|nf-kappa|interleukin',
      'metabolic':r'metabol|glycol|mitochond|fatty acid|oxidative phosphorylation|respirat'}
    audits=[]
    for axis in ['axis_a','axis_b']:
      qnames=[f'{axis}__{x}' for x in LABELS]
      z=result[result['query'].isin(qnames)].copy()
      for program,pattern in programs.items():
        for query in qnames: audits.append({'axis':axis.replace('_',' ').title(),'gene_set':query.split('__')[1],'program':program,'significant_matching_terms':int(z[z['query'].eq(query)].name.str.contains(pattern,case=False,regex=True).sum())})
      # Union of the six strongest terms per set; hierarchical row ordering on enrichment profiles.
      selected=z.groupby('query',group_keys=False).head(6)[['source','native','name']].drop_duplicates()
      selected['term']=selected.source+' | '+selected.name
      matrix=pd.DataFrame(0.0,index=selected.term,columns=list(LABELS))
      for _,r in z.iterrows():
        term=f'{r.source} | {r["name"]}'; kind=r['query'].split('__')[1]
        if term in matrix.index: matrix.loc[term,kind]=max(matrix.loc[term,kind],-np.log10(max(r.p_value,np.finfo(float).tiny)))
      if len(matrix)>2:
        order=leaves_list(linkage(pdist(matrix.values),method='average')); matrix=matrix.iloc[order]
      fig,ax=plt.subplots(figsize=(10,max(5,.38*len(matrix))))
      annot=matrix.map(lambda v:f'{v:.1f}' if v>0 else 'NS')
      shown=matrix.rename(columns=LABELS); image=ax.imshow(shown.values,aspect='auto',cmap='viridis',vmin=0)
      ax.set_xticks(np.arange(len(shown.columns)),shown.columns,rotation=25,ha='right'); ax.set_yticks(np.arange(len(shown.index)),shown.index)
      for yy in range(len(shown)):
        for xx in range(len(shown.columns)): ax.text(xx,yy,annot.iloc[yy,xx],ha='center',va='center',fontsize=8,color='white' if shown.iloc[yy,xx]>2 else 'black')
      fig.colorbar(image,ax=ax,label='−log10 adjusted p-value'); ax.set(title=f"{axis.replace('_',' ').title()}: DE versus IG pathway enrichment",xlabel='',ylabel=''); fig.tight_layout(); fig.savefig(OUT/f'{axis}_de_ig_enrichment.png',dpi=320,bbox_inches='tight'); fig.savefig(OUT/f'{axis}_de_ig_enrichment.pdf',bbox_inches='tight'); plt.close(fig)
      compact.append(z[['query','source','native','name','p_value','intersection_size','query_size']])
    pd.concat(compact).to_csv(OUT/'significant_pathway_comparison.csv',index=False)
    pd.DataFrame(audits).to_csv(OUT/'stress_inflammatory_metabolic_term_audit.csv',index=False)

def main():
    background=load_canonical_genes(ROOT/'data/ensembl/canonical_genes.csv')
    if len(background)!=15165 or len(set(background))!=15165: raise AssertionError('Expected exact 15,165-gene background')
    sets=build_sets(background); result,raw=enrichment(sets,background); figures_and_comparisons(result)
    provenance={'service':'g:Profiler g:GOSt','retrieved_utc':datetime.now(timezone.utc).isoformat(),'sources':['GO:BP','KEGG','REAC'],'multiple_testing':'g:SCS','threshold':0.05,'domain_scope':'custom','background_genes':15165,'de_consensus':'mean within-study absolute-edgeR-effect rank percentile; untested genes assigned worst percentile','definitions':{'DE Top-100':'top 100 axis-consensus DE magnitude','IG Top-100':'existing frozen axis-consensus IG Top 100','Shared IG+DE':'intersection of the two Top-100 sets','High-IG/Low-DE':'IG Top 100 excluding DE Top 500'},'queries':{k:len(v) for k,v in sets.items()},'service_meta':raw.get('meta',{})}
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(pd.read_csv(OUT/'de_ig_enrichment_summary.csv').to_string(index=False))
if __name__=='__main__': main()
