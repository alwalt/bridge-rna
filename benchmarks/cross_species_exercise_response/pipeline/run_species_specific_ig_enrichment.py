#!/usr/bin/env python3
"""Species-specific consensus IG enrichment within fixed exercise axes."""
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
OUT=HERE/'results/latent_axis_attribution/species_specific_enrichment'; OUT.mkdir(parents=True,exist_ok=True)
URL='https://biit.cs.ut.ee/gprofiler/api/gost/profile/'
AXES={'Axis A':['GSE108643','GSE86931','GSE126962','GSE132520'],
      'Axis B':['GSE71972','GSE87748','GSE97718']}

def build_rankings(genes):
    meta=pd.read_csv(HERE/'results/response_contrasts.csv').sort_values('contrast_id').reset_index(drop=True)
    values=np.load(HERE/'work/latent_axis_attribution/study_integrated_gradient_changes.npy')
    if values.shape!=(len(meta),len(genes)): raise AssertionError('IG matrix alignment failure')
    index={g:i for i,g in enumerate(meta.GSE)}; rows=[]; queries={}
    for axis,gses in AXES.items():
      for species in ['human','mouse']:
        selected=[g for g in gses if meta.loc[index[g],'species']==species]
        # Match the original axis consensus: average signed attribution changes,
        # then rank by magnitude. This allows consistent directional effects to reinforce.
        score=values[[index[g] for g in selected]].mean(axis=0); order=np.argsort(-np.abs(score))
        query=f"{axis.lower().replace(' ','_')}__{species}"; queries[query]=[genes[i] for i in order[:100]]
        for rank,i in enumerate(order,1): rows.append({'axis':axis,'species':species,'rank':rank,'gene':genes[i],'mean_ig_change':score[i],'absolute_mean_ig_change':abs(score[i]),'studies':len(selected)})
    ranking=pd.DataFrame(rows); ranking.to_parquet(OUT/'species_specific_ig_rankings.parquet',index=False)
    pd.DataFrame([{'query':q,'gene':g,'rank':i+1} for q,v in queries.items() for i,g in enumerate(v)]).to_csv(OUT/'species_specific_top100_genes.csv',index=False)
    return queries

def run_enrichment(queries,background):
    payload={'organism':'hsapiens','query':queries,'sources':['GO:BP','KEGG','REAC'],'user_threshold':0.05,'domain_scope':'custom','background':background,'no_evidences':False}
    response=requests.post(URL,json=payload,timeout=300); response.raise_for_status(); raw=response.json(); (OUT/'species_specific_enrichment_raw.json').write_text(json.dumps(raw,indent=2)+'\n')
    result=pd.DataFrame(raw.get('result',[])); keep=['query','source','native','name','p_value','significant','term_size','query_size','intersection_size','effective_domain_size','precision','recall']
    if not result.empty: result=result[[c for c in keep if c in result]].sort_values(['query','source','p_value'])
    result.to_parquet(OUT/'species_specific_enrichment.parquet',index=False); result.to_csv(OUT/'species_specific_enrichment.csv',index=False)
    return result,raw

def compare_and_plot(queries,result):
    comparisons=[]; pathway_rows=[]
    for axis in ['axis_a','axis_b']:
      h=f'{axis}__human'; m=f'{axis}__mouse'; hg=set(queries[h]); mg=set(queries[m]); rec={'axis':axis.replace('_',' ').title(),'human_top100':100,'mouse_top100':100,'shared_top100_genes':len(hg&mg),'gene_jaccard':len(hg&mg)/len(hg|mg)}
      for source in ['GO:BP','KEGG','REAC']:
        ht=set(result[(result['query']==h)&(result.source==source)].native); mt=set(result[(result['query']==m)&(result.source==source)].native)
        rec[f'{source}_human_terms']=len(ht); rec[f'{source}_mouse_terms']=len(mt); rec[f'{source}_shared_terms']=len(ht&mt); rec[f'{source}_human_only']=len(ht-mt); rec[f'{source}_mouse_only']=len(mt-ht); rec[f'{source}_jaccard']=len(ht&mt)/len(ht|mt) if ht|mt else np.nan
        for term in sorted(ht|mt): pathway_rows.append({'axis':rec['axis'],'source':source,'native':term,'name':result.loc[(result.source==source)&(result.native==term),'name'].iloc[0],'human_significant':term in ht,'mouse_significant':term in mt})
      comparisons.append(rec)
      z=result[result['query'].isin([h,m])].copy(); selected=z.groupby('query',group_keys=False).head(10)[['source','native','name']].drop_duplicates(); selected['term']=selected.source+' | '+selected.name
      matrix=pd.DataFrame(0.0,index=selected.term,columns=['human','mouse'])
      for _,r in z.iterrows():
        term=f'{r.source} | {r["name"]}'; species=r['query'].split('__')[1]
        if term in matrix.index: matrix.loc[term,species]=max(matrix.loc[term,species],-np.log10(max(r.p_value,np.finfo(float).tiny)))
      if len(matrix)>2: matrix=matrix.iloc[leaves_list(linkage(pdist(matrix.values),method='average'))]
      fig,ax=plt.subplots(figsize=(7,max(5,.38*len(matrix)))); image=ax.imshow(matrix.values,aspect='auto',cmap='viridis',vmin=0)
      ax.set_xticks([0,1],['Human Top-100','Mouse Top-100']); ax.set_yticks(np.arange(len(matrix)),matrix.index)
      for y in range(len(matrix)):
        for x in range(2):
          v=matrix.iloc[y,x]; ax.text(x,y,f'{v:.1f}' if v else 'NS',ha='center',va='center',fontsize=8,color='white' if v>2 else 'black')
      fig.colorbar(image,ax=ax,label='−log10 adjusted p-value'); ax.set_title(f"{axis.replace('_',' ').title()}: species-specific IG enrichment"); fig.tight_layout(); fig.savefig(OUT/f'{axis}_human_mouse_enrichment.png',dpi=320,bbox_inches='tight'); fig.savefig(OUT/f'{axis}_human_mouse_enrichment.pdf',bbox_inches='tight'); plt.close(fig)
    pd.DataFrame(comparisons).to_csv(OUT/'human_mouse_enrichment_comparison.csv',index=False); pd.DataFrame(pathway_rows).to_csv(OUT/'significant_pathway_overlap.csv',index=False)

def main():
    genes=load_canonical_genes(ROOT/'data/ensembl/canonical_genes.csv')
    if len(genes)!=15165 or len(set(genes))!=15165: raise AssertionError('Expected exact 15,165-gene background')
    queries=build_rankings(genes); result,raw=run_enrichment(queries,genes); compare_and_plot(queries,result)
    provenance={'retrieved_utc':datetime.now(timezone.utc).isoformat(),'service':'g:Profiler g:GOSt','sources':['GO:BP','KEGG','REAC'],'multiple_testing':'g:SCS','threshold':0.05,'domain_scope':'custom','background_genes':15165,'ranking':'species-specific mean signed IG change within each fixed axis, ranked by absolute mean','top_n':100,'queries':{k:len(v) for k,v in queries.items()},'service_meta':raw.get('meta',{})}
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n'); print(pd.read_csv(OUT/'human_mouse_enrichment_comparison.csv').to_string(index=False))
if __name__=='__main__': main()
