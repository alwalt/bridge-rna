#!/usr/bin/env python3
"""Independent protein-level validation of the frozen Task 3 response modes."""
from __future__ import annotations
import json, re, hashlib
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd, requests
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE=Path(__file__).resolve().parents[1]; R=HERE/'results'; W=HERE/'work/task3_proteomics'; O=R/'task3_proteomics'
O.mkdir(parents=True,exist_ok=True)
MEM=R/'task3b_contrast_sample_membership.csv'; MODES=R/'task3c_cluster_assignments.csv'
URL='https://biit.cs.ut.ee/gprofiler/api/gost/profile/'

def symbol(s):
 m=re.search(r'\bGN=([^ ]+)',str(s)); return m.group(1).upper() if m else ''
def ratio_cols(d): return [c for c in d if c.startswith('Ratios:')]
def read_xlsx(p): return pd.read_excel(p)
def read_txt(p): return pd.read_csv(p,sep='\t',low_memory=False)
def prep(d):
 d=d[d['Master'].astype(str).eq('IsMasterProtein')].copy(); d['gene_symbol']=d.Description.map(symbol)
 d=d[d.gene_symbol.ne('')].drop_duplicates('gene_symbol'); return d

def api_matches():
 mem=pd.read_csv(MEM); modes=pd.read_csv(MODES)[['contrast_id','geometry_cluster']]
 mem=mem[mem.OSD.isin(['OSD-47','OSD-48','OSD-137'])].merge(modes,on='contrast_id')
 rows=[]
 root=R/'task3_manifest_api_validation'
 for osd in mem.OSD.unique():
  a=pd.read_csv(root/f'api_{osd}_sample_metadata.csv',dtype=str,low_memory=False).fillna('')
  a=a[a['investigation.study assays.study assay measurement type'].str.contains('protein',case=False,na=False)]
  a=a.drop_duplicates('id.sample name')
  z=mem[mem.OSD.eq(osd)].merge(a,left_on='sample_id',right_on='id.sample name',how='left',indicator=True)
  for _,x in z.iterrows():
   rows.append({'OSD':osd,'contrast_id':x.contrast_id,'mode':f'Mode {x.geometry_cluster}',
    'rna_sample_id':x.sample_id,'proteomics_sample_id':x.get('id.sample name',''),
    'condition':x.condition,'exact_identifier_match':x['_merge']=='both','protein_assay_found':x['_merge']=='both'})
 out=pd.DataFrame(rows); out.to_csv(O/'animal_matching_validation.csv',index=False); return mem,out

def build_responses(mem):
 responses={}; scope=[]; gene_rows=[]
 # OSD-47: processed table reports group means, not individual reporter channels.
 p=next((W/'OSD-47').glob('*TargetProtein.xlsx')); d=prep(read_xlsx(p))
 fl=[c for c in d if 'FLIGHT_Casis' in c and c.startswith('Ratios:')][0]; gc=[c for c in d if 'GC_Casis' in c and c.startswith('Ratios:')][0]
 d=d.set_index('gene_symbol'); s=(np.log2(pd.to_numeric(d[fl],errors='coerce'))-np.log2(pd.to_numeric(d[gc],errors='coerce'))).dropna()
 responses['OSD-47_pooled']=s; scope.append({'profile':'OSD-47_pooled','OSD':'OSD-47','mode':'Mode 1','resolution':'mission-level pooled','constituent_contrasts':'C11__OSD-47__RR1-CASIS__21-day | C12__OSD-47__RR1-CASIS__22-day','proteins':len(s),'limitation':'cannot separate 21-day and 22-day contrasts'})
 # OSD-48: FLT and GC are separate TMT plexes tied by common pools; output is group-level.
 fp=next((W/'OSD-48').glob('*GroupC*TargetProtein.xlsx')); gp=next((W/'OSD-48').glob('*GroupB*TargetProtein.xlsx'))
 f=prep(read_xlsx(fp)).set_index('gene_symbol'); g=prep(read_xlsx(gp)).set_index('gene_symbol')
 fc=[c for c in f if 'NASA FLT' in c and c.startswith('Ratios:')][0]; gc=[c for c in g if 'NASA GC' in c and c.startswith('Ratios:')][0]
 common=f.index.intersection(g.index); s=np.log2(pd.to_numeric(f.loc[common,fc],errors='coerce'))-np.log2(pd.to_numeric(g.loc[common,gc],errors='coerce')); s=s.dropna()
 responses['OSD-48_pooled']=s; scope.append({'profile':'OSD-48_pooled','OSD':'OSD-48','mode':'Mixed Mode 1/2','resolution':'mission-level pooled','constituent_contrasts':'C13__OSD-48__RR1-NASA__37-day | C14__OSD-48__RR1-NASA__37-day','proteins':len(s),'limitation':'cannot separate immediate Mode 2 from carcass Mode 1'})
 # OSD-137: individual reporter ratios mapped authoritatively from assay TMT labels.
 api=pd.read_csv(R/'task3_manifest_api_validation/api_OSD-137_sample_metadata.csv',dtype=str,low_memory=False).fillna('')
 api=api[api['investigation.study assays.study assay measurement type'].str.contains('protein',case=False,na=False)]
 api=api[~api['id.sample name'].str.contains('Pool')].drop_duplicates('id.sample name')
 tables={}
 for p in (W/'OSD-137').rglob('*TargetProtein.txt'):
  d=prep(read_txt(p)); tables[p.name]=d.set_index('gene_symbol')
 values={}
 for x in api.itertuples(index=False):
  sid=getattr(x,api.columns.get_loc('id.sample name')) if False else x[api.columns.get_loc('id.sample name')]
  label=x[api.columns.get_loc('assay.label')].replace('N','_N').replace('C','_C')
  fn=x[api.columns.get_loc('assay.protein assignment file')]; tab=tables[fn]
  col=f'Ratios: ({label}) / (131)'; values[sid]=np.log2(pd.to_numeric(tab[col],errors='coerce'))
 mat=pd.DataFrame(values)
 for cid,z in mem[mem.OSD.eq('OSD-137')].groupby('contrast_id'):
  fl=z.loc[z.condition.eq('FLT'),'sample_id']; gcids=z.loc[z.condition.eq('GC'),'sample_id']
  s=mat[fl].mean(axis=1)-mat[gcids].mean(axis=1); s=s.dropna(); key=cid
  mode=f"Mode {int(z.geometry_cluster.iloc[0])}"; responses[key]=s
  scope.append({'profile':key,'OSD':'OSD-137','mode':mode,'resolution':'animal-level','constituent_contrasts':cid,'proteins':len(s),'limitation':'none'})
 for k,s in responses.items():
  gene_rows.extend({'profile':k,'gene_symbol':g,'log2_protein_effect':v} for g,v in s.items())
 return responses,pd.DataFrame(scope),pd.DataFrame(gene_rows)

def similarity(responses,scope):
 common=set.intersection(*(set(x.index) for x in responses.values())); names=list(responses); X=np.array([responses[n].loc[sorted(common)] for n in names])
 cos=X@X.T/(np.linalg.norm(X,axis=1)[:,None]*np.linalg.norm(X,axis=1)[None,:])
 sp=np.array([[spearmanr(X[i],X[j]).statistic for j in range(len(X))] for i in range(len(X))])
 pd.DataFrame(cos,index=names,columns=names).to_csv(O/'proteomic_cosine_similarity.csv'); pd.DataFrame(sp,index=names,columns=names).to_csv(O/'proteomic_spearman_similarity.csv')
 rows=[]; md=scope.set_index('profile')['mode']
 for i in range(len(names)):
  for j in range(i+1,len(names)):
   relation='same_mode' if md[names[i]]==md[names[j]] and not md[names[i]].startswith('Mixed') else ('different_mode' if not md[names[i]].startswith('Mixed') and not md[names[j]].startswith('Mixed') else 'mixed_mode')
   rows.append({'profile_a':names[i],'profile_b':names[j],'mode_relation':relation,'cosine':cos[i,j],'spearman':sp[i,j]})
 pairs=pd.DataFrame(rows); pairs.to_csv(O/'proteomic_pairwise_similarity.csv',index=False)
 pairs.groupby('mode_relation').agg(pairs=('cosine','size'),cosine_mean=('cosine','mean'),cosine_median=('cosine','median'),spearman_mean=('spearman','mean'),spearman_median=('spearman','median')).reset_index().to_csv(O/'same_vs_different_mode_similarity.csv',index=False)
 fig,ax=plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
 for a,m,title in zip(ax,[cos,sp],['Cosine','Spearman']):
  im=a.imshow(m,cmap='RdBu_r',vmin=-1,vmax=1); a.set(xticks=range(len(names)),yticks=range(len(names)),xticklabels=names,yticklabels=names,title=title); a.tick_params(axis='x',rotation=55,labelsize=8); fig.colorbar(im,ax=a,shrink=.8)
 fig.suptitle(f'Protein FLT−GC response similarity ({len(common):,} proteins shared)'); fig.savefig(O/'proteomic_similarity_heatmaps.png',dpi=400,bbox_inches='tight'); fig.savefig(O/'proteomic_similarity_heatmaps.pdf',bbox_inches='tight'); plt.close(fig)
 return pairs,sorted(common)

def enrich(responses):
 queries={}
 for k,s in responses.items():
  queries[f'{k}__up']=s.nlargest(min(250,len(s))).index.tolist(); queries[f'{k}__down']=s.nsmallest(min(250,len(s))).index.tolist()
 raw={'result':[]}
 for query_name, genes in queries.items():
  profile=query_name.rsplit('__',1)[0]
  payload={'organism':'mmusculus','query':genes,'sources':['GO:BP','KEGG','REAC'],'user_threshold':0.05,'no_evidences':False,
           'domain_scope':'custom','background':responses[profile].index.tolist()}
  r=requests.post(URL,json=payload,timeout=180)
  if not r.ok:
   raise RuntimeError(f'g:Profiler failed for {query_name}: HTTP {r.status_code}: {r.text[:1000]}')
  result=r.json().get('result',[])
  for row in result: row['query']=query_name
  raw['result'].extend(result)
 (O/'pathway_enrichment_raw.json').write_text(json.dumps(raw,indent=2))
 d=pd.DataFrame(raw['result']); d.to_csv(O/'pathway_enrichment.csv',index=False)
 themes={'Fatty-acid/lipid metabolism':r'fatty acid|lipid|beta.oxid|acyl|lipoprotein','Small-molecule/catabolic metabolism':r'small molecule|catabolic|organic acid|amino acid','Peroxisomal metabolism':r'peroxis','Cholesterol metabolism':r'cholesterol|sterol','Detoxification':r'detox|xenobiotic|glutathione|drug metabolism|cytochrome p450','Complement/coagulation':r'complement|coagulation|hemostasis'}
 rows=[]
 for q in queries:
  z=d[d['query'].eq(q)] if not d.empty else d
  for theme,pat in themes.items():
   hit=z[z.name.str.contains(pat,case=False,regex=True,na=False)].sort_values('p_value').head(1)
   rows.append({'query':q,'profile':q.rsplit('__',1)[0],'direction':q.rsplit('__',1)[1],'theme':theme,'adjusted_p':hit.p_value.iloc[0] if len(hit) else np.nan,'term':hit.name.iloc[0] if len(hit) else ''})
 t=pd.DataFrame(rows); t.to_csv(O/'metabolic_program_directionality.csv',index=False)
 piv=t.assign(score=lambda x:-np.log10(x.adjusted_p)).pivot(index='theme',columns='query',values='score')
 fig,ax=plt.subplots(figsize=(13,5.5),layout='constrained'); im=ax.imshow(piv.fillna(0),cmap='magma',aspect='auto'); ax.set(yticks=range(len(piv)),yticklabels=piv.index,xticks=range(len(piv.columns)),xticklabels=piv.columns); ax.tick_params(axis='x',rotation=60,labelsize=7); fig.colorbar(im,ax=ax,label='−log10 adjusted p'); ax.set_title('Directional protein-response enrichment (blank/0 = not significant)'); fig.savefig(O/'protein_pathway_directionality.png',dpi=400,bbox_inches='tight'); fig.savefig(O/'protein_pathway_directionality.pdf',bbox_inches='tight'); plt.close(fig)
 return d,t

def bridge_comparison(scope,pairs):
 b=pd.read_csv(R/'task3c_cosine_bridgerna.csv',index_col=0); cmap={}
 for x in scope.itertuples():
  cs=x.constituent_contrasts.split(' | ')
  cmap[x.profile]=cs
 rows=[]
 for x in pairs.itertuples():
  vals=[b.loc[a,c] for a in cmap[x.profile_a] for c in cmap[x.profile_b]]
  rows.append({'profile_a':x.profile_a,'profile_b':x.profile_b,'mode_relation':x.mode_relation,'proteomic_cosine':x.cosine,'BridgeRNA_cosine_mean_across_constituent_contrasts':np.mean(vals)})
 d=pd.DataFrame(rows); d.to_csv(O/'proteomics_vs_bridgerna_geometry.csv',index=False)
 fig,ax=plt.subplots(figsize=(6.5,5.2),layout='constrained')
 colors={'same_mode':'#2b8cbe','different_mode':'#d7301f','mixed_mode':'#777777'}
 for relation,z in d.groupby('mode_relation'):
  ax.scatter(z.BridgeRNA_cosine_mean_across_constituent_contrasts,z.proteomic_cosine,label=relation.replace('_',' '),s=55,color=colors[relation])
 ax.axhline(0,color='0.75',lw=1); ax.axvline(0,color='0.75',lw=1)
 ax.set(xlabel='BridgeRNA FLT−GC response cosine',ylabel='Proteomic FLT−GC response cosine',title='Proteomic vs BridgeRNA response geometry')
 ax.legend(frameon=False)
 fig.savefig(O/'proteomics_vs_bridgerna_geometry.png',dpi=400,bbox_inches='tight'); fig.savefig(O/'proteomics_vs_bridgerna_geometry.pdf',bbox_inches='tight'); plt.close(fig)
 return d

def main():
 mem,matches=api_matches(); responses,scope,genes=build_responses(mem); scope.to_csv(O/'proteomics_analysis_scope.csv',index=False); genes.to_parquet(O/'protein_effects_long.parquet',index=False)
 pairs,common=similarity(responses,scope); enr,themes=enrich(responses); bridge=bridge_comparison(scope,pairs)
 conclusion='inconclusive: only OSD-137 retains animal-level quantitative reporter channels; OSD-47 and OSD-48 published tables are mission-level aggregates, and OSD-48 pools one contrast from each mode'
 (O/'conclusion.txt').write_text(conclusion+'\n')
 prov={'created_utc':datetime.now(timezone.utc).isoformat(),'profiles':len(responses),'shared_proteins':len(common),'conclusion':conclusion,'OSD-168':'excluded; non-independent ERCC/resequencing material','mode_assignments':'frozen Task 3C labels; no reclustering','statistics':'descriptive'}; (O/'provenance.json').write_text(json.dumps(prov,indent=2))
 print(scope.to_string(index=False)); print('\n',pd.read_csv(O/'same_vs_different_mode_similarity.csv').to_string(index=False)); print('\nCONCLUSION:',conclusion)
if __name__=='__main__': main()
