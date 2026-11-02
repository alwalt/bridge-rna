#!/usr/bin/env python3
"""Per-sample combined-atlas retrieval-state heatmaps for largest liver/muscle studies."""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np,pandas as pd,seaborn as sns
import matplotlib.pyplot as plt
from common import RESULTS

OUT=RESULTS/'osdr_multispecies'; FIG=OUT/'figures'
STUDIES={'liver':'OSD-379','muscle':'OSD-665'}

def cat(row):
 s=' | '.join(str(row.get(c,'')) for c in ['title','source_name','characteristics']).lower()
 rules=[('adipose inflammatory state',r'adipos|adipocyte',r'inflamm|\blps\b|cytokine|infect'),('adipose metabolic state',r'adipos|adipocyte|white fat|brown fat',None),('skeletal-muscle/exercise state',r'skeletal muscle|vastus|soleus|quadriceps|gastrocnemius|tibialis|digitorum|exercise|training',None),('fibroblast/stromal state',r'fibroblast|stromal|stroma|mesenchymal|synovial|\bmsc[_ -]',None),('immune/inflammatory state',r'blood|immune|inflamm|bone marrow|plasma cell|lymphocyte|macrophage|allograft|myeloma|mgus',None),('liver state',r'liver|hepatic|hepatocyte',None),('brain state',r'brain|cortex|cerebell|hippocamp|neural|neuron',None),('kidney state',r'kidney|renal',None),('lung state',r'lung|pulmonary|bronch',None),('heart state',r'heart|cardiac|ventric',None),('infection/immune stimulation',r'infect|viral|virus|bacter|\blps\b',None),('drug/toxicant response',r'\bdrug\b|toxic|inhibitor|agonist|antagonist|chemotherapy',None),('oxidative/stress-response state',r'oxidative|reactive oxygen|hypoxi|stress response|heat shock',None),('mitochondrial/metabolic state',r'mitochond|metabolic|glycoly|respiration',None),('proliferative/tumor state',r'prolifer|cell.?cycle|tumou?r|cancer|carcinoma|melanoma|adenoma|sarcoma|leukemia',None)]
 for name,p1,p2 in rules:
  if re.search(p1,s) and (p2 is None or re.search(p2,s)): return name
 return 'other/unknown'

def main():
 FIG.mkdir(parents=True,exist_ok=True);hits=pd.read_parquet(OUT/'individual_and_centroid_top100_all_references.parquet');man=pd.read_parquet(RESULTS/'osdr_atlas/eligible_sample_manifest.parquet')
 h=hits[(hits.query_type=='individual')&(hits.reference_set=='combined')&hits.method.isin(['Bridge-512 cosine','Hallmark-50 cosine'])].copy();h['category']=h.apply(cat,axis=1);h['affinity']=h.score/np.log2(h['rank']+1)
 allmetrics=[];selection=[]
 for domain,study in STUDIES.items():
  sm=man[man.study==study].copy();sm['condition_order']=sm.condition.map({'GC':0,'FLT':1});sm=sm.sort_values(['condition_order','group_id','sex','genotype','sample_id','matrix_row']);ids=sm.sample_id.tolist();selection.append({'domain':domain,'study':study,'samples':len(sm),'FLT':int((sm.condition=='FLT').sum()),'GC':int((sm.condition=='GC').sum()),'groups':sm.group_id.nunique(),'tissues':'; '.join(sorted(sm.tissue.unique())),'tie_break':'largest eligible sample count; lexicographically first OSD ID if tied'})
  z=h[(h.study==study)&h.osdr_sample_id.isin(ids)];agg=z.groupby(['method','osdr_sample_id','category'],as_index=False).agg(affinity=('affinity','sum'),mouse_weight=('affinity',lambda x:0.0))
  # species dominance computed separately with the same affinity weights
  sp=z.groupby(['method','osdr_sample_id','category','reference_species'],as_index=False).affinity.sum();piv=sp.pivot_table(index=['method','osdr_sample_id','category'],columns='reference_species',values='affinity',fill_value=0).reset_index();agg=agg.drop(columns='mouse_weight').merge(piv,on=['method','osdr_sample_id','category'],how='left').fillna(0)
  agg['species_evidence']=np.where(agg.get('mouse',0)>1.5*agg.get('human',0),'mouse',np.where(agg.get('human',0)>1.5*agg.get('mouse',0),'human','mixed'))
  full=pd.MultiIndex.from_product([['Bridge-512 cosine','Hallmark-50 cosine'],ids,sorted(agg.category.unique())],names=['method','osdr_sample_id','category']).to_frame(index=False).merge(agg,how='left').fillna({'affinity':0,'human':0,'mouse':0,'species_evidence':'none'})
  full['z_affinity']=full.groupby(['method','category']).affinity.transform(lambda x:(x-x.mean())/(x.std(ddof=0) or 1))
  cond=sm.set_index('sample_id').condition;full['condition']=full.osdr_sample_id.map(cond)
  shifts=full.groupby(['method','category','condition']).z_affinity.agg(['mean','median']).reset_index();w=shifts.pivot(index=['method','category'],columns='condition',values='mean').reset_index();w['flt_minus_gc']=w.FLT-w.GC
  sep=[]
  for (method,c),q in full.groupby(['method','category']):
   f=q[q.condition=='FLT'].z_affinity;g=q[q.condition=='GC'].z_affinity; direction=np.sign(f.mean()-g.mean());cons=((f>g.median()).mean() if direction>=0 else (f<g.median()).mean());sep.append((method,c,cons,abs(f.mean()-g.mean())>=.5 and cons>=.75))
  sep=pd.DataFrame(sep,columns=['method','category','condition_consistency','reproducible_separation']);w=w.merge(sep,on=['method','category']);w.insert(0,'study',study);w.insert(0,'domain',domain);allmetrics.append(w)
  order=w.assign(a=w.flt_minus_gc.abs()).groupby('category').a.max().sort_values(ascending=False).index.tolist();fig,axes=plt.subplots(2,1,figsize=(max(13,.45*len(ids)),max(10,.42*len(order)*2)),sharex=True)
  for ax,method in zip(axes,['Bridge-512 cosine','Hallmark-50 cosine']):
   q=full[full.method==method];mat=q.pivot(index='category',columns='osdr_sample_id',values='z_affinity').reindex(index=order,columns=ids);sns.heatmap(mat,ax=ax,cmap='RdBu_r',center=0,vmin=-2.5,vmax=2.5,cbar_kws={'label':'standardized per-sample retrieval affinity'})
   ev=q.pivot(index='category',columns='osdr_sample_id',values='species_evidence').reindex(index=order,columns=ids)
   sym={'human':'H','mouse':'M','mixed':'X','none':''}
   for yi,c in enumerate(order):
    for xi,sid in enumerate(ids): ax.text(xi+.5,yi+.5,sym.get(ev.loc[c,sid],''),ha='center',va='center',fontsize=5,color='black')
   sig=set(w[(w.method==method)&w.reproducible_separation].category)
   ax.set_yticklabels([('* ' if c in sig else '')+c for c in order],fontsize=8);ax.set_title(method+'  (H human, M mouse, X mixed; * reproducible FLT–GC separation)',loc='left');ax.set_ylabel('ARCHS4 biological-state category')
  axes[-1].set_xticklabels([f'{cond[sid]} | {sid}' for sid in ids],rotation=75,ha='right',fontsize=7);axes[-1].set_xlabel('Individual OSDR sample (GC then FLT)');fig.suptitle(f'{study} {domain}: combined human+mouse ARCHS4 retrieval states',y=1.003);fig.tight_layout()
  fig.savefig(FIG/f'{domain}_largest_study_per_sample_heatmap.png',dpi=300,bbox_inches='tight');fig.savefig(FIG/f'{domain}_largest_study_per_sample_heatmap.pdf',bbox_inches='tight');plt.close(fig)
 pd.DataFrame(selection).to_csv(OUT/'largest_liver_muscle_study_selection.csv',index=False);pd.concat(allmetrics).to_csv(OUT/'largest_study_flt_gc_category_shifts.csv',index=False)
if __name__=='__main__':main()
