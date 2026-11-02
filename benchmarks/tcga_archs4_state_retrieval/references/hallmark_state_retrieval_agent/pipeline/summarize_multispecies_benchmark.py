#!/usr/bin/env python3
"""Create compact comparison tables for the multispecies extension."""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np,pandas as pd
from common import RESULTS,WORK
OUT=RESULTS/'osdr_multispecies';QW=WORK/'osdr_atlas';MW=WORK/'archs4_multispecies'
ALIASES={'liver':r'liver|hepatic|hepatocyte','muscle':r'muscle|soleus|gastrocnemius|quadriceps|tibialis|digitorum','brain':r'brain|cerebell|hippocamp|cortex','kidney':r'kidney|renal','lung':r'lung|pulmonary|bronch','heart':r'heart|cardiac|ventric','adipose':r'adipose|adipocyte','blood':r'blood|pbmc|leukocyte|lymphocyte|marrow'}
def tissue_key(x):
 for k,p in ALIASES.items():
  if re.search(p,str(x),re.I):return k
 return 'other'
def topmask(x,k):
 idx=np.argpartition(np.abs(x),-k)[-k:];m=np.zeros(len(x),bool);m[idx]=1;return m
def main():
 h=pd.read_parquet(OUT/'individual_and_centroid_top100_all_references.parquet');rep=pd.read_csv(OUT/'deterministic_study_representatives.csv');r=h[(h.query_type=='individual')&h.osdr_sample_id.isin(rep.sample_id)].copy();r['query_tissue_key']=r.tissue.map(tissue_key);txt=r[['title','source_name','characteristics']].fillna('').agg(' '.join,axis=1);r['same_tissue']=np.array([bool(re.search(ALIASES.get(k,'a^'),t,re.I)) for k,t in zip(r.query_tissue_key,txt)])
 rows=[]
 for k in [5,10,50,100]:
  for keys,b in r[r['rank']<=k].groupby(['reference_set','method']): rows.append({'reference_set':keys[0],'method':keys[1],'k':k,'mouse_fraction':(b.reference_species=='mouse').mean(),'human_fraction':(b.reference_species=='human').mean(),'tissue_coherence':b.same_tissue.mean(),'unique_gse':b.gse.nunique()})
 pd.DataFrame(rows).to_csv(OUT/'representative_topk_comparison.csv',index=False)
 # Functional signal coverage for representative combined hits from frozen profiles.
 qman=pd.read_parquet(RESULTS/'osdr_atlas/eligible_sample_manifest.parquet').sort_values('matrix_row');qh=np.load(QW/'hallmark_profiles.float32.npy');hh=np.load(WORK/'archs4_hallmark_signatures.float32.npy',mmap_mode='r');mh=np.load(MW/'mouse_hallmark_signatures.float32.npy',mmap_mode='r');lookup=dict(zip(qman.sample_id,qman.matrix_row));cov=[]
 for x in r[r.reference_set=='combined'].itertuples():
  q=qh[lookup[x.osdr_sample_id]];v=np.asarray(mh[x.reference_row] if x.reference_species=='mouse' else hh[x.reference_row]);
  rec={'osdr_sample_id':x.osdr_sample_id,'study':x.study,'condition':x.condition,'method':x.method,'rank':x.rank,'gsm':x.gsm,'reference_species':x.reference_species}
  for k in [5,10]:
   qm=topmask(q,k);rm=topmask(v,k);rec[f'signal_coverage_top{k}']=float(np.sum(qm&rm&(np.sign(q)==np.sign(v)))/k)
  cov.append(rec)
 pd.DataFrame(cov).to_parquet(OUT/'representative_combined_signal_coverage.parquet',index=False)
 # Retention/rank displacement of prior human-only analogs.
 prior=r[r.reference_set=='human'];comb=r[r.reference_set=='combined'];ret=[]
 for (sid,method),b in prior[prior['rank']<=10].groupby(['osdr_sample_id','method']):
  c=comb[(comb.osdr_sample_id==sid)&(comb.method==method)];rank=dict(zip(c.gsm,c['rank']))
  for x in b.itertuples():ret.append({'osdr_sample_id':sid,'method':method,'human_only_rank':x.rank,'gsm':x.gsm,'combined_rank_if_top100':rank.get(x.gsm,np.nan),'retained_combined_top100':x.gsm in rank})
 pd.DataFrame(ret).to_csv(OUT/'legacy_human_analog_rank_retention.csv',index=False)
 # Cases where a human sample outranks the first mouse result.
 first=r[(r.reference_set=='combined')].sort_values('rank').groupby(['osdr_sample_id','method']).first().reset_index();first[first.reference_species=='human'].to_csv(OUT/'human_ahead_of_mouse_cases.csv',index=False)
 d=pd.read_parquet(OUT/'combined_flt_vs_gc_differential_retrieval.parquet');strict=d[(d.bh_q<=.1)&(d.rrf_delta_flt_minus_gc>0)&(d.flt_replicate_prevalence>=.75)&(d.gc_replicate_prevalence<=.25)];strict.to_csv(OUT/'strict_combined_flt_enriched_hits.csv',index=False)
if __name__=='__main__':main()
