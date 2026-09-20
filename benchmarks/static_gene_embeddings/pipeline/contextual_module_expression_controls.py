#!/usr/bin/env python3
"""Expression-decile and TPM-residual controls for contextual gene modules."""
from __future__ import annotations
import json,time
from itertools import combinations
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score,silhouette_score
from contextual_gene_modules import ROOT,HERE,WORK,VOCAB,LAYERS,SEEDS
from run_static_embeddings import enrich,GMTS

OUT=HERE/'results/contextual_gene_modules'

def residualize(x,tpm):
 t=tpm-tpm.mean();beta=(t[:,None]*x).sum(0)/np.square(t).sum();return (x-t[:,None]*beta[None,:]).astype('float32')

def main():
 started=time.time();genes=pd.read_csv(VOCAB).sort_values('token_id');names=genes.gene_symbol.astype(str).str.upper().tolist();rows=[]
 for dataset in ['GTEx','TCGA']:
  tpm=np.load(WORK/f'{dataset}_mean_log1p_tpm.npy');decile=pd.qcut(pd.Series(tpm),10,labels=False,duplicates='drop').to_numpy()+1
  base=pd.DataFrame({'token_id':genes.token_id,'gene':names,'cluster':decile})
  erow={'dataset':dataset,'control':'expression_deciles'}
  for lib,path in GMTS.items():
   e,_=enrich(base,lib.upper(),path);e.to_csv(OUT/f'{lib}_enrichment_{dataset}_expression_deciles.csv',index=False);erow[f'significant_{lib}_terms']=int(e.significant.sum())
  rows.append(erow)
  for layer in LAYERS:
   x=np.load(WORK/f'{dataset}_{layer}_mean_tokens.float32.npy');z=residualize(x,tpm);labs=[KMeans(10,random_state=s,n_init=10).fit_predict(z) for s in SEEDS];labels=labs[0]+1;row={'dataset':dataset,'layer':layer,'control':'linear_tpm_residual','silhouette':silhouette_score(z,labels,metric='euclidean',sample_size=3000,random_state=42),'stability_ari':np.mean([adjusted_rand_score(a,b) for a,b in combinations(labs,2)])};table=pd.DataFrame({'token_id':genes.token_id,'gene':names,'cluster':labels})
   for lib,path in GMTS.items():
    e,_=enrich(table,lib.upper(),path);e.to_csv(OUT/f'{lib}_enrichment_{dataset}_{layer}_tpm_residual.csv',index=False);row[f'significant_{lib}_terms']=int(e.significant.sum())
   rows.append(row);print(dataset,layer,flush=True)
 pd.DataFrame(rows).to_csv(OUT/'expression_control_summary.csv',index=False);(OUT/'expression_control_provenance.json').write_text(json.dumps({'status':'complete','expression_only':'ten equal-frequency bins of cohort mean log1p(TPM)','residualization':'per dataset/layer, subtract least-squares linear association of each of 512 dimensions with centered mean log1p(TPM)','clustering':{'k':10,'seeds':SEEDS,'n_init':10,'space':'original 512-D residual vectors'},'caveat':'linear residualization does not remove nonlinear or higher-order expression effects','elapsed_seconds':time.time()-started},indent=2)+'\n')
if __name__=='__main__':main()
