#!/usr/bin/env python3
"""Characterize library-associated geometry in original Bridge embeddings."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr

HERE=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--dataset',action='append',type=Path,required=True); p.add_argument('--output',type=Path,default=HERE/'results/task4c_bridge_baseline'); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    frames=[]; arrays=[]
    for d in a.dataset:
        m=pd.read_parquet(d/'manifest.parquet'); z=np.load(d/'bridgerna_embeddings.npy'); assert len(m)==len(z); frames.append(m);arrays.append(z)
    m=pd.concat(frames,ignore_index=True); z=np.concatenate(arrays).astype(np.float64); rows=[]; diffs=[]; names=[]
    for (dataset,pair),q in m.groupby(['dataset','pair_id'],sort=False):
        if set(q.library_prep)!={'polyA','ribo'} or not q.same_rna_verified.all(): continue
        pa=q.index[q.library_prep.eq('polyA')];ri=q.index[q.library_prep.eq('ribo')]; a0=z[pa].mean(0);b0=z[ri].mean(0); d=b0-a0
        rows.append({'dataset':dataset,'pair_id':pair,'polyA_libraries':len(pa),'ribo_libraries':len(ri),'paired_cosine':cosine_similarity(a0[None],b0[None])[0,0],'euclidean_distance':euclidean(a0,b0),'difference_norm':np.linalg.norm(d)})
        diffs.append(d);names.append(f'{dataset}:{pair}')
    if not rows: raise ValueError('No authoritative same-RNA pairs')
    pairs=pd.DataFrame(rows); pairs.to_csv(a.output/'bridge_pair_metrics.csv',index=False)
    D=np.stack(diffs); C=cosine_similarity(D); pd.DataFrame(C,index=names,columns=names).to_csv(a.output/'library_difference_cosine.csv')
    # Exact cross-library donor retrieval in both directions.
    pa=[];ri=[]
    for (dataset,pair),q in m.groupby(['dataset','pair_id'],sort=False):
        if set(q.library_prep)=={'polyA','ribo'} and q.same_rna_verified.all():
            pa.append(z[q.index[q.library_prep.eq('polyA')]].mean(0));ri.append(z[q.index[q.library_prep.eq('ribo')]].mean(0))
    S=cosine_similarity(np.stack(pa),np.stack(ri)); ranks=np.r_[[1+(S[i]>S[i,i]).sum() for i in range(len(pa))],[1+(S[:,i]>S[i,i]).sum() for i in range(len(pa))]]
    n=min(len(D)-1,D.shape[1]); pca=PCA(n_components=max(1,n)).fit(D)
    pd.DataFrame({'component':np.arange(1,len(pca.explained_variance_ratio_)+1),'variance_explained':pca.explained_variance_ratio_,'cumulative_variance':np.cumsum(pca.explained_variance_ratio_)}).to_csv(a.output/'library_difference_svd.csv',index=False)
    summary={'samples':len(m),'datasets':m.dataset.nunique(),'verified_pairs':len(pairs),'median_pair_cosine':pairs.paired_cosine.median(),'mean_pair_cosine':pairs.paired_cosine.mean(),'pair_recall_at_1':float((ranks<=1).mean()),'pair_recall_at_5':float((ranks<=5).mean()),'pair_recall_at_10':float((ranks<=10).mean()),'pair_mrr':float((1/ranks).mean()),'median_pair_rank':float(np.median(ranks)),'difference_pc1_variance':float(pca.explained_variance_ratio_[0]),'interpretation':'A dominant direction is descriptive, not evidence of a universal causal library effect.'}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(pairs.describe().to_string());print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
