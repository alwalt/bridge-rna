#!/usr/bin/env python3
"""Matched random-gene controls and descriptive decoder/IG quadrants."""
import argparse
from pathlib import Path
import numpy as np,pandas as pd
from common import CONFIG,RESULTS

def main():
 p=argparse.ArgumentParser();p.add_argument("--results-dir",default="ig_validation");a=p.parse_args();base=RESULTS/a.results_dir if a.results_dir else RESULTS
 d=pd.read_parquet(base/"ig_de_gene_level_comparison.parquet");rng=np.random.default_rng(CONFIG["seed"]);controls=[];quadrants=[]
 contrast="contrast_label" if "contrast_label" in d else "contrast";keys=["sample_id",contrast,"cohort","hallmark"]
 for key,g in d.groupby(keys,sort=False):
  g=g.reset_index(drop=True);top=set(g.nlargest(50,"ig_absolute").index);members=np.flatnonzero(g.canonical_member.to_numpy());observed=len(top&set(members));null=[]
  for _ in range(1000):null.append(len(top&set(rng.choice(len(g),size=len(members),replace=False))))
  controls.append(dict(zip(keys,key))|{"top50_canonical_overlap":observed,"matched_random_mean_overlap":np.mean(null),"matched_random_empirical_p":(1+sum(x>=observed for x in null))/(len(null)+1),"random_replicates":len(null)})
  # Rank-based quartiles remain defined when decoder probabilities saturate.
  n=max(1,len(g)//4);dq=np.zeros(len(g),bool);iq=np.zeros(len(g),bool);dq[np.argsort(-g.decoder_score.to_numpy())[:n]]=True;iq[np.argsort(-g.ig_absolute.to_numpy())[:n]]=True
  label=np.select([dq&iq,dq&~iq,~dq&iq],["high decoder / high IG","high decoder / low IG","low decoder / high IG"],default="low decoder / low IG")
  q=g.assign(category=label).groupby("category").agg(genes=("gene","size"),canonical_fraction=("canonical_member","mean"),leading_edge_fraction=("gsea_leading_edge","mean"),median_abs_de=("de_statistic",lambda x:np.nanmedian(abs(x)))).reset_index()
  for row in q.to_dict("records"):quadrants.append(dict(zip(keys,key))|row)
 pd.DataFrame(controls).to_csv(base/"ig_matched_random_controls.csv",index=False);pd.DataFrame(quadrants).to_csv(base/"decoder_ig_quadrant_summary.csv",index=False)
if __name__=="__main__":main()
