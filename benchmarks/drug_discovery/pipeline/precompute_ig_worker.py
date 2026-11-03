#!/usr/bin/env python3
"""Parallel cache worker for Bridge sample-level Integrated Gradients."""
import argparse,re,time
from pathlib import Path
import numpy as np,pandas as pd
import run_bridge as rb

def main():
 p=argparse.ArgumentParser(); p.add_argument("--device",required=True); p.add_argument("--regex",required=True); p.add_argument("--ig-steps",type=int,default=rb.CONFIG["ig_steps"]); a=p.parse_args()
 rb.CACHE.mkdir(parents=True,exist_ok=True); model,device,_=rb.load_model(a.device)
 samples=pd.read_parquet(rb.HERE/"results/manifests/sample_manifest.parquet"); members=pd.read_parquet(rb.HERE/"results/manifests/contrast_members.parquet")
 sensitivity=pd.read_csv(rb.HERE/"results/manifests/GSE211204_unseen_subject_sensitivity.csv"); sensitivity["contrast_id"]="GSE211204_ULLS_unseen"; members=pd.concat([members,sensitivity],ignore_index=True,sort=False)
 selected=[(c,g.copy()) for c,g in members.groupby("contrast_id",sort=True) if re.search(a.regex,c)]; total=sum(len(g) for _,g in selected); done=0; started=time.monotonic()
 for cid,cm in selected:
  cache=rb.CACHE/f"{cid}_sample_ig.npy"
  if cache.exists(): rb.say(f"worker reuse {cid}"); continue
  gse=cm.dataset.iloc[0]; x=np.load(rb.PREP/f"{gse}_log1p_tpm.npy",mmap_mode="r"); mapping=pd.read_csv(rb.PREP/f"{gse}_matrix_rows.csv").set_index("sample_id").matrix_row.to_dict(); emb=np.load(rb.CACHE/f"{gse}_embeddings.npy")
  cm["matrix_row"]=cm.sample_id.map(mapping); cm=cm.dropna(subset=["matrix_row"]).reset_index(drop=True); cm["matrix_row"]=cm.matrix_row.astype(int)
  _,directions=rb.attribution_directions(cid,np.stack([emb[r] for r in cm.matrix_row]),cm); np.save(rb.CACHE/f"{cid}_sample_directions.npy",directions); attrs=[]
  for pos,row in cm.iterrows():
   value,_,_=rb.ig(model,np.asarray(x[row.matrix_row]),directions[pos],device,a.ig_steps); attrs.append(value); done+=1; elapsed=time.monotonic()-started; rb.say(f"worker {cid} {done}/{total} elapsed={elapsed/60:.1f}m eta={(total-done)/(done/max(elapsed,1e-9))/60:.1f}m")
  np.save(cache,np.stack(attrs)); rb.say(f"worker saved {cache.name}")
if __name__=="__main__":main()
