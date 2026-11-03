#!/usr/bin/env python3
"""Recompute complete IG endpoint checks, including parallel-worker caches."""
import argparse
import numpy as np,pandas as pd,torch
import run_bridge as rb

def main():
 p=argparse.ArgumentParser(); p.add_argument("--device",default="cuda:0"); a=p.parse_args(); model,device,_=rb.load_model(a.device)
 members=pd.read_parquet(rb.HERE/"results/manifests/contrast_members.parquet"); s=pd.read_csv(rb.HERE/"results/manifests/GSE211204_unseen_subject_sensitivity.csv");s["contrast_id"]="GSE211204_ULLS_unseen";members=pd.concat([members,s],ignore_index=True,sort=False)
 rows=[]
 baseline=torch.zeros((1,15165),device=device)
 with torch.no_grad(): baseline_embedding=model._encode_hidden(baseline).mean(1)[0]
 for cid,cm in members.groupby("contrast_id",sort=True):
  gse=cm.dataset.iloc[0]; mapping=pd.read_csv(rb.PREP/f"{gse}_matrix_rows.csv").set_index("sample_id").matrix_row.to_dict(); x=np.load(rb.PREP/f"{gse}_log1p_tpm.npy",mmap_mode="r"); attrs=np.load(rb.CACHE/f"{cid}_sample_ig.npy"); directions=np.load(rb.CACHE/f"{cid}_sample_directions.npy")
  cm=cm.reset_index(drop=True)
  for start in range(0,len(cm),2):
   batch=cm.iloc[start:start+2]; target=torch.as_tensor(directions[start:start+len(batch)],device=device); observed=torch.as_tensor(np.stack([np.asarray(x[mapping[s]]) for s in batch.sample_id]).copy(),device=device)
   with torch.no_grad(): embedding=model._encode_hidden(observed).mean(1); deltas=((embedding-baseline_embedding)*target).sum(1).cpu().numpy()
   for offset,(_,r) in enumerate(batch.iterrows()):
    pos=start+offset; delta=float(deltas[offset]); error=float(attrs[pos].sum()-delta)
    rows.append({"contrast_id":cid,"sample_id":r.sample_id,"score_difference":delta,"attribution_sum":float(attrs[pos].sum()),"completeness_delta":error,"absolute_completeness_delta":abs(error)})
 pd.DataFrame(rows).to_csv(rb.OUT/"ig_completeness.csv",index=False); rb.say("attribution completeness validation complete")
if __name__=="__main__":main()
