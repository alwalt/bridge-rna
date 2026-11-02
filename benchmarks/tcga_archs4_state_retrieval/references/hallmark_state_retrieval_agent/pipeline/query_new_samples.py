#!/usr/bin/env python3
"""Query new natural-log1p(TPM) samples against frozen ARCHS4 indexes."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from common import CONFIG, RESULTS, WORK, apply_probe, l2_rows

def main():
    p=argparse.ArgumentParser(); p.add_argument("--expression",type=Path,required=True,help=".npy [samples,15165], natural log1p(TPM), exact frozen gene order"); p.add_argument("--sample-ids",type=Path); p.add_argument("--output",type=Path,required=True); p.add_argument("--device",default="cuda:0"); p.add_argument("--batch-size",type=int,default=4); p.add_argument("--top-k",type=int,default=10); a=p.parse_args()
    x=np.load(a.expression,mmap_mode="r"); assert x.ndim==2 and x.shape[1]==15165 and np.isfinite(x).all() and x.min()>=0
    ids=[z.strip() for z in a.sample_ids.read_text().splitlines()] if a.sample_ids else [f"sample_{i}" for i in range(len(x))]; assert len(ids)==len(x) and len(set(ids))==len(ids)
    repo=Path(__file__).resolve().parents[3]; sys.path.insert(0,str(repo/"src")); from fm_embed.model import load_expression_performer
    model,device=load_expression_performer(CONFIG["checkpoint"],RESULTS/"frozen/bridge_model_config.json",15165,a.device); chunks=[]
    for left in range(0,len(x),a.batch_size):
        xb=torch.as_tensor(np.array(x[left:left+a.batch_size],copy=True),device=device)
        with torch.inference_mode(),torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"): chunks.append(model.encode(xb,normalize=False).float().cpu().numpy())
    z=np.concatenate(chunks); probe=np.load(RESULTS/"frozen/frozen_gtex_hallmark_probe.npz",allow_pickle=False); h=apply_probe(z,probe)
    meta=pd.read_parquet(RESULTS/"archs4/sample_metadata.parquet"); hi=np.load(WORK/"archs4_hallmark_cosine_index.float32.npy",mmap_mode="r"); bi=np.load(WORK/"archs4_bridge_cosine_index.float32.npy",mmap_mode="r"); rows=[]
    for method,q,index in [("Hallmark-50 cosine",l2_rows(h),hi),("Bridge-512 cosine",l2_rows(z),bi)]:
      for qi,v in enumerate(q):
        score=np.asarray(index@v); take=np.argpartition(score,-a.top_k)[-a.top_k:]; take=take[np.argsort(score[take])[::-1]]
        for rank,ri in enumerate(take,1): rows.append({"query_id":ids[qi],"method":method,"rank":rank,"archs4_sample_id":meta.iloc[ri].sample_id,"series_id":meta.iloc[ri].series_id,"similarity":float(score[ri]),"title":meta.iloc[ri].title,"source_name":meta.iloc[ri].source_name_ch1,"characteristics":meta.iloc[ri].characteristics_ch1})
    a.output.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(a.output,index=False)
if __name__=="__main__": main()
