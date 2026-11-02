#!/usr/bin/env python3
"""Frozen, versioned Bridge sample→Hallmark inference interface.

Input must already be natural log1p(TPM), sample×15,165, in the bundled
canonical-gene order. Outputs are pathway *scores*, never GSEA enrichment.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
from common import CONFIG,REPO,RESULTS,WORK,resolve,read_vocab,sha256,save_json
sys.path.insert(0,str(REPO/"src"))
from fm_embed.model import load_expression_performer

def main():
 p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--sample-ids",type=Path);p.add_argument("--device",default="cuda:0");p.add_argument("--batch-size",type=int,default=4);p.add_argument("--max-samples",type=int);a=p.parse_args()
 X=np.load(a.input,mmap_mode="r");
 if X.ndim!=2 or X.shape[1]!=15165 or not np.isfinite(X).all():raise ValueError(f"Expected finite [N,15165], got {X.shape}")
 if a.max_samples is not None:X=X[:a.max_samples]
 ids=([x.strip() for x in a.sample_ids.read_text().splitlines()][:len(X)] if a.sample_ids else [f"sample_{i}" for i in range(len(X))])
 if len(ids)!=len(X) or len(set(ids))!=len(ids):raise ValueError("sample IDs must be unique and aligned")
 probe_path=RESULTS/"frozen_gtex_hallmark_probe.npz";probe=np.load(probe_path);model,device=load_expression_performer(resolve(CONFIG["checkpoint"]),resolve(CONFIG["model_config"]),15165,a.device);out=[]
 for left in range(0,len(X),a.batch_size):
  xb=torch.as_tensor(np.array(X[left:left+a.batch_size],copy=True),device=device)
  with torch.inference_mode(),torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"):z=model.encode(xb,normalize=False).float().cpu().numpy()
  out.append(((z-probe["mean"])/probe["scale"])@probe["weight"].T+probe["bias"])
 frame=pd.DataFrame(np.concatenate(out),columns=probe["hallmarks"].astype(str));frame.insert(0,"sample_id",ids);a.output.parent.mkdir(parents=True,exist_ok=True);frame.to_parquet(a.output,index=False)
 save_json(a.output.with_suffix(".provenance.json"),{"input":str(a.input.resolve()),"input_sha256":sha256(a.input),"output":str(a.output.resolve()),"checkpoint":str(resolve(CONFIG["checkpoint"])),"checkpoint_sha256":sha256(resolve(CONFIG["checkpoint"])),"probe_sha256":sha256(probe_path),"normalization":"natural log1p(TPM)","canonical_genes_file":str(RESULTS/"frozen_canonical_genes.csv"),"pooling":"final-layer arithmetic mean","bridge_frozen":True})
if __name__=="__main__":main()
