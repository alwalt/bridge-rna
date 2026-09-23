#!/usr/bin/env python3
"""Encode observed GSE264130 samples only; never reads benchmark predictions."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np,pandas as pd,torch
HERE=Path(__file__).resolve().parents[1];WORK=HERE/'work';WORK.mkdir(exist_ok=True)
SRC=Path('/home/walt/bridge-rna.worktrees/attachment-pasted-text-1-30bae6d6/benchmarks/drug_perturbation_prediction')
sys.path.insert(0,'/home/walt/bridge-rna')
from src.fm_embed.model import load_expression_performer
def main():
 p=argparse.ArgumentParser();p.add_argument('--shard',type=int,required=True);p.add_argument('--shards',type=int,default=2);p.add_argument('--device',required=True);p.add_argument('--limit',type=int);a=p.parse_args()
 genes=pd.read_parquet(SRC/'results/perturbation_vectors/gene_order.parquet');canon=pd.read_csv('/home/walt/bridge-rna/data/ensembl/canonical_genes.csv')
 X=np.load(SRC/'work/prepared/sample_log1p_tpm.npy',mmap_mode='r');idx=np.arange(a.shard,len(X),a.shards);idx=idx[:a.limit] if a.limit else idx
 pos=genes.token_id.astype(int).to_numpy()-1;assert np.array_equal(canon.iloc[pos].gene_symbol.to_numpy(),genes.bridge_symbol.to_numpy())
 model,device=load_expression_performer(Path('/home/walt/bridge-rna/model/r7hnr92k/best_model.pt'),Path('/home/walt/bridge-rna/model/r7hnr92k/config.json'),len(canon),a.device)
 out=np.empty((len(idx),512),np.float32);start=time.time()
 # Match benchmark #6's validated encode_matrix behavior exactly: float32,
 # model.encode(normalize=False), batch size one, no autocast.
 with torch.no_grad():
  for k in range(len(idx)):
   rows=idx[k:k+1];aligned=np.zeros((1,len(canon)),np.float32);aligned[:,pos]=np.asarray(X[rows])
   values=torch.as_tensor(aligned,device=device);out[k]=model.encode(values,normalize=False).float().cpu().numpy()[0]
   if k%40==0:print(f'shard={a.shard} {k+1}/{len(idx)} elapsed_min={(time.time()-start)/60:.1f}',flush=True)
 np.save(WORK/f'observed_sample_embeddings_shard{a.shard}.npy',out);np.save(WORK/f'observed_sample_indices_shard{a.shard}.npy',idx)
 print(json.dumps({'shard':a.shard,'samples':len(idx),'device':a.device,'minutes':(time.time()-start)/60}),flush=True)
if __name__=='__main__':main()
