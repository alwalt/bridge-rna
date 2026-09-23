#!/usr/bin/env python3
"""Verify current frozen inference reproduces benchmark #6 control embeddings."""
from pathlib import Path
import json,sys
import numpy as np,pandas as pd,torch
HERE=Path(__file__).resolve().parents[1];OUT=HERE/'results';SRC=Path('/home/walt/bridge-rna.worktrees/attachment-pasted-text-1-30bae6d6/benchmarks/drug_perturbation_prediction')
sys.path.insert(0,'/home/walt/bridge-rna');from src.fm_embed.model import load_expression_performer
def main():
 genes=pd.read_parquet(SRC/'results/perturbation_vectors/gene_order.parquet');canon=pd.read_csv('/home/walt/bridge-rna/data/ensembl/canonical_genes.csv');x=np.load(SRC/'work/prepared/control_log1p_tpm.npy');expected=np.load(SRC/'work/prepared/bridge_control_embeddings.npy');pos=genes.token_id.astype(int).to_numpy()-1
 aligned=np.zeros((len(x),len(canon)),np.float32);aligned[:,pos]=x
 model,device=load_expression_performer(Path('/home/walt/bridge-rna/model/r7hnr92k/best_model.pt'),Path('/home/walt/bridge-rna/model/r7hnr92k/config.json'),len(canon),'cuda:0')
 values=[]
 with torch.no_grad():
  for row in aligned:values.append(model.encode(torch.as_tensor(row[None,:],device=device),normalize=False).float().cpu().numpy()[0])
 actual=np.asarray(values)
 diff=np.abs(actual-expected);r={'controls':4,'dimensions':512,'max_abs_difference':float(diff.max()),'mean_abs_difference':float(diff.mean()),'allclose_rtol_1e-5_atol_1e-5':bool(np.allclose(actual,expected,rtol=1e-5,atol=1e-5))}
 (OUT/'encoder_equivalence.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));assert r['allclose_rtol_1e-5_atol_1e-5']
if __name__=='__main__':main()
