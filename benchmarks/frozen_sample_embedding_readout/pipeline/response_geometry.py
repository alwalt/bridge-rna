#!/usr/bin/env python3
"""Evaluate final frozen readouts on established Task 3 response contrasts."""
from pathlib import Path
import sys
import numpy as np,pandas as pd,torch
from scipy.stats import spearmanr
from run_benchmark import HERE,REPO,T3,OUT,WORK,G,SPECS,QueryPool,hidden,cos

def vectors(A):
 m=pd.read_csv(T3/'results/sample_manifest.csv');mem=pd.read_csv(T3/'results/task3b_contrast_sample_membership.csv');ix=dict(zip(m.sample_id,range(len(m))));out={}
 for name,(cid,strict) in SPECS.items():
  q=mem[mem.contrast_id.eq(cid)].copy()
  if name=='RR1_original':q=q[~q.sample_id.str.endswith('_M27')]
  if name=='RR1_remeasurement':q=q[~q.sample_id.str.endswith('_M29')]
  if strict:q=q[~q.sample_id.str.endswith('_F5')]
  ids={c:q[q.condition.eq(c)].sample_id.map(ix).to_numpy(int) for c in ['FLT','GC']};out[name]=A[ids['FLT']].mean(0)-A[ids['GC']].mean(0)
 return out
def scores(label,A,fold=None):
 v=vectors(A);pairs={'RR1':('RR1_original','RR1_remeasurement'),'RR3_39':('RR3_39_original','RR3_39_remeasurement'),'RR3_40':('RR3_40_original','RR3_40_remeasurement'),'false_friend':('RR1_original','RR3_39_original')};row={'readout':label,'fold':fold}
 for n,(a,b) in pairs.items():row[n+'_cosine']=cos(v[a],v[b]);row[n+'_spearman']=spearmanr(v[a],v[b]).statistic
 return row
def main():
 dev=torch.device('cuda:0');sys.path.insert(0,str(REPO/'benchmarks/tcga_imputation/pipeline'));from model_adapters import load_ours
 model=load_ours(dev).eval();[p.requires_grad_(False) for p in model.parameters()];x=np.load(T3/'work/bridgerna_log1p_tpm_inputs.npy',mmap_mode='r');outs={'layer11_mean_plus_std':[],'layer12_mean_plus_std':[]};att={(fold,layer,kind):[] for fold in range(5) for layer in [11,12] for kind in ['single_query_attention','multihead_attention']}
 mods={}
 for key in att:
  fold,layer,kind=key;mod=QueryPool(1 if kind.startswith('single') else 4).attach(14).to(dev);mod.load_state_dict(torch.load(OUT/f'workers/fold_{fold}_layer{layer}_{kind}.pt',map_location=dev,weights_only=True));mod.eval();mods[key]=mod
 with torch.no_grad():
  for i in range(len(x)):
   z=torch.as_tensor(np.array(x[i:i+1],copy=True),device=dev)
   with torch.autocast('cuda',dtype=torch.float16):h11,h12=hidden(model,z)
   for layer,h in [(11,h11),(12,h12)]:
    outs[f'layer{layer}_mean_plus_std'].append(torch.cat([h.float().mean(1),h.float().std(1,unbiased=False)],1).cpu().numpy()[0])
    for fold in range(5):
     for kind in ['single_query_attention','multihead_attention']:att[(fold,layer,kind)].append(mods[(fold,layer,kind)].pool(h).cpu().numpy()[0])
 rows=[];base=np.load(T3/'work/bridgerna_embeddings.npy');rows.append(scores('current_layer12_mean',base))
 for n,a in outs.items():rows.append(scores(n,np.asarray(a)))
 for (fold,layer,kind),a in att.items():rows.append(scores(f'layer{layer}_{kind}',np.asarray(a),fold))
 d=pd.DataFrame(rows);d.to_csv(OUT/'response_geometry_fold_metrics.csv',index=False);summary=d.groupby('readout',dropna=False).agg(**{c:(c,'mean') for c in d.columns if c not in ['readout','fold']}).reset_index();summary.to_csv(OUT/'response_geometry_summary.csv',index=False);print(summary.to_string(index=False))
if __name__=='__main__':main()
