#!/usr/bin/env python3
"""Build exact frozen layer-12 contextual-gene cosine Top-20 graphs."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from numpy.lib.format import open_memmap

HERE=Path(__file__).resolve().parents[1];REPO=HERE.parents[1];OUT=HERE/'results/graph_fingerprint';WORK=HERE/'work/graph_fingerprint';G=15165;K=20
OLD=REPO/'benchmarks/library_prep_disentanglement';T2=REPO/'benchmarks/cross_species_exercise_response';T3=REPO/'benchmarks/osdr_batch_effect_representation';PREV=OLD/'results/task4_pca_vs_bridgerna_generalization';HWORK=T2/'work/hallmark_readout'
def say(x):print(f'[{time.strftime("%F %T")}] {x}',flush=True)
def source(name):
 if name=='tissue':
  m=pd.read_parquet(PREV/'sample_manifest.parquet').sort_values('matrix_row').reset_index(drop=True);x=np.memmap(HWORK/'archs4_log1p_tpm.float32.mmap',mode='r',dtype='float32',shape=(40000,G));return m,x,m.matrix_row.to_numpy(int)
 if name=='tcell':
  p=OLD/'work/datasets/chen_2020_tcells';m=pd.read_parquet(p/'manifest.parquet');x=np.load(p/'log1p_tpm.npy',mmap_mode='r');return m,x,np.arange(len(m))
 if name=='exercise':
  m=pd.read_parquet(T2/'results/matched_manifest.parquet');members=pd.read_parquet(T2/'results/contrast_members.parquet');keep=m.GSM.astype(str).isin(members.GSM.astype(str));m=m[keep].copy();m['source_row']=m.index;x=np.load(T2/'work/matched_log1p_tpm_corrected.npy',mmap_mode='r');return m.reset_index(drop=True),x,m.source_row.to_numpy(int)
 if name=='task3':
  m=pd.read_csv(T3/'results/sample_manifest.csv');x=np.load(T3/'work/bridgerna_log1p_tpm_inputs.npy',mmap_mode='r');return m,x,np.arange(len(m))
 raise KeyError(name)
def main():
 p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=['tissue','tcell','exercise','task3']);p.add_argument('--device',default='cuda:0');p.add_argument('--query-chunk',type=int,default=512);a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True)
 sys.path.insert(0,str(REPO/'benchmarks/tcga_imputation/pipeline'));from model_adapters import load_ours
 dev=torch.device(a.device);model=load_ours(dev).eval();[q.requires_grad_(False) for q in model.parameters()]
 rate=.94;counts={n:len(source(n)[0]) for n in a.datasets};say(f'estimated graph construction runtime={sum(counts.values())*rate/60:.1f} minutes for {sum(counts.values()):,} samples')
 for name in a.datasets:
  m,x,rows=source(name);d=WORK/name;d.mkdir(exist_ok=True);complete=d/'complete.json';npth=d/'neighbors_top20.uint16.npy';wpth=d/'weights_top20.float16.npy'
  if complete.exists():say(f'reuse {name}');continue
  m.to_parquet(d/'manifest.parquet',index=False);nbr=open_memmap(npth,mode='w+',dtype='uint16',shape=(len(m),G,K));wei=open_memmap(wpth,mode='w+',dtype='float16',shape=(len(m),G,K));started=time.time()
  for si,row in enumerate(rows):
   v=torch.as_tensor(np.array(x[row:row+1],copy=True),device=dev)
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16,enabled=dev.type=='cuda'):h=model._encode_hidden(v)
   h=torch.nn.functional.normalize(h[0].float(),dim=1);ni=[];nw=[]
   with torch.no_grad():
    for q0 in range(0,G,a.query_chunk):
     q1=min(q0+a.query_chunk,G);s=h[q0:q1]@h.T;r=torch.arange(q0,q1,device=dev);s[torch.arange(q1-q0,device=dev),r]=-2;z=torch.topk(s,K,dim=1);ni.append(z.indices.cpu().numpy().astype('uint16'));nw.append(z.values.cpu().numpy().astype('float16'))
   nbr[si]=np.concatenate(ni);wei[si]=np.concatenate(nw)
   if (si+1)%25==0 or si+1==len(m):
    elapsed=time.time()-started;rate=(si+1)/elapsed;remain=(len(m)-si-1)/rate;say(f'{name} {si+1}/{len(m)} elapsed={elapsed/60:.1f}m eta={remain/60:.1f}m rate={rate:.2f}/s')
  nbr.flush();wei.flush();complete.write_text(json.dumps({'dataset':name,'samples':len(m),'genes':G,'layer':12,'metric':'cosine','directed_top_k':20,'self_edges':False,'checkpoint':'r7hnr92k','input':'log1p_tpm'},indent=2)+'\n')
 say('all graph caches complete')
if __name__=='__main__':main()
