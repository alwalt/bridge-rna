#!/usr/bin/env python3
"""Parts III-VI: from-scratch held-out-gene decoder and streamed pathway profiles.

The interface deliberately stores only selected training contexts, 50-D profiles,
and per-sample top-gene attribution rows; full [sample,gene,512] tensors are never
materialized for a cohort.
"""
from __future__ import annotations
import argparse, json, sys, time
import numpy as np, pandas as pd, torch
from scipy.special import expit
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from common import CONFIG, REPO, RESULTS, WORK, resolve, read_vocab, read_gmt, save_json

sys.path.insert(0,str(REPO/"src"))
from fm_embed.model import load_expression_performer

def labels():
    genes,sets=read_vocab(),read_gmt(); Y=np.zeros((len(genes),len(sets)),np.uint8)
    for k,members in enumerate(sets.values()): Y[:,k]=np.isin(genes,list(members))
    return genes,list(sets),Y

def split_genes(Y):
    """Iterative-ish deterministic split retaining positives for every pathway."""
    rng=np.random.default_rng(CONFIG["seed"]); score=Y.sum(1); test=[]
    for burden in sorted(np.unique(score),reverse=True):
        ids=np.flatnonzero(score==burden); rng.shuffle(ids); test.extend(ids[:round(len(ids)*CONFIG["gene_test_fraction"])])
    test=np.array(sorted(set(test))); train=np.setdiff1d(np.arange(len(Y)),test)
    if np.any(Y[test].sum(0)==0) or np.any(Y[train].sum(0)==0): raise AssertionError("Gene split loses pathway positives")
    return train,test

def extract_hidden(model,x,device):
    with torch.inference_mode(),torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"):
        return model._encode_hidden(torch.as_tensor(x[None],dtype=torch.float32,device=device))[0].float().cpu().numpy()

def extract(args):
    X=np.load(args.expression,mmap_mode="r"); chosen=np.arange(min(args.samples,len(X)))
    model,device=load_expression_performer(resolve(CONFIG["checkpoint"]),resolve(CONFIG["model_config"]),15165,args.device)
    out=WORK/f"{args.dataset}_decoder_contexts.float16.npy"; mm=np.lib.format.open_memmap(out,mode="w+",dtype=np.float16,shape=(len(chosen),15165,512))
    start=time.time()
    for i,row in enumerate(chosen):
        mm[i]=extract_hidden(model,np.asarray(X[row]),device).astype(np.float16); mm.flush()
        elapsed=time.time()-start; rate=(i+1)/elapsed
        print(f"[contexts] {i+1}/{len(chosen)} elapsed={elapsed:.1f}s rate={rate:.3f}/s eta={(len(chosen)-i-1)/max(rate,1e-9):.1f}s",flush=True)
    save_json(WORK/f"{args.dataset}_decoder_contexts.json",{"sample_rows":chosen.tolist(),"shape":list(mm.shape),"dtype":"float16","final_layer":12,"checkpoint":CONFIG["checkpoint"]})

def train(args):
    genes,names,Y=labels(); train_ids,test_ids=split_genes(Y); contexts=np.load(args.contexts,mmap_mode="r")
    # Average training contexts is intentional: membership is gene-level; sample-specific states are evaluated downstream.
    F=np.asarray(contexts,dtype=np.float32).mean(0); mu=F[train_ids].mean(0); sd=np.maximum(F[train_ids].std(0),1e-5); X=(F-mu)/sd
    W=np.zeros((50,512),np.float32); b=np.zeros(50,np.float32); rows=[]
    for k,name in enumerate(names):
        clf=SGDClassifier(loss="log_loss",penalty="l2",alpha=1e-4,class_weight="balanced",max_iter=5000,tol=1e-5,random_state=CONFIG["seed"]+k)
        clf.fit(X[train_ids],Y[train_ids,k]); W[k]=clf.coef_[0]; b[k]=clf.intercept_[0]; prob=clf.predict_proba(X[test_ids])[:,1]; y=Y[test_ids,k]
        rows.append({"hallmark":name,"auroc":roc_auc_score(y,prob),"auprc":average_precision_score(y,prob),"prevalence":y.mean(),"random_auprc":y.mean(),"train_positive":int(Y[train_ids,k].sum()),"test_positive":int(y.sum())})
    metrics=pd.DataFrame(rows); metrics.to_csv(RESULTS/"heldout_gene_decoder_metrics.csv",index=False)
    np.savez(WORK/"hallmark_linear_decoder.npz",weight=W,bias=b,mean=mu,scale=sd,train_gene_ids=train_ids,test_gene_ids=test_ids,hallmarks=np.array(names),genes=np.array(genes))
    save_json(RESULTS/"decoder_summary.json",{"macro_auroc":metrics.auroc.mean(),"macro_auprc":metrics.auprc.mean(),"macro_prevalence":metrics.prevalence.mean(),"contexts":len(contexts),"bridge_frozen":True,"evaluation":"held-out genes"})

def score(args):
    pack=np.load(args.decoder); W,b,mu,sd=pack["weight"],pack["bias"],pack["mean"],pack["scale"]; genes,names,Y=labels(); X=np.load(args.expression,mmap_mode="r")
    meta=pd.read_parquet(args.metadata); model,device=load_expression_performer(resolve(CONFIG["checkpoint"]),resolve(CONFIG["model_config"]),15165,args.device)
    mean_scores=np.empty((len(X),50),np.float32); top_scores=np.empty_like(mean_scores); attrib=[]; topk=CONFIG["top_k_member_genes"]
    start=time.time()
    for i in range(len(X)):
        hidden=extract_hidden(model,np.asarray(X[i]),device); prob=expit(((hidden-mu)/sd)@W.T+b); 
        for k,name in enumerate(names):
            ids=np.flatnonzero(Y[:,k]); vals=prob[ids,k]; order=np.argsort(-vals)[:topk]; mean_scores[i,k]=vals.mean(); top_scores[i,k]=vals[order].mean()
            for rank,o in enumerate(order[:5],1): attrib.append({"sample_id":meta.iloc[i].sample_id,"hallmark":name,"rank":rank,"gene":genes[ids[o]],"probability":vals[o]})
        elapsed=time.time()-start; print(f"[profiles] {i+1}/{len(X)} elapsed={elapsed:.1f}s eta={(elapsed/(i+1))*(len(X)-i-1):.1f}s",flush=True)
    for method,array in [("mean_member",mean_scores),("topk_member",top_scores)]:
        pd.DataFrame(array,columns=names).assign(sample_id=meta.sample_id.values).to_parquet(RESULTS/f"{args.dataset}_bridge_hallmark_{method}.parquet",index=False)
    pd.DataFrame(attrib).to_parquet(RESULTS/f"{args.dataset}_top_gene_attributions.parquet",index=False)

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="command",required=True)
    e=sub.add_parser("extract"); e.add_argument("--dataset",required=True); e.add_argument("--expression",type=str,required=True); e.add_argument("--samples",type=int,default=32); e.add_argument("--device",default="cuda:0")
    t=sub.add_parser("train"); t.add_argument("--contexts",type=str,required=True)
    s=sub.add_parser("score"); s.add_argument("--dataset",required=True); s.add_argument("--expression",required=True); s.add_argument("--metadata",required=True); s.add_argument("--decoder",default=str(WORK/"hallmark_linear_decoder.npz")); s.add_argument("--device",default="cuda:0")
    a=p.parse_args(); {"extract":extract,"train":train,"score":score}[a.command](a)
if __name__=="__main__": main()

