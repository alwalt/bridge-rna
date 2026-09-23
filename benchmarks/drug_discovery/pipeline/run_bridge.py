#!/usr/bin/env python3
"""Frozen Bridge embeddings, condition vectors, and signed IG rankings."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parents[1]; WORK=HERE/"work"; PREP=WORK/"prepared"; CACHE=WORK/"bridge"; OUT=HERE/"results"/"bridge"
CONFIG=json.loads((HERE/"config.json").read_text()); ROOT=Path("/home/walt/bridge-rna")
sys.path.insert(0,str(ROOT))
from src.fm_embed.model import load_expression_performer

def say(x): print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {x}",flush=True)
def unit(x): return x/np.maximum(np.linalg.norm(x),1e-12)

def load_model(device):
    genes=pd.read_csv(CONFIG["canonical_genes"]).gene_symbol
    model,resolved=load_expression_performer(Path(CONFIG["checkpoint"]),Path(CONFIG["checkpoint_config"]),len(genes),device)
    for p in model.parameters(): p.requires_grad_(False)
    return model,resolved,genes.tolist()

def encode(model,x,device,batch=2):
    out=[]
    with torch.no_grad(),torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"):
        for start in range(0,len(x),batch):
            values=torch.as_tensor(np.asarray(x[start:start+batch]),dtype=torch.float32,device=device)
            out.append(model._encode_hidden(values).mean(1).float().cpu().numpy())
    return np.concatenate(out)

def score(model,x,direction): return (model._encode_hidden(x).mean(1)*direction).sum(1)

def ig(model,values,direction,device,steps=16,path_batch=4):
    baseline=torch.zeros((1,len(values)),device=device); observed=torch.as_tensor(np.array(values,copy=True),device=device).view(1,-1)
    target=torch.as_tensor(direction,device=device).view(1,-1); total=torch.zeros_like(baseline)
    alphas=(np.arange(steps,dtype=np.float32)+.5)/steps
    for start in range(0,steps,path_batch):
        alpha=torch.as_tensor(alphas[start:start+path_batch],device=device).view(-1,1)
        x=(baseline+alpha*(observed-baseline)).requires_grad_(True)
        gradient=torch.autograd.grad(score(model,x,target.expand(len(x),-1)).sum(),x)[0]
        total += gradient.detach().sum(0,keepdim=True)
    attr=((observed-baseline)*total/steps)[0].detach().cpu().numpy().astype(np.float32)
    with torch.no_grad():
        endpoint=float(score(model,observed,target).item()); zero=float(score(model,baseline,target).item())
    return attr,endpoint-zero,float(attr.sum()-(endpoint-zero))

def block_columns(cid):
    if cid.startswith("GSE297090_"): return ["donor","state"]
    if cid=="GSE184119_10Gy": return ["cell_type"]
    if cid in {"GSE211204_ULLS","GSE113165_bedrest","GSE211204_ULLS_unseen"}: return ["donor"]
    return []

def holdout_columns(cid):
    if cid.startswith("GSE297090_"): return ["donor"]
    return block_columns(cid)

def contributions(values,members,columns):
    if not columns:
        return np.stack([values[members.role.eq("case")].mean(0)-values[members.role.eq("control")].mean(0)]),["all"]
    arrays=[]; names=[]
    for key,g in members.groupby(columns,sort=True,dropna=False):
        if not g.role.eq("case").any() or not g.role.eq("control").any(): continue
        arrays.append(values[g.index[g.role.eq("case")]].mean(0)-values[g.index[g.role.eq("control")]].mean(0)); names.append(str(key))
    return np.stack(arrays),names

def contrast_contributions(cid,values,members):
    """Return block-wise effects, or the frozen covariate-adjusted coefficient."""
    if cid=="GSE276529_GIOP":
        # Match the DE design: intercept + centered age + case indicator.
        age=pd.to_numeric(members.age_numeric).to_numpy(float)
        case=members.role.eq("case").to_numpy(float)
        design=np.column_stack([np.ones(len(members)),age-age.mean(),case])
        coefficient=np.linalg.lstsq(design,np.asarray(values),rcond=None)[0][-1]
        return coefficient[None,:],["age_adjusted_condition"]
    return contributions(values,members,block_columns(cid))

def attribution_directions(cid,values,members):
    """Estimate a direction without the attributed sample's subject/block."""
    values=np.asarray(values); members=members.reset_index(drop=True)
    overall,_=contrast_contributions(cid,values,members)
    overall=unit(overall.mean(0)).astype(np.float32)
    directions=[]; columns=holdout_columns(cid)
    for pos,row in members.iterrows():
        if columns:
            held=np.ones(len(members),dtype=bool)
            for column in columns:
                held &= members[column].astype(str).eq(str(row[column])).to_numpy()
        else:
            held=np.zeros(len(members),dtype=bool); held[pos]=True
        kept=~held; subset=members.loc[kept].reset_index(drop=True)
        # Fall back only when exclusion makes the contrast unidentifiable.
        if not subset.role.eq("case").any() or not subset.role.eq("control").any():
            directions.append(overall); continue
        effect,_=contrast_contributions(cid,values[kept],subset)
        directions.append(unit(effect.mean(0)).astype(np.float32))
    return overall,np.stack(directions)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--device",default="cuda:0"); p.add_argument("--ig-steps",type=int,default=CONFIG["ig_steps"]); args=p.parse_args()
    CACHE.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
    samples=pd.read_parquet(HERE/"results/manifests/sample_manifest.parquet"); members=pd.read_parquet(HERE/"results/manifests/contrast_members.parquet")
    sensitivity=pd.read_csv(HERE/"results/manifests/GSE211204_unseen_subject_sensitivity.csv")
    sensitivity["contrast_id"]="GSE211204_ULLS_unseen"; members=pd.concat([members,sensitivity],ignore_index=True,sort=False)
    model,device,genes=load_model(args.device)

    embedding_rows=[]; expression={}; rowmaps={}; offsets={}
    for gse,g in samples.groupby("dataset",sort=True):
        x=np.load(PREP/f"{gse}_log1p_tpm.npy",mmap_mode="r"); expression[gse]=x
        mapping=pd.read_csv(PREP/f"{gse}_matrix_rows.csv").set_index("sample_id").matrix_row.to_dict(); rowmaps[gse]=mapping
        cache=CACHE/f"{gse}_embeddings.npy"
        if cache.exists(): emb=np.load(cache)
        else: emb=encode(model,x,device); np.save(cache,emb)
        offsets[gse]=emb
        for sid,row in mapping.items(): embedding_rows.append({"dataset":gse,"sample_id":sid,**{f"z{i}":v for i,v in enumerate(emb[row])}})
        say(f"embeddings {gse}: {len(emb)}")
    embeddings=pd.DataFrame(embedding_rows); embeddings.to_parquet(OUT/"sample_embeddings.parquet",index=False)

    vector_rows=[]; gene_rows=[]; complete=[]; block_meta=[]; started=time.monotonic(); done=0; total=sum(len(g) for _,g in members.groupby("contrast_id"))
    for cid,cm0 in members.groupby("contrast_id",sort=True):
        cm=cm0.copy(); gse=cm.dataset.iloc[0]; mapping=rowmaps[gse]; x=expression[gse]; emb=offsets[gse]
        cm["matrix_row"]=cm.sample_id.map(mapping); cm=cm.dropna(subset=["matrix_row"]); cm["matrix_row"]=cm.matrix_row.astype(int)
        emb_frame=np.stack([emb[r] for r in cm.matrix_row]); cm=cm.reset_index(drop=True); direction,sample_directions=attribution_directions(cid,emb_frame,cm)
        np.save(CACHE/f"{cid}_sample_directions.npy",sample_directions)
        for i,v in enumerate(direction): vector_rows.append({"contrast_id":cid,"dataset":gse,"dimension":i,"value":v})
        cache=CACHE/f"{cid}_sample_ig.npy"
        if cache.exists(): attrs=np.load(cache)
        else:
            attrs=[]
            for pos,row in cm.iterrows():
                value,delta,error=ig(model,np.asarray(x[row.matrix_row]),sample_directions[pos],device,args.ig_steps)
                attrs.append(value); complete.append({"contrast_id":cid,"sample_id":row.sample_id,"score_difference":delta,"attribution_sum":float(value.sum()),"completeness_delta":error})
                done+=1; elapsed=time.monotonic()-started; rate=done/max(elapsed,1e-9); say(f"IG {cid} {done}/{total} elapsed={elapsed/60:.1f}m eta={(total-done)/max(rate,1e-9)/60:.1f}m")
            attrs=np.stack(attrs); np.save(cache,attrs)
        attr_contrib,attr_names=contrast_contributions(cid,attrs,cm.reset_index(drop=True)); np.savez_compressed(CACHE/f"{cid}_block_attribution_contributions.npz",values=attr_contrib,names=np.array(attr_names))
        score_values=attr_contrib.mean(0); score_values=score_values/np.maximum(np.abs(score_values).sum(),1e-12)
        for gene,value in zip(genes,score_values): gene_rows.append({"contrast_id":cid,"dataset":gse,"gene":gene,"signed_bridge_score":value,"absolute_bridge_score":abs(value)})
        for name in attr_names: block_meta.append({"contrast_id":cid,"dataset":gse,"block":name})
    vectors=pd.DataFrame(vector_rows); vectors.to_parquet(OUT/"condition_vectors.parquet",index=False)
    contrast_scores=pd.DataFrame(gene_rows); contrast_scores.to_parquet(OUT/"contrast_gene_scores.parquet",index=False)
    if complete: pd.DataFrame(complete).to_csv(OUT/"ig_completeness.csv",index=False)
    pd.DataFrame(block_meta).to_csv(OUT/"attribution_blocks.csv",index=False)

    primary=contrast_scores[~contrast_scores.contrast_id.str.endswith("_unseen")].groupby(["dataset","gene"],as_index=False).signed_bridge_score.mean()
    primary["absolute_bridge_score"]=primary.signed_bridge_score.abs(); primary["rank"]=primary.groupby("dataset").absolute_bridge_score.rank(method="first",ascending=False).astype(int)
    primary.to_parquet(OUT/"study_gene_rankings.parquet",index=False)
    sensitivity_scores=contrast_scores[contrast_scores.contrast_id.eq("GSE211204_ULLS_unseen")].copy(); sensitivity_scores["rank"]=sensitivity_scores.absolute_bridge_score.rank(method="first",ascending=False).astype(int); sensitivity_scores.to_parquet(OUT/"GSE211204_unseen_subject_gene_ranking.parquet",index=False)
    provenance={"checkpoint":CONFIG["checkpoint"],"checkpoint_config":CONFIG["checkpoint_config"],"encoder_frozen":True,"input":"natural log1p(TPM), canonical 15,165 genes","pooling":"mean contextual gene token","condition_direction":"unit mean of within-block case-minus-control embedding contributions; GSE276529 is age-adjusted","attribution_direction":"leave-one-donor/block-out, or leave-one-sample-out for unblocked contrasts; fallback to overall only if exclusion makes a contrast unidentifiable","target":"dot product of pooled embedding and contrast direction","attribution":"signed Integrated Gradients; all-zero baseline; midpoint Riemann","ig_steps":args.ig_steps,"study_consensus":"arithmetic mean of per-contrast signed IG scores after L1 absolute normalization","sensitivity":"GSE211204 subjects with either primary endpoint in ARCHS4 train/validation removed"}
    (OUT/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n"); say("Bridge inference and attribution complete")

if __name__=="__main__": main()
