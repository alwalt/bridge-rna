#!/usr/bin/env python3
"""Integrated-Gradients attribution of fixed BridgeRNA exercise axes."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

HERE=Path(__file__).resolve().parents[1]; ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/"benchmarks/tcga_downstream/pipeline")); sys.path.insert(0,str(ROOT))
from run_attention_pooling import load_frozen_encoder
from src.fm_embed.vocab import load_canonical_genes

OUT=HERE/"results"/"latent_axis_attribution"; WORK=HERE/"work"/"latent_axis_attribution"; FIGURES=OUT/"figures"
AXES={"Axis A":["GSE108643","GSE86931","GSE126962","GSE132520"],"Axis B":["GSE71972","GSE87748","GSE97718"]}
CONTEXT_GENES={"NR4A3","CDKN1A","SIK1","ATF3","ANKRD1","EGR1","OTUD1","CX3CL1"}

def log(x): print(f"[{time.strftime('%H:%M:%S')}] {x}",flush=True)
def unit(x): return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)

def axis_directions(meta:pd.DataFrame,effects:np.ndarray)->dict[str,np.ndarray]:
    lookup={g:i for i,g in enumerate(meta.GSE)}; result={}
    for axis,gses in AXES.items(): result[axis]=unit(unit(effects[[lookup[g] for g in gses]]).mean(0)).astype(np.float32)
    return result

def sample_score(model,x:torch.Tensor,direction:torch.Tensor)->torch.Tensor:
    return (model._encode_hidden(x).mean(1)*direction).sum(1)

def integrated_gradients(model,values:np.ndarray,direction:np.ndarray,device:torch.device,steps:int,path_batch:int=4)->np.ndarray:
    baseline=torch.zeros((1,len(values)),device=device); observed=torch.from_numpy(np.array(values,copy=True)).to(device).unsqueeze(0); target=torch.from_numpy(direction).to(device).unsqueeze(0); total=torch.zeros_like(baseline)
    # Midpoint Riemann integration avoids endpoints and is stable at modest step counts.
    alphas=(np.arange(steps,dtype=np.float32)+.5)/steps
    for start in range(0,steps,path_batch):
        alpha=torch.from_numpy(alphas[start:start+path_batch]).to(device).view(-1,1)
        x=(baseline+alpha*(observed-baseline)).requires_grad_(True)
        score=sample_score(model,x,target.expand(len(x),-1)); gradient=torch.autograd.grad(score.sum(),x)[0]; total+=gradient.detach().sum(0,keepdim=True)
    return ((observed-baseline)*total/steps)[0].cpu().numpy().astype(np.float32)

def study_effect(matrix:np.ndarray,members:pd.DataFrame,manifest:pd.DataFrame)->tuple[pd.DataFrame,np.ndarray]:
    lookup=manifest.reset_index().set_index("GSM")["index"]; rows=[]; effects=[]
    for cid,g in members.groupby("contrast_id",sort=True):
        post=lookup.loc[g.loc[g.role.eq("post_exercise"),"GSM"]].to_numpy(int); pre=lookup.loc[g.loc[g.role.eq("pre_control"),"GSM"]].to_numpy(int)
        effects.append(matrix[post].mean(0)-matrix[pre].mean(0)); f=g.iloc[0]; rows.append({"contrast_id":cid,"species":f.species,"GSE":f.GSE,"post_n":len(post),"pre_n":len(pre)})
    return pd.DataFrame(rows),np.stack(effects)

def score_all(model,expression:np.ndarray,direction:np.ndarray,device:torch.device,mask:np.ndarray|None=None)->np.ndarray:
    output=[]; d=torch.from_numpy(direction).to(device).unsqueeze(0)
    with torch.no_grad():
        for start in range(0,len(expression),4):
            x=torch.from_numpy(np.array(expression[start:start+4],copy=True)).to(device)
            if mask is not None: x[:,mask]=-10.0
            output.append(sample_score(model,x,d).cpu().numpy())
    return np.concatenate(output)

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--ig-steps",type=int,default=16); p.add_argument("--random-panels",type=int,default=10); p.add_argument("--device",default="cuda:0"); p.add_argument("--seed",type=int,default=42); args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True); WORK.mkdir(parents=True,exist_ok=True); FIGURES.mkdir(parents=True,exist_ok=True)
    genes=load_canonical_genes(ROOT/"data/ensembl/canonical_genes.csv"); manifest=pd.read_parquet(HERE/"results/matched_manifest.parquet"); members=pd.read_parquet(HERE/"results/contrast_members.parquet"); meta=pd.read_csv(HERE/"results/response_contrasts.csv")
    expression=np.load(HERE/"work/matched_log1p_tpm_corrected.npy",mmap_mode="r"); embedding_effects=np.load(HERE/"work/response_effects_bridgerna.npy"); directions=axis_directions(meta,embedding_effects)
    np.savez(OUT/"axis_directions.npz",**{k.replace(" ","_"):v for k,v in directions.items()})
    device=torch.device(args.device if torch.cuda.is_available() else "cpu"); model=load_frozen_encoder(device); selected=members.merge(manifest[["GSM"]].reset_index(),on="GSM",validate="many_to_one"); relevant={g:a for a,gs in AXES.items() for g in gs}
    # GSE151066 was previously identified as weak/intermediate and was not a
    # member of either fixed axis. Attribute it to its nearer fixed direction
    # for completeness, but never include it in either consensus.
    meta_index={g:i for i,g in enumerate(meta.GSE)}
    if "GSE151066" in meta_index:
        response=embedding_effects[meta_index["GSE151066"]]
        relevant["GSE151066"]=max(directions,key=lambda axis:abs(float(response@directions[axis])))
    lookup=manifest.reset_index().set_index("GSM")["index"]; role_attr={}; completeness=[]
    role_profiles=[]
    for cid,group in members.groupby("contrast_id",sort=True):
        for role in ["post_exercise","pre_control"]:
            rows=lookup.loc[group.loc[group.role.eq(role),"GSM"]].to_numpy(int); role_profiles.append((cid,group.iloc[0].GSE,role,rows,np.asarray(expression[rows]).mean(0)))
    attr_rows=[]
    for cid,group in members.groupby("contrast_id",sort=True):
        f=group.iloc[0]; attr_rows.append({"contrast_id":cid,"species":f.species,"GSE":f.GSE,"post_n":int((group.role=="post_exercise").sum()),"pre_n":int((group.role=="pre_control").sum())})
    attr_meta=pd.DataFrame(attr_rows); attr_cache=WORK/"study_integrated_gradient_changes.npy"
    if attr_cache.exists() and (OUT/"integrated_gradients_completeness.csv").exists():
        attr_effect=np.load(attr_cache); log("Reusing completed study-level Integrated Gradients")
    else:
        for count,(cid,gse,role,rows,profile) in enumerate(role_profiles,1):
            axis=relevant[gse]; value=integrated_gradients(model,profile,directions[axis],device,args.ig_steps); role_attr[(cid,role)]=value
            endpoint=score_all(model,profile[None],directions[axis],device)[0]; baseline=score_all(model,np.zeros((1,len(genes)),dtype=np.float32),directions[axis],device)[0]
            completeness.append({"contrast_id":cid,"GSE":gse,"role":role,"samples":len(rows),"axis":axis,"score_difference":endpoint-baseline,"attribution_sum":value.sum(),"completeness_delta":value.sum()-(endpoint-baseline)})
            log(f"IG group profiles={count}/{len(role_profiles)} steps={args.ig_steps}")
        attr_effect=np.stack([role_attr[(cid,"post_exercise")]-role_attr[(cid,"pre_control")] for cid in attr_meta.contrast_id])
        np.save(attr_cache,attr_effect); pd.DataFrame(completeness).to_csv(OUT/"integrated_gradients_completeness.csv",index=False)
    de_meta,de_effect=study_effect(np.asarray(expression),members,manifest)
    if not attr_meta.equals(de_meta): raise AssertionError("Attribution/DE study alignment failure")
    study_rows=[]; comparison=[]; gene_lookup={g:i for i,g in enumerate(genes)}
    for i,row in attr_meta.iterrows():
        axis=relevant[row.GSE]; order=np.argsort(-np.abs(attr_effect[i])); de_order=np.argsort(-np.abs(de_effect[i])); top_de=set(de_order[:100]);
        comparison.append({"GSE":row.GSE,"species":row.species,"axis":axis,"absolute_attribution_vs_absolute_de_spearman":spearmanr(np.abs(attr_effect[i]),np.abs(de_effect[i])).statistic,"top100_overlap":len(set(order[:100])&top_de),"top100_jaccard":len(set(order[:100])&top_de)/len(set(order[:100])|top_de),"prior_context_genes_in_top100":len({gene_lookup[g] for g in CONTEXT_GENES}&set(order[:100]))})
        for rank,g in enumerate(order[:250],1): study_rows.append({"GSE":row.GSE,"species":row.species,"axis":axis,"rank":rank,"gene":genes[g],"attribution_change":attr_effect[i,g],"absolute_attribution_change":abs(attr_effect[i,g]),"de_log1p_tpm":de_effect[i,g],"prior_context_gene":genes[g] in CONTEXT_GENES})
    pd.DataFrame(study_rows).to_parquet(OUT/"study_top_attributed_genes.parquet",index=False); pd.DataFrame(comparison).to_csv(OUT/"attribution_vs_de_summary.csv",index=False)
    # Axis consensus ranks and rank stability.
    index={g:i for i,g in enumerate(attr_meta.GSE)}; consensus={}; stability=[]
    for axis,gses in AXES.items():
        idx=[index[g] for g in gses]; consensus[axis]=attr_effect[idx].mean(0); ranks=[np.argsort(-np.abs(attr_effect[i])) for i in idx]
        for a in range(len(idx)):
            for b in range(a+1,len(idx)):
                sa,sb=set(ranks[a][:100]),set(ranks[b][:100]); stability.append({"axis":axis,"GSE_1":gses[a],"GSE_2":gses[b],"top100_overlap":len(sa&sb),"top100_jaccard":len(sa&sb)/len(sa|sb),"absolute_rank_spearman":spearmanr(np.abs(attr_effect[idx[a]]),np.abs(attr_effect[idx[b]])).statistic})
    pd.DataFrame(stability).to_csv(OUT/"within_axis_ranking_stability.csv",index=False)
    consensus_rows=[]
    for axis,values in consensus.items():
        for rank,g in enumerate(np.argsort(-np.abs(values))[:500],1): consensus_rows.append({"axis":axis,"rank":rank,"gene":genes[g],"mean_attribution_change":values[g],"absolute_mean_attribution_change":abs(values[g]),"prior_context_gene":genes[g] in CONTEXT_GENES})
    pd.DataFrame(consensus_rows).to_csv(OUT/"axis_consensus_attributed_genes.csv",index=False)
    # Native mask-token deletion against deterministic size-matched random panels.
    rng=np.random.default_rng(args.seed); deletion=[]
    role_matrix=np.stack([profile for _,_,_,_,profile in role_profiles]); role_lookup={(cid,role):i for i,(cid,_,role,_,_) in enumerate(role_profiles)}
    def role_effects(scores):
        return np.array([scores[role_lookup[(cid,"post_exercise")]]-scores[role_lookup[(cid,"pre_control")]] for cid in attr_meta.contrast_id])
    for axis,gses in AXES.items():
        direction=directions[axis]; original_effect=role_effects(score_all(model,role_matrix,direction,device)); original_map={g:original_effect[index[g]] for g in gses}; ranking=np.argsort(-np.abs(consensus[axis]))
        for size in [25,50,100]:
            panels=[("top",0,ranking[:size])]+[("random",r,rng.choice(len(genes),size,replace=False)) for r in range(args.random_panels)]
            for panel_type,rep,panel in panels:
                effect=role_effects(score_all(model,role_matrix,direction,device,panel))
                for gse in gses:
                    changed=effect[index[gse]]; orig=original_map[gse]
                    deletion.append({"axis":axis,"GSE":gse,"species":attr_meta.loc[index[gse],"species"],"genes_masked":size,"panel_type":panel_type,"replicate":rep,"original_axis_effect":orig,"masked_axis_effect":changed,"absolute_score_change":abs(changed-orig),"fraction_signal_remaining":changed/orig if abs(orig)>1e-9 else np.nan})
    deletion=pd.DataFrame(deletion); deletion.to_parquet(OUT/"deletion_test_results.parquet",index=False); deletion.groupby(["axis","genes_masked","panel_type"],as_index=False).agg(mean_absolute_change=("absolute_score_change","mean"),sd_absolute_change=("absolute_score_change","std"),mean_fraction_remaining=("fraction_signal_remaining","mean"),sd_fraction_remaining=("fraction_signal_remaining","std"),observations=("GSE","size")).to_csv(OUT/"deletion_test_summary.csv",index=False)
    top=pd.DataFrame(consensus_rows).query("rank <= 30"); fig,axes=plt.subplots(1,2,figsize=(12,9),sharey=False)
    for ax,axis in zip(axes,AXES):
        x=top[top.axis.eq(axis)].sort_values("absolute_mean_attribution_change"); ax.barh(x.gene,x.mean_attribution_change,color=np.where(x.mean_attribution_change>0,"#C44E52","#4C72B0")); ax.set(title=f"{axis} consensus IG",xlabel="Post − pre attribution")
    fig.tight_layout(); fig.savefig(FIGURES/"axis_consensus_top_genes.png",dpi=320,bbox_inches="tight"); fig.savefig(FIGURES/"axis_consensus_top_genes.pdf",bbox_inches="tight"); plt.close(fig)
    summary=deletion.groupby(["axis","genes_masked","panel_type"],as_index=False).absolute_score_change.mean(); fig,ax=plt.subplots(figsize=(8,5))
    for (axis,panel),x in summary.groupby(["axis","panel_type"]): ax.plot(x.genes_masked,x.absolute_score_change,marker="o",label=f"{axis} {panel}")
    ax.set(xlabel="Genes masked",ylabel="Mean absolute latent-score change",title="Native-mask deletion test"); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES/"deletion_test.png",dpi=320); fig.savefig(FIGURES/"deletion_test.pdf"); plt.close(fig)
    provenance={"encoder":"frozen r7hnr92k","target":"dot product of mean-pooled sample embedding with unit latent-axis direction","direction":"normalized mean of normalized fixed study post-minus-pre response vectors","ig_baseline":"all-zero log1p(TPM) input","ig_steps":args.ig_steps,"ig_rule":"midpoint Riemann","deletion":"native -10 mask token","random_panels":args.random_panels,"contrasts":"unchanged","GSE151066":"unassigned/intermediate; attributed to nearer fixed direction by absolute projection and excluded from consensus"}; (OUT/"provenance.json").write_text(json.dumps(provenance,indent=2)); log("Latent-axis attribution complete")

if __name__=="__main__": main()
