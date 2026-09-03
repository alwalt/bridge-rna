#!/usr/bin/env python3
"""Decode Hallmark membership from frozen BridgeRNA contextual gene tokens."""

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
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

if not hasattr(sys, "get_int_max_str_digits"):
    def _get_int_max_str_digits() -> int: return 0
    sys.get_int_max_str_digits = _get_int_max_str_digits  # type: ignore[attr-defined]
if not hasattr(sys, "set_int_max_str_digits"):
    def _set_int_max_str_digits(maxdigits: int) -> None: del maxdigits
    sys.set_int_max_str_digits = _set_int_max_str_digits  # type: ignore[attr-defined]

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "benchmarks/tcga_downstream/pipeline"))
sys.path.insert(0, str(HERE / "pipeline"))
sys.path.insert(0, str(ROOT))

from run_attention_pooling import load_frozen_encoder
from run_hallmark_readout import map_hallmarks, resolve
from src.fm_embed.vocab import load_canonical_genes

OUT = HERE / "results" / "gene_context_hallmark"
WORK = HERE / "work" / "gene_context_hallmark"
FIGURES = OUT / "figures"
AXIS = {"Axis A":["GSE108643","GSE86931","GSE126962","GSE132520"],
        "Axis B":["GSE71972","GSE87748","GSE97718"]}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}",flush=True)


class GeneHallmarkHead(nn.Module):
    def __init__(self) -> None:
        super().__init__(); self.network=nn.Sequential(nn.Linear(512,256),nn.GELU(),nn.Linear(256,50))
    def forward(self,x:torch.Tensor)->torch.Tensor: return self.network(x)


def membership(names:list[str],sets:list[np.ndarray],genes:list[str])->tuple[np.ndarray,pd.DataFrame]:
    target=np.zeros((len(genes),len(names)),dtype=np.float32); rows=[]
    for j,(name,index) in enumerate(zip(names,sets)):
        target[index,j]=1
        for i in index: rows.append({"gene":genes[i],"gene_index":i,"hallmark":name,"hallmark_index":j})
    return target,pd.DataFrame(rows)


def choose_cohort(path:Path,total:int,seed:int)->pd.DataFrame:
    source=pd.read_parquet(path); allocation={"train":int(total*.75),"val":int(total*.125)}; allocation["test"]=total-sum(allocation.values())
    rng=np.random.default_rng(seed); selected=[]
    for split,n in allocation.items():
        frame=source[source.readout_split.eq(split)].copy(); frame=frame.iloc[rng.permutation(len(frame))]
        frame=frame.drop_duplicates("gse").head(n); selected.append(frame)
    result=pd.concat(selected,ignore_index=True)
    if result.gse.nunique()!=len(result) or result.groupby("gse").readout_split.nunique().max()!=1: raise AssertionError("Context cohort GSE leakage")
    result.to_parquet(OUT/"context_training_manifest.parquet",index=False); return result


def hidden(encoder:nn.Module,expression:np.ndarray,row:int,device:torch.device)->torch.Tensor:
    values=torch.as_tensor(np.array(expression[row:row+1],copy=True),dtype=torch.float32,device=device)
    with torch.no_grad(),torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"):
        return encoder._encode_hidden(values)[0].detach().float()


def evaluate(model:nn.Module,encoder:nn.Module,expression:np.ndarray,cohort:pd.DataFrame,target:np.ndarray,
             split:str,device:torch.device)->tuple[pd.DataFrame,np.ndarray,pd.DataFrame]:
    selected=cohort[cohort.readout_split.eq(split)]; sums=np.zeros_like(target,dtype=np.float64); study_rows=[]
    model.eval()
    for count,row in enumerate(selected.itertuples(),1):
        token=hidden(encoder,expression,int(row.matrix_row),device)
        with torch.no_grad(): score=torch.sigmoid(model(token)).cpu().numpy()
        sums+=score
        for j in range(target.shape[1]):
            study_rows.append({"split":split,"gsm":row.gsm,"gse":row.gse,"hallmark_index":j,
                               "auroc":roc_auc_score(target[:,j],score[:,j]),"auprc":average_precision_score(target[:,j],score[:,j])})
        if count%10==0 or count==len(selected): log(f"evaluate {split}: {count}/{len(selected)} studies")
    scores=(sums/len(selected)).astype(np.float32); per_study=pd.DataFrame(study_rows)
    summary=per_study.groupby(["split","hallmark_index"],as_index=False).agg(
        auroc=("auroc","mean"),auroc_sd=("auroc","std"),auprc=("auprc","mean"),auprc_sd=("auprc","std"),studies=("gse","nunique"))
    summary["positives"]=[int(target[:,j].sum()) for j in summary.hallmark_index]
    summary["prevalence"]=[float(target[:,j].mean()) for j in summary.hallmark_index]
    return summary,scores,per_study


def train_head(encoder,expression,cohort,target,names,device,epochs,gene_batch,seed):
    torch.manual_seed(seed); model=GeneHallmarkHead().to(device); y=torch.from_numpy(target).to(device)
    positive=y.sum(0); pos_weight=(len(y)-positive)/positive
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=pos_weight); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    best=-np.inf; best_state=None; history=[]; train=cohort[cohort.readout_split.eq("train")]
    for epoch in range(1,epochs+1):
        model.train(); order=np.random.default_rng(seed+epoch).permutation(len(train)); total_loss=0; steps=0; start=time.monotonic()
        for count,k in enumerate(order,1):
            token=hidden(encoder,expression,int(train.iloc[k].matrix_row),device)
            gene_order=torch.randperm(len(y),device=device)
            for offset in range(0,len(y),gene_batch):
                idx=gene_order[offset:offset+gene_batch]; optimizer.zero_grad(set_to_none=True)
                loss=loss_fn(model(token[idx]),y[idx]); loss.backward(); optimizer.step(); total_loss+=float(loss.detach()); steps+=1
            if count%10==0 or count==len(train): log(f"train epoch={epoch} studies={count}/{len(train)} loss={total_loss/steps:.4f} elapsed={(time.monotonic()-start)/60:.1f}m")
        val_metrics,_,_=evaluate(model,encoder,expression,cohort,target,"val",device); score=val_metrics.auprc.median()
        history.append({"epoch":epoch,"train_loss":total_loss/steps,"val_median_auroc":val_metrics.auroc.median(),"val_median_auprc":score})
        if score>best: best=score; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state); torch.save({"state_dict":best_state,"seed":seed,"membership_names":names,"validation_median_auprc":best},OUT/"gene_hallmark_head.pt")
    pd.DataFrame(history).to_csv(OUT/"training_history.csv",index=False); return model


def contrast_scores(model,encoder,expression,manifest,members,names,genes,device):
    lookup=manifest.reset_index().set_index("GSM")["index"]; role_sums={}; role_n={}
    selected=set(members.GSM)
    for count,gsm in enumerate(sorted(selected),1):
        row=int(lookup.loc[gsm]); token=hidden(encoder,expression,row,device)
        with torch.no_grad(): score=torch.sigmoid(model(token)).cpu().numpy().astype(np.float32)
        member=members[members.GSM.eq(gsm)].iloc[0]; key=(member.contrast_id,member.role)
        role_sums[key]=role_sums.get(key,0)+score; role_n[key]=role_n.get(key,0)+1
        if count%10==0 or count==len(selected): log(f"exercise contextual scoring: {count}/{len(selected)} samples")
    deltas=[]; study_hallmark=[]; top=[]
    metadata=[]
    for contrast_id,group in members.groupby("contrast_id",sort=True):
        delta=role_sums[(contrast_id,"post_exercise")]/role_n[(contrast_id,"post_exercise")]-role_sums[(contrast_id,"pre_control")]/role_n[(contrast_id,"pre_control")]
        deltas.append(delta); first=group.iloc[0]; metadata.append({"contrast_id":contrast_id,"species":first.species,"GSE":first.GSE})
        for j,name in enumerate(names):
            member_delta=delta[:,j][target_global[:,j]>0]
            study_hallmark.append({"contrast_id":contrast_id,"species":first.species,"GSE":first.GSE,"hallmark":name,"member_genes":len(member_delta),"mean_member_score_delta":member_delta.mean(),"median_member_score_delta":np.median(member_delta),"fraction_member_increased":np.mean(member_delta>0)})
        magnitude=np.sqrt(np.mean(delta**2,axis=1)); order=np.argsort(-magnitude)[:100]
        for rank,i in enumerate(order,1):
            j=int(np.argmax(np.abs(delta[i]))); top.append({"contrast_id":contrast_id,"species":first.species,"GSE":first.GSE,"rank":rank,"gene":genes[i],"rms_hallmark_score_delta":magnitude[i],"largest_change_hallmark":names[j],"largest_score_delta":delta[i,j]})
    return pd.DataFrame(metadata),np.stack(deltas),pd.DataFrame(study_hallmark),pd.DataFrame(top)


def cosine_rows(a,b):
    return (a@b.T)/(np.maximum(np.linalg.norm(a,axis=1)[:,None],1e-12)*np.maximum(np.linalg.norm(b,axis=1)[None,:],1e-12))


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--samples",type=int,default=256); parser.add_argument("--epochs",type=int,default=2); parser.add_argument("--gene-batch",type=int,default=2048); parser.add_argument("--device",default="cuda:0"); parser.add_argument("--seed",type=int,default=42); args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True); WORK.mkdir(parents=True,exist_ok=True); FIGURES.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((HERE/"config.json").read_text()); genes=load_canonical_genes(resolve(cfg["canonical_genes"])); names,sets,mapping=map_hallmarks(resolve(cfg["hallmark_readout"]["hallmark_gene_sets"]),genes)
    global target_global
    target_global,membership_table=membership(names,sets,genes); membership_table.to_parquet(OUT/"gene_hallmark_membership.parquet",index=False); mapping.to_csv(OUT/"hallmark_mapping.csv",index=False)
    source_manifest=pd.read_parquet(HERE/"results/hallmark_readout/sample_manifest.parquet"); cohort=choose_cohort(HERE/"results/hallmark_readout/sample_manifest.parquet",args.samples,args.seed)
    shape=(len(source_manifest),len(genes)); expression=np.memmap(HERE/"work/hallmark_readout/archs4_log1p_tpm.float32.mmap",dtype="float32",mode="r",shape=shape)
    device=torch.device(args.device if torch.cuda.is_available() else "cpu"); encoder=load_frozen_encoder(device); model=train_head(encoder,expression,cohort,target_global,names,device,args.epochs,args.gene_batch,args.seed)
    test_metrics,test_mean,per_study=evaluate(model,encoder,expression,cohort,target_global,"test",device); test_metrics["hallmark"]=[names[i] for i in test_metrics.hallmark_index]; test_metrics.to_csv(OUT/"heldout_per_hallmark_metrics.csv",index=False); np.save(WORK/"heldout_mean_gene_scores.npy",test_mean)
    per_study["hallmark"]=[names[i] for i in per_study.hallmark_index]; per_study.to_parquet(OUT/"heldout_per_study_hallmark_metrics.parquet",index=False)
    summary=pd.DataFrame([{"test_studies":int((cohort.readout_split=="test").sum()),"median_auroc":test_metrics.auroc.median(),"median_auroc_sd":test_metrics.auroc_sd.median(),"median_auprc":test_metrics.auprc.median(),"median_auprc_sd":test_metrics.auprc_sd.median(),"macro_auroc":test_metrics.auroc.mean(),"macro_auprc":test_metrics.auprc.mean()}]); summary.to_csv(OUT/"heldout_summary.csv",index=False)
    exercise_manifest=pd.read_parquet(HERE/"results/matched_manifest.parquet"); exercise_expression=np.load(HERE/"work/matched_log1p_tpm_corrected.npy",mmap_mode="r"); members=pd.read_parquet(HERE/"results/contrast_members.parquet")
    meta,deltas,study_hallmark,top=contrast_scores(model,encoder,exercise_expression,exercise_manifest,members,names,genes,device); study_hallmark.to_parquet(OUT/"exercise_study_hallmark_changes.parquet",index=False); top.to_csv(OUT/"exercise_top_contextual_gene_changes.csv",index=False)
    np.save(WORK/"exercise_gene_hallmark_deltas.npy",deltas)
    flat=deltas.reshape(len(deltas),-1); human=meta.species.eq("human").to_numpy(); mouse=meta.species.eq("mouse").to_numpy(); matrix=cosine_rows(flat[human],flat[mouse]); humans=meta.loc[human,"GSE"].tolist(); mice=meta.loc[mouse,"GSE"].tolist(); pd.DataFrame(matrix,index=humans,columns=mice).to_csv(OUT/"cross_species_contextual_response_similarity.csv")
    study={g:i for i,g in enumerate(meta.GSE)}; axis_rows=[]
    for axis,gses in AXIS.items():
        idx=[study[g] for g in gses]; within=cosine_rows(flat[idx],flat[idx]); axis_rows.append({"axis":axis,"within_mean_cosine":within[np.triu_indices(len(idx),1)].mean()})
    between=cosine_rows(flat[[study[g] for g in AXIS["Axis A"]]],flat[[study[g] for g in AXIS["Axis B"]]]).mean(); pd.DataFrame(axis_rows).assign(between_axes_cosine=between).to_csv(OUT/"axis_geometry.csv",index=False)
    fig,ax=plt.subplots(figsize=(7,5)); im=ax.imshow(matrix,cmap="coolwarm",vmin=-1,vmax=1); ax.set_xticks(range(len(mice)),mice,rotation=35,ha="right"); ax.set_yticks(range(len(humans)),humans); ax.set(title="Contextual gene-Hallmark exercise response",xlabel="Mouse study",ylabel="Human study"); fig.colorbar(im,ax=ax,label="Cosine"); fig.tight_layout(); fig.savefig(FIGURES/"cross_species_contextual_similarity.png",dpi=320); fig.savefig(FIGURES/"cross_species_contextual_similarity.pdf"); plt.close(fig)
    best=test_metrics.nlargest(10,"auprc"); worst=test_metrics.nsmallest(10,"auprc"); pd.concat([best.assign(group="best"),worst.assign(group="worst")]).to_csv(OUT/"best_worst_hallmarks.csv",index=False)
    provenance={"encoder":"frozen r7hnr92k","contextual_tensor_cached":False,"training_samples":args.samples,"training_GSEs":cohort.gse.nunique(),"split_unit":"GSE","epochs":args.epochs,"membership_target":"fixed binary gene x Hallmark","interpretation":"sigmoid outputs are contextual association scores, not pathway activity probabilities","seed":args.seed}; (OUT/"provenance.json").write_text(json.dumps(provenance,indent=2)); log("Gene-context Hallmark experiment complete")


if __name__=="__main__": main()
