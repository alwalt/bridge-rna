#!/usr/bin/env python3
"""Interpret fixed Task 2 exercise contrasts with the frozen Hallmark readout."""

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

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / "pipeline"))
sys.path.insert(0, str(ROOT))

from run_hallmark_readout import HallmarkHead, map_hallmarks, predict, resolve, ssgsea_scores
from src.fm_embed.vocab import load_canonical_genes

OUT = HERE / "results" / "hallmark_response_axes"
FIGURES = OUT / "figures"
WORK = HERE / "work" / "hallmark_response_axes"
AXIS_A = ["GSE108643", "GSE86931", "GSE126962", "GSE132520"]
AXIS_B = ["GSE71972", "GSE87748", "GSE97718"]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    bn = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return an @ bn.T


def heatmap(ax, values: np.ndarray, rows: list[str], columns: list[str], title: str,
            annotate: bool = False) -> None:
    image=ax.imshow(values,aspect="auto",cmap="coolwarm",vmin=-max(abs(values.min()),abs(values.max())),vmax=max(abs(values.min()),abs(values.max())))
    ax.set_xticks(range(len(columns)),columns,rotation=40,ha="right"); ax.set_yticks(range(len(rows)),rows); ax.set_title(title)
    if annotate:
        for i in range(len(rows)):
            for j in range(len(columns)): ax.text(j,i,f"{values[i,j]:.2f}",ha="center",va="center",fontsize=7)
    ax.figure.colorbar(image,ax=ax,shrink=.75)


def load_frozen_head(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = HallmarkHead(); model.load_state_dict(checkpoint["state_dict"]); model.eval()
    return {**checkpoint, "model": model}


def effect_vectors(scores: np.ndarray, members: pd.DataFrame, manifest: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    lookup = manifest.reset_index().set_index("GSM")["index"]
    rows, vectors = [], []
    for contrast_id, group in members.groupby("contrast_id", sort=True):
        post = lookup.loc[group.loc[group.role.eq("post_exercise"), "GSM"]].to_numpy(int)
        pre = lookup.loc[group.loc[group.role.eq("pre_control"), "GSM"]].to_numpy(int)
        vectors.append(scores[post].mean(0) - scores[pre].mean(0))
        first = group.iloc[0]
        rows.append({"contrast_id": contrast_id, "species": first.species, "GSE": first.GSE,
                     "post_n": len(post), "pre_n": len(pre), "rule": first.rule, "stratum": first.stratum})
    return pd.DataFrame(rows), np.stack(vectors)


def bootstrap_effects(scores: np.ndarray, metadata: pd.DataFrame, members: pd.DataFrame,
                      manifest: pd.DataFrame, replicates: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    index = manifest.reset_index().set_index("GSM")["index"]
    subject = manifest.set_index("GSM")["subject_id"]
    output = np.empty((replicates, len(metadata), scores.shape[1]), dtype=np.float32)
    for study_idx, row in metadata.iterrows():
        group = members[members.contrast_id.eq(row.contrast_id)]
        post_gsm = group.loc[group.role.eq("post_exercise"), "GSM"]
        pre_gsm = group.loc[group.role.eq("pre_control"), "GSM"]
        post = index.loc[post_gsm].to_numpy(int); pre = index.loc[pre_gsm].to_numpy(int)
        if row.species == "human":
            post_subject = subject.loc[post_gsm].astype(str).to_numpy()
            pre_subject = subject.loc[pre_gsm].astype(str).to_numpy()
            shared = sorted(set(post_subject) & set(pre_subject))
            if not shared:
                raise ValueError(f"No paired subjects for {row.GSE}")
            paired = np.stack([
                scores[post[post_subject == s]].mean(0) - scores[pre[pre_subject == s]].mean(0)
                for s in shared
            ])
            draws = rng.integers(0, len(paired), size=(replicates, len(paired)))
            output[:, study_idx] = paired[draws].mean(1)
        else:
            post_draws = rng.integers(0, len(post), size=(replicates, len(post)))
            pre_draws = rng.integers(0, len(pre), size=(replicates, len(pre)))
            output[:, study_idx] = scores[post[post_draws]].mean(1) - scores[pre[pre_draws]].mean(1)
    return output


def cross_matrix(metadata: pd.DataFrame, vectors: np.ndarray) -> tuple[np.ndarray, list[str], list[str]]:
    human = metadata.species.eq("human").to_numpy(); mouse = metadata.species.eq("mouse").to_numpy()
    return cosine(vectors[human], vectors[mouse]), metadata.loc[human, "GSE"].tolist(), metadata.loc[mouse, "GSE"].tolist()


def geometry_summary(source: str, metadata: pd.DataFrame, vectors: np.ndarray) -> dict:
    study = {g:i for i,g in enumerate(metadata.GSE)}
    def pair_mean(group: list[str]) -> float:
        x=cosine(vectors[[study[g] for g in group]],vectors[[study[g] for g in group]])
        return float(x[np.triu_indices(len(group),1)].mean())
    between=cosine(vectors[[study[g] for g in AXIS_A]],vectors[[study[g] for g in AXIS_B]])
    return {"source":source,"axis_a_within_cosine":pair_mean(AXIS_A),"axis_b_within_cosine":pair_mean(AXIS_B),
            "between_axes_cosine":float(between.mean())}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--bootstrap",type=int,default=1000); parser.add_argument("--seed",type=int,default=42); args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True); FIGURES.mkdir(parents=True,exist_ok=True); WORK.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((HERE/"config.json").read_text()); manifest=pd.read_parquet(HERE/"results/matched_manifest.parquet"); members=pd.read_parquet(HERE/"results/contrast_members.parquet")
    expression=np.load(HERE/"work/matched_log1p_tpm_corrected.npy",mmap_mode="r"); embeddings=np.load(HERE/"work/matched_embeddings.npy",mmap_mode="r")
    genes=load_canonical_genes(resolve(cfg["canonical_genes"])); names,sets,_=map_hallmarks(resolve(cfg["hallmark_readout"]["hallmark_gene_sets"]),genes)
    head=load_frozen_head(HERE/"results/hallmark_readout/hallmark_head_bridgerna.pt")
    predicted=predict(head,embeddings); direct=ssgsea_scores(expression,sets)
    # Use the ARCHS4-training target scale learned with the frozen head. This
    # makes pathway dimensions comparable without fitting on exercise studies.
    predicted_z=(predicted-head["y_mean"])/head["y_sd"]; direct_z=(direct-head["y_mean"])/head["y_sd"]
    np.save(WORK/"sample_hallmark_predicted_z.npy",predicted_z); np.save(WORK/"sample_hallmark_direct_ssgsea_z.npy",direct_z)
    metadata,pred_effect=effect_vectors(predicted_z,members,manifest); metadata2,direct_effect=effect_vectors(direct_z,members,manifest)
    if not metadata.equals(metadata2): raise AssertionError("Effect metadata misalignment")
    metadata.to_csv(OUT/"study_contrasts.csv",index=False)
    pd.DataFrame(pred_effect,index=metadata.GSE,columns=names).to_parquet(OUT/"predicted_hallmark_deltas.parquet")
    pd.DataFrame(direct_effect,index=metadata.GSE,columns=names).to_parquet(OUT/"direct_ssgsea_hallmark_deltas.parquet")

    matrices={}; geometry=[geometry_summary("original_bridgerna",metadata,np.load(HERE/"work/response_effects_bridgerna.npy"))]; similarity_rows=[]
    for source,vectors in [("bridgerna_predicted",pred_effect),("direct_ssgsea",direct_effect)]:
        matrix,humans,mice=cross_matrix(metadata,vectors); matrices[source]=matrix
        pd.DataFrame(matrix,index=humans,columns=mice).to_csv(OUT/f"cross_species_similarity_{source}.csv")
        geometry.append(geometry_summary(source,metadata,vectors))
        for i,h in enumerate(humans):
            for j,m in enumerate(mice): similarity_rows.append({"source":source,"human_GSE":h,"mouse_GSE":m,"cosine":matrix[i,j]})
    bridge=pd.read_csv(HERE/"results/response_similarity_bridgerna.csv",index_col=0).to_numpy()
    comparison=[]
    for source,matrix in matrices.items():
        comparison.append({"hallmark_source":source,"pearson_with_bridgerna_matrix":np.corrcoef(bridge.ravel(),matrix.ravel())[0,1],
                           "spearman_with_bridgerna_matrix":pd.Series(bridge.ravel()).corr(pd.Series(matrix.ravel()),method="spearman")})
    pd.DataFrame(comparison).to_csv(OUT/"matrix_comparison.csv",index=False); pd.DataFrame(geometry).to_csv(OUT/"axis_geometry.csv",index=False)

    boot_rows=[]; stability=[]
    study_index={g:i for i,g in enumerate(metadata.GSE)}
    for source,scores,point in [("bridgerna_predicted",predicted_z,pred_effect),("direct_ssgsea",direct_z,direct_effect)]:
        boot=bootstrap_effects(scores,metadata,members,manifest,args.bootstrap,args.seed)
        np.save(WORK/f"bootstrap_effects_{source}.npy",boot)
        human=metadata.species.eq("human").to_numpy(); mouse=metadata.species.eq("mouse").to_numpy()
        for b in range(args.bootstrap):
            mat=cosine(boot[b,human],boot[b,mouse])
            for i,h in enumerate(metadata.loc[human,"GSE"]):
                for j,m in enumerate(metadata.loc[mouse,"GSE"]): boot_rows.append((source,h,m,b,mat[i,j]))
        for axis,gses in [("Axis A",AXIS_A),("Axis B",AXIS_B)]:
            idx=[study_index[g] for g in gses]; axis_boot=boot[:,idx].mean(1); axis_point=point[idx].mean(0)
            for j,name in enumerate(names):
                stability.append({"source":source,"axis":axis,"hallmark":name,"point_delta":axis_point[j],"bootstrap_mean":axis_boot[:,j].mean(),
                                  "bootstrap_sd":axis_boot[:,j].std(ddof=1),"ci_low":np.quantile(axis_boot[:,j],.025),"ci_high":np.quantile(axis_boot[:,j],.975),
                                  "probability_positive":np.mean(axis_boot[:,j]>0)})
    boot_df=pd.DataFrame(boot_rows,columns=["source","human_GSE","mouse_GSE","bootstrap","cosine"])
    boot_summary=boot_df.groupby(["source","human_GSE","mouse_GSE"],as_index=False).cosine.agg(bootstrap_mean="mean",bootstrap_sd="std",ci_low=lambda x:x.quantile(.025),ci_high=lambda x:x.quantile(.975))
    boot_summary=boot_summary.merge(pd.DataFrame(similarity_rows).rename(columns={"cosine":"point_estimate"}),on=["source","human_GSE","mouse_GSE"],how="left",validate="one_to_one")
    boot_summary.to_csv(OUT/"cross_species_similarity_bootstrap.csv",index=False)
    stability_df=pd.DataFrame(stability); stability_df.to_csv(OUT/"axis_hallmark_bootstrap.csv",index=False)

    direction_rows=[]
    for source,vectors in [("bridgerna_predicted",pred_effect),("direct_ssgsea",direct_effect)]:
        frame=pd.DataFrame(vectors,index=metadata.GSE,columns=names)
        for axis,gses in [("Axis A",AXIS_A),("Axis B",AXIS_B)]:
            values=frame.loc[gses]
            for hallmark in names:
                direction_rows.append({"source":source,"axis":axis,"hallmark":hallmark,"mean_delta":values[hallmark].mean(),
                                       "minimum_delta":values[hallmark].min(),"maximum_delta":values[hallmark].max(),
                                       "studies_positive":int((values[hallmark]>0).sum()),"studies_negative":int((values[hallmark]<0).sum()),
                                       "studies":len(values),"consistent_increase":bool((values[hallmark]>0).all()),
                                       "consistent_decrease":bool((values[hallmark]<0).all())})
    pd.DataFrame(direction_rows).to_csv(OUT/"axis_hallmark_direction_consistency.csv",index=False)
    pred=pd.DataFrame(pred_effect,index=metadata.GSE,columns=names); axis_a=pred.loc[AXIS_A].mean(); axis_b=pred.loc[AXIS_B].mean()
    selected=(axis_a-axis_b).abs().sort_values(ascending=False).head(20).index.tolist()
    pd.DataFrame({"hallmark":names,"axis_a_mean":axis_a.reindex(names).values,"axis_b_mean":axis_b.reindex(names).values,"axis_difference":(axis_a-axis_b).reindex(names).values}).sort_values("axis_difference",key=abs,ascending=False).to_csv(OUT/"axis_hallmark_summary.csv",index=False)
    fig,axes=plt.subplots(2,1,figsize=(13,10),sharex=True)
    for ax,(title,frame) in zip(axes,[("BridgeRNA-predicted Hallmark Δ",pred),("Direct ssGSEA Hallmark Δ",pd.DataFrame(direct_effect,index=metadata.GSE,columns=names))]):
        heatmap(ax,frame[selected].T.to_numpy(),selected,frame.index.tolist(),title); ax.set(xlabel="",ylabel="")
    axes[-1].set_xlabel("Study"); fig.tight_layout(); fig.savefig(FIGURES/"informative_hallmark_heatmap.png",dpi=320,bbox_inches="tight"); fig.savefig(FIGURES/"informative_hallmark_heatmap.pdf",bbox_inches="tight"); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(17,5))
    for ax,(title,matrix) in zip(axes,[("BridgeRNA response",bridge),("Predicted Hallmark response",matrices["bridgerna_predicted"]),("Direct ssGSEA response",matrices["direct_ssgsea"])]):
        heatmap(ax,matrix,humans,mice,title,annotate=True)
    fig.tight_layout(); fig.savefig(FIGURES/"response_geometry_comparison.png",dpi=320,bbox_inches="tight"); fig.savefig(FIGURES/"response_geometry_comparison.pdf",bbox_inches="tight"); plt.close(fig)
    provenance={"hallmark_head":"frozen; no retraining","contrast_definitions":"unchanged Task 2 contrast_members.parquet","scale":"ARCHS4 Hallmark-head training target SD units","bootstrap_replicates":args.bootstrap,"human_bootstrap":"paired subjects","mouse_bootstrap":"independent within-role samples","seed":args.seed}
    (OUT/"provenance.json").write_text(json.dumps(provenance,indent=2)); log("Hallmark response-axis analysis complete")


if __name__ == "__main__": main()
