#!/usr/bin/env python3
"""Train/evaluate Task 4 disentanglement and required ablations.

Input dataset directories contain ``manifest.parquet`` and
``bridgerna_embeddings.npy`` in identical sample order. Split roles are fixed
in the manifests. This script refuses study overlap and pair leakage.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# The repository environment uses Python 3.11.0rc1; modern torch expects the
# final 3.11 integer-string guard API.
if not hasattr(sys, "get_int_max_str_digits"):
    def get_int_max_str_digits() -> int:
        return 4300
    sys.get_int_max_str_digits = get_int_max_str_digits  # type: ignore[attr-defined]
if not hasattr(sys, "set_int_max_str_digits"):
    def set_int_max_str_digits(maxdigits: int) -> None:
        return None
    sys.set_int_max_str_digits = set_int_max_str_digits  # type: ignore[attr-defined]
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parents[1]; ROOT = HERE.parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from task4_model import Disentangler, LinearResidualizer


def load_datasets(paths: list[Path]) -> tuple[np.ndarray, pd.DataFrame]:
    arrays, frames = [], []
    for path in paths:
        frame = pd.read_parquet(path / "manifest.parquet")
        z = np.load(path / "bridgerna_embeddings.npy")
        if len(frame) != len(z): raise ValueError(f"Order/length mismatch in {path}")
        arrays.append(z.astype(np.float32)); frames.append(frame)
    metadata = pd.concat(frames, ignore_index=True); embeddings = np.concatenate(arrays)
    required = {"sample_id","dataset","study","pair_id","library_prep","role","same_rna_verified"}
    if missing := required - set(metadata): raise ValueError(f"Missing manifest fields: {sorted(missing)}")
    if set(metadata.library_prep) - {"polyA","ribo"}: raise ValueError("Only explicit polyA/ribo labels are allowed")
    study_roles = metadata.groupby("study").role.nunique()
    if (study_roles > 1).any(): raise ValueError(f"Study leakage: {study_roles[study_roles > 1].index.tolist()}")
    pair_roles = metadata.groupby(["dataset","pair_id"]).role.nunique()
    if (pair_roles > 1).any(): raise ValueError("Biological pair leakage across roles")
    return embeddings, metadata


def probe_metrics(x_train, y_train, x_test, y_test) -> dict:
    scale = StandardScaler().fit(x_train); model = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=20260904)
    model.fit(scale.transform(x_train), y_train); prob = model.predict_proba(scale.transform(x_test))[:, 1]; pred = prob >= .5
    return {"auroc": roc_auc_score(y_test, prob), "balanced_accuracy": balanced_accuracy_score(y_test, pred), "macro_f1": f1_score(y_test, pred, average="macro")}


def pair_metrics(x: np.ndarray, meta: pd.DataFrame) -> dict:
    # Centroid each side for pairs with technical replicates, preventing an
    # arbitrary one-to-one mapping of replicate libraries.
    groups = []
    for (dataset, pair), sub in meta.groupby(["dataset","pair_id"], sort=False):
        if set(sub.library_prep) != {"polyA","ribo"} or not sub.same_rna_verified.all(): continue
        groups.append((dataset, pair, x[sub.index[sub.library_prep == "polyA"]].mean(0), x[sub.index[sub.library_prep == "ribo"]].mean(0)))
    if not groups: return {k: np.nan for k in ["pair_cosine","pair_r1","pair_r5","pair_r10","pair_mrr","median_rank"]} | {"pairs": 0}
    a = np.stack([g[2] for g in groups]); b = np.stack([g[3] for g in groups])
    a = a / np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12); b = b / np.maximum(np.linalg.norm(b,axis=1,keepdims=True),1e-12)
    sim = a @ b.T; ranks_ab = np.array([1 + (sim[i] > sim[i,i]).sum() for i in range(len(groups))]); ranks_ba = np.array([1 + (sim[:,i] > sim[i,i]).sum() for i in range(len(groups))]); ranks=np.r_[ranks_ab,ranks_ba]
    return {"pairs":len(groups),"pair_cosine":float(np.diag(sim).mean()),"pair_r1":float((ranks<=1).mean()),"pair_r5":float((ranks<=5).mean()),"pair_r10":float((ranks<=10).mean()),"pair_mrr":float((1/ranks).mean()),"median_rank":float(np.median(ranks))}


def train_variant(z, meta, variant, cfg, device, output):
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    model=Disentangler(hidden_dim=cfg["hidden_dim"],latent_dim=cfg["fe_dim"],dropout=cfg["dropout"]).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"])
    train=np.where(meta.role.eq("train"))[0]; val=np.where(meta.role.eq("validation"))[0]
    # With no independent validation dataset, train for the frozen epoch count
    # and do not select a checkpoint on train or test performance.
    fixed_epoch_mode = not len(val)
    y=meta.library_prep.map({"polyA":0,"ribo":1}).to_numpy(); y_train_target=y.copy()
    rng=np.random.default_rng(cfg["seed"])
    if variant == "shuffled_labels": y_train_target[train] = rng.permutation(y_train_target[train])
    pair_pairs=[]
    for _, sub in meta.iloc[train].groupby(["dataset","pair_id"],sort=False):
        pa=sub.index[sub.library_prep.eq("polyA")].tolist(); ri=sub.index[sub.library_prep.eq("ribo")].tolist()
        if pa and ri: pair_pairs.append((pa[0],ri[0]))
    if variant == "shuffled_pairs" and pair_pairs:
        ribo=rng.permutation([b for _,b in pair_pairs]); pair_pairs=[(a,int(b)) for (a,_),b in zip(pair_pairs,ribo)]
    best=np.inf; best_state=None; patience=40; stale=0; started=time.time(); rows=[]
    for epoch in range(1,cfg["epochs"]+1):
        model.train(); rng.shuffle(train); losses=[]
        for start in range(0,len(train),cfg["batch_size"]):
            ids=train[start:start+cfg["batch_size"]]; xb=torch.from_numpy(z[ids]).to(device); yb=torch.from_numpy(y_train_target[ids]).long().to(device)
            out=model(xb,cfg["gradient_reversal_lambda"]); recon=F.mse_loss(out["reconstructed"],xb); re_cls=F.cross_entropy(out["re_logits"],yb); adv=F.cross_entropy(out["fe_logits"],yb)
            # Pair loss is evaluated on all training pairs once per minibatch;
            # tiny controlled datasets make this inexpensive and deterministic.
            pids=np.array([v for pair in pair_pairs for v in pair]); pout=model(torch.from_numpy(z[pids]).to(device),cfg["gradient_reversal_lambda"])["fe"]
            pair_loss=torch.stack([1-F.cosine_similarity(pout[2*i:2*i+1],pout[2*i+1:2*i+2]).mean() for i in range(len(pair_pairs))]).mean()
            wr=0 if variant=="without_pair" else cfg["weight_pair"]; wa=0 if variant=="without_adversarial" else cfg["weight_adversarial"]
            loss=cfg["weight_reconstruction"]*recon+cfg["weight_re_classifier"]*re_cls+wa*adv+wr*pair_loss
            optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            if len(val):
                xv=torch.from_numpy(z[val]).to(device); ov=model(xv); vloss=F.mse_loss(ov["reconstructed"],xv).item()
            else: vloss=float('nan')
        rows.append({"epoch":epoch,"train_loss":np.mean(losses),"validation_reconstruction_loss":vloss})
        criterion=float(vloss) if len(val) else float(np.mean(losses))
        if criterion < best-1e-7: best=criterion; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale=0
        else: stale+=1
        if epoch==1 or epoch%10==0: print(f"[heartbeat] {variant} epoch={epoch}/{cfg['epochs']} val={vloss:.6g} elapsed={(time.time()-started)/60:.1f}m",flush=True)
        if not fixed_epoch_mode and stale>=patience: break
        if fixed_epoch_mode: best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state); torch.save({"model_state_dict":best_state,"config":cfg,"variant":variant},output/f"{variant}.pt")
    pd.DataFrame(rows).to_csv(output/f"{variant}_training_history.csv",index=False)
    return model


def representations(model,z,device):
    model.eval(); chunks={"FE":[],"RE":[]}
    with torch.no_grad():
        for start in range(0,len(z),256):
            out=model(torch.from_numpy(z[start:start+256]).to(device)); chunks["FE"].append(out["fe"].cpu().numpy()); chunks["RE"].append(out["re"].cpu().numpy())
    return {k:np.concatenate(v) for k,v in chunks.items()}


def plot_umaps(reps, meta, output):
    import umap
    colors={"polyA":"#377eb8","ribo":"#e41a1c"}
    test=meta.role.eq('test').to_numpy(); tm=meta.loc[test].reset_index(drop=True)
    fig,axes=plt.subplots(len(reps),1,figsize=(7,4*len(reps)),layout='constrained',squeeze=False)
    coord_rows=[]
    for ax,(name,xall) in zip(axes[:,0],reps.items()):
        x=xall[test]; xy=umap.UMAP(n_neighbors=max(2,min(10,len(x)-1)),min_dist=.15,metric='euclidean',random_state=cfg_seed).fit_transform(x)
        for lib in ['polyA','ribo']:
            keep=tm.library_prep.eq(lib).to_numpy();ax.scatter(xy[keep,0],xy[keep,1],s=45,label=lib,color=colors[lib],alpha=.85)
        for (_,pair),q in tm.groupby(['dataset','pair_id']):
            pa=q.index[q.library_prep.eq('polyA')];ri=q.index[q.library_prep.eq('ribo')]
            if len(pa) and len(ri):
                a=xy[pa].mean(0);b=xy[ri].mean(0);ax.plot([a[0],b[0]],[a[1],b[1]],color='0.45',lw=1,alpha=.8)
        ax.set(title=f'{name}: held-out samples',xlabel='UMAP1',ylabel='UMAP2');ax.legend(frameon=False)
        q=tm.copy();q['representation']=name;q['UMAP1']=xy[:,0];q['UMAP2']=xy[:,1];coord_rows.append(q)
    for ext in ['png','pdf']:fig.savefig(output/f'heldout_representation_umaps.{ext}',dpi=400,bbox_inches='tight')
    plt.close(fig);pd.concat(coord_rows,ignore_index=True).to_csv(output/'heldout_umap_coordinates.csv',index=False)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--dataset",action="append",type=Path,required=True); p.add_argument("--device",default="cuda:0"); p.add_argument("--config",type=Path,default=HERE/"config.json"); p.add_argument("--output",type=Path,default=HERE/"results/task4_disentanglement")
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True); cfg=json.loads(a.config.read_text()); z,meta=load_datasets(a.dataset); meta.to_parquet(a.output/"evaluation_manifest.parquet",index=False)
    global cfg_seed
    cfg_seed=cfg['seed']; device=torch.device(a.device if torch.cuda.is_available() else "cpu"); y=meta.library_prep.map({"polyA":0,"ribo":1}).to_numpy(); train=np.where(meta.role.eq("train"))[0]; test=np.where(meta.role.eq("test"))[0]
    if not len(test): raise ValueError("A completely held-out test dataset is required")
    rows=[]; reps={"Bridge":z}; residual=LinearResidualizer().fit(z[train],y[train]); reps["Linear baseline"]=residual.transform(z); joblib.dump(residual,a.output/"linear_residualizer.joblib")
    for variant in ["full","without_pair","without_adversarial","shuffled_labels","shuffled_pairs"]:
        checkpoint=a.output/f'{variant}.pt'
        if checkpoint.exists():
            saved=torch.load(checkpoint,map_location='cpu');m=Disentangler(hidden_dim=cfg['hidden_dim'],latent_dim=cfg['fe_dim'],dropout=cfg['dropout']).to(device);m.load_state_dict(saved['model_state_dict']);print(f'[cache] {checkpoint}',flush=True)
        else: m=train_variant(z,meta,variant,cfg,device,a.output)
        rr=representations(m,z,device)
        if variant=="full": reps.update(rr)
        else: reps[f"FE {variant}"]=rr["FE"]
    # Negative controls are separate deterministic fits, not post-hoc relabeling.
    for name,x in reps.items():
        probe=probe_metrics(x[train],y[train],x[test],y[test]); pair=pair_metrics(x,meta.reset_index(drop=True).iloc[test].set_axis(np.arange(len(test)))) if False else pair_metrics(x[test],meta.iloc[test].reset_index(drop=True))
        rows.append({"representation":name,**probe,**pair,"biology_metric":"cross_library_source_identity_mrr","biology_metric_value":pair["pair_mrr"]})
    pd.DataFrame(rows).to_csv(a.output/"main_summary.csv",index=False)
    np.savez_compressed(a.output/'sample_representations.npz',sample_id=meta.sample_id.to_numpy(dtype=object),**{k.replace(' ','_'):v for k,v in reps.items()})
    plot_umaps({k:v for k,v in reps.items() if k in {'Bridge','FE','RE'}},meta,a.output)
    limitations=["Task 3 is an independent downstream challenge and is not used here."]
    if not meta.role.eq('validation').any(): limitations.append("Only two verified controlled human datasets were available; training used a fixed epoch count with no validation-driven model selection, and SRP127360 remained completely held out.")
    (a.output/"provenance.json").write_text(json.dumps({"datasets":[str(x) for x in a.dataset],"config":cfg,"limitations":limitations},indent=2)+"\n")
    print(pd.DataFrame(rows).to_string(index=False)); print(f"[complete] {a.output}",flush=True)

if __name__=="__main__": main()
