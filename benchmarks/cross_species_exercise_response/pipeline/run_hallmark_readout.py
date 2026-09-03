#!/usr/bin/env python3
"""Frozen BridgeRNA Hallmark readout with study-held-out ARCHS4 evaluation."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fm_embed.vocab import load_canonical_genes

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "results" / "hallmark_readout"
WORK = HERE / "work" / "hallmark_readout"
FIGURES = OUT / "figures"

# This WSL Python build omits two optional integer-string safety APIs that
# recent torch._dynamo imports probe while constructing an optimizer.
if not hasattr(sys, "get_int_max_str_digits"):
    def _get_int_max_str_digits() -> int:
        return 0
    sys.get_int_max_str_digits = _get_int_max_str_digits  # type: ignore[attr-defined]
if not hasattr(sys, "set_int_max_str_digits"):
    def _set_int_max_str_digits(maxdigits: int) -> None:
        del maxdigits
    sys.set_int_max_str_digits = _set_int_max_str_digits  # type: ignore[attr-defined]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sample_study_diverse(manifest: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Round-robin across shuffled GSEs, preventing large studies from dominating."""
    rng = np.random.default_rng(seed)
    groups = {
        gse: group.iloc[rng.permutation(len(group))].copy()
        for gse, group in manifest.groupby("gse", sort=True)
    }
    gses = np.array(sorted(groups), dtype=object)
    chosen: list[pd.DataFrame] = []
    selected = 0
    depth = 0
    while selected < n:
        available = [g for g in rng.permutation(gses) if len(groups[g]) > depth]
        if not available:
            break
        take = available[: n - selected]
        chosen.extend(groups[g].iloc[[depth]] for g in take)
        selected += len(take)
        depth += 1
    if selected < n:
        raise ValueError(f"Only {selected:,} eligible samples were available")
    result = pd.concat(chosen, ignore_index=True)
    if result.gsm.duplicated().any():
        raise AssertionError("Study-diverse sampler produced duplicate GSMs")
    return result


def build_cohort(cfg: dict) -> pd.DataFrame:
    cache = OUT / "sample_manifest.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    manifest = pd.read_parquet(
        resolve(cfg["sample_manifest"]),
        columns=["gsm", "species", "gse_candidates_str", "gse_count", "mapping_status"],
    )
    locations = pd.read_parquet(resolve(cfg["embedding_directory"]) / "sample_locations.parquet")
    eligible = manifest[
        manifest.species.eq("human")
        & manifest.gse_count.eq(1)
        & manifest.gse_candidates_str.notna()
        & manifest.mapping_status.eq("mapped_single")
    ].rename(columns={"gse_candidates_str": "gse"})
    eligible = eligible.merge(
        locations, left_on="gsm", right_on="geo_accession", how="inner", validate="one_to_one"
    )
    selected = sample_study_diverse(
        eligible, int(cfg["hallmark_readout"]["sample_count"]), int(cfg["hallmark_readout"]["seed"])
    )
    rng = np.random.default_rng(int(cfg["hallmark_readout"]["seed"]))
    gses = selected.gse.drop_duplicates().to_numpy(object)
    rng.shuffle(gses)
    n_gse = len(gses)
    val_n = round(n_gse * 0.1)
    test_n = round(n_gse * 0.1)
    split_map = {g: "test" for g in gses[:test_n]}
    split_map.update({g: "val" for g in gses[test_n:test_n + val_n]})
    split_map.update({g: "train" for g in gses[test_n + val_n:]})
    selected["readout_split"] = selected.gse.map(split_map)
    selected["matrix_row"] = np.arange(len(selected))
    selected = selected.sort_values("matrix_row").reset_index(drop=True)
    leakage = selected.groupby("gse").readout_split.nunique().max()
    if leakage != 1:
        raise AssertionError("A GSE occurs in multiple Hallmark splits")
    selected.to_parquet(cache, index=False)
    selected[["gse"]].drop_duplicates().assign(
        split=lambda x: x.gse.map(split_map)
    ).sort_values(["split", "gse"]).to_csv(OUT / "gse_splits.csv", index=False)
    return selected


def extract_log1p_tpm(cohort: pd.DataFrame, genes: list[str], shard_dir: Path) -> np.memmap:
    path = WORK / "archs4_log1p_tpm.float32.mmap"
    shape = (len(cohort), len(genes))
    if path.exists() and path.stat().st_size == int(np.prod(shape)) * 4:
        log("Reusing cached ARCHS4 log1p(TPM) matrix")
        return np.memmap(path, dtype="float32", mode="r", shape=shape)
    output = np.memmap(path, dtype="float32", mode="w+", shape=shape)
    for shard_file, shard_rows in cohort.groupby("shard_file", sort=True):
        parquet = pq.ParquetFile(shard_dir / shard_file)
        if parquet.schema_arrow.names != [*genes, "geo_accession"]:
            raise ValueError(f"Vocabulary mismatch in {shard_file}")
        selected = shard_rows.assign(row_group=shard_rows.row_in_shard.astype(int) // 2048)
        for row_group, rows in selected.groupby("row_group"):
            block = parquet.read_row_group(int(row_group)).to_pandas(ignore_metadata=True)
            start = int(row_group) * 2048
            offsets = rows.row_in_shard.to_numpy(int) - start
            if not np.array_equal(block.iloc[offsets].geo_accession.to_numpy(str), rows.gsm.to_numpy(str)):
                raise ValueError(f"GSM alignment failure in {shard_file}, row group {row_group}")
            tpm = block.iloc[offsets, :-1].to_numpy(np.float32)
            sums = tpm.sum(axis=1, dtype=np.float64)
            if not np.allclose(sums, 1e6, rtol=5e-4, atol=250):
                raise ValueError(f"Non-TPM values in {shard_file}")
            output[rows.matrix_row.to_numpy(int)] = np.log1p(tpm)
        output.flush()
        log(f"Expression {shard_file}: {len(shard_rows):,} selected samples")
    return np.memmap(path, dtype="float32", mode="r", shape=shape)


def prepare_embeddings(cohort: pd.DataFrame, directory: Path) -> np.ndarray:
    path = WORK / "archs4_bridgerna_embeddings.npy"
    if path.exists():
        log("Reusing cached BridgeRNA embeddings")
        return np.load(path, mmap_mode="r")
    spec = json.loads((directory / "embedding_manifest.json").read_text())
    source = np.memmap(
        directory / f"sample_embeddings.{np.dtype(spec['embedding_dtype']).name}.mmap",
        dtype=np.dtype(spec["embedding_dtype"]), mode="r",
        shape=(int(spec["total_samples"]), int(spec["embedding_dim"])),
    )
    embeddings = np.asarray(source[cohort.global_index.to_numpy(int)], dtype=np.float32)
    np.save(path, embeddings)
    return np.load(path, mmap_mode="r")


def map_hallmarks(path: Path, genes: list[str]) -> tuple[list[str], list[np.ndarray], pd.DataFrame]:
    raw = json.loads(path.read_text())
    lookup = {gene.upper(): i for i, gene in enumerate(genes)}
    names, indices, rows = [], [], []
    for name, members in raw.items():
        clean = sorted({str(g).strip().upper() for g in members if str(g).strip()})
        mapped = np.array([lookup[g] for g in clean if g in lookup], dtype=np.int32)
        if len(mapped) < 2:
            raise ValueError(f"Hallmark {name} has fewer than two mapped genes")
        names.append(name); indices.append(mapped)
        rows.append({"hallmark": name, "source_genes": len(clean), "mapped_genes": len(mapped),
                     "mapping_percent": 100 * len(mapped) / len(clean)})
    return names, indices, pd.DataFrame(rows)


def ssgsea_scores(expression: np.ndarray, sets: list[np.ndarray], chunk_size: int = 128) -> np.ndarray:
    """Classic weighted ssGSEA enrichment scores (alpha=0.25), sample by sample.

    The running-sum integral is evaluated analytically from each set's ranked
    hit positions, avoiding construction of a genes x Hallmarks running matrix.
    """
    n_samples, n_genes = expression.shape
    scores = np.empty((n_samples, len(sets)), dtype=np.float32)
    alpha = 0.25
    total_position_sum = n_genes * (n_genes + 1) / 2
    for start in range(0, n_samples, chunk_size):
        stop = min(start + chunk_size, n_samples)
        values = np.asarray(expression[start:stop], dtype=np.float32)
        ranks = rankdata(values, axis=1, method="average").astype(np.float32)
        positions = n_genes - ranks + 1.0  # position 1 is highest expression
        weights = np.power(ranks, alpha)
        for j, gene_idx in enumerate(sets):
            hit_pos = positions[:, gene_idx]
            hit_weight = weights[:, gene_idx]
            hit_integral = (hit_weight * (n_genes - hit_pos + 1)).sum(1) / hit_weight.sum(1)
            miss_integral = (
                total_position_sum - (n_genes - hit_pos + 1).sum(1)
            ) / (n_genes - len(gene_idx))
            scores[start:stop, j] = hit_integral - miss_integral
        if stop == n_samples or stop % 2048 == 0:
            log(f"ssGSEA: {stop:,}/{n_samples:,} samples")
    return scores


def prepare_targets(expression: np.ndarray, names: list[str], sets: list[np.ndarray]) -> np.ndarray:
    path = WORK / "archs4_hallmark_ssgsea.npy"
    if path.exists():
        log("Reusing cached Hallmark ssGSEA targets")
        return np.load(path, mmap_mode="r")
    targets = ssgsea_scores(expression, sets)
    np.save(path, targets)
    return np.load(path, mmap_mode="r")


def prepare_pca(expression: np.ndarray, cohort: pd.DataFrame, components: int, seed: int) -> tuple[np.ndarray, PCA]:
    scores_path, model_path = WORK / "archs4_pca512.npy", WORK / "pca512.pkl"
    if scores_path.exists() and model_path.exists():
        log("Reusing cached PCA representation")
        with model_path.open("rb") as handle:
            return np.load(scores_path, mmap_mode="r"), pickle.load(handle)
    train = cohort.readout_split.eq("train").to_numpy()
    log(f"Fitting {components}-component PCA on {train.sum():,} training samples only")
    pca = PCA(n_components=components, svd_solver="randomized", iterated_power=2, random_state=seed)
    pca.fit(np.asarray(expression[train]))
    scores = np.empty((len(expression), components), dtype=np.float32)
    for start in range(0, len(expression), 1024):
        scores[start:start + 1024] = pca.transform(np.asarray(expression[start:start + 1024])).astype(np.float32)
    np.save(scores_path, scores)
    with model_path.open("wb") as handle:
        pickle.dump(pca, handle)
    pd.DataFrame({"pc": np.arange(1, components + 1), "variance_explained": pca.explained_variance_ratio_,
                  "cumulative_variance": np.cumsum(pca.explained_variance_ratio_)}).to_csv(OUT / "pca_variance.csv", index=False)
    return np.load(scores_path, mmap_mode="r"), pca


class HallmarkHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 50))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def fit_head(name: str, features: np.ndarray, targets: np.ndarray, cohort: pd.DataFrame, cfg: dict) -> dict:
    seed = int(cfg["seed"]); torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint_path = OUT / f"hallmark_head_{name}.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = HallmarkHead().to(device); model.load_state_dict(checkpoint["state_dict"])
        log(f"Reusing trained {name} Hallmark head")
        return {**checkpoint, "model": model}
    masks = {s: cohort.readout_split.eq(s).to_numpy() for s in ["train", "val", "test"]}
    x_mean = np.asarray(features[masks["train"]]).mean(0, dtype=np.float64).astype(np.float32)
    x_sd = np.asarray(features[masks["train"]]).std(0, dtype=np.float64).astype(np.float32); x_sd[x_sd < 1e-6] = 1
    y_mean = np.asarray(targets[masks["train"]]).mean(0, dtype=np.float64).astype(np.float32)
    y_sd = np.asarray(targets[masks["train"]]).std(0, dtype=np.float64).astype(np.float32); y_sd[y_sd < 1e-6] = 1

    def dataset(mask: np.ndarray) -> TensorDataset:
        x = (np.asarray(features[mask], dtype=np.float32) - x_mean) / x_sd
        y = (np.asarray(targets[mask], dtype=np.float32) - y_mean) / y_sd
        return TensorDataset(torch.from_numpy(x), torch.from_numpy(y))

    train_loader = DataLoader(dataset(masks["train"]), batch_size=int(cfg["batch_size"]), shuffle=True)
    val_loader = DataLoader(dataset(masks["val"]), batch_size=1024)
    model = HallmarkHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=1e-4)
    loss_fn = nn.MSELoss(); best_loss = np.inf; best_state = None; stale = 0; history = []
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train(); train_loss = 0.0; seen = 0
        for x, y in train_loader:
            x=x.to(device); y=y.to(device); optimizer.zero_grad(set_to_none=True)
            loss=loss_fn(model(x), y); loss.backward(); optimizer.step()
            train_loss += loss.item()*len(x); seen += len(x)
        model.eval(); val_loss=0.0; val_seen=0
        with torch.inference_mode():
            for x,y in val_loader:
                x=x.to(device); y=y.to(device); loss=loss_fn(model(x),y)
                val_loss += loss.item()*len(x); val_seen += len(x)
        train_loss /= seen; val_loss /= val_seen
        history.append({"representation": name, "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        log(f"{name} epoch={epoch:02d} train={train_loss:.5f} val={val_loss:.5f}")
        if val_loss < best_loss - 1e-5:
            best_loss=val_loss; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale=0
        else:
            stale += 1
            if stale >= int(cfg["patience"]): break
    model.load_state_dict(best_state)
    checkpoint = {"state_dict": best_state, "x_mean": x_mean, "x_sd": x_sd, "y_mean": y_mean, "y_sd": y_sd,
                  "best_val_loss": best_loss, "representation": name, "seed": seed}
    torch.save(checkpoint, checkpoint_path)
    pd.DataFrame(history).to_csv(OUT / f"training_history_{name}.csv", index=False)
    return {**checkpoint, "model": model}


def predict(fit: dict, features: np.ndarray) -> np.ndarray:
    model = fit["model"]; device = next(model.parameters()).device; model.eval(); chunks=[]
    with torch.inference_mode():
        for start in range(0, len(features), 1024):
            x=(np.asarray(features[start:start+1024],dtype=np.float32)-fit["x_mean"])/fit["x_sd"]
            chunks.append(model(torch.from_numpy(x).to(device)).cpu().numpy())
    return np.concatenate(chunks)*fit["y_sd"]+fit["y_mean"]


def metrics(dataset: str, representation: str, truth: np.ndarray, pred: np.ndarray, names: list[str]) -> pd.DataFrame:
    rows=[]
    for j,name in enumerate(names):
        p = pearsonr(truth[:,j],pred[:,j]).statistic if np.std(truth[:,j]) and np.std(pred[:,j]) else np.nan
        s = spearmanr(truth[:,j],pred[:,j]).statistic if np.std(truth[:,j]) and np.std(pred[:,j]) else np.nan
        rows.append({"dataset":dataset,"representation":representation,"hallmark":name,"samples":len(truth),
                     "pearson":p,"spearman":s,"mse":float(np.mean((truth[:,j]-pred[:,j])**2))})
    return pd.DataFrame(rows)


def plot_results(all_metrics: pd.DataFrame, predictions: dict[str, tuple[np.ndarray,np.ndarray]], names: list[str]) -> None:
    test = all_metrics[all_metrics.dataset.eq("archs4_test")]
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,metric in zip(axes,["pearson","spearman"]):
        values=[test.loc[test.representation.eq(r),metric].dropna() for r in ["bridgerna","pca"]]
        ax.boxplot(values, tick_labels=["BridgeRNA","PCA"], showmeans=True)
        ax.set(title=f"Held-out GSE Hallmark {metric}",ylabel=metric.capitalize(),ylim=(-0.1,1.0))
    fig.tight_layout(); fig.savefig(FIGURES/"heldout_performance_distribution.png",dpi=320,bbox_inches="tight"); fig.savefig(FIGURES/"heldout_performance_distribution.pdf",bbox_inches="tight"); plt.close(fig)
    truth,pred=predictions["bridgerna"]
    ranked=test[test.representation.eq("bridgerna")].sort_values("pearson")
    selected=pd.concat([ranked.head(3),ranked.tail(3)]).hallmark.tolist()
    fig,axes=plt.subplots(2,3,figsize=(13,8))
    for ax,name in zip(axes.flat,selected):
        j=names.index(name); ax.scatter(truth[:,j],pred[:,j],s=7,alpha=.35)
        score=ranked.set_index("hallmark").loc[name,"pearson"]
        ax.set(title=f"{name}\nr={score:.3f}",xlabel="Actual ssGSEA",ylabel="Predicted")
    fig.tight_layout(); fig.savefig(FIGURES/"bridgerna_best_worst_scatter.png",dpi=320,bbox_inches="tight"); fig.savefig(FIGURES/"bridgerna_best_worst_scatter.pdf",bbox_inches="tight"); plt.close(fig)


def external_assets(cfg: dict) -> list[tuple[str, Path, Path, Path, str | None]]:
    h=cfg["hallmark_readout"]
    return [("tcga",resolve(h["external_tcga_expression"]),resolve(h["external_tcga_embeddings"]),resolve(h["external_tcga_manifest"]),None),
            ("gtex",resolve(h["external_gtex_expression"]),resolve(h["external_gtex_embeddings"]),resolve(h["external_gtex_manifest"]),"human")]


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--skip-external",action="store_true"); args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True); WORK.mkdir(parents=True,exist_ok=True); FIGURES.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((HERE/"config.json").read_text()); hcfg=cfg["hallmark_readout"]
    genes=load_canonical_genes(resolve(cfg["canonical_genes"])); cohort=build_cohort(cfg)
    log(f"Cohort: {len(cohort):,} samples across {cohort.gse.nunique():,} GSEs")
    names,sets,mapping=map_hallmarks(resolve(hcfg["hallmark_gene_sets"]),genes); mapping.to_csv(OUT/"hallmark_gene_mapping.csv",index=False)
    expression=extract_log1p_tpm(cohort,genes,resolve(cfg["archs4_shard_directory"])); embeddings=prepare_embeddings(cohort,resolve(cfg["embedding_directory"]))
    targets=prepare_targets(expression,names,sets); pca_features,pca_model=prepare_pca(expression,cohort,int(hcfg["pca_components"]),int(hcfg["seed"]))
    fits={"bridgerna":fit_head("bridgerna",embeddings,targets,cohort,hcfg),"pca":fit_head("pca",pca_features,targets,cohort,hcfg)}
    test=cohort.readout_split.eq("test").to_numpy(); metric_tables=[]; test_predictions={}
    for name,features in [("bridgerna",embeddings),("pca",pca_features)]:
        pred=predict(fits[name],np.asarray(features[test])); truth=np.asarray(targets[test]); test_predictions[name]=(truth,pred)
        np.save(WORK/f"archs4_test_predictions_{name}.npy",pred); metric_tables.append(metrics("archs4_test",name,truth,pred,names))
    if not args.skip_external:
        for dataset,expr_path,embed_path,manifest_path,species in external_assets(cfg):
            if not all(p.exists() for p in [expr_path,embed_path,manifest_path]):
                log(f"Skipping {dataset}: one or more local assets missing"); continue
            expr=np.load(expr_path,mmap_mode="r"); emb=np.load(embed_path,mmap_mode="r"); meta=pd.read_parquet(manifest_path)
            if species and "species" in meta and len(meta)!=len(expr): meta=meta[meta.species.eq(species)].reset_index(drop=True)
            if len(expr)!=len(emb): raise ValueError(f"{dataset} expression/embedding row mismatch")
            if len(meta)!=len(expr): log(f"{dataset}: manifest length differs; metrics remain row-aligned by prepared matrices")
            truth=ssgsea_scores(expr,sets); np.save(WORK/f"{dataset}_hallmark_ssgsea.npy",truth)
            representations={"bridgerna":emb,"pca":pca_model.transform(np.asarray(expr)).astype(np.float32)}
            for name,features in representations.items():
                pred=predict(fits[name],features); np.save(WORK/f"{dataset}_predictions_{name}.npy",pred)
                metric_tables.append(metrics(dataset,name,truth,pred,names))
    all_metrics=pd.concat(metric_tables,ignore_index=True); all_metrics.to_csv(OUT/"per_hallmark_metrics.csv",index=False); all_metrics.to_parquet(OUT/"per_hallmark_metrics.parquet",index=False)
    summary=all_metrics.groupby(["dataset","representation"],as_index=False).agg(hallmarks=("hallmark","size"),median_pearson=("pearson","median"),median_spearman=("spearman","median"),median_mse=("mse","median"))
    summary.to_csv(OUT/"summary_metrics.csv",index=False); plot_results(all_metrics,test_predictions,names)
    targets_frame=pd.DataFrame(np.asarray(targets),columns=names); targets_frame.insert(0,"gsm",cohort.gsm.to_numpy()); targets_frame.insert(1,"gse",cohort.gse.to_numpy()); targets_frame.insert(2,"split",cohort.readout_split.to_numpy()); targets_frame.to_parquet(OUT/"archs4_hallmark_targets.parquet",index=False)
    provenance={"backbone":"frozen precomputed BridgeRNA embeddings","samples":len(cohort),"gses":cohort.gse.nunique(),"split_unit":"GSE","expression":"natural log1p(TPM), 15,165 canonical genes","ssgsea":"weighted rank-based ssGSEA running-sum integral, alpha=0.25","pca_fit":"ARCHS4 training GSE split only","seed":int(hcfg["seed"])}
    (OUT/"provenance.json").write_text(json.dumps(provenance,indent=2))
    lines=["# Exploratory Hallmark readout", "", "```text", summary.to_string(index=False), "```", "", "BridgeRNA remained frozen; both readout heads use the same architecture and study-held-out split."]
    (OUT/"SUMMARY.md").write_text("\n".join(lines)+"\n"); log("Hallmark readout complete")


if __name__ == "__main__":
    main()
