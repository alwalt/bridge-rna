#!/usr/bin/env python3
"""Analyze the frozen BridgeRNA static gene embedding."""

from __future__ import annotations

import argparse, hashlib, json, time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import hypergeom
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parents[1]
DEFAULT_OUT, WORK = HERE / "results/static_embeddings", HERE / "work"
CHECKPOINT = ROOT / "model/r7hnr92k/best_model.pt"
VOCAB = ROOT / "data/ensembl/canonical_genes.csv"
SPLIT = ROOT / "data/archs4/training/sample_split/train_samples.parquet"
H5 = {"human": ROOT / "data/archs4/human_gene_v2.5.h5", "mouse": ROOT / "data/archs4/mouse_gene_v2.5.h5"}
LENGTHS = {"human": ROOT / "data/gencode/gencode_v49_gene_exon_lengths.csv", "mouse": ROOT / "data/gencode/gencode_v49_mouse_gene_exon_lengths.csv"}
ORTHOLOGS = ROOT / "data/ensembl/orthologs_one2one.txt"
GMT_DIR = ROOT / "benchmarks/cross_species_exercise_response/results/per_study_ranked_gsea"
GMTS = {"go": GMT_DIR / "GO_Biological_Process_2026.gmt", "kegg": GMT_DIR / "KEGG_2026.gmt"}
CACHED_EXPRESSION = ROOT / "benchmarks/cross_species_exercise_response/work/hallmark_readout/archs4_log1p_tpm.float32.mmap"
CACHED_MANIFEST = ROOT / "benchmarks/cross_species_exercise_response/results/hallmark_readout/sample_manifest.parquet"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Logger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True); self.path = path; path.write_text("")
    def __call__(self, message: str) -> None:
        line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"; print(line, flush=True)
        with self.path.open("a") as handle: handle.write(line + "\n")


def decode(values: np.ndarray) -> list[str]:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in values]


def mouse_mapping() -> dict[str, str]:
    table = pd.read_csv(ORTHOLOGS, sep="\t")
    table = table.loc[table["Human homology type"].eq("ortholog_one2one") & table["Human orthology confidence [0 low, 1 high]"].eq(1), ["Gene name", "Human gene name"]].dropna().drop_duplicates()
    table = table.loc[~table["Gene name"].duplicated(False) & ~table["Human gene name"].duplicated(False)]
    return dict(zip(table["Human gene name"].astype(str), table["Gene name"].astype(str)))


def aligned_rows_and_lengths(species: str, genes: list[str], h5_symbols: list[str]):
    lengths = pd.read_csv(LENGTHS[species]).drop_duplicates("gene_symbol").set_index("gene_symbol")["exon_length"]
    targets = genes if species == "human" else [mouse_mapping().get(g) for g in genes]
    symbol_rows: dict[str, list[int]] = {}
    for row, symbol in enumerate(h5_symbols): symbol_rows.setdefault(symbol, []).append(row)
    all_rows = [symbol_rows.get(symbol, []) if symbol is not None else [] for symbol in targets]
    width = max((len(x) for x in all_rows), default=0)
    rows = np.full((len(genes), width), -1, dtype=np.int64)
    for i, indices in enumerate(all_rows): rows[i, :len(indices)] = indices
    length_vector = np.array([lengths.get(symbol, np.nan) if symbol is not None else np.nan for symbol in targets], float)
    usable = (rows[:, 0] >= 0) & np.isfinite(length_vector) & (length_vector > 0)
    info = {"genes_with_h5_row": int((rows[:, 0] >= 0).sum()), "genes_with_length": int(np.isfinite(length_vector).sum()), "genes_used": int(usable.sum()), "duplicate_source_rows": int(sum(max(0, len(x)-1) for x in all_rows))}
    return rows, length_vector, info


def training_mean_tpm(genes: list[str], max_samples: int | None, chunk_size: int, log: Logger):
    split = pd.read_parquet(SPLIT, columns=["sample_id", "species"])
    if split.sample_id.duplicated().any(): raise AssertionError("Training split contains duplicate sample IDs")
    if max_samples is None:
        selected = split.copy()
    else:
        per_species = max_samples // 2
        selected = pd.concat([part.sample(min(per_species, len(part)), random_state=20260919) for _, part in split.groupby("species", sort=True)], ignore_index=True)
    total_sum, processed, mapping_info, started = np.zeros(len(genes), np.float64), 0, {}, time.monotonic()
    log(f"Training TPM scan starting: {len(selected):,} samples; estimated 15-35 minutes for full cohort")
    for species in ("human", "mouse"):
        wanted = set(selected.loc[selected.species.eq(species), "sample_id"].astype(str))
        with h5py.File(H5[species], "r") as handle:
            accessions = decode(handle["meta/samples/geo_accession"][:]); accession_to_col = {x: i for i, x in enumerate(accessions)}
            missing = wanted - accession_to_col.keys()
            if missing: raise AssertionError(f"{len(missing)} {species} training samples absent from H5")
            columns = np.array(sorted(accession_to_col[x] for x in wanted), np.int64)
            rows, lengths, info = aligned_rows_and_lengths(species, genes, decode(handle["meta/genes/symbol"][:]))
            mapping_info[species] = info; usable = (rows[:, 0] >= 0) & np.isfinite(lengths) & (lengths > 0); expr = handle["data/expression"]
            for start in range(0, len(columns), chunk_size):
                cols = columns[start:start + chunk_size]; raw = np.asarray(expr[:, cols], np.float64)
                counts = np.zeros((len(genes), len(cols)), np.float64)
                for slot in range(rows.shape[1]):
                    valid = usable & (rows[:, slot] >= 0); counts[valid] += raw[rows[valid, slot]]
                rates = np.zeros_like(counts); rates[usable] = counts[usable] / (lengths[usable, None] / 1000.0)
                denominator = rates.sum(axis=0)
                if np.any(~np.isfinite(denominator) | (denominator <= 0)): raise AssertionError("Non-positive TPM denominator")
                total_sum += (rates / denominator[None, :] * 1e6).sum(axis=1); processed += len(cols)
                elapsed = time.monotonic() - started; rate = processed / max(elapsed, 1e-9); remaining = (len(selected)-processed) / max(rate, 1e-9)
                if start == 0 or processed % (chunk_size * 20) < len(cols) or processed == len(selected):
                    log(f"TPM {processed:,}/{len(selected):,}; elapsed={elapsed/60:.1f}m; rate={rate:.1f} samples/s; ETA={remaining/60:.1f}m")
    if processed != len(selected): raise AssertionError(f"Processed {processed}, expected {len(selected)}")
    info = {"samples": processed, "samples_by_species": selected.groupby("species").size().astype(int).to_dict(), "mapping": mapping_info, "sampled": max_samples is not None}
    return total_sum / processed, info


def cached_training_mean_tpm(genes: list[str], out: Path, log: Logger):
    """Use the available canonical expression cache, restricted to model-train GSMs."""
    split = pd.read_parquet(SPLIT, columns=["sample_id", "species"])
    manifest = pd.read_parquet(CACHED_MANIFEST)
    if len(manifest) != 40_000 or sorted(manifest.matrix_row.tolist()) != list(range(40_000)):
        raise AssertionError("Cached expression manifest invariant failed")
    expected_bytes = len(manifest) * len(genes) * np.dtype("float32").itemsize
    if CACHED_EXPRESSION.stat().st_size != expected_bytes:
        raise AssertionError("Cached expression shape/size mismatch")
    train_ids = set(split.sample_id.astype(str))
    cohort = manifest.loc[manifest.gsm.astype(str).isin(train_ids)].copy().sort_values("matrix_row")
    if cohort.gsm.duplicated().any() or cohort.empty:
        raise AssertionError("Invalid cached training-reference cohort")
    cohort_out = cohort[["gsm", "species", "gse", "matrix_row"]].rename(columns={"gsm": "sample_id"})
    cohort_out.to_parquet(out / "training_expression_cohort.parquet", index=False)
    cohort_out.to_csv(out / "training_expression_cohort.csv", index=False)
    matrix = np.memmap(CACHED_EXPRESSION, mode="r", dtype="float32", shape=(len(manifest), len(genes)))
    total = np.zeros(len(genes), np.float64)
    for start in range(0, len(cohort), 1024):
        rows = cohort.matrix_row.iloc[start:start+1024].to_numpy(int)
        total += np.expm1(np.asarray(matrix[rows], dtype=np.float64)).sum(axis=0)
    log(f"Computed mean TPM from {len(cohort):,} cached samples belonging to the model training split")
    return total / len(cohort), {"samples": len(cohort), "samples_by_species": cohort.groupby("species").size().astype(int).to_dict(), "sampled": True, "sampling_frame": "intersection of existing 40,000-sample canonical expression cache with model training split", "cache_representation": "natural log1p(TPM); inverted with expm1 before arithmetic mean", "full_training_samples": len(split), "coverage_fraction": len(cohort) / len(split)}


def load_gmt(path: Path) -> dict[str, set[str]]:
    result = {}
    with path.open() as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 3: result[fields[0]] = {x.upper() for x in fields[1:] if x}
    return result


def enrich(assignments: pd.DataFrame, library: str, path: Path):
    terms = load_gmt(path); vocab = set(assignments.gene); annotated = vocab & set().union(*terms.values())
    filtered = {name: genes & annotated for name, genes in terms.items()}; filtered = {name: x for name, x in filtered.items() if 5 <= len(x) <= 2000}
    rows = []
    for cluster in range(1, 11):
        query = set(assignments.loc[assignments.cluster.eq(cluster), "gene"]) & annotated
        for term, members in filtered.items():
            overlap = sorted(query & members); p = hypergeom.sf(len(overlap)-1, len(annotated), len(members), len(query))
            rows.append({"library": library, "cluster": cluster, "term": term, "p_value": float(p), "overlap_size": len(overlap), "cluster_annotated_size": len(query), "term_size": len(members), "background_size": len(annotated), "overlap_genes": ";".join(overlap)})
    out = pd.DataFrame(rows); out["fdr"] = np.nan
    for _, idx in out.groupby("cluster").groups.items(): out.loc[idx, "fdr"] = multipletests(out.loc[idx, "p_value"], method="fdr_bh")[1]
    out["significant"] = out.fdr < 0.05; out = out.sort_values(["cluster", "fdr", "p_value", "term"]).reset_index(drop=True)
    return out, {"library_terms_loaded": len(terms), "terms_tested_per_cluster": len(filtered), "annotated_background_genes": len(annotated)}


def bar_comparison(enrichment: pd.DataFrame, clusters: tuple[int, int], title: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for ax, cluster in zip(axes, clusters):
        frame = enrichment.loc[enrichment.cluster.eq(cluster)].nsmallest(10, ["fdr", "p_value"]).copy().iloc[::-1]
        frame["label"] = frame.term.str.replace(r" \(GO:\d+\)$", "", regex=True).str.slice(0, 55); frame["score"] = -np.log10(frame.fdr.clip(lower=np.finfo(float).tiny))
        ax.barh(frame.label, frame.score, color=np.where(frame.significant, "#2a788e", "#bdbdbd")); ax.axvline(-np.log10(.05), color="black", linestyle="--", linewidth=1)
        ax.set_title(f"Cluster {cluster} (n={int(frame.cluster_annotated_size.iloc[0]):,} annotated)"); ax.set_xlabel(r"$-\log_{10}$(BH FDR)")
    fig.suptitle(title)
    for suffix in ("png", "pdf"): fig.savefig(path.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT); parser.add_argument("--exact-full-training", action="store_true", help="Reconstruct TPM for all 640,000 training samples from slow compressed H5 sources"); parser.add_argument("--max-training-samples", type=int, help="Only with --exact-full-training; deterministic balanced smoke-test subset"); parser.add_argument("--expression-chunk-size", type=int, default=128); parser.add_argument("--seed", type=int, default=42); args = parser.parse_args()
    out = args.output_dir.resolve(); figures = out / "figures"; out.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True); WORK.mkdir(parents=True, exist_ok=True); log = Logger(out / "run.log"); started = time.monotonic(); log("Static embedding benchmark started")
    vocab = pd.read_csv(VOCAB).sort_values("token_id").reset_index(drop=True)
    if len(vocab) != 15165 or vocab.token_id.tolist() != list(range(1, 15166)) or vocab.gene_symbol.duplicated().any(): raise AssertionError("Canonical vocabulary invariant failed")
    genes = vocab.gene_symbol.astype(str).str.upper().tolist(); checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False); embedding = checkpoint["model_state_dict"]["gene_embedding.weight"].detach().cpu().numpy().astype(np.float32)
    if embedding.shape != (15165, 512) or not np.isfinite(embedding).all(): raise AssertionError(f"Unexpected embedding: {embedding.shape}")
    np.save(WORK / "static_gene_embedding.float32.npy", embedding); log("Extracted model_state_dict.gene_embedding.weight (15,165 x 512)")
    if args.max_training_samples is not None and not args.exact_full_training: raise ValueError("--max-training-samples requires --exact-full-training")
    mean_tpm, expression_info = (training_mean_tpm(genes, args.max_training_samples, args.expression_chunk_size, log) if args.exact_full_training else cached_training_mean_tpm(genes, out, log))
    if not np.isfinite(mean_tpm).all() or (mean_tpm < 0).any(): raise AssertionError("Invalid mean TPM")
    log("Fitting PCA(50) followed by t-SNE"); pca = PCA(n_components=50, random_state=args.seed).fit_transform(embedding); tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", max_iter=1500, random_state=args.seed, method="barnes_hut").fit_transform(pca)
    log("Fitting k-means k=10 on original 512-D embedding"); km = KMeans(n_clusters=10, random_state=args.seed, n_init=50, algorithm="lloyd").fit(embedding); cluster = km.labels_.astype(int) + 1
    assignments = pd.DataFrame({"token_id": vocab.token_id, "gene": genes, "average_training_tpm": mean_tpm, "log10_average_training_tpm": np.log10(mean_tpm+1e-3), "tsne_1": tsne[:, 0], "tsne_2": tsne[:, 1], "cluster": cluster}); assignments.to_csv(out / "gene_embedding_analysis.csv", index=False)
    summary = assignments.groupby("cluster", as_index=False).agg(genes=("gene", "size"), mean_average_training_tpm=("average_training_tpm", "mean"), median_average_training_tpm=("average_training_tpm", "median")); summary.to_csv(out / "cluster_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True); scatter = ax.scatter(tsne[:, 0], tsne[:, 1], c=assignments.log10_average_training_tpm, s=5, alpha=.75, cmap="viridis", linewidths=0, rasterized=True); fig.colorbar(scatter, ax=ax, label=r"$\log_{10}$(mean training TPM + 0.001)"); ax.set(title="BridgeRNA static gene embeddings", xlabel="t-SNE 1", ylabel="t-SNE 2")
    for suffix in ("png", "pdf"): fig.savefig(figures / f"tsne_by_training_tpm.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig); fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True); scatter = ax.scatter(tsne[:, 0], tsne[:, 1], c=cluster, s=5, alpha=.8, cmap="tab10", vmin=.5, vmax=10.5, linewidths=0, rasterized=True); fig.colorbar(scatter, ax=ax, ticks=range(1, 11), label="K-means cluster"); ax.set(title="K-means clusters in static embedding space", xlabel="t-SNE 1", ylabel="t-SNE 2")
    for suffix in ("png", "pdf"): fig.savefig(figures / f"tsne_by_cluster.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig); enrich_meta = {}
    for name, gmt in GMTS.items():
        log(f"Testing {name.upper()} enrichment for all clusters"); result, meta = enrich(assignments, name.upper(), gmt); result.to_csv(out / f"{name}_enrichment.csv", index=False); enrich_meta[name] = meta
    bar_comparison(pd.read_csv(out / "go_enrichment.csv"), (6, 9), "GO Biological Process enrichment", figures / "go_clusters_6_9"); bar_comparison(pd.read_csv(out / "kegg_enrichment.csv"), (2, 7), "KEGG enrichment", figures / "kegg_clusters_2_7")
    provenance = {"status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(), "checkpoint": str(CHECKPOINT.relative_to(ROOT)), "checkpoint_sha256": sha256(CHECKPOINT), "embedding_tensor": "model_state_dict.gene_embedding.weight", "embedding_shape": list(embedding.shape), "canonical_vocabulary": str(VOCAB.relative_to(ROOT)), "canonical_vocabulary_sha256": sha256(VOCAB), "training_split": str(SPLIT.relative_to(ROOT)), "training_split_sha256": sha256(SPLIT), "expression": {**expression_info, "statistic": "arithmetic mean of per-sample TPM", "normalization": "species-specific exon-length TPM on canonical vocabulary", "pseudocount_for_color_only": .001}, "tsne": {"input": "PCA(50) of raw 512-D embedding", "perplexity": 30, "init": "pca", "learning_rate": "auto", "max_iter": 1500, "seed": args.seed}, "kmeans": {"input": "raw 512-D embedding", "k": 10, "n_init": 50, "algorithm": "lloyd", "seed": args.seed, "reported_cluster": "zero-based sklearn label + 1"}, "enrichment": {"method": "one-sided hypergeometric over-representation; BH within cluster/library", "term_size_filter": [5, 2000], **enrich_meta, **{f"{name}_gmt_sha256": sha256(path) for name, path in GMTS.items()}}, "limitations": ["the cached training-expression reference contains human samples only and covers a subset of the full model training manifest", "t-SNE preserves local neighborhoods but not global distances or cluster geometry", "k-means cluster identifiers are arbitrary deterministic labels, not biological ranks", "enrichment is descriptive and does not validate that embedding clusters encode causal biology"], "elapsed_seconds": time.monotonic()-started}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n"); log(f"Complete in {(time.monotonic()-started)/60:.1f} minutes")


if __name__ == "__main__": main()
