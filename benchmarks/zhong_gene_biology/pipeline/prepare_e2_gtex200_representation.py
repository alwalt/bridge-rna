#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "benchmarks" / "zhong_gene_biology"
REP_DIR = BENCH / "representations"
COVERAGE_DIR = BENCH / "coverage"
LOGS_DIR = BENCH / "logs"

DEFAULT_SHARED_ROOT = Path("/home/walt/bridge-rna")

CHECKPOINT_REL = Path("model/r7hnr92k/best_model.pt")
CONFIG_REL = Path("model/r7hnr92k/config.json")
VOCAB_REL = Path("data/ensembl/canonical_genes.csv")

MATCHED_GTEX_TCGA_MANIFEST_REL = Path(
    "benchmarks/static_gene_embeddings/results/matched_gtex_tcga/cohort_manifest.csv"
)
MATCHED_GTEX_TCGA_PROVENANCE_REL = Path(
    "benchmarks/static_gene_embeddings/results/matched_gtex_tcga/provenance.json"
)
CONTEXTUAL_MODULES_SCRIPT_REL = Path(
    "benchmarks/static_gene_embeddings/pipeline/contextual_gene_modules.py"
)
MATCHED_SCRIPT_REL = Path("benchmarks/static_gene_embeddings/pipeline/matched_gtex_tcga.py")
GTEX_LOG1P_TPM_REL = Path("benchmarks/landmark_gene_sufficiency/work/gtex_model_log1p_tpm.npy")
SHARED_GTEX_L12_MEAN_CACHE_REL = Path(
    "benchmarks/static_gene_embeddings/work/contextual_gene_modules/GTEx_L12_final_mean_tokens.float32.npy"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_prefer_shared(relative_path: Path, shared_root: Path) -> tuple[Path, str]:
    shared = shared_root / relative_path
    local = ROOT / relative_path
    if shared.exists():
        return shared, "shared_root"
    if local.exists():
        return local, "worktree"
    return shared, "missing"


class Logger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        path.write_text("")

    def __call__(self, msg: str) -> None:
        line = msg.rstrip("\n")
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


@dataclass
class Assets:
    checkpoint: Path
    checkpoint_source: str
    config: Path
    config_source: str
    vocab: Path
    vocab_source: str
    gtex_manifest: Path
    gtex_manifest_source: str
    gtex_manifest_provenance: Path
    gtex_manifest_provenance_source: str
    gtex_log1p_tpm: Path
    gtex_log1p_tpm_source: str
    contextual_modules_script: Path
    matched_script: Path
    shared_gtex_l12_cache: Path
    shared_gtex_l12_cache_source: str


def load_assets(shared_root: Path) -> Assets:
    checkpoint, checkpoint_source = resolve_prefer_shared(CHECKPOINT_REL, shared_root)
    config, config_source = resolve_prefer_shared(CONFIG_REL, shared_root)
    vocab, vocab_source = resolve_prefer_shared(VOCAB_REL, shared_root)
    gtex_manifest, gtex_manifest_source = resolve_prefer_shared(
        MATCHED_GTEX_TCGA_MANIFEST_REL, shared_root
    )
    gtex_manifest_provenance, gtex_manifest_provenance_source = resolve_prefer_shared(
        MATCHED_GTEX_TCGA_PROVENANCE_REL, shared_root
    )
    gtex_log1p_tpm, gtex_log1p_tpm_source = resolve_prefer_shared(GTEX_LOG1P_TPM_REL, shared_root)
    contextual_modules_script, _ = resolve_prefer_shared(CONTEXTUAL_MODULES_SCRIPT_REL, shared_root)
    matched_script, _ = resolve_prefer_shared(MATCHED_SCRIPT_REL, shared_root)
    shared_gtex_l12_cache, shared_gtex_l12_cache_source = resolve_prefer_shared(
        SHARED_GTEX_L12_MEAN_CACHE_REL, shared_root
    )

    required = {
        "checkpoint": checkpoint,
        "config": config,
        "vocab": vocab,
        "gtex_manifest": gtex_manifest,
        "gtex_manifest_provenance": gtex_manifest_provenance,
        "gtex_log1p_tpm": gtex_log1p_tpm,
        "contextual_modules_script": contextual_modules_script,
        "matched_script": matched_script,
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required assets: {missing}")

    return Assets(
        checkpoint=checkpoint,
        checkpoint_source=checkpoint_source,
        config=config,
        config_source=config_source,
        vocab=vocab,
        vocab_source=vocab_source,
        gtex_manifest=gtex_manifest,
        gtex_manifest_source=gtex_manifest_source,
        gtex_manifest_provenance=gtex_manifest_provenance,
        gtex_manifest_provenance_source=gtex_manifest_provenance_source,
        gtex_log1p_tpm=gtex_log1p_tpm,
        gtex_log1p_tpm_source=gtex_log1p_tpm_source,
        contextual_modules_script=contextual_modules_script,
        matched_script=matched_script,
        shared_gtex_l12_cache=shared_gtex_l12_cache,
        shared_gtex_l12_cache_source=shared_gtex_l12_cache_source,
    )


def load_vocab(vocab_path: Path) -> pd.DataFrame:
    vocab = pd.read_csv(vocab_path)
    required = {"token_id", "gene_symbol"}
    if not required.issubset(vocab.columns):
        raise ValueError(f"Vocabulary missing required columns: {required - set(vocab.columns)}")
    vocab = vocab[["token_id", "gene_symbol"]].copy()
    vocab["gene_symbol"] = vocab["gene_symbol"].astype(str).str.upper().str.strip()
    vocab = vocab.sort_values("token_id").reset_index(drop=True)
    expected = np.arange(1, len(vocab) + 1)
    if not np.array_equal(vocab["token_id"].to_numpy(), expected):
        raise AssertionError("token_id must be contiguous 1..N")
    if vocab["gene_symbol"].duplicated().any():
        raise AssertionError("Duplicate gene_symbol in canonical vocabulary")
    return vocab


def detect_gtex200_manifest_candidates(shared_root: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    roots = [ROOT, shared_root]
    seen = set()
    for base in roots:
        search_root = base / "benchmarks/static_gene_embeddings/results"
        if not search_root.exists():
            continue
        for path in sorted(search_root.rglob("*manifest*.csv")):
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            needed = {"dataset", "context", "sample_id", "matrix_row"}
            if not needed.issubset(df.columns):
                continue
            subset = df[df["dataset"].astype(str).eq("GTEx")].copy()
            if len(subset) != 200:
                continue
            context_counts = subset.groupby("context").size().sort_index()
            balanced = bool(context_counts.nunique() == 1 and int(context_counts.iloc[0]) == 20)
            if subset["sample_id"].duplicated().any():
                continue
            candidates.append(
                {
                    "path": resolved,
                    "sha256": sha256(path),
                    "rows_total": int(len(df)),
                    "rows_gtex": int(len(subset)),
                    "n_tissues": int(subset["context"].nunique()),
                    "balanced_20_per_tissue": balanced,
                    "context_counts": {str(k): int(v) for k, v in context_counts.items()},
                }
            )
    return candidates


def load_model(checkpoint: Path, config: Path, num_genes: int, device: str):
    import sys

    sys.path.insert(0, str((ROOT / "src").resolve()))
    from fm_embed.model import load_expression_performer

    model, resolved_device = load_expression_performer(
        checkpoint_path=checkpoint,
        config_path=config,
        num_genes=num_genes,
        device=device,
    )
    return model, resolved_device


def cosine_per_gene(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)
    denom = np.maximum(a_norm * b_norm, 1e-12)
    return np.sum(a * b, axis=1) / denom


def pairwise_geometry_corr(a: np.ndarray, b: np.ndarray, subset_genes: int = 2000, seed: int = 42) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = a.shape[0]
    choose = min(n, subset_genes)
    idx = rng.choice(n, size=choose, replace=False)
    a_sub = a[idx]
    b_sub = b[idx]

    a_sub = a_sub / np.maximum(np.linalg.norm(a_sub, axis=1, keepdims=True), 1e-12)
    b_sub = b_sub / np.maximum(np.linalg.norm(b_sub, axis=1, keepdims=True), 1e-12)

    sim_a = a_sub @ a_sub.T
    sim_b = b_sub @ b_sub.T
    iu = np.triu_indices(choose, k=1)
    va = sim_a[iu].astype(np.float64)
    vb = sim_b[iu].astype(np.float64)
    corr = np.corrcoef(va, vb)[0, 1]
    return {
        "subset_genes": float(choose),
        "pair_count": float(len(va)),
        "pearson_corr": float(corr),
    }


def summarize_vector(x: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(x)),
        "p25": float(np.percentile(x, 25)),
        "median": float(np.median(x)),
        "mean": float(np.mean(x)),
        "p75": float(np.percentile(x, 75)),
        "max": float(np.max(x)),
    }


def save_representation(out_dir: Path, emb: np.ndarray, vocab: pd.DataFrame, provenance: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", emb)
    vocab[["token_id", "gene_symbol"]].to_csv(out_dir / "genes.tsv", sep="\t", index=False)
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate E2 GTEx-200 real-context L12 representation")
    parser.add_argument("--shared-root", type=Path, default=DEFAULT_SHARED_ROOT)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--geometry-subset-genes", type=int, default=2000)
    args = parser.parse_args()

    started = time.time()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    COVERAGE_DIR.mkdir(parents=True, exist_ok=True)
    logger = Logger(LOGS_DIR / "prepare_e2_gtex200.log")

    assets = load_assets(args.shared_root)
    logger(f"[assets] checkpoint={assets.checkpoint} ({assets.checkpoint_source})")
    logger(f"[assets] config={assets.config} ({assets.config_source})")
    logger(f"[assets] vocab={assets.vocab} ({assets.vocab_source})")
    logger(f"[assets] gtex_manifest={assets.gtex_manifest} ({assets.gtex_manifest_source})")
    logger(f"[assets] gtex_log1p_tpm={assets.gtex_log1p_tpm} ({assets.gtex_log1p_tpm_source})")

    candidates = detect_gtex200_manifest_candidates(args.shared_root)
    balanced_candidates = [c for c in candidates if c["balanced_20_per_tissue"]]
    candidate_report = {
        "all_candidates": candidates,
        "balanced_gtex200_candidates": balanced_candidates,
        "selected_manifest": str(assets.gtex_manifest.resolve()),
    }
    (COVERAGE_DIR / "gtex200_manifest_candidates.json").write_text(
        json.dumps(candidate_report, indent=2) + "\n", encoding="utf-8"
    )

    distinct_balanced_hashes = {c["sha256"] for c in balanced_candidates}
    if len(distinct_balanced_hashes) > 1:
        raise RuntimeError(
            "Multiple distinct tissue-balanced GTEx-200 manifests detected. "
            "Inspect coverage/gtex200_manifest_candidates.json and disambiguate before proceeding."
        )

    vocab = load_vocab(assets.vocab)
    num_genes = len(vocab)

    manifest = pd.read_csv(assets.gtex_manifest)
    required = {"analysis_index", "dataset", "context", "sample_id", "matrix_row"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Manifest missing columns: {required - set(manifest.columns)}")
    gtex_manifest = manifest[manifest["dataset"].astype(str).eq("GTEx")].copy().reset_index(drop=True)
    if len(gtex_manifest) != 200:
        raise AssertionError(f"Expected 200 GTEx samples, found {len(gtex_manifest)}")
    tissue_counts = gtex_manifest.groupby("context").size().sort_index()
    if tissue_counts.nunique() != 1 or int(tissue_counts.iloc[0]) != 20:
        raise AssertionError("GTEx panel is not tissue-balanced 10x20")
    if gtex_manifest["sample_id"].duplicated().any():
        raise AssertionError("Duplicate GTEx sample IDs in manifest")

    x = np.load(assets.gtex_log1p_tpm, mmap_mode="r")
    if x.ndim != 2 or x.shape[1] != num_genes:
        raise AssertionError(f"Unexpected GTEx expression matrix shape: {x.shape}, expected (*, {num_genes})")

    model, device = load_model(assets.checkpoint, assets.config, num_genes, args.device)
    model.eval()

    ids = torch.arange(num_genes, device=device)
    running_sum = np.zeros((num_genes, 512), dtype=np.float64)
    processed_rows: list[dict[str, Any]] = []
    sample_ids_processed: list[str] = []

    loop_started = time.time()
    for idx, row in enumerate(gtex_manifest.itertuples(index=False), start=1):
        matrix_row = int(row.matrix_row)
        sample_id = str(row.sample_id)
        context = str(row.context)

        expr = np.array(x[matrix_row], dtype=np.float32, copy=True)
        sample_ids_processed.append(sample_id)
        with torch.no_grad():
            expr_t = torch.as_tensor(expr[None, :], dtype=torch.float32, device=device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                hidden = model._encode_hidden(expr_t)
            l12 = hidden[0].detach().float().cpu().numpy().astype(np.float32)

        if l12.shape != (num_genes, 512):
            raise AssertionError(f"Unexpected hidden shape {l12.shape} for sample {sample_id}")
        if not np.isfinite(l12).all():
            raise AssertionError(f"Non-finite values in hidden state for sample {sample_id}")

        running_sum += l12
        processed_rows.append(
            {
                "analysis_index": int(row.analysis_index),
                "dataset": "GTEx",
                "context": context,
                "sample_id": sample_id,
                "matrix_row": matrix_row,
                "processed_order": idx,
            }
        )

        if idx % 25 == 0 or idx == len(gtex_manifest):
            elapsed = time.time() - loop_started
            rate = idx / max(elapsed, 1e-9)
            eta = (len(gtex_manifest) - idx) / max(rate, 1e-9)
            logger(
                f"[E2] processed={idx}/{len(gtex_manifest)} elapsed={elapsed/60:.1f}m "
                f"rate={rate:.2f} samples/s eta={eta/60:.1f}m"
            )

    if len(processed_rows) != 200:
        raise AssertionError(f"Processed {len(processed_rows)} samples, expected 200")

    e2 = (running_sum / 200.0).astype(np.float32)
    if e2.shape != (num_genes, 512):
        raise AssertionError(f"E2 shape mismatch: {e2.shape}")
    if not np.isfinite(e2).all():
        raise AssertionError("E2 contains non-finite values")

    processed_df = pd.DataFrame(processed_rows).sort_values("processed_order").reset_index(drop=True)
    if set(processed_df["sample_id"]) != set(gtex_manifest["sample_id"]):
        raise AssertionError("Processed sample IDs do not match target GTEx-200 manifest")

    e0 = np.load(REP_DIR / "E0_static" / "embeddings.npy")
    e1 = np.load(REP_DIR / "E1_null_L12" / "embeddings.npy")
    if e0.shape != e2.shape or e1.shape != e2.shape:
        raise AssertionError(f"Shape mismatch among E0/E1/E2: {e0.shape}, {e1.shape}, {e2.shape}")

    e0_e1_cos = cosine_per_gene(e0, e1)
    e0_e2_cos = cosine_per_gene(e0, e2)
    e1_e2_cos = cosine_per_gene(e1, e2)

    geom_e0_e1 = pairwise_geometry_corr(e0, e1, subset_genes=args.geometry_subset_genes, seed=args.seed)
    geom_e0_e2 = pairwise_geometry_corr(e0, e2, subset_genes=args.geometry_subset_genes, seed=args.seed)
    geom_e1_e2 = pairwise_geometry_corr(e1, e2, subset_genes=args.geometry_subset_genes, seed=args.seed)

    e0_var = np.var(e0, axis=0)
    e1_var = np.var(e1, axis=0)
    e2_var = np.var(e2, axis=0)

    if assets.shared_gtex_l12_cache.exists():
        shared_cache = np.load(assets.shared_gtex_l12_cache)
        cache_cmp = {
            "path": str(assets.shared_gtex_l12_cache),
            "source": assets.shared_gtex_l12_cache_source,
            "shape": list(shared_cache.shape),
            "shape_match": bool(shared_cache.shape == e2.shape),
        }
        if shared_cache.shape == e2.shape:
            diff = np.abs(shared_cache.astype(np.float32) - e2)
            cache_cmp.update(
                {
                    "max_abs_diff": float(diff.max()),
                    "mean_abs_diff": float(diff.mean()),
                    "allclose_atol_1e-6": bool(np.allclose(shared_cache, e2, atol=1e-6, rtol=0.0)),
                }
            )
    else:
        cache_cmp = {
            "path": str(assets.shared_gtex_l12_cache),
            "source": assets.shared_gtex_l12_cache_source,
            "exists": False,
        }

    rep_dir = REP_DIR / "E2_GTEx200_L12"
    rep_dir.mkdir(parents=True, exist_ok=True)
    np.save(rep_dir / "embeddings.npy", e2)
    vocab[["token_id", "gene_symbol"]].to_csv(rep_dir / "genes.tsv", sep="\t", index=False)
    processed_df.to_csv(rep_dir / "gtex200_samples.tsv", sep="\t", index=False)

    manifest_provenance = json.loads(assets.gtex_manifest_provenance.read_text())
    e2_provenance = {
        "representation": "E2_GTEx200_L12",
        "definition": "Mean final transformer hidden state (L12) per gene across frozen GTEx-200 tissue-balanced panel",
        "is_pre_transformer": False,
        "is_post_transformer": True,
        "is_sample_contextual": True,
        "shape": [int(e2.shape[0]), int(e2.shape[1])],
        "checkpoint": str(assets.checkpoint),
        "checkpoint_sha256": sha256(assets.checkpoint),
        "checkpoint_source": assets.checkpoint_source,
        "config": str(assets.config),
        "config_sha256": sha256(assets.config),
        "vocabulary": str(assets.vocab),
        "vocabulary_sha256": sha256(assets.vocab),
        "vocabulary_source": assets.vocab_source,
        "gene_order": "token_id ascending; row_index = token_id - 1",
        "input_expression_matrix": str(assets.gtex_log1p_tpm),
        "input_expression_matrix_source": assets.gtex_log1p_tpm_source,
        "input_expression_contract": {
            "expected_input": "log1p(TPM)",
            "mask_token_value": -10.0,
            "species_embedding_used": False,
            "normalization": "none inside model; upstream expression matrix already log1p(TPM)",
            "clipping": "none in inference path",
        },
        "panel_manifest": str(assets.gtex_manifest),
        "panel_manifest_sha256": sha256(assets.gtex_manifest),
        "panel_manifest_source": assets.gtex_manifest_source,
        "panel_provenance": manifest_provenance,
        "panel_samples": {
            "total": int(len(processed_df)),
            "n_tissues": int(processed_df["context"].nunique()),
            "samples_per_tissue": {str(k): int(v) for k, v in processed_df.groupby("context").size().items()},
            "sample_ids": processed_df["sample_id"].tolist(),
        },
        "extraction": {
            "model_call": "model._encode_hidden(expr)",
            "output_layer": "final transformer hidden state after layer 12",
            "output_map_used": False,
            "vector_normalization_before_aggregation": "none",
            "centering_before_aggregation": "none",
            "pca_before_aggregation": "none",
            "aggregation": "mean across 200 GTEx samples",
        },
        "eval_mode": True,
        "gradients_enabled": False,
        "parameter_updates": False,
        "deterministic_note": "Deterministic for fixed checkpoint/input/model implementation on same backend",
        "comparison_to_shared_contextual_cache": cache_cmp,
        "related_method_scripts": {
            "matched_script": str(assets.matched_script),
            "contextual_modules_script": str(assets.contextual_modules_script),
        },
        "created_utc": pd.Timestamp.utcnow().isoformat(),
    }
    (rep_dir / "provenance.json").write_text(json.dumps(e2_provenance, indent=2) + "\n", encoding="utf-8")

    qc_gene_cos = pd.DataFrame(
        {
            "token_id": vocab["token_id"].to_numpy(),
            "gene_symbol": vocab["gene_symbol"].to_numpy(),
            "cosine_e0_e1": e0_e1_cos,
            "cosine_e0_e2": e0_e2_cos,
            "cosine_e1_e2": e1_e2_cos,
        }
    )
    qc_gene_cos.to_csv(COVERAGE_DIR / "e0_e1_e2_gene_cosines.csv", index=False)

    qc_summary = {
        "status": "complete",
        "elapsed_seconds": time.time() - started,
        "representation_shapes": {
            "E0": list(e0.shape),
            "E1": list(e1.shape),
            "E2": list(e2.shape),
        },
        "finite_checks": {
            "E0": bool(np.isfinite(e0).all()),
            "E1": bool(np.isfinite(e1).all()),
            "E2": bool(np.isfinite(e2).all()),
        },
        "sample_processing": {
            "target_samples": 200,
            "processed_samples": int(len(processed_df)),
            "silently_dropped_samples": int(200 - len(processed_df)),
            "all_target_samples_processed": bool(len(processed_df) == 200),
        },
        "cosine_distributions": {
            "E0_vs_E1": summarize_vector(e0_e1_cos),
            "E0_vs_E2": summarize_vector(e0_e2_cos),
            "E1_vs_E2": summarize_vector(e1_e2_cos),
        },
        "pairwise_geometry_correlation": {
            "E0_vs_E1": geom_e0_e1,
            "E0_vs_E2": geom_e0_e2,
            "E1_vs_E2": geom_e1_e2,
        },
        "variance_across_genes": {
            "E0_dimension_variance": summarize_vector(e0_var),
            "E1_dimension_variance": summarize_vector(e1_var),
            "E2_dimension_variance": summarize_vector(e2_var),
            "E0_mean_dimension_variance": float(e0_var.mean()),
            "E1_mean_dimension_variance": float(e1_var.mean()),
            "E2_mean_dimension_variance": float(e2_var.mean()),
        },
        "outputs": {
            "e2_dir": str(rep_dir.resolve()),
            "gene_cosines_csv": str((COVERAGE_DIR / "e0_e1_e2_gene_cosines.csv").resolve()),
        },
    }
    (COVERAGE_DIR / "representation_qc_e0_e1_e2.json").write_text(
        json.dumps(qc_summary, indent=2) + "\n", encoding="utf-8"
    )

    logger("[done] E2 generated and validated")
    logger(f"[done] wrote {rep_dir / 'embeddings.npy'}")
    logger(f"[done] wrote {COVERAGE_DIR / 'representation_qc_e0_e1_e2.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
