#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
STATIC_CACHE_REL = Path("benchmarks/static_gene_embeddings/work/static_gene_embedding.float32.npy")
ARCHS4_LOG1P_TPM_MMAP_REL = Path(
    "benchmarks/cross_species_exercise_response/work/hallmark_readout/archs4_log1p_tpm.float32.mmap"
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
    static_cache: Path
    static_cache_source: str
    archs4_mmap: Path
    archs4_mmap_source: str


def load_assets(shared_root: Path) -> Assets:
    checkpoint, checkpoint_source = resolve_prefer_shared(CHECKPOINT_REL, shared_root)
    config, config_source = resolve_prefer_shared(CONFIG_REL, shared_root)
    vocab, vocab_source = resolve_prefer_shared(VOCAB_REL, shared_root)
    static_cache, static_cache_source = resolve_prefer_shared(STATIC_CACHE_REL, shared_root)
    archs4_mmap, archs4_mmap_source = resolve_prefer_shared(ARCHS4_LOG1P_TPM_MMAP_REL, shared_root)

    missing = [
        name
        for name, path in [
            ("checkpoint", checkpoint),
            ("config", config),
            ("vocab", vocab),
            ("static_cache", static_cache),
        ]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing required assets: {missing}")

    return Assets(
        checkpoint=checkpoint,
        checkpoint_source=checkpoint_source,
        config=config,
        config_source=config_source,
        vocab=vocab,
        vocab_source=vocab_source,
        static_cache=static_cache,
        static_cache_source=static_cache_source,
        archs4_mmap=archs4_mmap,
        archs4_mmap_source=archs4_mmap_source,
    )


def load_vocab(vocab_path: Path) -> pd.DataFrame:
    vocab = pd.read_csv(vocab_path)
    required = {"token_id", "gene_symbol"}
    if not required.issubset(vocab.columns):
        raise ValueError(f"Vocab missing columns: {required - set(vocab.columns)}")
    vocab = vocab[["token_id", "gene_symbol"]].copy()
    vocab["gene_symbol"] = vocab["gene_symbol"].astype(str).str.upper().str.strip()
    vocab = vocab.sort_values("token_id").reset_index(drop=True)
    expected = np.arange(1, len(vocab) + 1)
    if not np.array_equal(vocab["token_id"].to_numpy(), expected):
        raise AssertionError("Vocabulary token_id must be contiguous 1..N")
    if vocab["gene_symbol"].duplicated().any():
        raise AssertionError("Duplicate gene_symbol in vocabulary")
    return vocab


def load_checkpoint_state(checkpoint: Path) -> dict[str, Any]:
    return torch.load(checkpoint, map_location="cpu", weights_only=False)


def extract_e0_from_checkpoint(payload: dict[str, Any]) -> np.ndarray:
    arr = payload["model_state_dict"]["gene_embedding.weight"].detach().cpu().numpy().astype(np.float32)
    return arr


def compare_e0_to_static_cache(e0: np.ndarray, cache_path: Path) -> dict[str, Any]:
    cache = np.load(cache_path)
    if cache.shape != e0.shape:
        return {
            "cache_shape": list(cache.shape),
            "checkpoint_shape": list(e0.shape),
            "shape_match": False,
            "allclose": False,
        }
    diff = np.abs(e0 - cache)
    return {
        "cache_shape": list(cache.shape),
        "checkpoint_shape": list(e0.shape),
        "shape_match": True,
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "allclose": bool(np.allclose(e0, cache, rtol=0.0, atol=0.0)),
    }


def estimate_input_distribution_constants(mmap_path: Path, genes: int, sample_size_values: int, seed: int = 42) -> dict[str, float]:
    if not mmap_path.exists():
        return {
            "global_min": float("nan"),
            "global_max": float("nan"),
            "sample_mean": float("nan"),
            "sample_median": float("nan"),
            "sample_p25": float("nan"),
            "sample_p75": float("nan"),
            "samples": 0.0,
        }

    total_values = mmap_path.stat().st_size // 4
    if total_values % genes != 0:
        raise ValueError(f"Unexpected mmap size for gene width {genes}: {mmap_path}")
    samples = total_values // genes

    arr = np.memmap(mmap_path, dtype=np.float32, mode="r", shape=(samples, genes))
    rng = np.random.default_rng(seed)
    flat_idx = rng.integers(0, total_values, size=min(sample_size_values, total_values), endpoint=False)
    rows = flat_idx // genes
    cols = flat_idx % genes
    sample_values = arr[rows, cols].astype(np.float64)

    return {
        "global_min": float(np.min(sample_values)),
        "global_max": float(np.max(sample_values)),
        "sample_mean": float(np.mean(sample_values)),
        "sample_median": float(np.median(sample_values)),
        "sample_p25": float(np.percentile(sample_values, 25)),
        "sample_p75": float(np.percentile(sample_values, 75)),
        "samples": float(samples),
    }


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


def extract_l12_null(model, device: torch.device, num_genes: int, c_value: float) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.full((1, num_genes), float(c_value), dtype=torch.float32, device=device)
        hidden = model._encode_hidden(x)  # [1, genes, dim]
        out = hidden[0].detach().cpu().numpy().astype(np.float32)
    return out


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
    return {"subset_genes": float(choose), "pair_count": float(len(va)), "pearson_corr": float(corr)}


def save_representation(out_dir: Path, emb: np.ndarray, vocab: pd.DataFrame, provenance: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", emb)
    vocab[["token_id", "gene_symbol"]].to_csv(out_dir / "genes.tsv", sep="\t", index=False)
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


def summarize_vector(x: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(x)),
        "p25": float(np.percentile(x, 25)),
        "median": float(np.median(x)),
        "mean": float(np.mean(x)),
        "p75": float(np.percentile(x, 75)),
        "max": float(np.max(x)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and validate E0/E1 representations for Zhong ablation")
    parser.add_argument("--shared-root", type=Path, default=DEFAULT_SHARED_ROOT)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size-values", type=int, default=5_000_000)
    parser.add_argument(
        "--selected-null",
        choices=["zero", "one", "empirical_median", "log1p_one_tpm"],
        default="zero",
        help="Which candidate null constant to save as E1",
    )
    args = parser.parse_args()

    started = time.time()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = Logger(LOGS_DIR / "representation_ablation.log")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    assets = load_assets(args.shared_root)
    vocab = load_vocab(assets.vocab)
    n_genes = len(vocab)

    logger(f"[assets] checkpoint={assets.checkpoint} ({assets.checkpoint_source})")
    logger(f"[assets] config={assets.config} ({assets.config_source})")
    logger(f"[assets] vocab={assets.vocab} ({assets.vocab_source})")
    logger(f"[assets] static_cache={assets.static_cache} ({assets.static_cache_source})")

    payload = load_checkpoint_state(assets.checkpoint)
    e0 = extract_e0_from_checkpoint(payload)
    if e0.shape[0] != n_genes:
        raise AssertionError(f"E0 rows {e0.shape[0]} != vocab genes {n_genes}")
    if not np.isfinite(e0).all():
        raise AssertionError("E0 has non-finite values")

    e0_cache_cmp = compare_e0_to_static_cache(e0, assets.static_cache)
    logger(f"[E0] shape={e0.shape}, cache_allclose={e0_cache_cmp.get('allclose', False)}")

    dist_stats = estimate_input_distribution_constants(
        assets.archs4_mmap,
        genes=n_genes,
        sample_size_values=args.sample_size_values,
        seed=args.seed,
    )
    logger(
        "[null] sampled input distribution "
        f"median={dist_stats['sample_median']:.6f}, mean={dist_stats['sample_mean']:.6f}, "
        f"p25={dist_stats['sample_p25']:.6f}, p75={dist_stats['sample_p75']:.6f}"
    )

    candidate_constants = {
        "zero": 0.0,
        "one": 1.0,
        "log1p_one_tpm": float(np.log1p(1.0)),
        "empirical_median": float(dist_stats["sample_median"]),
    }

    model, device = load_model(assets.checkpoint, assets.config, n_genes, args.device)
    model.eval()
    if model.training:
        raise AssertionError("Model is not in eval mode")

    candidate_qc_rows: list[dict[str, Any]] = []
    selected_name = args.selected_null
    selected_c = candidate_constants[selected_name]
    e1_selected = None

    for name, c in candidate_constants.items():
        e1 = extract_l12_null(model, device, n_genes, c)
        if not np.isfinite(e1).all():
            raise AssertionError(f"E1 candidate {name} has non-finite values")

        cos = cosine_per_gene(e0, e1)
        geom = pairwise_geometry_corr(e0, e1, subset_genes=2000, seed=args.seed)

        candidate_qc_rows.append(
            {
                "candidate": name,
                "c_value": float(c),
                "e1_shape_rows": int(e1.shape[0]),
                "e1_shape_dim": int(e1.shape[1]),
                "cosine_median": float(np.median(cos)),
                "cosine_mean": float(np.mean(cos)),
                "cosine_p25": float(np.percentile(cos, 25)),
                "cosine_p75": float(np.percentile(cos, 75)),
                "e1_gene_variance_mean": float(np.var(e1, axis=0).mean()),
                "geometry_pearson": float(geom["pearson_corr"]),
            }
        )

        if name == selected_name:
            e1_selected = e1

    if e1_selected is None:
        raise AssertionError("Selected E1 candidate was not generated")

    cos_selected = cosine_per_gene(e0, e1_selected)
    geom_selected = pairwise_geometry_corr(e0, e1_selected, subset_genes=2000, seed=args.seed)

    if e1_selected.shape != e0.shape:
        raise AssertionError(f"E1 shape mismatch: {e1_selected.shape} vs {e0.shape}")

    e0_var_dim = np.var(e0, axis=0)
    e1_var_dim = np.var(e1_selected, axis=0)

    e0_dir = REP_DIR / "E0_static"
    e1_dir = REP_DIR / "E1_null_L12"

    e0_prov = {
        "representation": "E0_static",
        "definition": "model_state_dict['gene_embedding.weight']",
        "is_pre_transformer": True,
        "is_post_transformer": False,
        "is_sample_contextual": False,
        "checkpoint": str(assets.checkpoint),
        "checkpoint_sha256": sha256(assets.checkpoint),
        "checkpoint_source": assets.checkpoint_source,
        "checkpoint_tensor": "model_state_dict.gene_embedding.weight",
        "shape": [int(e0.shape[0]), int(e0.shape[1])],
        "vocabulary": str(assets.vocab),
        "vocabulary_sha256": sha256(assets.vocab),
        "vocabulary_source": assets.vocab_source,
        "gene_order": "token_id ascending; row_index = token_id - 1",
        "static_cache_comparison": e0_cache_cmp,
        "expression_contribution": "none (direct learned parameter lookup)",
        "created_utc": pd.Timestamp.utcnow().isoformat(),
    }

    e1_prov = {
        "representation": "E1_null_L12",
        "definition": "final transformer hidden state per gene with uniform expression input",
        "is_pre_transformer": False,
        "is_post_transformer": True,
        "is_sample_contextual": False,
        "selected_null_candidate": selected_name,
        "selected_null_value": float(selected_c),
        "null_candidates": candidate_constants,
        "model_input_contract": {
            "expected_input": "log1p(TPM)",
            "mask_token_value": -10.0,
            "expression_embedding": "RotaryExpressionEmbedding: concat(sin(x*inv_freq), cos(x*inv_freq))",
            "gene_token": "gene_embedding + expression_rotary_embedding",
            "species_embedding_used": False,
            "dropout_used_in_inference_model": False,
        },
        "checkpoint": str(assets.checkpoint),
        "checkpoint_sha256": sha256(assets.checkpoint),
        "checkpoint_source": assets.checkpoint_source,
        "config": str(assets.config),
        "config_sha256": sha256(assets.config),
        "shape": [int(e1_selected.shape[0]), int(e1_selected.shape[1])],
        "vocabulary": str(assets.vocab),
        "vocabulary_sha256": sha256(assets.vocab),
        "vocabulary_source": assets.vocab_source,
        "gene_order": "token_id ascending; row_index = token_id - 1",
        "eval_mode": True,
        "gradients_enabled": False,
        "deterministic_note": "deterministic for fixed checkpoint/model implementation/input constant on same hardware/backend",
        "created_utc": pd.Timestamp.utcnow().isoformat(),
    }

    save_representation(e0_dir, e0, vocab, e0_prov)
    save_representation(e1_dir, e1_selected, vocab, e1_prov)

    qc_dir = COVERAGE_DIR
    qc_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(candidate_qc_rows).to_csv(qc_dir / "e1_null_candidates_qc.csv", index=False)
    pd.DataFrame(
        {
            "token_id": vocab["token_id"].to_numpy(),
            "gene_symbol": vocab["gene_symbol"].to_numpy(),
            "cosine_e0_e1": cos_selected,
        }
    ).to_csv(qc_dir / "e0_e1_gene_cosine.csv", index=False)

    summary = {
        "status": "complete",
        "elapsed_seconds": time.time() - started,
        "selected_null_candidate": selected_name,
        "selected_null_value": float(selected_c),
        "input_distribution_sample": dist_stats,
        "e0_shape": list(e0.shape),
        "e1_shape": list(e1_selected.shape),
        "e0_finite": bool(np.isfinite(e0).all()),
        "e1_finite": bool(np.isfinite(e1_selected).all()),
        "cosine_e0_e1_summary": summarize_vector(cos_selected),
        "pairwise_geometry_correlation": geom_selected,
        "variance_across_genes": {
            "e0_dimension_variance_summary": summarize_vector(e0_var_dim),
            "e1_dimension_variance_summary": summarize_vector(e1_var_dim),
            "e0_mean_dimension_variance": float(e0_var_dim.mean()),
            "e1_mean_dimension_variance": float(e1_var_dim.mean()),
        },
        "candidate_qc_csv": str((qc_dir / "e1_null_candidates_qc.csv").resolve()),
        "gene_cosine_csv": str((qc_dir / "e0_e1_gene_cosine.csv").resolve()),
        "e0_provenance": str((e0_dir / "provenance.json").resolve()),
        "e1_provenance": str((e1_dir / "provenance.json").resolve()),
    }

    (qc_dir / "representation_ablation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    logger(f"[done] selected_null={selected_name} value={selected_c:.6f}")
    logger(f"[done] wrote {(qc_dir / 'representation_ablation_summary.json').resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
