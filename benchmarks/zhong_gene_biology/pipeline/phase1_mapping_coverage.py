#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import pickle
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "benchmarks" / "zhong_gene_biology"
REFERENCES = BENCH / "references"
COVERAGE_DIR = BENCH / "coverage"
LOGS_DIR = BENCH / "logs"
DEFAULT_SHARED_ROOT = Path("/home/walt/bridge-rna")

BRIDGE_CHECKPOINT_REL = Path("model/r7hnr92k/best_model.pt")
BRIDGE_CONFIG_REL = Path("model/r7hnr92k/config.json")
BRIDGE_VOCAB_REL = Path("data/ensembl/canonical_genes.csv")
BRIDGE_STATIC_CACHE_REL_CANDIDATES = [
    Path("benchmarks/zhong_gene_biology/work/static_gene_embedding.float32.npy"),
    Path("benchmarks/static_gene_embeddings/work/static_gene_embedding.float32.npy"),
    Path("benchmarks/static_gene_embeddings/results/static_embeddings/static_gene_embedding.float32.npy"),
]

ZHONG_REPO_URL = "https://github.com/ylaboratory/gene-embedding-benchmarks"
ZHONG_REPO_LOCAL = REFERENCES / "gene-embedding-benchmarks"

NCBI_GENE_INFO_URL = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz"
NCBI_GENE_INFO_LOCAL = REFERENCES / "Homo_sapiens.gene_info.gz"
COGNOMA_GENES_URL = "https://raw.githubusercontent.com/cognoma/genes/master/data/genes.tsv"
COGNOMA_GENES_LOCAL = REFERENCES / "cognoma_genes.tsv"

MIN_RETAINED_POSITIVE = 20


def resolve_asset(relative_path: Path, shared_root: Path | None = None) -> tuple[Path, str]:
    local_path = ROOT / relative_path
    if local_path.exists():
        return local_path, "worktree"
    if shared_root is not None:
        shared_path = shared_root / relative_path
        if shared_path.exists():
            return shared_path, "shared_root"
    return local_path, "missing"


def resolve_asset_prefer_shared(relative_path: Path, shared_root: Path | None = None) -> tuple[Path, str]:
    if shared_root is not None:
        shared_path = shared_root / relative_path
        if shared_path.exists():
            return shared_path, "shared_root"
    return resolve_asset(relative_path, shared_root)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class TeeLogger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = path.open("w", encoding="utf-8")

    def log(self, msg: str) -> None:
        line = msg.rstrip("\n")
        print(line, flush=True)
        self._f.write(line + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


@dataclass
class ZhongPaths:
    go_cv1: Path
    go_cv2: Path
    go_cv3: Path
    go_holdout: Path
    omim_cv1: Path
    omim_cv2: Path
    omim_cv3: Path
    omim_holdout: Path
    pair_sl: Path
    pair_ng: Path
    pair_tf: Path
    go_obo: Path
    doid_obo: Path


def ensure_zhong_repo(log: TeeLogger, force_refresh: bool = False) -> Path:
    if force_refresh and ZHONG_REPO_LOCAL.exists():
        log.log(f"[setup] Removing existing Zhong repo at {ZHONG_REPO_LOCAL}")
        subprocess.run(["rm", "-rf", str(ZHONG_REPO_LOCAL)], check=True)
    if not ZHONG_REPO_LOCAL.exists():
        REFERENCES.mkdir(parents=True, exist_ok=True)
        log.log(f"[setup] Cloning Zhong repository to {ZHONG_REPO_LOCAL}")
        subprocess.run(
            ["git", "clone", "--depth", "1", ZHONG_REPO_URL, str(ZHONG_REPO_LOCAL)],
            check=True,
        )
    else:
        log.log(f"[setup] Reusing local Zhong repository: {ZHONG_REPO_LOCAL}")
    return ZHONG_REPO_LOCAL


def zhong_paths(repo_root: Path) -> ZhongPaths:
    data = repo_root / "data"
    return ZhongPaths(
        go_cv1=data / "data_splits/gene_level_benchmark/go_folds_splits/go_cv_fold1_dict_all.pkl",
        go_cv2=data / "data_splits/gene_level_benchmark/go_folds_splits/go_cv_fold2_dict_all.pkl",
        go_cv3=data / "data_splits/gene_level_benchmark/go_folds_splits/go_cv_fold3_dict_all.pkl",
        go_holdout=data / "data_splits/gene_level_benchmark/go_folds_splits/go_holdout_dict_all.pkl",
        omim_cv1=data / "data_splits/gene_level_benchmark/omim_folds_splits/omim_cv_fold1_dict_all.pkl",
        omim_cv2=data / "data_splits/gene_level_benchmark/omim_folds_splits/omim_cv_fold2_dict_all.pkl",
        omim_cv3=data / "data_splits/gene_level_benchmark/omim_folds_splits/omim_cv_fold3_dict_all.pkl",
        omim_holdout=data / "data_splits/gene_level_benchmark/omim_folds_splits/omim_holdout_dict_all.pkl",
        pair_sl=data / "data_splits/gene_pair_benchmark/sl_nested_cv_splits.pkl",
        pair_ng=data / "data_splits/gene_pair_benchmark/ng_nested_cv_splits.pkl",
        pair_tf=data / "data_splits/gene_pair_benchmark/tf_nested_cv_splits.pkl",
        go_obo=data / "obo/go.obo",
        doid_obo=data / "obo/doid.obo",
    )


def load_vocab(vocab_path: Path) -> pd.DataFrame:
    vocab = pd.read_csv(vocab_path)
    required = {"token_id", "gene_symbol"}
    if not required.issubset(set(vocab.columns)):
        raise ValueError(f"Vocabulary missing required columns: {required - set(vocab.columns)}")
    vocab = vocab[["token_id", "gene_symbol"]].copy()
    vocab["gene_symbol"] = vocab["gene_symbol"].astype(str).str.upper().str.strip()
    vocab = vocab.sort_values("token_id").reset_index(drop=True)
    expected = np.arange(1, len(vocab) + 1)
    if not np.array_equal(vocab["token_id"].to_numpy(), expected):
        raise AssertionError("token_id is not contiguous 1..N; embedding row order cannot be guaranteed")
    if vocab["gene_symbol"].duplicated().any():
        raise AssertionError("Duplicate gene_symbol entries in canonical Bridge vocabulary")
    return vocab


def resolve_static_embedding_shape(
    checkpoint_path: Path,
    config_path: Path,
    static_provenance_path: Path,
    log: TeeLogger,
) -> tuple[list[int] | None, str]:
    if checkpoint_path.exists():
        import torch

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        shape = list(payload["model_state_dict"]["gene_embedding.weight"].shape)
        return shape, "checkpoint_tensor"

    if static_provenance_path.exists():
        data = json.loads(static_provenance_path.read_text())
        shape = data.get("embedding_shape")
        if isinstance(shape, list) and len(shape) == 2:
            return [int(shape[0]), int(shape[1])], "static_embedding_provenance"

    if config_path.exists():
        cfg = json.loads(config_path.read_text())
        n = cfg.get("num_genes")
        d = cfg.get("hidden_dim")
        if n is not None and d is not None:
            return [int(n), int(d)], "model_config"

    log.log("[phase0] Could not resolve static embedding shape from checkpoint/provenance/config")
    return None, "unknown"


def find_existing_embedding_caches(shared_root: Path | None = None) -> list[str]:
    found: list[str] = []
    for rel in BRIDGE_STATIC_CACHE_REL_CANDIDATES:
        candidate = ROOT / rel
        if candidate.exists():
            found.append(str(candidate))
        if shared_root is not None:
            shared_candidate = shared_root / rel
            if shared_candidate.exists():
                found.append(str(shared_candidate))
    for path in ROOT.glob("benchmarks/**/work/*.npy"):
        if "gene_embedding" in path.name:
            found.append(str(path))
    if shared_root is not None:
        for path in shared_root.glob("benchmarks/**/work/*.npy"):
            if "gene_embedding" in path.name:
                found.append(str(path))
    return sorted(set(found))


def parse_obo_names(path: Path) -> dict[str, str]:
    names: dict[str, str] = {}
    current_id: str | None = None
    current_name: str | None = None
    in_term = False
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line == "[Term]":
                if current_id and current_name:
                    names[current_id] = current_name
                current_id, current_name = None, None
                in_term = True
                continue
            if not in_term:
                continue
            if line.startswith("id: "):
                current_id = line[4:].strip()
            elif line.startswith("name: "):
                current_name = line[6:].strip()
            elif line == "":
                if current_id and current_name:
                    names[current_id] = current_name
                current_id, current_name = None, None
                in_term = False
    if current_id and current_name:
        names[current_id] = current_name
    return names


def ensure_ncbi_gene_info(log: TeeLogger, force_refresh: bool = False) -> Path:
    REFERENCES.mkdir(parents=True, exist_ok=True)
    if force_refresh and NCBI_GENE_INFO_LOCAL.exists():
        NCBI_GENE_INFO_LOCAL.unlink()
    if not NCBI_GENE_INFO_LOCAL.exists():
        log.log(f"[setup] Downloading NCBI gene_info: {NCBI_GENE_INFO_URL}")
        urlretrieve(NCBI_GENE_INFO_URL, NCBI_GENE_INFO_LOCAL)
    else:
        log.log(f"[setup] Reusing NCBI gene_info: {NCBI_GENE_INFO_LOCAL}")
    return NCBI_GENE_INFO_LOCAL


def ensure_cognoma_genes(log: TeeLogger, force_refresh: bool = False) -> Path:
    REFERENCES.mkdir(parents=True, exist_ok=True)
    if force_refresh and COGNOMA_GENES_LOCAL.exists():
        COGNOMA_GENES_LOCAL.unlink()
    if not COGNOMA_GENES_LOCAL.exists():
        log.log(f"[setup] Downloading fallback symbol map: {COGNOMA_GENES_URL}")
        urlretrieve(COGNOMA_GENES_URL, COGNOMA_GENES_LOCAL)
    else:
        log.log(f"[setup] Reusing fallback symbol map: {COGNOMA_GENES_LOCAL}")
    return COGNOMA_GENES_LOCAL


def ensure_symbol_mapping_resource(log: TeeLogger, force_refresh: bool = False) -> tuple[Path, str]:
    try:
        path = ensure_ncbi_gene_info(log, force_refresh=force_refresh)
        return path, "ncbi_gene_info"
    except Exception as exc:
        log.log(f"[setup] NCBI gene_info unavailable ({exc}); falling back to cognoma/genes")
        path = ensure_cognoma_genes(log, force_refresh=force_refresh)
        return path, "cognoma_genes"


def build_symbol_to_entrez_maps(gene_info_gz: Path) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    official: dict[str, set[str]] = defaultdict(set)
    aliases: dict[str, set[str]] = defaultdict(set)
    with gzip.open(gene_info_gz, "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("#tax_id") != "9606":
                continue
            gene_id = (row.get("GeneID") or "").strip()
            symbol = (row.get("Symbol") or "").strip().upper()
            synonyms = (row.get("Synonyms") or "").strip()
            if not gene_id or not symbol:
                continue
            official[symbol].add(gene_id)
            if synonyms and synonyms != "-":
                for raw in synonyms.split("|"):
                    alias = raw.strip().upper()
                    if alias and alias != "-":
                        aliases[alias].add(gene_id)
    return dict(official), dict(aliases)


def build_symbol_to_entrez_maps_from_cognoma(path: Path) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    official: dict[str, set[str]] = defaultdict(set)
    aliases: dict[str, set[str]] = defaultdict(set)
    frame = pd.read_csv(path, sep="\t", dtype=str)
    for row in frame.itertuples(index=False):
        entrez = str(row.entrez_gene_id).strip()
        symbol = str(row.symbol).strip().upper()
        if not entrez or not symbol or entrez == "nan" or symbol == "NAN":
            continue
        official[symbol].add(entrez)
        for col in ("synonyms", "aliases"):
            value = getattr(row, col)
            if value is None or value == "nan":
                continue
            for raw in str(value).split("|"):
                alias = raw.strip().upper()
                if alias and alias != "-":
                    aliases[alias].add(entrez)
    return dict(official), dict(aliases)


def map_bridge_to_entrez(vocab: pd.DataFrame, official: dict[str, set[str]], aliases: dict[str, set[str]]) -> pd.DataFrame:
    rows = []
    for row in vocab.itertuples(index=False):
        token_id = int(row.token_id)
        symbol = str(row.gene_symbol).upper().strip()
        official_candidates = sorted(official.get(symbol, set()))
        alias_candidates = sorted(aliases.get(symbol, set()))

        mapped_entrez = ""
        status = "unmapped"
        source = ""

        if len(official_candidates) == 1:
            mapped_entrez = official_candidates[0]
            status = "mapped"
            source = "official_symbol"
        elif len(official_candidates) > 1:
            status = "ambiguous_official_symbol"
            source = "official_symbol"
        elif len(alias_candidates) == 1:
            mapped_entrez = alias_candidates[0]
            status = "mapped"
            source = "synonym"
        elif len(alias_candidates) > 1:
            status = "ambiguous_synonym"
            source = "synonym"

        rows.append(
            {
                "token_id": token_id,
                "bridge_gene_symbol": symbol,
                "mapped_entrez_id": mapped_entrez,
                "map_status": status,
                "map_source": source,
                "official_entrez_candidates": "|".join(official_candidates),
                "synonym_entrez_candidates": "|".join(alias_candidates),
            }
        )

    mapped = pd.DataFrame(rows).sort_values("token_id").reset_index(drop=True)

    counts = mapped.loc[mapped["mapped_entrez_id"] != "", "mapped_entrez_id"].value_counts()
    mapped["is_duplicate_entrez_mapping"] = mapped["mapped_entrez_id"].map(counts).fillna(0).astype(int) > 1
    mapped["duplicate_group_size"] = mapped["mapped_entrez_id"].map(counts).fillna(0).astype(int)
    return mapped


def load_term_split_dict(path: Path) -> dict[str, pd.DataFrame]:
    with path.open("rb") as f:
        obj = pickle.load(f)
    return obj


def combine_term_splits(cv1: dict[str, pd.DataFrame], cv2: dict[str, pd.DataFrame], cv3: dict[str, pd.DataFrame], holdout: dict[str, pd.DataFrame]) -> dict[str, dict[str, pd.DataFrame]]:
    terms = sorted(holdout.keys())
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for term in terms:
        train = pd.concat([cv1[term], cv2[term], cv3[term]], ignore_index=True)
        test = holdout[term].copy()
        train["gene"] = train["gene"].astype(str)
        test["gene"] = test["gene"].astype(str)
        out[term] = {"train": train, "test": test}
    return out


def term_coverage_rows(task_name: str, split_dict: dict[str, dict[str, pd.DataFrame]], mapped_entrez_set: set[str], term_names: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, (term, split) in enumerate(split_dict.items(), start=1):
        train = split["train"]
        test = split["test"]
        all_df = pd.concat([train, test], ignore_index=True)

        if all_df.duplicated(subset=["gene"]).any():
            all_df = all_df.drop_duplicates(subset=["gene"], keep="first")

        keep = all_df["gene"].isin(mapped_entrez_set)
        retained = all_df[keep]

        orig_pos = int((all_df["result"] == 1).sum())
        orig_neg = int((all_df["result"] == 0).sum())
        ret_pos = int((retained["result"] == 1).sum())
        ret_neg = int((retained["result"] == 0).sum())

        train_keep = train["gene"].isin(mapped_entrez_set)
        test_keep = test["gene"].isin(mapped_entrez_set)

        rows.append(
            {
                "task": task_name,
                "term_id": term,
                "term_name": term_names.get(term, ""),
                "original_total": int(len(all_df)),
                "retained_total": int(len(retained)),
                "original_positive_count": orig_pos,
                "retained_positive_count": ret_pos,
                "original_negative_count": orig_neg,
                "retained_negative_count": ret_neg,
                "retained_positive_coverage_pct": (100.0 * ret_pos / orig_pos) if orig_pos else np.nan,
                "retained_total_coverage_pct": (100.0 * len(retained) / len(all_df)) if len(all_df) else np.nan,
                "n_train_pos_retained": int((train.loc[train_keep, "result"] == 1).sum()),
                "n_train_neg_retained": int((train.loc[train_keep, "result"] == 0).sum()),
                "n_test_pos_retained": int((test.loc[test_keep, "result"] == 1).sum()),
                "n_test_neg_retained": int((test.loc[test_keep, "result"] == 0).sum()),
                "passes_min_positive_20": ret_pos >= MIN_RETAINED_POSITIVE,
            }
        )

        if i % 25 == 0:
            pass

    return rows


def pair_coverage_rows(name: str, path: Path, mapped_entrez_set: set[str]) -> tuple[dict[str, Any], set[str]]:
    with path.open("rb") as f:
        data = pickle.load(f)
    pairs = data["pairs"]
    labels = np.asarray(data["labels"], dtype=int)

    keep_mask = np.array([(g1 in mapped_entrez_set and g2 in mapped_entrez_set) for g1, g2 in pairs], dtype=bool)
    retained_pairs = [pairs[i] for i in np.flatnonzero(keep_mask)]
    retained_labels = labels[keep_mask]

    original_pos = int((labels == 1).sum())
    retained_pos = int((retained_labels == 1).sum())
    original_neg = int((labels == 0).sum())
    retained_neg = int((retained_labels == 0).sum())

    retained_genes_all = set()
    retained_genes_pos = set()
    for (g1, g2), label in zip(retained_pairs, retained_labels):
        retained_genes_all.add(g1)
        retained_genes_all.add(g2)
        if label == 1:
            retained_genes_pos.add(g1)
            retained_genes_pos.add(g2)

    row = {
        "interaction_type": name,
        "original_total_pairs": int(len(pairs)),
        "retained_total_pairs": int(keep_mask.sum()),
        "original_positive_pairs": original_pos,
        "retained_positive_pairs": retained_pos,
        "retained_positive_pct": (100.0 * retained_pos / original_pos) if original_pos else np.nan,
        "original_negative_pairs": original_neg,
        "retained_negative_pairs": retained_neg,
        "retained_unique_genes_all_pairs": len(retained_genes_all),
        "retained_unique_genes_positive_pairs": len(retained_genes_pos),
    }
    return row, retained_genes_all


def benchmark_gene_sets(go_terms: dict[str, dict[str, pd.DataFrame]], omim_terms: dict[str, dict[str, pd.DataFrame]], pair_paths: list[Path]) -> tuple[set[str], set[str], set[str]]:
    single_gene_set: set[str] = set()
    for split_dict in (go_terms, omim_terms):
        for split in split_dict.values():
            single_gene_set.update(split["train"]["gene"].astype(str).tolist())
            single_gene_set.update(split["test"]["gene"].astype(str).tolist())

    pair_gene_set: set[str] = set()
    for path in pair_paths:
        with path.open("rb") as f:
            data = pickle.load(f)
        for g1, g2 in data["pairs"]:
            pair_gene_set.add(str(g1))
            pair_gene_set.add(str(g2))

    all_genes = single_gene_set | pair_gene_set
    return single_gene_set, pair_gene_set, all_genes


def phase0_audit(
    vocab: pd.DataFrame,
    embedding_shape: list[int] | None,
    shape_source: str,
    existing_embedding_matrices: list[str],
    checkpoint_path: Path,
    checkpoint_source: str,
    config_path: Path,
    vocab_path: Path,
) -> dict[str, Any]:
    candidate_representations = [
        {
            "name": "static_gene_identity_embedding",
            "tensor": "model_state_dict.gene_embedding.weight",
            "shape": embedding_shape,
            "requires_sample_context": False,
            "notes": "Sample-invariant embedding lookup table from the frozen checkpoint.",
            "recommended_for_phase1": True,
        },
        {
            "name": "contextual_gene_token_embeddings",
            "tensor": "hidden states after Transformer layers (e.g., L1/L6/L12)",
            "shape": [len(vocab), embedding_shape[1] if embedding_shape else None],
            "requires_sample_context": True,
            "notes": "Requires a reference cohort and aggregation choice; context-dependent by design.",
            "recommended_for_phase1": False,
        },
        {
            "name": "sample_level_embeddings",
            "tensor": "model.encode(...).mean(dim=1)",
            "shape": [None, embedding_shape[1] if embedding_shape else None],
            "requires_sample_context": True,
            "notes": "Produces per-sample vectors, not per-gene embeddings.",
            "recommended_for_phase1": False,
        },
    ]

    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_exists": checkpoint_path.exists(),
        "checkpoint_source": checkpoint_source,
        "checkpoint_config_path": str(config_path),
        "vocabulary_path": str(vocab_path),
        "vocabulary_exists": vocab_path.exists(),
        "existing_embedding_matrices": existing_embedding_matrices,
        "embedding_shape": embedding_shape,
        "embedding_shape_source": shape_source,
        "gene_identifier_type": "HGNC-style human gene symbols (uppercase strings)",
        "vocabulary_size": int(len(vocab)),
        "embedding_dim": int(embedding_shape[1]) if embedding_shape else None,
        "row_order_definition": "Rows follow canonical vocabulary sorted by token_id; embedding row index = token_id - 1.",
        "candidate_representations": candidate_representations,
        "phase0_recommendation": "Use static_gene_identity_embedding for Zhong-style fixed-gene benchmark unless/until a separate contextual protocol is approved.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 0 + Phase 1 Bridge-to-Zhong mapping and coverage audit")
    parser.add_argument("--force-refresh-zhong", action="store_true", help="Delete and reclone Zhong repo before running")
    parser.add_argument("--force-refresh-gene-info", action="store_true", help="Re-download NCBI gene_info before running")
    parser.add_argument(
        "--shared-root",
        type=Path,
        default=DEFAULT_SHARED_ROOT,
        help="Canonical shared asset root used when local worktree assets are missing",
    )
    args = parser.parse_args()

    started = time.time()
    COVERAGE_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log = TeeLogger(LOGS_DIR / "phase1_mapping_coverage.log")

    try:
        log.log("[start] Bridge Zhong Phase0/1 audit")
        repo = ensure_zhong_repo(log, force_refresh=args.force_refresh_zhong)
        zpaths = zhong_paths(repo)

        checkpoint_path, checkpoint_source = resolve_asset_prefer_shared(
            BRIDGE_CHECKPOINT_REL, args.shared_root
        )
        config_path, _ = resolve_asset_prefer_shared(BRIDGE_CONFIG_REL, args.shared_root)
        vocab_path, _ = resolve_asset_prefer_shared(BRIDGE_VOCAB_REL, args.shared_root)
        static_provenance_path, _ = resolve_asset(
            Path("benchmarks/static_gene_embeddings/results/static_embeddings/provenance.json"),
            args.shared_root,
        )

        vocab = load_vocab(vocab_path)
        embedding_shape, shape_source = resolve_static_embedding_shape(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            static_provenance_path=static_provenance_path,
            log=log,
        )
        existing_embedding_matrices = find_existing_embedding_caches(args.shared_root)

        phase0 = phase0_audit(
            vocab=vocab,
            embedding_shape=embedding_shape,
            shape_source=shape_source,
            existing_embedding_matrices=existing_embedding_matrices,
            checkpoint_path=checkpoint_path,
            checkpoint_source=checkpoint_source,
            config_path=config_path,
            vocab_path=vocab_path,
        )
        phase0_path = COVERAGE_DIR / "phase0_bridge_embedding_audit.json"
        phase0_path.write_text(json.dumps(phase0, indent=2) + "\n", encoding="utf-8")
        log.log(f"[phase0] wrote {phase0_path}")

        symbol_resource, symbol_resource_type = ensure_symbol_mapping_resource(
            log, force_refresh=args.force_refresh_gene_info
        )
        if symbol_resource_type == "ncbi_gene_info":
            official_map, alias_map = build_symbol_to_entrez_maps(symbol_resource)
        else:
            official_map, alias_map = build_symbol_to_entrez_maps_from_cognoma(symbol_resource)
        mapped = map_bridge_to_entrez(vocab, official_map, alias_map)

        mapping_path = BENCH / "bridge_gene_mapping.tsv"
        mapped.to_csv(mapping_path, sep="\t", index=False)
        log.log(f"[phase1] wrote mapping table: {mapping_path}")

        mapped_ok = mapped[mapped["map_status"] == "mapped"].copy()
        mapped_entrez_set = set(mapped_ok["mapped_entrez_id"].astype(str))
        duplicate_groups = mapped_ok[mapped_ok["is_duplicate_entrez_mapping"]]

        go_cv1 = load_term_split_dict(zpaths.go_cv1)
        go_cv2 = load_term_split_dict(zpaths.go_cv2)
        go_cv3 = load_term_split_dict(zpaths.go_cv3)
        go_holdout = load_term_split_dict(zpaths.go_holdout)

        omim_cv1 = load_term_split_dict(zpaths.omim_cv1)
        omim_cv2 = load_term_split_dict(zpaths.omim_cv2)
        omim_cv3 = load_term_split_dict(zpaths.omim_cv3)
        omim_holdout = load_term_split_dict(zpaths.omim_holdout)

        go_terms = combine_term_splits(go_cv1, go_cv2, go_cv3, go_holdout)
        omim_terms = combine_term_splits(omim_cv1, omim_cv2, omim_cv3, omim_holdout)

        go_names = parse_obo_names(zpaths.go_obo)
        doid_names = parse_obo_names(zpaths.doid_obo)

        log.log("[phase1] computing term coverage")
        term_rows = []
        term_rows.extend(term_coverage_rows("GO_BP", go_terms, mapped_entrez_set, go_names))
        term_rows.extend(term_coverage_rows("OMIM_DOID", omim_terms, mapped_entrez_set, doid_names))
        term_df = pd.DataFrame(term_rows)
        term_path = COVERAGE_DIR / "term_coverage.csv"
        term_df.to_csv(term_path, index=False)
        log.log(f"[phase1] wrote term coverage: {term_path}")

        log.log("[phase1] computing pair coverage")
        pair_rows = []
        retained_pair_gene_union: set[str] = set()
        for name, path in [("synthetic_lethal", zpaths.pair_sl), ("negative_genetic", zpaths.pair_ng), ("tf_target", zpaths.pair_tf)]:
            row, genes = pair_coverage_rows(name, path, mapped_entrez_set)
            pair_rows.append(row)
            retained_pair_gene_union.update(genes)
        pair_df = pd.DataFrame(pair_rows)
        pair_path = COVERAGE_DIR / "pair_coverage.csv"
        pair_df.to_csv(pair_path, index=False)
        log.log(f"[phase1] wrote pair coverage: {pair_path}")

        single_gene_set, pair_gene_set, benchmark_genes = benchmark_gene_sets(
            go_terms, omim_terms, [zpaths.pair_sl, zpaths.pair_ng, zpaths.pair_tf]
        )
        overlap = benchmark_genes & mapped_entrez_set

        go_eligible = int(((term_df["task"] == "GO_BP") & term_df["passes_min_positive_20"]).sum())
        omim_eligible = int(((term_df["task"] == "OMIM_DOID") & term_df["passes_min_positive_20"]).sum())

        summary_rows = [
            {"section": "overall", "metric": "bridge_total_genes", "value": int(len(vocab))},
            {"section": "overall", "metric": "bridge_mapped_rows", "value": int((mapped["map_status"] == "mapped").sum())},
            {"section": "overall", "metric": "bridge_mapped_unique_entrez", "value": int(len(mapped_entrez_set))},
            {"section": "overall", "metric": "bridge_unmapped_rows", "value": int((mapped["map_status"] == "unmapped").sum())},
            {"section": "overall", "metric": "bridge_ambiguous_rows", "value": int(mapped["map_status"].str.startswith("ambiguous").sum())},
            {"section": "overall", "metric": "bridge_duplicate_entrez_rows", "value": int(duplicate_groups.shape[0])},
            {"section": "overall", "metric": "benchmark_genes_single_gene_tasks", "value": int(len(single_gene_set))},
            {"section": "overall", "metric": "benchmark_genes_pair_tasks", "value": int(len(pair_gene_set))},
            {"section": "overall", "metric": "benchmark_genes_union", "value": int(len(benchmark_genes))},
            {"section": "overall", "metric": "overlapping_genes", "value": int(len(overlap))},
            {
                "section": "overall",
                "metric": "overlap_pct_of_benchmark",
                "value": 100.0 * len(overlap) / len(benchmark_genes) if benchmark_genes else np.nan,
            },
            {
                "section": "overall",
                "metric": "overlap_pct_of_bridge_mapped",
                "value": 100.0 * len(overlap) / len(mapped_entrez_set) if mapped_entrez_set else np.nan,
            },
            {"section": "single_gene", "metric": "go_terms_total", "value": int((term_df["task"] == "GO_BP").sum())},
            {"section": "single_gene", "metric": "go_terms_pass_min_positive_20", "value": go_eligible},
            {"section": "single_gene", "metric": "omim_terms_total", "value": int((term_df["task"] == "OMIM_DOID").sum())},
            {"section": "single_gene", "metric": "omim_terms_pass_min_positive_20", "value": omim_eligible},
            {
                "section": "pair",
                "metric": "synthetic_lethal_retained_positive_pct",
                "value": float(pair_df.loc[pair_df["interaction_type"] == "synthetic_lethal", "retained_positive_pct"].iloc[0]),
            },
            {
                "section": "pair",
                "metric": "negative_genetic_retained_positive_pct",
                "value": float(pair_df.loc[pair_df["interaction_type"] == "negative_genetic", "retained_positive_pct"].iloc[0]),
            },
            {
                "section": "pair",
                "metric": "tf_target_retained_positive_pct",
                "value": float(pair_df.loc[pair_df["interaction_type"] == "tf_target", "retained_positive_pct"].iloc[0]),
            },
        ]

        summary_df = pd.DataFrame(summary_rows)
        summary_path = COVERAGE_DIR / "coverage_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        log.log(f"[phase1] wrote coverage summary: {summary_path}")

        problems: list[str] = []
        if int(mapped["map_status"].str.startswith("ambiguous").sum()) > 0:
            problems.append("ambiguous_symbol_to_entrez_mappings")
        if int(duplicate_groups.shape[0]) > 0:
            problems.append("multiple_bridge_symbols_map_to_same_entrez")
        if len(overlap) < 0.5 * len(benchmark_genes):
            problems.append("low_benchmark_gene_overlap_lt_50pct")
        if go_eligible < 10:
            problems.append("few_go_terms_pass_min_positive_threshold")
        if omim_eligible < 10:
            problems.append("few_omim_terms_pass_min_positive_threshold")

        report = {
            "status": "complete",
            "created_utc": pd.Timestamp.utcnow().isoformat(),
            "elapsed_seconds": time.time() - started,
            "phase0": {
                "phase0_json": str(phase0_path),
                "checkpoint_exists": checkpoint_path.exists(),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_source": checkpoint_source,
                "config_path": str(config_path),
                "vocab_path": str(vocab_path),
                "embedding_shape": embedding_shape,
                "shape_source": shape_source,
                "existing_embedding_matrices": existing_embedding_matrices,
            },
            "phase1": {
                "mapping_tsv": str(mapping_path),
                "symbol_to_entrez_resource": str(symbol_resource),
                "symbol_to_entrez_resource_type": symbol_resource_type,
                "coverage_summary_csv": str(summary_path),
                "term_coverage_csv": str(term_path),
                "pair_coverage_csv": str(pair_path),
                "major_problems": problems,
            },
        }
        report_path = COVERAGE_DIR / "phase1_report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        log.log(f"[done] wrote report: {report_path}")

    finally:
        log.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
