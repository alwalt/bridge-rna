#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "benchmarks" / "zhong_gene_biology"
REP_DIR = BENCH / "representations"
RES_DIR = BENCH / "results"
FIG_DIR = BENCH / "figures"
LOGS_DIR = BENCH / "logs"
REF_REPO = BENCH / "references" / "gene-embedding-benchmarks"
MAP_PATH = BENCH / "bridge_gene_mapping.tsv"

GO_CV1 = REF_REPO / "data/data_splits/gene_level_benchmark/go_folds_splits/go_cv_fold1_dict_all.pkl"
GO_CV2 = REF_REPO / "data/data_splits/gene_level_benchmark/go_folds_splits/go_cv_fold2_dict_all.pkl"
GO_CV3 = REF_REPO / "data/data_splits/gene_level_benchmark/go_folds_splits/go_cv_fold3_dict_all.pkl"
GO_HOLDOUT = REF_REPO / "data/data_splits/gene_level_benchmark/go_folds_splits/go_holdout_dict_all.pkl"

OMIM_CV1 = REF_REPO / "data/data_splits/gene_level_benchmark/omim_folds_splits/omim_cv_fold1_dict_all.pkl"
OMIM_CV2 = REF_REPO / "data/data_splits/gene_level_benchmark/omim_folds_splits/omim_cv_fold2_dict_all.pkl"
OMIM_CV3 = REF_REPO / "data/data_splits/gene_level_benchmark/omim_folds_splits/omim_cv_fold3_dict_all.pkl"
OMIM_HOLDOUT = REF_REPO / "data/data_splits/gene_level_benchmark/omim_folds_splits/omim_holdout_dict_all.pkl"

GO_OBO = REF_REPO / "data/obo/go.obo"
DOID_OBO = REF_REPO / "data/obo/doid.obo"

C_VALUES = [0.1, 1.0, 10.0, 100.0, 1000.0]


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


def precision_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int = 10) -> float:
    if len(y_scores) == 0:
        return float("nan")
    order = np.argsort(y_scores)
    if len(y_scores) > k:
        idx = order[-k:][::-1]
    else:
        idx = order[::-1]
    return float(np.mean(y_true[idx]))


def load_split_dict(path: Path) -> dict[str, pd.DataFrame]:
    with path.open("rb") as f:
        raw = pickle.load(f)
    out: dict[str, pd.DataFrame] = {}
    for term, df in raw.items():
        q = df.copy()
        q["gene"] = q["gene"].astype(str)
        q["result"] = q["result"].astype(int)
        out[str(term)] = q[["gene", "result"]]
    return out


def parse_obo_names(path: Path) -> dict[str, str]:
    names: dict[str, str] = {}
    current_id: str | None = None
    current_name: str | None = None
    in_term = False
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line == "[Term]":
                if in_term and current_id and current_name:
                    names[current_id] = current_name
                in_term = True
                current_id = None
                current_name = None
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
                in_term = False
                current_id = None
                current_name = None
    if in_term and current_id and current_name:
        names[current_id] = current_name
    return names


@dataclass
class Representation:
    name: str
    emb: np.ndarray
    genes: pd.DataFrame


def load_representation(name: str) -> Representation:
    rep_root = REP_DIR / name
    emb_path = rep_root / "embeddings.npy"
    genes_path = rep_root / "genes.tsv"
    if not emb_path.exists() or not genes_path.exists():
        raise FileNotFoundError(f"Missing representation assets for {name}: {emb_path}, {genes_path}")
    emb = np.load(emb_path)
    genes = pd.read_csv(genes_path, sep="\t")
    req = {"token_id", "gene_symbol"}
    if not req.issubset(genes.columns):
        raise ValueError(f"{genes_path} missing columns: {req - set(genes.columns)}")
    genes = genes[["token_id", "gene_symbol"]].copy()
    genes["token_id"] = genes["token_id"].astype(int)
    genes["gene_symbol"] = genes["gene_symbol"].astype(str).str.upper().str.strip()
    genes = genes.sort_values("token_id").reset_index(drop=True)
    if emb.shape[0] != len(genes):
        raise AssertionError(f"{name}: embedding rows {emb.shape[0]} != genes {len(genes)}")
    if emb.shape[1] != 512:
        raise AssertionError(f"{name}: expected 512 dims, got {emb.shape[1]}")
    if not np.isfinite(emb).all():
        raise AssertionError(f"{name}: non-finite values")
    return Representation(name=name, emb=emb.astype(np.float32), genes=genes)


def prepare_gene_mapping() -> pd.DataFrame:
    if not MAP_PATH.exists():
        raise FileNotFoundError(f"Missing mapping table: {MAP_PATH}")
    m = pd.read_csv(MAP_PATH, sep="\t")
    expected = {
        "token_id",
        "bridge_gene_symbol",
        "mapped_entrez_id",
        "map_status",
        "is_duplicate_entrez_mapping",
    }
    if not expected.issubset(m.columns):
        raise ValueError(f"Mapping table missing columns: {expected - set(m.columns)}")
    m["token_id"] = m["token_id"].astype(int)
    m["bridge_gene_symbol"] = m["bridge_gene_symbol"].astype(str).str.upper().str.strip()
    m["map_status"] = m["map_status"].astype(str)
    m["is_duplicate_entrez_mapping"] = m["is_duplicate_entrez_mapping"].astype(bool)
    m["mapped_entrez_id"] = m["mapped_entrez_id"].astype("Int64")

    mapped = m[(m["map_status"] == "mapped") & (~m["is_duplicate_entrez_mapping"])].copy()
    mapped["mapped_entrez_id"] = mapped["mapped_entrez_id"].astype(str)
    if mapped["mapped_entrez_id"].duplicated().any():
        dup = mapped[mapped["mapped_entrez_id"].duplicated(keep=False)].sort_values("mapped_entrez_id")
        raise AssertionError(f"Duplicate mapped Entrez IDs remain:\n{dup.head(20).to_string(index=False)}")
    return mapped


def build_feature_matrix(mapped: pd.DataFrame, rep: Representation) -> tuple[np.ndarray, np.ndarray]:
    g = rep.genes[["token_id", "gene_symbol"]].copy()
    merged = mapped.merge(
        g,
        left_on=["token_id", "bridge_gene_symbol"],
        right_on=["token_id", "gene_symbol"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(mapped):
        raise AssertionError(f"{rep.name}: mapping rows {len(mapped)} but aligned rows {len(merged)}")
    merged = merged.sort_values("token_id").reset_index(drop=True)
    row_index = merged["token_id"].to_numpy() - 1
    x = rep.emb[row_index]
    entrez = merged["mapped_entrez_id"].to_numpy(dtype=str)
    return entrez, x


def filter_split(df: pd.DataFrame, allowed_genes: set[str]) -> pd.DataFrame:
    q = df.copy()
    q["gene"] = q["gene"].astype(str)
    q = q[q["gene"].isin(allowed_genes)].copy()
    q["result"] = q["result"].astype(int)
    return q[["gene", "result"]]


def evaluate_term(
    term: str,
    term_name: str,
    rep_name: str,
    cv1: pd.DataFrame,
    cv2: pd.DataFrame,
    cv3: pd.DataFrame,
    holdout: pd.DataFrame,
    gene_to_row: dict[str, int],
    x: np.ndarray,
    seed: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    all_cv = pd.concat([cv1, cv2, cv3], ignore_index=True)
    orig_pos = int(all_cv["result"].sum() + holdout["result"].sum())
    orig_total = int(len(all_cv) + len(holdout))

    fold_splits = [
        (cv1, pd.concat([cv2, cv3], ignore_index=True)),
        (cv2, pd.concat([cv1, cv3], ignore_index=True)),
        (cv3, pd.concat([cv1, cv2], ignore_index=True)),
    ]

    train_all = pd.concat([cv1, cv2, cv3], ignore_index=True)
    retained_pos = int(train_all["result"].sum() + holdout["result"].sum())
    retained_total = int(len(train_all) + len(holdout))

    n_train_pos = int(train_all["result"].sum())
    n_train_neg = int((train_all["result"] == 0).sum())
    n_test_pos = int(holdout["result"].sum())
    n_test_neg = int((holdout["result"] == 0).sum())

    if len(train_all) == 0:
        return None, {
            "term": term,
            "term_name": term_name,
            "representation": rep_name,
            "reason": "empty_train_after_bridge_filter",
        }
    if len(holdout) == 0:
        return None, {
            "term": term,
            "term_name": term_name,
            "representation": rep_name,
            "reason": "empty_holdout_after_bridge_filter",
        }
    if n_train_pos == 0 or n_train_neg == 0:
        return None, {
            "term": term,
            "term_name": term_name,
            "representation": rep_name,
            "reason": "single_class_train_after_bridge_filter",
        }
    if n_test_pos == 0 or n_test_neg == 0:
        return None, {
            "term": term,
            "term_name": term_name,
            "representation": rep_name,
            "reason": "single_class_holdout_after_bridge_filter",
        }

    c_records: list[dict[str, Any]] = []
    for c in C_VALUES:
        fold_aurocs: list[float] = []
        fold_auprcs: list[float] = []
        fold_pr10s: list[float] = []
        fold_valid = 0
        for test_df, train_df in fold_splits:
            if len(train_df) == 0 or len(test_df) == 0:
                continue
            y_train = train_df["result"].to_numpy(dtype=np.int64)
            y_test = test_df["result"].to_numpy(dtype=np.int64)
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue

            train_idx = np.array([gene_to_row[g] for g in train_df["gene"].to_list()], dtype=np.int64)
            test_idx = np.array([gene_to_row[g] for g in test_df["gene"].to_list()], dtype=np.int64)
            x_train = x[train_idx]
            x_test = x[test_idx]

            clf = SVC(C=c, kernel="rbf", class_weight="balanced", random_state=seed)
            clf.fit(x_train, y_train)
            y_score = clf.decision_function(x_test)
            fold_aurocs.append(float(roc_auc_score(y_test, y_score)))
            fold_auprcs.append(float(average_precision_score(y_test, y_score)))
            fold_pr10s.append(precision_at_k(y_test, y_score, k=10))
            fold_valid += 1

        if fold_valid == 0:
            c_records.append(
                {
                    "C": c,
                    "cv_mean_auroc": float("nan"),
                    "cv_mean_auprc": float("nan"),
                    "cv_mean_p10": float("nan"),
                    "cv_valid_folds": 0,
                }
            )
        else:
            c_records.append(
                {
                    "C": c,
                    "cv_mean_auroc": float(np.mean(fold_aurocs)),
                    "cv_mean_auprc": float(np.mean(fold_auprcs)),
                    "cv_mean_p10": float(np.mean(fold_pr10s)),
                    "cv_valid_folds": int(fold_valid),
                }
            )

    c_df = pd.DataFrame(c_records)
    c_df = c_df.sort_values(by=["cv_mean_auroc", "cv_mean_auprc", "cv_mean_p10"], ascending=False)
    if c_df["cv_valid_folds"].max() == 0:
        return None, {
            "term": term,
            "term_name": term_name,
            "representation": rep_name,
            "reason": "no_valid_cv_folds_after_bridge_filter",
        }
    best = c_df.iloc[0]
    best_c = float(best["C"])

    train_idx = np.array([gene_to_row[g] for g in train_all["gene"].to_list()], dtype=np.int64)
    holdout_idx = np.array([gene_to_row[g] for g in holdout["gene"].to_list()], dtype=np.int64)
    x_train_all = x[train_idx]
    x_holdout = x[holdout_idx]
    y_train_all = train_all["result"].to_numpy(dtype=np.int64)
    y_holdout = holdout["result"].to_numpy(dtype=np.int64)

    clf = SVC(C=best_c, kernel="rbf", class_weight="balanced", random_state=seed)
    clf.fit(x_train_all, y_train_all)
    y_score = clf.decision_function(x_holdout)

    auroc = float(roc_auc_score(y_holdout, y_score))
    auprc = float(average_precision_score(y_holdout, y_score))
    p10 = precision_at_k(y_holdout, y_score, k=10)

    row = {
        "term": term,
        "term_name": term_name,
        "representation": rep_name,
        "n_train_pos": n_train_pos,
        "n_train_neg": n_train_neg,
        "n_test_pos": n_test_pos,
        "n_test_neg": n_test_neg,
        "selected_C": best_c,
        "AUROC": auroc,
        "AUPRC": auprc,
        "Precision@10": p10,
        "bridge_coverage_total": retained_total / max(orig_total, 1),
        "bridge_coverage_positive": retained_pos / max(orig_pos, 1),
        "retained_total": retained_total,
        "original_total": orig_total,
        "retained_positive": retained_pos,
        "original_positive": orig_pos,
    }
    return row, {}


def bootstrap_median_ci(values: np.ndarray, n_boot: int = 5000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = values[np.isfinite(values)]
    if len(x) == 0:
        return float("nan"), float("nan")
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = rng.choice(x, size=len(x), replace=True)
        boots[i] = np.median(sample)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def summarize_metric(df: pd.DataFrame, metric: str) -> dict[str, Any]:
    x = df[metric].to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {
            "n_terms": 0,
            "median": float("nan"),
            "mean": float("nan"),
            "iqr": float("nan"),
        }
    q1, q3 = np.percentile(x, [25, 75])
    return {
        "n_terms": int(len(x)),
        "median": float(np.median(x)),
        "mean": float(np.mean(x)),
        "iqr": float(q3 - q1),
    }


def paired_comparison(df: pd.DataFrame, metric: str, left: str, right: str, seed: int) -> dict[str, Any]:
    delta = df[f"{right}_{metric}"].to_numpy(dtype=float) - df[f"{left}_{metric}"].to_numpy(dtype=float)
    delta = delta[np.isfinite(delta)]
    if len(delta) == 0:
        return {
            "comparison": f"{right}-{left}",
            "n_terms": 0,
            "median_delta": float("nan"),
            "mean_delta": float("nan"),
            "iqr_delta": float("nan"),
            "median_delta_ci_low": float("nan"),
            "median_delta_ci_high": float("nan"),
            "fraction_positive_delta": float("nan"),
            "wilcoxon_stat": float("nan"),
            "wilcoxon_p": float("nan"),
        }
    q1, q3 = np.percentile(delta, [25, 75])
    ci_low, ci_high = bootstrap_median_ci(delta, n_boot=5000, seed=seed)
    try:
        w = wilcoxon(delta, alternative="two-sided", zero_method="wilcox", correction=False, mode="auto")
        w_stat = float(w.statistic)
        w_p = float(w.pvalue)
    except ValueError:
        w_stat = float("nan")
        w_p = float("nan")
    return {
        "comparison": f"{right}-{left}",
        "n_terms": int(len(delta)),
        "median_delta": float(np.median(delta)),
        "mean_delta": float(np.mean(delta)),
        "iqr_delta": float(q3 - q1),
        "median_delta_ci_low": ci_low,
        "median_delta_ci_high": ci_high,
        "fraction_positive_delta": float(np.mean(delta > 0)),
        "wilcoxon_stat": w_stat,
        "wilcoxon_p": w_p,
    }


def plot_task_figures(task: str, paired: pd.DataFrame) -> None:
    task_label = task.upper()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    x_coords = np.array([0, 1, 2], dtype=float)
    reps = ["E0", "E1", "E2"]

    for metric in ["AUROC", "AUPRC"]:
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        for _, row in paired.iterrows():
            ys = [row[f"{rep}_{metric}"] for rep in reps]
            ax.plot(x_coords, ys, color="#1f77b4", alpha=0.12, linewidth=0.8)
        med = [np.median(paired[f"{rep}_{metric}"].to_numpy(dtype=float)) for rep in reps]
        ax.plot(x_coords, med, color="#d62728", marker="o", linewidth=2.0, label="Median")
        ax.set_xticks(x_coords, ["E0 static", "E1 null L12", "E2 GTEx200 L12"])
        ax.set_ylabel(metric)
        ax.set_title(f"{task_label} paired {metric}: E0 → E1 → E2")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"{task.lower()}_paired_{metric.lower()}_e0_e1_e2.png", dpi=220)
        fig.savefig(FIG_DIR / f"{task.lower()}_paired_{metric.lower()}_e0_e1_e2.pdf")
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for j, metric in enumerate(["AUROC", "AUPRC"]):
        data = [paired[f"E0_{metric}"], paired[f"E1_{metric}"], paired[f"E2_{metric}"]]
        axes[0, j].boxplot(data, tick_labels=["E0", "E1", "E2"], showfliers=False)
        axes[0, j].set_title(f"{task_label} {metric} distribution")
        axes[0, j].set_ylabel(metric)
        axes[0, j].grid(alpha=0.2)

        deltas = [
            paired[f"E1_{metric}"] - paired[f"E0_{metric}"],
            paired[f"E2_{metric}"] - paired[f"E1_{metric}"],
            paired[f"E2_{metric}"] - paired[f"E0_{metric}"],
        ]
        axes[1, j].boxplot(deltas, tick_labels=["E1-E0", "E2-E1", "E2-E0"], showfliers=False)
        axes[1, j].axhline(0.0, color="black", linestyle="--", linewidth=1)
        axes[1, j].set_title(f"{task_label} Δ{metric} distribution")
        axes[1, j].set_ylabel(f"Δ{metric}")
        axes[1, j].grid(alpha=0.2)

    fig.savefig(FIG_DIR / f"{task.lower()}_metric_distributions.png", dpi=220)
    fig.savefig(FIG_DIR / f"{task.lower()}_metric_distributions.pdf")
    plt.close(fig)


def run_task(
    task_name: str,
    cv_paths: tuple[Path, Path, Path],
    holdout_path: Path,
    term_names: dict[str, str],
    representations: list[Representation],
    mapped: pd.DataFrame,
    seed: int,
    logger: Logger,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cv1_all = load_split_dict(cv_paths[0])
    cv2_all = load_split_dict(cv_paths[1])
    cv3_all = load_split_dict(cv_paths[2])
    holdout_all = load_split_dict(holdout_path)

    terms = sorted(holdout_all.keys())
    if not (set(terms) == set(cv1_all.keys()) == set(cv2_all.keys()) == set(cv3_all.keys())):
        raise AssertionError(f"{task_name}: split term-key mismatch")

    rep_term_results: dict[str, pd.DataFrame] = {}
    invalid_records_all: list[dict[str, Any]] = []

    for rep in representations:
        entrez, x = build_feature_matrix(mapped, rep)
        gene_to_row = {g: i for i, g in enumerate(entrez)}
        allowed_genes = set(gene_to_row.keys())

        rows: list[dict[str, Any]] = []
        invalid_records: list[dict[str, Any]] = []
        started = time.time()

        for idx, term in enumerate(terms, start=1):
            cv1 = filter_split(cv1_all[term], allowed_genes)
            cv2 = filter_split(cv2_all[term], allowed_genes)
            cv3 = filter_split(cv3_all[term], allowed_genes)
            hold = filter_split(holdout_all[term], allowed_genes)

            term_name = term_names.get(term, "")
            row, invalid = evaluate_term(
                term=term,
                term_name=term_name,
                rep_name=rep.name,
                cv1=cv1,
                cv2=cv2,
                cv3=cv3,
                holdout=hold,
                gene_to_row=gene_to_row,
                x=x,
                seed=seed,
            )
            if row is None:
                invalid_records.append(invalid)
            else:
                rows.append(row)

            if idx % 20 == 0 or idx == len(terms):
                elapsed = time.time() - started
                rate = idx / max(elapsed, 1e-9)
                eta = (len(terms) - idx) / max(rate, 1e-9)
                logger(
                    f"[{task_name}/{rep.name}] terms={idx}/{len(terms)} elapsed={elapsed/60:.1f}m "
                    f"rate={rate:.2f}/s eta={eta/60:.1f}m"
                )

        rep_df = pd.DataFrame(rows).sort_values("term").reset_index(drop=True)
        rep_term_results[rep.name] = rep_df
        if invalid_records:
            invalid_records_all.extend(invalid_records)

    common_terms = sorted(set.intersection(*[set(df["term"]) for df in rep_term_results.values()]))
    paired = pd.DataFrame({"term": common_terms})
    paired["term_name"] = paired["term"].map(term_names).fillna("")
    for rep in representations:
        df = rep_term_results[rep.name].set_index("term")
        paired[f"{rep.name}_AUROC"] = paired["term"].map(df["AUROC"])
        paired[f"{rep.name}_AUPRC"] = paired["term"].map(df["AUPRC"])
        paired[f"{rep.name}_Precision@10"] = paired["term"].map(df["Precision@10"])

    paired["delta01_AUROC"] = paired["E1_null_L12_AUROC"] - paired["E0_static_AUROC"]
    paired["delta12_AUROC"] = paired["E2_GTEx200_L12_AUROC"] - paired["E1_null_L12_AUROC"]
    paired["delta02_AUROC"] = paired["E2_GTEx200_L12_AUROC"] - paired["E0_static_AUROC"]
    paired["delta01_AUPRC"] = paired["E1_null_L12_AUPRC"] - paired["E0_static_AUPRC"]
    paired["delta12_AUPRC"] = paired["E2_GTEx200_L12_AUPRC"] - paired["E1_null_L12_AUPRC"]
    paired["delta02_AUPRC"] = paired["E2_GTEx200_L12_AUPRC"] - paired["E0_static_AUPRC"]

    comp_auroc = pd.DataFrame(
        [
            {"task": task_name, **paired_comparison(paired.rename(columns={"E0_static_AUROC": "E0_AUROC", "E1_null_L12_AUROC": "E1_AUROC", "E2_GTEx200_L12_AUROC": "E2_AUROC"}), "AUROC", "E0", "E1", seed)},
            {"task": task_name, **paired_comparison(paired.rename(columns={"E0_static_AUROC": "E0_AUROC", "E1_null_L12_AUROC": "E1_AUROC", "E2_GTEx200_L12_AUROC": "E2_AUROC"}), "AUROC", "E1", "E2", seed)},
            {"task": task_name, **paired_comparison(paired.rename(columns={"E0_static_AUROC": "E0_AUROC", "E1_null_L12_AUROC": "E1_AUROC", "E2_GTEx200_L12_AUROC": "E2_AUROC"}), "AUROC", "E0", "E2", seed)},
        ]
    )
    comp_auprc = pd.DataFrame(
        [
            {"task": task_name, **paired_comparison(paired.rename(columns={"E0_static_AUPRC": "E0_AUPRC", "E1_null_L12_AUPRC": "E1_AUPRC", "E2_GTEx200_L12_AUPRC": "E2_AUPRC"}), "AUPRC", "E0", "E1", seed)},
            {"task": task_name, **paired_comparison(paired.rename(columns={"E0_static_AUPRC": "E0_AUPRC", "E1_null_L12_AUPRC": "E1_AUPRC", "E2_GTEx200_L12_AUPRC": "E2_AUPRC"}), "AUPRC", "E1", "E2", seed)},
            {"task": task_name, **paired_comparison(paired.rename(columns={"E0_static_AUPRC": "E0_AUPRC", "E1_null_L12_AUPRC": "E1_AUPRC", "E2_GTEx200_L12_AUPRC": "E2_AUPRC"}), "AUPRC", "E0", "E2", seed)},
        ]
    )

    invalid_df = pd.DataFrame(invalid_records_all)
    return rep_term_results, paired, comp_auroc, comp_auprc, invalid_df


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Zhong single-gene GO/OMIM benchmark for E0/E1/E2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    started = time.time()
    np.random.seed(args.seed)

    RES_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = Logger(LOGS_DIR / "run_single_gene_ablation.log")

    for p in [GO_CV1, GO_CV2, GO_CV3, GO_HOLDOUT, OMIM_CV1, OMIM_CV2, OMIM_CV3, OMIM_HOLDOUT, GO_OBO, DOID_OBO]:
        if not p.exists():
            raise FileNotFoundError(f"Missing benchmark input: {p}")

    reps = [
        load_representation("E0_static"),
        load_representation("E1_null_L12"),
        load_representation("E2_GTEx200_L12"),
    ]
    logger("[setup] Loaded E0/E1/E2 representations")

    for rep in reps[1:]:
        if not np.array_equal(reps[0].genes["token_id"].to_numpy(), rep.genes["token_id"].to_numpy()):
            raise AssertionError(f"token_id mismatch between {reps[0].name} and {rep.name}")
        if not np.array_equal(reps[0].genes["gene_symbol"].to_numpy(), rep.genes["gene_symbol"].to_numpy()):
            raise AssertionError(f"gene ordering mismatch between {reps[0].name} and {rep.name}")

    mapped = prepare_gene_mapping()
    map_status_counts = pd.read_csv(MAP_PATH, sep="\t")["map_status"].value_counts().to_dict()
    logger(f"[mapping] using mapped non-duplicate genes: {len(mapped)}")
    logger(f"[mapping] full status counts: {map_status_counts}")

    go_names = parse_obo_names(GO_OBO)
    doid_names = parse_obo_names(DOID_OBO)

    go_rep, go_paired, go_comp_auroc, go_comp_auprc, go_invalid = run_task(
        task_name="GO",
        cv_paths=(GO_CV1, GO_CV2, GO_CV3),
        holdout_path=GO_HOLDOUT,
        term_names=go_names,
        representations=reps,
        mapped=mapped,
        seed=args.seed,
        logger=logger,
    )
    omim_rep, omim_paired, omim_comp_auroc, omim_comp_auprc, omim_invalid = run_task(
        task_name="OMIM",
        cv_paths=(OMIM_CV1, OMIM_CV2, OMIM_CV3),
        holdout_path=OMIM_HOLDOUT,
        term_names=doid_names,
        representations=reps,
        mapped=mapped,
        seed=args.seed,
        logger=logger,
    )

    output_map = {
        "E0_static": "E0",
        "E1_null_L12": "E1",
        "E2_GTEx200_L12": "E2",
    }
    for rep_name, short in output_map.items():
        go_rep[rep_name].to_csv(RES_DIR / f"{short}_GO.csv", index=False)
        omim_rep[rep_name].to_csv(RES_DIR / f"{short}_OMIM.csv", index=False)

    go_vs = go_paired[["term", "term_name", "E0_static_AUROC", "E1_null_L12_AUROC", "delta01_AUROC", "E0_static_AUPRC", "E1_null_L12_AUPRC", "delta01_AUPRC"]].rename(
        columns={
            "E0_static_AUROC": "E0_AUROC",
            "E1_null_L12_AUROC": "E1_AUROC",
            "E0_static_AUPRC": "E0_AUPRC",
            "E1_null_L12_AUPRC": "E1_AUPRC",
        }
    )
    omim_vs = omim_paired[["term", "term_name", "E0_static_AUROC", "E1_null_L12_AUROC", "delta01_AUROC", "E0_static_AUPRC", "E1_null_L12_AUPRC", "delta01_AUPRC"]].rename(
        columns={
            "E0_static_AUROC": "E0_AUROC",
            "E1_null_L12_AUROC": "E1_AUROC",
            "E0_static_AUPRC": "E0_AUPRC",
            "E1_null_L12_AUPRC": "E1_AUPRC",
        }
    )
    go_vs.to_csv(RES_DIR / "E0_vs_E1_GO.csv", index=False)
    omim_vs.to_csv(RES_DIR / "E0_vs_E1_OMIM.csv", index=False)

    go_paired.to_csv(RES_DIR / "E0_E1_E2_GO_paired.csv", index=False)
    omim_paired.to_csv(RES_DIR / "E0_E1_E2_OMIM_paired.csv", index=False)

    comp_auroc = pd.concat([go_comp_auroc, omim_comp_auroc], ignore_index=True)
    comp_auprc = pd.concat([go_comp_auprc, omim_comp_auprc], ignore_index=True)
    comp_auroc.to_csv(RES_DIR / "comparison_summary_AUROC.csv", index=False)
    comp_auprc.to_csv(RES_DIR / "comparison_summary_AUPRC.csv", index=False)

    invalid = pd.concat([go_invalid.assign(task="GO"), omim_invalid.assign(task="OMIM")], ignore_index=True)
    invalid.to_csv(RES_DIR / "excluded_terms.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    for task_name, rep_dict in [("GO", go_rep), ("OMIM", omim_rep)]:
        for rep_name, df in rep_dict.items():
            auroc = df["AUROC"].to_numpy(dtype=float)
            auprc = df["AUPRC"].to_numpy(dtype=float)
            p10 = df["Precision@10"].to_numpy(dtype=float)
            q1, q3 = np.percentile(auroc, [25, 75]) if len(auroc) else (np.nan, np.nan)
            summary_rows.append(
                {
                    "task": task_name,
                    "representation": rep_name,
                    "N_terms": int(len(df)),
                    "median_AUROC": float(np.median(auroc)) if len(auroc) else float("nan"),
                    "IQR_AUROC": float(q3 - q1) if len(auroc) else float("nan"),
                    "median_AUPRC": float(np.median(auprc)) if len(auprc) else float("nan"),
                    "median_Precision@10": float(np.median(p10)) if len(p10) else float("nan"),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RES_DIR / "representation_summary.csv", index=False)

    plot_task_figures("GO", go_paired.rename(columns={
        "E0_static_AUROC": "E0_AUROC",
        "E1_null_L12_AUROC": "E1_AUROC",
        "E2_GTEx200_L12_AUROC": "E2_AUROC",
        "E0_static_AUPRC": "E0_AUPRC",
        "E1_null_L12_AUPRC": "E1_AUPRC",
        "E2_GTEx200_L12_AUPRC": "E2_AUPRC",
    }))
    plot_task_figures("OMIM", omim_paired.rename(columns={
        "E0_static_AUROC": "E0_AUROC",
        "E1_null_L12_AUROC": "E1_AUROC",
        "E2_GTEx200_L12_AUROC": "E2_AUROC",
        "E0_static_AUPRC": "E0_AUPRC",
        "E1_null_L12_AUPRC": "E1_AUPRC",
        "E2_GTEx200_L12_AUPRC": "E2_AUPRC",
    }))

    report = {
        "status": "complete",
        "elapsed_seconds": time.time() - started,
        "seed": args.seed,
        "mapping": {
            "mapping_tsv": str(MAP_PATH),
            "mapped_genes": int(len(mapped)),
            "excluded_ambiguous_or_unmapped": int(pd.read_csv(MAP_PATH, sep="\t").shape[0] - len(mapped)),
        },
        "outputs": {
            "results_dir": str(RES_DIR.resolve()),
            "figures_dir": str(FIG_DIR.resolve()),
            "log": str((LOGS_DIR / "run_single_gene_ablation.log").resolve()),
        },
        "summary_table": summary_df.to_dict(orient="records"),
    }
    (RES_DIR / "run_single_gene_ablation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    logger("[done] GO+OMIM E0/E1/E2 benchmark complete")
    logger(f"[done] results: {RES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
