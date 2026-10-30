#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "benchmarks" / "zhong_gene_biology"
REP_DIR = BENCH / "representations"
RES_DIR = BENCH / "results"
CONTROL_DIR = RES_DIR / "controls" / "single_gene"
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
    idx = order[-k:][::-1] if len(y_scores) > k else order[::-1]
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


def load_rep(name: str) -> tuple[np.ndarray, pd.DataFrame]:
    emb = np.load(REP_DIR / name / "embeddings.npy")
    genes = pd.read_csv(REP_DIR / name / "genes.tsv", sep="\t")
    genes = genes[["token_id", "gene_symbol"]].copy()
    genes["token_id"] = genes["token_id"].astype(int)
    genes["gene_symbol"] = genes["gene_symbol"].astype(str).str.upper().str.strip()
    genes = genes.sort_values("token_id").reset_index(drop=True)
    if emb.shape != (len(genes), 512):
        raise AssertionError(f"Unexpected shape for {name}: {emb.shape}")
    return emb.astype(np.float32), genes


def load_mapped_genes() -> pd.DataFrame:
    m = pd.read_csv(MAP_PATH, sep="\t")
    m["token_id"] = m["token_id"].astype(int)
    m["bridge_gene_symbol"] = m["bridge_gene_symbol"].astype(str).str.upper().str.strip()
    m["map_status"] = m["map_status"].astype(str)
    m["is_duplicate_entrez_mapping"] = m["is_duplicate_entrez_mapping"].astype(bool)
    m["mapped_entrez_id"] = m["mapped_entrez_id"].astype("Int64")
    mapped = m[(m["map_status"] == "mapped") & (~m["is_duplicate_entrez_mapping"])].copy()
    mapped["mapped_entrez_id"] = mapped["mapped_entrez_id"].astype(str)
    if mapped["mapped_entrez_id"].duplicated().any():
        raise AssertionError("Duplicate Entrez IDs remain after filtering")
    return mapped


def filter_split(df: pd.DataFrame, allowed: set[str]) -> pd.DataFrame:
    q = df.copy()
    q["gene"] = q["gene"].astype(str)
    q = q[q["gene"].isin(allowed)].copy()
    q["result"] = q["result"].astype(int)
    return q[["gene", "result"]]


def evaluate_task(
    task_name: str,
    cv_paths: tuple[Path, Path, Path],
    holdout_path: Path,
    x: np.ndarray,
    gene_to_row: dict[str, int],
    seed: int,
    logger: Logger,
) -> pd.DataFrame:
    cv1_all = load_split_dict(cv_paths[0])
    cv2_all = load_split_dict(cv_paths[1])
    cv3_all = load_split_dict(cv_paths[2])
    hold_all = load_split_dict(holdout_path)
    terms = sorted(hold_all.keys())
    allowed = set(gene_to_row.keys())

    rows: list[dict[str, Any]] = []
    started = time.time()
    for i, term in enumerate(terms, start=1):
        cv1 = filter_split(cv1_all[term], allowed)
        cv2 = filter_split(cv2_all[term], allowed)
        cv3 = filter_split(cv3_all[term], allowed)
        hold = filter_split(hold_all[term], allowed)

        train_all = pd.concat([cv1, cv2, cv3], ignore_index=True)
        if len(train_all) == 0 or len(hold) == 0:
            continue
        if train_all["result"].nunique() < 2 or hold["result"].nunique() < 2:
            continue

        fold_splits = [
            (cv1, pd.concat([cv2, cv3], ignore_index=True)),
            (cv2, pd.concat([cv1, cv3], ignore_index=True)),
            (cv3, pd.concat([cv1, cv2], ignore_index=True)),
        ]

        best = None
        for c in C_VALUES:
            vals = []
            for test_df, tr_df in fold_splits:
                if len(test_df) == 0 or len(tr_df) == 0:
                    continue
                if tr_df["result"].nunique() < 2 or test_df["result"].nunique() < 2:
                    continue
                tr_idx = np.array([gene_to_row[g] for g in tr_df["gene"].to_list()], dtype=np.int64)
                te_idx = np.array([gene_to_row[g] for g in test_df["gene"].to_list()], dtype=np.int64)
                x_tr, x_te = x[tr_idx], x[te_idx]
                y_tr = tr_df["result"].to_numpy(dtype=np.int64)
                y_te = test_df["result"].to_numpy(dtype=np.int64)
                clf = SVC(C=c, kernel="rbf", class_weight="balanced", random_state=seed)
                clf.fit(x_tr, y_tr)
                y_sc = clf.decision_function(x_te)
                vals.append((roc_auc_score(y_te, y_sc), average_precision_score(y_te, y_sc), precision_at_k(y_te, y_sc, 10)))
            if vals:
                auc = float(np.mean([v[0] for v in vals]))
                auprc = float(np.mean([v[1] for v in vals]))
                p10 = float(np.mean([v[2] for v in vals]))
                score = (auc, auprc, p10)
                if best is None or score > best[0]:
                    best = (score, c)
        if best is None:
            continue
        best_c = best[1]

        tr_idx = np.array([gene_to_row[g] for g in train_all["gene"].to_list()], dtype=np.int64)
        te_idx = np.array([gene_to_row[g] for g in hold["gene"].to_list()], dtype=np.int64)
        x_tr, x_te = x[tr_idx], x[te_idx]
        y_tr = train_all["result"].to_numpy(dtype=np.int64)
        y_te = hold["result"].to_numpy(dtype=np.int64)

        clf = SVC(C=best_c, kernel="rbf", class_weight="balanced", random_state=seed)
        clf.fit(x_tr, y_tr)
        y_sc = clf.decision_function(x_te)

        rows.append(
            {
                "term": term,
                "n_train_pos": int(train_all["result"].sum()),
                "n_train_neg": int((train_all["result"] == 0).sum()),
                "n_test_pos": int(hold["result"].sum()),
                "n_test_neg": int((hold["result"] == 0).sum()),
                "selected_C": float(best_c),
                "AUROC": float(roc_auc_score(y_te, y_sc)),
                "AUPRC": float(average_precision_score(y_te, y_sc)),
                "Precision@10": precision_at_k(y_te, y_sc, 10),
            }
        )

        if i % 20 == 0 or i == len(terms):
            elapsed = time.time() - started
            rate = i / max(elapsed, 1e-9)
            eta = (len(terms) - i) / max(rate, 1e-9)
            logger(f"[{task_name}] terms={i}/{len(terms)} elapsed={elapsed/60:.1f}m rate={rate:.2f}/s eta={eta/60:.1f}m")

    return pd.DataFrame(rows).sort_values("term").reset_index(drop=True)


def summarize_seed(df: pd.DataFrame) -> dict[str, float]:
    return {
        "median_AUROC": float(df["AUROC"].median()),
        "median_AUPRC": float(df["AUPRC"].median()),
        "median_Precision@10": float(df["Precision@10"].median()),
    }


def describe_seed_distribution(values: np.ndarray) -> dict[str, float]:
    values = values.astype(float)
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "median": float(np.median(values)),
        "p2_5": float(np.percentile(values, 2.5)),
        "p97_5": float(np.percentile(values, 97.5)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run single-gene permutation and Gaussian controls for E0/E1/E2")
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103, 104, 105])
    args = parser.parse_args()

    started = time.time()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    logger = Logger(LOGS_DIR / "run_single_gene_controls.log")

    mapped = load_mapped_genes()
    reps = ["E0_static", "E1_null_L12", "E2_GTEx200_L12"]

    real_summary = pd.read_csv(RES_DIR / "representation_summary.csv")
    records: list[dict[str, Any]] = []

    for rep_name in reps:
        emb, genes = load_rep(rep_name)
        merged = mapped.merge(
            genes,
            left_on=["token_id", "bridge_gene_symbol"],
            right_on=["token_id", "gene_symbol"],
            how="inner",
            validate="one_to_one",
        ).sort_values("token_id").reset_index(drop=True)
        idx = merged["token_id"].to_numpy(dtype=np.int64) - 1
        x_real = emb[idx]
        entrez = merged["mapped_entrez_id"].to_numpy(dtype=str)
        gene_to_row = {g: i for i, g in enumerate(entrez)}

        logger(f"[setup] {rep_name}: mapped genes={len(entrez)}")
        rep_out = CONTROL_DIR / rep_name
        rep_out.mkdir(parents=True, exist_ok=True)

        gauss_mu = float(np.mean(x_real))
        gauss_sigma = float(np.std(x_real))
        if gauss_sigma <= 0:
            raise AssertionError(f"{rep_name} has non-positive sigma for Gaussian control")

        per_seed_summary: list[dict[str, Any]] = []
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(entrez))
            x_shuffle = x_real[perm]
            x_gauss = rng.normal(loc=gauss_mu, scale=gauss_sigma, size=x_real.shape).astype(np.float32)

            for ctrl_name, x_ctrl in [("shuffled", x_shuffle), ("gaussian", x_gauss)]:
                for task_name, cv_paths, holdout in [
                    ("GO", (GO_CV1, GO_CV2, GO_CV3), GO_HOLDOUT),
                    ("OMIM", (OMIM_CV1, OMIM_CV2, OMIM_CV3), OMIM_HOLDOUT),
                ]:
                    logger(f"[run] rep={rep_name} control={ctrl_name} seed={seed} task={task_name}")
                    df = evaluate_task(task_name, cv_paths, holdout, x_ctrl, gene_to_row, seed, logger)
                    out_dir = rep_out / task_name
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_file = out_dir / f"{ctrl_name}_seed{seed}.csv"
                    df.to_csv(out_file, index=False)
                    s = summarize_seed(df)
                    per_seed_summary.append(
                        {
                            "task": task_name,
                            "representation": rep_name,
                            "control": ctrl_name,
                            "seed": int(seed),
                            "n_terms": int(len(df)),
                            **s,
                        }
                    )

        seed_df = pd.DataFrame(per_seed_summary)
        seed_df.to_csv(rep_out / "seed_level_summary.csv", index=False)

        for task in ["GO", "OMIM"]:
            real_row = real_summary[(real_summary["task"] == task) & (real_summary["representation"] == rep_name)].iloc[0]
            for ctrl in ["shuffled", "gaussian"]:
                sub = seed_df[(seed_df["task"] == task) & (seed_df["control"] == ctrl)]
                for metric in ["median_AUROC", "median_AUPRC", "median_Precision@10"]:
                    dist = describe_seed_distribution(sub[metric].to_numpy(dtype=float))
                    records.append(
                        {
                            "task": task,
                            "representation": rep_name,
                            "control": ctrl,
                            "metric": metric,
                            "real_value": float(real_row[metric.replace("median_", "median_")]),
                            "seed_mean": dist["mean"],
                            "seed_sd": dist["sd"],
                            "seed_median": dist["median"],
                            "seed_p2_5": dist["p2_5"],
                            "seed_p97_5": dist["p97_5"],
                            "seed_min": dist["min"],
                            "seed_max": dist["max"],
                            "n_seeds": int(len(sub)),
                            "gaussian_mean_parameter": gauss_mu,
                            "gaussian_sd_parameter": gauss_sigma,
                        }
                    )

    summary_df = pd.DataFrame(records)
    summary_df.to_csv(CONTROL_DIR / "single_gene_control_summary.csv", index=False)

    # requested compact comparison table for AUROC
    rows = []
    for task in ["GO", "OMIM"]:
        for rep in reps:
            real_row = real_summary[(real_summary.task == task) & (real_summary.representation == rep)].iloc[0]
            sh = summary_df[
                (summary_df.task == task)
                & (summary_df.representation == rep)
                & (summary_df.control == "shuffled")
                & (summary_df.metric == "median_AUROC")
            ].iloc[0]
            ga = summary_df[
                (summary_df.task == task)
                & (summary_df.representation == rep)
                & (summary_df.control == "gaussian")
                & (summary_df.metric == "median_AUROC")
            ].iloc[0]
            rows.append(
                {
                    "Task": task,
                    "Representation": rep,
                    "Real_AUROC_median": float(real_row["median_AUROC"]),
                    "Shuffled_AUROC_median_over_seeds": float(sh["seed_median"]),
                    "Shuffled_AUROC_mean_over_seeds": float(sh["seed_mean"]),
                    "Shuffled_AUROC_sd_over_seeds": float(sh["seed_sd"]),
                    "Gaussian_AUROC_median_over_seeds": float(ga["seed_median"]),
                    "Gaussian_AUROC_mean_over_seeds": float(ga["seed_mean"]),
                    "Gaussian_AUROC_sd_over_seeds": float(ga["seed_sd"]),
                }
            )
    pd.DataFrame(rows).to_csv(CONTROL_DIR / "real_vs_controls_auroc_table.csv", index=False)

    report = {
        "status": "complete",
        "elapsed_seconds": time.time() - started,
        "seeds": args.seeds,
        "controls": {
            "shuffled": "permute gene->embedding row mapping; preserve matrix geometry and all splits/labels",
            "gaussian": "N(mu, sigma) using global mu/sigma of representation-specific filtered matrix",
        },
        "outputs": {
            "control_root": str(CONTROL_DIR.resolve()),
            "summary": str((CONTROL_DIR / "single_gene_control_summary.csv").resolve()),
            "compact_table": str((CONTROL_DIR / "real_vs_controls_auroc_table.csv").resolve()),
        },
    }
    (CONTROL_DIR / "single_gene_controls_report.json").write_text(json.dumps(report, indent=2) + "\n")
    logger("[done] single-gene controls complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
