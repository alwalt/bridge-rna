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
from sklearn.metrics import average_precision_score, make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "benchmarks" / "zhong_gene_biology"
REP_DIR = BENCH / "representations"
RES_DIR = BENCH / "results"
PAIR_DIR = RES_DIR / "gene_pair"
CONTROL_DIR = RES_DIR / "controls" / "gene_pair"
LOGS_DIR = BENCH / "logs"
MAP_PATH = BENCH / "bridge_gene_mapping.tsv"
REF_REPO = BENCH / "references" / "gene-embedding-benchmarks"

PAIR_SPLITS = {
    "SL": REF_REPO / "data/data_splits/gene_pair_benchmark/sl_nested_cv_splits.pkl",
    "NG": REF_REPO / "data/data_splits/gene_pair_benchmark/ng_nested_cv_splits.pkl",
    "TF": REF_REPO / "data/data_splits/gene_pair_benchmark/tf_nested_cv_splits.pkl",
}

OPS = ["sum", "product", "concat"]
C_VALUES = [0.1, 1.0, 10.0, 100.0, 1000.0]


class Logger:
    def __init__(self, path: Path, append: bool = False):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        if not append:
            path.write_text("")

    def __call__(self, msg: str) -> None:
        line = msg.rstrip("\n")
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def precision_at_k(y_true, y_scores, k=10):
    order = np.argsort(y_scores)[::-1]
    return float(np.mean(np.asarray(y_true)[order[: min(k, len(order))]]))


def load_rep(name: str) -> tuple[np.ndarray, pd.DataFrame]:
    emb = np.load(REP_DIR / name / "embeddings.npy").astype(np.float32)
    genes = pd.read_csv(REP_DIR / name / "genes.tsv", sep="\t")
    genes = genes[["token_id", "gene_symbol"]].copy()
    genes["token_id"] = genes["token_id"].astype(int)
    genes["gene_symbol"] = genes["gene_symbol"].astype(str).str.upper().str.strip()
    genes = genes.sort_values("token_id").reset_index(drop=True)
    if emb.shape != (len(genes), 512):
        raise AssertionError(f"Unexpected representation shape for {name}: {emb.shape}")
    return emb, genes


def load_mapped() -> pd.DataFrame:
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


def build_gene_row_mapping(rep_name: str, mapped: pd.DataFrame) -> tuple[np.ndarray, dict[str, int]]:
    emb, genes = load_rep(rep_name)
    merged = mapped.merge(
        genes,
        left_on=["token_id", "bridge_gene_symbol"],
        right_on=["token_id", "gene_symbol"],
        how="inner",
        validate="one_to_one",
    ).sort_values("token_id").reset_index(drop=True)
    idx = merged["token_id"].to_numpy(dtype=np.int64) - 1
    rep_emb = emb[idx]
    entrez = merged["mapped_entrez_id"].to_numpy(dtype=str)
    g2i = {g: i for i, g in enumerate(entrez)}
    return rep_emb, g2i


def filter_pairs(pairs: list[tuple[str, str]], labels: list[int], g2i: dict[str, int]):
    keep_local = []
    row1 = []
    row2 = []
    y = []
    for i, ((g1, g2), lbl) in enumerate(zip(pairs, labels)):
        g1 = str(g1)
        g2 = str(g2)
        if g1 in g2i and g2 in g2i:
            keep_local.append(i)
            row1.append(g2i[g1])
            row2.append(g2i[g2])
            y.append(int(lbl))
    return np.array(keep_local, dtype=np.int64), np.array(row1, dtype=np.int64), np.array(row2, dtype=np.int64), np.array(y, dtype=np.int64)


def build_features(emb: np.ndarray, row1: np.ndarray, row2: np.ndarray, op: str) -> np.ndarray:
    a = emb[row1]
    b = emb[row2]
    if op == "sum":
        return a + b
    if op == "product":
        return a * b
    if op == "concat":
        return np.hstack([a, b])
    raise ValueError(op)


def leakage_audit(task: str, fold: int, pairs: list[tuple[str, str]], train_idx: list[int], test_idx: list[int]) -> dict[str, Any]:
    tr_genes = set()
    te_genes = set()
    for idx in train_idx:
        g1, g2 = pairs[idx]
        tr_genes.add(str(g1))
        tr_genes.add(str(g2))
    for idx in test_idx:
        g1, g2 = pairs[idx]
        te_genes.add(str(g1))
        te_genes.add(str(g2))
    return {
        "task": task,
        "outer_fold": int(fold),
        "train_unique_genes": int(len(tr_genes)),
        "test_unique_genes": int(len(te_genes)),
        "train_test_gene_overlap": int(len(tr_genes & te_genes)),
        "n_train_pairs": int(len(train_idx)),
        "n_test_pairs": int(len(test_idx)),
    }


def run_nested_cv_for_condition(
    task: str,
    rep_name: str,
    op: str,
    emb: np.ndarray,
    pairs: list[tuple[str, str]],
    labels: list[int],
    cv_splits: dict[int, dict[str, Any]],
    g2i: dict[str, int],
    seed: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    keep_idx, row1, row2, y = filter_pairs(pairs, labels, g2i)
    if len(keep_idx) == 0:
        raise ValueError(f"No filtered pairs survive for {task}/{rep_name}/{op}")

    x = build_features(emb, row1, row2, op)
    kept_set = set(keep_idx.tolist())
    master_to_local = {m: i for i, m in enumerate(keep_idx.tolist())}
    filtered_pos = int(y.sum())
    filtered_neg = int((y == 0).sum())

    fold_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []

    pr10_scorer = make_scorer(precision_at_k, response_method="decision_function", k=10)

    for fold, split in cv_splits.items():
        train_master = [m for m in split["train_idx"] if m in kept_set]
        test_master = [m for m in split["test_idx"] if m in kept_set]
        if not train_master or not test_master:
            raise ValueError(f"Fold {fold} has empty train/test after filtering for {task}/{rep_name}/{op}")

        leakage_rows.append(leakage_audit(task, int(fold), pairs, train_master, test_master))

        train_local = np.array([master_to_local[m] for m in train_master], dtype=np.int64)
        test_local = np.array([master_to_local[m] for m in test_master], dtype=np.int64)

        x_train = x[train_local]
        y_train = y[train_local]
        x_test = x[test_local]
        y_test = y[test_local]

        train_master_set = set(train_master)
        train_master_to_pos = {m: i for i, m in enumerate(train_master)}
        inner_splits = []
        for tr_m, val_m in split["inner_splits"]:
            tr = [train_master_to_pos[m] for m in tr_m if m in train_master_set]
            va = [train_master_to_pos[m] for m in val_m if m in train_master_set]
            if tr and va:
                inner_splits.append((tr, va))
        if not inner_splits:
            raise ValueError(f"No valid inner splits in fold {fold} after filtering for {task}/{rep_name}/{op}")

        grid = GridSearchCV(
            SVC(class_weight="balanced", kernel="rbf", probability=False, random_state=seed),
            param_grid={"C": C_VALUES},
            scoring={"AUC": "roc_auc", "AUPRC": "average_precision", "PR@10": pr10_scorer},
            refit="AUC",
            cv=inner_splits,
            n_jobs=n_jobs,
            return_train_score=False,
        )
        grid.fit(x_train, y_train)

        scores = grid.decision_function(x_test)
        fold_rows.append(
            {
                "interaction_type": task,
                "representation": rep_name,
                "joining_operation": op,
                "outer_fold": int(fold),
                "best_C": float(grid.best_params_["C"]),
                "n_train": int(len(train_local)),
                "n_test": int(len(test_local)),
                "n_positive_test": int(y_test.sum()),
                "AUROC": float(roc_auc_score(y_test, scores)),
                "AUPRC": float(average_precision_score(y_test, scores)),
                "Precision@10": float(precision_at_k(y_test, scores, 10)),
            }
        )

    coverage = {
        "task": task,
        "representation": rep_name,
        "joining_operation": op,
        "original_pairs": int(len(labels)),
        "original_positive_pairs": int(np.sum(labels)),
        "original_negative_pairs": int(len(labels) - np.sum(labels)),
        "retained_pairs": int(len(y)),
        "retained_positive_pairs": int(filtered_pos),
        "retained_negative_pairs": int(filtered_neg),
        "retained_positive_percentage": float(filtered_pos / max(np.sum(labels), 1)),
    }
    return pd.DataFrame(fold_rows), leakage_rows, coverage


def summarize_conditions(all_fold_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, rep, op), grp in all_fold_rows.groupby(["interaction_type", "representation", "joining_operation"]):
        rows.append(
            {
                "interaction_type": task,
                "representation": rep,
                "joining_operation": op,
                "outer_folds": int(len(grp)),
                "AUROC_mean": float(grp["AUROC"].mean()),
                "AUROC_sd": float(grp["AUROC"].std(ddof=1)),
                "AUPRC_mean": float(grp["AUPRC"].mean()),
                "AUPRC_sd": float(grp["AUPRC"].std(ddof=1)),
                "Precision@10_mean": float(grp["Precision@10"].mean()),
                "Precision@10_sd": float(grp["Precision@10"].std(ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def comparison_summary(summary_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    metric_col = f"{metric}_mean"
    for (task, op), grp in summary_df.groupby(["interaction_type", "joining_operation"]):
        ref = {r.representation: getattr(r, metric_col) for r in grp.itertuples(index=False)}
        if not {"E0_static", "E1_null_L12", "E2_GTEx200_L12"}.issubset(ref.keys()):
            continue
        rows.extend(
            [
                {
                    "interaction_type": task,
                    "joining_operation": op,
                    "comparison": "E1-E0",
                    f"delta_{metric}": float(ref["E1_null_L12"] - ref["E0_static"]),
                },
                {
                    "interaction_type": task,
                    "joining_operation": op,
                    "comparison": "E2-E1",
                    f"delta_{metric}": float(ref["E2_GTEx200_L12"] - ref["E1_null_L12"]),
                },
                {
                    "interaction_type": task,
                    "joining_operation": op,
                    "comparison": "E2-E0",
                    f"delta_{metric}": float(ref["E2_GTEx200_L12"] - ref["E0_static"]),
                },
            ]
        )
    return pd.DataFrame(rows)


def run_real_pair_benchmark(
    seed: int,
    n_jobs: int,
    logger: Logger,
    tasks: list[str],
    reps: list[str],
    ops: list[str],
    resume: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mapped = load_mapped()
    rep_assets = {rep: build_gene_row_mapping(rep, mapped) for rep in reps}

    all_folds = []
    all_leak = []
    all_cov = []

    for task in tasks:
        path = PAIR_SPLITS[task]
        logger(f"[task] {task} loading {path}")
        with path.open("rb") as f:
            d = pickle.load(f)
        pairs = [(str(a), str(b)) for a, b in d["pairs"]]
        labels = [int(v) for v in d["labels"]]
        cv_splits = d["cv_splits"]

        for rep in reps:
            emb, g2i = rep_assets[rep]
            for op in ops:
                task_dir = PAIR_DIR / task
                task_dir.mkdir(parents=True, exist_ok=True)
                short = {"E0_static": "E0", "E1_null_L12": "E1", "E2_GTEx200_L12": "E2"}[rep]
                result_path = task_dir / f"{short}_{op}.csv"
                leak_path = task_dir / f"{short}_{op}_leakage.csv"
                coverage_path = task_dir / f"{short}_{op}_coverage.json"
                if resume and result_path.exists() and leak_path.exists() and coverage_path.exists():
                    logger(f"[resume] task={task} rep={rep} op={op} from {result_path}")
                    fold_df = pd.read_csv(result_path)
                    expected = {"interaction_type", "representation", "joining_operation", "outer_fold"}
                    if not expected.issubset(fold_df.columns) or len(fold_df) != len(cv_splits):
                        raise AssertionError(f"Invalid resumable result: {result_path}")
                    all_folds.append(fold_df)
                    all_leak.extend(pd.read_csv(leak_path).to_dict(orient="records"))
                    all_cov.append(json.loads(coverage_path.read_text()))
                    continue
                logger(f"[run] task={task} rep={rep} op={op}")
                fold_df, leak_rows, cov = run_nested_cv_for_condition(
                    task=task,
                    rep_name=rep,
                    op=op,
                    emb=emb,
                    pairs=pairs,
                    labels=labels,
                    cv_splits=cv_splits,
                    g2i=g2i,
                    seed=seed,
                    n_jobs=n_jobs,
                )
                all_folds.append(fold_df)
                all_leak.extend(leak_rows)
                all_cov.append(cov)

                fold_df.to_csv(result_path, index=False)
                pd.DataFrame(leak_rows).to_csv(leak_path, index=False)
                coverage_path.write_text(json.dumps(cov, indent=2) + "\n")

    fold_all = pd.concat(all_folds, ignore_index=True)
    leak_df = pd.DataFrame(all_leak)
    cov_df = pd.DataFrame(all_cov)
    return fold_all, leak_df, cov_df


def run_permutation_controls(seed_list: list[int], n_jobs: int, logger: Logger) -> pd.DataFrame:
    mapped = load_mapped()
    reps_all = ["E0_static", "E1_null_L12", "E2_GTEx200_L12"]
    reps_control = ["E1_null_L12", "E2_GTEx200_L12", "E0_static"]
    base_assets = {rep: build_gene_row_mapping(rep, mapped) for rep in reps_all}

    rows = []
    for task, path in PAIR_SPLITS.items():
        with path.open("rb") as f:
            d = pickle.load(f)
        pairs = [(str(a), str(b)) for a, b in d["pairs"]]
        labels = [int(v) for v in d["labels"]]
        cv_splits = d["cv_splits"]

        for rep in reps_control:
            emb_real, g2i = base_assets[rep]
            n = emb_real.shape[0]
            for seed in seed_list:
                rng = np.random.default_rng(seed)
                perm = rng.permutation(n)
                emb_perm = emb_real[perm]

                for op in OPS:
                    logger(f"[ctrl] task={task} rep={rep} op={op} seed={seed}")
                    fold_df, _, cov = run_nested_cv_for_condition(
                        task=task,
                        rep_name=rep,
                        op=op,
                        emb=emb_perm,
                        pairs=pairs,
                        labels=labels,
                        cv_splits=cv_splits,
                        g2i=g2i,
                        seed=seed,
                        n_jobs=n_jobs,
                    )
                    fold_df["control"] = "shuffled_gene_to_embedding"
                    fold_df["seed"] = int(seed)
                    fold_df["coverage_retained_pairs"] = int(cov["retained_pairs"])
                    rows.append(fold_df)

                    out_dir = CONTROL_DIR / task
                    out_dir.mkdir(parents=True, exist_ok=True)
                    short = {"E0_static": "E0", "E1_null_L12": "E1", "E2_GTEx200_L12": "E2"}[rep]
                    fold_df.to_csv(out_dir / f"{short}_{op}_seed{seed}.csv", index=False)

    return pd.concat(rows, ignore_index=True)


def summarize_control(control_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, rep, op), grp in control_df.groupby(["interaction_type", "representation", "joining_operation"]):
        seed_scores = grp.groupby("seed", as_index=False).agg(
            AUROC=("AUROC", "mean"),
            AUPRC=("AUPRC", "mean"),
            Precision10=("Precision@10", "mean"),
        )
        for metric, col in [("AUROC", "AUROC"), ("AUPRC", "AUPRC"), ("Precision@10", "Precision10")]:
            v = seed_scores[col].to_numpy(dtype=float)
            rows.append(
                {
                    "interaction_type": task,
                    "representation": rep,
                    "joining_operation": op,
                    "metric": metric,
                    "seed_mean": float(np.mean(v)),
                    "seed_sd": float(np.std(v, ddof=1) if len(v) > 1 else 0.0),
                    "seed_median": float(np.median(v)),
                    "seed_p2_5": float(np.percentile(v, 2.5)),
                    "seed_p97_5": float(np.percentile(v, 97.5)),
                    "n_seeds": int(len(v)),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Zhong paired-gene E0/E1/E2 benchmarks and permutation controls")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=3)
    parser.add_argument("--control-seeds", type=int, nargs="+", default=[101, 102, 103, 104, 105])
    parser.add_argument("--skip-controls", action="store_true")
    parser.add_argument("--tasks", nargs="+", choices=sorted(PAIR_SPLITS), default=sorted(PAIR_SPLITS))
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=["E0_static", "E1_null_L12", "E2_GTEx200_L12"],
        default=["E0_static", "E1_null_L12", "E2_GTEx200_L12"],
    )
    parser.add_argument("--operations", nargs="+", choices=OPS, default=OPS)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    started = time.time()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = Logger(LOGS_DIR / "run_gene_pair_ablation.log", append=not args.no_resume)
    PAIR_DIR.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)

    n_conditions = len(args.tasks) * len(args.representations) * len(args.operations)
    logger(f"[startup] real conditions={n_conditions}; outer folds=3; C grid={len(C_VALUES)}; resume={not args.no_resume}")
    fold_all, leak_df, cov_df = run_real_pair_benchmark(
        args.seed,
        args.n_jobs,
        logger,
        args.tasks,
        args.representations,
        args.operations,
        not args.no_resume,
    )
    fold_all.to_csv(RES_DIR / "pair_outer_fold_results_all.csv", index=False)
    leak_df.to_csv(RES_DIR / "pair_leakage_qc.csv", index=False)
    cov_df.to_csv(RES_DIR / "pair_coverage_recomputed.csv", index=False)

    # replicate requested filenames
    for task, short in [("SL", "sl"), ("NG", "ng"), ("TF", "tf")]:
        if task not in args.tasks:
            continue
        fold_all[fold_all["interaction_type"] == task].to_csv(RES_DIR / f"pair_{short}_results.csv", index=False)

    summary = summarize_conditions(fold_all)
    summary.to_csv(RES_DIR / "pair_representation_summary.csv", index=False)
    comparison_summary(summary, "AUROC").to_csv(RES_DIR / "pair_comparison_summary_AUROC.csv", index=False)
    comparison_summary(summary, "AUPRC").to_csv(RES_DIR / "pair_comparison_summary_AUPRC.csv", index=False)

    if not args.skip_controls:
        control_df = run_permutation_controls(args.control_seeds, args.n_jobs, logger)
        control_df.to_csv(CONTROL_DIR / "pair_permutation_fold_results.csv", index=False)
        summarize_control(control_df).to_csv(CONTROL_DIR / "pair_permutation_summary.csv", index=False)

    report = {
        "status": "complete_selected_matrix",
        "elapsed_seconds": time.time() - started,
        "seed": args.seed,
        "control_seeds": args.control_seeds,
        "selection": {
            "tasks": args.tasks,
            "representations": args.representations,
            "operations": args.operations,
            "controls_run": not args.skip_controls,
        },
        "outputs": {
            "pair_results": str((RES_DIR / "pair_outer_fold_results_all.csv").resolve()),
            "pair_summary": str((RES_DIR / "pair_representation_summary.csv").resolve()),
            "leakage_qc": str((RES_DIR / "pair_leakage_qc.csv").resolve()),
            "coverage": str((RES_DIR / "pair_coverage_recomputed.csv").resolve()),
            "controls_dir": str(CONTROL_DIR.resolve()),
        },
    }
    (RES_DIR / "run_gene_pair_ablation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    logger("[done] gene-pair ablation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
