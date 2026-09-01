#!/usr/bin/env python3
"""Prepare, embed, and evaluate frozen models on TCGA downstream tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from common import CONFIG, REPO_ROOT, RESULTS, WORK

TCGA_H5 = REPO_ROOT / "data/tcga/tcga_matrix.h5"
CDR = REPO_ROOT / "data/tcga/Survival_SupplementalTable_S1_20171025_xena_sp.tsv"
CDR_URL = (
    "https://pancanatlas.xenahubs.net/download/"
    "Survival_SupplementalTable_S1_20171025_xena_sp"
)
OUR_LENGTHS = REPO_ROOT / "data/gencode/gencode_v49_gene_exon_lengths.csv"
BULK_INFO = REPO_ROOT / "model/BulkFormer/data/bulkformer_gene_info.csv"
IMPUTATION_WORK = REPO_ROOT / "benchmarks/tcga_imputation/work"
IMPUTATION_RESULTS = REPO_ROOT / "benchmarks/tcga_imputation/results"
MODELS = ("ours_45.6m", "bulkformer_50m", "bulkformer_147m")

# The repository environment uses a Python 3.11 release candidate while current
# torch._dynamo expects APIs added in the final 3.11 release.
if not hasattr(sys, "get_int_max_str_digits"):
    def get_int_max_str_digits() -> int:
        return 4300
    sys.get_int_max_str_digits = get_int_max_str_digits  # type: ignore[attr-defined]
if not hasattr(sys, "set_int_max_str_digits"):
    def set_int_max_str_digits(maxdigits: int) -> None:
        del maxdigits
    sys.set_int_max_str_digits = set_int_max_str_digits  # type: ignore[attr-defined]


def say(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def download_cdr() -> None:
    if CDR.is_file():
        return
    say(f"downloading TCGA PanCanAtlas survival endpoints: {CDR_URL}")
    with urllib.request.urlopen(CDR_URL, timeout=120) as response:
        payload = response.read()
    CDR.write_bytes(payload)
    say(f"saved {CDR} ({len(payload):,} bytes)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def patient_table() -> pd.DataFrame:
    with h5py.File(TCGA_H5, "r") as handle:
        table = pd.DataFrame({
            "h5_row": np.arange(len(handle["meta/sampleid"])),
            "sample_id": decode(handle["meta/sampleid"][:]),
            "patient_id": decode(handle["meta/gdc_cases.submitter_id"][:]),
            "project_id": decode(handle["meta/gdc_cases.project.project_id"][:]),
            "sample_type": decode(handle["meta/gdc_cases.samples.sample_type"][:]),
        })
    table["cohort"] = table.project_id.str.removeprefix("TCGA-")
    table = table.loc[table.sample_type.eq("Primary Tumor")].copy()
    # One deterministic expression aliquot per patient prevents split leakage.
    table = table.sort_values(["patient_id", "sample_id"]).drop_duplicates("patient_id")
    table = table.reset_index(drop=True)
    return table


def build_cohorts() -> pd.DataFrame:
    download_cdr()
    samples = patient_table()
    cdr = pd.read_csv(CDR, sep="\t", low_memory=False)
    survival = cdr.rename(columns={"_PATIENT": "patient_id", "OS": "event", "OS.time": "time_days",
                                   "cancer type abbreviation": "cdr_cohort"})
    survival = survival[["patient_id", "event", "time_days", "cdr_cohort"]].drop_duplicates("patient_id")
    survival["event"] = pd.to_numeric(survival.event, errors="coerce")
    survival["time_days"] = pd.to_numeric(survival.time_days, errors="coerce")
    samples = samples.merge(survival, on="patient_id", how="left", validate="one_to_one")
    samples["survival_usable"] = samples.event.isin([0, 1]) & samples.time_days.gt(0)
    samples["classification_label"] = samples.cohort.replace({"GBM": "GBMLGG", "LGG": "GBMLGG"})
    wanted = set(CONFIG["classification_cohorts"])
    samples["classification_usable"] = samples.classification_label.isin(wanted)
    samples["matrix_row"] = np.arange(len(samples))
    samples.to_parquet(RESULTS / "cohort_manifest.parquet", index=False)
    summary = samples.groupby("cohort", as_index=False).agg(
        patients=("patient_id", "size"), survival_usable=("survival_usable", "sum"),
        events=("event", "sum"), classification_usable=("classification_usable", "sum"))
    summary.to_parquet(RESULTS / "cohort_summary.parquet", index=False)
    say(f"cohort: {len(samples):,} primary-tumor patients; "
        f"classification={samples.classification_usable.sum():,}; "
        f"survival={samples.survival_usable.sum():,}; cohorts={samples.cohort.nunique()}")
    return samples


def tpm_log1p(counts: np.ndarray, lengths_bp: np.ndarray) -> np.ndarray:
    rates = counts.astype(np.float64) / (lengths_bp[None, :] / 1000.0)
    totals = rates.sum(axis=1, keepdims=True)
    return np.log1p(np.divide(rates * 1e6, totals, out=np.zeros_like(rates), where=totals > 0)).astype(np.float32)


def prepare_expression(samples: pd.DataFrame) -> None:
    ours_path = WORK / "ours_log1p_tpm.npy"
    bulk_path = WORK / "bulkformer_log1p_tpm.npy"
    full_path = WORK / "tcga_full_log1p_cpm.npy"
    if ours_path.is_file() and bulk_path.is_file() and full_path.is_file():
        say("reusing prepared native expression matrices")
        return
    ours_genes = pd.read_parquet(IMPUTATION_WORK / "ours_genes.parquet")
    bulk_genes = pd.read_parquet(IMPUTATION_WORK / "bulkformer_genes.parquet")
    crosswalk = pd.read_csv(IMPUTATION_RESULTS / "tcga_hgnc_crosswalk.csv")
    usable = crosswalk.loc[crosswalk.mapping_status.isin(
        ["approved_symbol", "previous_symbol", "alias_symbol"])]
    approved_to_source = dict(zip(usable.approved_symbol.astype(str), usable.tcga_symbol.astype(str)))
    with h5py.File(TCGA_H5, "r") as handle:
        source_genes = decode(handle["meta/genes"][:])
        source_index = {gene.upper(): i for i, gene in enumerate(source_genes)}
        rows = samples.h5_row.to_numpy(int)
        row_order = np.argsort(rows)
        raw_sorted = np.asarray(handle["data/expression"][rows[row_order], :], dtype=np.float32)
        raw = raw_sorted[np.argsort(row_order)]
        def extract(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
            indices = np.asarray([source_index.get(approved_to_source.get(str(g), "").upper(), -1)
                                  if pd.notna(g) else -1 for g in table.approved_symbol], dtype=int)
            observed = indices >= 0
            out = np.zeros((len(rows), len(table)), dtype=np.float32)
            selected = indices[observed]
            out[:, observed] = raw[:, selected]
            return out, observed
        our_counts, our_observed = extract(ours_genes)
        bulk_counts, bulk_observed = extract(bulk_genes)
    lengths = pd.read_csv(OUR_LENGTHS)
    length_lookup = dict(zip(lengths.gene_symbol.astype(str), lengths.exon_length))
    our_lengths = np.asarray([length_lookup[g] for g in ours_genes.gene.astype(str)], dtype=float)
    bulk_info = pd.read_csv(BULK_INFO)
    ours = tpm_log1p(our_counts, our_lengths)
    bulk = tpm_log1p(bulk_counts, bulk_info.gene_length.to_numpy(float))
    bulk[:, ~bulk_observed] = -10.0
    totals = raw.sum(axis=1, keepdims=True, dtype=np.float64)
    full = np.log1p(np.divide(raw * 1e6, totals, out=np.zeros_like(raw), where=totals > 0))
    np.save(ours_path, ours); np.save(bulk_path, bulk)
    np.save(full_path, full.astype(np.float32))
    ours_genes.assign(tcga_observed=our_observed).to_parquet(WORK / "ours_genes.parquet", index=False)
    bulk_genes.assign(tcga_observed=bulk_observed).to_parquet(WORK / "bulkformer_genes.parquet", index=False)
    say(f"prepared expression: ours={ours.shape}; BulkFormer={bulk.shape}")


def load_model(name: str, device: torch.device):
    sys.path.insert(0, str(REPO_ROOT / "benchmarks/tcga_imputation/pipeline"))
    from model_adapters import load_bulkformer, load_ours
    return load_ours(device) if name == "ours_45.6m" else load_bulkformer(name, device)


@torch.inference_mode()
def extract_embeddings(name: str, device: torch.device, heartbeat: int) -> None:
    output = WORK / f"{name}_embeddings.npy"
    if output.is_file():
        say(f"reusing {output.name}"); return
    space = "ours" if name == "ours_45.6m" else "bulkformer"
    matrix = np.load(WORK / f"{space}_log1p_tpm.npy", mmap_mode="r")
    batch_size = int(CONFIG["embedding_batch_sizes"][name])
    model = load_model(name, device).eval()
    vectors, started, last = [], time.monotonic(), time.monotonic()
    say(f"embedding {name}: samples={len(matrix):,} batch={batch_size} device={device}")
    for start in range(0, len(matrix), batch_size):
        batch = torch.as_tensor(matrix[start:start + batch_size], dtype=torch.float32, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            if name == "ours_45.6m":
                embedding = model.encode(batch, normalize=False)
            else:
                tokens = model(batch, mask_prob=0.0, output_expr=False)
                embedding = tokens.mean(dim=1)
        vectors.append(embedding.float().cpu().numpy())
        now = time.monotonic()
        if now - last >= heartbeat or start + batch_size >= len(matrix):
            done = min(start + batch_size, len(matrix)); elapsed = now - started
            say(f"heartbeat {name}: {done:,}/{len(matrix):,} elapsed={elapsed/60:.1f}m "
                f"rate={done/max(elapsed, 1e-9):.2f}/s")
            last = now
    np.save(output, np.concatenate(vectors))
    say(f"saved {output}")
    del model
    torch.cuda.empty_cache() if device.type == "cuda" else None


def representation(name: str) -> np.ndarray:
    if name == "raw_expression":
        return np.load(WORK / "ours_log1p_tpm.npy", mmap_mode="r")
    return np.load(WORK / f"{name}_embeddings.npy", mmap_mode="r")


def reduced(train: np.ndarray, test: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    components = min(int(CONFIG["probe_pca_components"]), train.shape[1], len(train) - 1)
    pipe = make_pipeline(StandardScaler(), PCA(components, random_state=seed, svd_solver="randomized"))
    return pipe.fit_transform(train), pipe.transform(test)


def run_classification(samples: pd.DataFrame) -> None:
    output = RESULTS / "classification_per_split.parquet"
    if output.is_file(): say("reusing classification results"); return
    cohort = samples.loc[samples.classification_usable].copy()
    indices = cohort.matrix_row.to_numpy(int); y = cohort.classification_label.to_numpy(str)
    rows = []
    for seed in CONFIG["split_seeds"]:
        train, test = train_test_split(np.arange(len(cohort)), test_size=CONFIG["test_fraction"],
            random_state=seed, stratify=y)
        for name in (*MODELS, "raw_expression"):
            x = representation(name)[indices]
            xtrain, xtest = reduced(np.asarray(x[train]), np.asarray(x[test]), seed)
            for probe, estimator in (
                ("logistic_regression", LogisticRegression(max_iter=3000, class_weight="balanced")),
                ("linear_svm", LinearSVC(class_weight="balanced", max_iter=10000))):
                estimator.fit(xtrain, y[train]); predicted = estimator.predict(xtest)
                rows.append({"seed": seed, "representation": name, "probe": probe,
                    "train_patients": len(train), "test_patients": len(test),
                    "macro_f1": f1_score(y[test], predicted, average="macro"),
                    "weighted_f1": f1_score(y[test], predicted, average="weighted"),
                    "balanced_accuracy": balanced_accuracy_score(y[test], predicted)})
                say(f"classification seed={seed} model={name} probe={probe} "
                    f"macro_f1={rows[-1]['macro_f1']:.4f}")
    pd.DataFrame(rows).to_parquet(output, index=False)


def safe_cindex(frame: pd.DataFrame, risk: np.ndarray) -> float:
    if len(frame) < 2 or frame.event.sum() == 0:
        return np.nan
    try: return float(concordance_index(frame.time_days, -risk, frame.event))
    except ZeroDivisionError: return np.nan


def run_survival(samples: pd.DataFrame) -> None:
    output = RESULTS / "survival_per_split.parquet"
    if output.is_file(): say("reusing survival results"); return
    cohort = samples.loc[samples.survival_usable].copy()
    indices = cohort.matrix_row.to_numpy(int)
    strata = cohort.cohort.astype(str)
    rows = []
    for seed in CONFIG["split_seeds"]:
        train, test = train_test_split(np.arange(len(cohort)), test_size=CONFIG["test_fraction"],
            random_state=seed, stratify=strata)
        for name in (*MODELS, "raw_expression"):
            x = representation(name)[indices]
            xtrain, xtest = reduced(np.asarray(x[train]), np.asarray(x[test]), seed)
            columns = [f"x{i}" for i in range(xtrain.shape[1])]
            fit = pd.DataFrame(xtrain, columns=columns)
            fit["time_days"] = cohort.time_days.to_numpy(float)[train]
            fit["event"] = cohort.event.to_numpy(int)[train]
            model = CoxPHFitter(penalizer=float(CONFIG["survival_penalizer"]))
            model.fit(fit, duration_col="time_days", event_col="event", show_progress=False)
            risk = model.predict_partial_hazard(pd.DataFrame(xtest, columns=columns)).to_numpy()
            held = cohort.iloc[test].reset_index(drop=True)
            per_cohort = held.assign(risk=risk).groupby("cohort").apply(
                lambda group: safe_cindex(group, group.risk.to_numpy()), include_groups=False)
            valid = per_cohort.dropna()
            sizes = held.cohort.value_counts().reindex(valid.index)
            row = {"seed": seed, "representation": name, "probe": "pca128_cox",
                "train_patients": len(train), "test_patients": len(test),
                "events_test": int(held.event.sum()), "c_index": safe_cindex(held, risk),
                "weighted_c_index": float(np.average(valid, weights=sizes)) if len(valid) else np.nan,
                "macro_c_index": float(valid.mean()) if len(valid) else np.nan,
                "cohorts_scored": len(valid)}
            rows.append(row)
            say(f"survival seed={seed} model={name} c_index={row['c_index']:.4f} "
                f"weighted={row['weighted_c_index']:.4f}")
    pd.DataFrame(rows).to_parquet(output, index=False)


class ClassificationHead(nn.Module):
    def __init__(self, inputs: int, classes: int):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(inputs, 256), nn.SELU(),
            nn.Linear(256, 128), nn.SELU(), nn.Linear(128, classes))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class SurvivalHead(nn.Module):
    def __init__(self, inputs: int):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(inputs, 512), nn.SELU(),
            nn.Linear(512, 256), nn.SELU(), nn.LayerNorm(256), nn.Linear(256, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


def head_input(name: str) -> np.ndarray:
    if name == "full_expression_25150":
        return np.load(WORK / "tcga_full_log1p_cpm.npy", mmap_mode="r")
    return representation(name)


def standardize_native(matrix: np.ndarray, train: np.ndarray,
                       *others: np.ndarray) -> tuple[np.ndarray, ...]:
    train_values = np.asarray(matrix[train], dtype=np.float32)
    mean = train_values.mean(axis=0, dtype=np.float64).astype(np.float32)
    sd = train_values.std(axis=0, dtype=np.float64).astype(np.float32)
    sd[sd < 1e-6] = 1.0
    result = [(train_values - mean) / sd]
    result.extend((np.asarray(matrix[index], dtype=np.float32) - mean) / sd for index in others)
    return tuple(result)


def best_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def run_mlp_classification(samples: pd.DataFrame, device: torch.device) -> None:
    output = RESULTS / "classification_mlp_per_split.parquet"
    if output.is_file(): say("reusing native MLP classification results"); return
    cohort = samples.loc[samples.classification_usable].copy()
    source_rows = cohort.matrix_row.to_numpy(int)
    labels = sorted(cohort.classification_label.unique())
    label_index = {label: index for index, label in enumerate(labels)}
    y = cohort.classification_label.map(label_index).to_numpy(int)
    names = (*MODELS, "full_expression_25150")
    rows = []
    for seed in CONFIG["split_seeds"]:
        train_all, test = train_test_split(np.arange(len(cohort)), test_size=CONFIG["test_fraction"],
            random_state=seed, stratify=y)
        train, validation = train_test_split(train_all, test_size=.125,
            random_state=seed + 1000, stratify=y[train_all])
        for name in names:
            torch.manual_seed(seed); np.random.seed(seed)
            matrix = head_input(name)
            xtrain, xvalidation, xtest = standardize_native(matrix, source_rows[train],
                source_rows[validation], source_rows[test])
            model = ClassificationHead(xtrain.shape[1], len(labels)).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(CONFIG["head_learning_rate"]),
                weight_decay=float(CONFIG["head_weight_decay"]))
            loss_fn = nn.CrossEntropyLoss()
            train_y = torch.as_tensor(y[train], dtype=torch.long)
            validation_x = torch.as_tensor(xvalidation, dtype=torch.float32, device=device)
            validation_y = torch.as_tensor(y[validation], dtype=torch.long, device=device)
            generator = torch.Generator().manual_seed(seed)
            loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
                torch.as_tensor(xtrain), train_y), batch_size=int(CONFIG["head_batch_size"]),
                shuffle=True, generator=generator)
            optimum, state, stale = np.inf, None, 0
            for epoch in range(int(CONFIG["head_max_epochs"])):
                model.train()
                for batch_x, batch_y in loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = loss_fn(model(batch_x), batch_y); loss.backward(); optimizer.step()
                model.eval()
                with torch.no_grad(): value = float(loss_fn(model(validation_x), validation_y))
                if value < optimum - 1e-5:
                    optimum, state, stale = value, best_state(model), 0
                else: stale += 1
                if stale >= int(CONFIG["head_patience"]): break
            assert state is not None; model.load_state_dict(state); model.eval()
            with torch.no_grad():
                predicted = model(torch.as_tensor(xtest, dtype=torch.float32, device=device)).argmax(1).cpu().numpy()
            row = {"seed": seed, "representation": name, "probe": "native_mlp_256_128",
                "native_features": matrix.shape[1], "epochs": epoch + 1,
                "train_patients": len(train), "validation_patients": len(validation),
                "test_patients": len(test), "macro_f1": f1_score(y[test], predicted, average="macro"),
                "weighted_f1": f1_score(y[test], predicted, average="weighted"),
                "balanced_accuracy": balanced_accuracy_score(y[test], predicted)}
            rows.append(row); say(f"native classification seed={seed} model={name} "
                f"weighted_f1={row['weighted_f1']:.4f} epochs={epoch+1}")
            del model, optimizer, xtrain, xvalidation, xtest, validation_x, validation_y
            if device.type == "cuda": torch.cuda.empty_cache()
    pd.DataFrame(rows).to_parquet(output, index=False)


def cox_loss(risk: torch.Tensor, duration: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(duration, descending=True)
    ordered_risk, ordered_event = risk[order], event[order]
    log_risk = torch.logcumsumexp(ordered_risk, dim=0)
    return -((ordered_risk - log_risk) * ordered_event).sum() / ordered_event.sum().clamp_min(1)


def run_mlp_survival(samples: pd.DataFrame, device: torch.device) -> None:
    output = RESULTS / "survival_mlp_per_split.parquet"
    if output.is_file(): say("reusing native MLP survival results"); return
    cohort = samples.loc[samples.survival_usable].copy()
    source_rows = cohort.matrix_row.to_numpy(int); strata = cohort.cohort.astype(str)
    durations = cohort.time_days.to_numpy(np.float32); events = cohort.event.to_numpy(np.float32)
    names = (*MODELS, "full_expression_25150"); rows = []
    for seed in CONFIG["split_seeds"]:
        train_all, test = train_test_split(np.arange(len(cohort)), test_size=CONFIG["test_fraction"],
            random_state=seed, stratify=strata)
        train, validation = train_test_split(train_all, test_size=.125,
            random_state=seed + 1000, stratify=strata.iloc[train_all])
        for name in names:
            torch.manual_seed(seed); np.random.seed(seed)
            matrix = head_input(name)
            xtrain, xvalidation, xtest = standardize_native(matrix, source_rows[train],
                source_rows[validation], source_rows[test])
            model = SurvivalHead(xtrain.shape[1]).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(CONFIG["head_learning_rate"]),
                weight_decay=float(CONFIG["head_weight_decay"]))
            tensors = [torch.as_tensor(value, dtype=torch.float32, device=device) for value in
                (xtrain, durations[train], events[train], xvalidation, durations[validation], events[validation])]
            train_x, train_t, train_e, val_x, val_t, val_e = tensors
            optimum, state, stale = np.inf, None, 0
            for epoch in range(int(CONFIG["head_max_epochs"])):
                model.train(); optimizer.zero_grad(set_to_none=True)
                loss = cox_loss(model(train_x), train_t, train_e); loss.backward(); optimizer.step()
                model.eval()
                with torch.no_grad(): value = float(cox_loss(model(val_x), val_t, val_e))
                if value < optimum - 1e-5:
                    optimum, state, stale = value, best_state(model), 0
                else: stale += 1
                if stale >= int(CONFIG["head_patience"]): break
            assert state is not None; model.load_state_dict(state); model.eval()
            with torch.no_grad(): risk = model(torch.as_tensor(xtest, dtype=torch.float32, device=device)).cpu().numpy()
            held = cohort.iloc[test].reset_index(drop=True)
            per_cohort = held.assign(risk=risk).groupby("cohort").apply(
                lambda group: safe_cindex(group, group.risk.to_numpy()), include_groups=False).dropna()
            sizes = held.cohort.value_counts().reindex(per_cohort.index)
            row = {"seed": seed, "representation": name, "probe": "native_mlp_512_256_cox",
                "native_features": matrix.shape[1], "epochs": epoch + 1,
                "train_patients": len(train), "validation_patients": len(validation),
                "test_patients": len(test), "events_test": int(held.event.sum()),
                "c_index": safe_cindex(held, risk),
                "weighted_c_index": float(np.average(per_cohort, weights=sizes)),
                "macro_c_index": float(per_cohort.mean()), "cohorts_scored": len(per_cohort)}
            rows.append(row); say(f"native survival seed={seed} model={name} "
                f"c_index={row['c_index']:.4f} weighted={row['weighted_c_index']:.4f} epochs={epoch+1}")
            del model, optimizer, xtrain, xvalidation, xtest, tensors
            if device.type == "cuda": torch.cuda.empty_cache()
    pd.DataFrame(rows).to_parquet(output, index=False)


def aggregate() -> None:
    for task, metrics in {"classification": ["macro_f1", "weighted_f1", "balanced_accuracy"],
                          "survival": ["c_index", "weighted_c_index", "macro_c_index"]}.items():
        data = pd.read_parquet(RESULTS / f"{task}_per_split.parquet")
        rows = []
        for keys, frame in data.groupby(["representation", "probe"]):
            row = {"representation": keys[0], "probe": keys[1], "splits": len(frame)}
            for metric in metrics:
                row[f"{metric}_mean"] = frame[metric].mean()
                row[f"{metric}_sd"] = frame[metric].std(ddof=1)
            rows.append(row)
        summary = pd.DataFrame(rows)
        summary.to_parquet(RESULTS / f"{task}_summary.parquet", index=False)
        summary.to_csv(RESULTS / f"{task}_summary.csv", index=False)
    for task, metrics in {"classification_mlp": ["macro_f1", "weighted_f1", "balanced_accuracy"],
                          "survival_mlp": ["c_index", "weighted_c_index", "macro_c_index"]}.items():
        data = pd.read_parquet(RESULTS / f"{task}_per_split.parquet")
        rows = []
        for keys, frame in data.groupby(["representation", "probe"]):
            row = {"representation": keys[0], "probe": keys[1], "splits": len(frame),
                   "native_features": int(frame.native_features.iloc[0])}
            for metric in metrics:
                row[f"{metric}_mean"] = frame[metric].mean()
                row[f"{metric}_sd"] = frame[metric].std(ddof=1)
            rows.append(row)
        summary = pd.DataFrame(rows)
        summary.to_parquet(RESULTS / f"{task}_summary.parquet", index=False)
        summary.to_csv(RESULTS / f"{task}_summary.csv", index=False)
    published = pd.DataFrame([
        {"task": "five_cohort_classification", "method": "NMF-128+SVM (published)",
         "metric": "weighted_f1", "mean": .898, "sd": .018},
        {"task": "five_cohort_classification", "method": "PCA-256+SVM (published)",
         "metric": "weighted_f1", "mean": .968, "sd": .010},
        {"task": "five_cohort_classification", "method": "BulkRNABert(GTEx ENCODE)+SVM",
         "metric": "weighted_f1", "mean": .977, "sd": .008},
        {"task": "five_cohort_classification", "method": "BulkRNABert(TCGA)+SVM",
         "metric": "weighted_f1", "mean": .991, "sd": .005},
        {"task": "pan_cancer_classification", "method": "BulkRNABert(GTEx ENCODE)+SVM",
         "metric": "macro_f1", "mean": .887, "sd": .005},
        {"task": "pan_cancer_classification", "method": "BulkRNABert(GTEx ENCODE)+SVM",
         "metric": "weighted_f1", "mean": .918, "sd": .005},
        {"task": "pan_cancer_survival", "method": "BulkRNABert(GTEx ENCODE)",
         "metric": "c_index", "mean": .753, "sd": .009},
        {"task": "pan_cancer_survival", "method": "BulkRNABert(GTEx ENCODE)",
         "metric": "weighted_c_index", "mean": .621, "sd": .015},
        {"task": "pan_cancer_survival", "method": "BulkRNABert(TCGA)",
         "metric": "c_index", "mean": .765, "sd": .011},
        {"task": "pan_cancer_survival", "method": "BulkRNABert(TCGA)",
         "metric": "weighted_c_index", "mean": .642, "sd": .014},
        {"task": "pan_cancer_survival", "method": "MAE (published)",
         "metric": "c_index", "mean": .756, "sd": .010},
        {"task": "pan_cancer_survival", "method": "CustOmics RNA-only (published)",
         "metric": "weighted_c_index", "mean": .630, "sd": .020},
    ])
    published.to_parquet(RESULTS / "published_reference_results.parquet", index=False)
    provenance = {"tcga_h5": str(TCGA_H5), "tcga_h5_sha256": sha256(TCGA_H5),
        "survival_source": CDR_URL, "survival_sha256": sha256(CDR), "config": CONFIG,
        "frozen_encoders": True, "fine_tuning": False,
        "published_results_are_not_on_local_splits": True}
    (RESULTS / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--heartbeat-seconds", type=int, default=60)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    args = parser.parse_args()
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    samples = build_cohorts(); prepare_expression(samples)
    for name in args.models: extract_embeddings(name, device, args.heartbeat_seconds)
    missing = [name for name in MODELS if not (WORK / f"{name}_embeddings.npy").is_file()]
    if missing: raise RuntimeError(f"Missing required embeddings before evaluation: {missing}")
    run_classification(samples); run_survival(samples)
    run_mlp_classification(samples, device); run_mlp_survival(samples, device); aggregate()
    say("benchmark complete")


if __name__ == "__main__":
    main()
