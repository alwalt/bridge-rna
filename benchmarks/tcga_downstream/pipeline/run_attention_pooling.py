#!/usr/bin/env python3
"""Compare frozen-FM mean pooling with learned attention pooling on TCGA tasks.

The encoder is always frozen. Contextual gene tokens are generated on demand so
the approximately 150 GB float16 token tensor is never materialized on disk.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CONFIG, REPO_ROOT, RESULTS, WORK  # noqa: E402
from run_benchmark import (  # noqa: E402
    ClassificationHead,
    SurvivalHead,
    best_state,
    build_cohorts,
    cox_loss,
    prepare_expression,
    safe_cindex,
)


MODEL_CONFIG = REPO_ROOT / "model/r7hnr92k/config.json"
MEAN_EMBEDDINGS = WORK / "ours_45.6m_embeddings.npy"
EXPRESSION = WORK / "ours_log1p_tpm.npy"


def say(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def verify_input_contract() -> None:
    config = json.loads(MODEL_CONFIG.read_text())
    if config.get("normalization") != "log1p_tpm":
        raise RuntimeError(f"Checkpoint normalization is {config.get('normalization')!r}, not log1p_tpm")
    if not EXPRESSION.is_file():
        raise FileNotFoundError(f"Missing FM input matrix: {EXPRESSION}")
    matrix = np.load(EXPRESSION, mmap_mode="r")
    if matrix.shape[1] != int(config["num_genes"]):
        raise RuntimeError(f"FM input has {matrix.shape[1]} genes; checkpoint expects {config['num_genes']}")
    say(f"verified FM input: log1p(TPM), shape={matrix.shape}, checkpoint={config['run_id']}")
    (RESULTS / "attention_pooling_provenance.json").write_text(json.dumps({
        "checkpoint": str(MODEL_CONFIG), "checkpoint_run_id": config["run_id"],
        "checkpoint_normalization": config["normalization"], "input_matrix": str(EXPRESSION),
        "input_shape": list(matrix.shape), "encoder_frozen": True,
        "token_pooling": ["mean", "learned_attention"], "tokens_flattened": False,
    }, indent=2) + "\n")


def load_frozen_encoder(device: torch.device) -> nn.Module:
    sys.path.insert(0, str(REPO_ROOT / "benchmarks/tcga_imputation/pipeline"))
    from model_adapters import load_ours

    encoder = load_ours(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise AssertionError("The FM encoder is not fully frozen")
    return encoder


class AttentionPool(nn.Module):
    """Content-based attention assigning one normalized weight to every gene."""

    def __init__(self, hidden: int = 512):
        super().__init__()
        self.score = nn.Linear(hidden, 1, bias=False)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.score(tokens).squeeze(-1).float()
        weights = torch.softmax(logits, dim=1)
        pooled = torch.sum(tokens.float() * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class AttentionClassification(nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        self.pool = AttentionPool(512)
        self.head = ClassificationHead(512, classes)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled, weights = self.pool(tokens)
        return self.head(pooled), weights


class AttentionSurvival(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = AttentionPool(512)
        self.head = SurvivalHead(512)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled, weights = self.pool(tokens)
        return self.head(pooled), weights


def split_indices(cohort: pd.DataFrame, seed: int, task: str) -> tuple[np.ndarray, ...]:
    indices = np.arange(len(cohort))
    strata = cohort.classification_label if task == "classification" else cohort.cohort.astype(str)
    train_all, test = train_test_split(indices, test_size=CONFIG["test_fraction"],
        random_state=seed, stratify=strata)
    train, validation = train_test_split(train_all, test_size=.125,
        random_state=seed + 1000, stratify=strata.iloc[train_all])
    return train, validation, test


def mean_pool_scaler(source_rows: np.ndarray, train: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    means = np.load(MEAN_EMBEDDINGS, mmap_mode="r")
    values = np.asarray(means[source_rows[train]], dtype=np.float32)
    center = values.mean(0, dtype=np.float64).astype(np.float32)
    scale = values.std(0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return torch.from_numpy(center), torch.from_numpy(scale)


def contextual_tokens(encoder: nn.Module, expression: np.ndarray, rows: np.ndarray,
                      device: torch.device) -> torch.Tensor:
    values = torch.as_tensor(np.asarray(expression[rows]), dtype=torch.float32, device=device)
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16,
                                        enabled=device.type == "cuda"):
        tokens = encoder._encode_hidden(values)
    return tokens.detach()


def batches(indices: np.ndarray, batch_size: int, shuffle_seed: int | None = None):
    order = np.asarray(indices).copy()
    if shuffle_seed is not None:
        np.random.default_rng(shuffle_seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        yield order[start:start + batch_size]


def evaluate_classification(model, encoder, expression, source_rows, indices, labels,
                            center, scale, device, batch_size):
    del labels
    model.eval(); outputs = []
    with torch.no_grad():
        for batch in batches(indices, batch_size):
            tokens = contextual_tokens(encoder, expression, source_rows[batch], device)
            logits, _ = model((tokens.float() - center) / scale)
            outputs.append(logits.cpu())
    return torch.cat(outputs)


def run_classification(samples: pd.DataFrame, device: torch.device, batch_size: int,
                       heartbeat: int, seeds: list[int]) -> None:
    output = RESULTS / "classification_attention_per_split.parquet"
    existing = pd.read_parquet(output) if output.is_file() else pd.DataFrame()
    cohort = samples.loc[samples.classification_usable].copy().reset_index(drop=True)
    source_rows = cohort.matrix_row.to_numpy(int)
    names = sorted(cohort.classification_label.unique())
    y = cohort.classification_label.map({name: i for i, name in enumerate(names)}).to_numpy(int)
    expression = np.load(EXPRESSION, mmap_mode="r")
    encoder = load_frozen_encoder(device)
    rows = existing.to_dict("records")
    for seed in seeds:
        if not existing.empty and seed in set(existing.seed):
            say(f"classification seed={seed}: reusing completed result"); continue
        train, validation, test = split_indices(cohort, seed, "classification")
        center, scale = mean_pool_scaler(source_rows, train)
        center, scale = center.to(device), scale.to(device)
        torch.manual_seed(seed); np.random.seed(seed)
        model = AttentionClassification(len(names)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(CONFIG["head_learning_rate"]),
            weight_decay=float(CONFIG["head_weight_decay"]))
        loss_fn = nn.CrossEntropyLoss(); optimum, state, stale = np.inf, None, 0
        started = time.monotonic(); last_heartbeat = started
        for epoch in range(int(CONFIG["head_max_epochs"])):
            model.train(); total_loss = 0.0; seen = 0
            for batch in batches(train, batch_size, seed * 100000 + epoch):
                tokens = contextual_tokens(encoder, expression, source_rows[batch], device)
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model((tokens.float() - center) / scale)
                target = torch.as_tensor(y[batch], dtype=torch.long, device=device)
                loss = loss_fn(logits, target); loss.backward(); optimizer.step()
                total_loss += float(loss) * len(batch); seen += len(batch)
                now = time.monotonic()
                if now - last_heartbeat >= heartbeat:
                    say(f"heartbeat attention classification seed={seed} epoch={epoch+1} "
                        f"samples={seen:,}/{len(train):,} elapsed={(now-started)/60:.1f}m")
                    last_heartbeat = now
            validation_logits = evaluate_classification(model, encoder, expression, source_rows,
                validation, y, center, scale, device, batch_size)
            value = float(loss_fn(validation_logits,
                torch.as_tensor(y[validation], dtype=torch.long)))
            if value < optimum - 1e-5: optimum, state, stale = value, best_state(model), 0
            else: stale += 1
            say(f"attention classification seed={seed} epoch={epoch+1} train_loss={total_loss/seen:.5f} "
                f"val_loss={value:.5f} elapsed={(time.monotonic()-started)/60:.1f}m")
            if stale >= int(CONFIG["head_patience"]): break
        assert state is not None; model.load_state_dict(state)
        predicted = evaluate_classification(model, encoder, expression, source_rows,
            test, y, center, scale, device, batch_size).argmax(1).numpy()
        row = {"seed": seed, "representation": "ours_45.6m_attention", "pooling": "learned_attention",
            "probe": "attention_pool_mlp_256_128", "native_features": 512, "epochs": epoch + 1,
            "train_patients": len(train), "validation_patients": len(validation), "test_patients": len(test),
            "macro_f1": f1_score(y[test], predicted, average="macro"),
            "weighted_f1": f1_score(y[test], predicted, average="weighted"),
            "balanced_accuracy": balanced_accuracy_score(y[test], predicted)}
        rows.append(row); pd.DataFrame(rows).to_parquet(output, index=False)
        say(f"attention classification seed={seed} weighted_f1={row['weighted_f1']:.4f}")
    del encoder


def risks_for(model, encoder, expression, source_rows, indices, center, scale, device, batch_size,
              with_grad: bool = False, external_gradient: torch.Tensor | None = None,
              heartbeat: int | None = None, label: str = ""):
    output = []
    offset = 0; started = last = time.monotonic()
    for batch in batches(indices, batch_size):
        tokens = contextual_tokens(encoder, expression, source_rows[batch], device)
        risk, _ = model((tokens.float() - center) / scale)
        if with_grad:
            assert external_gradient is not None
            risk.backward(external_gradient[offset:offset + len(batch)])
        else:
            output.append(risk.detach())
        offset += len(batch)
        now = time.monotonic()
        if heartbeat and (now - last >= heartbeat or offset == len(indices)):
            say(f"heartbeat {label} samples={offset:,}/{len(indices):,} elapsed={(now-started)/60:.1f}m")
            last = now
    return None if with_grad else torch.cat(output)


def run_survival(samples: pd.DataFrame, device: torch.device, batch_size: int,
                 heartbeat: int, seeds: list[int]) -> None:
    output = RESULTS / "survival_attention_per_split.parquet"
    existing = pd.read_parquet(output) if output.is_file() else pd.DataFrame()
    cohort = samples.loc[samples.survival_usable].copy().reset_index(drop=True)
    source_rows = cohort.matrix_row.to_numpy(int)
    duration = torch.as_tensor(cohort.time_days.to_numpy(np.float32), device=device)
    event = torch.as_tensor(cohort.event.to_numpy(np.float32), device=device)
    expression = np.load(EXPRESSION, mmap_mode="r"); encoder = load_frozen_encoder(device)
    rows = existing.to_dict("records")
    for seed in seeds:
        if not existing.empty and seed in set(existing.seed):
            say(f"survival seed={seed}: reusing completed result"); continue
        train, validation, test = split_indices(cohort, seed, "survival")
        center, scale = mean_pool_scaler(source_rows, train)
        center, scale = center.to(device), scale.to(device)
        torch.manual_seed(seed); np.random.seed(seed)
        model = AttentionSurvival().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(CONFIG["head_learning_rate"]),
            weight_decay=float(CONFIG["head_weight_decay"]))
        optimum, state, stale = np.inf, None, 0; started = time.monotonic()
        for epoch in range(int(CONFIG["head_max_epochs"])):
            # Exact full-risk-set Cox gradient without retaining all 15,165 x 512
            # token activations: obtain dL/drisk, then recompute frozen tokens and
            # backpropagate that fixed gradient through the trainable pooling/head.
            model.train(); optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(): detached_risk = risks_for(model, encoder, expression,
                source_rows, train, center, scale, device, batch_size, heartbeat=heartbeat,
                label=f"attention survival seed={seed} epoch={epoch+1} risk pass")
            risk_leaf = detached_risk.detach().requires_grad_(True)
            loss = cox_loss(risk_leaf, duration[train], event[train])
            gradient = torch.autograd.grad(loss, risk_leaf)[0]
            risks_for(model, encoder, expression, source_rows, train, center, scale,
                device, batch_size, with_grad=True, external_gradient=gradient,
                heartbeat=heartbeat, label=f"attention survival seed={seed} epoch={epoch+1} gradient pass")
            optimizer.step(); model.eval()
            with torch.no_grad(): validation_risk = risks_for(model, encoder, expression,
                source_rows, validation, center, scale, device, batch_size)
            value = float(cox_loss(validation_risk, duration[validation], event[validation]))
            if value < optimum - 1e-5: optimum, state, stale = value, best_state(model), 0
            else: stale += 1
            say(f"attention survival seed={seed} epoch={epoch+1} train_loss={float(loss):.5f} "
                f"val_loss={value:.5f} elapsed={(time.monotonic()-started)/60:.1f}m")
            if stale >= int(CONFIG["head_patience"]): break
        assert state is not None; model.load_state_dict(state); model.eval()
        with torch.no_grad(): risk = risks_for(model, encoder, expression, source_rows,
            test, center, scale, device, batch_size).cpu().numpy()
        held = cohort.iloc[test].reset_index(drop=True)
        per_cohort = held.assign(risk=risk).groupby("cohort").apply(
            lambda group: safe_cindex(group, group.risk.to_numpy()), include_groups=False).dropna()
        sizes = held.cohort.value_counts().reindex(per_cohort.index)
        row = {"seed": seed, "representation": "ours_45.6m_attention", "pooling": "learned_attention",
            "probe": "attention_pool_mlp_512_256_cox", "native_features": 512, "epochs": epoch + 1,
            "train_patients": len(train), "validation_patients": len(validation), "test_patients": len(test),
            "events_test": int(held.event.sum()), "c_index": safe_cindex(held, risk),
            "weighted_c_index": float(np.average(per_cohort, weights=sizes)),
            "macro_c_index": float(per_cohort.mean()), "cohorts_scored": len(per_cohort)}
        rows.append(row); pd.DataFrame(rows).to_parquet(output, index=False)
        say(f"attention survival seed={seed} c_index={row['c_index']:.4f} "
            f"weighted={row['weighted_c_index']:.4f} macro={row['macro_c_index']:.4f}")
    del encoder


def summarize() -> None:
    specifications = {
        "classification": ("classification_mlp_per_split.parquet", "classification_attention_per_split.parquet",
            ["macro_f1", "weighted_f1", "balanced_accuracy"]),
        "survival": ("survival_mlp_per_split.parquet", "survival_attention_per_split.parquet",
            ["c_index", "weighted_c_index", "macro_c_index"]),
    }
    for task, (mean_file, attention_file, metrics) in specifications.items():
        mean = pd.read_parquet(RESULTS / mean_file)
        mean = mean.loc[mean.representation.eq("ours_45.6m")].assign(pooling="mean")
        attention_path = RESULTS / attention_file
        attention = pd.read_parquet(attention_path) if attention_path.is_file() else pd.DataFrame()
        combined = pd.concat([mean, attention], ignore_index=True)
        combined.to_parquet(RESULTS / f"ours_pooling_{task}_per_split.parquet", index=False)
        rows = []
        for pooling, frame in combined.groupby("pooling"):
            row = {"pooling": pooling, "splits": len(frame)}
            for metric in metrics:
                row[f"{metric}_mean"] = frame[metric].mean()
                row[f"{metric}_sd"] = frame[metric].std(ddof=1)
            rows.append(row)
        summary = pd.DataFrame(rows)
        summary.to_parquet(RESULTS / f"ours_pooling_{task}_summary.parquet", index=False)
        summary.to_csv(RESULTS / f"ours_pooling_{task}_summary.csv", index=False)
        print(f"\n{task}\n{summary.to_string(index=False)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tasks", nargs="+", choices=["classification", "survival"],
                        default=["classification", "survival"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(CONFIG["split_seeds"]))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--heartbeat-seconds", type=int, default=60)
    args = parser.parse_args()
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    samples = build_cohorts(); prepare_expression(samples); verify_input_contract()
    if not MEAN_EMBEDDINGS.is_file():
        raise FileNotFoundError(f"Mean-pooling baseline embeddings are missing: {MEAN_EMBEDDINGS}")
    if "classification" in args.tasks:
        run_classification(samples, device, args.batch_size, args.heartbeat_seconds, args.seeds)
    if "survival" in args.tasks:
        run_survival(samples, device, args.batch_size, args.heartbeat_seconds, args.seeds)
    summarize()


if __name__ == "__main__":
    main()
