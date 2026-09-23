#!/usr/bin/env python3
"""Recreate native Cox risks and summarize C-index separately by TCGA cancer."""

from __future__ import annotations

import argparse
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from common import CONFIG, RESULTS
from run_benchmark import (
    MODELS,
    SurvivalHead,
    best_state,
    build_cohorts,
    cox_loss,
    head_input,
    safe_cindex,
    standardize_native,
)

NAMES = (*MODELS, "full_expression_25150")


def say(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def run(device: torch.device) -> None:
    output = RESULTS / "cox_mlp_per_cancer_per_split.csv"
    prediction_output = RESULTS / "cox_mlp_risk_predictions.csv"
    if output.is_file() and prediction_output.is_file():
        say("reusing Cox per-cancer results")
        return
    samples = build_cohorts()
    cohort = samples.loc[samples.survival_usable].copy().reset_index(drop=True)
    source_rows = cohort.matrix_row.to_numpy(int)
    strata = cohort.cohort.astype(str)
    durations = cohort.time_days.to_numpy(np.float32)
    events = cohort.event.to_numpy(np.float32)
    rows, predictions = [], []
    for seed in CONFIG["split_seeds"]:
        train_all, test = train_test_split(
            np.arange(len(cohort)), test_size=CONFIG["test_fraction"],
            random_state=seed, stratify=strata)
        train, validation = train_test_split(
            train_all, test_size=.125, random_state=seed + 1000,
            stratify=strata.iloc[train_all])
        for name in NAMES:
            torch.manual_seed(seed)
            np.random.seed(seed)
            matrix = head_input(name)
            xtrain, xvalidation, xtest = standardize_native(
                matrix, source_rows[train], source_rows[validation], source_rows[test])
            model = SurvivalHead(xtrain.shape[1]).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=float(CONFIG["head_learning_rate"]),
                weight_decay=float(CONFIG["head_weight_decay"]))
            tensors = [torch.as_tensor(value, dtype=torch.float32, device=device) for value in
                       (xtrain, durations[train], events[train], xvalidation,
                        durations[validation], events[validation])]
            train_x, train_t, train_e, val_x, val_t, val_e = tensors
            optimum, state, stale = np.inf, None, 0
            for epoch in range(int(CONFIG["head_max_epochs"])):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = cox_loss(model(train_x), train_t, train_e)
                loss.backward()
                optimizer.step()
                model.eval()
                with torch.no_grad():
                    value = float(cox_loss(model(val_x), val_t, val_e))
                if value < optimum - 1e-5:
                    optimum, state, stale = value, best_state(model), 0
                else:
                    stale += 1
                if stale >= int(CONFIG["head_patience"]):
                    break
            assert state is not None
            model.load_state_dict(state)
            model.eval()
            with torch.no_grad():
                risk = model(torch.as_tensor(xtest, dtype=torch.float32, device=device)).cpu().numpy()
            held = cohort.iloc[test].copy().reset_index(drop=True)
            held["risk"] = risk
            for cancer, frame in held.groupby("cohort"):
                rows.append({
                    "seed": seed,
                    "cancer_label": cancer,
                    "representation": name,
                    "test_patients": len(frame),
                    "events": int(frame.event.sum()),
                    "c_index": safe_cindex(frame, frame.risk.to_numpy()),
                })
            predictions.extend({
                "seed": seed,
                "patient_id": row.patient_id,
                "cancer_label": row.cohort,
                "representation": name,
                "time_days": row.time_days,
                "event": int(row.event),
                "risk": float(row.risk),
            } for row in held.itertuples())
            say(f"seed={seed} representation={name} epochs={epoch + 1} "
                f"scored_cancers={pd.Series([r['cancer_label'] for r in rows if r['seed'] == seed and r['representation'] == name]).nunique()}")
            del model, optimizer, tensors, xtrain, xvalidation, xtest
            if device.type == "cuda":
                torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(output, index=False)
    pd.DataFrame(predictions).to_csv(prediction_output, index=False)


def summarize_and_plot() -> None:
    data = pd.read_csv(RESULTS / "cox_mlp_per_cancer_per_split.csv")
    summary = data.groupby(["cancer_label", "representation"], as_index=False).agg(
        splits=("c_index", "count"),
        test_patients_mean=("test_patients", "mean"),
        events_mean=("events", "mean"),
        c_index_mean=("c_index", "mean"),
        c_index_sd=("c_index", "std"),
    )
    summary.to_csv(RESULTS / "cox_mlp_per_cancer_summary.csv", index=False)
    labels = {"ours_45.6m": "Bridge 45.6M", "bulkformer_50m": "BulkFormer-50M",
              "bulkformer_147m": "BulkFormer-147M",
              "full_expression_25150": "Full raw expression"}
    order = list(NAMES)
    matrix = summary.pivot(index="cancer_label", columns="representation", values="c_index_mean")
    matrix = matrix.reindex(columns=order)
    matrix = matrix.loc[matrix.mean(axis=1, skipna=True).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(8.4, 10.8))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="RdYlBu", vmin=0.3, vmax=0.9)
    ax.set_xticks(np.arange(len(order)), [labels[value] for value in order], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(matrix)), matrix.index)
    ax.set_title("Per-cancer Cox C-index on held-out patients (mean across 5 splits)")
    for row in range(len(matrix)):
        for column in range(len(order)):
            value = matrix.iloc[row, column]
            ax.text(column, row, "NA" if pd.isna(value) else f"{value:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="black" if pd.isna(value) or 0.43 < value < 0.78 else "white")
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("C-index")
    ax.set_xlabel("Representation")
    ax.set_ylabel("TCGA cancer type")
    fig.tight_layout()
    figures = RESULTS / "figures"
    fig.savefig(figures / "cox_mlp_per_cancer_c_index.png", dpi=220)
    fig.savefig(figures / "cox_mlp_per_cancer_c_index.pdf")
    plt.close(fig)

    bridge = summary.loc[summary.representation.eq("ours_45.6m")].copy()
    bridge = bridge.sort_values("c_index_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(8.6, 10.5))
    y = np.arange(len(bridge))
    ax.barh(y, bridge.c_index_mean, xerr=bridge.c_index_sd.fillna(0),
            color="#4C78A8", alpha=0.9, capsize=3)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.3,
               label="Chance concordance (0.5)")
    ax.set_yticks(y, bridge.cancer_label)
    ax.set_xlim(0.2, 1.0)
    ax.set_xlabel("Held-out C-index (mean ± SD across 5 splits)")
    ax.set_ylabel("TCGA cancer type")
    ax.set_title("Bridge 45.6M: per-cancer Cox time-to-event discrimination")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(figures / "bridge_cox_per_cancer_c_index_bar.png", dpi=220)
    fig.savefig(figures / "bridge_cox_per_cancer_c_index_bar.pdf")
    plt.close(fig)

    binary = pd.read_csv(RESULTS / "bulkformer_parity_alive_dead_prognosis_per_cancer.csv")
    binary_matrix = binary.pivot(
        index="cancer_label", columns="representation", values="auroc").reindex(columns=[
            "ours_45.6m", "bulkformer_50m", "bulkformer_147m", "full_raw_expression"])
    binary_matrix = binary_matrix.rename(columns={"full_raw_expression": "full_expression_25150"})
    cancers = sorted(set(binary_matrix.index) | set(matrix.index))
    binary_matrix = binary_matrix.reindex(index=cancers, columns=order)
    cox_matrix = matrix.reindex(index=cancers, columns=order)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 11.0), sharey=True)
    panels = [(axes[0], binary_matrix, "Alive/dead classification", "AUROC"),
              (axes[1], cox_matrix, "Cox time-to-event survival", "C-index")]
    for ax, values, title, metric in panels:
        heatmap = ax.imshow(values.to_numpy(), aspect="auto", cmap="RdYlBu", vmin=0.3, vmax=0.9)
        ax.set_xticks(np.arange(len(order)), [labels[value] for value in order], rotation=25, ha="right")
        ax.set_yticks(np.arange(len(cancers)), cancers)
        ax.set_title(title)
        ax.set_xlabel("Representation")
        for row in range(len(values)):
            for column in range(len(order)):
                value = values.iloc[row, column]
                ax.text(column, row, "NA" if pd.isna(value) else f"{value:.2f}",
                        ha="center", va="center", fontsize=6.8,
                        color="black" if pd.isna(value) or 0.43 < value < 0.78 else "white")
        colorbar = fig.colorbar(heatmap, ax=ax, pad=0.02, fraction=0.046)
        colorbar.set_label(metric)
    axes[0].set_ylabel("TCGA cancer type")
    fig.suptitle("Per-cancer prognosis: binary status versus censoring-aware survival", y=0.995)
    fig.tight_layout()
    fig.savefig(figures / "per_cancer_alive_dead_vs_cox_comparison.png", dpi=220)
    fig.savefig(figures / "per_cancer_alive_dead_vs_cox_comparison.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    run(device)
    summarize_and_plot()
    say("Cox per-cancer analysis complete")


if __name__ == "__main__":
    main()
