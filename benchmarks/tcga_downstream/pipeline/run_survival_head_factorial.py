#!/usr/bin/env python3
"""Head-controlled 3 representation x 2 Cox-head TCGA OOF comparison."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import CONFIG, RESULTS, WORK
from run_benchmark import SurvivalHead, best_state, cox_loss, representation
from run_survival_5fold_oof import (BOOTSTRAPS, FOLDS, OOF_SEED,
    fold_aware_cindex, load_cohort, make_folds, stability)

ANALYSIS = "survival_head_factorial"
BOOTSTRAP_SEED = 20261102
PERMUTATIONS = 100_000
MAX_EPOCHS = 10_000
REPRESENTATIONS = ("Bridge 45.6M", "PCA-128", "Full raw expression")
HEADS = ("Linear Cox", "Cox MLP [512,256]")
COLORS = {"Bridge 45.6M": "#1f5a94", "PCA-128": "#d17a22",
          "Full raw expression": "#4f8b3a"}
FIGURES = RESULTS / "figures"


class LinearCoxHead(torch.nn.Module):
    def __init__(self, inputs: int):
        super().__init__()
        # A Cox intercept is not identifiable and is therefore omitted.
        self.network = torch.nn.Linear(inputs, 1, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


def say(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def inner_split(cohort: pd.DataFrame, train_all: np.ndarray, fold: int) -> tuple[np.ndarray, np.ndarray]:
    train, validation = train_test_split(
        train_all, test_size=.125, random_state=OOF_SEED + 1000 + fold,
        stratify=cohort.cohort.iloc[train_all])
    return np.asarray(train), np.asarray(validation)


def standardized(matrix: np.ndarray, train: np.ndarray, validation: np.ndarray,
                 test: np.ndarray, source_rows: np.ndarray | None = None) -> tuple[np.ndarray, ...]:
    rows = np.arange(len(matrix)) if source_rows is None else source_rows
    train_values = np.asarray(matrix[rows[train]], dtype=np.float32)
    mean = train_values.mean(axis=0, dtype=np.float64).astype(np.float32)
    sd = train_values.std(axis=0, dtype=np.float64).astype(np.float32)
    sd[sd < 1e-6] = 1.0
    return tuple((np.asarray(matrix[rows[index]], dtype=np.float32) - mean) / sd
                 for index in (train, validation, test))


def train_head(xtrain: np.ndarray, xvalidation: np.ndarray, xtest: np.ndarray,
               train_time: np.ndarray, train_event: np.ndarray,
               validation_time: np.ndarray, validation_event: np.ndarray,
               head: str, fold: int, device: torch.device,
               max_epochs: int | None = None) -> tuple[np.ndarray, int]:
    torch.manual_seed(OOF_SEED + fold)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(OOF_SEED + fold)
    model = (LinearCoxHead(xtrain.shape[1]) if head == HEADS[0]
             else SurvivalHead(xtrain.shape[1])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(CONFIG["head_learning_rate"]),
        weight_decay=float(CONFIG["head_weight_decay"]))
    tensors = [torch.as_tensor(value, dtype=torch.float32, device=device) for value in
               (xtrain, train_time, train_event, xvalidation,
                validation_time, validation_event)]
    train_x, train_t, train_e, val_x, val_t, val_e = tensors
    optimum, state, stale = np.inf, None, 0
    # The original MLPs all stopped before 200 epochs, but a linear-head QA run
    # reached that ceiling in every Bridge/PCA fold. Use a nonbinding common
    # ceiling and retain the same validation-patience stopping rule for both heads.
    epochs = int(MAX_EPOCHS if max_epochs is None else max_epochs)
    for epoch in range(epochs):
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss = cox_loss(model(train_x), train_t, train_e)
        loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            value = float(cox_loss(model(val_x), val_t, val_e))
        if value < optimum - 1e-5:
            optimum, state, stale = value, best_state(model), 0
        else:
            stale += 1
        if stale >= int(CONFIG["head_patience"]):
            break
    assert state is not None and np.isfinite(optimum)
    model.load_state_dict(state); model.eval()
    with torch.no_grad():
        risk = model(torch.as_tensor(xtest, dtype=torch.float32, device=device)).cpu().numpy()
    assert np.isfinite(risk).all()
    del model, optimizer, tensors, train_x, val_x
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return risk, epoch + 1


def fold_pca(cohort: pd.DataFrame, train_all: np.ndarray, test: np.ndarray,
             fold: int) -> np.ndarray:
    cache = WORK / f"{ANALYSIS}_pca128_fold{fold}.npy"
    if cache.is_file():
        values = np.load(cache, mmap_mode="r")
        assert values.shape == (len(cohort), 128) and np.isfinite(np.asarray(values)).all()
        return values
    matrix = representation("raw_expression")
    rows = cohort.matrix_row.to_numpy(int)
    components = min(128, len(train_all) - 1, matrix.shape[1])
    pipe = make_pipeline(StandardScaler(), PCA(
        n_components=components, svd_solver="randomized", random_state=OOF_SEED + fold))
    transformed = np.empty((len(cohort), components), dtype=np.float32)
    transformed[train_all] = pipe.fit_transform(np.asarray(matrix[rows[train_all]])).astype(np.float32)
    transformed[test] = pipe.transform(np.asarray(matrix[rows[test]])).astype(np.float32)
    assert np.isfinite(transformed).all()
    np.save(cache, transformed)
    return np.load(cache, mmap_mode="r")


def smoke_test(device: torch.device) -> None:
    cohort = load_cohort(); folds = make_folds(cohort)
    fold_lookup = folds.set_index("patient_id").fold
    cohort = cohort.copy(); cohort["fold"] = cohort.patient_id.map(fold_lookup).astype(int)
    test_all = np.flatnonzero(cohort.fold.to_numpy() == 0)
    train_all = np.flatnonzero(cohort.fold.to_numpy() != 0)
    train, validation = inner_split(cohort, train_all, 0)
    train, validation, test = train[:512], validation[:128], test_all[:128]
    source_rows = cohort.matrix_row.to_numpy(int)
    matrices = {
        "Bridge 45.6M": (representation("ours_45.6m"), source_rows),
        "Full raw expression": (np.load(WORK / "tcga_full_log1p_cpm.npy", mmap_mode="r"), source_rows),
    }
    durations = cohort.time_days.to_numpy(np.float32); events = cohort.event.to_numpy(np.float32)
    for label, (matrix, rows) in matrices.items():
        arrays = standardized(matrix, train, validation, test, rows)
        for head in HEADS:
            risk, epochs = train_head(*arrays, durations[train], events[train],
                                      durations[validation], events[validation],
                                      head, 0, device, max_epochs=2)
            say(f"smoke {label}/{head}: n={len(risk)} epochs={epochs}")
    say("smoke test passed")


def reused_mlp_predictions() -> pd.DataFrame:
    source = pd.read_csv(RESULTS / "survival_5fold_oof_predictions.csv")
    source = source[source.representation.isin([REPRESENTATIONS[0], REPRESENTATIONS[2]])].copy()
    source["head"] = HEADS[1]
    return source[["patient_id", "cancer", "survival_time", "event", "fold",
                   "representation", "head", "risk_score"]]


def generate_predictions(cohort: pd.DataFrame, folds: pd.DataFrame,
                         device: torch.device) -> pd.DataFrame:
    output = RESULTS / f"{ANALYSIS}_predictions.csv"
    fold_lookup = folds.set_index("patient_id").fold
    cohort = cohort.copy(); cohort["fold"] = cohort.patient_id.map(fold_lookup).astype(int)
    durations = cohort.time_days.to_numpy(np.float32); events = cohort.event.to_numpy(np.float32)
    source_rows = cohort.matrix_row.to_numpy(int)
    bridge = representation("ours_45.6m")
    raw = np.load(WORK / "tcga_full_log1p_cpm.npy", mmap_mode="r")
    pieces = [reused_mlp_predictions()]
    for fold in range(FOLDS):
        test = np.flatnonzero(cohort.fold.to_numpy() == fold)
        train_all = np.flatnonzero(cohort.fold.to_numpy() != fold)
        train, validation = inner_split(cohort, train_all, fold)
        pca = fold_pca(cohort, train_all, test, fold)
        inputs = {
            REPRESENTATIONS[0]: (bridge, source_rows),
            REPRESENTATIONS[1]: (pca, None),
            REPRESENTATIONS[2]: (raw, source_rows),
        }
        for representation_name, (matrix, rows) in inputs.items():
            for head in HEADS:
                if head == HEADS[1] and representation_name != REPRESENTATIONS[1]:
                    continue  # exact existing OOF predictions were loaded above
                slug = representation_name.lower().replace(" ", "_").replace(".", "_")
                head_slug = "linear" if head == HEADS[0] else "mlp"
                part = WORK / f"{ANALYSIS}_{slug}_{head_slug}_fold{fold}.csv"
                if part.is_file():
                    frame = pd.read_csv(part); say(f"reusing {part.name}")
                else:
                    started = time.monotonic()
                    arrays = standardized(matrix, train, validation, test, rows)
                    risk, epochs = train_head(
                        *arrays, durations[train], events[train],
                        durations[validation], events[validation], head, fold, device)
                    held = cohort.iloc[test]
                    frame = pd.DataFrame({
                        "patient_id": held.patient_id, "cancer": held.cohort,
                        "survival_time": held.time_days, "event": held.event.astype(int),
                        "fold": fold, "representation": representation_name,
                        "head": head, "risk_score": risk, "epochs": epochs})
                    frame.to_csv(part, index=False)
                    say(f"completed {fold=}/{representation_name}/{head} n={len(test)} "
                        f"epochs={epochs} elapsed={(time.monotonic()-started)/60:.1f}m")
                pieces.append(frame)
    result = pd.concat(pieces, ignore_index=True)
    trained = result.dropna(subset=["epochs"])
    assert len(trained) and trained.epochs.max() < MAX_EPOCHS, "training still reached epoch ceiling"
    (trained.groupby(["fold", "representation", "head"], as_index=False)
     .epochs.first().to_csv(RESULTS/f"{ANALYSIS}_training_qc.csv", index=False))
    result.to_csv(output, index=False)
    return result


def qa(predictions: pd.DataFrame, cohort: pd.DataFrame, folds: pd.DataFrame) -> None:
    expected = set(cohort.patient_id)
    combinations = pd.MultiIndex.from_product(
        [REPRESENTATIONS, HEADS], names=["representation", "head"])
    assert set(map(tuple, predictions[["representation", "head"]].drop_duplicates().to_numpy())) == set(combinations)
    reference = None
    keys = ["patient_id", "cancer", "survival_time", "event", "fold"]
    for representation_name, head in combinations:
        frame = predictions[(predictions.representation == representation_name) & (predictions["head"] == head)]
        assert len(frame) == len(cohort) and frame.patient_id.is_unique
        assert set(frame.patient_id) == expected and np.isfinite(frame.risk_score).all()
        ordered = frame[keys].sort_values("patient_id").reset_index(drop=True)
        if reference is None: reference = ordered
        else: pd.testing.assert_frame_equal(reference, ordered, check_dtype=False)
    assert folds.patient_id.is_unique and set(folds.patient_id) == expected


def metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (representation_name, head), frame in predictions.groupby(["representation", "head"]):
        value, pairs = fold_aware_cindex(frame)
        rows.append({"representation": representation_name, "head": head,
                     "oof_c_index": value, "comparable_pairs": pairs,
                     "aggregation": "fold-aware pair-weighted"})
    result = pd.DataFrame(rows)
    result.to_csv(RESULTS / f"{ANALYSIS}_metrics.csv", index=False)
    return result


def per_cancer(predictions: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED); rows = []
    for number, cancer in enumerate(sorted(predictions.cancer.unique()), 1):
        data = predictions[predictions.cancer == cancer]
        support = data[(data.representation == REPRESENTATIONS[0]) & (data["head"] == HEADS[0])].sort_values("patient_id")
        patients = support.patient_id.to_numpy(str)
        draws = rng.integers(0, len(patients), size=(BOOTSTRAPS, len(patients)))
        counts = np.apply_along_axis(np.bincount, 1, draws, minlength=len(patients))
        for representation_name in REPRESENTATIONS:
            for head in HEADS:
                frame = data[(data.representation == representation_name) & (data["head"] == head)]
                frame = frame.set_index("patient_id").loc[patients].reset_index()
                estimate, pairs = fold_aware_cindex(frame); boot = []
                for multiplicity in counts:
                    value, _ = fold_aware_cindex(frame.loc[frame.index.repeat(multiplicity)])
                    if np.isfinite(value): boot.append(value)
                low, high = np.quantile(boot, [.025, .975]) if boot else (np.nan, np.nan)
                width = float(high-low) if np.isfinite(low) else np.nan
                events = int(support.event.sum())
                rows.append({"representation": representation_name, "head": head,
                    "cancer": cancer, "n_patients": len(support), "n_events": events,
                    "n_censored": len(support)-events,
                    "censoring_fraction": 1-events/len(support),
                    "comparable_pairs": pairs, "c_index": estimate,
                    "ci_low": low, "ci_high": high, "ci_width": width,
                    "bootstrap_valid": len(boot),
                    "stability_flag": stability(events, len(boot), width)})
        say(f"bootstrap {number}/32 cancer={cancer}")
    result = pd.DataFrame(rows)
    result.to_csv(RESULTS / f"{ANALYSIS}_per_cancer.csv", index=False)
    return result


def signflip(values: np.ndarray, rng: np.random.Generator) -> float:
    observed = abs(values.mean()); extreme = 0
    for start in range(0, PERMUTATIONS, 10_000):
        size = min(10_000, PERMUTATIONS-start)
        signs = rng.choice([-1., 1.], size=(size, len(values)))
        extreme += int((np.abs((signs*values).mean(axis=1)) >= observed).sum())
    return (extreme+1)/(PERMUTATIONS+1)


def bh_adjust(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().sort_values()
    if valid.empty: return result
    adjusted = valid.to_numpy()*len(valid)/np.arange(1, len(valid)+1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(max=1.)
    result.loc[valid.index] = adjusted
    return result


def summaries(per: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED+1); rows = []
    for representation_name in REPRESENTATIONS:
        for head in HEADS:
            frame = per[(per.representation == representation_name) & (per["head"] == head)]
            rows.append({"comparison_type": "cell_summary", "representation": representation_name,
                "head": head, "contrast": "", "median_c_index": frame.c_index.median(),
                "patient_weighted_c_index": np.average(frame.c_index, weights=frame.n_patients),
                "median_difference": np.nan, "mean_difference": np.nan,
                "cancers_first_gt_second": np.nan, "signflip_p": np.nan, "wilcoxon_p": np.nan})
    contrasts = []
    for head in HEADS:
        wide = per[per["head"] == head].pivot(index="cancer", columns="representation", values="c_index")
        contrasts.extend(("representation_within_head", head, f"{a} - {b}",
                          (wide[a]-wide[b]).dropna().to_numpy())
                         for a, b in [(REPRESENTATIONS[0], REPRESENTATIONS[1]),
                                      (REPRESENTATIONS[0], REPRESENTATIONS[2])])
    for representation_name in REPRESENTATIONS:
        wide = per[per.representation == representation_name].pivot(index="cancer", columns="head", values="c_index")
        contrasts.append(("head_within_representation", representation_name,
                          f"{HEADS[1]} - {HEADS[0]}", (wide[HEADS[1]]-wide[HEADS[0]]).dropna().to_numpy()))
    for kind, group, contrast, values in contrasts:
        rows.append({"comparison_type": kind,
            "representation": group if kind == "head_within_representation" else "",
            "head": group if kind == "representation_within_head" else "",
            "contrast": contrast, "median_c_index": np.nan,
            "patient_weighted_c_index": np.nan, "median_difference": np.median(values),
            "mean_difference": values.mean(), "cancers_first_gt_second": int((values > 0).sum()),
            "signflip_p": signflip(values, rng),
            "wilcoxon_p": wilcoxon(values, alternative="two-sided").pvalue})
    result = pd.DataFrame(rows)
    result["signflip_bh_p"] = bh_adjust(result.signflip_p)
    result["wilcoxon_bh_p"] = bh_adjust(result.wilcoxon_p)
    result.to_csv(RESULTS / f"{ANALYSIS}_statistical_summary.csv", index=False)
    return result


def plots(metric: pd.DataFrame, per: pd.DataFrame) -> None:
    FIGURES.mkdir(exist_ok=True)
    x = np.arange(len(REPRESENTATIONS)); offsets = [-.12, .12]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for head, offset, marker in zip(HEADS, offsets, ("o", "s")):
        values = metric.set_index(["representation", "head"]).loc[
            [(name, head) for name in REPRESENTATIONS], "oof_c_index"].to_numpy()
        ax.scatter(x+offset, values, marker=marker, s=70, label=head)
        for xx, value in zip(x+offset, values):
            ax.text(xx, value+.0013, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, REPRESENTATIONS); ax.set_ylabel("Fold-aware OOF C-index")
    ax.set_title("TCGA survival: representation and Cox-head sensitivity")
    ax.axhline(.5, color="#555", ls="--", lw=1); ax.grid(axis="y", color="#ddd")
    ax.legend(frameon=False); fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES/f"{ANALYSIS}_overall.{ext}", dpi=300 if ext=="png" else None)
    plt.close(fig)

    order = sorted(per.cancer.unique()); y = np.arange(len(order))
    valid_bounds = pd.concat([per.ci_low, per.ci_high]).dropna()
    xlim = (max(0., valid_bounds.min()-.04), min(1., valid_bounds.max()+.04))
    fig, axes = plt.subplots(1, 2, figsize=(18, max(9, .35*len(order)+2)), sharey=True)
    for panel, (ax, head) in enumerate(zip(axes, HEADS)):
        for representation_name, offset in zip(REPRESENTATIONS, (-.2, 0, .2)):
            frame = per[(per["head"] == head) & (per.representation == representation_name)].set_index("cancer").reindex(order)
            ok = frame.c_index.notna() & frame.ci_low.notna(); yy = y+offset
            ax.errorbar(frame.loc[ok,"c_index"], yy[ok],
                xerr=np.vstack([frame.loc[ok,"c_index"]-frame.loc[ok,"ci_low"],
                                frame.loc[ok,"ci_high"]-frame.loc[ok,"c_index"]]),
                fmt="o", color=COLORS[representation_name], label=representation_name,
                capsize=2, ms=3.8, lw=1)
        ax.axvline(.5, color="#555", ls="--", lw=1); ax.set_xlim(xlim)
        ax.set(yticks=y, yticklabels=order, xlabel="Held-out C-index", title=f"{chr(65+panel)}. {head}")
        ax.grid(axis="x", color="#ddd", lw=.7)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("TCGA cancer type"); axes[1].legend(frameon=False, loc="upper center", bbox_to_anchor=(.5,-.04))
    fig.suptitle("Head-controlled within-cancer survival discrimination", fontsize=15)
    fig.tight_layout(rect=(0,.03,1,.98))
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES/f"{ANALYSIS}_per_cancer_two_panel.{ext}",
                    dpi=300 if ext=="png" else None, bbox_inches="tight")
    plt.close(fig)

    delta = per.pivot(index=["cancer","representation"], columns="head", values="c_index").reset_index()
    delta["delta"] = delta[HEADS[1]]-delta[HEADS[0]]
    matrix = delta.pivot(index="cancer", columns="representation", values="delta").reindex(order)[list(REPRESENTATIONS)]
    bound = max(.05, float(np.nanmax(np.abs(matrix.to_numpy()))))
    fig, ax = plt.subplots(figsize=(8, 10))
    image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-bound, vmax=bound)
    ax.set_xticks(range(3), REPRESENTATIONS); ax.set_yticks(range(len(order)), order)
    ax.set_title("Cox-head effect by cancer (MLP − linear C-index)")
    fig.colorbar(image, ax=ax, label="Δ held-out C-index"); fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES/f"{ANALYSIS}_head_effect_heatmap.{ext}",
                    dpi=300 if ext=="png" else None, bbox_inches="tight")
    plt.close(fig)


def provenance() -> None:
    record = {
        "analysis": "Head-controlled 3 representation x 2 Cox-head TCGA OOF comparison",
        "representations": list(REPRESENTATIONS), "heads": list(HEADS),
        "cohort": "same 9,668-patient, 32-cancer complete OOF survival cohort",
        "folds": "same saved five mutually exclusive OOF folds",
        "objective": "separate representation effects from downstream Cox-head effects",
        "linear_head": "single bias-free linear layer trained with Cox partial likelihood",
        "mlp_head": "existing [512,256] Cox MLP",
        "shared_training": {"optimizer": "AdamW", "learning_rate": CONFIG["head_learning_rate"],
            "weight_decay": CONFIG["head_weight_decay"], "max_epochs": MAX_EPOCHS,
            "patience": CONFIG["head_patience"], "inner_validation_fraction": .125,
            "seed": OOF_SEED},
        "preprocessing": "feature standardization learned on inner-training patients only",
        "pca": "StandardScaler + randomized PCA-128 fitted on each outer training fold only",
        "reused_predictions": "exact existing Bridge and raw-expression MLP OOF predictions",
        "bootstrap_replicates": BOOTSTRAPS, "bootstrap_seed": BOOTSTRAP_SEED,
        "encoder_inference": "none; cached frozen Bridge embeddings reused",
        "optimization_qa": "initial 200-epoch ceiling was nonbinding for every reused MLP but was reached by all Bridge/PCA linear fits; the common ceiling was raised until nonbinding (10000) before accepting results, with all other settings unchanged",
        "prespecification": "architecture and hyperparameters fixed before factorial results were inspected; the only post-run correction was the outcome-independent nonconvergence rule described in optimization_qa",
        "limitation": "identical optimizer hyperparameters do not guarantee equal effective regularization across input dimensions",
    }
    (RESULTS/f"{ANALYSIS}_provenance.json").write_text(json.dumps(record, indent=2)+"\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args(); device = torch.device(args.device)
    if args.smoke_test:
        smoke_test(device); return
    say("startup estimate: 15-35 minutes from cached embeddings; no encoder inference")
    cohort = load_cohort(); folds = make_folds(cohort)
    predictions = generate_predictions(cohort, folds, device); qa(predictions, cohort, folds)
    overall = metrics(predictions); cancer = per_cancer(predictions)
    summaries(cancer); plots(overall, cancer); provenance()
    say("head-controlled factorial survival analysis complete")


if __name__ == "__main__":
    main()
