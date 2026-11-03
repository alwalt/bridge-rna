#!/usr/bin/env python3
"""Build and validate the cached TCGA complete-OOF survival notebook."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
from lifelines.utils.concordance import _concordance_summary_statistics


ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "benchmarks" / "tcga_downstream"
RESULTS = BENCH / "results"
PACKAGE = ROOT / "notebooks" / "tcga_downstream"
CACHE = PACKAGE / "cached_evidence"
FIGURES = PACKAGE / "figures"

EVIDENCE = [
    "survival_head_factorial_metrics.csv",
    "survival_head_factorial_per_cancer.csv",
    "survival_head_factorial_statistical_summary.csv",
    "survival_head_factorial_training_qc.csv",
    "survival_head_factorial_provenance.json",
    "survival_5fold_oof_metrics.csv",
    "survival_5fold_oof_per_cancer.csv",
    "survival_5fold_oof_representation_comparison.csv",
    "bridge_layerwise_survival_summary.csv",
    "final_qa.json",
]

COLORS = {
    "Bridge 45.6M": "#1f5a94",
    "PCA-128": "#d17a22",
    "Full raw expression": "#4f8b3a",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fold_aware_cindex(frame: pd.DataFrame) -> tuple[float, int]:
    correct = tied = 0.0
    pairs = 0
    for _, group in frame.groupby("fold"):
        if len(group) < 2 or group.event.sum() == 0:
            continue
        c, t, p = _concordance_summary_statistics(
            group.survival_time.to_numpy(float),
            -group.risk_score.to_numpy(float),
            group.event.to_numpy(bool),
        )
        correct += float(c)
        tied += float(t)
        pairs += int(p)
    return ((correct + 0.5 * tied) / pairs if pairs else np.nan), pairs


def validate_predictions() -> dict:
    predictions = pd.read_csv(RESULTS / "survival_head_factorial_predictions.csv")
    keys = ["representation", "head"]
    sizes = predictions.groupby(keys).size()
    assert len(sizes) == 6 and sizes.eq(9668).all()
    assert not predictions.duplicated(["patient_id", *keys]).any()
    reference = None
    for _, group in predictions.groupby(keys):
        current = group[["patient_id", "cancer", "survival_time", "event", "fold"]].sort_values("patient_id").reset_index(drop=True)
        if reference is None:
            reference = current
        else:
            pd.testing.assert_frame_equal(reference, current, check_dtype=False)

    saved = pd.read_csv(RESULTS / "survival_head_factorial_metrics.csv")
    recalculated = []
    for (representation, head), group in predictions.groupby(keys):
        value, pairs = fold_aware_cindex(group)
        expected = saved[(saved.representation == representation) & (saved["head"] == head)].iloc[0]
        assert np.isclose(value, expected.oof_c_index, atol=1e-12)
        assert pairs == int(expected.comparable_pairs)
        recalculated.append({
            "representation": representation,
            "head": head,
            "recalculated_c_index": value,
            "comparable_pairs": pairs,
            "absolute_difference_from_saved": abs(value - expected.oof_c_index),
        })
    return {
        "status": "passed",
        "prediction_rows_checked": int(len(predictions)),
        "patients_per_representation_head": 9668,
        "identical_patients_folds_outcomes": True,
        "unique_prediction_per_patient_representation_head": True,
        "recalculated_metrics": recalculated,
    }


def copy_evidence() -> dict[str, str]:
    CACHE.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in EVIDENCE:
        source = RESULTS / name
        assert source.is_file(), source
        target = CACHE / name
        shutil.copy2(source, target)
        hashes[name] = sha256(target)
    return hashes


def plot_overall() -> None:
    data = pd.read_csv(CACHE / "survival_head_factorial_metrics.csv")
    heads = ["Linear Cox", "Cox MLP [512,256]"]
    reps = ["Bridge 45.6M", "PCA-128", "Full raw expression"]
    x = np.arange(len(reps)); width = 0.34
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for index, head in enumerate(heads):
        values = [float(data[(data.representation == rep) & (data["head"] == head)].oof_c_index.iloc[0]) for rep in reps]
        bars = ax.bar(x + (index - 0.5) * width, values, width, label=head, alpha=0.92)
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=3, fontsize=9)
    ax.set_xticks(x, reps)
    ax.set_ylabel("Fold-aware OOF C-index")
    ax.set_ylim(0.72, 0.78)
    ax.set_title("Complete five-fold OOF survival discrimination")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"oof_survival_overall.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_forest() -> None:
    data = pd.read_csv(CACHE / "survival_5fold_oof_per_cancer.csv")
    order = sorted(data.cancer.unique())
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(9.8, 12.0))
    offsets = {"Bridge 45.6M": -0.22, "PCA-128": 0.0, "Full raw expression": 0.22}
    for rep, offset in offsets.items():
        frame = data[data.representation == rep].set_index("cancer").reindex(order)
        valid = frame.c_index.notna() & frame.ci_low.notna() & frame.ci_high.notna()
        yy = y + offset
        ax.errorbar(
            frame.loc[valid, "c_index"], yy[valid],
            xerr=np.vstack([
                frame.loc[valid, "c_index"] - frame.loc[valid, "ci_low"],
                frame.loc[valid, "ci_high"] - frame.loc[valid, "c_index"],
            ]),
            fmt="o", ms=4.5, capsize=2, lw=1.2,
            color=COLORS[rep], label=rep,
        )
    support = data[data.representation == "Bridge 45.6M"].set_index("cancer").reindex(order)
    ax.set_yticks(y, order)
    ax.invert_yaxis()
    ax.axvline(0.5, color="black", ls="--", lw=1)
    ax.set_xlabel("Held-out within-cancer C-index")
    ax.set_title("Within-cancer survival discrimination by representation")
    ax.set_xlim(0.12, 1.02)
    for idx, row in enumerate(support.itertuples()):
        ax.text(1.025, idx, f"{int(row.n_events)} / {int(row.n_patients)}", va="center", fontsize=8, clip_on=False)
    ax.text(1.025, -0.8, "Events / N", fontsize=8, fontweight="bold", clip_on=False)
    ax.legend(frameon=False, loc="lower left")
    ax.grid(axis="x", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"oof_survival_forest.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def notebook() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata.language_info = {"name": "python", "version": f"{sys.version_info.major}.{sys.version_info.minor}"}
    cells = []
    cells.append(nbf.v4.new_markdown_cell(
        "# Complete five-fold OOF TCGA survival evaluation\n\n"
        "This analysis tests whether a frozen Bridge representation retains censoring-aware prognostic information relative to PCA-128 and full raw expression."
    ))
    cells.append(nbf.v4.new_markdown_cell(
        "## Data and endpoint\n\nThe cohort contains **9,668 patients**, **32 cancer types**, and **2,774 overall-survival events**. "
        "Every patient receives exactly one prediction from a Cox head that was not trained on that patient."
    ))
    cells.append(nbf.v4.new_code_cell(
        "from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display, Image\n"
        "HERE = Path.cwd()\nif not (HERE / 'cached_evidence').exists():\n    HERE = Path('notebooks/tcga_downstream')\n"
        "CACHE = HERE / 'cached_evidence'\nFIGURES = HERE / 'figures'\n"
        "qa = json.loads((CACHE / 'package_validation.json').read_text())\nassert qa['status'] == 'passed'\nqa['status'], qa['prediction_rows_checked']"
    ))
    cells.append(nbf.v4.new_markdown_cell(
        "## Methods\n\nBridge remains frozen. Scaling and PCA are fitted on training folds only. "
        "All representations use identical folds and outcomes and are tested with both linear Cox and a `[512,256]` Cox MLP. "
        "Concordance is aggregated within held-out folds. Cancer-specific intervals use 1,000 patient-level bootstrap draws retaining fold membership."
    ))
    cells.append(nbf.v4.new_code_cell(
        "metrics = pd.read_csv(CACHE / 'survival_head_factorial_metrics.csv')\n"
        "table = metrics.pivot(index='representation', columns='head', values='oof_c_index').loc[['Bridge 45.6M','PCA-128','Full raw expression']]\n"
        "display(table.round(4))"
    ))
    cells.append(nbf.v4.new_code_cell("display(Image(filename=str(FIGURES / 'oof_survival_overall.png')))"))
    cells.append(nbf.v4.new_markdown_cell(
        "Bridge achieves C-index **0.7589** with linear Cox and **0.7580** with the Cox MLP. "
        "PCA-128 achieves 0.7615/0.7638 and full expression 0.7533/0.7590. Bridge is nearly head-insensitive, "
        "and no representation or head contrast remains significant after multiplicity correction."
    ))
    cells.append(nbf.v4.new_markdown_cell("## Within-cancer survival discrimination"))
    cells.append(nbf.v4.new_code_cell("display(Image(filename=str(FIGURES / 'oof_survival_forest.png')))"))
    cells.append(nbf.v4.new_code_cell(
        "per = pd.read_csv(CACHE / 'survival_5fold_oof_per_cancer.csv')\n"
        "bridge = per[per.representation == 'Bridge 45.6M'].copy()\n"
        "supported = bridge[(bridge.ci_low > 0.5) & (bridge.n_events >= 20)][['cancer','n_patients','n_events','c_index','ci_low','ci_high']]\n"
        "display(supported.sort_values('c_index', ascending=False).round({'c_index':3,'ci_low':3,'ci_high':3}))"
    ))
    cells.append(nbf.v4.new_markdown_cell(
        "Bridge has median within-cancer C-index **0.6239** and exceeds each conventional comparator in 15/32 cancers. "
        "Sparse events make several cancer-specific estimates uncertain; pointwise intervals are not multiplicity-adjusted discoveries."
    ))
    cells.append(nbf.v4.new_markdown_cell(
        "## Literature context\n\nPublished frozen bulk-RNA embedding studies report pooled Cox C-indices of approximately **0.750–0.765**, "
        "placing Bridge's 0.7580–0.7589 in the same numerical range ([Gélard et al.](https://proceedings.mlr.press/v259/gelard25a.html); "
        "[multimodal survival study](https://arxiv.org/abs/2505.07683)). A separate scFoundation study found embedding performance comparable to raw expression under Ridge–Cox "
        "but better under Lasso–Cox, reinforcing that conclusions depend on the downstream estimator and regularization "
        "([Liu et al.](https://pubmed.ncbi.nlm.nih.gov/41909608/)). These cross-study values are contextual, not directly paired."
    ))
    cells.append(nbf.v4.new_markdown_cell(
        "## Limitations\n\nTCGA is retrospective and no clinical covariates, calibration, external validation, or prospective testing are included. "
        "Event support is heterogeneous, pooled concordance can reflect between-cancer prognosis differences, and practical benefits of compression were not tested."
    ))
    cells.append(nbf.v4.new_markdown_cell(
        "## Conclusion\n\nA frozen 512-dimensional Bridge representation preserves substantial censoring-aware prognostic information. "
        "Complete OOF evaluation supports stable within-cancer discrimination in a subset of cancers but does not demonstrate systematic superiority over PCA-128 or full expression."
    ))
    nb.cells = cells
    nbf.write(nb, PACKAGE / "tcga_downstream.ipynb")


def provenance(hashes: dict[str, str], validation: dict) -> None:
    source_commit = "unknown"
    try:
        import subprocess
        source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        pass
    record = {
        "analysis": "complete five-fold OOF TCGA survival",
        "source_benchmark": "benchmarks/tcga_downstream",
        "source_git_commit": source_commit,
        "cohort": {"patients": 9668, "cancers": 32, "events": 2774},
        "folds": 5,
        "bootstrap_replicates": 1000,
        "representations": ["Bridge 45.6M", "PCA-128", "Full raw expression"],
        "heads": ["Linear Cox", "Cox MLP [512,256]"],
        "bridge_frozen": True,
        "pca_training_fold_only": True,
        "concordance": "Harrell C-index aggregated over fold-valid comparable pairs",
        "approval_scope": "complete OOF survival only",
        "evidence_sha256": hashes,
        "dependency_versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "matplotlib", "lifelines", "nbformat", "nbconvert", "scipy")
        },
        "environment_note": "lifelines 0.30.0 was used for cached-report validation because the repository-pinned 0.30.3 was unavailable from the configured index and 0.30.2 required a newer Python datetime API; recalculated concordance matched every saved value exactly.",
        "prediction_level_validation": validation,
        "limitations": [
            "retrospective TCGA cohort",
            "no clinical-covariate adjustment",
            "no external validation",
            "heterogeneous event support",
            "cross-study literature values are not paired",
        ],
    }
    (CACHE / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")


def main() -> None:
    PACKAGE.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    hashes = copy_evidence()
    validation = validate_predictions()
    (CACHE / "package_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    plot_overall()
    plot_forest()
    notebook()
    provenance(hashes, validation)
    print(f"built {PACKAGE}")


if __name__ == "__main__":
    main()
