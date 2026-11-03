#!/usr/bin/env python3
"""Build the cached scientific notebook for the drug-response benchmark."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
import nbformat as nbf
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error


BENCHMARK = Path(__file__).resolve().parents[1]
REPO = BENCHMARK.parents[1]
RESULTS = BENCHMARK / "results"
WORK = BENCHMARK / "work"
PACKAGE = REPO / "notebooks" / "drug_response"
TABLES = PACKAGE / "cached_evidence" / "tables"
FIGURES = PACKAGE / "figures"

TABLE_NAMES = [
    "primary_results.csv",
    "per_drug_metrics.csv",
    "paired_comparisons.csv",
    "pca_component_sensitivity.csv",
    "bridge_vs_pca4_per_screen.csv",
    "fold_diagnostics.csv",
    "variance_controls.csv",
    "published_results.csv",
]
FIGURE_NAMES = [
    "pca_component_sensitivity.png",
    "pca_component_sensitivity.pdf",
    "bridge_vs_pca4_per_screen.png",
    "bridge_vs_pca4_per_screen.pdf",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_predictions() -> dict[str, object]:
    sys.path.insert(0, str(BENCHMARK / "pipeline"))
    from evaluate import matrices  # noqa: PLC0415

    samples, screens, targets = matrices()
    config = json.loads((BENCHMARK / "config.json").read_text())
    metrics = pd.read_csv(RESULTS / "per_drug_metrics.csv")
    saved = metrics.set_index(["representation", "screen_id"])

    assert len(samples) == 700
    assert samples.depmap_id.is_unique
    assert samples.groupby("fold").size().eq(140).all()
    assert targets.shape == (700, 355)
    assert int(np.isfinite(targets).sum()) == 212_299

    max_errors = {"pcc": 0.0, "scc": 0.0, "rmse": 0.0}
    for name in config["representations"]:
        predictions = np.load(WORK / f"{name}_oof_predictions.npy", mmap_mode="r")
        assert predictions.shape == targets.shape
        assert np.array_equal(np.isfinite(predictions), np.isfinite(targets))
        for index, screen in enumerate(screens):
            keep = np.isfinite(targets[:, index])
            truth = targets[keep, index]
            estimate = predictions[keep, index]
            recalculated = {
                "pcc": float(pearsonr(truth, estimate).statistic),
                "scc": float(spearmanr(truth, estimate).statistic),
                "rmse": float(mean_squared_error(truth, estimate) ** 0.5),
            }
            row = saved.loc[(name, screen)]
            for metric, value in recalculated.items():
                error = abs(value - float(row[metric]))
                max_errors[metric] = max(max_errors[metric], error)
                assert error < 1e-6, (name, screen, metric, error)

    summary = metrics.groupby("representation", as_index=False).agg(
        mean_pcc=("pcc", "mean"),
        median_pcc=("pcc", "median"),
        mean_scc=("scc", "mean"),
        rmse=("rmse", "mean"),
        evaluable_drugs=("screen_id", "nunique"),
    )
    primary = pd.read_csv(RESULTS / "primary_results.csv")
    checked = primary.merge(summary, on="representation", suffixes=("_saved", "_recalculated"))
    for metric in ["mean_pcc", "median_pcc", "mean_scc", "rmse"]:
        assert np.allclose(
            checked[f"{metric}_saved"], checked[f"{metric}_recalculated"], rtol=0, atol=1e-14
        )
    assert checked.evaluable_drugs_saved.eq(355).all()
    return {
        "status": "passed",
        "representations": len(config["representations"]),
        "screens": len(screens),
        "cell_lines": len(samples),
        "observed_responses": int(np.isfinite(targets).sum()),
        "prediction_mask": "exactly one finite prediction per observed response; none at missing responses",
        "maximum_absolute_metric_error": max_errors,
    }


def build_notebook() -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            """# Frozen transcriptomic representations predict drug sensitivity in held-out cancer cell lines

**Scientific question.** Does a frozen BridgeRNA representation of baseline cell-line expression contain information that predicts measured IC50 variation for known drug screens in cell lines excluded from downstream readout training?

**Hypothesis.** Under identical held-out-cell-line splits and a common linear readout, Bridge-45.6M will yield higher within-screen out-of-fold Pearson correlation than conventional expression summaries and alternative frozen transcriptomic representations.

This analysis evaluates **known drug screen → unseen cell line** prediction. It does not test unseen compounds, post-treatment expression, causal biomarkers, clinical efficacy, or drug repurposing. Data derive from the released [BulkFormer drug-response record](https://doi.org/10.5281/zenodo.15744294)."""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import json
import pandas as pd
from IPython.display import Image, display

ROOT = Path.cwd()
TABLES = ROOT / "cached_evidence" / "tables"
FIGURES = ROOT / "figures"
PROVENANCE = json.loads((ROOT / "cached_evidence" / "provenance.json").read_text())
assert PROVENANCE["verification"]["status"] == "passed"
PROVENANCE["cohort"]"""
        ),
        nbf.v4.new_markdown_cell(
            """## Data and cohort

The cohort contains 700 distinct DepMap cancer cell-line models spanning 27 broad lineages, not repeated samples from one tumor. One released baseline expression profile per cell line is reused across 212,299 measured responses from 355 GDSC1/GDSC2 version-specific screens (295 drug identifiers; 255 PubChem compounds). Screens contain 134–697 observed cell lines (median 658).

Five deterministic, lineage-stratified folds hold out 140 complete cell lines each. No response from a held-out cell line enters downstream readout fitting. Because screens reuse cell lines and some compounds occur in both GDSC versions, the 355 screens and 212,299 responses are not independent biological replicates."""
        ),
        nbf.v4.new_markdown_cell(
            """## Preprocessing and representations

The raw and PCA controls use all 19,098 released processed expression features unchanged. The matrix contains negative values and its exact transform is undocumented. For frozen-model inference, negative values were clipped to zero and remaining values were treated as `log2(TPM+1)` before multiplication by `ln(2)`. This is an explicit approximation and is not equivalent to canonical natural `log1p(TPM)` if the release was batch-corrected or otherwise transformed.

Bridge uses its frozen standard 512-dimensional sample representation with 15,130 of 15,165 canonical genes observed; 35 absent genes are zero-filled. Local BulkFormer controls use frozen mean-pooled final contextual-token representations (259 dimensions for BulkFormer-50M; 643 for BulkFormer-147M). Conventional controls include raw expression, PCA-1–8 and PCA-16/32/64/128/256/512, and PCA-128 concatenated with Bridge. PCA is independently fitted on standardized training-fold expression and applied to held-out cells; PCA scores, not loadings, enter ridge regression."""
        ),
        nbf.v4.new_markdown_cell(
            """## Evaluation

Every representation uses the same standardized multi-output ridge framework with a separate output for each version-specific screen. The fixed penalty is `alpha = 1000 × dimensions / 128`. Missing training responses receive one-pass training-screen-mean completion; depending on screen and fold, 0.18%–81.43% of training rows are completed. Predictions are assembled out of fold.

PCC, SCC, and RMSE are calculated within each screen and summarized as unweighted means across 355 screens; median PCC is also reported. Paired intervals use 10,000 bootstrap resamples of screens and are descriptive because screens share cell lines and repeated compounds. One split partition and one representation extraction per model are available, so no across-seed mean or SD is reported."""
        ),
        nbf.v4.new_code_cell(
            """primary = pd.read_csv(TABLES / "primary_results.csv")
metrics = pd.read_csv(TABLES / "per_drug_metrics.csv")
recalculated = metrics.groupby("representation", as_index=False).agg(
    mean_pcc=("pcc", "mean"), median_pcc=("pcc", "median"),
    mean_scc=("scc", "mean"), rmse=("rmse", "mean"),
    evaluable_drugs=("screen_id", "nunique"))
check = primary.merge(recalculated, on="representation", suffixes=("_saved", "_recalculated"))
for metric in ["mean_pcc", "median_pcc", "mean_scc", "rmse"]:
    assert (check[f"{metric}_saved"] - check[f"{metric}_recalculated"]).abs().max() < 1e-12
primary.sort_values("mean_pcc", ascending=False).round(3)"""
        ),
        nbf.v4.new_markdown_cell(
            """## Results

Bridge-45.6M achieved mean per-screen PCC 0.448 (median 0.461), mean SCC 0.424, and mean per-screen RMSE 1.235. Locally evaluated BulkFormer-50M and BulkFormer-147M achieved mean PCC 0.438 and 0.436. The Bridge differences were small: mean ΔPCC 0.0102 and 0.0120, with Bridge higher on 71.8% and 74.6% of screens, respectively. The published BulkFormer PCC 0.373 is literature context only because its final folds, readout, drug grouping, and aggregation cannot be reconstructed."""
        ),
        nbf.v4.new_code_cell(
            """paired = pd.read_csv(TABLES / "paired_comparisons.csv")
paired.round(4)"""
        ),
        nbf.v4.new_markdown_cell(
            """### PCA dimensionality sensitivity

The complete exploratory PCA curve has an interior optimum at PCA-4 (mean PCC 0.360); PCA-3 and PCA-5 achieve 0.359 and 0.357. Bridge exceeds PCA-4 by mean ΔPCC 0.0883 (95% screen-bootstrap interval 0.0810–0.0959) and is higher on 342 of 355 screens (96.3%). Because component count was expanded after inspecting the original PCA-128/PCA-512 controls, PCA-4 is the **best observed exploratory setting**, not a prospectively selected hyperparameter."""
        ),
        nbf.v4.new_code_cell(
            """pca = pd.read_csv(TABLES / "pca_component_sensitivity.csv")
display(Image(filename=str(FIGURES / "pca_component_sensitivity.png")))
pca.round(3)"""
        ),
        nbf.v4.new_code_cell(
            """per_screen = pd.read_csv(TABLES / "bridge_vs_pca4_per_screen.csv")
assert len(per_screen) == 355 and per_screen["screen_id"].is_unique
display(Image(filename=str(FIGURES / "bridge_vs_pca4_per_screen.png")))
per_screen.groupby("dataset_version").agg(
    screens=("screen_id", "size"),
    mean_bridge_pcc=("bridge_pcc", "mean"),
    mean_pca4_pcc=("pca4_pcc", "mean"),
    mean_delta_pcc=("delta_pcc", "mean"),
).round(3)"""
        ),
        nbf.v4.new_markdown_cell(
            """### Negative and metric-dependent findings

Raw expression has mean PCC 0.196 but the highest mean SCC, 0.440, showing that rank association, linear correlation, and absolute error support different conclusions. PCA-128 concatenated with Bridge reaches mean PCC 0.284, below Bridge alone, so this run does not establish useful complementarity from appending PCA features."""
        ),
        nbf.v4.new_markdown_cell(
            """## Limitations

- The source expression transform is undocumented; frozen-model inputs use an approximate conversion that may affect architectures differently.
- Exact Bridge pretraining exposure is unresolved because GEO/SRA sample and study accessions are unavailable.
- Results use one deterministic five-fold partition; screen-bootstrap intervals do not capture split variability and may understate dependence among screens.
- Mean-completed targets and a fixed, dimension-scaled ridge penalty were not replaced by observed-only fitting and nested training-only hyperparameter selection.
- Every screen has training observations and its own output; unseen-drug generalization is not tested.
- The local BulkFormer comparison is not a reproduction of the published PCC 0.373.

## Conclusion

Frozen Bridge representations contain baseline cellular-state information that predicts IC50 variation in held-out cell lines under a common lightweight readout. Bridge is higher than every tested PCA dimension and shows a small local advantage over both frozen BulkFormer checkpoints, but the evidence supports representation utility under this protocol—not state-of-the-art pharmacogenomic prediction, unseen-drug generalization, or direct improvement over the published BulkFormer result."""
        ),
    ]
    return notebook


def main() -> None:
    verification = validate_predictions()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for name in TABLE_NAMES:
        shutil.copy2(RESULTS / name, TABLES / name)
    for name in FIGURE_NAMES:
        shutil.copy2(RESULTS / name, FIGURES / name)

    asset_manifest = json.loads((RESULTS / "asset_manifest.json").read_text())
    sanitized_assets = [
        {
            "name": Path(item["path"]).name,
            "bytes": item["bytes"],
            "sha256": item["sha256"],
            "zenodo_md5": item["zenodo_md5"],
        }
        for item in asset_manifest["assets"]
    ]
    source_artifacts = [RESULTS / name for name in TABLE_NAMES] + [RESULTS / name for name in FIGURE_NAMES]
    provenance = {
        "source_git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "approval_scope": "Supplementary frozen drug-response representation benchmark approved by explicit benchmark-notebook invocation.",
        "source_benchmark": "benchmarks/drug_response",
        "source_release": {
            "doi": asset_manifest["zenodo_record"],
            "assets": sanitized_assets,
        },
        "cohort": {
            "cell_lines": 700,
            "broad_lineages": 27,
            "response_observations": 212_299,
            "version_specific_screens": 355,
            "gdsc_drug_ids": 295,
            "pubchem_compounds": 255,
            "folds": 5,
            "test_cell_lines_per_fold": 140,
        },
        "model": {
            "bridge_name": "Bridge-45.6M",
            "checkpoint_recorded_in_source_benchmark": "model/r7hnr92k/best_model.pt",
            "frozen": True,
            "representation_dimensions": 512,
            "canonical_genes": 15_165,
            "observed_canonical_genes": 15_130,
        },
        "protocol": {
            "split": "deterministic lineage-stratified held-out cell lines",
            "seed": 20_260_922,
            "readout": "standardized independent ridge output per version-specific screen",
            "alpha": "1000 * dimensions / 128",
            "missing_targets": "one-pass training-screen-mean completion",
            "metric": "PCC/SCC/RMSE per screen over pooled OOF predictions; unweighted mean across screens",
            "pca_dimensions": [1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256, 512],
        },
        "verification": verification,
        "dependencies": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
            "nbformat": nbf.__version__,
        },
        "source_artifacts": [
            {
                "path": str(path.relative_to(REPO)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in source_artifacts
        ],
        "limitations": [
            "Released expression preprocessing is undocumented and frozen-model input conversion is approximate.",
            "One split realization; screen-bootstrap intervals do not capture split variability or full dependence.",
            "One-pass mean completion and fixed dimension-scaled ridge penalty were not nested-CV optimized.",
            "Exact Bridge pretraining exposure is unresolved.",
            "Known-screen to unseen-cell-line prediction only; unseen compounds are not tested.",
            "Published BulkFormer PCC 0.373 is not protocol-equivalent to the local comparison.",
        ],
    }
    evidence = PACKAGE / "cached_evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    nbf.write(build_notebook(), PACKAGE / "drug_response.ipynb")

    readme = """# Frozen drug-response representation benchmark

This directory contains an executed scientific report built from validated cached evidence in `benchmarks/drug_response/`. The notebook runs without model inference, network access, raw expression matrices, checkpoints, embeddings, or prediction arrays.

## Lightweight cached report

Tested with Python 3.11, NumPy 1.26, pandas 2.3, SciPy 1.17, scikit-learn 1.8, Matplotlib 3.10, nbformat 5.11, and Jupyter nbconvert.

```bash
python -m pip install numpy pandas scipy scikit-learn matplotlib nbformat jupyter
cd notebooks/drug_response
jupyter nbconvert --to notebook --execute --inplace drug_response.ipynb \\
  --ExecutePreprocessor.timeout=1200
```

Cached execution should complete in seconds on CPU. The notebook reads only `cached_evidence/` and `figures/`.

## Rebuild the cached package

From the repository root, after the full benchmark outputs are available:

```bash
python benchmarks/drug_response/pipeline/build_scientific_notebook.py
```

This validates saved out-of-fold predictions against per-screen metrics, copies compact evidence, and regenerates the unexecuted notebook. Execute it with the command above.

## Full benchmark

The source release is [Zenodo record 10.5281/zenodo.15744294](https://doi.org/10.5281/zenodo.15744294). Full execution additionally requires the Bridge checkpoint, both BulkFormer checkpoints and vocabularies, the canonical Bridge gene vocabulary, a configured PyTorch environment, and sufficient local storage for the released expression/response tables and cached arrays. Paths are configured in `benchmarks/drug_response/config.json` and may need local adjustment.

```bash
python benchmarks/drug_response/pipeline/audit_data.py
python benchmarks/drug_response/pipeline/prepare_inputs.py
python benchmarks/drug_response/pipeline/make_splits.py
python benchmarks/drug_response/pipeline/extract_embeddings.py --model bridge_45.6m --device cuda:0
python benchmarks/drug_response/pipeline/extract_embeddings.py --model bulkformer_50m --device cuda:0
python benchmarks/drug_response/pipeline/extract_embeddings.py --model bulkformer_147m --device cuda:0
python benchmarks/drug_response/pipeline/evaluate.py
python benchmarks/drug_response/pipeline/make_figures.py
```

Neural representation extraction is the expensive stage and should use an available GPU; runtime depends on hardware. The saved run processes 700 cell lines. The downstream PCA/ridge evaluation is CPU-light (minutes on the validation machine). Pipeline commands emit progress heartbeats; redirect output to a log and monitor with `tail -f <log>` for long inference runs.

## Evidence and interpretation

- `cached_evidence/tables/`: compact numerical source tables.
- `cached_evidence/provenance.json`: source hashes, cohort, protocol, dependencies, validation outcomes, and limitations.
- `figures/`: PNG and PDF figures paired with CSV source tables in cached evidence.
- Scientific review: [`../../benchmarks/drug_response/PUBLICATION_REVIEW.md`](../../benchmarks/drug_response/PUBLICATION_REVIEW.md)
- Manuscript outline: [`../../benchmarks/drug_response/MANUSCRIPT_OUTLINE.md`](../../benchmarks/drug_response/MANUSCRIPT_OUTLINE.md)

The research pipeline under `benchmarks/drug_response/pipeline/` remains the computational source of truth.
"""
    (PACKAGE / "README.md").write_text(readme)
    print(f"validated {verification['representations']} representations and wrote {PACKAGE}")


if __name__ == "__main__":
    main()
