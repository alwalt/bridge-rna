# Expression imputation on TCGA

Primary report: [imputation.ipynb](imputation.ipynb).

Fresh inference compares frozen BridgeRNA (45.6M), BulkFormer-50M, and
BulkFormer-147M on the existing frozen 1,000 TCGA primary tumors. All models
use six mask ratios (15%, 30%, 50%, 70%, 90%, 100%) and three mask seeds (0–2).
There are **54 model conditions** using native vocabulary only.

## Protocol

The active run uses native vocabulary only: six masking ratios and three seeds
(0, 1, 2), giving 18 conditions per model and 54 model conditions in total.
All models use the same frozen 1,000 samples. Each masks its own measured genes;
target gene sets and exact masks differ between vocabularies. Thus this run
is not a matched-target comparison and cannot isolate model quality from gene
panel differences. Earlier shared-vocabulary condition caches are preserved
but excluded from the final aggregation.

BridgeRNA has 15,105 measured TCGA genes and BulkFormer 18,949. Each retains
its intended natural log1p(TPM) preprocessing. Gene lengths and TPM denominators
differ, so cross-model MSE is not measured against identical numerical targets.
Pearson/Spearman are calculated within samples across masked genes; summaries
average samples first. Error bars show SD across three mask seeds, not patient
confidence intervals. Undefined correlations remain NaN, with counts reported.

Training-gene mean and median use frozen profiles fitted to 5,000 ARCHS4 human
training samples. Both are retained numerically; the main figure displays mean.

## Layout and provenance

- `pipeline/`: reproducible preparation, inference, aggregation, and report code.
- `results/`: saved manifest, provenance, timing pilots, logs, numerical results,
  correlation QC, and final PNG/PDF/SVG figures.
- `work/`: ignored prepared matrices and fresh per-condition score caches.
- `imputation.ipynb`: executed report; it does not trigger model inference.

The initial run uses relative symlinks to audited prepared matrices in
`research/tcga_imputation/work/`, avoiding duplicate large inputs. No old model
prediction/score cache is read. Frozen input hashes and checkpoint hashes are
recorded in `results/run_provenance.json`. Old research results remain untouched.
If preparing independent inputs, remove **only these symlinks** first, then run
`prepare_tcga.py` and `prepare_frozen_baselines.py`; both write inside this benchmark.
The copied preparation code uses the same deterministic cohort and gene mapping.
Preparation requires local TCGA H5, ARCHS4 H5/training split, canonical genes,
GENCODE lengths, pinned HGNC annotations, and the original model resources.
See `results/preparation_manifest.json` for the original input provenance.

## Commands

Run from the repository root using the existing `.venv` environment. Dependencies
are listed in `requirements.txt`; notebook reporting additionally needs nbformat,
nbclient, ipykernel, and matplotlib. The local CUDA/PyG environment and original
BulkFormer model implementation are required for inference.

```bash
# Only needed when rebuilding the prepared inputs rather than reusing them:
.venv/bin/python benchmarks/imputation/pipeline/prepare_tcga.py
.venv/bin/python benchmarks/imputation/pipeline/prepare_frozen_baselines.py

# Pilot each model before the full run:
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u benchmarks/imputation/pipeline/fresh_run.py --model ours_45.6m --pilot
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u benchmarks/imputation/pipeline/fresh_run.py --model bulkformer_50m --pilot
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u benchmarks/imputation/pipeline/fresh_run.py --model bulkformer_147m --pilot

# Validate both protocols, then run BridgeRNA and 50M on GPU 0, 147M on GPU 1:
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u benchmarks/imputation/pipeline/validate_protocol.py
.venv/bin/python -u benchmarks/imputation/pipeline/launch_fresh.py > benchmarks/imputation/results/run.log 2>&1

# Rebuild summaries and execute the report after inference:
.venv/bin/python benchmarks/imputation/pipeline/fresh_run.py --aggregate
.venv/bin/python benchmarks/imputation/pipeline/build_notebook.py
```

`launch_fresh.py` automatically aggregates and executes the report after both
workers finish successfully. Restarting it resumes only this fresh run's own
condition caches. A genuinely new protocol must use a separate results/work
location rather than overwrite these outputs.

Monitor individual workers:

```bash
tail -f benchmarks/imputation/results/run_ours_45.6m.log benchmarks/imputation/results/run_bulkformer_50m.log benchmarks/imputation/results/run_bulkformer_147m.log
```

Results should support only claims tested here: random masked-expression
reconstruction in this TCGA cohort. TCGA exposure during checkpoint pretraining
is not established. Random masks do not capture every experimental missingness
mechanism. The existing graph resources are reused without asserting they were
constructed independently of TCGA.

## Runtime and numerical repeatability

The active native-only, three-seed run is estimated to take 65–90 minutes from
restart on two RTX 3090s. Earlier runtime estimates refer to the superseded
10-seed/two-protocol plan and are archived. Small-cohort protocol validation
passed for all models at 15% and 100% masking. Identical BulkFormer forward
passes differed slightly numerically, with maximum Pearson/Spearman differences
below 0.000007 in that check; exact values are in `results/repeatability_*.json`.

## Figure styling

The notebook displays Pearson/Spearman together and MSE in a separate block.
`pipeline/plot_figures.py` matches the existing manuscript style: Arial,
183 mm width, 7–8 pt labels, thin axes, and colorblind-safe blue/green/vermillion
model colors. Only the BRIDGE-target training mean is shown as a dashed gray line; both target-space baselines remain in the numerical tables. Exports are
`results/native_imputation_correlations.{png,pdf,svg}` and
`results/native_imputation_mse.{png,pdf,svg}`; PNG is 600 dpi and SVG text remains
editable. Numerical source and styling provenance are saved alongside them.
