# TCGA downstream benchmark

This benchmark compares frozen RNA foundation-model representations on two
patient-level TCGA tasks:

1. five-cohort cancer classification (BLCA, BRCA, GBM+LGG, LUAD, UCEC);
2. pan-cancer overall survival.

It also contains a separately labeled BulkFormer-parity extension:

3. 33-class pan-cancer classification on 9,942 unique patients;
4. alive/dead prediction on 9,921 patients, which is not treated as
   time-to-event survival.

All locally evaluated models use the same expression samples, patient-grouped
splits, and preprocessing native to each model. The primary analysis uses each
model's complete frozen representation with matched nonlinear downstream heads:
a 256/128 classification MLP and a 512/256 Cox survival MLP. Our model uses its
native 15,165-gene input, BulkFormer uses its native 20,010-gene input, and the
raw-expression baseline uses all 25,150 TCGA expression features. The frozen
encoders are never fine-tuned. PCA-128 plus linear probes are retained only as
secondary controls. Published BulkRNABert and literature
baseline values are displayed separately and are never presented as results on
our splits.

The benchmark is closed for the current manuscript. The manuscript-quality
record is [SCIENTIFIC_REPORT.md](SCIENTIFIC_REPORT.md), and the concise parity
summary is [BULKFORMER_PARITY_SUMMARY.md](BULKFORMER_PARITY_SUMMARY.md). The
executed human-readable analysis remains
[tcga_downstream_benchmark.ipynb](tcga_downstream_benchmark.ipynb).

## Final analysis inventory

- Five-cohort patient-held-out classification with native MLP heads and PCA
  controls.
- Locally paired 33-class pan-cancer classification and separately labeled
  final-release BulkFormer literature references.
- BulkFormer-style alive/dead prognosis parity.
- Original repeated 80/20 censoring-aware Cox survival analysis.
- Complete true five-fold OOF Cox survival for all 9,668 eligible patients,
  including fold-aware C-index, 1,000-draw patient-bootstrap intervals, event
  support, and matched Bridge/PCA/raw-expression forest plots.

Primary OOF records are:

- `results/survival_5fold_oof_predictions.csv`
- `results/survival_5fold_oof_metrics.csv`
- `results/survival_5fold_oof_per_cancer.csv`
- `results/survival_5fold_oof_representation_comparison.csv`
- `results/survival_5fold_oof_provenance.json`
- `results/final_qa.json`

Primary figures are under `results/figures/`:

- `bridge_5fold_oof_per_cancer_survival_forest.{png,pdf}`
- `pca128_5fold_oof_per_cancer_survival_forest.{png,pdf}`
- `raw_expression_5fold_oof_per_cancer_survival_forest.{png,pdf}`
- `bridge_pca_raw_5fold_oof_survival_forest.{png,pdf}`
- `bridge_pca_raw_5fold_oof_survival_three_panel.{png,pdf}`
- `bridge_vs_conventional_5fold_oof_delta_cindex.{png,pdf}`

The final BulkFormer release reports TCGA weighted F1 = 0.907, but its released
TCGA object contains 10,429 sample rows for 9,679 patients, including repeated
patients and normal-tissue aliquots. The local parity extension therefore uses
one eligible specimen per patient and is not an exact reproduction of that
literature value. See `bulkformer_tcga_gap_audit.md` and
`results/bulkformer_protocol_references.md` for the evidence and unresolved
final-release details.

## Layout

```text
tcga_downstream_benchmark.ipynb  human-readable final record
config.json                      frozen experimental settings
pipeline/                        reproducible preparation/inference/evaluation
results/                         compact tables, figures, logs, provenance
work/                            regenerable matrices, embeddings, split caches
```

## Run

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  benchmarks/tcga_downstream/pipeline/run_benchmark.py \
  --device cuda:0 --heartbeat-seconds 60 \
  2>&1 | tee benchmarks/tcga_downstream/results/final_run.log
```

Follow progress from another terminal:

```bash
tail -f benchmarks/tcga_downstream/results/final_run.log
```

### Frozen-FM pooling ablation

The pooling ablation compares the existing mean-pooled 512-D representation
with learned attention over all 15,165 contextual gene tokens. The FM remains
frozen, and the runner reuses the same splits and head hyperparameters. Because
the full token tensor would occupy about 150 GB in float16, tokens are generated
on demand and results are checkpointed after every completed seed:

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  benchmarks/tcga_downstream/pipeline/run_attention_pooling.py \
  --device cuda:0 --heartbeat-seconds 60 \
  2>&1 | tee benchmarks/tcga_downstream/results/attention_pooling.log
```

The runner explicitly checks that the FM checkpoint and its 15,165-gene input
use `log1p(TPM)`. The 25,150-feature full-expression baseline remains separately
labeled as `log1p(CPM)`.

The runner is resumable: prepared matrices and completed embedding/model tables
are reused. Run the notebook after the result tables exist:

```bash
cd benchmarks/tcga_downstream
../../.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  tcga_downstream_benchmark.ipynb --ExecutePreprocessor.timeout=1200
```

### Per-cancer survival uncertainty

The forest-plot analysis reuses the saved held-out Bridge and full-expression
Cox risks. Because patient-level PCA risks were not retained by the original
run, it refits only the original training-only PCA-128 plus penalized Cox
control on the exact five splits and verifies its pooled C-indices against the
existing result table. It then computes 1,000 patient-clustered bootstrap
replicates per cancer:

```bash
.venv/bin/python \
  benchmarks/tcga_downstream/pipeline/per_cancer_survival_uncertainty.py
```

The resulting C-indices compare patients only within a cancer and within a
held-out split. They are distinct from the pooled pan-cancer C-index.

### Complete five-fold OOF survival

The manuscript-facing survival extension assigns every eligible patient to one
of five mutually exclusive folds, reuses cached frozen Bridge embeddings, and
fits only the existing downstream heads. PCA-128 is refitted inside every
training fold. Its fold-aware C-index never compares risk scores produced by
different fitted heads:

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  benchmarks/tcga_downstream/pipeline/run_survival_5fold_oof.py \
  --device cuda:0 \
  2>&1 | tee benchmarks/tcga_downstream/results/survival_5fold_oof_run.log
```

Completed fold predictions are checkpointed, and all final tables and figures
are regenerated from the consolidated OOF prediction table.

### BulkFormer-parity extension

The extension reuses every existing 9,816-patient embedding cache and performs
new frozen inference only for the 126 LAML patients needed to complete the 33
TCGA labels. It then runs fixed 10-fold patient-level comparisons using
training-fold-only PCA-128 and random forests:

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  benchmarks/tcga_downstream/pipeline/run_bulkformer_parity.py \
  --device cuda:0 --heartbeat-seconds 60 \
  2>&1 | tee benchmarks/tcga_downstream/results/bulkformer_parity_run.log
```

The final repository does not release its downstream training code, PCA
dimension, fold assignments, or random seed. The local frozen configuration is
therefore a conservative, reproducible reconstruction. No setting was changed
after observing test results.

## Clinical endpoint provenance

Pan-cancer overall survival uses the TCGA PanCanAtlas curated endpoint table
(`Survival_SupplementalTable_S1_20171025_xena_sp`) and joins to expression by
TCGA patient barcode. `data/tcga/tcga_survival_labels.csv` is independently
audited but covers only BRCA, KIRC, LUAD, LUSC, and SKCM, so it is not used as
the pan-cancer source.

## Interpretation caveat

Pretraining exposure differs by model. BulkFormer uses TCGA-specific resources;
our model was trained on its documented pretraining manifest. Published
BulkRNABert checkpoints/results may also use TCGA pretraining. The notebook
reports these differences explicitly; this is a downstream-utility comparison,
not a strict uniform unseen-data test.
