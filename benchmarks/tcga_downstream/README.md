# TCGA downstream benchmark

This benchmark compares frozen RNA foundation-model representations on two
patient-level TCGA tasks:

1. five-cohort cancer classification (BLCA, BRCA, GBM+LGG, LUAD, UCEC);
2. pan-cancer overall survival.

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
