# Task 3 — OSDR batch-effect representation benchmark

This benchmark reproduces the uncorrected PCA analysis in Figure 1 of Sanders
et al. (2023) for OSD-47, -48, -137, -168, -173, -242, and -245, then applies
the same sample cohort to frozen BridgeRNA embeddings. It asks whether known
mission/library-preparation structure is retained, reduced, or reorganized.

The expression reproduction uses DESeq2-compatible median-of-ratios
normalization followed by `log2(normalized count + 1)` and centered, unscaled
PCA. The paper does not explicitly state the log step, but it is disclosed here
because it is required to closely reproduce the published PC1/PC2 fractions.

No batch correction, alignment, fine-tuning, response-vector analysis,
attribution, or pathway analysis is performed.

## Layout

- `pipeline/run_task3.py`: reproducible data assembly, normalization, inference,
  PCA, quantification, and plotting.
- `references/`: paper, Figure 1, and source-method notes.
- `results/`: compact tables, figures, provenance, and executed outputs.
- `work/`: regenerable matrices and embeddings (gitignored).
- `osdr_batch_effect_representation.ipynb`: primary human-readable report.

## Run

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  benchmarks/osdr_batch_effect_representation/pipeline/run_task3.py \
  --device cuda:0 \
  2>&1 | tee benchmarks/osdr_batch_effect_representation/results/task3_run.log
```

Re-execute the notebook after regenerating results:

```bash
.venv/bin/python -m nbconvert --to notebook --execute \
  benchmarks/osdr_batch_effect_representation/osdr_batch_effect_representation.ipynb \
  --output osdr_batch_effect_representation.ipynb \
  --output-dir benchmarks/osdr_batch_effect_representation \
  --ExecutePreprocessor.timeout=1200
```

## Sanders Figure 2 correction reproduction

The four ComBat/ComBat-seq panels use the exact Bioconductor `sva`
implementations in a benchmark-local R library. Install once and run with:

```bash
mkdir -p benchmarks/osdr_batch_effect_representation/.r-lib
Rscript -e '.libPaths(c("benchmarks/osdr_batch_effect_representation/.r-lib", .libPaths())); BiocManager::install("sva", ask=FALSE, update=FALSE)'

.venv/bin/python \
  benchmarks/osdr_batch_effect_representation/pipeline/run_combat_reproduction.py \
  2>&1 | tee benchmarks/osdr_batch_effect_representation/results/combat_reproduction.log
```

ComBat is applied to log2 median-ratio-normalized expression. ComBat-seq is
applied to nonnegative rounded counts, followed by median-ratio normalization
and log2(count + 1) for PCA. FLT/GC is supplied as the biological covariate.
Rounding is necessary because current RSEM estimated counts can be fractional,
while `ComBat_seq` requires integer counts.
