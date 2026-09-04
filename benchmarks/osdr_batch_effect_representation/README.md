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

## Task 3B response vectors

Construct strict within-study and within-stratum FLT-minus-GC responses from
the cached Task 3 representations:

```bash
.venv/bin/python \
  benchmarks/osdr_batch_effect_representation/pipeline/run_task3b_response_vectors.py \
  2>&1 | tee benchmarks/osdr_batch_effect_representation/results/task3b_run.log
```

Task 3B saves full-gene expression responses and 512-D BridgeRNA responses as
compressed NPZ files, PC responses and contrast metadata as CSV files, exact
sample memberships, and the BridgeRNA GC-to-FLT centroid-arrow figure. It does
not calculate cross-contrast similarity, clustering, differential expression,
attribution, or enrichment.

## Task 3C response geometry

Compare the saved Task 3B FLT-minus-GC responses without rerunning inference:

```bash
.venv/bin/python \
  benchmarks/osdr_batch_effect_representation/pipeline/run_task3c_response_geometry.py \
  2>&1 | tee benchmarks/osdr_batch_effect_representation/results/task3c_run.log
```

Task 3C calculates matched cosine-similarity matrices for full expression,
20-PC expression, and full 512-D BridgeRNA response vectors. Its primary
cross-study summaries exclude pairs from the same OSD. It also reports
concordance across mission, library-preparation, sequencing, facility, and
strain boundaries and performs an explicitly exploratory BridgeRNA geometry
cluster scan. It does not run inference, differential expression, attribution,
or pathway analysis.

Audit the two exploratory BridgeRNA geometry clusters against all available
contrast metadata with 9,999 label permutations per variable:

```bash
.venv/bin/python \
  benchmarks/osdr_batch_effect_representation/pipeline/run_task3c_metadata_audit.py \
  2>&1 | tee benchmarks/osdr_batch_effect_representation/results/task3c_metadata_audit.log
```

## Task 3D mode Integrated Gradients

Attribute the two fixed Task 3C response modes on separate GPUs, combine the
contrast-level scores, and run GO Biological Process, KEGG, and Reactome
enrichment against the exact 15,165-gene model universe:

```bash
nohup env CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python \
  benchmarks/osdr_batch_effect_representation/pipeline/run_task3d_mode_ig.py \
  --devices cuda:0 cuda:1 --ig-steps 16 --path-batch 4 \
  > benchmarks/osdr_batch_effect_representation/results/task3d_mode_ig/run.log 2>&1 < /dev/null &
echo $! > benchmarks/osdr_batch_effect_representation/results/task3d_mode_ig/run.pid
```

The frozen target is the projection of the mean-pooled 512-D sample embedding
onto each fixed unit mode direction. Integrated Gradients uses the all-zero
`log1p(TPM)` profile as its documented baseline. No edgeR, conventional GSEA,
batch correction, or deletion experiment is included.

Validate the completed mode IG rankings with frozen-model deletion:

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python \
  benchmarks/osdr_batch_effect_representation/pipeline/run_task3d_deletion_validation.py \
  --devices cuda:0 cuda:1 --random-replicates 10 --heartbeat-seconds 30 \
  2>&1 | tee benchmarks/osdr_batch_effect_representation/results/task3d_mode_ig/deletion_validation.log
```

This recomputes full 512-D embeddings for all 112 samples after masking the
Top 25–1,000 IG genes or 10 deterministic size-matched random panels. It then
reconstructs the strict Task 3B FLT-minus-GC contrasts and measures the fraction
of each fixed mode projection remaining. It does not use edgeR, DE rankings,
GSEA, or additional enrichment.

Calculate signed FLT-minus-GC expression changes for the shared and
mode-specific Top-100 IG genes and the major enriched terms, without rerunning
the encoder or attribution:

```bash
.venv/bin/python \
  benchmarks/osdr_batch_effect_representation/pipeline/run_task3d_signed_expression.py \
  2>&1 | tee benchmarks/osdr_batch_effect_representation/results/task3d_mode_ig/signed_expression_run.log
```

The pathway readout is the median signed `log1p(TPM)` change among mapped IG
genes contributing to each displayed enrichment term. Raw values are retained;
row scaling is used only in the compact heatmap.

The audit preserves condition-specific metadata differences, reports Cramér's
V and permutation p-values, and explicitly flags sparse, missing, constant, and
study-confounded variables. It does not run model inference or biological
interpretation.
