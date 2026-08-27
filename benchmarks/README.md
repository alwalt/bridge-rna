# Foundation-model benchmarks

This directory contains paper-oriented, reproducible benchmark notebooks and
their supporting outputs. Notebooks should use reusable code from `src/fm_embed`
rather than reimplementing cohort selection or embedding preprocessing.

## Planned workflow

1. Load and validate the two ARCHS4 held-out comparison cohorts.
2. Generate/load GTEx embeddings using TPM-based preprocessing (not CPM).
3. Generate/load TCGA embeddings using TPM-based preprocessing.
4. Add OSDR as the external mouse benchmark.
5. Run matched classification, retrieval, and study-generalization analyses.

`01_cross_dataset_benchmark.ipynb` begins with the ARCHS4 cohort-loading and
leakage-validation section. It intentionally reads embeddings lazily in batches.

## Paired processing benchmark

[`paired_recount3/README.md`](paired_recount3/README.md) defines the paired
ARCHS4–recount3 benchmark. It processes identical held-out GSMs through recount3
and tests expression agreement, embedding cosine similarity, and paired-sample
retrieval while stratifying results by training-study exposure.
