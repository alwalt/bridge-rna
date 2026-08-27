# Paired ARCHS4–recount3 benchmark

This benchmark asks whether the foundation model gives a stable representation
to the **same biological sample** after two independent RNA-seq processing
pipelines. It compares identical human GSMs from two held-out ARCHS4 cohorts:

- `unseen_sample_seen_study`: the GSM was held out, but its GSE occurred in training.
- `strict_unseen_single_gse`: neither the GSM nor its unambiguous GSE occurred in training.

The original target was 1,000 matched GSMs per cohort. A complete metadata audit
found only 574 unambiguous exact matches (562 study-unseen and 12 seen-study), so
the configured benchmark retains every available pair. Selection happens after
recount3 matching so missing coverage is reported rather than silently ignored.

## Important design rules

- The sample manifest remains the authority for model exposure.
- recount3 is the only source used for recount3 matching and expression.
- ARCHS4 and recount3 counts use the same frozen exon lengths and canonical genes.
- Normalization is natural `log1p(TPM)`. CPM is never used.
- Multiple sequencing runs for one GSM are summed before TPM.
- No z-scoring, batch correction, feature selection, or ComBat is applied.
- Only human, bulk, single-GSE manifest samples are included initially.
- Existing ARCHS4 embeddings are used as the reference representation.

## Requirements

Python commands use the repository environment:

```bash
.venv/bin/python --version
```

The download steps require R and the official Bioconductor package:

```r
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("recount3")
```

On the current R 4.1 / Bioconductor 3.14 environment, recount3 1.4.0 requires
the legacy dbplyr collection interface. If `collect(Inf)` fails, install the
compatible release used for this run:

```r
install.packages(
  "https://cran.r-project.org/src/contrib/Archive/dbplyr/dbplyr_2.3.4.tar.gz",
  repos = NULL, type = "source"
)
```

The checkpoint is expected at `model/r7hnr92k/best_model.pt`. The file currently
links to the original checkpoint location; replace it with a durable local file
before archiving or transferring this repository.

## Run order

### 1. Freeze ARCHS4 candidates (metadata only)

```bash
.venv/bin/python benchmarks/paired_recount3/select_candidates.py
```

This writes candidate Parquet and CSV files under `outputs/`. The CSV is the R
interchange file. Candidate selection is deterministic using `config.json`.

### 2. Find candidates in recount3 (metadata only)

```bash
Rscript benchmarks/paired_recount3/match_recount3_samples.R
```

This scans recount3 project metadata and writes `recount3_matches.csv`. It does
not download gene-expression matrices. Inspect its match-status table before
continuing. Preserve `not_found`, `matched_no_run`, and `multiple_projects` rows
as the benchmark attrition record.

Convert the result to Parquet for final selection:

```bash
.venv/bin/python -c "import pandas as pd; p='benchmarks/paired_recount3/outputs/recount3_matches'; pd.read_csv(p+'.csv').to_parquet(p+'.parquet', index=False)"
```

### 3. Select the final study-balanced pairs

```bash
.venv/bin/python benchmarks/paired_recount3/select_candidates.py \
  --mode finalize \
  --matches benchmarks/paired_recount3/outputs/recount3_matches.parquet \
  --output benchmarks/paired_recount3/outputs/final_pairs.parquet
```

The script retains all eligible matches and reports the achieved count. It never
substitutes samples from another exposure class. Because the seen-study arm has
only 12 samples, exposure-stratified results from that arm are exploratory; the
primary processing-stability analysis uses all exact pairs.

### 4. Download only required recount3 projects and counts

```bash
Rscript benchmarks/paired_recount3/download_recount3_counts.R
```

recount3 expression is distributed by project. Project files are cached by
BiocFileCache; the exported Matrix Market artifact contains only selected GSMs.

### 5. Reconstruct paired log1p(TPM)

```bash
.venv/bin/python benchmarks/paired_recount3/preprocess_pairs.py
```

Outputs include two identically ordered sample-by-gene arrays, `samples.parquet`,
`genes.parquet`, and a preprocessing manifest with checksums. Before a full run,
make a small final-pairs file with about 20 GSMs and inspect numerical agreement.

### 6. Encode recount3 expression

```bash
.venv/bin/python benchmarks/paired_recount3/generate_recount3_embeddings.py \
  --device cuda
```

The existing ARCHS4 embeddings are not regenerated or overwritten.

### 7. Analyze paired agreement and retrieval

```bash
.venv/bin/python benchmarks/paired_recount3/analyze_pairs.py
```

Primary outputs are:

- per-pair expression Pearson and Spearman correlations;
- paired embedding cosine similarity;
- matching ARCHS4 sample rank for every recount3 query;
- Recall@1, Recall@5, and Recall@10 by exposure cohort;
- paper-ready PNG/PDF agreement distributions.

## Interpretation

The strongest result is not merely high cosine similarity. The exact paired GSM
should rank ahead of unrelated ARCHS4 samples, and performance should be reported
separately for seen-study and study-unseen samples. A low expression correlation
with a high embedding similarity suggests useful processing robustness. A high
expression correlation with poor retrieval suggests the embedding is losing
sample-level information.

This is a processing-stability benchmark, not an independent biological cohort:
both representations describe the same held-out sample.

## Completed run summary

The complete recount3 metadata audit covered 84,896 eligible human GSMs and
found 574 unambiguous exact pairs from 42 recount3 projects:

| Cohort | Pairs | GSEs |
|---|---:|---:|
| Study-unseen, single GSE | 562 | 40 |
| Unseen sample, seen study | 12 | 2 |

Across all 574 pairs:

| Metric | Result |
|---|---:|
| Median expression Pearson | 0.9680 |
| Median expression Spearman | 0.9600 |
| Median embedding cosine | 0.9992 |
| Exact GSM Recall@1 | 57.8% |
| Exact GSM Recall@5 | 82.8% |
| Exact GSM Recall@10 | 90.8% |
| Top-1 retrieved sample from same GSE | 97.4% |
| Top-5 contains same-GSE sample | 99.1% |

The per-pair table and figures are under `outputs/results/`. The seen-study arm
is too small for a powered exposure comparison and must remain exploratory.
