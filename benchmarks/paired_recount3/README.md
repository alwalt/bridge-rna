# Paired ARCHS4–recount3 benchmark

This benchmark asks whether the foundation model gives a stable representation
to the **same biological sample** after two independent RNA-seq processing
pipelines. The benchmark uses 562 exact human GSM pairs for which neither the
sample nor its unambiguous GSE occurred in training (`strict_unseen_single_gse`).

The original target was 1,000 matched GSMs per cohort. A metadata audit
found 562 eligible unseen-study matches. Selection happens after
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
.venv/bin/python benchmarks/paired_recount3/pipeline/select_candidates.py
```

This writes candidate Parquet and CSV files under `work/`. The CSV is the R
interchange file. Candidate selection is deterministic using `config.json`.

### 2. Find candidates in recount3 (metadata only)

```bash
Rscript benchmarks/paired_recount3/pipeline/match_recount3_samples.R
```

This scans recount3 project metadata and writes `recount3_matches.csv`. It does
not download gene-expression matrices. Inspect its match-status table before
continuing. Preserve `not_found`, `matched_no_run`, and `multiple_projects` rows
as the benchmark attrition record.

Convert the result to Parquet for final selection:

```bash
.venv/bin/python -c "import pandas as pd; p='benchmarks/paired_recount3/work/recount3_matches'; pd.read_csv(p+'.csv').to_parquet(p+'.parquet', index=False)"
```

### 3. Select the final study-balanced pairs

```bash
.venv/bin/python benchmarks/paired_recount3/pipeline/select_candidates.py \
  --mode finalize \
  --matches benchmarks/paired_recount3/work/recount3_matches.parquet \
  --output benchmarks/paired_recount3/work/final_pairs.parquet
```

The script retains all eligible matches and reports the achieved count. It never
substitutes unmatched samples. Exposure groups are not treated as comparative
benchmark arms; the processing-stability analysis uses all exact pairs.

### 4. Download only required recount3 projects and counts

```bash
Rscript benchmarks/paired_recount3/pipeline/download_recount3_counts.R
```

recount3 expression is distributed by project. Project files are cached by
BiocFileCache; the exported Matrix Market artifact contains only selected GSMs.

### 5. Reconstruct paired log1p(TPM)

```bash
.venv/bin/python benchmarks/paired_recount3/pipeline/preprocess_pairs.py
```

Outputs include two identically ordered sample-by-gene arrays, `samples.parquet`,
`genes.parquet`, and a preprocessing manifest with checksums. Before a full run,
make a small final-pairs file with about 20 GSMs and inspect numerical agreement.

### 6. Encode recount3 expression

```bash
.venv/bin/python benchmarks/paired_recount3/pipeline/generate_recount3_embeddings.py \
  --device cuda
```

The existing ARCHS4 embeddings are not regenerated or overwritten.

### 7. Run the publication notebook

Open and run `paired_recount3_benchmark.ipynb`. It computes both 562-sample and
full-human-ARCHS4 retrieval directly from the frozen inputs. Primary outputs are:

- per-pair expression Pearson and Spearman correlations;
- paired embedding cosine similarity;
- matching ARCHS4 sample rank for every recount3 query;
- Recall@1, Recall@5, Recall@10, and Recall@100 where applicable;
- paired Top-1 failure annotations and full-corpus Top-10 neighbors;
- a single publication summary table.

The notebook is the analysis source of truth. It never downloads data or
regenerates embeddings.

## Folder map

- `paired_recount3_benchmark.ipynb`: primary analysis and collaborator-facing report.
- `results/`: final small tables and figures produced by the notebook.
- `pipeline/`: upstream selection, download, preprocessing, and embedding scripts.
- `work/paired_expression/` and `work/recount3_embeddings.npy`: frozen notebook inputs.
- `work/recount3_counts/`: intermediate recount3 matrix retained for provenance.
- `work/candidate_samples.*`, `work/recount3_matches.*`, and `work/final_pairs.*`:
  sample-selection audit trail.
- `work/recount3_metadata_cache/`: restart cache, not an analysis result.

## Interpretation

The strongest result is not merely high cosine similarity. The exact paired GSM
should rank ahead of unrelated ARCHS4 samples in both the paired and complete
human search spaces. A low expression correlation
with a high embedding similarity suggests useful processing robustness. A high
expression correlation with poor retrieval suggests the embedding is losing
sample-level information.

This is a processing-stability benchmark, not an independent biological cohort:
both representations describe the same held-out sample.

## Completed run summary

The recount3 metadata audit considered the eligible human GSMs and found 562
unambiguous unseen-study pairs from 40 recount3 projects.

Across all 562 pairs:

| Metric | Result |
|---|---:|
| Median expression Pearson | 0.9680 |
| Median expression Spearman | 0.9600 |
| Median embedding cosine | 0.9992 |
| Exact GSM Recall@1 | 58.7% |
| Exact GSM Recall@5 | 84.2% |
| Exact GSM Recall@10 | 92.0% |
| Top-1 retrieved sample from same GSE | 98.2% |
| Top-5 contains same-GSE sample | 100.0% |

Against all 510,709 human ARCHS4 embeddings, exact-GSM Recall@1, Recall@10,
and Recall@100 are 28.5%, 77.2%, and 91.8%, respectively; median rank is 2.

The publication tables are under `results/`; they contain only the 562
unseen-study queries.
