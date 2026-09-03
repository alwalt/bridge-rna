# Mouse ENCODE benchmark

This benchmark evaluates Task 1A: zero-shot human↔mouse tissue retrieval using
GTEx human RNA-seq and selected adult, unperturbed ENCODE mouse total RNA-seq.
The frozen BridgeRNA representation is compared with raw ortholog expression
and PCA without fine-tuning or cross-species normalization.

## Layout

```text
mouse_encode_benchmark.ipynb  primary human-readable analysis
config.json                   frozen dataset selection and settings
pipeline/                     preparation and evaluation scripts
results/                      final compact tables and figures
work/                         regenerable intermediates
```

## Source data

The user-exported ENCODE batch manifest is `data/encode/files.txt`. Its first
line is the metadata query, and the remaining lines are the 123 selected file
URLs. The downloaded metadata are stored at `data/encode/metadata.tsv`; payload
files are stored under `data/encode/files/` and must pass the ENCODE-provided
MD5 checksums before use.

Selection represented by the manifest:

- organism: *Mus musculus*
- assay: total RNA-seq
- biosample type: tissue
- life stage: adult
- unperturbed samples
- 41 experiments and 123 files
- 82 strand-signal bigWigs and 41 gene-quantification TSVs

Raw and downloaded data remain outside the benchmark directory. The notebook
should consume prepared artifacts, while reproducible preparation belongs in
`pipeline/`.

## Task 1A cohort

The corrected primary cohort contains 17 expression-ready, fully unseen ENCODE
experiment profiles across 11 directly matched tissues: adrenal,
subcutaneous adipose, cerebellum, heart, liver, lung, mammary gland, ovary,
spleen, stomach, and testis. GTEx contributes 4,260 human samples. A prespecified
robustness subset contains the five tissues with at least two independent mouse
experiments: heart, liver, lung, spleen, and testis (11 profiles total).

Experiments classified as `same_study_seen` or `exact_sample_seen` are excluded
from the primary analysis. The exposure audit and excluded secondary strata are
retained under `results/`.

## Reproduce

From the repository root:

```bash
.venv/bin/python benchmarks/mouse_encode/pipeline/run_task1a_expanded.py \
  --device cuda:0 --batch-size 4
```

Use `--reuse-prepared` to reuse compatible TPM and frozen-embedding arrays in
`work/`. The script emits a heartbeat during BridgeRNA inference. Final compact
tables and figures are written to `results/`; large regenerable arrays remain
under `work/`.

The representation-geometry diagnostic reuses those frozen outputs:

```bash
.venv/bin/python benchmarks/mouse_encode/pipeline/run_task1a_geometry.py \
  --device cuda:0
```

It compares original/centered centroid cosine, cross-species cosine kNN, a
linear softmax probe, and a fixed shallow MLP. Centering and probe fitting use
the source species only; target labels and target distributions are not used
for transformation or hyperparameter selection.

The balanced geometry extension is stored separately from the strict
fully-unseen analysis and includes all eligible healthy expression-ready mouse
experiments while retaining their pretraining-exposure labels:

```bash
.venv/bin/python benchmarks/mouse_encode/pipeline/run_task1a_balanced_geometry.py \
  --device cuda:0 --batch-size 8 --seeds 20 --base-seed 42
```

Its outputs and representation cache are under
`results/task1a_balanced_geometry/`. The expanded expression, joint-PCA, and
frozen BridgeRNA representations are calculated once; only GTEx sample indices
are resampled across seeds. Existing strict Task 1A outputs are not modified.

The primary human-readable report is
[`mouse_encode_benchmark.ipynb`](mouse_encode_benchmark.ipynb).

## Preprocessing contract

- GTEx v11 gene counts are converted to TPM with GENCODE v49 human exon-union
  lengths.
- ENCODE M21 RSEM `expected_count` values are converted to TPM with the
  species- and sample-specific `effective_length` supplied in each quant file;
  only `ENSMUSG` genome genes enter the TPM denominator.
- Mouse genes are mapped through the repository's one-to-one ortholog table,
  and both species are ordered in the exact 15,165-gene BridgeRNA vocabulary.
- The model input is natural `log1p(TPM)`. No batch correction,
  cross-species normalization, or z-scoring is applied.
