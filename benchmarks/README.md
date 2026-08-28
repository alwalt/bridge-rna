# Foundation-model benchmarks

This directory is the index for paper-oriented, reproducible benchmarks. Each
benchmark lives in its own self-contained folder. Start with the linked notebook;
the folder README documents provenance and rerun instructions.

## Benchmark index

| Benchmark | Question | Primary notebook | Status |
|---|---|---|---|
| Paired ARCHS4–recount3 | Are expression and FM embeddings robust to two independent processing pipelines for the same unseen-study samples? | [`paired_recount3/paired_recount3_benchmark.ipynb`](paired_recount3/paired_recount3_benchmark.ipynb) | Complete |

## Folder convention

Every new benchmark should use:

```text
benchmarks/
├── README.md                         # this index
└── <benchmark_name>/
    ├── <benchmark_name>_benchmark.ipynb  # primary human-readable analysis
    ├── README.md                         # design, inputs, and reproduction
    ├── config.json                       # frozen benchmark parameters, if needed
    ├── pipeline/                         # upstream preparation code
    ├── results/                          # final tables and figures
    └── work/                             # ignored intermediates and caches
```

Use one folder per scientific question—not one folder per dataset. For example,
a GTEx/TCGA/ARCHS4 comparison belongs in one cross-dataset benchmark folder if
all three datasets answer the same question. The notebook should be the obvious
entry point and should state which files are final results versus restart caches.

Reusable cohort selection, embedding access, preprocessing, and metrics belong
in `src/fm_embed/`; benchmark folders should contain only benchmark-specific
orchestration and reporting.
