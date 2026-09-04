# BridgeRNA Repository Guide

## Purpose

This repository develops and evaluates BridgeRNA, a frozen bulk RNA-seq foundation model. Keep analyses reproducible, scientifically conservative, and easy for collaborators to navigate.

## Repository organization

- Put reusable library code in `src/`.
- Put data inspection and manifest utilities in `scripts/data_audit/`.
- Put genuinely one-off analyses in `scripts/one_off/`.
- Give each benchmark its own directory under `benchmarks/<benchmark_name>/`.
- Follow this benchmark layout:

```text
benchmarks/<benchmark_name>/
├── README.md
├── <benchmark_name>.ipynb
├── pipeline/
├── results/
├── work/
└── references/          # only when needed
```

- `pipeline/` contains the reproducible source-of-truth scripts.
- `work/` contains regenerable caches, downloaded intermediates, and large arrays. It must remain git-ignored.
- `results/` contains compact machine-readable tables, provenance, logs, and final figures.
- Keep one primary notebook per benchmark. It is the human-readable report, not a second implementation of the pipeline.
- Do not place benchmark notebooks loose in `benchmarks/`.

## Benchmark workflow

1. Inspect existing assets and utilities before downloading or recomputing anything.
2. Define and save the exact cohort/sample manifest before analysis.
3. Implement computations in reusable scripts under `pipeline/`.
4. Save raw numerical results without rounding.
5. Build figures and summary tables from saved results, never invented or hard-coded values.
6. Add a concise section to the benchmark notebook and execute it successfully.
7. Update the benchmark `README.md` when the workflow or primary commands change.

Do not overwrite prior analyses when adding a materially different cohort or protocol. Use a clearly named results subdirectory and preserve previous outputs.

## Data and model rules

- Treat source data as read-only. Derived artifacts belong in `work/` or `results/`.
- Reuse the canonical 15,165-gene vocabulary and exact ordering from `data/ensembl/canonical_genes.csv` when evaluating BridgeRNA.
- Use species-correct gene annotations and gene lengths when deriving TPM.
- BridgeRNA input is natural `log1p(TPM)`. Do not substitute CPM or another transformation unless the benchmark explicitly tests it and labels it clearly.
- Record gene mapping losses, ambiguities, missing genes, and the final evaluated gene count.
- Use the existing checkpoint and standard inference utilities. Keep the backbone frozen unless a task explicitly authorizes training or fine-tuning.
- Cache expensive deterministic representations when practical, with sample order and provenance.
- Never count technical resequencing or repeated material as independent biological replication.

## Experimental rigor

- Split samples by study when study leakage is possible.
- Keep discovery/ranking data separate from final evaluation data.
- Preserve frozen panels, rankings, cohorts, contrasts, and labels once validation begins.
- Use the same samples, masks, seeds, and scoring definitions for model comparisons where applicable.
- Make random seeds deterministic and save selected samples/panels.
- For paired or stratified studies, preserve the established pairing, condition direction, and biological strata.
- Do not pool distinct missions, strains, durations, preservation methods, or technical strata merely to increase sample size.
- Keep sample exposure and study exposure separate. Prefer fully unseen studies for strict external evaluation.
- Do not redefine clusters or modes using a validation assay intended to test them.
- Distinguish biological replication, technical replication, same-cohort evidence, and publication-only evidence.

## Metrics and statistics

- Reuse existing metric implementations whenever possible.
- Evaluate reconstruction only on masked positions.
- Report mean, SD, and replicate/seed count for repeated experiments.
- Preserve per-sample, per-seed, or per-gene outputs needed to audit summaries.
- Use effect sizes and individual observations for very small cohorts; do not overstate exploratory p-values.
- Use the declared eligible-gene universe as the enrichment background. For assay-specific enrichment, use the genes or proteins actually measurable in that assay.
- Treat PCA correctly: fit it once on the specified sample-level matrix, then form contrasts in the fitted space. Report rank-limited dimensionality and cumulative explained variance.
- Verify invariants with assertions, including sample order, vocabulary order, split overlap, dimensions, and mask/probe separation.

## Results and interpretation

- Save CSV or Parquet tables alongside PNG and PDF figures.
- Prefer publication-quality figures with readable labels, consistent ordering, explicit units, and appropriate chance lines/error bars.
- Keep raw stored values unrounded; round only display tables and plot annotations.
- Every analysis should save a provenance JSON or equivalent containing inputs, cohort size, parameters, seeds, checkpoint, preprocessing, and limitations.
- Report negative results and failures as clearly as positive results.
- Distinguish representation robustness from batch correction. Do not claim BridgeRNA is batch-corrected unless that was directly tested.
- Stronger clustering alone is not evidence of better biology; prioritize held-out performance, controlled comparisons, and technical reproducibility.
- End benchmark sections with a short conclusion that answers the stated scientific question and names major limitations.

## Long-running jobs

- Print a startup estimate when feasible.
- Emit a flushed heartbeat containing completed/total work, elapsed time, rate, and estimated remaining time.
- Write continuous output to `results/<analysis>/run.log`.
- Provide a simple monitoring command such as:

```bash
tail -f benchmarks/<benchmark_name>/results/<analysis>/run.log
```

- Use available GPUs for independent inference work when supported.
- Run a one-sample or small-cohort end-to-end validation before launching a large job.
- Do not continuously poll a background job after giving the user a monitoring command.

## Git and storage hygiene

- Never commit raw datasets, model checkpoints, embeddings, H5 files, large NumPy caches, archives, or generated package environments.
- Before staging, inspect `git status` and large untracked files.
- Keep compact result tables, scripts, notebooks, README files, provenance, and final figures versioned when reasonable.
- Avoid duplicating the same data in multiple benchmark directories.
- Do not delete source data or prior results unless the user explicitly requests it and the target has been verified.

## Completion checklist

Before calling a benchmark task complete, confirm:

- The pipeline runs successfully.
- Required assertions and QA checks pass.
- Outputs are saved under the correct benchmark directory.
- The primary notebook loads those outputs and executes without error.
- The notebook explains cohort, preprocessing, results, and limitations.
- No large unintended files are staged for Git.
