# Landmark-gene sufficiency benchmark

This side analysis asks whether fixed reduced gene panels preserve unusually
large amounts of global transcriptomic information for the frozen 45.6M model.
It compares the official 978-gene L1000 landmark set with size-matched random
panels and builds a random-panel sufficiency curve.

Gene panels are fixed across samples and species. Every model position outside
the visible panel is set to the model mask token; metrics are calculated only
over masked genes with measured ground truth. Human discovery studies are
disjoint from both final evaluation cohorts. Mouse is never used for selection.

Run order will be:

```bash
.venv/bin/python benchmarks/landmark_gene_sufficiency/pipeline/fetch_l1000_landmarks.py
.venv/bin/python benchmarks/landmark_gene_sufficiency/pipeline/freeze_cohorts.py
.venv/bin/python benchmarks/landmark_gene_sufficiency/pipeline/prepare_expression.py
.venv/bin/python benchmarks/landmark_gene_sufficiency/pipeline/build_gene_panels.py
.venv/bin/python benchmarks/landmark_gene_sufficiency/pipeline/run_sufficiency.py --pilot
```

An exploratory, leakage-safe first-pass ranking can be generated from the
human discovery cohort only:

```bash
.venv/bin/python benchmarks/landmark_gene_sufficiency/pipeline/prepare_discovery_expression.py
.venv/bin/python benchmarks/landmark_gene_sufficiency/pipeline/rank_informative_genes.py
```

The pilot uses cross-fitted probe-gene folds. Random visible panels are scored
only on a disjoint probe fold, and a candidate gene receives credit when its
inclusion improves probe reconstruction. This is a screening rank—not the
final panel—and neither human nor mouse evaluation samples influence it.

The final cross-species attribution run uses two GPUs, caches every condition,
and emits a 60-second orchestrator heartbeat:

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python \
  benchmarks/landmark_gene_sufficiency/pipeline/run_informative_gene_benchmark.py \
  --mode final --species human mouse --devices cuda:0 cuda:1 --heartbeat-seconds 60
```

For the definitive paper ranking, use the study-balanced mode below. It draws
500 samples from 500 distinct unseen studies per species, runs 48 deterministic
panels in each of four leakage-safe probe folds, and writes separate
`definitive_*` artifacts without replacing the earlier ranking:

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python \
  benchmarks/landmark_gene_sufficiency/pipeline/run_informative_gene_benchmark.py \
  --mode definitive --species human mouse --devices cuda:0 cuda:1 --heartbeat-seconds 60
.venv/bin/python \
  benchmarks/landmark_gene_sufficiency/pipeline/run_gene_set_enrichment.py \
  --ranking-prefix definitive
```

Enrichment uses GO Biological Process and Reactome through g:Profiler with
`domain_scope=custom`. The script asserts that the supplied background is
exactly the 15,165 unique genes in the model vocabulary.

The frozen Shared-451 GTEx tissue-pattern analysis is reproducible with:

```bash
MPLBACKEND=Agg .venv/bin/python \
  benchmarks/landmark_gene_sufficiency/pipeline/analyze_gtex_tissue_patterns.py
```

The exact all-gene GTEx coexpression summaries use GPU-blocked matrix
multiplication without saving the full correlation matrix:

```bash
.venv/bin/python \
  benchmarks/landmark_gene_sufficiency/pipeline/analyze_gtex_coexpression.py \
  --device cuda:0 --block-size 1024
```

Freeze and validate the locked panels on independent TCGA human and OSDR mouse
data (the freeze command intentionally refuses to overwrite an existing lock):

```bash
.venv/bin/python benchmarks/landmark_gene_sufficiency/pipeline/freeze_validation_panels.py
.venv/bin/python benchmarks/landmark_gene_sufficiency/pipeline/prepare_external_validation.py
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python \
  benchmarks/landmark_gene_sufficiency/pipeline/run_external_panel_validation.py \
  --datasets tcga_human osdr_mouse --devices cuda:0 cuda:1 --heartbeat-seconds 60
.venv/bin/python benchmarks/landmark_gene_sufficiency/pipeline/run_gene_set_enrichment.py
```

The notebook is the publication-facing record. `results/` contains mappings,
frozen panels, cohort provenance, unrounded tables, and figures; `work/`
contains restartable expression and condition caches.
