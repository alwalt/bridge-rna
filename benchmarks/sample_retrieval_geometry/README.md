# Sample retrieval geometry

This benchmark asks whether dense-neighborhood, hubness, and local-geometry
corrections improve retrieval in frozen 512-dimensional Bridge sample
embeddings. It is independent of `paired_recount3`; that benchmark and all
shared assets are read-only inputs.

Only Stages 0 and 1 are in scope. No Bridge parameter, adapter, autoencoder, or
other representation is trained.

## Run

From the repository root:

```bash
/home/walt/bridge-rna/.venv/bin/python \
  benchmarks/sample_retrieval_geometry/pipeline/run_geometry_benchmark.py
```

Use `--force` only to rebuild the ignored ~1.15 GB query-by-reference cosine
cache. A normal rerun reuses it. Monitor with:

```bash
tail -f benchmarks/sample_retrieval_geometry/results/run.log
```

The configuration records every resolved shared input. Compact tables,
provenance, and figures are in `results/`; regenerable arrays are in ignored
`work/`. The executed notebook is the collaborator-facing report, and
`REPORT.md` is the concise final interpretation.

Regenerate presentation plots from saved tables without rerunning retrieval:

```bash
/home/walt/bridge-rna/.venv/bin/python \
  benchmarks/sample_retrieval_geometry/pipeline/plot_results.py
```

Run the composition-controlled biological validation after Stage 0/1 has
created the ignored query/corpus score cache:

```bash
/home/walt/bridge-rna/.venv/bin/python \
  benchmarks/sample_retrieval_geometry/pipeline/run_biological_validation.py
```

This Stage 1A–1H extension excludes the exact paired GSM from biological
retrieval and additionally excludes the query GSE for cross-study retrieval.
Its compact tables and figures are under `results/biological_validation/`.

Run the metadata-only upstream discovery audit and study-balancing analysis:

```bash
/home/walt/bridge-rna/.venv/bin/python \
  benchmarks/sample_retrieval_geometry/pipeline/run_recount3_audit.py
```

This reads the preserved `paired_recount3` selection tables and recount3
metadata restart cache, inventories predefined Bridge evaluation cohorts, and
reuses saved exact ranks for study-macro and deterministic cap-5/cap-10
analyses. It does not access expression services, download data, or generate
embeddings. Outputs are under `results/recount3_audit/`.

Run the cached absolute-similarity versus retrieval-uniqueness extension:

```bash
/home/walt/bridge-rna/.venv/bin/python \
  benchmarks/sample_retrieval_geometry/pipeline/run_absolute_similarity_extension.py
```

This reuses the saved 562-query ranks, query/corpus cosine memmap, and
two-million-pair ARCHS4 random-cosine sample. It adds rank-binned paired cosine,
expanded R@K and study-macro metrics, near-exact neighborhood counts, native
method-score summaries, and figures under `results/absolute_similarity/`.

Run the frozen-space anisotropy/common-component diagnostic:

```bash
/home/walt/bridge-rna/.venv/bin/python \
  benchmarks/sample_retrieval_geometry/pipeline/run_anisotropy_analysis.py
```

This fits the global mean and PCA basis on all 510,709 ARCHS4 references, uses
the same deterministic two-million pair indices for every geometry, applies
the reference-derived transforms unchanged to recount3 queries, and performs
chunked full-corpus retrieval. Compact results and eight figure families are
under `results/anisotropy/`; regenerated pair indices, transformed random
cosines, and reference transform parameters are retained only in ignored
`work/`.

Run the frozen literature post-processing comparison:

```bash
/home/walt/bridge-rna/.venv/bin/python \
  benchmarks/sample_retrieval_geometry/pipeline/run_literature_postprocessing.py
```

This reuses the exact two-million random-pair indices and established rankings,
fits a full 512-dimensional BERT-style whitening transform on ARCHS4 references
only, and compares raw cosine, centered cosine, frozen ABTT D=10, whitening,
NICDM, local scaling, and supplemental CSLS. Compact results and eight figure
families are under `results/literature_postprocessing/`; the full transform and
sampled whitening cosines remain under ignored `work/`.

Run the independent frozen GTEx biological-retrieval benchmark:

```bash
/home/walt/bridge-rna/.venv/bin/python \
  benchmarks/sample_retrieval_geometry/pipeline/run_gtex_biological_retrieval.py
/home/walt/bridge-rna/.venv/bin/python \
  benchmarks/sample_retrieval_geometry/pipeline/summarize_gtex_biological_retrieval.py
```

This reuses 6,314 existing GTEx v11 Bridge embeddings and authoritative tissue
labels, excludes explicitly identified GTEx-origin ARCHS4 references, and
evaluates the frozen raw, ABTT D=10, whitening, NICDM k=10, local-scaling k=10,
and supplemental CSLS configurations. It does not refit or tune any transform
on GTEx. Compact outputs and eleven figure families are under
`results/gtex_biological_retrieval/`; Top-100 neighbor caches remain ignored.

Run the retrospective representation × metric controls:

```bash
/home/walt/bridge-rna/.venv/bin/python \
  benchmarks/sample_retrieval_geometry/pipeline/prepare_representation_metric_controls.py
/home/walt/bridge-rna/.venv/bin/python \
  benchmarks/sample_retrieval_geometry/pipeline/run_representation_metric_controls.py
```

The preparation step reconstructs the full 510,709 × 15,165 ARCHS4
`log1p(TPM)` matrix from existing local counts, verifies paired rows against the
read-only upstream benchmark, and fits the fixed reference-only PCA-512. Large
expression, PCA-score, and transformed-Bridge caches remain under ignored
`work/representation_metric_controls/`. Compact retrieval, study-macro,
separation, runtime/storage, PCA, provenance, and figure outputs are written to
`results/representation_metric_controls/`. The completed GTEx analyses are not
rerun.

Run the frozen GTEx representation controls after preparing those caches:

```bash
/home/walt/bridge-rna/.venv/bin/python \
  benchmarks/sample_retrieval_geometry/pipeline/run_gtex_representation_controls.py
```

This reuses the exact 6,064-query, 13-harmonized-tissue, 510,376-reference
GTEx benchmark definition and the established Bridge Top-100 rankings. It
evaluates only the recount3-frozen expression Canberra/Pearson, PCA-512
Canberra, and PCA-whitened cosine/Canberra controls. Large Top-100 caches remain
under ignored `work/`; compact metrics, different-donor controls,
confusion/overlap diagnostics, runtimes, provenance, and figures are under
`results/gtex_representation_controls/`.

Run the contextual metadata audit of the frozen GTEx Top-100 results:

```bash
/home/walt/bridge-rna/.venv/bin/python \
  benchmarks/sample_retrieval_geometry/pipeline/run_contextual_metadata_analysis.py
```

This deduplicates the union of retrieved GSMs and classifies each once using a
prespecified, explicit-evidence rule set over `source_name_ch1`, sample title,
and `characteristics_ch1`. It reuses all six saved rankings and does not read
embeddings or expression. Missing and conflicting evidence remains unknown;
compact audit tables and examples are written under
`results/contextual_metadata/`.

## Methods

- cosine baseline, with published upstream exact-pair scores/ranks retained;
- CSLS, primary `k=10`, with prespecified `k=5,10,50` sensitivity;
- Zelnik-Manor–Perona local scaling on cosine distance;
- NICDM contextual distance normalization;
- reciprocal cross-domain kNN as set retrieval;
- cross-domain SNN Jaccard overlap as set retrieval;
- a two-million-pair global empirical cosine null with add-one P-values and
  Bonferroni correction across 510,709 references.

The exact equations, provenance, limitations, and results are in `REPORT.md`.
