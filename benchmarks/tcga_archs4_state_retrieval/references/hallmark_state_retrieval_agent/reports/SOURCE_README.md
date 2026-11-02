# Bridge Hallmark-state retrieval

**Status: COMPLETE — benchmark #9 is closed.** No inference or training is required for the final validation layer.

This benchmark applies the frozen GTEx-only 512-D → 50-Hallmark ridge probe to the compatible human ARCHS4 Bridge embedding cache, then compares exact cosine retrieval in Hallmark and raw Bridge space.

## Reproduce

```bash
cd /home/walt/bridge-rna.worktrees/we-are-starting-a-new-self-contained-benchmark/benchmarks/hallmark_state_retrieval
/home/walt/bridge-rna/.venv/bin/python pipeline/audit_and_build.py
/home/walt/bridge-rna/.venv/bin/python pipeline/run_queries.py
/home/walt/bridge-rna/.venv/bin/python pipeline/make_report.py
```

For a new sample, construct natural `log1p(TPM)` in the exact bundled 15,165-gene order, obtain the unnormalized final-layer arithmetic-mean Bridge embedding with checkpoint `r7hnr92k`, then apply `((z - mean) / scale) @ weight.T + bias` from `results/frozen/frozen_gtex_hallmark_probe.npz`. L2-normalize the resulting 50-vector and multiply it by `work/archs4_hallmark_cosine_index.float32.npy.T`.

```bash
/home/walt/bridge-rna/.venv/bin/python pipeline/query_new_samples.py \
  --expression /absolute/path/new_samples_log1p_tpm.npy \
  --sample-ids /absolute/path/sample_ids.txt \
  --output results/new_query/retrieval.csv --device cuda:0 --top-k 10
```

The expression array must be finite, nonnegative, natural `log1p(TPM)`, and ordered exactly as `results/frozen/frozen_canonical_genes.csv`.

## Similarity-scaling audit

The frozen-output-only follow-up is reproduced with:

```bash
/home/walt/bridge-rna/.venv/bin/python pipeline/audit_similarity_scaling.py
```

Its comparison table, provenance, Hallmark dominance audit, liver-versus-adipose contribution table, and PNG/PDF figure are under `results/similarity_scaling_audit/`. Raw, GTEx-centered, and GTEx-z-scored cosine were identical because the GTEx targets were already standardized. Pearson remained substantially below Bridge-512, so no contextual-program database was built.

## All-33 TCGA evaluation

The pan-cancer extension uses the existing 9,816-sample solid-tumor cache plus the compatible frozen 126-patient LAML parity cache. It applies the already-frozen linear head to those 126 embeddings, but performs no Bridge inference, fitting, or tuning. Run:

```bash
/home/walt/bridge-rna/.venv/bin/python pipeline/run_tcga33_retrieval.py
```

This performs exact top-100 retrieval for all 9,942 individual-patient queries and 33 cancer-type centroids, then derives P@10/P@50/P@100 without repeating retrieval. Results are in `results/tcga33_retrieval/`. `confirmed_precision` counts only explicit metadata matches and leaves ambiguous metadata unconfirmed; `conditional_precision` is computed among evaluable annotations and is always reported with `annotation_coverage`.

## Dominant-signal matching audit

To compare whole-vector cosine with signed rank, top-5/top-10 overlap, query-weighted, and asymmetric-containment retrieval using frozen outputs only:

```bash
/home/walt/bridge-rna/.venv/bin/python pipeline/audit_signal_matching.py
/home/walt/bridge-rna/.venv/bin/python pipeline/update_signal_notebook.py
```

The audit evaluates all 33 cancer centroids at top 5/10/50, measures signed top-5/top-10 signal coverage, computes score-background quantiles against all 510,709 ARCHS4 samples, and saves hub/study-diversity analyses under `results/signal_matching_audit/`.

For the focused frozen-output LUAD representative-patient and all-patient-centroid audit:

```bash
/home/walt/bridge-rna/.venv/bin/python pipeline/audit_luad_retrieval.py
```

Exact top-five metadata and shared Hallmark contributions are saved under
`results/luad_focused_audit/`. LUAD molecular/clinical subgroup retrieval is
not run because the authoritative local manifest lacks suitable subgroup labels.

Final LUAD provenance, cross-space quantification, scatter, and the final TCGA geometry figure are reproduced with:

```bash
/home/walt/bridge-rna/.venv/bin/python pipeline/audit_luad_geo_leakage.py
/home/walt/bridge-rna/.venv/bin/python pipeline/finalize_luad_and_tcga_visuals.py
/home/walt/bridge-rna/.venv/bin/python pipeline/test_invariants.py
```

The final publication-facing figure list is in `results/FINAL_FIGURE_INVENTORY.md`.
The conclusive scientific synthesis is in `FINAL_REPORT.md`; the corrected
human+mouse analysis remains in `MULTISPECIES_FINAL_REPORT.md`. Contextual
pathway retrieval, improved cross-species alignment, subtype-specific cancer
retrieval, and drug-reversal search are future work.

## All-eligible OSDR reference-atlas analysis

```bash
/home/walt/bridge-rna/.venv/bin/python pipeline/audit_osdr_atlas_cohort.py
/home/walt/bridge-rna/.venv/bin/python pipeline/prepare_osdr_atlas_representations.py
/home/walt/bridge-rna/.venv/bin/python pipeline/run_osdr_atlas_retrieval.py
/home/walt/bridge-rna/.venv/bin/python pipeline/summarize_osdr_atlas.py
/home/walt/bridge-rna/.venv/bin/python pipeline/update_osdr_atlas_notebook.py
```

The frozen cohort contains 1,471 mouse bulk RNA-seq samples from 64 OSDR studies, 45 tissues, and 88 exact matched FLT/GC groups. Derived arrays are under `work/osdr_atlas/`; manifests, retrievals, differential-rank tests, controls, top-hit audits, figures, and provenance are under `results/osdr_atlas/`.


The database has 510,709 human ARCHS4 samples. Large database/index files are deliberately ignored by Git but reproducible from shared read-only assets.
# Multispecies ARCHS4 extension

The original `results/osdr_atlas/` analysis is preserved as **mouse OSDR → human ARCHS4 cross-species retrieval**. The complete compatible reference contains 510,709 human and 429,746 mouse samples. Mouse Hallmark values are zero-shot predictions from the unchanged human GTEx-trained head.

Reproduce the extension from frozen caches:

```bash
python3 pipeline/build_multispecies_archs4.py
python3 pipeline/run_osdr_multispecies_retrieval.py
python3 pipeline/make_osdr_multispecies_heatmaps.py
python3 pipeline/summarize_multispecies_benchmark.py
python3 pipeline/run_tcga_pan_cancer_positive_control.py
PYTHONNOUSERSITE=1 /home/walt/bridge-rna/.venv/bin/python pipeline/build_multispecies_manifold.py
```

The human and mouse indexes are searched independently and their exact scores are merged for the combined reference; this avoids duplicating a two-gigabyte combined Bridge index. Query-new-sample preprocessing remains natural `log1p(TPM)` in the frozen ordered 15,165 one-to-one ortholog vocabulary. Generate the Bridge-512 query with the frozen r7hnr92k checkpoint, apply `((z - mean) / scale) @ weight.T + bias` from `results/frozen/frozen_gtex_hallmark_probe.npz`, L2-normalize the query, search each species index, and merge results by descending score.
