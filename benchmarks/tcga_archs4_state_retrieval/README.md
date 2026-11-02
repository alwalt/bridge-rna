# TCGA → ARCHS4 transcriptomic and functional-state retrieval

This self-contained benchmark compares exact retrieval of 510,709 human ARCHS4 samples from 9,816 TCGA primary-tumor queries in two frozen spaces:

1. the Bridge 512-dimensional mean-pooled sample representation; and
2. the frozen GTEx-trained 50-dimensional Bridge Hallmark readout.

A conventional expression-space control uses cancer centroids and the existing deterministic 40,000-sample ARCHS4 natural-log1p(TPM) cache in the canonical 15,165-gene order. It is deliberately scoped so that expression retrieval does not dominate runtime.

The frozen head is applied exactly as `((z - mean) / scale) @ weight.T + bias`. Its outputs are continuous Bridge Hallmark scores, not probabilities and not enrichment scores. No Bridge or Hallmark parameters are fit here.

## Run

```bash
bash benchmarks/tcga_archs4_state_retrieval/pipeline/run_all.sh
```

For a fast end-to-end software check:

```bash
/home/walt/miniconda3/envs/biofm/bin/python benchmarks/tcga_archs4_state_retrieval/pipeline/run_benchmark.py --smoke --skip-hash-embedding
```

The primary pipeline emits progress and ETA heartbeats to `results/run.log`. Large regenerable Hallmark matrices are stored under `work/`; ordered indexes and hashes make them auditable.

## Outputs

- `results/provenance.json` — immutable input paths, hashes, ordering hashes, and contracts.
- `results/qa.json` — dimensions, finiteness, uniqueness, and leakage checks.
- `results/top100_individual_*.parquet` — complete per-patient top-100 results and metadata.
- `results/top100_centroid_*.parquet` — cancer-centroid results.
- `results/individual_method_overlap.csv` and `results/centroid_method_overlap.csv` — direct neighborhood comparisons.
- `results/metadata_coherence.csv`, `results/retrieval_hubs.csv` — label coverage, study diversity, concentration, and hubs.
- `results/pathway_contributions.csv` — additive per-Hallmark cosine contributions for centroid matches.
- `results/representative_functional_matches.csv` — exploratory discordant examples with dominant-pathway robustness flags.
- `results/SCIENTIFIC_REPORT.md` — conclusions and limitations.
- `tcga_archs4_state_retrieval.ipynb` — reproducible human-readable report over frozen outputs.

Source datasets and earlier benchmark outputs are read-only dependencies. The benchmark does not modify them.

## Consolidated validation evidence

The benchmark now also contains a compact, hash-audited copy of the independent Hallmark-state validation and extension analyses from the sibling research worktree under `references/hallmark_state_retrieval_agent/`. The bundle includes the frozen head and contracts, source pipeline scripts, compact numerical results, provenance, and reports. Large expression matrices, embedding arrays, and full retrieval caches remain external and are identified in the copy manifest rather than duplicated.

Key additions are:

- an independent TCGA positive control comparing Bridge Hallmark profiles with expression-derived Hallmark profiles;
- a 33-cancer retrieval audit separating patient-level disease retrieval from centroid-level signal matching;
- OSDR replicate and functional-state retrieval analyses; and
- a combined human/mouse ARCHS4 analysis that exposes the species-domain boundary of the frozen human-trained representation.

See `references/hallmark_state_retrieval_agent/README.md` for scope and interpretation, and `references/hallmark_state_retrieval_agent/COPY_MANIFEST.csv` for source paths, sizes, and SHA-256 hashes. The consolidated publication decisions are recorded in `PUBLICATION_REVIEW.md` and `MANUSCRIPT_OUTLINE.md`.

Reusable frozen representations were promoted to the canonical, Git-ignored `/home/walt/bridge-rna/embeddings/` asset store before retirement of the producer worktrees. This includes row-aligned Bridge-512 and Hallmark-50 matrices for GTEx, TCGA, the separate TCGA-LAML extension, and OSDR, plus a 940,455-sample ARCHS4 Hallmark matrix reordered to exactly match the canonical combined human/mouse Bridge matrix and sample-location manifest. Dataset-level representation manifests record shapes, row contracts, and hashes; retrieval indexes and top-k analysis tables were not promoted because they are regenerable derivatives.

The publication notebook and its cached evidence are under `notebooks/tcga_archs4_state_retrieval/`. Regenerate it with `python benchmarks/tcga_archs4_state_retrieval/pipeline/make_notebook.py`; this reads saved evidence and does not launch the full research pipeline.
