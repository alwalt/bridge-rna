# Consolidated evidence from the Hallmark-state retrieval agent

This directory is a read-only copy of the compact, scientifically necessary evidence from:

`/home/walt/bridge-rna.worktrees/we-are-starting-a-new-self-contained-benchmark/benchmarks/hallmark_state_retrieval/`

The source worktree was at Git revision `cbd48b7d2bd0eb87bc71225de6f58f6b74924ea6`. The frozen GTEx Hallmark head has SHA-256 `f3f6c2ddc75d872b1e78af189f9286ff2be64ee43e90047916fca5ffe0538931`, identical to the head used by this benchmark.

## Included

- `frozen/`: head weights, Hallmark ordering, canonical genes, Bridge configuration, preprocessing contract, and frozen-system manifests.
- `pipeline/`: the complete source pipeline from the agent benchmark, excluding generated `__pycache__` files.
- `results/tcga_pan_cancer_positive_control/`: independent conventional-expression versus Bridge-Hallmark validation.
- `results/tcga33_retrieval/`: compact 33-cancer individual and centroid retrieval summaries.
- `results/signal_matching_audit/`: whole-vector and signed-program matching comparisons and controls.
- `results/luad_focused_audit/`: focused LUAD neighborhood and TCGA/GEO leakage audit evidence.
- `results/osdr_atlas/`: compact OSDR cohort, reproducibility, differential-retrieval, overlap, robustness, and hub evidence.
- `results/archs4_multispecies/` and `results/osdr_multispecies/`: combined human/mouse atlas provenance and compact species-composition results.
- `reports/`: the source benchmark README, performance summary, final report, and multispecies final report.

`COPY_MANIFEST.csv` records every copied file, its size, SHA-256, and original source path.

## Deliberately excluded

Large deterministic caches and raw retrieval tables were not duplicated: human/mouse Hallmark matrices, combined manifests, 30–91 MB top-100 tables, full embedding stores, and source expression matrices. Their provenance remains recorded in the copied reports and JSON manifests. This consolidation is sufficient to reproduce the manuscript summaries and notebook figures without pretending to reproduce full inference from compact files alone.

The local `results/frozen` compatibility link resolves the source pipeline's original frozen-artifact path to this bundle's `frozen/` directory. The copied full invariant script still requires the intentionally excluded ARCHS4 metadata, large arrays, and retrieval caches, so it is preserved for audit but cannot run end to end from this compact bundle alone. The current benchmark's own invariant suite remains the executable QA authority.

## Interpretation boundary

The copied evidence supports independent pathway validation and complementary Hallmark-state retrieval. It does not show that Hallmark retrieval is generally superior to Bridge-512. Individual-patient disease, tissue, and malignancy coherence generally favored Bridge, while cancer-centroid Hallmark retrieval was approximately competitive and better preserved named pathway signals. Combined human/mouse retrieval was overwhelmingly species-conditioned.
