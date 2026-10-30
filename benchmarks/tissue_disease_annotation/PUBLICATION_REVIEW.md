# Benchmark review

**Recommendation: Supplementary material**, with a concise main-text reference.

This is a rigorous and useful representation-quality benchmark, but it does not establish a Bridge advantage over simpler transcriptomic representations. Its strongest role is demonstrating that a compact, frozen Bridge representation retains substantial biological identity.

## Scientific question

- How linearly accessible are tissue and cancer identities from frozen Bridge sample representations?
- Bridge was evaluated as a frozen 512-dimensional representation using the same linear classifier, samples, and held-out folds as PCA and raw expression.

## Evaluated cohorts

- GTEx detailed tissue:

  - 8,883 profiles
  - 549 donors
  - 51 tissue classes
  - Donor-grouped five-fold evaluation

- TCGA pan-cancer:

  - 9,942 unique patients
  - 33 cancer types
  - One specimen per patient
  - Five-fold stratified evaluation

- Supplementary GTEx broad tissue:

  - Same 8,883 profiles and donor-grouped folds
  - 30 broad tissue classes

The frozen cohorts are recorded in [gtex_manifest.csv](results/gtex_manifest.csv) and [tcga_manifest.csv](results/tcga_manifest.csv).

## Principal results

| Task | Bridge macro F1 | PCA macro F1 | Raw macro F1 |
|---|---:|---:|---:|
| GTEx detailed, 51 classes | 0.877 ± 0.023 | 0.925 ± 0.028 | 0.924 ± 0.022 |
| TCGA cancer, 33 classes | 0.910 ± 0.005 | 0.938 ± 0.009 | 0.942 ± 0.010 |
| GTEx broad, 30 classes | 0.926 ± 0.013 | 0.962 ± 0.024 | 0.960 ± 0.024 |

Positive findings:

- Frozen Bridge embeddings make tissue and cancer identity highly linearly accessible.
- The 512-dimensional Bridge representation preserves:

  - 94.9% of PCA macro-F1 performance for detailed GTEx annotation.
  - 97.0% of PCA macro-F1 performance for TCGA annotation.

- Bridge compresses the 15,165-gene transcriptome by approximately 29.6-fold.
- Performance was strong across both normal-tissue and pan-cancer cohorts without model fine-tuning or label-informed representation construction.

Negative findings:

- Bridge did not outperform PCA or raw expression.
- Bridge was below both baselines in every paired fold.
- Mean Bridge-minus-PCA and Bridge-minus-raw macro-F1 differences were:

  - GTEx detailed: −0.0476 and −0.0468
  - TCGA: −0.0278 and −0.0325
  - GTEx broad: −0.0359 and −0.0331

The exact summaries are in [metrics_summary.csv](results/metrics_summary.csv) and [paired_macro_f1_differences.csv](results/paired_macro_f1_differences.csv).

## Class-level findings

- Rare GTEx classes have unstable estimates:

  - Cervix – Ectocervix: six samples, Bridge F1 = 0
  - Cervix – Endocervix: five samples
  - Fallopian Tube: seven samples

  These outcomes should not be interpreted as absence of biological information.

- Better-powered GTEx errors were predominantly between biologically adjacent tissues, including:

  - Gastroesophageal junction → esophagus muscularis
  - Transverse colon → sigmoid colon
  - Non-sun-exposed → sun-exposed skin
  - Related brain regions

- The strongest TCGA confusion was READ → COAD:

  - 106/166 READ patients, or 63.9%
  - Reverse COAD → READ confusion was 46/458, or 10.0%

- ESCA → STAD occurred in 41/184 patients, or 22.3%.
- CHOL estimates are uncertain because it contains only 36 patients.

Complete class-level results are available in [per_class_summary.csv](results/per_class_summary.csv).

## Verification and QC

The saved predictions were independently checked against the metric tables:

- Recomputed accuracy, macro F1, and weighted F1 match the reported fold metrics to numerical precision.
- Held-out samples are identical across Bridge, PCA, and raw expression.
- GTEx donors never cross folds.
- GTEx biospecimen identifiers are unique after repeated-material removal.
- TCGA patient and sample identifiers are unique.
- PCA and all fitted scalers are fold-local.
- Bridge remained frozen.
- No annotation labels entered representation construction.
- 15,105 of 15,165 canonical genes were observed; the 60 unavailable genes were zero-filled before natural `log1p(TPM)`.
- All 45 logistic-regression fits converged.

The saved assertions are in [final_qc.json](results/final_qc.json), with provenance in [provenance.json](results/provenance.json).

## Limitations and unresolved evidence

- ARCHS4 pretraining may have contained public GTEx or TCGA profiles. This demonstrates label-free biological separability, not guaranteed sample-unseen generalization.
- Fold variability comes from one fixed five-fold partition, not independent model-training replicates.
- Rare classes make macro F1 sensitive to very small numbers of predictions.
- Fold-level PCA cumulative explained variance was not saved. It should be added during publication packaging using cached matrices, without rerunning Bridge inference.
- Provenance for the reused TCGA expression and embedding caches should be strengthened with explicit cache hashes.
- The executed [tissue_disease_annotation.ipynb](tissue_disease_annotation.ipynb) is valid but minimal; publication packaging should expand its narrative rather than introduce new analyses.

## Proposed publication scope

- Place the benchmark in supplementary material.
- Mention it briefly in the main text as evidence that frozen Bridge embeddings retain tissue and cancer identity under a linear readout.
- Include the existing [main_annotation_macro_f1.png](results/figures/main_annotation_macro_f1.png) as the primary supplementary figure.
- Include fold metrics, class counts, per-class F1, and the normalized confusion matrices as supporting material.
- Preserve the conclusion that PCA and raw expression remain more linearly separable.
- Avoid mechanistic interpretations of individual confusions and exclude GTEx-versus-TCGA normal/tumor claims.

No benchmark computations were rerun and no scientific interpretation was changed when this review was saved.

Explicit approval of this supplementary-material scope is required before publication packaging. Silence is not approval.
