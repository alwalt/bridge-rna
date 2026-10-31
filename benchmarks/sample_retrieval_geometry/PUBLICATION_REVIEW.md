# Publication review: sample retrieval geometry

**Review status:** Proposed publication scope pending explicit user approval  
**Benchmark reviewed:** `benchmarks/sample_retrieval_geometry`  
**Read-only upstream benchmark:** `/home/walt/bridge-rna/benchmarks/paired_recount3`  
**Source Git revision:** `7b2c757b3aeaaa067b33f7faa32218faefbe1454`  
**Working-tree qualification:** the benchmark directory is currently untracked at that revision, so the revision identifies the surrounding repository state rather than a committed benchmark snapshot.  
**Review date:** 2026-10-30

## Benchmark objective

This benchmark asks three related but distinct questions about frozen BridgeRNA sample embeddings:

1. **Representation agreement:** do independently processed ARCHS4 and recount3 versions of the same human RNA-seq sample map to similar Bridge representations?
2. **Retrieval uniqueness and geometry:** does the exact counterpart rank highly among 510,709 human ARCHS4 samples, and how are rank, hubness, local density, anisotropy, and dominant components related?
3. **Biological retrieval:** do frozen Bridge representations retrieve samples with the same tissue, and do they add value beyond full expression or a matched 512-dimensional PCA representation?

The testable hypothesis evolved appropriately as controls accumulated. The initial hypothesis was that local-density correction would improve exact retrieval. The stronger final hypothesis was that reference-only post-processing could expose useful Bridge geometry without training, while preserving biological retrieval. The final conventional controls also test whether any apparent Bridge advantage is specific to the learned representation rather than ordinary expression geometry.

The intended scientific contribution is therefore not a universal “best metric” claim. It is an objective-specific characterization of frozen Bridge retrieval: high cross-pipeline agreement can coexist with poor uniqueness in crowded neighborhoods; anisotropy correction markedly changes ranking; and technical identity, tissue similarity, and storage efficiency favor different representations.

## Evidence inspected

The review inspected the executed [primary notebook](sample_retrieval_geometry.ipynb), [README](README.md), [full report](REPORT.md), [configuration](config.json), pipeline source under [pipeline/](pipeline/), result tables, figures, logs, and provenance under [results/](results/). In particular, the following evidence was traced directly:

- cohort discovery and exposure audit: [original_cohort_funnel.csv](results/recount3_audit/original_cohort_funnel.csv), [paired_exposure_inventory.csv](results/recount3_audit/paired_exposure_inventory.csv), [expansion_inventory.csv](results/recount3_audit/expansion_inventory.csv), and [recount3 audit provenance](results/recount3_audit/provenance.json);
- exact retrieval and composition: [ranked_method_comparison.csv](results/ranked_method_comparison.csv), [study_macro_retrieval_metrics.csv](results/recount3_audit/study_macro_retrieval_metrics.csv), [composition_summary.csv](results/biological_validation/composition_summary.csv), and per-query saved ranks;
- absolute similarity and crowding: [paired_cosine_micro_macro_summary.csv](results/absolute_similarity/paired_cosine_micro_macro_summary.csv), [random_vs_exact_cosine.csv](results/absolute_similarity/random_vs_exact_cosine.csv), [near_exact_neighbor_counts_by_rank.csv](results/absolute_similarity/near_exact_neighbor_counts_by_rank.csv), and [high_similarity_poor_rank_frequency.csv](results/absolute_similarity/high_similarity_poor_rank_frequency.csv);
- anisotropy and post-processing: [anisotropy_summary.csv](results/anisotropy/anisotropy_summary.csv), [pca_summary.csv](results/anisotropy/pca_summary.csv), [geometry_vs_local_corrections.csv](results/anisotropy/geometry_vs_local_corrections.csv), [main_comparison.csv](results/literature_postprocessing/main_comparison.csv), [alignment_uniformity.csv](results/literature_postprocessing/alignment_uniformity.csv), and [frozen_for_gtex.json](results/literature_postprocessing/frozen_for_gtex.json);
- GTEx biology: [GTEx cohort summary](results/gtex_biological_retrieval/gtex_cohort_summary.csv), [ARCHS4 annotation summary](results/gtex_biological_retrieval/archs4_annotation_summary.csv), [tissue harmonization](results/gtex_biological_retrieval/tissue_harmonization.csv), micro and tissue-macro results, different-donor controls, confusion matrices, and per-query results;
- expression/PCA controls: [asset audit](results/representation_metric_controls/asset_audit.json), [PCA provenance](results/representation_metric_controls/pca_provenance.csv), [micro metrics](results/representation_metric_controls/micro_metrics.csv), [study-macro metrics](results/representation_metric_controls/study_macro_metrics.csv), and [GTEx representation comparison](results/gtex_representation_controls/identity_vs_biological_retrieval.csv);
- exploratory context annotation: [annotation coverage](results/contextual_metadata/annotation_coverage.csv), [rules](results/contextual_metadata/annotation_rules.json), and [provenance](results/contextual_metadata/provenance.json).

The notebook contains 82 cells, including 39 code cells, and has no saved execution errors. Review calculations were bounded, read-only summaries of saved tables; no inference, fitting, retrieval, download, or pipeline execution was performed.

## Datasets and cohort

| Cohort | Role | Samples/queries | Biological units | Exposure and leakage control | Key qualification |
|---|---|---:|---|---|---|
| ARCHS4↔recount3 exact pairs | Technical cross-pipeline identity | 562 paired GSMs | 40 GSEs | All are repository-defined `unseen_study`; same GSM is the intended positive | These are technical reprocessings of the same biological material, not 562 independent biological replications |
| Human ARCHS4 reference corpus | Retrieval database and transform fitting | 510,709 samples | 26,262 GSE labels in master manifest | Frozen reference corpus; reference-only PCA/whitening fits | Study sizes are highly unequal and samples within a GSE are correlated |
| GTEx v11 | Tissue retrieval queries | 6,314 samples; 6,064 eligible | 937 donors; 13 eligible harmonized tissues | GTEx not used to fit Bridge transformations; 333 explicit GTEx-origin ARCHS4 records excluded | Multiple samples per donor mean 6,064 queries are not independent biological replicates |
| GTEx→GTEx control | Fully labeled tissue control | 6,064 queries | 937 donors, 16 authoritative tissue categories in control | Self and all same-donor references excluded | Query-level summaries still weight donors with more tissues/samples more heavily |
| ARCHS4 contextual metadata | Exploratory state/material audit | 24,938 unique retrieved GSMs | Heterogeneous GEO submissions | Annotation is text-only and independent of embeddings | State coverage 32.0%; material coverage 17.8%; not disease ground truth |

### recount3 discovery and composition

The executed discovery began with 510,709 human Bridge samples, selected 110,709 in the predefined unseen split, retained 84,896 unambiguous single-GSE candidates, found 575 recount3 metadata matches, and processed 574 pairs. The headline cohort contains 562 `unseen_study` pairs from 40 studies; 12 additional pairs are unseen samples from two seen studies and are correctly kept separate.

The discovery was broad but cannot be called strictly exhaustive. It read 8,199 of 8,803 contemporaneous human recount3 project metadata files; 604 files failed to download or read. No mapped-single candidate remained in the readable snapshot, but inaccessible project metadata prevents proof of exhaustiveness. The 257 additional unseen-study candidates in the audit are all `mapped_multiple`, outside the original single-GSE eligibility definition, and were not processed.

The 562-pair cohort is compositionally imbalanced. GSE234297/peripheral blood contributes 144 samples (25.6%), the five largest studies contribute 51.1%, and that single study contributes 122 of 187 dense queries. Effective study counts are 19.2 by Shannon entropy and 10.6 by inverse Simpson concentration. Study-macro results are therefore more defensible than sample-micro results for generalization across studies.

### Gene coverage and preprocessing

The benchmark verifies the canonical 15,165-gene ordering. Expression controls reconstruct ARCHS4 natural `log1p(TPM)` from counts using GENCODE v49 exon-union gene lengths, and the paired ARCHS4 rows reproduce the upstream matrix with maximum absolute difference zero. GTEx uses an existing 6,314 × 15,165 natural `log1p(TPM)` matrix with asserted sample and canonical gene order. PCA-512 is fit once on all 510,709 ARCHS4 references only; recount3 and GTEx queries are transformed after fitting. No Bridge parameter is trained or fine-tuned.

## Methods and evaluation actually run

### Frozen representations and retrieval methods

- Raw Bridge cosine in the original 512-dimensional sample-embedding space.
- Mean-centering and ABTT, with the final ABTT D=10 setting chosen from the prespecified recount3 settings D=0, 1, 2, 5, and 10.
- Full 512-dimensional ARCHS4-fitted covariance whitening with eigenvalue floor `lambda_max × 1e-12`; no dimensions required flooring.
- Retrieval-score corrections: NICDM k=10, local scaling k=10, and supplemental CSLS k=10.
- Conventional controls: full 15,165-gene expression with frozen metrics, ARCHS4-fitted PCA-512, and PCA-512 whitening.

All Bridge transforms are fit on the ARCHS4 reference corpus and then applied unchanged to recount3 and GTEx. The GTEx Bridge comparison was frozen before examining GTEx performance. The later expression/PCA GTEx extension was retrospective after GTEx had already been inspected, but its method list and transformations were frozen from the recount3 controls rather than selected using new GTEx scores.

### Metrics and aggregation

- Exact technical identity: R@1/5/10/100/1,000, MRR, exact rank, and study-macro counterparts.
- Representation agreement: exact-pair cosine and random-pair separation using a deterministic two-million-pair ARCHS4 sample.
- Geometry: mean direction, covariance spectrum, participation ratio, hubness, and local density.
- Biological retrieval: conservative tissue annotated precision, absolute confirmed hit rate, unknown rate, fold enrichment, per-tissue values, and tissue-macro averages.
- GTEx→GTEx: same-tissue precision after excluding self and every sample from the same donor.
- Context annotation: explicitly matched GEO text only, with conflicting/missing cases left unknown.

Competition rank is `1 + count(score > exact score)`, so exact ties share rank. Micro metrics weight samples equally; recount3 macro metrics calculate within each GSE and then weight 40 studies equally; GTEx tissue-macro metrics weight 13 eligible tissues equally. The study-macro audit includes deterministic bootstrap intervals over studies. GTEx micro annotated-precision tables include Wilson intervals over labeled hit positions, but those intervals do not model repeated hits within query, donor, or study.

## Validated results

### Headline results traced to saved evidence

| Result | Saved estimate | Effect or comparison | Uncertainty / replication | Verification | Supporting artifact |
|---|---:|---|---|---|---|
| Exact paired Bridge cosine | mean 0.997806; median 0.999218 | Exact pairs occupy the extreme tail of ordinary ARCHS4 similarity | 562 pairs, 40 studies; SD 0.003510; study-macro mean 0.996569 | Validated | [paired cosine summary](results/absolute_similarity/paired_cosine_micro_macro_summary.csv) |
| Raw Bridge exact retrieval | R@1 28.47%; R@10 77.22%; MRR 0.4764 | Median exact rank is 2 despite low R@1 | 562 pairs; study-macro R@10 75.92% | Validated against saved upstream ranks | [ranked methods](results/ranked_method_comparison.csv), [study macro](results/recount3_audit/study_macro_retrieval_metrics.csv) |
| Crowded-neighborhood interpretation | 33 queries have rank >100 with cosine >0.99 | 71.7% of rank >100 cases still exceed cosine 0.99 | Deterministic full-corpus ranks; no sampling for this count | Validated | [poor-rank frequency](results/absolute_similarity/high_similarity_poor_rank_frequency.csv) |
| Raw ARCHS4 anisotropy | random-pair median 0.912954; directional mean norm² 0.878774 | Mean-centering moves median random cosine to −0.002471 | 2,000,000 fixed random pairs; sampled distribution, not exhaustive | Validated | [cosine distributions](results/anisotropy/cosine_distribution_summaries.csv), [anisotropy summary](results/anisotropy/anisotropy_summary.csv) |
| Dominant centered components | PC1 57.66%; PCs 1–10 82.96% | Centered participation ratio 2.91 of 512 | One reference-only decomposition over 510,709 references | Validated | [PCA summary](results/anisotropy/pca_summary.csv) |
| Local corrections | NICDM R@10 82.56% versus raw 77.22% | +5.34 points micro; macro +3.04 points | 40-study macro NICDM R@10 78.96%, 95% bootstrap interval 68.94–88.18%; intervals overlap raw | Validated; improvement is modest, not decisive | [ranked methods](results/ranked_method_comparison.csv), [study macro](results/recount3_audit/study_macro_retrieval_metrics.csv) |
| ABTT D=10 exact retrieval | R@1 37.19%; R@10 85.41%; macro R@10 84.46% | +8.19 R@10 points over raw | Single frozen deterministic cohort; D selected on recount3 among prespecified settings | Validated, exploratory on recount3 | [geometry comparison](results/anisotropy/geometry_vs_local_corrections.csv) |
| Bridge whitening exact retrieval | R@1 46.62%; R@10 94.48%; macro R@10 94.12% | +17.26 R@10 points over raw; +9.07 over ABTT | Single deterministic cohort; reference-only transform; no paired method CI saved | Validated | [literature comparison](results/literature_postprocessing/main_comparison.csv) |
| Full-expression exact identity | Canberra R@1 78.29%; R@10 98.04%; macro R@10 95.72% | Exceeds Bridge whitening R@10 by 3.56 points | Same 562 queries and 510,709 references; one deterministic run | Validated | [expression/PCA metrics](results/representation_metric_controls/micro_metrics.csv), [study macro](results/representation_metric_controls/study_macro_metrics.csv) |
| Matched-dimensional identity | PCA-512 Canberra R@10 92.70%; Bridge-white cosine 94.48% | Bridge-white +1.78 points versus best reported PCA-512 R@10 | One seeded PCA fit; no PCA-fit replicate analysis | Validated, with solver-variation unmeasured | [expression/PCA metrics](results/representation_metric_controls/micro_metrics.csv) |
| GTEx→ARCHS4 Bridge tissue retrieval | raw macro P@10 86.73%; ABTT 90.60%; Bridge-white 91.21% | Bridge whitening +4.48 points versus raw | 6,064 queries, 13 tissue classes; approximately 74% Top-10 labels unknown | Validated conditional on conservative annotations | [GTEx macro metrics](results/gtex_biological_retrieval/retrieval_metrics_tissue_macro.csv) |
| GTEx→GTEx different-donor control | raw macro P@10 90.61%; ABTT 93.35%; Bridge-white 94.86% | Bridge whitening +4.25 points versus raw | 6,064 queries, 937 donors; same donor excluded from references | Validated; query-level donor clustering remains | [GTEx control](results/gtex_biological_retrieval/gtex_to_gtex_control.csv) |
| GTEx conventional controls | PCA-white cosine macro P@10 92.08%; Bridge-white 91.21% | PCA-white +0.87 point on GTEx→ARCHS4 | Same frozen cohort; retrospective method panel; no tissue/donor bootstrap CI | Validated | [identity-versus-biology table](results/gtex_representation_controls/identity_vs_biological_retrieval.csv) |
| Fully labeled GTEx conventional control | expression Pearson macro P@10 96.12%; PCA-white cosine 95.42%; Bridge-white 94.86% | Expression Pearson +1.26 points versus Bridge-white | Different-donor reference exclusion; no donor-cluster CI | Validated | [GTEx representation control](results/gtex_representation_controls/gtex_to_gtex_control.csv) |

### Positive findings

1. **Cross-pipeline representation agreement is strong.** Exact counterparts have median cosine 0.999218, far above random-pair median 0.912954. Rank failures often occur despite very high absolute similarity, directly supporting the distinction between representation agreement and retrieval uniqueness.
2. **The high raw cosine baseline is quantitatively explained by anisotropy.** The squared norm of the mean unit direction (0.878774) closely matches mean random-pair cosine (0.878748), and mean-centering moves the random-pair center near zero.
3. **Reference-only post-processing materially improves Bridge identity retrieval.** Whitening gives the strongest Bridge exact retrieval and substantially improves equal-study macro performance without model training.
4. **The improvement is not merely an R@1 artifact.** Whitening improves R@10, R@100, MRR, and study-macro metrics and reaches R@1,000 of 100% in this cohort.
5. **Bridge whitening improves Bridge tissue organization.** Both GTEx→ARCHS4 and the fully labeled different-donor GTEx control improve over raw Bridge, so the gain is not confined to exact GSM matching.
6. **The benchmark includes unusually strong conventional controls.** Full expression, matched PCA-512, PCA whitening, and multiple metrics prevent attributing all retrieval performance to Bridge.

### Negative and null findings

1. **Local density is real, but its association with failure is cohort-sensitive.** Removing GSE234297 reverses the dense-minus-sparse failure difference from +6.8 to −1.7 points, and adjusted models are imprecise. Density should not be described as a general causal determinant of failure from this cohort.
2. **NICDM and local scaling offer only modest technical-identity gains.** Their equal-study intervals overlap raw cosine, and they reduce the available source-label coherence in dense neighborhoods.
3. **CSLS underperforms after equal-study weighting.** Its favorable micro result does not generalize across studies.
4. **Bridge is not the strongest exact-identity representation.** Full `log1p(TPM)` with Canberra reaches R@10 98.04% versus 94.48% for Bridge whitening.
5. **Bridge does not beat matched PCA on the GTEx tissue task.** PCA-white cosine reaches tissue-macro P@10 92.08%, modestly above Bridge-white at 91.21%; expression Pearson leads the fully labeled GTEx control.
6. **Context-state evidence is too sparse for a disease/state claim.** State annotation is 68.0% unknown and material annotation 82.2% unknown. The text-derived analysis is exploratory only.

### Discrepant or superseded statements

- The earlier paragraph in [REPORT.md](REPORT.md) under “Direct attribution and biological context” says raw expression and expression-PCA were not evaluated on GTEx. That was true at that stage but is superseded by the later “Retrospective GTEx representation controls” section. Publication text must use the later results and remove the stale implication.
- “Independent GTEx validation” is accurate for the originally frozen Bridge transforms and tissue mapping, but not for the entire final method ensemble: the expression/PCA extension was retrospective after GTEx had been inspected. It was prespecified from recount3 controls, which reduces but does not eliminate the distinction from a never-inspected confirmatory cohort.
- The original recount3 search should not be called exhaustive without qualification because 604 project metadata files were unreadable. The correct classification is “comprehensive over readable metadata and documented eligibility; strict exhaustiveness cannot be determined.”

### Unverified claims

- The exact model checkpoint is named (`model/r7hnr92k/best_model.pt`) in GTEx provenance, but a checkpoint hash is not recorded in this benchmark. Frozen behavior is supported by cached embedding hashes and by the absence of training code, but checkpoint identity is not independently reconstructable from this benchmark alone.
- Latent or unmarked GTEx-origin samples may remain in ARCHS4; the exclusion covers 333 explicit metadata matches, not sequence-level duplicate detection.
- Biological state, disease, treatment, and cell type are not validated ground truth at corpus scale.
- No independent cross-species or perturbational benchmark is part of this review, so transfer claims beyond human sample/tissue retrieval remain untested here.

## Limitations and unresolved issues

1. **Cohort imbalance and non-independence.** The recount3 cohort has one dominant study/source. GTEx has 6,064 eligible samples but only 937 donors, and donors contribute multiple tissues. Point estimates based on samples or hit positions are more precise-looking than the biological replication supports.
2. **Sparse ARCHS4 tissue annotation.** Only 73,338 of 510,376 post-exclusion references (14.37%) receive any conservative tissue label; only 19,507 references (3.82% of the corpus) belong to the 13 eligible GTEx tissue labels. Top-10 unknown rates are approximately 73–76%. Annotated precision is conditional on the labeled subset and must always be paired with absolute hit and unknown rates.
3. **Tissue imbalance.** Blood is 65.2% of all labeled ARCHS4 tissue references, although blood is not an eligible GTEx query tissue in the cached cohort. Tissue-macro aggregation helps but does not repair annotation missingness or heterogeneity.
4. **Selection and multiplicity.** ABTT D=10 was selected using recount3 from six prespecified geometry settings. Many representations, metrics, and endpoints were examined. GTEx freezes the selected Bridge transforms, but recount3 comparisons are exploratory selection rather than untouched confirmation.
5. **Uncertainty is incomplete.** Study-bootstrap intervals are available for the balancing audit, and per-hit Wilson intervals are available for GTEx micro annotated precision. There are no paired study-bootstrap intervals for most representation comparisons, no donor-cluster bootstrap for GTEx, and no uncertainty for tissue-macro differences. Small differences such as PCA-white 92.08% versus Bridge-white 91.21% should be described as near parity, not a decisive win.
6. **PCA solver sensitivity is unmeasured.** PCA uses one deterministic randomized fit with 32 oversamples and one power iteration. The 512-dimensional conclusion is well controlled operationally, but variability across PCA fits was not assessed.
7. **Same-study technical structure.** Same-GSE retrieval can reflect both biology and study-specific processing. The benchmark appropriately includes different-donor GTEx and cross-study analyses, but exact source labels transfer poorly across studies and are not reliable biological ground truth.
8. **No batch-correction claim.** Improved retrieval after whitening does not establish batch correction or removal of all processing effects.
9. **Publication figure readiness.** The existing [identity-versus-biology figure](results/gtex_representation_controls/figures/identity_vs_biology.png) has overlapping labels near the PCA points. Several figures show point estimates without biological-unit uncertainty. They are analytically valid but should be redrawn for publication rather than copied unchanged.
10. **Versioning.** The entire benchmark is currently untracked in Git. Reproducibility requires committing the compact scripts, tables, provenance, notebook, and selected figures while excluding large `work/` caches.

## Publication recommendation

**Recommendation: include a focused version in the main manuscript, with the full geometry, audit, and diagnostic analyses in supplementary material.**

The benchmark merits main-manuscript placement because it establishes a scientifically important boundary on BridgeRNA retrieval claims rather than merely reporting a favorable score. It demonstrates strong cross-pipeline representation stability, diagnoses why raw cosine is misleading, shows a large reference-only whitening gain, and then subjects that gain to full-expression and matched-PCA controls. The negative result—that Bridge does not outperform conventional controls for broad human tissue retrieval—is scientifically valuable and prevents overstatement.

The main text should not present every method sweep. A compact integrated figure and table can support the following defensible claims:

1. Exact ARCHS4↔recount3 pairs have extremely high Bridge similarity even when exact rank is poor, demonstrating agreement without uniqueness.
2. Bridge sample embeddings are strongly anisotropic; ARCHS4-fitted whitening improves exact identity retrieval from R@10 77.22% to 94.48% and study-macro R@10 from 75.92% to 94.12%.
3. Full expression remains stronger for exact identity (Canberra R@10 98.04%), while Bridge whitening modestly exceeds matched PCA on the recount3 identity task.
4. On GTEx tissue retrieval, Bridge whitening improves raw Bridge but is approximately tied with or slightly below matched PCA and expression controls. Therefore Bridge provides a compact useful representation, not demonstrated superiority over conventional tissue geometry.
5. Density and local-normalization effects are cohort-dependent; they belong in the supplement rather than the headline claim.

### Prerequisites before packaging

1. Add paired study-level bootstrap intervals for the main recount3 representation contrasts and donor-cluster/tissue-level uncertainty for GTEx comparisons using existing saved per-query outputs.
2. Make the fully labeled GTEx→GTEx different-donor control the primary biological comparison; present GTEx→ARCHS4 annotated precision as secondary and always show Hit@K and Unknown@K.
3. Reconcile the superseded prose in [REPORT.md](REPORT.md) so the final narrative includes the expression/PCA GTEx controls throughout.
4. Redraw the main identity-versus-biology figure with non-overlapping labels and uncertainty; avoid treating sub-percentage tissue-macro differences as decisive.
5. Record the frozen checkpoint hash or explicitly state that only cached-embedding hashes are available.
6. Commit the benchmark and verify that large caches under `work/` remain excluded.

## Proposed publication scope and approval status

### Main manuscript package

**Cohorts**

- 562 strict unseen-study ARCHS4↔recount3 pairs across 40 studies against 510,709 human ARCHS4 references.
- 6,064 eligible GTEx queries across 13 harmonized tissues against 510,376 post-exclusion ARCHS4 references.
- 6,064-query different-donor GTEx→GTEx control across 937 donors.

**Methods shown**

- Raw Bridge cosine, ABTT D=10, full 512-dimensional Bridge whitening.
- Full-expression Canberra and Pearson.
- PCA-512 Canberra and PCA-white cosine.
- NICDM/local scaling only as compact supplementary comparators.

**Main claims**

- high cross-pipeline representation agreement is distinct from exact retrieval uniqueness;
- anisotropy explains the high ordinary cosine baseline;
- reference-only whitening improves frozen Bridge retrieval without training;
- full expression is strongest for exact identity;
- Bridge whitening is compact and useful but not superior to matched PCA for broad human tissue retrieval;
- no batch-correction, disease-state, cross-species, or universal biological-retrieval superiority claim.

**Proposed main table**

- Exact R@10 and study-macro R@10 on recount3, GTEx→ARCHS4 tissue-macro P@10 plus Hit@10/Unknown@10, and GTEx→GTEx donor-excluded tissue-macro P@10 for the frozen representation controls.

**Proposed main figure**

- Panel A: raw versus centered random-pair cosine distribution;
- Panel B: exact-pair cosine versus rank/crowding;
- Panel C: raw, ABTT, whitening, expression, and PCA exact-identity performance with study uncertainty;
- Panel D: GTEx biological retrieval with donor/tissue uncertainty and explicit annotation missingness;
- Panel E: dimensionality/storage versus the two retrieval objectives.

### Supplementary package

- recount3 discovery funnel, study composition, cap-5/cap-10 and leave-GSE234297-out analyses;
- full cosine distribution and sampled minima;
- hubness, local density, rank changes, and local-scaling/NICDM/CSLS comparisons;
- full PCA/whitening geometry, alignment/uniformity, per-study and per-tissue tables;
- GTEx confusion matrices, per-tissue performance, neighborhood overlap, and runtime/storage;
- exploratory contextual metadata annotations, clearly labeled non-authoritative;
- mapped-multiple candidate inventory as an unprocessed future cohort, not benchmark evidence.

### Excluded from publication claims

- the 257 unprocessed mapped-multiple recount3 candidates;
- the 12 seen-study pairs as a headline cohort;
- disease, treatment, cell-state, batch-correction, cross-species, or representation-learning claims;
- any assertion that the original recount3 search was strictly exhaustive;
- any claim that density generally causes failure or that Bridge is universally the best tissue retriever.

**Approval status: pending.** Saving this review does not approve publication packaging. The concrete scope above requires explicit user approval before creating a public notebook, modifying other benchmark artifacts, committing, or pushing.
