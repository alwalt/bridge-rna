# Publication review: cross-species MTLE benchmark

**Review status:** pending user approval
**Reviewed benchmark:** `benchmarks/cross_species_mtle/`
**Worktree:** `/home/walt/bridge-rna.worktrees/bridge-cross-species-mtle`
**Branch:** `agents/bridge-cross-species-mtle`
**Repository revision:** `cbd48b7d2bd0eb87bc71225de6f58f6b74924ea6`
**Revision qualification:** the entire benchmark directory is currently untracked, so the revision identifies the surrounding repository state but does not version the reviewed benchmark contents.
**Review date:** 2026-11-02

## Executive assessment

This is a technically substantial, unusually transparent negative/qualified benchmark. It tests a valuable translational question under a strict study-unseen design, uses valid Bridge inputs, preserves biologically appropriate mouse phases, compares Bridge with raw expression, PCA, differential expression, coexpression, a frozen Hallmark head, and published biology, and retains null findings rather than optimizing them away.

The prespecified primary endpoint is not positive. Bridge disease-direction cosines are larger than raw-expression/PCA point estimates, especially acutely, but none exceeds its phenotype-label permutation null. Whole-gene DE and cross-fitted Integrated Gradients (IG) are weakly discordant in acute/intermediate phases. The separately trained frozen Hallmark head is also globally discordant and nonsignificant. Individual-gene and signed network-rewiring conservation are essentially absent.

The narrow positive result is at the pathway/program and functional-organization level. Exact inflammatory-response terms are supported by contextual rankings and acute IG, and Bridge network edges are strongly enriched for GO/KEGG/Hallmark co-membership even after removing matched coexpression top edges. In addition, the frozen human GTEx-trained Hallmark head recovers much of the **mouse** conventional-GSEA landscape zero-shot: across all 50 Hallmarks, mouse GSEA versus mouse Bridge-head Spearman correlation is 0.660 acute, 0.347 intermediate, and 0.567 chronic, with 78%, 92%, and 86% sign agreement. This is a useful transfer result, but it is not an independent assay—the head was trained against expression-derived Hallmark targets—and it does not overturn the negative human–mouse global Hallmark correspondence. These findings do not establish conserved mechanism, conserved individual genes, conserved rewiring, Bridge superiority, or model-fidelity ranking.

**Publication recommendation: supplementary material.** The benchmark is scientifically useful as a rigorous boundary-case and multiscale diagnostic, but its irreducible human procurement confounding, one-study-per-species primary design, nonsignificant primary endpoint, extensive post-primary analyses, and method-dependent pathway conclusions make it unsuitable as a principal efficacy figure. A concise main-text sentence may reference it as a study-unseen translational stress test; the detailed result belongs in the supplement.

## Benchmark objective

### Scientific question

Can a frozen BridgeRNA encoder, using the shared canonical human/mouse ortholog vocabulary and no fitted cross-species alignment, identify disease-associated transcriptional programs that correspond between human mesial temporal lobe epilepsy (MTLE) and phase-specific mouse kainate epilepsy?

### Testable primary hypothesis

The human disease direction

\[
\Delta_H = \operatorname{mean}(z_{MTLE})-\operatorname{mean}(z_{control})
\]

should align with independently estimated phase-specific mouse disease directions

\[
\Delta_{M,k} = \operatorname{mean}(z_{kainate,k})-\operatorname{mean}(z_{vehicle,k})
\]

more strongly than expected under valid within-study/phase phenotype-label permutations. The principal statistic is cosine similarity. Raw `log1p(TPM)` and PCA-50 use the same strict samples and common measured genes.

### Intended contribution

The benchmark is intended to test cross-species disease-program fidelity rather than species classification. Its strongest contribution is the simultaneous examination of sample geometry, conventional DE, contextual gene representations, attribution, functional-state readouts, pathways, and networks under the same frozen strict cohort.

## Datasets and cohort

### Strict primary cohort

| Species | Study | Tissue/model | Contrast | Biological N | Exposure | Bridge-ready input |
|---|---|---|---|---:|---|---|
| Human | GSE256068 / PRJNA1077986 | Whole hippocampal tissue; predominantly MTLE-HS | 58 MTLE vs 10 controls | 68 | All study-unseen | NCBI-deposited TPM |
| Mouse | GSE319769 / PRJNA1425071 | Intra-amygdala kainate, ipsilateral hippocampus | Day 3 acute: 6 vs 6 | 12 | All study-unseen | HTSeq counts converted with Ensembl-112 exon-union lengths |
| Mouse | GSE319769 / PRJNA1425071 | Same model | Day 7 intermediate: 6 vs 6 | 12 | All study-unseen | Same |
| Mouse | GSE319769 / PRJNA1425071 | Same model | Day 14 chronic: 6 vs 6 | 12 | All study-unseen | Same |

The saved primary manifest contains 104 unique GSMs and no duplicate GSM identifiers. Mouse phases are destructive, independent biological samples rather than repeated longitudinal measures. All mouse animals are male. Human sex is represented in both phenotypes (cases 36 male/22 female; controls 6 male/4 female).

Supporting evidence: [final primary manifest](./results/final_primary_manifest.csv), [sample manifest](./results/sample_manifest.csv), [sample-level provenance](./results/sample_level_provenance.json), and [Stage 0–2 audit](./STAGE_0_2_AUDIT.md).

### Gene coverage and preprocessing

- Canonical model vocabulary: 15,165 human/mouse one-to-one ortholog genes.
- Human NCBI TPM contains 15,141 vocabulary genes (99.842%); 24 absent genes are zero-filled through the established vocabulary alignment behavior.
- Mouse Ensembl-112 count-derived TPM contains all 15,165 vocabulary genes.
- TPM is computed over the complete valid source-gene denominator before Bridge subsetting.
- Inputs use natural `log1p(TPM)` with no post-subset renormalization.
- The mouse Ensembl-112 annotation is primary; GENCODE-v49 is retained as sensitivity. Sample-embedding robustness is high despite measurable TPM differences.

Supporting evidence: [gene coverage](./results/bridge_gene_coverage.csv), [input provenance](./results/stage3a_input_provenance.json), [human QC](./results/human_bridge_input_qc.csv), [mouse QC](./results/mouse_bridge_input_qc.csv), and [annotation sensitivity](./results/mouse_bridge_annotation_sensitivity.csv).

### Biological replication and confounding

The primary analysis has one human study and one mouse study. Individual subjects/animals are biological replicates within those studies; genes, pathways, permutations, and bootstrap draws are not independent biological cohorts.

The dominant limitation is irreducible human phenotype/procurement confounding: MTLE cases are surgical tissue, whereas controls are post-mortem neurotypical tissue. Controls are older and more mitochondrial-rich. Procurement, post-mortem status, age, and disease cannot be separated within this study. The cohort is almost entirely MTLE-HS (57 of 58 cases), so MTLE-non-HS and HS-specific fidelity are not estimable.

The post-primary expansion adds GSE186334 (human) and GSE99577 (mouse), but exposure category is confounded with study, model, tissue fraction, and acquisition. It does not supply a clean exposure-effect experiment or robust independent replication.

## Methods and evaluation actually run

### Frozen model and sample geometry

- Frozen checkpoint: `r7hnr92k/best_model.pt`, SHA256 `f3491bea...5541`.
- Frozen configuration SHA256: `1cccc2a0...b805`.
- Output: finite 512-dimensional sample embeddings.
- No species embedding and no Harmony, ComBat, CCA, Procrustes, domain adaptation, or fitted cross-species alignment.
- Biological-sample bootstrap: 5,000 replicates.
- Phenotype-label permutations: 10,000, within human study and within each mouse phase.
- PCA: fitted once on pooled 104 × 15,141 `log1p(TPM)` matrix after global per-gene centering; 50 components; 98.664% cumulative explained variance.

The implementation is traceable in [run_sample_level_benchmark.py](./pipeline/run_sample_level_benchmark.py) and [sample-level provenance](./results/sample_level_provenance.json).

### Contextual gene analysis

Condition-level contextual gene representations are case/control means of saved 512-dimensional token states at L1, L6, and L12. L0 is correctly treated as sample-invariant and has no disease shift. Shift support uses phenotype-label permutations; correspondence uses gene-identity permutations; BH correction is applied within layer/phase. Only 250 permutations were used, limiting null resolution and precision for 15,165-gene inference.

### Conventional DE and IG

- DE replication uses saved DESeq2 log2FC values and one-to-one ortholog matching. Gene-identity permutations and gene bootstrap quantify cross-species effect correspondence.
- IG uses a detached, cross-fitted disease axis, five human folds and six folds per mouse phase, an all-zero natural-log1p baseline, eight interpolation steps, 10,000 gene-identity permutations, and 5,000 biological-fold bootstraps.
- IG describes model reliance, not causal biology.

Supporting evidence: [DE summary](./results/exposure_extension/de_signature_cross_species.csv), [IG summary](./results/exposure_extension/ig_cross_species_summary.csv), [IG QC](./results/exposure_extension/ig_qc.csv), and [IG provenance](./results/exposure_extension/ig_analysis_provenance.json).

### Pathways, Hallmark head, and external-reference mapping

- Contextual and IG pathway tests use GO, KEGG, and Hallmark with a declared 15,165-gene universe and BH correction.
- The frozen GTEx-trained 512→50 Hallmark head reuses cached Bridge embeddings; mouse results are zero-shot predictions from a human-trained head.
- Kim et al. themes were frozen and mapped before Bridge lookup in the final validation layer.

Kim concordance is best described as **literature/method concordance**, not independent-cohort replication: the published study and this benchmark draw from the same broader source collection, and the strict studies are part of the publication's analysis universe. Independence applies to the frozen Bridge analyses and pre-specified mapping, not to biological samples.

### Networks

L12 networks use the validated per-sample mean-centered cosine union-kNN method (`k=10`), retaining an edge when present in at least 50% of biological samples in a state. Pearson and Spearman networks use the same genes, samples, and `k`. Rewiring is disease edge weight minus matched-control weight. The final null uses 500 expression-decile-stratified mouse gene-label permutations. GO/KEGG/Hallmark edge co-membership is functional annotation evidence, not physical-interaction evidence; STRING/BioGRID was not tested.

## Validated results

### Prespecified primary endpoint and matched baselines

| Representation | Mouse phase | Cosine | Biological-bootstrap 95% CI | Label-permutation p | Verification |
|---|---|---:|---:|---:|---|
| Bridge | Acute | 0.5326 | [0.3885, 0.6426] | 0.1888 | Validated from saved metrics/nulls |
| Bridge | Intermediate | 0.2753 | [0.1413, 0.4270] | 0.2448 | Validated |
| Bridge | Chronic | 0.2356 | [-0.0518, 0.5730] | 0.3063 | Validated |
| Raw `log1p(TPM)` | Acute | 0.0975 | [0.0144, 0.2147] | 0.2349 | Validated |
| PCA-50 | Acute | 0.0975 | [0.0145, 0.2159] | 0.2674 | Validated |

Bridge has the largest point estimate, especially acutely, but no representation is significant against its own valid label null. The bootstrap interval and permutation test answer different questions; a bootstrap interval above zero does not supersede the label-null result. Bridge superiority is not established.

Supporting artifacts: [direction metrics](./results/cross_species_direction_metrics.csv), [permutation draws](./results/cross_species_permutation_results.csv), [bootstrap draws](./results/cross_species_bootstrap_results.csv), and [primary null figure](./figures/observed_vs_null_disease_directions.png).

### Gene-level and attribution results

| Analysis | Acute | Intermediate | Chronic | Interpretation |
|---|---:|---:|---:|---|
| DE log2FC Pearson | -0.0930 | -0.0463 | 0.0032 | Acute/intermediate significantly discordant by gene-identity null; chronic null |
| IG Pearson | -0.0649 | -0.0766 | -0.0103 | Weak acute/intermediate signed discordance; chronic near zero |
| IG cosine | -0.0612 | -0.0738 | -0.0095 | Same qualitative result |
| L12 mean contextual cosine | 0.1681 | 0.0593 | 0.0417 | Positive means, but zero genes pass joint conserved/discordant criteria |

At L12, 15,146/15,165 acute genes, 15,137 intermediate genes, and 14,967 chronic genes remain unresolved. The data do not support individual-gene conservation claims. Strongly IG-discordant genes are enriched for opposite-direction DE, but continuous DE–IG correspondence is weak.

Supporting artifacts: [contextual summary](./results/contextual_gene_shift_summary.csv), [DE summary](./results/exposure_extension/de_signature_cross_species.csv), [IG summary](./results/exposure_extension/ig_cross_species_summary.csv), and [DE/IG association](./results/hallmark_head/ig_discordant_gene_de_association.csv).

### Program-level literature concordance

Fourteen Kim themes were frozen. Eight are testable with the saved Bridge outputs:

| Final category | Themes | Verified meaning |
|---|---:|---|
| Replicated biological program | 1 | Neuroinflammation supported by contextual analysis and IG with compatible direction |
| Partial replication | 1 | TNF signaling supported contextually but not independently by the exact IG mapping |
| Opposite result | 2 | Synaptic plasticity and neurotransmission are contextually divergent in this strict intra-amygdala comparison |
| No corrected evidence | 4 | Gliogenesis, neurogenesis, axonogenesis, neuronal death |
| Not testable | 6 | Includes HS-specific fidelity, model ranking, astrocytic fidelity, PPS remodeling, and marker-gene theme |

The exact inflammatory-response term is contextually conserved across all phases, with independent acute IG support. This is biological-program concordance, not exact-gene, sample-level, network-rewiring, or mechanistic replication.

The older broad “gliogenesis conservation” wording in [README.md](./README.md) and the first broad-theme section of [FINAL_REPORT.md](./reports/FINAL_REPORT.md) is superseded by the final frozen term-mapped analysis: exact gliogenesis terms are not supported. Publication packaging must use the corrected conclusion.

Supporting artifacts: [validation counts](./results/kim_external_validation/kim_validation_counts.csv), [independent-method table](./results/kim_external_validation/independent_bridge_replication.csv), [mapping](./results/kim_external_validation/kim_to_bridge_term_mapping.csv), and [validation provenance](./results/kim_external_validation/validation_provenance.json).

### Frozen Hallmark-head result

The independent 50-Hallmark functional-state head does not confirm global positive correspondence. Complete-vector Pearson/cosine values are negative in every phase and nonsignificant relative to phenotype-label nulls. Cosines are -0.484 acute, -0.420 intermediate, and -0.350 chronic; family q is 0.870 for all nine global tests. Chronic appears closer partly because the mouse effect norm contracts from 9.27 acute to 4.69 chronic. This is relative contraction toward a still-discordant state, not functional conservation.

The head leaves inflammatory response and TNF directionally compatible but unresolved, while interferon/IL-6 components are discordant. Therefore broad pathway conclusions are method-dependent.

Supporting artifacts: [global correspondence](./results/hallmark_head/strict_hallmark_correspondence.csv), [temporal metrics](./results/hallmark_head/direct_head_temporal_metrics.csv), [individual Hallmarks](./results/hallmark_head/strict_individual_hallmarks.csv), and [four-level landscape](./figures/hallmark_head/de_gsea_hallmark_head_ig_4x3_landscape.png).

### Zero-shot mouse Hallmark recovery and phase-resolved programs

The same frozen head provides a narrower positive result when evaluated as a **mouse functional-state readout**, rather than as a human–mouse conservation statistic. Across all 50 Hallmarks, the saved mouse disease-minus-control head deltas broadly track mouse conventional GSEA:

| Phase | Mouse GSEA vs mouse Bridge Pearson | Spearman | Same-sign fraction | Verification |
|---|---:|---:|---:|---|
| Acute | 0.5819 | 0.6596 | 0.78 | Validated from saved 50-term vectors |
| Intermediate | 0.3759 | 0.3466 | 0.92 | Validated |
| Chronic | 0.4066 | 0.5673 | 0.86 | Validated |

This shows that a head trained on human GTEx samples transfers meaningful pathway ordering and direction to the mouse kainate model without mouse retraining. It does **not** show that the head independently validates GSEA, because its training targets were ssGSEA-like Hallmark activities, and these summary correlations have no saved inferential null. It also does not show human–mouse disease-state conservation: the corresponding human-versus-mouse Bridge-head Spearman values remain negative (-0.485, -0.287, and -0.280).

The selected phase-resolved figure uses observed disease-minus-matched-control values for 22 prespecified epilepsy-relevant programs. Acute mouse injury is characterized by high unfolded-protein response, p53, interferon, apoptosis, TNF/NF-kB, IL-6/JAK/STAT3, and hypoxia scores. Several acute responses weaken by chronic disease. The Bridge head also resolves dynamic energy/redox and growth/injury states that are muted or unresolved by conventional GSEA: peroxisome, fatty-acid metabolism, and oxidative phosphorylation are negative acutely and positive at intermediate/chronic phases; mTORC1 peaks at the intermediate phase; TGF-beta attenuates by chronic disease. These are descriptive phase patterns from independent animals at destructive time points, not longitudinal trajectories or direct between-phase tests.

In the focused plot, filled Bridge markers denote biological-sample bootstrap 95% intervals excluding zero, whereas filled GSEA markers denote BH q<0.05. Those support definitions are not calibrated equivalents. Pathways were selected from epilepsy biology, but their within-block order uses observed Bridge phase swing; the figure is therefore a biologically focused secondary visualization, not an unbiased confirmatory ranking.

Supporting artifacts: [mouse recovery values](./results/hallmark_head/mouse_hallmark_recovery_values.csv), [mouse recovery metrics](./results/hallmark_head/mouse_hallmark_recovery_metrics.csv), [mouse recovery summary](./figures/hallmark_head/mouse_hallmark_recovery_summary.png), [focused pathway table](./results/hallmark_head/epilepsy_energy_three_phase_hallmark_states.csv), [focused pathway figure](./figures/hallmark_head/epilepsy_energy_three_phase_hallmark_states.png), and [focused-figure provenance](./results/hallmark_head/epilepsy_energy_three_phase_hallmark_states_provenance.json).

### Incremental-value result

Adding continuous IG features to an unsigned conventional-DE baseline improves held-out AUPRC by 0.0075–0.0137 across phases and 0.0128 pooled, while reducing AUROC by 0.0035–0.0073. The pooled opposite-IG-sign adjusted odds ratio is 1.49 (gene-cluster-robust 95% CI 1.37–1.62). Hard top-k IG indicators change AUROC/AUPRC by less than 0.001.

This is a small, metric-dependent prioritization gain, not broadly superior classification. Cross-validation and uncertainty operate over genes, not independent biological samples, and the outcome itself is derived from DE significance and direction.

Supporting artifacts: [model performance](./results/hallmark_head/incremental_value/model_performance.csv), [nested models](./results/hallmark_head/incremental_value/nested_model_tests.csv), [predictions](./results/hallmark_head/incremental_value/oof_predictions.csv.gz), and [provenance](./results/hallmark_head/incremental_value/provenance.json).

### Network result

Signed cross-species rewiring is essentially zero: Bridge cosines are -0.00093, 0.00128, and 0.00182, with only 20, 13, and 9 shared top-1,000 edges. Gain/loss agreement is not significant after correction. Module-level rewiring cosine is negative acute/intermediate and approximately zero chronic.

Bridge network structure is nevertheless functionally coherent. Bridge-specific top rewiring edges are enriched approximately 3.8–4.2× for GO, 4.6–5.0× for KEGG, and 14.1–14.6× for Hallmark co-membership after removal of coexpression top edges. This supports functional organization beyond matched coexpression, not conserved disease rewiring or physical interactions. Because ontology co-membership is an annotation-based validation and no PPI resource was tested, it should remain supplementary.

Supporting artifacts: [network metrics](./results/network_final/rewiring_conservation_metrics.csv), [method summary](./results/network_final/network_method_summary.csv), [functional validation](./results/network_final/edge_functional_validation.csv), [network provenance](./results/network_final/provenance.json), and [network QC](./results/network_final/qc_summary.json).

## Positive findings

1. **Technically valid strict benchmark:** Bridge-ready TPM was obtained without FASTQ reprocessing, with 99.84–100% vocabulary coverage and annotation-matched mouse lengths.
2. **Study-unseen primary cohort:** every strict human and mouse sample is study-unseen to Bridge pretraining.
3. **Larger Bridge point estimate:** the acute Bridge disease-direction cosine is materially larger than raw/PCA, although not significant against the label null.
4. **Narrow program-level concordance:** neuroinflammation is supported by two distinct frozen Bridge views, contextual embeddings and acute IG.
5. **Functionally coherent networks:** Bridge-specific network edges recover independent ontology/pathway co-membership beyond matched Pearson/Spearman top edges.
6. **Zero-shot mouse functional-state recovery:** the human GTEx-trained Hallmark head broadly tracks mouse GSEA across all 50 programs and resolves plausible phase-specific stress, metabolism, and growth/injury patterns without mouse retraining.
7. **Transparent negative evidence:** DE, IG, global human–mouse Hallmark-head, gene-level, and rewiring analyses collectively define where cross-species correspondence fails.

## Negative findings and scientific significance

1. **Primary endpoint null:** positive cosine alone does not demonstrate disease conservation.
2. **No demonstrated Bridge superiority:** raw/PCA are smaller, but all methods are nonsignificant under their valid nulls.
3. **Gene-level discordance:** DE and IG are weakly negative acute/intermediate; L12 has no jointly supported conserved genes.
4. **No conserved network rewiring:** tiny statistically detectable values against near-zero topology nulls have negligible effect size.
5. **Hallmark-head boundary:** the probe recovers the mouse pathway landscape but does not independently confirm a conserved human–mouse 50-program state.
6. **No clean exposure effect:** seen/unseen status is inseparable from study and biology.
7. **No robust phase/model ranking:** acute has the largest primary Bridge point estimate, chronic is least discordant in some secondary functional/network summaries, and Kim's intermediate-fidelity claim is not directly testable with this intra-amygdala cohort. No single “best model phase” is supported across endpoints.

These negatives are scientifically informative: Bridge may organize genes into coherent higher-order biological neighborhoods while human and mouse disease perturbations act through different gene-level and edge-level implementations. The data do not establish that this organization improves cross-species disease prediction.

## Discrepant and superseded claims

| Claim/location | Evidence status | Required publication treatment |
|---|---|---|
| “Inflammatory/gliogenic conservation” in README and early broad-theme summary | Partly superseded | Retain neuroinflammation; remove broad gliogenesis replication claim |
| Pathway conservation as rescue of primary endpoint | Unsupported | Explicitly state that pathway aggregation does not rescue the label-null primary result |
| Kim findings as independent external replication | Overstated if “independent” refers to data | Describe as pre-mapped literature concordance on overlapping source studies |
| Chronic Hallmark convergence | Mostly contraction | Report negative correlations and shrinking mouse norm together |
| Significant network cosine/Jaccard as substantial rewiring conservation | Unsupported by effect size | Emphasize near-zero effect and sparse overlap despite small null p-values |
| Exposure effect | Not identifiable | Report heterogeneous study-confounded sensitivity only |

## Limitations and unresolved issues

### Biological design

- Surgical cases versus post-mortem controls is perfectly confounded with phenotype.
- Human controls are only 10, older, and more mitochondrial-rich.
- The primary cohort contains one human and one mouse study; the mouse model cannot establish generality across epilepsy models.
- Mouse phases contain six animals per condition, producing broad biological-sample uncertainty.
- Human disease is almost entirely MTLE-HS; non-HS contrasts are unavailable.
- Human “whole” tissue and mouse hippocampus are broadly matched, but cellular composition and acquisition differ materially.

### Statistical and methodological

- Primary bootstrap intervals and label-permutation tests answer different questions and must be shown together.
- Contextual gene inference uses only 250 permutations for 15,165 genes.
- Pathway sets overlap; pathway-panel resampling is not biological-sample replication.
- The Hallmark head was trained against expression-derived ssGSEA-like targets; its agreement with conventional GSEA is transfer/recovery evidence, not independent biological validation.
- The focused 22-program Hallmark plot uses prespecified biological themes but orders pathways by observed Bridge phase swing. Bridge bootstrap support and GSEA FDR support answer different questions and must not be compared as equivalent significance calls.
- Many analyses are post-primary and should not be presented as confirmatory.
- Full-cohort whitening is transductive and sensitive to retained rank; PC removal is diagnostic only.
- Gene-level incremental-value CV measures generalization across genes, not subjects or studies.
- Network annotation enrichment is not PPI validation and may reflect stable functional organization rather than disease-specific information.

### Reproducibility and packaging

- [cross_species_mtle.ipynb](./cross_species_mtle.ipynb) is executed but contains only five cells and indexes the whitening extension. It does not present the primary endpoint or the later DE/IG, Hallmark, external-validation, incremental-value, or network evidence.
- [README.md](./README.md) does not fully document the final extensions and retains superseded broad gliogenesis wording.
- The benchmark directory is untracked at review time. The source revision therefore cannot reproduce the benchmark without a future curated commit.
- Several result files are large (for example the full contextual-gene table and 17 MB network state-edge table). Publication packaging should include compact validated tables and figures, not caches or raw embeddings.
- The historical Stage 0–2 audit correctly records the earlier mouse length caveat, but the final publication narrative must state that Stage 3 used annotation-matched Ensembl-112 lengths.

## Publication recommendation

### Placement: supplementary material

The benchmark should be included as a **supplementary translational stress test**, not as a principal performance claim. Its value is the rigor of the study-unseen design, matched baselines, multiscale negative evidence, and careful distinction between pathway organization and perturbational fidelity. Its primary endpoint is null, and the strongest positive findings are narrow and method-dependent.

A concise main-text sentence is reasonable, for example:

> In a fully study-unseen human–mouse MTLE benchmark, Bridge produced a larger acute disease-direction cosine than raw expression or PCA but did not exceed phenotype-label nulls; secondary analyses supported neuroinflammatory program concordance while gene-level attribution and disease-network rewiring remained weak or discordant.

The benchmark should not be used to claim that Bridge identifies a superior mouse model, demonstrates mechanistic conservation, corrects batch/species effects, or generally outperforms conventional expression analysis.

## Proposed publication scope

### Cohorts and protocol to package

- Primary only: study-unseen GSE256068 human MTLE/control and phase-matched GSE319769 mouse intra-amygdala kainate/control.
- Natural `log1p(TPM)`, canonical 15,165-gene order, no cross-species alignment.
- Bridge/raw/PCA disease-direction comparison with biological bootstrap and phenotype-label nulls.
- Conventional DE, cross-fitted IG, frozen Hallmark head, term-mapped Kim concordance, and final networks as explicitly secondary analyses.
- Exposure/full-cohort/whitening analyses should be relegated to a methods appendix or omitted from the first publication package unless space permits.

### Claims to package

1. The strict benchmark is feasible and study-unseen.
2. Bridge shows suggestive acute-weighted sample geometry but no significant primary correspondence or demonstrated superiority.
3. Neuroinflammation is the only Kim theme independently supported by both contextual and IG analyses; this is program-level literature concordance, not independent-cohort replication.
4. DE and IG are weakly discordant acute/intermediate; individual-gene conservation is unsupported.
5. The frozen Hallmark head does not confirm global program-state conservation.
6. The same head nevertheless recovers much of the mouse GSEA landscape zero-shot and identifies phase-resolved stress, metabolic, and growth/injury states; this is a secondary transfer result, not evidence that the human and mouse disease programs are globally aligned.
7. Bridge networks are functionally coherent beyond matched coexpression, but signed cross-species disease rewiring is negligible.

### Proposed supplementary figures

1. **Cohort and primary endpoint:** [cohort design](./figures/cohort_design.png), [observed-vs-null distributions](./figures/observed_vs_null_disease_directions.png), and [Bridge/raw/PCA comparison](./figures/bridge_raw_pca_comparison.png).
2. **Zero-shot mouse pathway recovery:** [mouse Hallmark recovery summary](./figures/hallmark_head/mouse_hallmark_recovery_summary.png), explicitly separating mouse functional recovery from human–mouse conservation.
3. **Phase-resolved pathway biology:** [epilepsy, energy, and growth/injury Hallmark states](./figures/hallmark_head/epilepsy_energy_three_phase_hallmark_states.png), with captions defining the two non-equivalent support markers and noting that phases are independent animals.
4. **Multiscale boundary:** [DE/GSEA/Hallmark-head/IG four-row landscape](./figures/hallmark_head/de_gsea_hallmark_head_ig_4x3_landscape.png), with a caption stating that numerical scales differ by row, or move this to an extended-data panel if space is limited.
5. **Pre-mapped literature concordance:** [Kim validation heatmap](./figures/kim_external_validation/kim_bridge_validation_heatmap.png).
6. **Network boundary result:** [rewiring conservation summary](./figures/network_final/rewiring_conservation_summary.png), accompanied by effect sizes rather than p-values alone.

### Proposed supplementary tables

1. Cohort, exposure, input provenance, and gene coverage.
2. Primary Bridge/raw/PCA metrics with bootstrap CIs and label-null p-values.
3. DE and IG cross-species correspondence with explicit null type.
4. Frozen Kim-theme mapping and validation categories.
5. Hallmark-head global correspondence and magnitude contraction.
6. All-50 mouse GSEA/head recovery metrics and the focused 22-program phase table.
7. Network effect sizes and functional-enrichment caveats.

### Exclusions from publication claims

- Broad gliogenesis replication.
- Exact gene-level conservation.
- Conserved disease-network rewiring.
- A causal pretraining-exposure effect.
- Whitening or PC removal as a validated improvement.
- A general acute/intermediate/chronic model-fidelity ranking.
- Physical interaction claims for contextual edges.

## Prerequisites before packaging

1. Replace the current sparse notebook with an executed, self-contained public notebook that loads compact frozen tables and presents the approved hierarchy.
2. Reconcile the README and report so the superseded gliogenesis statement cannot be mistaken for the final conclusion.
3. Add a compact, frozen claim/evidence table distinguishing primary, secondary, exploratory, and literature-concordance results.
4. Ensure every publication figure is generated from saved tables and has an explicit caption defining its null and uncertainty unit.
5. For the two selected Hallmark figures, state explicitly that mouse head predictions are zero-shot outputs of a human GTEx-trained head, that the head targets were expression-derived, and that mouse recovery does not imply human–mouse conservation.
6. Curate versionable outputs; exclude embeddings, raw matrices, contextual tensors, work caches, and unnecessarily large edge/gene tables.
7. Commit the approved benchmark/package so the publication revision is traceable.

No new inference or statistical computation is required for the proposed scope. Packaging can regenerate compact tables and figures cheaply from validated cached evidence.

## Evidence inspected

- [README](./README.md), [Stage 0–2 audit](./STAGE_0_2_AUDIT.md), [primary notebook](./cross_species_mtle.ipynb), and [final report](./reports/FINAL_REPORT.md).
- Primary manifests, exposure manifests, expression audit, gene coverage, input QC, checkpoint/config provenance, and annotation sensitivity.
- Primary sample-direction metrics, 5,000 bootstrap draws, 10,000 label-permutation draws, raw/PCA baselines, and plotting pipeline.
- Contextual gene summaries/tables, pathway tables, null definitions, and extraction provenance.
- DE, cross-fitted IG, exposure sensitivity, full-cohort/whitening, and associated provenance/QC.
- Frozen Hallmark-head compatibility, sample predictions, global/individual statistics, temporal magnitude analysis, and conventional GSEA comparison.
- All-50 mouse Hallmark recovery values/metrics, the focused 22-program phase table, both selected Hallmark figures, their provenance, and their plotting definitions.
- Pre-mapped Kim reference/mapping/validation tables.
- Incremental-value folds, predictions, model metrics, sensitivities, and provenance.
- Final network edge/module summaries, nulls, functional validation, QC, and provenance.
- Saved publication figures and the scripts defining primary, contextual, IG-temporal, Hallmark, incremental-value, and network metrics.

## Approval boundary

This review proposes a supplementary publication scope and is **pending explicit user approval**. Saving this file does not authorize notebook packaging, documentation reconciliation, committing, or pushing. Those actions must wait for approval of this concrete scope or requested revisions.
