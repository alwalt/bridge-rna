# Scientific report: frozen Bridge representations and Hallmark biology

## Scope and provenance

Bridge remained frozen throughout. TCGA embeddings were regenerated from the authoritative checkpoint and 9,816 × 15,165 natural `log1p(TPM)` matrix, using exact canonical order and unweighted final-layer mean pooling. The cohort is one deterministic primary-tumor aliquot per patient across 32 cancer types. No legacy TCGA embedding or previous exploratory pathway-head implementation was used.

Independent Hallmark targets are deterministic within-sample rank-mean scores calculated from expression and standardized within dataset. Bridge pathway scores are linear-probe predictions, not GSEA enrichment scores. Conventional GSEA uses Welch differential-expression statistics on `log1p(CPM)` and 1,000 preranked permutations. Sample-level evaluation is donor/patient-disjoint.

## 1. Final-representation Hallmark biology

GTEx mean held-out Pearson was 0.746 for the best training-selected coordinate, 0.825/0.882/0.947 for 5/10/25 coordinates, and 0.991 for all 512; the null was -0.002. GTEx entropy effective rank was 3.05 and 99% of variance required 12 PCs. The information is distributed rather than one-pathway/one-coordinate.

TCGA replicated this across 9,816 patients. Mean patient-held-out Pearson was 0.564, 0.691, 0.738, 0.830, and 0.967 for 1/5/10/25/512 coordinates; the null was 0.0016. Mean full-model Spearman was 0.960 and R2 was 0.935. TCGA entropy rank was 5.31 and 99% variance required 38 PCs.

## 2. GTEx→TCGA zero-shot transfer

The GTEx-only scaler and 512-D ridge map were frozen before using TCGA targets. All 50 TCGA correlations were positive. Zero-shot Pearson was 0.794 mean, 0.852 median, and 0.305–0.944 range. The strongest included Xenobiotic Metabolism (0.944), E2F Targets (0.938), EMT (0.935), and Myc Targets V1 (0.931). GTEx/TCGA coefficient cosines were only 0.365 mean, showing that stable prediction can coexist with different fitted vectors in a correlated representation.

## 3. Individual-sample profiles

Every GTEx and TCGA sample has a common-order 50-Hallmark vector from the frozen GTEx probe. Individual and group-mean heatmaps are descriptive; target correlations and transfer provide validation. The contextual-decoder mean-member profile remains heterogeneous (GTEx median target r=0.195), while top-k pooling was negative because probabilities saturated.

## 4. Vocabulary effect on conventional DE/GSEA

The controlled comparison used 25,150 available TCGA genes versus 14,656 exact Bridge-vocabulary intersections, holding samples, statistics, Hallmarks, and GSEA parameters fixed. Whole-versus-restricted NES Pearson was 0.984 for LUAD–LUSC and 0.965 for KIRC–KIRP; direction agreement was 98% and 96%, rank Spearman 0.972 and 0.983, and leading-edge Jaccard 0.856 and 0.855. Restriction removed five and three whole-genome-significant pathways, introduced none, and retained significant-set Jaccard 0.80 and 0.897. Vocabulary restriction alters some marginal calls but not most ordering or direction.

## 5. Bridge versus GSEA

For LUAD–LUSC, Bridge-delta rank correlation was 0.599 with whole-genome NES and 0.613 with vocabulary-matched NES; direction agreement was 82% and 80%. For KIRC–KIRP the correlations were 0.736 and 0.775, with 82% direction agreement. Vocabulary matching modestly improves agreement, so some discordance reflects vocabulary, but most remaining disagreement reflects different estimands: per-sample model scores versus group-level DE ranks. Bridge-only signals are not discoveries.

## 6. Contextual decoder, IG, and DE

The held-out-gene decoder achieved macro AUROC 0.750 and AUPRC 0.0570 versus prevalence 0.00866, establishing above-baseline membership geometry. The decoder measures Hallmark-like information in a contextual gene representation. IG instead measures an input gene's contribution to one sample's frozen GTEx-probe Hallmark output relative to an all-zero natural `log1p(TPM)` reference. It is sample-specific and is not LUAD–LUSC or KIRC–KIRP attribution.

The original eight-step right-endpoint integration failed numerical validation: median relative completeness error was 47.6% and the worst error was 393%. Composite-trapezoid integration greatly improved completeness while preserving the same straight zero-to-sample path. Under the prespecified all-profile gate (relative completeness error ≤1%, absolute-rank Spearman versus the reference ≥0.99, and top-50 overlap ≥90%), 8, 16, 32, and 64 steps failed completeness; **128 steps was the lowest accepted setting**. At 128 steps the median relative completeness error was 0.0856%, the maximum was 0.664%, and all 16 sample/pathway profiles had exact top-50 agreement with the 128-step reference. The failed right-endpoint and incomplete 64-step trapezoid audits are preserved separately.

With validated 128-step IG, decoder-versus-absolute-IG Spearman averaged 0.184, compared with 0.185 originally. Mean top-10/25/50 decoder–IG overlaps were 0.063/0.188/0.750 genes, and top-50 overlap with vocabulary-matched GSEA leading edges remained 1.0 gene. Canonical members averaged 1.875 of the top 50 IG genes versus 0.601 in 1,000 size-matched random sets; 4/16 empirical tests had p<0.05, compared with the earlier 5/16. Thus the main negative conclusion is unchanged—decoder identity and IG contribution are weakly related and distinct—but the evidence for canonical-member enrichment is slightly weaker after numerical validation.

The gene-level table records DE effect/statistic/FDR, canonical membership, GSEA leading edge, decoder score, signed IG, and absolute IG. Rank-based decoder×IG quadrants preserve discordant genes for follow-up without labeling nonmembers as new pathway members or treating discordance as model superiority.

## 7. Exploratory matched organs

GTEx lung↔TCGA LUAD and GTEx mammary tissue↔TCGA BRCA were examined only after within-dataset validation. Bridge-delta versus whole-GSEA Spearman was 0.796 and 0.810; vocabulary-matched values were 0.797 and 0.815. Whole/restricted NES Pearson was 0.967 and 0.965. These are cross-cohort displacements, not tumor effects, and are confounded by cohort, processing, collection, demographics, composition, purity, and technical effects.

## 7b. Pan-cancer Bridge versus conventional GSEA

Seven scientifically related, patient-disjoint primary-tumor contrasts were evaluated: LUAD–LUSC, all three pairwise renal subtype comparisons among KIRC/KIRP/KICH, GBM–LGG, COAD–READ, and UCEC–UCS. These are labeled subtype/site comparisons. Tumor-versus-adjacent-normal analyses were not run because the frozen Bridge-profile cohort contains only primary tumors; adding normals would require a new inference cohort.

Across contrasts, Bridge delta versus whole-transcriptome GSEA had median Spearman 0.705 (IQR 0.654–0.772), median Pearson 0.697, and median pathway-direction agreement 82%. All 7/7 correlations were positive. Against vocabulary-matched GSEA, median Spearman was 0.712 (IQR 0.644–0.801), median Pearson 0.720, median direction agreement remained 82%, and all 7/7 correlations were positive. Median top-5 and top-10 pathway overlap were 40% and 50% for both comparisons.

Whole-transcriptome and Bridge-vocabulary GSEA were much closer to each other: median Spearman 0.988, median Pearson 0.982, median direction agreement 98%, median top-5 overlap 100%, and median top-10 overlap 90%. Vocabulary matching therefore produces only a small median improvement in Bridge/GSEA agreement and does not explain most residual discordance.

Thirty-three Hallmarks met the descriptive consistent-agreement rule (direction agreement ≥80% against both GSEA universes). Particularly consistent examples included DNA Repair, Myc Targets V1, Oxidative Phosphorylation, Epithelial Mesenchymal Transition, Reactive Oxygen Species Pathway, Hedgehog Signaling, TGF-beta Signaling, IL-6/JAK/STAT3 Signaling, p53 Pathway, and G2-M Checkpoint. No pathway met the stringent consistent-disagreement rule (≤20% direction agreement against both universes). The weakest/context-dependent pathways included Pancreas Beta Cells, Adipogenesis, Cholesterol Homeostasis, and KRAS Signaling Dn. These results support broad but context-dependent agreement; they do not make Bridge-only pathways discoveries.

## 8. Frozen-system decision

The prespecified acceptance checks passed: median TCGA within-cohort and zero-shot correlations are positive, and all 50 zero-shot correlations are positive. The sample→Bridge→frozen GTEx Hallmark system and optional contextual decoder/IG tools are hashed and versioned. This supports later unchanged application to the cross-species benchmark; no cross-species sample was analyzed here.

## Limitations

Hallmarks overlap and do not exhaust biology. Rank-mean targets are not running-sum ssGSEA. Tissue/cancer identity, composition, RNA quality, and global expression can drive associations. Ridge coefficients are unstable under correlated features despite stable predictions. GSEA uses unadjusted Welch statistics; no clinical covariates were introduced. The pan-cancer contrasts compare related primary-tumor cohorts but do not adjust for clinical or molecular covariates and do not include adjacent normals. IG uses a biologically artificial zero-expression reference, four pathways, and one deterministic sample per cohort; even with validated 128-step integration it remains targeted and exploratory. Cross-cohort comparisons are deeply confounded. Correlation is not causation.
