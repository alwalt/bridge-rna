# Final report: Bridge Hallmark functional-state retrieval

**Benchmark status: COMPLETE.** The frozen benchmark is closed; remaining ideas are future work, not unfinished analyses.

## Frozen system and database

The benchmark reused, without retraining, the validated GTEx-only ridge probe and exactly compatible ARCHS4 embeddings. The database contains 510,709 human samples × 50 ordered Hallmark scores. The 512-D cache passed a 40,000-row bitwise compatibility witness. TCGA was evaluation-only.

## TCGA validation

Across representative samples and cohort-centroid queries, mean metadata keyword precision@10 was 0.777 in Hallmark space and 0.809 in Bridge space. The paired mean difference was -0.032 (median -0.050; Hallmark better for 6/22 queries, tied for 5/22). This keyword metric is deliberately external to Hallmark labels, but is coarse because ARCHS4 metadata are free text.

## OSDR liver test

The repository-defined cohort contains 112 mouse liver RNA-seq samples (spaceflight FLT and ground-control GC across missions). These were not relabeled as human fatty-liver cases. Across 28 representative/group queries, liver-keyword precision@10 was 0.061 for Hallmark retrieval versus 0.986 for global Bridge retrieval. The Hallmark top hits were predominantly adipose samples, driven chiefly by Unfolded Protein Response, Wnt-beta Catenin Signaling, mTORC1 Signaling, KRAS Signaling Dn, Protein Secretion, and/or Cholesterol Homeostasis; the 512-D baseline overwhelmingly retrieved liver/hepatocyte samples and included a MAFLD liver sample for one RR6-LAR ground-control representative. This is evidence of cross-species/domain-transfer failure for this frozen Hallmark retrieval, not evidence of improved functional matching. Applying the human GTEx probe to mouse is exploratory despite the shared one-to-one vocabulary and species-agnostic checkpoint.

## Scientific conclusion

These results test retrieval behavior, not Hallmark-label recovery. The aggregate answer is negative: 50-D retrieval did not outperform 512-D global retrieval on TCGA and failed the OSDR liver-coherence check. There are narrow TCGA examples where functional retrieval improved metadata coherence—KIRC (+0.30 precision@10), BRCA (+0.10), and LUAD/LUSC cohort centroids (+0.10)—but these heterogeneous qualitative wins do not offset the overall result and require independent biological review. Nothing here establishes batch correction, causal program equivalence, or clinical validity.

## Limitations

- ARCHS4 titles/source/characteristics are incomplete and noisy; keyword coherence undercounts implicit biology.
- The Hallmark scores are frozen linear predictions of GTEx-derived ssGSEA-like targets, not direct pathway measurements.
- Cosine contributions are coordinate-wise decompositions of similarity, not gene-level attribution or mechanism.
- TCGA and OSDR were not used to fit or tune the head, queries, or scoring rule.

## Similarity-scaling audit

This follow-up used frozen outputs only: no Bridge inference and no head retraining. The saved probe contains a 512-D input-feature mean and scale from the GTEx embedding `StandardScaler`; these are not Hallmark-output statistics. Exact 50-D training-target statistics were recovered from the frozen 4,260-sample GTEx target table. The targets had already been standardized at construction to per-Hallmark mean zero and population SD one.

| Similarity method | TCGA metadata precision@10 | OSDR liver precision@10 | OSDR adipose fraction@10 |
|---|---:|---:|---:|
| Raw 50-D cosine | 0.777 | 0.061 | 0.939 |
| GTEx-target-centered cosine | 0.777 | 0.061 | 0.939 |
| GTEx-target-z-scored cosine | 0.777 | 0.061 | 0.939 |
| Pearson correlation | 0.768 | 0.175 | 0.796 |
| Bridge-512 cosine | **0.809** | **0.986** | **0.000** |

Cosine already includes row-wise L2 normalization, so separate L2-normalized centered and z-scored methods would be exact duplicates. Raw, centered, and z-scored retrieval were identical because the valid frozen target normalization is already essentially the identity. Pearson changed rankings and reduced OSDR adipose retrieval, but still failed badly relative to Bridge-512 and slightly reduced TCGA coherence.

ARCHS4 transfer does produce unequal coordinate magnitudes: the five Hallmarks with the largest raw second moments contribute 30.2% of squared norm, the top ten contribute 48.0%, and the inverse-concentration effective dimensionality is 29.9 of 50. Dominant dimensions include heme Metabolism, E2F Targets, Myc Targets V1, DNA Repair, and G2-M Checkpoint. Yet this anisotropy is not the primary cause: every legitimate frozen GTEx scaling produces the same failure, and correlation similarity remains far below the baseline.

OSDR adipose confusion is driven most strongly by Cholesterol Homeostasis, mTORC1 Signaling, TNF-alpha Signaling via NF-kB, Mitotic Spindle, Fatty Acid Metabolism, Protein Secretion, Adipogenesis, and Myogenesis. Pearson reduces adipose dominance from 0.939 to 0.796 but does not restore liver retrieval.

The evidence therefore supports information loss/domain-transfer failure in the 512→50 functional projection, not a poorly scaled cosine space. Properly normalized 50-D methods still substantially underperform Bridge-512, so this benchmark stops here and does not build the contextual-program database.

## All-33 TCGA frozen-output extension

### Design and metadata adjudication

This extension used all 9,942 frozen, one-specimen-per-patient TCGA queries across 33 cancer types: 9,816 solid-primary samples from the original cache and 126 LAML peripheral-blood samples from the pre-existing compatible parity cache. It also formed one arithmetic-mean centroid per cancer before cosine normalization. Exact top-100 retrieval against 510,709 ARCHS4 samples was run in both spaces; P@10, P@50, and P@100 are nested cuts of those same rankings. No Bridge inference, head retraining, or TCGA tuning occurred.

Free-text ARCHS4 annotations were assessed separately for exact disease, tissue/organ, malignancy, and an identifiable other cancer from the same tissue. Ambiguous records are not called confirmed negatives: the machine-readable results contain confirmed precision, annotation coverage, and conditional precision among evaluable records. The primary per-cancer comparison below is confirmed precision for exact disease or an identifiable same-tissue cancer. This is conservative and reproducible, but remains a metadata benchmark rather than pathology review.

### Individual-patient retrieval: all 33 cancers

| Cancer | H@10 | B@10 | Δ10 | H@50 | B@50 | Δ50 | H@100 | B@100 | Δ100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ACC | .004 | .000 | +.004 | .003 | .001 | +.002 | .003 | .001 | +.002 |
| BLCA | .273 | .363 | -.090 | .192 | .270 | -.078 | .156 | .217 | -.061 |
| BRCA | .789 | .821 | -.032 | .696 | .763 | -.067 | .622 | .700 | -.078 |
| CESC | .093 | .062 | +.031 | .089 | .059 | +.030 | .080 | .057 | +.023 |
| CHOL | .406 | .497 | -.092 | .262 | .292 | -.030 | .193 | .193 | -.000 |
| COAD | .279 | .241 | +.037 | .271 | .249 | +.021 | .267 | .242 | +.025 |
| DLBC | .890 | .865 | +.025 | .811 | .826 | -.015 | .751 | .754 | -.003 |
| ESCA | .104 | .129 | -.026 | .086 | .089 | -.003 | .073 | .075 | -.002 |
| GBM | .562 | .645 | -.083 | .566 | .631 | -.065 | .568 | .613 | -.045 |
| HNSC | .143 | .185 | -.042 | .120 | .153 | -.033 | .104 | .132 | -.028 |
| KICH | .474 | .515 | -.041 | .164 | .241 | -.077 | .094 | .162 | -.068 |
| KIRC | .511 | .555 | -.043 | .420 | .420 | -.001 | .334 | .336 | -.002 |
| KIRP | .331 | .437 | -.106 | .196 | .267 | -.072 | .144 | .211 | -.067 |
| LAML | .475 | .563 | -.089 | .481 | .510 | -.029 | .482 | .499 | -.017 |
| LGG | .519 | .546 | -.027 | .463 | .478 | -.015 | .407 | .426 | -.019 |
| LIHC | .826 | .868 | -.042 | .789 | .830 | -.041 | .758 | .799 | -.042 |
| LUAD | .216 | .250 | -.034 | .168 | .226 | -.058 | .147 | .213 | -.066 |
| LUSC | .338 | .355 | -.017 | .264 | .304 | -.040 | .217 | .265 | -.048 |
| MESO | .000 | .013 | -.013 | .001 | .006 | -.004 | .001 | .003 | -.002 |
| OV | .635 | .767 | -.132 | .487 | .694 | -.207 | .407 | .584 | -.177 |
| PAAD | .127 | .168 | -.041 | .104 | .102 | +.001 | .091 | .077 | +.014 |
| PCPG | .570 | .904 | -.334 | .287 | .514 | -.227 | .180 | .301 | -.121 |
| PRAD | .839 | .914 | -.075 | .798 | .847 | -.049 | .768 | .803 | -.035 |
| READ | .557 | .611 | -.054 | .532 | .574 | -.042 | .523 | .563 | -.040 |
| SARC | .166 | .177 | -.010 | .139 | .140 | -.001 | .118 | .113 | +.006 |
| SKCM | .800 | .798 | +.002 | .685 | .769 | -.084 | .626 | .750 | -.124 |
| STAD | .316 | .377 | -.061 | .244 | .289 | -.046 | .214 | .255 | -.042 |
| TGCT | .029 | .015 | +.015 | .030 | .019 | +.011 | .027 | .017 | +.010 |
| THCA | .760 | .855 | -.095 | .651 | .772 | -.121 | .593 | .684 | -.091 |
| THYM | .000 | .000 | +.000 | .000 | .000 | +.000 | .000 | .000 | +.000 |
| UCEC | .052 | .121 | -.070 | .036 | .060 | -.023 | .032 | .044 | -.013 |
| UCS | .000 | .002 | -.002 | .003 | .005 | -.002 | .003 | .004 | -.001 |
| UVM | .608 | .501 | +.106 | .281 | .237 | +.044 | .187 | .187 | +.001 |

H is Hallmark-50 cosine, B is Bridge-512 cosine, and Δ is H−B. With an absolute 0.10 P@50 threshold fixed for descriptive classification, no cancer clearly improved; OV (-0.207), PCPG (-0.227), and THCA (-0.121) clearly worsened. UVM is the strongest narrow Hallmark result at P@10 (+0.106), but the gain contracts at P@50 (+0.044) and disappears at P@100 (+0.001). CESC, COAD, and TGCT show smaller persistent improvements. THYM's zero is not evidence of equivalence: neither representation retrieved explicitly identifiable thymoma metadata.

### Criterion-specific hypothesis test

For individual-patient queries, macro confirmed P@50 was 0.305 versus 0.338 for exact disease, 0.395 versus 0.424 for tissue, 0.707 versus 0.777 for malignancy, and 0.008 versus 0.015 for identifiable same-tissue related cancer (Hallmark versus Bridge). The composite disease/related-cancer P@50 was 0.313 versus 0.353. Thus Hallmark-50 did not show the proposed disease/program advantage, and Bridge-512 was also stronger for tissue/context.

The Hallmark deficit for identifiable same-tissue related cancer (-0.006 P@50) was numerically smaller than its tissue deficit (-0.029), but both rates were very low and Hallmark had no positive absolute advantage. Therefore this difference does not substantiate the hypothesis. Conditional precision and annotation coverage lead to the same cautious interpretation because coverage is incomplete and method-dependent.

Centroids reveal a few biologically plausible ranking changes without reversing the aggregate result. Hallmark's UVM centroid had 8/10 exact uveal-melanoma hits versus 6/10 for Bridge, and CHOL had 9/10 versus 6/10. The Hallmark LUAD centroid retrieved three LUSC samples from GSE181043—plausible same-lung malignancies—while Bridge retrieved none in its top ten; the dominant positive Hallmark coordinates were E2F Targets, DNA Repair, G2-M Checkpoint, Myc Targets V1, Estrogen Response Late, and Interferon Alpha Response. UVM matches were dominated by DNA Repair, Myc Targets V2, UV Response Up/Down, Mitotic Spindle, and TGF-beta Signaling; CHOL matches by DNA Repair, Myc Targets V1/V2, E2F Targets, Glycolysis, and G2-M Checkpoint. These are qualitative coordinate-wise similarity decompositions, not causal explanations.

Across centroids, macro P@10 was nearly tied: exact disease 0.482 Hallmark versus 0.488 Bridge, tissue 0.512 versus 0.506, malignancy 0.855 versus 0.864, and identifiable same-tissue related cancer 0.015 versus 0.012. The centroid result suggests occasional useful functional reranking, but the all-patient evaluation is more stable and favors Bridge overall.

### Conclusion of the 33-cancer extension

The full pan-cancer result reinforces the earlier negative conclusion. Hallmark-50 sometimes retrieves a different, biologically plausible disease state—most clearly LUAD→LUSC and the UVM/CHOL centroid improvements—but it does not generally improve disease/program matching, and it loses substantial tissue and malignancy information for several cancers. The strongest evidence remains consistent with information lost in the 512→50 projection rather than a broad functional-retrieval advantage.

The OSDR analysis above remains a separate cross-species/tissue stress test and is not pooled with this TCGA conclusion.

## Dominant-signal matching versus whole-vector cosine

### Frozen-output design

This follow-up used only the existing 510,709 × 50 ARCHS4 Hallmark matrix, the frozen Hallmark profiles underlying all 33 TCGA centroids, and the saved Bridge-512 centroid rankings. It compared whole-vector cosine, cosine of signed within-sample absolute ranks, signed top-5 and top-10 overlap, a query-absolute-value-weighted top-10 score, and an asymmetric top-10 rank-containment score that ignores non-query programs and applies only a 0.25 penalty to sign mismatches. Discrete overlap scores use whole-vector cosine solely as a deterministic tie-breaker. No Bridge inference, training, or tuning occurred.

Signal coverage is intentionally independent of retrieval score: it is the fraction of the query's top 5 or 10 absolute Hallmarks that also occur in the retrieved sample's corresponding top set with the same sign. Score-background quantiles were calculated against all 510,709 ARCHS4 samples for every query and method.

### Macro top-10 comparison across 33 centroids

| Method | Exact disease | Tissue | Related cancer | Signal cov.@5 | Signal cov.@10 | Unique studies | Largest-study fraction | Bridge overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Whole-vector cosine | **0.482** | **0.512** | 0.015 | 0.821 | 0.791 | 5.24 | 0.442 | 0.300 |
| Signed absolute rank | 0.433 | 0.461 | 0.006 | 0.781 | 0.763 | 5.76 | 0.394 | 0.261 |
| Signed top-5 overlap | 0.270 | 0.318 | 0.009 | **0.945** | 0.679 | 5.21 | 0.464 | 0.130 |
| Signed top-10 overlap | 0.388 | 0.452 | 0.009 | 0.790 | **0.888** | **6.64** | **0.333** | 0.179 |
| Query-weighted top-10 | 0.227 | 0.279 | 0.009 | 0.688 | 0.693 | 6.00 | 0.370 | 0.073 |
| Asymmetric containment | 0.388 | 0.467 | 0.003 | 0.806 | 0.854 | 6.21 | 0.358 | 0.182 |
| Bridge-512 reference | 0.488 | 0.506 | 0.012 | 0.730 | 0.707 | 6.18 | 0.348 | 1.000 |

Signal-selective retrieval does what its objective requests: top-10 overlap raises matched signed top-10 coverage from 0.791 to 0.888, and containment raises it to 0.854. This comes at a substantial identity cost: exact-disease P@10 falls from 0.482 for Hallmark cosine to 0.388 for both methods. Signed absolute rank is the least disruptive alternative, but neither its signal coverage nor its disease/tissue precision exceeds whole-vector cosine. Bridge is included as a global-transcriptome reference: it has lower signal coverage than Hallmark cosine but slightly higher exact-disease precision and similar tissue precision.

### Deep examples

- **UVM:** whole cosine returns 8/10 exact UVM hits and 0.71 signal coverage. Signed top-10 returns 6/10 exact hits with 0.76 coverage. Its first five remain UVM samples, while top-5 overlap admits an untreated primary melanocyte sample. Query-weighting fails completely (0/10 UVM; 0.28 coverage), illustrating score saturation without high-rank agreement.
- **CHOL:** whole cosine yields 9/10 exact cholangiocarcinoma hits with 0.88 coverage. Top-10 overlap increases coverage to 0.92 but reduces exact disease to 6/10; a non-viral HCC liver sample is a plausible hepatic cancer analog. Top-5 overlap retrieves no explicit CHOL samples and includes SARS-CoV-2 and unrelated tumor/cell contexts.
- **LUAD:** whole cosine gives 2/10 exact LUAD, 9/10 lung tissue, and 0.80 coverage. Top-10 overlap raises coverage to 0.90 but retains only 2/10 exact and 2/10 tissue. Containment also reaches 0.90 coverage yet its top five are Crohn's-disease intestinal samples from two studies—shared programs, but not convincing lung-cancer analogs. Query-weighting retrieves two LUSC samples, a plausible same-lung cancer state, but its broader precision is poor.
- **PCPG:** whole cosine and Bridge each return 10/10 exact disease from one study. Containment diversifies retrieval and retains 7/10 exact disease, including a plausible pituitary neuroendocrine tumor analog. Top-10 overlap and query-weighting increasingly substitute ovarian follicles, DNA-damage cell models, and other contexts.
- **OV:** whole cosine returns 8/10 exact ovarian cancer with 0.85 coverage. Top-10 overlap reaches 0.94 coverage but only 2/10 exact disease; containment reaches 0.92 with 3/10 exact. Several cross-context hits are xenografts or other tumors, so high signal coverage alone does not establish biological equivalence.
- **THCA:** signed rank returns 10/10 exact thyroid cancer, matching Bridge and slightly improving over whole cosine's 9/10. Top-5 overlap also returns 10/10. Query-weighting collapses to 0/10 disease and 0.36 signal coverage despite nominal scores of 1.0.
- **KICH:** signed top-5/top-10 and containment modestly improve exact disease from 6/10 to 7/10 while maintaining 0.9–1.0 tissue precision; top-10 also retrieves a clear-cell kidney cancer as a related same-tissue cancer. Query-weighting again fails (0/10 disease and tissue).

The complete top-five audit records GSM, GSE, metadata, score, full-background percentile, dominant query Hallmarks, matched signed signals, and conservative interpretation. Automated “candidate analog” labels are flags for review, not biological validation.

### Background, ubiquity, and hub checks

All selected top-five scores lie at approximately the 99.999th percentile of their full ARCHS4 score backgrounds, but this does not imply specificity. Query-weighted scores frequently saturate at 1.0 for unrelated samples because capping per-coordinate magnitude ratios discards rank information. Overlap methods also have extensive ties; the full-background 99.9th-percentile mean scores are 0.818 for top-5 overlap and 0.752 for top-10 overlap.

ARCHS4's most frequent strongest coordinates are E2F Targets (13.0%), heme Metabolism (12.8%), DNA Repair (11.3%), Myc Targets V1 (10.9%), and Cholesterol Homeostasis (9.4%). Retrieved top tens are enriched further for DNA Repair (33–35% for cosine, signed rank, top-10 overlap, and containment) and often E2F/Myc. Thus some apparent functional analogs are driven by ubiquitous proliferation/DNA-repair programs rather than disease-specific combinations.

No single study dominates globally, but top-5 overlap is most hub-prone: GSE108383/GSE108397 occupies 8.8% and GSE108322 8.2% of all 330 top-ten slots. Whole cosine's largest study occupies 3.6%; signed rank and top-10 overlap are each about 3.0%; containment is 4.5%. The increased study diversity of top-10 overlap is real, but it accompanies lower disease and tissue coherence.

### Scientific conclusion

Dominant-signal matching is useful as a secondary retrieval view, not as a replacement for the complete Hallmark state. Signed top-10 overlap and asymmetric containment recover substantially more of the requested signed program set and sometimes expose plausible same-tissue or neuroendocrine analogs. However, they also retrieve cross-context samples sharing generic proliferation, DNA-repair, or stress programs, reducing exact disease and tissue correspondence. The strongest practical formulation in this frozen benchmark is therefore two-stage or multi-view: retain whole-vector cosine (or signed rank) for contextual coherence, then expose top-10 overlap/containment as an interpretable functional reranking or annotation. The data do not support using sparse signal matching alone as the primary nearest-neighbor metric.

## All-eligible OSDR spaceflight reference-atlas analysis

### Cohort and frozen representations

Eligibility required bulk RNA-seq, a unique metadata/count-column match, and both FLT and GC within the exact study, organism, tissue, strain, sex, age, and genotype stratum. This yielded 1,471 samples (731 FLT, 740 GC), 64 studies, one organism (mouse), 45 tissues, and 88 matched groups; every group has at least two replicates per condition. Basal/vivarium controls, single-cell assays, unmatched strata, ambiguous columns, and technical repeats were excluded or retained only in the eligibility audit.

All 112 rows in the prior liver cache regenerated within (10^{-12}) and were reused. The remaining 1,359 samples were inferred once with the unchanged checkpoint. Preprocessing was mouse raw counts → exon-length TPM → one-to-one human orthologs → canonical 15,165 genes → natural `log1p`. The unchanged GTEx head generated the 50-D profiles. No FLT/GC labels were used for fitting or correction.

### Replicate reproducibility

At top 10, mean neighbor/GSE/metadata-category Jaccard was 0.366/0.510/0.586 for Bridge FLT replicates and 0.386/0.515/0.618 for GC. Hallmark cosine was more repetitive at 0.613/0.650/0.690 for FLT and 0.629/0.665/0.720 for GC; signed top-10 overlap was intermediate. Bridge retained much more tissue context (same-tissue fraction 0.649) than Hallmark cosine (0.119) or signed overlap (0.196). Signed-signal coverage@10 was 0.230–0.236 for Bridge, 0.455–0.457 for Hallmark cosine, and 0.629–0.630 for signed overlap.

High Hallmark consistency partly reflects hubs. GSE134570 occupies 14.4% of FLT and 13.8% of GC Hallmark-cosine top-ten slots. Bridge's largest hub, GSE128072, occupies 8.8%/7.9%; signed overlap's largest hub occupies 8.2%/7.9%. Reproducibility is therefore genuine but inflated by representation-specific concentration.

### FLT-versus-GC neighborhood shifts

The primary test used reciprocal-rank enrichment over individual top-100 lists and 1,000 label permutations inside each exact biological group. After within-group/method BH correction, requiring at least 75% FLT prevalence and at most 25% GC prevalence, 435 ARCHS4 samples were FLT-enriched across 24/88 groups: 233 Bridge hits in 12 groups, 179 Hallmark-cosine hits in 20 groups, and 23 signed-overlap hits in five groups.

These are group-specific shifts, not a universal spaceflight neighborhood. Macro FLT−GC changes across automated cancer, inflammation, oxidative, mitochondrial, hypoxic, fibrosis, metabolic, radiation, exercise/disuse, and drug categories were all near zero. Nominal cross-study patterns—cancer-like Hallmark retrieval in cerebellum, Bridge hypoxia annotations in heart, exercise/disuse in quadriceps—are small and method-dependent.

Fourteen liver groups from 12 studies provide the strongest repeated-tissue test. Only four groups produced robust FLT-enriched samples. OSD-168 Bridge retrieval preferentially recovered normal liver samples. Hallmark shifts elsewhere included adipose samples, a melanoma/stroma sample, and one liver sample; there is no consistent fatty-liver or metabolic-disease state.

### Global versus functional atlas views

The views are almost disjoint: a centroid's Hallmark-cosine top 100 overlaps Bridge by only 1.83 samples on average; signed overlap shares 2.68. Bridge supplies global tissue/context analogs—kidney to kidney, thymus to medullary thymic epithelium, and liver to liver. Hallmark supplies cross-tissue programs dominated by UPR, heme metabolism, Pancreas Beta Cells, KRAS Signaling Dn, Wnt-beta-catenin, Peroxisome, Protein Secretion, lipid, and cell-cycle programs.

Some functional matches are coherent. OSD-457 brown-adipose FLT centroids retrieve adipose/BAT samples with Peroxisome, Adipogenesis, Fatty Acid Metabolism, Cholesterol Homeostasis, and Wnt programs. OSD-666 quadriceps and OSD-99 extensor digitorum longus retrieve trained vastus-lateralis samples under signed overlap. Bone-marrow Nrf2-knockout FLT retrieves hematopoietic/drug-perturbed samples with heme, E2F, G2-M, Myc, mTORC1, and UPR signals. Other matches—particularly thymus to pancreas-secretory states or generic tumor/cell-line samples—are not convincing analogs.

Leaving out each query's dominant Hallmark preserves 75% of FLT and 70% of GC Hallmark-cosine top-five hits within the top 100. Most are therefore not literal single-coordinate matches. Sensitivity is weakest for Pancreas Beta Cells (48% survival) and heme metabolism (61%); UPR is dominant for 265/880 tested hits. Hallmark similarity remains a hypothesis, not mechanism or causal equivalence.

### Top-hit audit and priority candidates

The audit contains all 2,640 centroid top-five rows with score, percentile, GSM/GSE, metadata, tissue and category flags, signal coverage, dominant programs, and leave-one-Hallmark-out sensitivity. Thirty FLT top-five hits with same-tissue evidence or ≥0.6 signal coverage are absent from the corresponding GC top 100.

Highest-priority literature candidates are OSD-457 thymus → GSE127825 medullary thymic epithelial cells; OSD-253 kidney → GSE271143 kidney samples; OSD-457 brown adipose → GSE97156 adipose and GSE76404 LPS-exposed adipose; OSD-666 quadriceps and OSD-99 EDL → trained vastus lateralis; and OSD-164 liver → GSE128072 liver. The first two have strong replicate-consistent Bridge shifts; the adipose and muscle cases are interpretable Hallmark-specific hypotheses. The GSE134570 hub, thymus-to-pancreas matches, heterogeneous liver-to-adipose results, and generic UPR/heme/cell-cycle tumor matches should not be prioritized.

### Final answers

1. Retrieval is moderately reproducible for Bridge and highly repetitive for Hallmark, with hub inflation.
2. FLT shifts occur in 24/88 groups, but no universal FLT neighborhood is supported.
3. No single disease or perturbation state consistently resembles FLT across the atlas.
4. Bridge's strongest findings are tissue-specific; kidney, thymus, liver, brain, and muscle are clearest.
5. Cross-study recurrence is weak after establishing within-study effects; liver does not converge on one disease state.
6. Bridge retrieves tissue/global context; Hallmark retrieves almost orthogonal program combinations.
7. Some adipose and muscle cross-tissue Hallmark matches are coherent and sensitivity-stable; many others are generic.
8. Kidney, thymic-epithelial, adipose/LPS, trained-muscle, and liver examples merit literature review.
9. GSE hubs, incomplete metadata, human/mouse transfer, and dominant UPR/heme/pancreas-beta coordinates materially affect results.
10. Bridge supports a qualified reference-atlas search use case for tissue/context and group-specific analog discovery, not an autonomous atlas of universal spaceflight disease states. Hallmark is best used as a secondary hypothesis-generating view with hub and sensitivity controls.

## Final benchmark synthesis and closure

The completed benchmark supports three distinct, non-interchangeable search modes:

- **Bridge-512 — global transcriptomic similarity:** *What public transcriptomes most closely resemble my sample overall?* It is the strongest default for tissue, disease identity, and experimental context.
- **Hallmark-50 — functional-state similarity:** *What public transcriptomes share a similar configuration of biological programs?* It reproduces cancer-level conventional Hallmark geometry, but individual retrieval loses information and is best interpreted as a complementary hypothesis-generating view.
- **Species-conditioned retrieval:** *What is the closest within-species state or forced cross-species analog?* Human-only and mouse-only searches answer different questions and must be labeled explicitly.

The TCGA positive control is the clearest validation of the frozen Hallmark representation. Across 9,816 matched tumors and 32 cancer types, mean per-cancer conventional-versus-Bridge-Hallmark profile Pearson correlation is 0.756; cancer-similarity geometry Spearman correlation is 0.862 for Bridge Hallmark-50 versus 0.388 for Bridge-512; median pathway-wise cancer-ranking Spearman is 0.824. The local conventional scores are rank-mean ssGSEA-like values rather than the publication's exact ssGSEA 2.0 NES, so this validates broad functional organization rather than an exact paper reproduction.

The focused LUAD example makes the distinction concrete. The 516-patient centroid's Bridge top five are all lung adenocarcinoma, whereas Hallmark retrieves one LUAD, one LUSC, and three broader NSCLC/lung-tumor contexts. Across all 510,709 human ARCHS4 samples, Bridge and Hallmark cosine scores are moderately related (Pearson 0.487; Spearman 0.578), but the selected neighborhoods diverge: top-100 Jaccard is 0.105 and top-five Jaccard is zero. Hallmark top-100 samples have median Bridge rank 893.5, while Bridge top-100 samples have median Hallmark rank 596.5. The union-only correlation is negative because conditioning on membership in either extreme top-100 set creates a selection tradeoff; it is not the global relationship. The result supports functional reranking, not disease equivalence.

The formal leakage audit found no TCGA-derived/reused hit, no TCGA barcode, and no duplicated TCGA query identifier among 200 LUAD centroid top-100 rows. Affirmative independent-cohort evidence supports 128 rows; 72 remain metadata-insufficient rather than being assumed independent. All method-specific top-five hits have affirmative GEO cohort provenance, so exclusion does not change the LUAD interpretation.

Adding 429,746 mouse samples to 510,709 human ARCHS4 samples exposes a strong limitation: combined nearest-neighbor search is overwhelmingly species-conditioned. Hallmark-50 species NN purity is 0.9993 and Bridge-512 species NN purity is 0.9976; combined OSDR top-ten hits are 99.53% mouse for Bridge and 100% mouse for both Hallmark modes. Therefore unconstrained combined search is inappropriate when the scientific question is specifically cross-species analogy. Mouse-only search is appropriate for within-species reference states; human-only retrieval remains useful when explicitly framed as forced **mouse → human analog search**.

OSDR results remain mixed and mostly negative. Adding mouse references improves tissue coherence and replaces nearly all forced human analogs. FLT/GC shifts are group-specific, not universal; liver studies do not converge on a single fatty-liver or injury state. The OSD-665 muscle Hallmark shift toward muscle/exercise references is a descriptive example, not mechanism or causal equivalence.

### Closed scope and future work

The benchmark is complete. Contextual pathway retrieval, improved cross-species alignment/calibration, subtype-specific cancer retrieval, and drug-reversal search are explicitly **future work**. They are not required to close benchmark #9. No result here establishes mechanism, disease equivalence, therapeutic response, or causality.
# Multispecies ARCHS4 correction

The complete human+mouse extension is reported in [MULTISPECIES_FINAL_REPORT.md](./MULTISPECIES_FINAL_REPORT.md). All earlier OSDR results in `results/osdr_atlas/` are preserved and should now be read explicitly as **mouse OSDR → human ARCHS4 cross-species retrieval**.
