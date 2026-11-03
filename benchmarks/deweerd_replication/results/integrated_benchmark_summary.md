# Integrated BridgeRNA biological and drug-recovery benchmark

This report separates de Weerd replication-style evaluations from the
previously completed cross-study space-condition extension. The frozen
ExpressionPerformer checkpoint, natural `log1p(TPM)` input, signed
leave-donor-out Integrated Gradients attribution, and top-500 primary
module definition are unchanged.

## Evaluation 1: Disease-gene recovery

The primary reference is exact-CUI DisGeNET v7.0 intersected with each
dataset's expressed canonical-gene universe. This is the release closest
to de Weerd, but not an exact reconstruction: their filtered gene lists
were not published and their reported reference totals differ markedly.

| Disease | Method | Recovered / reference | OR | raw p | BH q | de Weerd VAE |
|---|---|---:|---:|---:|---:|---:|
| MS active lesion | Bridge | 72/1330 | 1.77 | 2.55e-05 | 3.06e-05 | 141/300, OR 10.03 |
| MS active lesion | DE | 87/1330 | 2.25 | 5.86e-10 | 1.76e-09 | — |
| Crohn ileum | Bridge | 70/859 | 1.95 | 2.22e-06 | 3.33e-06 | 61/168, OR 6.97 |
| Crohn ileum | DE | 79/859 | 2.28 | 2.26e-09 | 4.53e-09 | — |
| SLE blood | Bridge | 106/1289 | 2.76 | 2.36e-16 | 1.42e-15 | 6/95, OR 0.67 |
| SLE blood | DE | 72/1289 | 1.67 | 1.22e-04 | 1.22e-04 | — |

Module-size sensitivity covers 50–500 genes in increments of 50;
the complete machine-readable table reports raw p, BH q, overlap, and
odds ratio for every size. A DisGeNET score ≥0.1 sensitivity is kept
separate from the all-association primary analysis.

At top 500, DE recovers more reference genes than Bridge for MS
(87 vs 72) and Crohn (79 vs 70), while Bridge recovers more for SLE
(106 vs 72). Therefore these cohorts do not support a general claim
that Bridge improves disease-gene recovery over DE; the advantage is
disease-specific and strongest for SLE.

All three studies are overwhelmingly pretraining-exposed (39/41 MS,
71/74 Crohn, 116/117 SLE in ARCHS4 train/validation), so this evaluates
recovery/reproducibility rather than unseen-study generalization.

## Evaluation 2: Published drug recovery

Exact DrugBank replication was not authorized. De Weerd tested 328
DrugBank drugs with >10 targets against 16,600 proteins. The frozen
ChEMBL 37 graph has 985 approved small-molecule mechanism drugs, 323
targets, and only one drug with >10 targets. Thus the strict analogue
cannot meaningfully compare published-drug recovery. The ≥3-target
ChEMBL endpoint and ≥1-target reference audit are labeled sensitivities.

| Disease | Published drug (#) | ChEMBL targets | Bridge rank | DE rank | Bridge overlap | DE overlap | Bridge p/q | DE p/q |
|---|---|---:|---:|---:|---|---|---:|---:|
| MS active lesion | Muromonab (1) | 0 | absent | absent | — | — | NA/NA | NA/NA |
| MS active lesion | Ibrutinib (2) | 1 | 446 | 484 | — | — | 1.00e+00/1.00e+00 | 1.00e+00/1.00e+00 |
| MS active lesion | Daclizumab (3) | 0 | absent | absent | — | — | NA/NA | NA/NA |
| MS active lesion | Zanubrutinib (4) | 1 | 661 | 686 | — | — | 1.00e+00/1.00e+00 | 1.00e+00/1.00e+00 |
| MS active lesion | Alemtuzumab (5) | 0 | absent | absent | — | — | NA/NA | NA/NA |
| Crohn ileum | Zinc (1) | 0 | absent | absent | — | — | NA/NA | NA/NA |
| Crohn ileum | Zinc Acetate (2) | 0 | absent | absent | — | — | NA/NA | NA/NA |
| Crohn ileum | Dilmapimod (3) | 0 | absent | absent | — | — | NA/NA | NA/NA |
| Crohn ileum | Glucosamine (4) | 0 | absent | absent | — | — | NA/NA | NA/NA |
| Crohn ileum | Vx-702 (5) | 0 | absent | absent | — | — | NA/NA | NA/NA |
| SLE blood | Gemcitabine (1) | 1 | 916 | 917 | — | — | 1.00e+00/1.00e+00 | 1.00e+00/1.00e+00 |
| SLE blood | Enzastaurin (2) | 0 | absent | absent | — | — | NA/NA | NA/NA |
| SLE blood | Sunitinib (3) | 9 | 14 | 43 | CSF1R | — | 2.82e-01/1.00e+00 | 1.00e+00/1.00e+00 |
| SLE blood | Fostamatinib (4) | 1 | 487 | 477 | — | — | 1.00e+00/1.00e+00 | 1.00e+00/1.00e+00 |
| SLE blood | Cladribine (5) | 0 | absent | absent | — | — | NA/NA | NA/NA |

Ranks in the table use the strict endpoint when eligible, then the
≥3 sensitivity, then the ≥1 reference-audit universe. Missing compounds
are absent from the approved direct-mechanism graph; below-threshold
ranks are not primary benchmark recoveries. The full table contains
target counts, raw p, FDR, and endpoint-specific ranks.

Only Sunitinib overlaps a top-500 module (Bridge: CSF1R; raw
p=0.282, FDR=1.0), and it is not significant. Ibrutinib,
zanubrutinib, gemcitabine, and fostamatinib are represented but have
zero module-target overlap for both methods. Consequently, this
ChEMBL audit does not establish that Bridge reproduces de Weerd's
published drug signals better than DE.

Top independently generated nonzero-overlap drugs (≥3 targets):

- MS active lesion, Bridge: AFATINIB, NERATINIB, DACOMITINIB ANHYDROUS, ETRASIMOD, VADADUSTAT
- MS active lesion, DE: PAZOPANIB, VADADUSTAT, UMBRALISIB, TRIMIPRAMINE
- Crohn ileum, Bridge: VADADUSTAT, DICHLORPHENAMIDE, METHAZOLAMIDE, ACETAZOLAMIDE
- Crohn ileum, DE: none
- SLE blood, Bridge: ETRASIMOD, AFATINIB, DOXYCYCLINE ANHYDROUS, NERATINIB, IBUDILAST
- SLE blood, DE: REGORAFENIB, PEMIGATINIB, PERHEXILINE, DASATINIB ANHYDROUS, VERNAKALANT

## Evaluation 3: Cross-study space-condition convergence

This is the previously completed Bridge-specific extension, not a de
Weerd replication. No Phase 2 analysis was rerun.

| Condition | Bridge Jaccard | DE Jaccard | Difference | exact permutation p |
|---|---:|---:|---:|---:|
| Radiation | 0.155 | 0.042 | 0.114 | 0.250 |
| Bone loss | 0.142 | 0.181 | -0.038 | 0.750 |
| Muscle atrophy | 0.714 | 0.182 | 0.532 | 0.250 |
| Overall | 0.337 | 0.135 | 0.202 | 0.059 |

The saved muscle interpretation remains: Bridge Jaccard 0.714; four
drugs recur across all three studies; universal target recurrence is
primarily ERBB2 and PDE5A. Removing ERBB2 lowers Jaccard to 0.500,
removing PDE5A lowers it to 0.619, and PDE10A supplies only
spaceflight-specific ibudilast support. Corrected GO/KEGG enrichment
of the strict two-gene recurrent set is not significant. This is a
low-dimensional target recurrence amplified by ChEMBL connectivity,
not evidence of therapeutic efficacy.

## Overall conclusion

The three evaluations answer different questions and must not be pooled.
Disease-gene recovery compares representation-derived modules with known
biology and favors Bridge only in SLE at the primary endpoint; published-
drug recovery is strongly limited by non-equivalent
DrugBank/ChEMBL coverage; cross-study Jaccard tests reproducibility across
space-condition cohorts. None is a direction-aware reversal test or
evidence that an enriched drug is therapeutically effective.
