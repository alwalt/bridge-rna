# Muscle Bridge drug-convergence target decomposition

## Scope and counterfactual

This analysis uses the frozen Phase 2 top-500 gene modules and the frozen ChEMBL
37 endpoint requiring at least three targets per drug. It does **not** rerun
Bridge, differential expression, attribution, module construction, or drug
enrichment.

For a target-removal counterfactual, the target is deleted from each drug's
saved `overlap_genes`. A drug is removed from a dataset only if it then has no
remaining module-overlapping target. This asks how dependent the reported drug
sets are on the observed target support; it is not a newly corrected enrichment
test and does not model a changed module background.

The reported convergence is the mean of the three pairwise Jaccard indices,

\[
\bar J = \frac{1}{3}\sum_{a<b}\frac{|D_a\cap D_b|}{|D_a\cup D_b|}.
\]

The frozen Bridge pairs are 5/5, 4/7, and 4/7, giving
`(1 + 0.571429 + 0.571429) / 3 = 0.714286`.

## Exact all-study intersection

Four Bridge drugs occur in all three muscle studies:

| Drug | GSE211204 overlap | GSE113165 overlap | GSE234465 overlap | Target recurring in all three |
|---|---|---|---|---|
| Afatinib | ERBB2 | ERBB2 | ERBB2 | ERBB2 |
| Dacomitinib anhydrous | ERBB2 | ERBB2 | ERBB2 | ERBB2 |
| Neratinib | ERBB2 | ERBB2 | ERBB2 | ERBB2 |
| Ibudilast | PDE5A | PDE5A | PDE10A; PDE5A | PDE5A |

Thus there are four shared drug identifiers, two strict recurrent target genes
(ERBB2 and PDE5A), and three genes in the union of overlaps supporting those
four drugs (ERBB2, PDE5A, PDE10A). PDE10A is present only in GSE234465 and is not
required for ibudilast's recurrence.

Collapsed by target:

| Target | Drugs supported | Studies | Drug-by-study support edges |
|---|---:|---:|---:|
| ERBB2 | 3 (afatinib, dacomitinib, neratinib) | 3 | 9 |
| PDE5A | 1 (ibudilast) | 3 | 3 |
| PDE10A | 1 (ibudilast) | 1 | 1 |
| CPT1B / CPT2 | 1 (perhexiline, via different paralogs) | 1 each | 1 each |
| ADRB2 | 1 (sotalol) | 1 | 1 |
| SCN5A | 1 (vernakalant) | 1 | 1 |

ERBB2 is the only target combination responsible for multiple recurrent drugs.
No shared drug requires a genuinely multi-target overlap in all three studies.

## Bridge target ranks and signed attribution scores

| Target | GSE211204 rank / score | GSE113165 rank / score | GSE234465 rank / score |
|---|---:|---:|---:|
| ERBB2 | 147 / -0.000990 | 151 / -0.000829 | 15 / +0.003406 |
| PDE5A | 150 / +0.000978 | 88 / -0.001461 | 344 / +0.000426 |
| PDE10A | 5692 / +0.000015 | 4536 / +0.000021 | 299 / +0.000483 |

The recurrence is based on absolute Bridge rank. Neither ERBB2 nor PDE5A has a
consistent signed attribution across studies, so the result is not evidence for
a reproducible direction of dysregulation or for drug reversal.

## Sensitivity of the 0.714 Jaccard

| Removal | Mean Jaccard | Absolute change | Fraction of baseline lost |
|---|---:|---:|---:|
| None | 0.714 | 0.000 | 0.0% |
| ERBB2 | 0.500 | -0.214 | 30.0% |
| PDE5A | 0.619 | -0.095 | 13.3% |
| PDE10A | 0.714 | 0.000 | 0.0% |
| Ibudilast (most supported drug) | 0.667 | -0.048 | 6.7% |
| CPT1B | 0.679 | -0.035 | 4.9% |
| CPT2 | 0.679 | -0.035 | 4.9% |
| ADRB2 | 0.778 | +0.063 | -8.9% |
| SCN5A | 0.778 | +0.063 | -8.9% |

The most promiscuous target is ERBB2 by observed drug-by-study support (nine
edges). Ibudilast is the most promiscuous drug by observed target-by-study
support (four edges); all four shared drugs otherwise occur in all three
datasets, so occurrence count alone cannot distinguish them.

Jaccard is nonlinear, so leave-one-target-out changes are not additive causal
fractions. Two complementary summaries are nevertheless clear:

- ERBB2 generates three of four all-study shared drug identifiers (75%), while
  PDE5A generates the fourth (25%).
- Removing ERBB2 loses 30.0% of the numerical Jaccard; removing PDE5A loses
  13.3%; removing PDE10A loses none. Other study-specific targets do not create
  the all-study intersection, and removing the two GSE234465-only targets
  actually raises Jaccard by shrinking that study's union.

There is therefore no measurable residual all-study drug convergence that can
be assigned to a broader, multi-target shared program at this endpoint. The
strong value is chiefly a database-multiplicity effect: three ChEMBL drugs map
to ERBB2 and one maps to PDE5A. It still reflects reproducible Bridge gene ranks,
but the conversion from two recurrent genes to four recurrent drug IDs amplifies
the apparent breadth.

## Comparison with differential expression

DE's frozen mean pairwise drug Jaccard is 0.182. Only pacritinib appears in all
three studies, but its supporting target changes: ACVR1 in GSE113165 and
GSE211204, IRAK1 in GSE234465. Consequently DE has one all-study recurrent drug
and **zero** target genes recurrent across all three studies. At the two-study
level, ACVR1 supports both momelotinib and pacritinib in GSE113165/GSE211204,
and EGLN3 supports vadadustat in GSE113165/GSE234465. RET and KCNK2 each fan out
to five drugs within only one study, illustrating the same target-to-many-drugs
amplification without cross-study recurrence.

Bridge therefore has stronger target-level recurrence than DE (two versus zero
strict recurrent genes), but the large drug-level effect size overstates its
biological breadth.

## GO/pathway analysis and muscle interpretation

The strict recurrent set has only two genes. No GO Biological Process or KEGG
term passes BH FDR 0.05 when correction includes the full eligible term library
(best strict-set q = 0.420). Descriptively, ERBB2 annotates ERBB/EGF, PI3K-AKT,
MAPK, growth, and membrane-localization terms; PDE5A annotates cyclic-nucleotide
signaling. Adding the study-specific PDE10A produces suggestive but non-significant
cAMP/PKA (`q = 0.086`) and purine-metabolism (`q = 0.054`) terms. These are
database annotations driven by one or two genes, not evidence of pathway-wide
enrichment.

The biology is plausible but narrow. ERBB2/neuregulin signaling is present in
human skeletal muscle and is connected to myoblast survival, regeneration, and
neuromuscular-junction biology. PDE5A/PDE10A point to cyclic-nucleotide control,
which can intersect vascular, metabolic, and nitric-oxide/cGMP responses relevant
to loading. However, neither target is a canonical broad unloading/atrophy
program in this result, and the module does not recurrently recover a larger set
of proteostasis, autophagy, FOXO, mitochondrial, or contractile targets through
ChEMBL.

## Conclusion

The 0.714 result is best described as **a real but very low-dimensional Bridge
target recurrence amplified by ChEMBL target multiplicity**, not broad drug or
pathway convergence. ERBB2 and PDE5A are reproducibly high by absolute Bridge
rank, which is more target-coherent than DE, but three ERBB2 inhibitors should
not be counted as three independent biological discoveries. None of the drug
enrichments is FDR-significant, signs are inconsistent, and this direction-free
analysis does not establish therapeutic relevance or efficacy.

