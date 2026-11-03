# Expanded ChEMBL Phase 2 sensitivity and Bridge–DE complementarity

This is a new sensitivity layer over the frozen Phase 2 top-500 modules.
Expression preprocessing, embeddings, condition vectors, Integrated
Gradients, differential expression, and module construction were not rerun.

## Expanded-universe drug convergence

The Evaluation 2 ChEMBL 37 graph (4,389 drugs, 2,046 targets, 16,986
edges) was filtered to the 366 drugs with >10 targets. Enrichment uses
right-sided hypergeometric tests over the fixed 15,165-gene canonical
background with BH correction. Convergence is the original mean of three
pairwise top-25 nonzero-overlap drug-list Jaccards.

| Condition | Expanded Bridge | Expanded DE | Expanded difference | exact p | Restrictive Bridge | Restrictive DE |
|---|---:|---:|---:|---:|---:|---:|
| Radiation | 0.064 | 0.074 | -0.009 | 0.750 | 0.155 | 0.042 |
| Bone loss | 0.097 | 0.035 | 0.062 | 0.250 | 0.142 | 0.181 |
| Muscle atrophy | 0.164 | 0.007 | 0.157 | 0.250 | 0.714 | 0.182 |

The expanded resource changes the headline substantially. Muscle
remains the strongest Bridge condition and Bridge remains far above DE
(0.164 vs 0.0068), but Bridge's absolute muscle Jaccard falls 77% from
0.714. Thus the original magnitude was highly dependent on the sparse
direct-mechanism graph; the qualitative Bridge>DE ordering survives.

Three Bridge drugs recur across all muscle datasets: METFORMIN, BMS-690514, PENTOXIFYLLINE.
Their same-gene recurrence across all three datasets comprises ERBB2, MAP3K20, NDUFS6, PDE4B, PDE5A.
This broadens the strict universal structure beyond ERBB2/PDE5A to
mitochondrial complex-I (NDUFS6), MAP3K20, and PDE4B, but remains a
compact five-target/three-drug structure rather than a broad drug program.

## Bridge/DE complementarity

| Condition | Partition | Recurs in another dataset | OSDR Jaccard | External overlaps | minimum external q | Targetable | Mean deletion loss |
|---|---|---:|---:|---:|---:|---:|---:|
| Bone loss | Bridge-only | 0.423 | 0.173 | 148 | 1.18e-06 | 0.166 | 0.1130 |
| Bone loss | DE-only | 0.071 | 0.017 | 86 | 0.0322 | 0.132 | 0.0558 |
| Bone loss | Shared | 0.031 | 0.011 | 13 | 0.132 | 0.151 | 0.0142 |
| Muscle atrophy | Bridge-only | 0.503 | 0.175 | 59 | 5.16e-05 | 0.161 | 0.0491 |
| Muscle atrophy | DE-only | 0.197 | 0.020 | 33 | 0.0708 | 0.168 | 0.0087 |
| Muscle atrophy | Shared | 0.210 | 0.019 | 10 | 0.301 | 0.149 | 0.0264 |
| Radiation | Bridge-only | 0.380 | 0.129 | 1 | 0.91 | 0.164 | 0.0283 |
| Radiation | DE-only | 0.085 | 0.011 | 4 | 0.376 | 0.138 | 0.0092 |
| Radiation | Shared | 0.125 | 0.009 | 1 | 0.366 | 0.184 | 0.0243 |

Across all conditions, Bridge-only genes recur in another independent
dataset at 0.435 on average, versus 0.117 for DE-only genes. Their mean
terrestrial–OSDR Jaccard is 0.159 versus 0.016. Targetability is similar
but slightly higher for Bridge-only genes (0.163 vs 0.146).

Observed partition deletion is also representation-consistent: Bridge-only
masking changes the frozen projection by 0.0635 on average (0.0143 per
100 genes), versus 0.0246 (0.00549 per 100 genes) for DE-only masking.
Shared genes have the largest loss per gene. These are effect sizes, not
partition-level significance tests; the frozen Phase 2 whole-module
expression-matched deletion null remains the inferential control.

## Repeated Bridge-only biology

| Condition | Recurrent genes | Present in all 3 | Includes OSDR | External support | ChEMBL targets | Targeted by eligible drug | Top pathway | adjusted p |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| Bone loss | 261 | 51 | 213 | 32 | 44 | 25 | extracellular structure organization | 6.95e-13 |
| Muscle atrophy | 270 | 76 | 167 | 19 | 43 | 24 | Cytoskeleton in muscle cells | 2.01e-24 |
| Radiation | 212 | 33 | 143 | 0 | 37 | 18 | epithelial cell differentiation | 0.000318 |

The clearest condition-specific evidence is:

- Bone: Bridge-only genes are externally enriched in all three datasets
  after FDR correction and converge on extracellular-matrix organization.
  Recurrent Bridge-only drugs include luteolin, ocriplasmin, and
  collagenase *C. histolyticum*; these are target recurrences, not efficacy claims.
- Muscle: Bridge-only genes are enriched for the skeletal-muscle-atrophy
  reference in all three datasets and converge on muscle cytoskeleton.
  Repeated genes include LAMB2, ITGA7, DES, NEB, TPM2, ACTA1, ERBB2,
  PDE5A, and PDE4B. BMS-690514 and pentoxifylline recur specifically from
  Bridge-only partitions.
- Radiation: Bridge-only recurrence and pathway coherence are strong, but
  the narrow 39-gene canonical DisGeNET Radiation Damage reference provides
  no corroborating Bridge-only overlap. External support is therefore
  unresolved rather than negative evidence against the representation.

## Answer to the central question

Qualified yes for bone and muscle: Bridge prioritizes gene sets that DE does
not, that recur much more strongly across independent and OSDR datasets,
carry frozen-representation signal, match external condition genes, form
coherent pathways, and contain recurrent druggable targets. The conclusion
is not universal (radiation lacks adequate external-reference confirmation),
and pharmacological actionability is modest: no direction-aware reversal was
tested, expanded-universe drug convergence is smaller than the restrictive
result, and target enrichment is not evidence of therapeutic efficacy.
