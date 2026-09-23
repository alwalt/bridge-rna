# Evaluation 2 — published and independent drug recovery

This evaluation reuses the frozen top-500 Bridge and DE modules. It did not
rerun attribution, differential expression, Evaluation 1, or the Phase 2
space-condition benchmark.

## Final drug-target universe

The final approximation uses ChEMBL 37 clinical-stage parent drugs
(`max_phase >= 2`; cells, genes, and vaccine components excluded). Targets
are human protein components with a canonical HGNC symbol. Evidence is either
a curated ChEMBL drug mechanism or a high-confidence human single-protein
binding assay (`confidence_score = 9`, pChEMBL ≥ 6, valid, non-duplicate).
Complex and family mechanisms are expanded only through ChEMBL's explicit
component membership. Drug–gene edges are parent-normalized and deduplicated.

- Broad graph: 4,389 drugs, 2,046 targets, 16,986 edges.
- Strict endpoint: 366 drugs with >10 targets.
- Restrictive Phase 2 comparator: 985 drugs, 323 targets, 1,344 edges; 1 drug with >10 targets.
- de Weerd DrugBank setup: 328 tested drugs and 16,600-protein background.
- Fisher background here: the fixed 15,165 canonical genes, identical for
  Bridge and DE; BH correction is performed separately within each
  disease × method family of 366 eligible drugs.

The 366-drug eligible set is much closer in scale to DrugBank's 328 than the
restrictive graph, but it is not an exact replication: ChEMBL bioactivity and
component-expanded mechanisms are not equivalent to DrugBank target curation,
and the gene backgrounds differ.

Target-count quantiles (all drugs with edges): 0%=1, 25%=1, 50%=1, 75%=3, 90%=9, 95%=14, 99%=37.12, 100%=226

## Published drugs

`Strict rank/p/q` applies only to >10-target drugs. `Audit rank/p/q` includes
all drugs with at least one mapped target and is descriptive only. A ChEMBL
entity without a qualifying canonical protein edge is not counted as a Bridge
failure.

| Disease | de Weerd drug (#) | Target status | Targets | Method | Strict rank | Overlap | Strict p/q | Audit rank/p/q |
|---|---|---|---:|---|---:|---|---:|---:|
| Multiple sclerosis — active lesion | muromonab (1) | mapped, ≤10 | 3 | Bridge | ineligible | — | —/— | 427/1/1 |
| Multiple sclerosis — active lesion | muromonab (1) | mapped, ≤10 | 3 | DE | ineligible | CD3E;CD3G | —/— | 6/0.00318/0.653 |
| Multiple sclerosis — active lesion | ibrutinib (2) | eligible | 32 | Bridge | 17 | ERBB2;FGR | 0.285/1 | 261/0.285/1 |
| Multiple sclerosis — active lesion | ibrutinib (2) | eligible | 32 | DE | 7 | BLK;ITK;LCK;TEC | 0.0202/0.938 | 35/0.0202/0.653 |
| Multiple sclerosis — active lesion | daclizumab (3) | mapped, ≤10 | 3 | Bridge | ineligible | — | —/— | 427/1/1 |
| Multiple sclerosis — active lesion | daclizumab (3) | mapped, ≤10 | 3 | DE | ineligible | IL2RG | —/— | 285/0.0957/1 |
| Multiple sclerosis — active lesion | zanubrutinib (4) | eligible | 15 | Bridge | 11 | ERBB2;FGR | 0.0858/1 | 127/0.0858/1 |
| Multiple sclerosis — active lesion | zanubrutinib (4) | eligible | 15 | DE | 1 | BLK;ITK;LCK;TEC | 0.00119/0.284 | 3/0.00119/0.653 |
| Multiple sclerosis — active lesion | alemtuzumab (5) | entity, no eligible gene edge | 0 | Bridge | ineligible | — | —/— | ineligible/—/— |
| Multiple sclerosis — active lesion | alemtuzumab (5) | entity, no eligible gene edge | 0 | DE | ineligible | — | —/— | ineligible/—/— |
| Crohn's disease — ileum | zinc (1) | absent | 0 | Bridge | ineligible | — | —/— | ineligible/—/— |
| Crohn's disease — ileum | zinc (1) | absent | 0 | DE | ineligible | — | —/— | ineligible/—/— |
| Crohn's disease — ileum | zinc acetate (2) | entity, no eligible gene edge | 0 | Bridge | ineligible | — | —/— | ineligible/—/— |
| Crohn's disease — ileum | zinc acetate (2) | entity, no eligible gene edge | 0 | DE | ineligible | — | —/— | ineligible/—/— |
| Crohn's disease — ileum | dilmapimod (3) | mapped, ≤10 | 1 | Bridge | ineligible | — | —/— | 521/1/1 |
| Crohn's disease — ileum | dilmapimod (3) | mapped, ≤10 | 1 | DE | ineligible | — | —/— | 635/1/1 |
| Crohn's disease — ileum | glucosamine (4) | entity, no eligible gene edge | 0 | Bridge | ineligible | — | —/— | ineligible/—/— |
| Crohn's disease — ileum | glucosamine (4) | entity, no eligible gene edge | 0 | DE | ineligible | — | —/— | ineligible/—/— |
| Crohn's disease — ileum | VX-702 (5) | mapped, ≤10 | 5 | Bridge | ineligible | — | —/— | 521/1/1 |
| Crohn's disease — ileum | VX-702 (5) | mapped, ≤10 | 5 | DE | ineligible | — | —/— | 635/1/1 |
| Systemic lupus erythematosus — blood | gemcitabine (1) | eligible | 13 | Bridge | 114 | — | 1/1 | 452/1/1 |
| Systemic lupus erythematosus — blood | gemcitabine (1) | eligible | 13 | DE | 117 | — | 1/1 | 291/1/1 |
| Systemic lupus erythematosus — blood | enzastaurin (2) | eligible | 13 | Bridge | 114 | — | 1/1 | 452/1/1 |
| Systemic lupus erythematosus — blood | enzastaurin (2) | eligible | 13 | DE | 35 | GSK3A | 0.353/1 | 209/0.353/1 |
| Systemic lupus erythematosus — blood | sunitinib (3) | eligible | 137 | Bridge | 22 | BLK;CAMK1;CAMK1D;CAMK2G;CSF1R;EIF2AK2;LRRK2 | 0.167/1 | 325/0.167/1 |
| Systemic lupus erythematosus — blood | sunitinib (3) | eligible | 137 | DE | 101 | EIF2AK2;FGFR2;MAP2K1;NTRK1 | 0.666/1 | 275/0.666/1 |
| Systemic lupus erythematosus — blood | fostamatinib (4) | mapped, ≤10 | 6 | Bridge | ineligible | — | —/— | 452/1/1 |
| Systemic lupus erythematosus — blood | fostamatinib (4) | mapped, ≤10 | 6 | DE | ineligible | — | —/— | 291/1/1 |
| Systemic lupus erythematosus — blood | cladribine (5) | entity, no eligible gene edge | 0 | Bridge | ineligible | — | —/— | ineligible/—/— |
| Systemic lupus erythematosus — blood | cladribine (5) | entity, no eligible gene edge | 0 | DE | ineligible | — | —/— | ineligible/—/— |

## Bridge versus DE

Coverage is 14/15 as ChEMBL clinical entities, 10/15 with
usable target edges, and 5/15 at the strict >10-target endpoint.
The five strictly eligible reference drugs are ibrutinib, zanubrutinib,
gemcitabine, enzastaurin, and sunitinib.

Across those five drugs, mean Bridge-minus-DE rank percentile is -0.009
(exact paired sign-flip p=0.750).
This is a negligible effect slightly favoring DE, with no evidence that
Bridge preferentially recovers the published drug set. No highlighted
drug is significant after FDR correction for either method.

Disease-specific comparison:

| Disease | Eligible reference drugs | Bridge−DE rank-percentile | Exact p |
|---|---:|---:|---:|
| Multiple sclerosis — active lesion | 2 | -0.0274 | 0.5 |
| Crohn's disease — ileum | 0 | — | — |
| Systemic lupus erythematosus — blood | 3 | 0.00274 | 1 |

## Independently highest-ranked drugs

These are enrichment rankings, not therapeutic recommendations or
direction-aware reversal results.

| Disease | Method | Top five drugs (raw p; FDR) |
|---|---|---|
| Multiple sclerosis — active lesion | Bridge | LUTEOLIN (0.0575; 1); RESVERATROL (0.0666; 1); ANG1005 (0.076; 1); ANVATABART OPADOTIN (0.076; 1); DISITAMAB VEDOTIN (0.076; 1) |
| Multiple sclerosis — active lesion | DE | ZANUBRUTINIB (0.00119; 0.284); APITOLISIB (0.00155; 0.284); PICTILISIB (0.00482; 0.588); RUBOXISTAURIN (0.0131; 0.938); CANERTINIB (0.0162; 0.938) |
| Crohn's disease — ileum | Bridge | TOSEDOSTAT (0.00372; 0.766); OCRIPLASMIN (0.00619; 0.766); CURCUMIN (0.00628; 0.766); COLLAGENASE CLOSTRIDIUM HISTOLYTICUM (0.0431; 1); NILOTINIB (0.0479; 1) |
| Crohn's disease — ileum | DE | LUTEOLIN (0.00628; 1); VORTIOXETINE (0.0121; 1); MARIZOMIB (0.0233; 1); NIGULDIPINE (0.0431; 1); RESVERATROL (0.0666; 1) |
| Systemic lupus erythematosus — blood | Bridge | CELECOXIB (0.0145; 1); MOLIBRESIB (0.0179; 1); IBRUTINIB (0.0202; 1); BRIGATINIB (0.026; 1); BMS-690514 (0.0431; 1) |
| Systemic lupus erythematosus — blood | DE | MOLIBRESIB (5.87e-06; 0.00215); PONATINIB (0.0368; 1); VEMURAFENIB (0.0575; 1); CAPIVASERTIB (0.0666; 1); UPROSERTIB (0.0666; 1) |

## Statistical result

Of six disease × method families, 1 drug results pass FDR < 0.05.
- Systemic lupus erythematosus — blood, DE: MOLIBRESIB, overlap 22/226, raw p=5.87e-06, FDR=0.00215.

The sole FDR-significant result is not one of de Weerd's highlighted
drugs. No Bridge drug enrichment passes FDR. Target enrichment does not
establish drug efficacy, direction of effect, or expression reversal.

## Exact-replication limitation

No authorized DrugBank export is available. The redistributed object in
the paper's code materials does not establish downstream data rights.
Consequently this is a reproducible ChEMBL approximation, not an exact
DrugBank replication. Exact replication requires an authorized versioned
DrugBank export and the authors' precise drug/protein identifier mapping.
