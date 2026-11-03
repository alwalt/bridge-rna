# Locked de Weerd replication cohorts

Lock date: 2026-09-21. These contrasts were resolved before Bridge attribution,
DE, disease-gene testing, or drug enrichment was run.

## Resolution evidence

The de Weerd supplement names the three contrasts as:

- `ms_modalyser_al_vs_control`, displayed as **MS Active Lesion**;
- `E-GEOD-101794_chron_a1a_vs_control`, displayed as **Crohn Ileum 1**; and
- `E-GEOD-72509_sle_vs_control`, displayed as **Systemic Lupus Erythematosus Blood**.

These labels were matched to the cited primary studies and GEO sample-level
metadata. In particular, `chron_a1a` resolves which of the two age-stratified
GSE101794 contrasts de Weerd called Crohn Ileum 1.

The original 2019 MS article was retracted after a row omission shifted ten
sample labels. The authors corrected the labels, reanalyzed the data, and
republished the corrected study (PMID 31829262); de Weerd cites that corrected
publication. This lock uses the current GSE138614 GEO metadata/count matrix and
the corrected 98-sample design, not the obsolete label file from the retracted
article.

## Frozen contrasts

| Disease | Source/tissue | Case | Control | Samples | Independent donors | Canonical coverage | ARCHS4 exposure |
|---|---|---|---|---:|---:|---:|---:|
| MS active lesion | GSE138614, brain white matter | 16 Active (AL) lesion pieces | 25 non-neurological white-matter pieces | 41 | 7 case, 5 control | 15,069/15,165 (99.4%) | 35 train, 4 validation, 2 absent |
| Crohn ileum | GSE101794, terminal ileum | 61 Crohn, Paris A1a | 13 non-IBD, Paris A1a | 74 | 61 case, 13 control | 10,738/15,165 (70.8%) | 54 train, 17 validation, 3 absent |
| SLE blood | GSE72509, whole blood | 99 SLE | 18 healthy | 117 | 99 case, 18 control | 13,874/15,165 (91.5%) | 88 train, 28 validation, 1 absent |

Coverage counts genes mapped into the canonical input, while expressed counts
are 14,991 for MS, 10,738 for Crohn, and 13,821 for SLE. Crohn's lower coverage
reflects the deposited 13,151-row symbol-level TPM files; missing canonical genes
are set to zero under the established input contract, not imputed.

## Covariates and independence

- **MS:** several tissue pieces come from the same donor. Donor is the unit of
  independence. Bridge condition/attribution consensus and DE therefore weight
  donors, not pieces; exercise-style case-control pairing is not invented. GEO
  does not provide sample-level age or sex in the series metadata.
- **Crohn:** cases and controls are restricted to the same Paris A1a age stratum.
  Case ages are 6.2–9.8 years (median 8.8) and control ages are 6.6–9.9 (median
  8.5). Cases comprise 33 male/28 female participants; controls 10 male/3 female.
  Each biopsy is from a different participant. The primary contrast remains the
  unadjusted de Weerd case-control contrast.
- **SLE:** each whole-blood sample is a different donor. GEO records anti-Ro
  class (24 high, 23 medium, 52 none among cases) and interferon-signature metric
  (75 high, 24 low), but not age/sex in the series metadata. These disease
  strata are retained, not used to redefine de Weerd's all-SLE contrast.

## Expression sources

- GSE138614: deposited Ensembl count matrix; counts were mapped to HGNC,
  converted to TPM with the repository's frozen gene lengths, then transformed
  with natural `log1p`.
- GSE101794: per-sample deposited gene-symbol TPM files; duplicate symbols were
  summed, aligned, and transformed with natural `log1p`.
- GSE72509: deposited gene-symbol RPKM matrix; duplicate symbols were summed,
  each sample was rescaled from RPKM to TPM, aligned, and transformed with
  natural `log1p`.

All final inputs contain the canonical 15,165 columns in the frozen order.

## Generalization boundary

Because 226 of 232 selected samples occur in ARCHS4 train/validation, this is a
reproducibility and biological-recovery evaluation. It must not be presented as
evidence of generalization to unseen studies. The exposure status is analyzed
as provenance, not used to remove samples from the published contrasts.
