# LUAD → ARCHS4 leakage and provenance audit

All 200 method-specific LUAD centroid top-100 rows were audited: 100
Bridge-512 hits and 100 Hallmark-50 hits, representing 181 unique GSMs and 51
individual GSE accessions. GEO quick-series records were checked for TCGA
barcodes, explicit TCGA import/reuse language, and affirmative evidence of
original cohort collection or experimental generation. Local GSM metadata were
also checked against every TCGA sample UUID and patient identifier in the frozen
query cohort.

## Result

| Classification | Hit rows | GSE accessions |
|---|---:|---:|
| Independent GEO cohort with affirmative evidence | 128 | 18 |
| Ambiguous/insufficient metadata | 72 | 33 |
| TCGA-derived/reused | **0** | **0** |

No exact TCGA query identifier duplicates, TCGA barcodes, or explicit
TCGA-import statements were found. Absence of TCGA language was never treated
as proof of independence; records lacking affirmative provenance remain
ambiguous.

All method-specific centroid top-five hits have affirmative original-cohort
evidence. Examples include GSE120622 (81 lung-cancer and 19 adjacent tissues
RNA-sequenced), GSE256091 (141 never-smoker LUAD tissues selected for a
proteogenomic cohort), GSE166720/GSE166722 (53 resected early-stage LUAD tumors
under an IRB-approved protocol), GSE181043 (an independent 109-patient Chinese
LUSC cohort), and GSE81089 (199 surgically treated NSCLC patients at Uppsala
University Hospital). GSE81089 mentions TCGA only as an independent external
comparison; its GEO samples are explicitly described as the Uppsala cohort.

Because no hit was classified as TCGA-derived/reused, exclusion does not alter
the focused LUAD top-five interpretation. The saved exclusion table is identical
to the original top five for both methods.

Machine-readable evidence is in `luad_centroid_top100_leakage_audit.csv`,
`luad_geo_series_provenance_audit.csv`, and
`luad_leakage_audit_summary.json`. Each record includes the GEO URL, evidence
excerpt, classification flags, and exact-identifier check.

