# BulkFormer–TCGA gap audit

Audit date: 2026-09-22

This audit was written before running or modifying any TCGA benchmark pipeline. It records the state of the existing benchmark and the protocol information recoverable from the final 2026 Cell Systems release, its official repository, and its released data. Where the final release does not document an implementation detail, it is explicitly marked as unresolved rather than inferred from the 2025 preprint.

## Existing Bridge benchmark

The benchmark already implements two locally paired tasks on one deterministic primary-tumor expression aliquot per TCGA patient:

- Five-cohort classification: BLCA, BRCA, GBM+LGG (reported as `GBMLGG`), LUAD, and UCEC; 3,227 patients. Frozen Bridge 45.6M, BulkFormer-50M, BulkFormer-147M, and full-expression representations use identical patient-level splits. The primary head is an MLP `[256, 128]`; secondary PCA-128 logistic-regression and linear-SVM probes are retained. Metrics are macro F1, weighted F1, and balanced accuracy.
- Pan-cancer overall survival: 9,668 patients with positive follow-up and a valid PanCanAtlas OS event indicator across 32 represented TCGA projects. Frozen Bridge 45.6M, BulkFormer-50M, BulkFormer-147M, and full expression use identical patient-level splits. The primary head is an MLP `[512, 256]` optimized with Cox partial likelihood; secondary PCA-128 Cox probes are retained. Metrics are pooled, sample-count-weighted within-cohort, and macro within-cohort C-index.

The existing benchmark also contains:

- five deterministic 80/20 patient splits (seeds 0–4), with 12.5% of the training partition used for validation by the primary MLP heads;
- native model inputs: Bridge uses its ordered 15,165-gene `log1p(TPM)` space; BulkFormer uses its ordered 20,010-gene `log1p(TPM)` space; the full-expression head uses all 25,150 TCGA features as `log1p(CPM)`;
- complete cached matrices and sample embeddings for all 9,816 selected primary-tumor patients (Bridge 512-D, BulkFormer-50M 259-D, BulkFormer-147M 643-D);
- an exploratory one-seed Bridge mean-pooling versus learned-attention pooling ablation;
- published BulkRNABert and conventional reference values kept separate from local results;
- an executed notebook, compact result tables, figures, logs, and provenance.

No existing result needs to be rerun for the requested extension.

## Final BulkFormer release evidence

Authoritative identifiers:

- Kang et al., *Cell Systems* 17(7), 101657 (2026), DOI `10.1016/j.cels.2026.101657`, PII `S2405471226001390`.
- Official repository: `https://github.com/KangBoming/BulkFormer`, inspected at commit `5bcf5b99e799cd55f41e4d6fe8200b2cdf3d9589` (2026-07-03).
- Released data record: Zenodo `10.5281/zenodo.15744294` (2025-12-11 release used by the final repository).

The final official README reports `TCGA classification (weighted F1) = 0.907`. The final repository does **not** release downstream classifier/training code or split definitions. It releases model/inference code and an embedding-extraction notebook. The Zenodo record releases a single `TCGA_survival.h5ad` object, not a separate TCGA classification object. Direct inspection found 10,429 sample rows, 9,679 unique 12-character patient barcodes, 33 cancer labels, 750 repeated-patient rows, and 709 sample-type-code-11 normal aliquots. It contains both binary `OS` and continuous `OS.time` fields. Therefore the released data are sample-level and are not an exact patient-level comparison cohort.

The released labels are ACC, BLCA, BRCA, CESC, CHOL, COAD, DLBC, ESCA, GBM, HNSC, KICH, KIRC, KIRP, LAML, LGG, LIHC, LUAD, LUSC, MESO, OV, PAAD, PCPG, PRAD, READ, SARC, SKCM, STAD, TGCT, THCA, THYM, UCEC, UCS, and UVM. GBM and LGG are separate.

The final feature-extraction notebook establishes native count-to-`log1p(TPM)` preprocessing, the 20,010-gene vocabulary, missing-gene fill value −10, and support for mean/max/median token pooling. Its example uses mean pooling. The 2025 preprint instead stated max pooling for sample-level tasks. Until the final paper/supplement or released object resolves this difference, pooling for the reported 0.907 must be treated as ambiguous.

The 2025 preprint is used only to identify possible protocol continuity and changes, not as final authority. It described:

- TCGA classification across 33 cancer types, PCA of frozen sample embeddings, random forest, 10-fold cross-validation, weighted F1 = 0.833;
- prognosis as alive/dead classification for approximately 10,000 patients across 33 cancer types, PCA, random forest, 10-fold cross-validation, AUROC = 0.747 and AUPRC = 0.549.

The final README changes TCGA weighted F1 from 0.833 to 0.907 and expands the baseline set, confirming that the final classification result is not simply the preprint result. The final abstract retains prognosis modeling as one of five tasks, but the final README table does not report prognosis metrics. The released object's fields are consistent with the preprint's alive/dead task but do not prove the final headline endpoint. Because final downstream code and supplementary methods were not retrievable from the available public endpoints, the precise final prognosis implementation and values remain unresolved.

## Gap table (pre-execution)

| Component | Final BulkFormer | Existing Bridge benchmark | Match? | Action |
| --- | --- | --- | --- | --- |
| TCGA source | GDC/TCGA; exact release unresolved in public code | Local `data/tcga/tcga_matrix.h5`; GDC metadata | Partial | Record source hashes and compare released sample identifiers |
| Cancer cohorts/classes | Final result is pan-cancer; preprint used 33 types; final exact label set pending released-data inspection | Five labels for classification; 32 projects in selected primary tumors | No | Add pan-cancer classification after freezing eligible labels |
| Patient/sample count | Released object: 10,429 samples, 9,679 unique patients | 9,816 primary-tumor patients total; 3,227 in five-class task | No | Save a 33-class patient-level local manifest |
| Sample selection | Released object includes 709 code-11 normals and other non-primary codes | Primary Tumor only; one sorted aliquot per patient | No | Preserve patient-level local policy and document divergence |
| Aliquot handling | 750 rows repeat a patient barcode | One aliquot per patient | No | Do not present 0.907 as paired with local results |
| Expression preprocessing | Final inference utility: raw counts → natural `log1p(TPM)` | Same for both FMs; full baseline is `log1p(CPM)` | FM match | Reuse matrices; label full-expression transform clearly |
| Vocabulary | Native BulkFormer 20,010 protein-coding genes | Native Bridge 15,165; native BulkFormer 20,010 | Yes | Preserve native vocabularies |
| Sample pooling | Final notebook example mean; preprint methods max; final benchmark setting unresolved | Mean for cached FM embeddings | Unknown | If parity setting is resolved and differs, infer only the missing pooled representation |
| Embedding dimensions | Model dependent; final 147M is 643 with appended summary features; 50M is 259 | 512 / 259 / 643 | Yes | Reuse caches where pooling matches |
| Dimensionality reduction | Preprint: PCA to a common feature space; final details unresolved | PCA-128 only for secondary probes; none for primary MLP | Partial | Fit parity PCA on training folds only once dimensions are documented |
| Downstream classifier | Preprint: random forest; final code not released | MLP `[256,128]` primary, LR/SVM secondary | No | Add separate parity RF analysis; retain existing MLP task |
| Splitting | Preprint: 10-fold CV; final split details not released | Five seeded 80/20 splits | No | Add frozen reproducible patient-level folds; distinguish reproduction from local common splits |
| Seeds/folds | Preprint: 10 folds; final seed/repeats unresolved | Five seeds | No | Use documented folds only; record unresolved random seed |
| Classification metrics | Final: weighted F1 0.907; other metrics not in README | Macro F1, weighted F1, balanced accuracy | Partial | Report all local metrics; keep 0.907 literature-only |
| Prognosis endpoint | Final abstract retains prognosis; exact final formulation pending; preprint was alive/dead | Proper OS `(time,event)` | Likely no | Retain Cox benchmark; add a separately labeled alive/dead parity task only if final evidence confirms it |
| Censoring | Preprint binary alive/dead ignores time/censoring structure | Explicit event and follow-up time | No | Never equate AUROC/AUPRC with C-index |
| Prognosis metrics | Preprint AUROC/AUPRC; final values unresolved | Three C-index summaries | No | Add separate AUROC/AUPRC table only if final binary task is confirmed |
| Pretraining exposure | BulkFormer pretraining set is released without row-level sample provenance; possible TCGA overlap cannot be excluded | Bridge uses its documented manifest; BulkFormer caches use released checkpoints | Not uniform | State exposure limitation; do not call strict external validation |

## Implemented minimal extension

1. Audited the final released object and saved machine-readable and human-readable provenance, including unresolved details.
2. Added 33-class pan-cancer classification on 9,942 unique patients: the existing 9,816 primary-tumor patients plus 126 LAML peripheral-blood patients.
3. Added a separately labeled alive/dead prognosis reconstruction on 9,921 patients with OS status; the existing time-to-event Cox benchmark is unchanged.
4. Reused all 9,816 cached embeddings per model and inferred only the 126 missing LAML patients.
5. Used identical patient-level folds across Bridge, both BulkFormer sizes, raw expression, and PCA controls. PCA is fitted within each training fold.
