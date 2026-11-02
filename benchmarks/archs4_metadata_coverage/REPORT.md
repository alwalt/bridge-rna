# GEOMeta and Gemma coverage of the Bridge ARCHS4 corpus

Audit date: 2026-09-25. This is a metadata audit only; no embeddings or retrieval indices were changed.

## Executive answer

The frozen Bridge ARCHS4 embedding corpus contains **940,455 unique GSMs**: **510,709 human** and **429,746 mouse**. Every row is unique and `global_index` is contiguous from 0 to 940,454, so row *i* of the embedding memmap is identified by row *i* of the location table.

For the human corpus, the observed union of the currently downloadable GEOMeta CSV release and successfully retrieved Gemma sample records covers **444,541/510,709 GSMs (87.04%)**. The remaining measured gap is **66,168 GSMs (12.96%)**. GEOMeta supplies much broader human GSM coverage; Gemma adds ontology-backed fields and mouse coverage but is a selective curated collection rather than an ARCHS4-wide catalogue.

**Recommendation:** build the next implementation as a versioned metadata sidecar keyed by GSM and embedding `global_index`. Ingest GEOMeta first for broad human coverage, overlay Gemma's structured sample characteristics and experimental factors without discarding provenance, and reserve Jev/Lydia for the union-unmatched set plus field-level gaps. Before production integration, manually review the category mapping and the records marked as cross-resource conflicts; do not collapse divergent values into one unqualified label.

## 1. Exact Bridge source artifacts

- Embeddings: `embeddings/archs4/sample_embeddings.float16.mmap` (940,455 × 512 float16; 963,025,920 bytes).
- Row map: `embeddings/archs4/sample_locations.parquet` (`global_index`, shard and row location, GSM, species ID).
- ARCHS4 metadata: `data/manifests/archs4_sample_metadata_v2.5.parquet` (GSM, GSE, organism, title, source, characteristics, platform/library fields, date and single-cell probability).
- Manifest: `embeddings/archs4/embedding_manifest.json`; it records ARCHS4 v2.5-derived natural `log1p_tpm`, a 512-dimensional frozen-model representation, and 940,455 samples.

Human and mouse occupy one combined memmap, distinguished by `species_id` (0 human, 1 mouse); they are not separate embedding files. The location and ARCHS4 metadata tables join one-to-one by GSM, and each GSM occurs exactly once.

## 2. GEOMeta release and schema

Official resources: [dataset](https://huggingface.co/datasets/binchenlab/GEOMeta), [source](https://github.com/Bin-Chen-Lab/GEOMeta), and the 2026 preprint DOI `10.64898/2026.08.19.745739`.

Only the three metadata CSVs were downloaded. Together they contain 474,373 rows, **473,688 unique GSMs** and **18,477 GSEs**, all human ARCHS4-derived records. Their exact 12 columns are: `GSE_ID`, `GSM_ID`, `year`, `contact_country`, `instrument_model`, `RNA_Library`, `Exp_Setting`, `GSE_Pert`, `Disease`, `Organ`, `Gender`, and `Age_Group`. They are directly GSM-keyed. The repository also publishes three expression-linked H5AD files totaling roughly 124 GB; those were not downloaded.

Release caveat: the preprint says 594,989 GSM–GSE rows / 594,304 unique GSMs and describes richer perturbation, dose and duration fields. Those records/fields are **not present in the pinned public metadata CSVs** (HF revision `d0cd1e117143b32a392bbe3130e1c4a2a3c8e740`). Results below therefore describe reproducible, observed downloadable data rather than the manuscript headline.

## 3. Gemma resources and observed schema

Official resources: [Gemma](https://gemma.msl.ubc.ca/), [REST v2 documentation](https://gemma.msl.ubc.ca/resources/restapidocs/), and Lim et al. 2021 (`doi:10.1093/database/baab006`). The live API reported 23,741 catalogue records; after accession de-duplication this audit observed 23,103 GSE accessions: 9,626 human, 11,911 mouse and 1,566 rat. Bridge-overlapping GSE sample records were requested from `/rest/v2/datasets/{GSE}/samples`; no expression data were downloaded. Of 8,508 catalogue-overlapping Bridge GSEs, **432** consistently returned HTTP 404 from that endpoint and are conservatively treated as unresolved rather than matched.

Gemma exposes GEO GSE/GSM accessions, study- and sample-level ontology-backed characteristics, and sample experimental-factor values. Observed categories include tissue/organism part, disease, cell type, sex, age/developmental stage, strain, genotype, treatment, dose/duration-like fields, and experimental factors. Category spelling is heterogeneous, so the machine-readable observed schema records every category and its explicit audit mapping.

Top observed mapped categories:

| source | category | occurrences | mapped_fields |
| --- | --- | --- | --- |
| experimental_factor | treatment | 90237 | treatment_perturbation |
| experimental_factor | genotype | 84945 | genotype |
| experimental_factor | organism part | 37712 | tissue |
| experimental_factor | timepoint | 31573 | duration |
| experimental_factor | cell type | 24038 | cell_type |
| experimental_factor | biological sex | 23973 | sex |
| experimental_factor | disease | 23860 | disease \| disease_status |
| experimental_factor | developmental stage | 19252 | age_development |
| experimental_factor | strain | 6747 | strain |
| experimental_factor | phenotype | 4549 | experimental_factor_group |
| experimental_factor | disease model | 3805 | disease \| disease_status |
| experimental_factor | age | 3727 | age_development |
| experimental_factor | growth condition | 1598 | experimental_factor_group |
| experimental_factor | disease staging | 1121 | disease \| disease_status |
| experimental_factor | dose | 381 | dose |
| experimental_factor | study design | 34 | experimental_factor_group |
| sample_characteristic | tissue | 77940 | tissue |
| sample_characteristic | genotype | 69526 | genotype |
| sample_characteristic | treatment | 69159 | treatment_perturbation |
| sample_characteristic | cell type | 68999 | cell_type |
| sample_characteristic | strain | 55107 | strain |
| sample_characteristic | age | 46010 | age_development |
| sample_characteristic | Sex | 36861 | sex |
| sample_characteristic | organism part | 18772 | tissue |
| sample_characteristic | gender | 8999 | sex |
| sample_characteristic | genotype/variation | 8503 | genotype |
| sample_characteristic | timepoint | 6409 | duration |
| sample_characteristic | developmental stage | 5326 | age_development |
| sample_characteristic | time point | 5200 | duration |
| sample_characteristic | disease state | 5047 | disease \| disease_status |

## 4. Actual GSM and GSE coverage

| level | species | source | total | matched | unmatched | coverage_pct |
| --- | --- | --- | --- | --- | --- | --- |
| GSM | human | geometa | 510709 | 438119 | 72590 | 85.79 |
| GSM | human | gemma | 510709 | 72952 | 437757 | 14.28 |
| GSM | human | union | 510709 | 444541 | 66168 | 87.04 |
| GSM | human | intersection | 510709 | 66530 | 444179 | 13.03 |
| GSE | human | geometa | 30482 | 21588 | 8894 | 70.82 |
| GSE | human | gemma | 30482 | 3729 | 26753 | 12.23 |
| GSE | human | union | 30482 | 21994 | 8488 | 72.15 |
| GSE | human | intersection | 30482 | 3323 | 27159 | 10.9 |
| GSM | mouse | geometa | 429746 | 0 | 429746 | 0.0 |
| GSM | mouse | gemma | 429746 | 104503 | 325243 | 24.32 |
| GSM | mouse | union | 429746 | 104503 | 325243 | 24.32 |
| GSM | mouse | intersection | 429746 | 0 | 429746 | 0.0 |
| GSE | mouse | geometa | 30423 | 0 | 30423 | 0.0 |
| GSE | mouse | gemma | 30423 | 5907 | 24516 | 19.42 |
| GSE | mouse | union | 30423 | 5907 | 24516 | 19.42 |
| GSE | mouse | intersection | 30423 | 0 | 30423 | 0.0 |

Gemma matching is conservative: a GSM counts only when returned in the sample endpoint, not merely because its GSE exists in the catalogue. GSE matching likewise uses successfully observed sample records. Comma-separated ARCHS4 series memberships are exploded for GSE denominators.

## 5. Field-level completeness

Percent below is of the full Bridge species corpus, not merely matched samples. The source table also supplies matched-sample denominators and percentages.

| species | field | gemma | geometa | union |
| --- | --- | --- | --- | --- |
| human | age_development | 3.44 | 15.3 | 16.87 |
| human | cell_type | 7.77 | 0.0 | 7.77 |
| human | disease | 4.28 | 57.94 | 59.09 |
| human | disease_status | 4.29 | 57.94 | 59.1 |
| human | dose | 0.87 | 0.0 | 0.87 |
| human | duration | 3.24 | 0.0 | 3.24 |
| human | experimental_factor_group | 14.11 | 85.79 | 87.02 |
| human | genotype | 5.82 | 0.0 | 5.82 |
| human | sex | 2.58 | 27.88 | 28.18 |
| human | strain | 0.08 | 0.0 | 0.08 |
| human | tissue | 5.42 | 75.12 | 76.09 |
| human | treatment_perturbation | 7.94 | 85.79 | 86.13 |
| mouse | age_development | 11.52 | 0.0 | 11.52 |
| mouse | cell_type | 7.73 | 0.0 | 7.73 |
| mouse | disease | 1.84 | 0.0 | 1.84 |
| mouse | disease_status | 1.84 | 0.0 | 1.84 |
| mouse | dose | 0.26 | 0.0 | 0.26 |
| mouse | duration | 4.64 | 0.0 | 4.64 |
| mouse | experimental_factor_group | 24.24 | 0.0 | 24.24 |
| mouse | genotype | 15.9 | 0.0 | 15.9 |
| mouse | sex | 7.61 | 0.0 | 7.61 |
| mouse | strain | 14.62 | 0.0 | 14.62 |
| mouse | tissue | 17.75 | 0.0 | 17.75 |
| mouse | treatment_perturbation | 12.67 | 0.0 | 12.67 |

GEOMeta's public CSV has no sample-level cell type, strain, genotype, dose, duration, or detailed treatment term. `GSE_Pert` is retained as a study-level yes/no perturbation indicator and is not presented as a specific treatment. Gemma availability is counted only from semantically mapped characteristic/factor categories; generic free-text `BioSource` is not promoted to structured tissue.

## 6. Agreement where both resources exist

| field | relation | n |
| --- | --- | --- |
| cell_type | missing_both | 30233 |
| cell_type | missing_geometa | 36297 |
| disease | different_vocabulary_or_candidate_conflict | 13491 |
| disease | exact | 280 |
| disease | missing_both | 20946 |
| disease | missing_gemma | 27166 |
| disease | missing_geometa | 2421 |
| disease | normalized | 2226 |
| sex | exact | 2283 |
| sex | missing_both | 47821 |
| sex | missing_gemma | 7004 |
| sex | missing_geometa | 57 |
| sex | normalized | 9329 |
| sex | obvious_conflict | 36 |
| tissue | different_vocabulary_or_candidate_conflict | 17369 |
| tissue | exact | 2336 |
| tissue | missing_both | 8156 |
| tissue | missing_gemma | 33490 |
| tissue | missing_geometa | 2175 |
| tissue | normalized | 3004 |
| treatment_perturbation | missing_gemma | 27746 |
| treatment_perturbation | normalized | 12 |
| treatment_perturbation | not_comparable_granularity | 38772 |

`exact` compares source strings and `normalized` applies case/punctuation normalization plus straightforward sex synonyms. `obvious_conflict` is reserved for direct male/female disagreements. Disease differences remain in `different_vocabulary_or_candidate_conflict` because reference-role factors can coexist with specific diagnoses and require ontology-aware review. GEOMeta's yes/no study perturbation is marked `not_comparable_granularity` against Gemma treatment terms. Tissue granularity is often different (for example organ system versus specific organism part), and Gemma factors may encode roles rather than GEOMeta disease labels. Cell type comparisons are missing-on-GEOMeta because that field is absent from the downloadable CSV schema.

## 7. Ten retrieval-ready examples

| global_index | gsm | species | series_id | geometa_matched | gemma_matched | union_tissue | union_disease | union_disease_status | union_cell_type | union_sex | union_age_development | union_strain | union_genotype | union_treatment_perturbation | union_experimental_factor_group | union_dose | union_duration |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 316470 | GSM4212421 | mouse | GSE141721 | False | True | Spinal cord \| spinal cord | experimental autoimmune encephalomyelitis | experimental autoimmune encephalomyelitis \| disease/case | macrophages \| macrophage | female | 12wk | LysM-Cre | wild type \| wild type genotype | untreated \| reference substance role | disease: experimental autoimmune encephalomyelitis \| cell type: macrophage \| treatment: reference substance role \| organism part: spinal cord \| block: Device=D00796:Run=354:Flowcell=CCF53ANXX:Lane=5 \| genotype: wild type genotype |  |  |
| 316471 | GSM4212422 | mouse | GSE141721 | False | True | Spinal cord \| spinal cord | experimental autoimmune encephalomyelitis | experimental autoimmune encephalomyelitis \| disease/case | macrophages \| macrophage | male | 12wk | GRIP1 fl/fl | GRIP1 -/- \| Homozygous negative  Grip1 [mouse] glutamate receptor interacting protein 1 | untreated \| reference substance role | disease: experimental autoimmune encephalomyelitis \| genotype: Homozygous negative  Grip1 [mouse] glutamate receptor interacting protein 1 \| cell type: macrophage \| treatment: reference substance role \| organism part: spinal cord \| block: Device=D00796:Run=354:Flowcell=CCF53ANXX:Lane=5 |  |  |
| 316472 | GSM4212423 | mouse | GSE141721 | False | True | Spinal cord \| spinal cord | experimental autoimmune encephalomyelitis | experimental autoimmune encephalomyelitis \| disease/case | microglia \| microglial cell | male | 12wk | GRIP1 fl/fl | GRIP1 -/- \| Homozygous negative  Grip1 [mouse] glutamate receptor interacting protein 1 | untreated \| reference substance role | disease: experimental autoimmune encephalomyelitis \| genotype: Homozygous negative  Grip1 [mouse] glutamate receptor interacting protein 1 \| treatment: reference substance role \| organism part: spinal cord \| block: Device=D00796:Run=354:Flowcell=CCF53ANXX:Lane=5 \| cell type: microglial cell |  |  |
| 316473 | GSM4212424 | mouse | GSE141721 | False | True | Spinal cord \| spinal cord | experimental autoimmune encephalomyelitis | experimental autoimmune encephalomyelitis \| disease/case | microglia \| microglial cell | female | 12wk | GRIP1 fl/fl | GRIP1 -/- \| Homozygous negative  Grip1 [mouse] glutamate receptor interacting protein 1 | untreated \| reference substance role | block: Device=D00796:Run=374:Flowcell=CCT1LANXX:Lane=6 \| disease: experimental autoimmune encephalomyelitis \| genotype: Homozygous negative  Grip1 [mouse] glutamate receptor interacting protein 1 \| treatment: reference substance role \| organism part: spinal cord \| cell type: microglial cell |  |  |
| 316476 | GSM4212427 | mouse | GSE141721 | False | True | Spinal cord \| spinal cord | experimental autoimmune encephalomyelitis | experimental autoimmune encephalomyelitis \| disease/case | macrophages \| macrophage | female | 12wk | GRIP1 fl/fl | GRIP1 -/- \| Homozygous negative  Grip1 [mouse] glutamate receptor interacting protein 1 | untreated \| reference substance role | disease: experimental autoimmune encephalomyelitis \| genotype: Homozygous negative  Grip1 [mouse] glutamate receptor interacting protein 1 \| cell type: macrophage \| block: Device=GPL17021 \| treatment: reference substance role \| organism part: spinal cord |  |  |
| 316477 | GSM4212428 | mouse | GSE141721 | False | True | Spinal cord \| spinal cord | experimental autoimmune encephalomyelitis | experimental autoimmune encephalomyelitis \| disease/case | microglia \| microglial cell | female | 12wk | GRIP1 fl/fl | GRIP1 -/- \| Homozygous negative  Grip1 [mouse] glutamate receptor interacting protein 1 | untreated \| reference substance role | disease: experimental autoimmune encephalomyelitis \| genotype: Homozygous negative  Grip1 [mouse] glutamate receptor interacting protein 1 \| block: Device=GPL17021 \| treatment: reference substance role \| organism part: spinal cord \| cell type: microglial cell |  |  |
| 316482 | GSM4212433 | mouse | GSE141721 | False | True | Spinal cord \| spinal cord | reference subject role | reference subject role \| control/reference | microglia \| microglial cell | male | 12wk | GRIP1 fl/fl | GRIP1 -/- \| Homozygous negative  Grip1 [mouse] glutamate receptor interacting protein 1 | untreated \| reference substance role | genotype: Homozygous negative  Grip1 [mouse] glutamate receptor interacting protein 1 \| block: Device=GPL17021 \| treatment: reference substance role \| organism part: spinal cord \| disease: reference subject role \| cell type: microglial cell |  |  |
| 316484 | GSM4212435 | mouse | GSE141721 | False | True | Spinal cord \| spinal cord | reference subject role | reference subject role \| control/reference | microglia \| microglial cell | female | 12wk | GRIP1 fl/fl | GRIP1 -/- \| Homozygous negative  Grip1 [mouse] glutamate receptor interacting protein 1 | untreated \| reference substance role | genotype: Homozygous negative  Grip1 [mouse] glutamate receptor interacting protein 1 \| block: Device=GPL17021 \| treatment: reference substance role \| organism part: spinal cord \| disease: reference subject role \| cell type: microglial cell |  |  |
| 316485 | GSM4212436 | mouse | GSE141721 | False | True | Spinal cord \| spinal cord | reference subject role | reference subject role \| control/reference | microglia \| microglial cell | female | 12wk | GRIP1 fl/fl | GRIP1 -/- \| Homozygous negative  Grip1 [mouse] glutamate receptor interacting protein 1 | untreated \| reference substance role | genotype: Homozygous negative  Grip1 [mouse] glutamate receptor interacting protein 1 \| block: Device=GPL17021 \| treatment: reference substance role \| organism part: spinal cord \| disease: reference subject role \| cell type: microglial cell |  |  |
| 727779 | GSM6062917 | human | GSE201407 | True | True | Brain | Normal | control/normal | iPSC-derived motor neurons | Female | Adults-40s |  | Healthy control \| wild type genotype | Yes | In Vitro |  | 21 d |

Values remain source-faithful and provenance-specific in the machine-readable table. A production view can later choose a display precedence while preserving both originals.

## 8. Union-unmatched GEO audit (25 deterministic random GSMs)

GEO SOFT records were fetched directly for 25 deterministic random union-unmatched GSMs. The flags below are conservative keyword/evidence checks over titles, sources, characteristics, descriptions and protocols—not LLM annotations.

| field | recoverable_pct_in_25_sample_audit |
| --- | --- |
| tissue_recoverable | 96.0 |
| disease_state_recoverable | 40.0 |
| treatment_recoverable | 68.0 |
| experimental_role_recoverable | 24.0 |

| gsm | species | gse | geo_title | geo_source | geo_characteristics | tissue_recoverable | disease_state_recoverable | treatment_recoverable | experimental_role_recoverable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GSM8250089 | mouse | GSE266549 | DO371_GT20_00290_adipose | adipose | tissue: adipose \| diet: 45 percent  fat | True | False | True | False |
| GSM2264597 | mouse | GSE85316 | Tfh | helper T cell | strain: C57BL/6 \| tissue: spleen \| age: 8 weeks \| genotype: wild type | True | True | False | True |
| GSM5666484 | human | GSE187008 | MDAMB231 cells with knockout of OTUD6B at 96h replicate 2 [sample_702] | triple negative metastatic mammary adenocarcinoma | cell_type: triple negative metastatic mammary adenocarcinoma \| cell_line: MDAMB231 \| treatment_type: knockout \| treatment: OTUD6B \| concentration: -- \| time: 96 \| replicate: 2 \| library: FC_04750_S2 \| well: K20 \| barcode: CGGACG | False | True | True | False |
| GSM3561066 | mouse | GSE125007,GSE125008 | WT, baseline rep 2 | mouse liver | strain: C57Bl/6 \| tissue: liver \| genotype: WT \| treatment: baseline \| time: baseline | True | True | True | True |
| GSM2978509 | mouse | GSE110144 | TLR7_24h Rep4 | Splenocytes from OT-1 Mice | strain: OT-1 \| jackson labs: Stock No: 003831 \| cell type: CD8+ T cells \| agent: TLR7 \| time: 24h | True | False | True | False |
| GSM1634467 | mouse | GSE66931,GSE73382 | Hira-KO1_RNA-Seq [Gdf9-Cre+] | MII oocyte, Hira-KO | strain: C57BL/6 \| genotype/variation: Hiraf/f, Gdf9-Cre+ \| cell type: oocyte \| developmental stage: MII | True | False | False | False |
| GSM5923839 | human | GSE197619 | P2 CD4+ T cells | blood | disease: Cutaneous T Cell Lymphoma | True | True | False | False |
| GSM5448400 | mouse | GSE180020 | pDC.IL15.Sp#2 | Plasmacytoid Dendritic Cell | strain: C57BL/6 \| tissue: Spleen \| cell type: Plasmacytoid Dendritic Cell | True | False | True | False |
| GSM2037736 | mouse | GSE76779 | 3C-HFSCs-rep2 | epidermis | cell type: keratinocyte \| cell type: Cultured hair follicle stem cells (HFSCs) \| genotype: wild type | True | True | False | True |
| GSM2895417 | mouse | GSE108328 | FT8_2a_moderate chronic liver injury for 8 weeks | moderate chronic liver injury for 8 weeks | strain background: 129S4 \| genotype/variation: Fah-/- \| treatment: chronic injury 8 weeks \| tissue type: liver | True | False | True | False |
| GSM5448082 | mouse | GSE180012 | 16P_Rep1 | liver_16 h after PHx | strain: C57BL/6 \| gender: male \| treatment: PHx \| age: 8 weeks \| tissue: liver \| time point: 16 h after PHx | True | False | True | False |
| GSM3521687 | mouse | GSE124146 | Ctrl2 | Jejunum | cell type: Enteroid in culture | True | False | True | False |
| GSM4983311 | mouse | GSE123909 | KO_NF_12_Untx_M_2 | Liver | tissue: Liver \| genotype: BMAL1 KO \| feeding (day or night): NF \| zt: 12 \| treatment: Untx \| Sex: M | True | False | True | False |
| GSM7825744 | human | GSE244740 | Sunitinib, Con4, Optimization plate 2, Well H5 [S4881Nr2_H5] | hiPSC-CM | cell line: hiPSC-CM \| genotype: WT \| treatment: Sunitinib, Con4 \| time: 24h | True | True | True | True |
| GSM2185691 | mouse | GSE82180,GSE82182 | Colon GFP+ Rep 1 | sorted from AtohGFP/GFP mice | cell type: Atoh1 GFP+ cells \| genotype: GFP+ \| tissue: colon | True | False | False | False |
| GSM5399370 | mouse | GSE178856 | Lrig1-low cells from colonic tumors_S07 | Lrig1-low cells from colonic tumors | disease state: Tumor \| lrgi1-expression: low \| tissue: colon | True | True | False | False |
| GSM5241866 | mouse | GSE172104,GSE172106 | Lung_KPC_2 | KPC lung | tissue: lung \| strain: C57BL/6 \| cell type: Non-immune cells enriched from KPC-bearing lungs | True | False | False | False |
| GSM7498317 | mouse | GSE234914,GSE235282 | MQ, DKO, RNA, stimulated 2h, r2 | Bone marrow | strain: C57/Bl6 \| tissue: Bone marrow \| cell line: J2 vitus transformed cell line \| cell type: bone marrow-derived macrophages \| genotype: Rela-/-Rel-/- \| treatment: 2 hours | True | False | True | False |
| GSM5944502 | mouse | GSE198297 | Mouse_WT_heart_RNA-seq_rep1 [Sample1_S1] | Embryonic heart tissue | tissue: Embryonic heart tissue \| genotype: WT \| developmental stage: E11.5 | True | False | False | False |
| GSM3889282 | mouse | GSE132706,GSE132885 | vWAT, 20 weeks, high fat diet, rep 6 | visceral (epididymal) adipose tissue | strain: C57BL/6 \| age: 26 weeks \| Sex: male \| diet: High fat diet (D12492) \| time: 20 weeks \| tissue: visceral (epididymal) adipose tissue | True | True | True | True |
| GSM6057952 | mouse | GSE201292 | SL1344-ILC3-LI-2 | Large intestine ILC3 | tissue: Large intestine \| cell type: ILC3 \| strain: C57BL/6 \| genotype: WT \| infection: S. typhimurium (SL1344) | True | True | True | False |
| GSM5340979 | human | GSE175533 | RS_PDL20_TP1_C | WI-38 | cell line: WI-38 \| time course: RS timecourse \| population doublings: 20 \| time point/day: TP1 \| extract protocol: Cells were lysed on the tissue culture plate with trizol and RNA was extracted using Zymo DirectZol Miniprep Plus Kit. Manufacturer protocol was followed exactly. | True | False | True | True |
| GSM7127664 | mouse | GSE228590 | run_17_seq_1_PD-02F_S302 | sci-RNA-seq3 of mouse organogenesis and fetal development | developmental stage: mouse embryos spanning late gastrulation (E8) to birth (P0) \| strain: C57BL/6 | True | False | True | False |
| GSM5956967 | mouse | GSE198790 | ST7 | HNSCC Tumor | strain: BALB/C \| genotype: WT mouse \| injected cell line: LY2 \| tissue: Tumor | True | True | True | False |
| GSM5222214 | mouse | GSE171292 | 99 | liver | strain: B6/129 \| tissue: liver \| age_weeks: 41 \| genotype: wt \| treatment: Saline | True | False | True | False |

This sample indicates whether the residual problem is absent metadata or unstructured metadata. Positive recoverability flags support a targeted Jev/Lydia gap-filling layer; negative flags should be treated as insufficient source evidence rather than guessed labels.

## 9. Limitations and interpretation

- GEOMeta is human bulk RNA-seq only and the observed public CSV release is smaller/narrower than the preprint description.
- Gemma is actively updated; these counts are a dated API snapshot, and failed/unavailable API responses are not silently counted as matches.
- Field completeness reflects explicit observed columns/categories. It intentionally does not infer cell type from cell-line names or tissue from generic free text.
- Normalized string agreement cannot resolve ontology hierarchy, mixed samples, or granularity differences. The conflict table is a triage input for later ontology-aware reconciliation.
- The 25-record GEO audit estimates recoverability qualitatively and is not a large-scale annotation result.

## 10. Machine-readable outputs

The `results/` directory contains Bridge↔GEOMeta and Bridge↔Gemma Parquet matches, combined coverage, unmatched GSMs, field-level completeness, the observed Gemma schema, agreement summaries/conflicts, retrieval-ready examples, the 25-record GEO audit, and provenance. Raw numerical values are unrounded in those files.
