# ARCHS4 v2.5 versus NCBI GEO metadata audit

## Decision

Choose **D: retrieve both GSE and selected GSM metadata, but only for the 59,804
GSE families represented by the 940,455 Bridge GSMs**. Use official per-series
family SOFT archives or the metadata-only accession endpoint, discard expression
tables, and store only fields absent from ARCHS4. This gets the value of option C
without mirroring all GEO or duplicating existing ARCHS4 metadata.

Every audited GSE supplied a summary and overall design that ARCHS4 lacks. In
addition, 75/100 GSMs supplied at least one useful sample-only field
(`treatment_protocol_ch1`, `growth_protocol_ch1`, or `description`). Existing
ARCHS4 characteristics were identical or formatting-equivalent to GEO for all
100 samples and should not be duplicated.

## Sources and complete ARCHS4 inventory

- `data/archs4/human_gene_v2.5.h5`: 47,865,298,452 bytes;
  888,821 H5 sample-metadata rows.
- `data/archs4/mouse_gene_v2.5.h5`: 38,960,132,574 bytes;
  997,515 H5 sample-metadata rows.
- `data/manifests/archs4_sample_metadata_v2.5.parquet`:
  940,455 Bridge-indexed samples and 15 columns.
- `embeddings/archs4/sample_locations.parquet` and existing
  metadata-coverage outputs were read only. No embedding or index was modified.

Bridge contains 510,709 human and 429,746 mouse GSMs in 59,804 unique GSEs
(30,482 human memberships and 30,423 mouse memberships; some GSEs contain both).

Both H5 files have the same 29 `meta/samples` fields:

- Identifiers: `geo_accession`, `series_id`, `organism_ch1`, `taxid_ch1`,
  `platform_id`.
- Biological/free text: `title`, `source_name_ch1`, `characteristics_ch1`.
  Tissue, disease, cell type, sex, age, strain, genotype, treatment, dose, and
  timepoint can occur inside characteristics but are not canonical H5 columns.
- Technical/provenance: `channel_count`, `data_processing`,
  `extract_protocol_ch1`, `instrument_model`, `library_selection`,
  `library_source`, `library_strategy`, `molecule_ch1`, `relation`, `sample`,
  `singlecellprobability`, `status`, `submission_date`, `last_update_date`,
  `type`.
- Submitter contact: `contact_name`, `contact_institute`, `contact_address`,
  `contact_city`, `contact_zip`, `contact_country`.

The derived Parquet retains GSM, GSE, organism, title, source, characteristics,
platform, library strategy/source, instrument, molecule, submission date, and
single-cell probability, plus source-file/version provenance. It omits 17 H5
fields, including library selection, extraction protocol, data processing,
relations, taxonomy, status/update date, and contact information. These are
recoverable locally without GEO.

The H5 has no GSE entity table. It contains no GSE title, summary, overall
design, PubMed ID, study type, study dates, study protocols, or study-level
organism/platform metadata. `meta/info` has only ARCHS4 file author, contact,
creation date, laboratory, and version; `meta/genes` has symbol, Ensembl ID, and
biotype.

## Official GEO metadata resources

NCBI's [download documentation](https://www.ncbi.nlm.nih.gov/geo/info/download.html),
[programmatic-access guide](https://www.ncbi.nlm.nih.gov/geo/info/geo_paccess.html),
and [SOFT specification](https://www.ncbi.nlm.nih.gov/geo/info/soft.html) describe
the official distributions. NCBI provides each GSE family as `{GSE}_family.soft.gz` (gzip SOFT) and
`{GSE}_family.xml.tgz` (tar/gzip MINiML). These contain the original GSE, linked
GSMs and GPL references, and can also contain submitted data tables. The
accession viewer with `targ=self&view=brief&form=text` returns metadata-only SOFT
for one GSM or GSE; this audit used it for exactly 100 GSMs and 100 GSEs.
E-Utilities/GEO DataSets can discover records and updates.

The FTP hierarchy is accession-ranged and updated as records are revised.
Records expose status, submission date, and last-update date. A production
ingest should record retrieval time and re-fetch records whose GEO update date
changes. Both human and mouse are represented.

Relevant records are selectively retrievable. One GSE family contains its
linked GSMs, so one request per relevant GSE is preferable to 940,455 individual
GSM requests. Do not fetch Series Matrix, supplementary expression files, SRA,
or FASTQ.

Observed brief-record means were 3.67 KB/GSM and 8.00 KB/GSE. Individual brief
retrieval extrapolates to about 3.45 GB for GSMs plus 0.48 GB for GSEs. Twenty
deterministic relevant family SOFT files had a 6.5 KB compressed median (range
1.9–13.7 KB), suggesting roughly 0.4 GB at the sample mean; allow 0.4–2 GB for
the long tail. Parsed minimal text is about 0.22 GB for the three GSM extras and
0.095 GB for selected GSE fields before indexes and provenance.

## Deterministic 100-sample comparison

Seed `bridge-archs4-geo-audit-v1` produced 50 human and 50 mouse samples, 100
unique GSEs, GEOMeta/Gemma both/one/neither strata where possible, and balanced
simple/moderate/complex characteristics. It is diversity-stratified rather than
prevalence-weighted.

| Field | Result |
|---|---:|
| GSM title/source/platform | 100 identical |
| GSM characteristics | 2 identical; 98 formatting-equivalent |
| GSM extract protocol | 7 identical; 93 formatting-equivalent |
| GSM treatment protocol | GEO-only 36; missing both 64 |
| GSM growth protocol | GEO-only 41; missing both 59 |
| GSM description | GEO-only 49; missing both 51 |
| GSE title/summary/overall design | GEO-only 100 each |
| GSE PubMed | GEO-only 87; absent 13 |
| GSE type/platform/dates | GEO-only 100 each |
| GSE relation | GEO-only 96 |

Two GSM update dates and one relation showed post-ARCHS4 GEO additions. Preserve
source timestamps rather than overwriting the frozen ARCHS4 values.

## Information gain for Laya

| GEO addition | Samples with useful evidence |
|---|---:|
| GSE summary | 100% |
| Overall design | 100% |
| PubMed identifier | 87% |
| Additional GSM characteristics | 0% |
| Treatment protocol | 36% |
| Growth protocol | 41% |
| Sample description | 49% |
| Any of the three GSM extras | 75% |
| Added genotype context | 39% |
| Added disease context | 66% |
| Added experimental-role context | 63% |
| Added tissue context | 58% |
| Added treatment context | 42% |

Context categories are conservative keyword evidence checks, not Laya output.
GEO added study context for all 100 samples. For 25 samples it added no unique
GSM text, so their gain was entirely study-level. GSE-only retrieval captures
the universal gain, but omitting GSM extras discards useful evidence for 75% of
this deliberately diverse cohort.

## Minimal eventual schema

1. `samples`: GSM, species/taxid, all GSE memberships, platform, Bridge
   `global_index`, index version, and row provenance.
2. `archs4_sample_metadata`: frozen ARCHS4 values, including locally recoverable
   library selection, extraction protocol, data processing, relations, status,
   and update date. Exclude contact/address fields from the serving database.
3. `studies`: GSE, title, summary, overall design, PubMed IDs, type, dates,
   organism/taxid sets, platforms, and BioProject/SRA relations—stored once.
4. `geo_sample_extras`: GSM, treatment protocol, growth protocol, description,
   GEO update/retrieval timestamps, and source locator. Do not duplicate title,
   source, characteristics, extraction/library fields, instrument, or platform
   except in a source-conflict record.
5. `source_values`: source, entity/field, raw and normalized value/hash, source
   release/update date, retrieval date, parser version, and conflict status.
6. `canonical_annotations`: tissue, disease/status, cell type, sex,
   age/development, strain, genotype, treatment, group, dose, and duration, each
   linked to source evidence.
7. `model_annotations`: Laya checkpoint, schema version, candidate set, selected
   value, scores, calibration state, evidence IDs, timestamp, and human review.

This avoids nearly a million copies of study text and keeps frozen ARCHS4 values
separate from current GEO revisions and later annotations.

## Reproduction and limitations

Run `pipeline/audit_archs4_vs_geo.py`. Brief records are cached under ignored
`data/geo/geo_archs4_audit_100/`. No expression data or large GEO archive was
downloaded. Full unrounded outputs are under `results/`.

The cohort estimates information diversity, not corpus prevalence. Family-file
sizing uses a small HEAD sample and needs a larger sizing pass before retrieval.
The absence of study context from ARCHS4 follows direct enumeration of both H5
schemas, not an assumption based on the derived Parquet.
