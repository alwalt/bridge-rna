# Bridge ARCHS4 metadata coverage

This directory contains the read-only metadata sidecar for the frozen Bridge
ARCHS4 retrieval index:

- `bridge_metadata.sqlite`: normalized sample, study, source-annotation,
  canonical-label, provenance, and conflict tables for every indexed sample.
- `samples.global_index`: the immutable join from a GSM to its Bridge embedding
  row. Embeddings are not stored here and were not modified by this audit.

## Corpus

| Species | GSM samples | GSE studies |
|---|---:|---:|
| Human | 510,709 | 30,482 |
| Mouse | 429,746 | 30,423 |
| Total | 940,455 | 59,804 unique GSEs across the combined corpus |

## Curated-source identifier coverage

Coverage means that the GSM was present in the downloaded resource. It does not
mean that every biological field was populated.

| Species | Source | Matched GSMs | Missing GSMs | Coverage |
|---|---|---:|---:|---:|
| Human | GEOMeta | 438,119 | 72,590 | 85.79% |
| Human | Gemma | 72,952 | 437,757 | 14.28% |
| Human | GEOMeta or Gemma | 444,541 | 66,168 | 87.04% |
| Human | GEOMeta and Gemma | 66,530 | 444,179 | 13.03% |
| Mouse | GEOMeta | 0 | 429,746 | 0.00% |
| Mouse | Gemma | 104,503 | 325,243 | 24.32% |
| Mouse | GEOMeta or Gemma | 104,503 | 325,243 | 24.32% |

The examined GEOMeta release is human-only. Official GEO brief metadata was
retrieved successfully for 59,791 of 59,804 target GSEs; 13 accessions remain
explicit download failures.

## Accepted canonical-field coverage

These are the materialized, user-facing selections in `canonical_metadata`.
Missing values are SQL `NULL` and must not be interpreted as confirmed negative
biology.

| Canonical field | Available | Missing | Coverage |
|---|---:|---:|---:|
| Organ | 404,266 | 536,189 | 42.99% |
| Disease | 297,603 | 642,852 | 31.64% |
| Gender | 175,090 | 765,365 | 18.62% |
| Age group | 78,125 | 862,330 | 8.31% |
| Experimental setting | 438,119 | 502,336 | 46.59% |
| Perturbation | 438,119 | 502,336 | 46.59% |
| Cell type | 72,985 | 867,470 | 7.76% |
| Strain | 63,227 | 877,228 | 6.72% |
| Genotype | 98,795 | 841,660 | 10.51% |
| Treatment | 96,172 | 844,283 | 10.23% |
| Dose | 5,570 | 934,885 | 0.59% |
| Duration/development | 93,309 | 847,146 | 9.92% |

## Additional context and provenance

The database also retains original ARCHS4 sample fields, official GEO study
titles/summaries/designs, selected GEO sample descriptions and protocols,
source-faithful GEOMeta rows, and Gemma characteristics/experimental factors.
Use `metadata_assertions` and the source-specific tables to display evidence and
provenance behind a canonical value.

Downloaded source files are stored separately under `data/geo/`, `data/gemma/`,
and `data/geometa/`. The database and all downloaded source files are ignored by
Git; this Markdown summary is intentionally versionable.

## Key limitations

- Curated-resource membership is not equivalent to field completeness.
- GEOMeta contributes no mouse coverage in the pinned release.
- Gemma coverage is selective and field availability varies substantially.
- GEO study summaries provide useful context but should not override explicit
  sample-level evidence or generate sample-level canonical labels by themselves.
- Laya validation outputs are marked `validation_only`; no Laya or Jev labels
  were promoted into canonical metadata.
- Deterministic Organ-recovery proposals remain separate audit outputs and were
  not promoted because their measured precision did not meet the desired bar.
- Thirty-six explicit cross-source conflicts remain preserved for review.

For table definitions, source-precedence rules, and detailed audit results, see
`../../benchmarks/archs4_metadata_coverage/DATABASE_SCHEMA.md`,
`../../benchmarks/archs4_metadata_coverage/UNIFIED_METADATA_REPORT.md`, and
`../../benchmarks/archs4_metadata_coverage/ORGAN_RECOVERY_REPORT.md`.
