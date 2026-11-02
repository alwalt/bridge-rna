# Bridge unified ARCHS4 metadata database

`data/archs4/bridge_metadata.sqlite` is a metadata sidecar for the existing frozen
ARCHS4 Bridge retrieval index. `samples.global_index` is the immutable join to
row `global_index` of `sample_embeddings.float16.mmap`; the embedding file is
not stored or modified by this workflow.

## Tables

| Table | Grain and purpose |
|---|---|
| `samples` | One row per Bridge GSM, with `global_index`, species, and embedding shard location. |
| `sample_studies` | One row per GSM–GSE membership; preserves multiple series. |
| `archs4_metadata` | One row per GSM with all 29 original H5 sample fields plus ARCHS4 source/version. |
| `geo_downloads` | One row per targeted GSE download, including status, URL, digest, size, and error. |
| `studies` | One row per parsed GEO GSE with title, summary, design, publication, dates, organism/platform, and relations. |
| `geo_sample_extras` | Only GSM treatment protocol, growth protocol, description, and GEO provenance absent from ARCHS4. |
| `geometa` | Source-faithful rows from the downloaded GEOMeta release. |
| `gemma_annotations` | One row per Gemma characteristic or experimental-factor value, retaining category/value URIs. |
| `canonical_terms` | Versioned canonical vocabulary; initial GEOMeta terms and conservative ontology links. |
| `canonical_mappings` | Auditable source-value-to-canonical mappings and mapping status. |
| `metadata_assertions` | Append-only GSM × field × source assertions, including model evidence and scores. |
| `canonical_metadata` | Materialized user-facing selection following the documented source hierarchy. |
| `metadata_conflicts` | Unresolved cross-source conflicts; they are never silently overwritten. |
| `annotation_runs` | Model/tool/runtime/question/vocabulary provenance for annotation runs. |

## Canonical selection policy

For GEOMeta-supported fields, select GEOMeta first. Use Gemma only if its single
normalized value maps exactly and unambiguously to the observed GEOMeta
vocabulary. A later Laya assertion may fill a still-null value but cannot resolve
a GEOMeta/Gemma conflict. Gemma-specific cell type, strain, genotype, treatment,
dose, and duration remain available at their original specificity.

The initial vocabulary is derived only from actual values in the pinned GEOMeta
CSV release. UBERON mappings are accepted only for unique, case-insensitive exact
label matches returned by OLS4; uncertain mappings remain null. Gemma ontology
URIs are preserved on their own assertions and are not collapsed into UBERON.

Laya's `model_score` is stored verbatim and is not named or treated as calibrated
confidence. Validation predictions have `status='validation_only'` when imported.

## Provenance rules

- Source tables preserve raw values.
- Canonical selection does not delete or overwrite assertions.
- GEO and source update dates remain separate from retrieval timestamps.
- Every model assertion links to `annotation_runs` and stores its exact evidence.
- Schema version: `bridge-metadata-v1`.
- Initial canonical vocabulary version: `geometa-hf-d0cd1e1`.

The executable DDL is `pipeline/schema.sql`; example queries are in
`results/example_queries.sql`.
