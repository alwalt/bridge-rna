# Human + mouse ARCHS4 pre-inference audit

Status: **stopped before expensive inference**.

## Actual sample availability

| Asset | Human | Mouse | Combined |
|---|---:|---:|---:|
| Raw ARCHS4 v2.5 H5 columns | 888,821 | 997,515 | 1,886,336 |
| Samples in the compatible frozen Bridge cache | 510,709 | 429,746 | 940,455 |
| Unique GSMs in that cache | 510,709 | 429,746 | 940,455 |
| Samples requiring new Bridge inference | 0 | 0 | 0 |

The previously reported human reference size of 510,709 is verified from the cache row manifest, the repository sample manifest, and the ARCHS4 metadata manifest. The corresponding compatible mouse reference contains **429,746 samples**.

The larger H5 column counts are not the model-ready atlas size. The frozen embedding archive represents the 940,455 samples that passed the existing shared-ortholog preprocessing and sample-manifest construction. Expanding directly to every H5 column would require defining a new eligibility protocol and is outside the frozen compatible cache.

## Compatibility decision

The authoritative cache at `/home/walt/bridge-rna/embeddings/archs4/` already contains both species in one exact global order. All 940,455 rows join one-to-one to the sample and metadata manifests; there are zero missing metadata rows and zero species disagreements. A full matrix scan found no non-finite values and no zero-norm embeddings in either species.

The cache declares natural `log1p(TPM)`, 512 dimensions, no L2 normalization, and the same r7hnr92k frozen checkpoint family. The local checkpoint and config hashes match the frozen Hallmark benchmark manifest. The model config uses 15,165 genes and explicitly disables a species embedding, so human and mouse were encoded in the same representation function after species-correct mapping into the shared one-to-one ortholog ordering. All 15,165 canonical human symbols are unique, map to the one-to-one ortholog table, and have exactly one mouse gene ID in that table.

The frozen Hallmark transform remains:

`((Bridge-512 - GTEx mean) / GTEx SD) @ weight.T + bias`

The probe is unchanged. Mouse scores will be labeled **zero-shot cross-species predictions from the human GTEx-trained head**; they are not mouse-calibrated pathway measurements.

## Cheapest valid route

No Bridge inference is needed. The next step is a matrix projection of the 429,746 cached mouse embeddings through the frozen 512→50 probe, followed by normalization/index construction. Based on the measured human build, Hallmark projection should take roughly 16–30 seconds; index and metadata writing should take roughly 1–5 minutes.

The incremental storage estimate is about 1.0–1.2 GB: approximately 880 MB for a float32 mouse Bridge cosine index, 86 MB each for mouse Hallmark signatures and its cosine index, plus compact metadata/manifests. A combined atlas need not duplicate these arrays: exact combined retrieval can search the human and mouse indexes independently and merge their scores, while species-specific retrieval uses the corresponding view.

## Existing OSDR result policy

The existing `results/osdr_atlas/` directory will remain unchanged. It is now explicitly designated the legacy **mouse OSDR → human ARCHS4 cross-species retrieval** benchmark. New multispecies outputs will be written under `results/archs4_multispecies/` and a separate OSDR multispecies results directory.

No inference, retrieval recomputation, commit, merge, or push was performed during this audit.
