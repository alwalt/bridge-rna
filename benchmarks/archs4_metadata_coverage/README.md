# ARCHS4 metadata coverage audit

This benchmark audits how much of the frozen BridgeRNA ARCHS4 embedding corpus
can be annotated from the published GEOMeta release and the Gemma REST API.
It does not modify the retrieval index or regenerate embeddings.

## Run

```bash
.venv/bin/python benchmarks/archs4_metadata_coverage/pipeline/download_geometa.py
.venv/bin/python benchmarks/archs4_metadata_coverage/pipeline/fetch_gemma.py
.venv/bin/python benchmarks/archs4_metadata_coverage/pipeline/analyze_coverage.py
.venv/bin/python benchmarks/archs4_metadata_coverage/pipeline/build_notebook.py
```

Downloaded source tables and resumable API caches are written to `data/geo/`,
`data/gemma/`, and `data/geometa/`.
Compact audit tables, provenance, and the executed report are written to
`results/`.

## Laya pilot environment

Laya is pinned to commit `4066d5d5fbf08b66c6757ddeedbd797bd7655bc0`
(version 0.3.20). Set up its isolated CUDA 13.0 environment and run the installation
smoke test with:

```bash
bash benchmarks/archs4_metadata_coverage/pipeline/setup_laya.sh
```

The source checkout, virtual environment, and Hugging Face checkpoint cache are
stored under the git-ignored `work/` directory. The smoke test deliberately uses
Laya's typed `choice` and `noul` interface. Laya is a decision model rather than
a free-text generator, so the metadata pilot must supply explicit candidate
labels (including an `unknown` option) and manually score the resulting fields.
See `results/laya_setup_provenance.json` for the tested versions and known
calibration warning.

## ARCHS4 versus GEO audit

The read-only comparison of the complete ARCHS4 v2.5 H5 metadata schema against
100 selectively retrieved GEO GSM/GSE brief records is documented in
`ARCHS4_VS_GEO_AUDIT.md`. Reproduce it with:

```bash
python3 benchmarks/archs4_metadata_coverage/pipeline/audit_archs4_vs_geo.py
```

## Unified metadata database

The normalized SQLite sidecar and controlled Laya validation are documented in
`DATABASE_SCHEMA.md` and `UNIFIED_METADATA_REPORT.md`. The reproducible sequence
is:

```bash
python3 benchmarks/archs4_metadata_coverage/pipeline/map_geometa_uberon.py
python3 benchmarks/archs4_metadata_coverage/pipeline/download_geo_families.py --workers 48
python3 benchmarks/archs4_metadata_coverage/pipeline/build_unified_database.py
HF_HOME=benchmarks/archs4_metadata_coverage/work/laya-hf \
  benchmarks/archs4_metadata_coverage/work/laya-gpu-venv/bin/python \
  benchmarks/archs4_metadata_coverage/pipeline/run_laya_validation.py
python3 benchmarks/archs4_metadata_coverage/pipeline/build_unified_report.py
```

The downloader retrieves only GEO `brief` GSE and linked-GSM SOFT metadata and
compresses it locally; it does not retrieve platforms, expression tables,
supplementary files, FASTQ, or SRA reads. It is resumable and writes an explicit
manifest and failure log. For long builds, `build_unified_database.py
--local-only` stages ARCHS4/GEOMeta/Gemma and `--resume` adds GEO and finalizes.

`data/archs4/bridge_metadata.sqlite` uses `samples.global_index` as the stable link
to the existing Bridge embedding mmap. The Laya pilot stores 800 assertions as
validation-only and does not promote them into production canonical metadata.

## Conservative Organ recovery audit

`ORGAN_RECOVERY_REPORT.md` evaluates the GEOMeta Organ vocabulary, reviewed
UBERON/synonym matching, and a bounded shortlisted Laya fallback. It opens the
completed SQLite database read-only and writes proposed labels only beneath
`results/organ_recovery/`.

```bash
python3 benchmarks/archs4_metadata_coverage/pipeline/audit_organ_recovery.py
HF_HOME=benchmarks/archs4_metadata_coverage/work/laya-hf \
  benchmarks/archs4_metadata_coverage/work/laya-gpu-venv/bin/python \
  benchmarks/archs4_metadata_coverage/pipeline/run_laya_organ_shortlist.py
python3 benchmarks/archs4_metadata_coverage/pipeline/summarize_organ_recovery.py
```

The deterministic run uses a fixed 5,000-sample held-out GEOMeta cohort. The
Laya run is restricted to the 140 cases that remain ambiguous in that cohort;
it does not run across the missing corpus. Laya outputs remain
`validation_only`, and `model_score` is not treated as calibrated confidence.
