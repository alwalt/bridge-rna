#!/usr/bin/env python3
"""Render the audit tables into a self-contained Markdown report and notebook."""
from pathlib import Path
import json

import nbformat
import pandas as pd
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


def md_table(df: pd.DataFrame, decimals: int = 2) -> str:
    x = df.round(decimals).fillna("").astype(str)
    clean = lambda v: v.replace("|", "\\|").replace("\n", " ")
    header = "| " + " | ".join(clean(c) for c in x.columns) + " |"
    rule = "| " + " | ".join("---" for _ in x.columns) + " |"
    rows = ["| " + " | ".join(clean(v) for v in row) + " |" for row in x.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *rows])


def main() -> None:
    cov = pd.read_csv(R / "coverage_summary.csv")
    comp = pd.read_csv(R / "field_level_completeness.csv")
    agree = pd.read_csv(R / "agreement_summary.csv")
    examples = pd.read_csv(R / "retrieval_ready_examples.csv").head(10)
    gaps = pd.read_csv(R / "unmatched_geo_audit_25.csv")
    cats = pd.read_csv(R / "gemma_observed_schema.csv")
    prov = json.loads((R / "provenance.json").read_text())
    human = cov[(cov.level == "GSM") & (cov.species == "human")].set_index("source")
    union_pct = human.at["union", "coverage_pct"]
    gap_pct = 100 - union_pct
    recover = gaps[["tissue_recoverable","disease_state_recoverable","treatment_recoverable","experimental_role_recoverable"]].mean().mul(100).reset_index()
    recover.columns = ["field", "recoverable_pct_in_25_sample_audit"]
    field_pivot = comp.pivot_table(index=["species","field"], columns="source", values="availability_pct_of_bridge").reset_index()
    report = f"""# GEOMeta and Gemma coverage of the Bridge ARCHS4 corpus

Audit date: 2026-09-25. This is a metadata audit only; no embeddings or retrieval indices were changed.

## Executive answer

The frozen Bridge ARCHS4 embedding corpus contains **940,455 unique GSMs**: **510,709 human** and **429,746 mouse**. Every row is unique and `global_index` is contiguous from 0 to 940,454, so row *i* of the embedding memmap is identified by row *i* of the location table.

For the human corpus, the observed union of the currently downloadable GEOMeta CSV release and successfully retrieved Gemma sample records covers **{human.at['union','matched']:,.0f}/{human.at['union','total']:,.0f} GSMs ({union_pct:.2f}%)**. The remaining measured gap is **{human.at['union','unmatched']:,.0f} GSMs ({gap_pct:.2f}%)**. GEOMeta supplies much broader human GSM coverage; Gemma adds ontology-backed fields and mouse coverage but is a selective curated collection rather than an ARCHS4-wide catalogue.

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

Release caveat: the preprint says 594,989 GSM–GSE rows / 594,304 unique GSMs and describes richer perturbation, dose and duration fields. Those records/fields are **not present in the pinned public metadata CSVs** (HF revision `{prov['geometa_revision']}`). Results below therefore describe reproducible, observed downloadable data rather than the manuscript headline.

## 3. Gemma resources and observed schema

Official resources: [Gemma](https://gemma.msl.ubc.ca/), [REST v2 documentation](https://gemma.msl.ubc.ca/resources/restapidocs/), and Lim et al. 2021 (`doi:10.1093/database/baab006`). The live API reported 23,741 catalogue records; after accession de-duplication this audit observed 23,103 GSE accessions: 9,626 human, 11,911 mouse and 1,566 rat. Bridge-overlapping GSE sample records were requested from `/rest/v2/datasets/{{GSE}}/samples`; no expression data were downloaded. Of 8,508 catalogue-overlapping Bridge GSEs, **{prov['gemma_overlapping_gse_sample_endpoint_404s']:,}** consistently returned HTTP 404 from that endpoint and are conservatively treated as unresolved rather than matched.

Gemma exposes GEO GSE/GSM accessions, study- and sample-level ontology-backed characteristics, and sample experimental-factor values. Observed categories include tissue/organism part, disease, cell type, sex, age/developmental stage, strain, genotype, treatment, dose/duration-like fields, and experimental factors. Category spelling is heterogeneous, so the machine-readable observed schema records every category and its explicit audit mapping.

Top observed mapped categories:

{md_table(cats[cats.mapped_fields.fillna('').ne('')].head(30))}

## 4. Actual GSM and GSE coverage

{md_table(cov)}

Gemma matching is conservative: a GSM counts only when returned in the sample endpoint, not merely because its GSE exists in the catalogue. GSE matching likewise uses successfully observed sample records. Comma-separated ARCHS4 series memberships are exploded for GSE denominators.

## 5. Field-level completeness

Percent below is of the full Bridge species corpus, not merely matched samples. The source table also supplies matched-sample denominators and percentages.

{md_table(field_pivot)}

GEOMeta's public CSV has no sample-level cell type, strain, genotype, dose, duration, or detailed treatment term. `GSE_Pert` is retained as a study-level yes/no perturbation indicator and is not presented as a specific treatment. Gemma availability is counted only from semantically mapped characteristic/factor categories; generic free-text `BioSource` is not promoted to structured tissue.

## 6. Agreement where both resources exist

{md_table(agree)}

`exact` compares source strings and `normalized` applies case/punctuation normalization plus straightforward sex synonyms. `obvious_conflict` is reserved for direct male/female disagreements. Disease differences remain in `different_vocabulary_or_candidate_conflict` because reference-role factors can coexist with specific diagnoses and require ontology-aware review. GEOMeta's yes/no study perturbation is marked `not_comparable_granularity` against Gemma treatment terms. Tissue granularity is often different (for example organ system versus specific organism part), and Gemma factors may encode roles rather than GEOMeta disease labels. Cell type comparisons are missing-on-GEOMeta because that field is absent from the downloadable CSV schema.

## 7. Ten retrieval-ready examples

{md_table(examples[[c for c in ['global_index','gsm','species','series_id','geometa_matched','gemma_matched','union_tissue','union_disease','union_disease_status','union_cell_type','union_sex','union_age_development','union_strain','union_genotype','union_treatment_perturbation','union_experimental_factor_group','union_dose','union_duration'] if c in examples]])}

Values remain source-faithful and provenance-specific in the machine-readable table. A production view can later choose a display precedence while preserving both originals.

## 8. Union-unmatched GEO audit (25 deterministic random GSMs)

GEO SOFT records were fetched directly for 25 deterministic random union-unmatched GSMs. The flags below are conservative keyword/evidence checks over titles, sources, characteristics, descriptions and protocols—not LLM annotations.

{md_table(recover)}

{md_table(gaps[['gsm','species','gse','geo_title','geo_source','geo_characteristics','tissue_recoverable','disease_state_recoverable','treatment_recoverable','experimental_role_recoverable']])}

This sample indicates whether the residual problem is absent metadata or unstructured metadata. Positive recoverability flags support a targeted Jev/Lydia gap-filling layer; negative flags should be treated as insufficient source evidence rather than guessed labels.

## 9. Limitations and interpretation

- GEOMeta is human bulk RNA-seq only and the observed public CSV release is smaller/narrower than the preprint description.
- Gemma is actively updated; these counts are a dated API snapshot, and failed/unavailable API responses are not silently counted as matches.
- Field completeness reflects explicit observed columns/categories. It intentionally does not infer cell type from cell-line names or tissue from generic free text.
- Normalized string agreement cannot resolve ontology hierarchy, mixed samples, or granularity differences. The conflict table is a triage input for later ontology-aware reconciliation.
- The 25-record GEO audit estimates recoverability qualitatively and is not a large-scale annotation result.

## 10. Machine-readable outputs

The `results/` directory contains Bridge↔GEOMeta and Bridge↔Gemma Parquet matches, combined coverage, unmatched GSMs, field-level completeness, the observed Gemma schema, agreement summaries/conflicts, retrieval-ready examples, the 25-record GEO audit, and provenance. Raw numerical values are unrounded in those files.
"""
    (ROOT / "REPORT.md").write_text(report)
    nb = nbformat.v4.new_notebook()
    nb["cells"] = [nbformat.v4.new_markdown_cell(report), nbformat.v4.new_code_cell(
        "from pathlib import Path\nimport pandas as pd\nR=Path('results')\ncoverage=pd.read_csv(R/'coverage_summary.csv')\ncompleteness=pd.read_csv(R/'field_level_completeness.csv')\nassert set(coverage['species']) == {'human','mouse'}\nassert (coverage['matched'] <= coverage['total']).all()\nprint('QA passed:', len(coverage), 'coverage rows and', len(completeness), 'field rows')\ncoverage"
    )]
    nb["metadata"]["kernelspec"] = {"display_name":"Python 3","language":"python","name":"python3"}
    path = ROOT / "archs4_metadata_coverage.ipynb"
    nbformat.write(nb, path)
    client = NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata":{"path":str(ROOT)}})
    client.execute()
    nbformat.write(nb, path)
    print(f"wrote {ROOT/'REPORT.md'} and executed {path}")


if __name__ == "__main__":
    main()
