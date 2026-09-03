# Task 2 — Exploratory cross-species exercise response

This benchmark matches GEPREP skeletal-muscle metadata to existing ARCHS4
expression rows and frozen BridgeRNA embeddings, then explores whether
within-study acute aerobic exercise responses are similar across human and
mouse. It does not download or process new expression data, fine-tune the
model, align species, or filter exploratory samples by pretraining exposure.

## Layout

```text
cross_species_exercise_response.ipynb  primary human-readable report
config.json                            fixed local inputs and analysis settings
pipeline/                              reproducible matching and analysis code
results/                               compact tables and figures
work/                                  regenerable matched matrices
results/hallmark_readout/              frozen-embedding Hallmark readout outputs
```

## Reproduce

From the repository root:

```bash
.venv/bin/python \
  benchmarks/cross_species_exercise_response/pipeline/run_exploratory_analysis.py \
  2>&1 | tee benchmarks/cross_species_exercise_response/results/run.log
```

The analysis reads the precomputed ARCHS4 expression shards and embedding
memmap. Use `--reuse-prepared` to reuse aligned matrices after the first run.

## Exploratory Hallmark readout

The Hallmark analysis uses 40,000 study-diverse human ARCHS4 samples, splits
by GSE, computes 50 ssGSEA targets, and compares identical nonlinear heads on
frozen BridgeRNA and 512-component PCA representations. It also evaluates the
trained heads without retraining on local TCGA and GTEx assets:

```bash
.venv/bin/python \
  benchmarks/cross_species_exercise_response/pipeline/run_hallmark_readout.py \
  2>&1 | tee benchmarks/cross_species_exercise_response/results/hallmark_readout_run.log
```

Interpret the fixed exercise contrasts with that frozen head and 1,000 paired/
group-aware bootstrap replicates:

```bash
.venv/bin/python \
  benchmarks/cross_species_exercise_response/pipeline/analyze_hallmark_response_axes.py \
  --bootstrap 1000 \
  2>&1 | tee benchmarks/cross_species_exercise_response/results/hallmark_response_axes_run.log
```

Run the separate streamed contextual-gene Hallmark membership experiment:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  benchmarks/cross_species_exercise_response/pipeline/run_gene_context_hallmark.py \
  --samples 256 --epochs 2 --device cuda:0 \
  2>&1 | tee benchmarks/cross_species_exercise_response/results/gene_context_hallmark_run.log
```

Attribute the fixed latent exercise axes directly to input genes with
Integrated Gradients and a study-mean native-mask deletion control:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  benchmarks/cross_species_exercise_response/pipeline/attribute_latent_axes.py \
  --ig-steps 8 --random-panels 3 --device cuda:0 \
  2>&1 | tee benchmarks/cross_species_exercise_response/results/latent_axis_attribution_run.log
```

Run full-transcriptome count-based differential expression and relate it to
the completed frozen-model IG/context rankings:

```bash
.venv/bin/python benchmarks/cross_species_exercise_response/pipeline/run_full_transcriptome_de.py \
  2>&1 | tee benchmarks/cross_species_exercise_response/results/full_transcriptome_de_run.log
```

Enrich the frozen Axis A/B IG Top-100 and conserved human–mouse IG sets using
GO Biological Process, KEGG, and Reactome with the exact 15,165-gene universe:

```bash
.venv/bin/python benchmarks/cross_species_exercise_response/pipeline/run_ig_pathway_enrichment.py \
  2>&1 | tee benchmarks/cross_species_exercise_response/results/ig_pathway_enrichment_run.log
```

Compare axis-consensus edgeR and IG pathway biology without new inference:

```bash
.venv/bin/python benchmarks/cross_species_exercise_response/pipeline/compare_de_ig_enrichment.py \
  2>&1 | tee benchmarks/cross_species_exercise_response/results/de_ig_enrichment_run.log
```

Compare species-specific human and mouse IG rankings within each fixed axis:

```bash
.venv/bin/python benchmarks/cross_species_exercise_response/pipeline/run_species_specific_ig_enrichment.py \
  2>&1 | tee benchmarks/cross_species_exercise_response/results/species_specific_ig_enrichment_run.log
```

Build the final Pattern A/B synthesis heatmaps with distinct DE and IG enrichment backgrounds:

```bash
.venv/bin/python benchmarks/cross_species_exercise_response/pipeline/build_task2_synthesis_heatmaps.py \
  2>&1 | tee benchmarks/cross_species_exercise_response/results/final_synthesis_run.log
```

Audit whether study biology or sequencing metadata visibly track Pattern A/B:

```bash
.venv/bin/python benchmarks/cross_species_exercise_response/pipeline/audit_pattern_metadata.py \
  2>&1 | tee benchmarks/cross_species_exercise_response/results/pattern_metadata_audit_run.log
```

Run the two-GPU extended latent-axis deletion sweep:

```bash
.venv/bin/python benchmarks/cross_species_exercise_response/pipeline/run_extended_deletion_sweep.py \
  --devices cuda:0 cuda:1 --heartbeat-seconds 30 \
  2>&1 | tee benchmarks/cross_species_exercise_response/results/latent_axis_attribution/extended_deletion_sweep.log
```
