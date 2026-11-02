# Bridge Hallmark representation benchmark

This self-contained benchmark asks (1) whether biology in Bridge's frozen
512-dimensional sample embedding is axis-aligned or distributed and (2) whether
final-layer contextual gene states support independently validated, per-sample
Hallmark readouts in GTEx and TCGA.

The implementation is new and does not import, modify, or depend on any prior
pathway-head notebook or code. Authoritative repository assets are inputs only.
Bridge is always frozen.

## Layout

- `config.json`: frozen paths, seeds, folds, controls, and matched comparisons.
- `pipeline/audit.py`: provenance, reuse decision, gene-set mapping, cohort manifest.
- `pipeline/sample_space.py`: Parts I-II geometry, associations, independent
  expression targets, donor-aware nested probes, and null controls.
- `pipeline/contextual_decoder.py`: Parts III-IV streamed final-layer extraction,
  held-out-gene linear decoder, compact profiles, and top-gene audit records.
- `pipeline/validate_profiles.py`: independent pathway validation and sample-pooling recovery.
- `pipeline/tcga_representation.py`: authoritative TCGA inference and patient-disjoint probes.
- `pipeline/transfer_and_heatmaps.py`: frozen GTEx probe, zero-shot transfer, and heatmaps.
- `pipeline/de_gsea.py`: whole-transcriptome versus Bridge-vocabulary DE/GSEA.
- `pipeline/integrated_gradients.py`: targeted pathway IG and gene-level audit table.
- `pipeline/validate_integrated_gradients.py`: nested 8/16/32/64-step IG
  completeness and gene-ranking convergence, followed by validated targeted outputs.
- `pipeline/pan_cancer_gsea.py`: seven defensible primary-tumor subtype/site
  comparisons with matched whole-transcriptome and Bridge-vocabulary GSEA.
- `pipeline/matched_organs.py`: exploratory GTEx–TCGA matched-organ analysis.
- `pipeline/frozen_pathway_inference.py`: frozen future sample-to-pathway interface.
- `results/`: compact tables, manifests, figures, report, and logs.
- `work/`: ignored regenerable arrays and decoder parameters.

## Run

```bash
bash benchmarks/bridge_hallmark_representation/pipeline/run_all.sh
```

After the GTEx stage, run the TCGA/transfer/GSEA/IG continuation with:

```bash
bash benchmarks/bridge_hallmark_representation/pipeline/run_tcga_and_validation.sh
```

Monitor the non-contextual run with:

```bash
tail -f benchmarks/bridge_hallmark_representation/results/run.log
```

The contextual phases are deliberately explicit because full-cohort inference is
expensive. First make a small end-to-end decoder validation:

```bash
PY=/home/walt/bridge-rna/.venv/bin/python
$PY benchmarks/bridge_hallmark_representation/pipeline/contextual_decoder.py extract \
  --dataset gtex --expression /home/walt/bridge-rna/benchmarks/mouse_encode/work/expanded_gtex_log1p_tpm.npy \
  --samples 32 --device cuda:0
$PY benchmarks/bridge_hallmark_representation/pipeline/contextual_decoder.py train \
  --contexts benchmarks/bridge_hallmark_representation/work/gtex_decoder_contexts.float16.npy
```

Only after held-out-gene metrics pass should `score` be run on frozen GTEx and
newly prepared TCGA matrices. It emits 50-D mean-member and top-k-member profiles
plus the top five responsible member genes per sample/pathway; it does not retain
full contextual tensors.

TCGA and validation stages are independently resumable. Regenerated embeddings,
the frozen probe, and decoder parameters remain in ignored `work/`; compact
manifests, metrics, profiles, tables, and figures are versioned in `results/`.

Run only the focused IG and pan-cancer validations with:

```bash
bash benchmarks/bridge_hallmark_representation/pipeline/run_focused_validations.sh
```

The validated IG remains sample-specific: it attributes an individual frozen
Bridge→Hallmark score relative to an all-zero natural `log1p(TPM)` vector. It is
not a cancer-pair attribution. The pan-cancer analysis uses the existing frozen
per-sample profiles and therefore covers subtype/site contrasts among primary
tumors. Tumor-versus-adjacent-normal analysis is gated because adjacent normals
are absent from that frozen profile cohort.

## Prespecified interpretation

- The raw uncentered mean-pooled embedding is primary; post-processing is an ablation.
- Axis-aligned and distributed models use identical donor/patient-aware outer folds.
- Dimension selection and ridge tuning happen inside each training fold.
- AUPRC, compared with prevalence, is primary for the imbalanced gene decoder.
- Pathway profiles are called *Bridge-derived Hallmark pathway scores*, not enrichment scores.
- GTEx–TCGA subtraction is withheld until all preceding validation succeeds and is
  interpreted only as cross-cohort displacement, never a pure tumor effect.
