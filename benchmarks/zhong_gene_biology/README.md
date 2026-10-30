# Zhong et al. Bridge-only gene biology benchmark

This benchmark adapts the Zhong et al. 2025 gene-level and gene-pair probe design to **BridgeRNA only**.

Current status: **frozen and scientifically complete.** No further classifiers, controls, variants, or optimization are part of the final benchmark.

Reference: Zhong et al., *Benchmarking gene embeddings from sequence, expression, network, and text models for functional prediction tasks*. Upstream implementation: <https://github.com/ylaboratory/gene-embedding-benchmarks>, frozen at commit `d1320026a2a4ee033d49517f91e2d1c2ccc8df1e`.

## Scope

- Reuse Zhong split files and benchmark design.
- Restrict evaluation to genes present in Bridge vocabulary after Entrez mapping.
- Preserve original split assignments; do not reshuffle by default.
- Test frozen Bridge representations at the single-gene and gene-pair scales.

## Frozen representations

- **E0:** direct `model_state_dict['gene_embedding.weight']`; no REE or transformer processing.
- **E1:** final L12 state with uniform `log1p(TPM) = 0`; this is a null context, not a biological average.
- **E2:** final L12 states averaged without normalization across a fixed GTEx panel of 200 samples (10 tissues × 20 samples) using natural `log1p(TPM)`.

All representations contain 15,165 genes × 512 dimensions in canonical token order. The frozen checkpoint SHA256 is `f3491beaa697a0408dc1daeb6f8d648566bd46bafc6e44938d723e7ffc325541`.

## Final headline results

| Task | E0 mean AUROC | E1 mean AUROC | E2 mean AUROC |
|---|---:|---:|---:|
| GO (56 terms) | 0.5177 | 0.7722 | 0.7811 |
| OMIM (103 terms) | 0.4891 | 0.7354 | 0.7596 |

Best observed pair AUROC conditions were E2-concat for negative genetic interaction (0.580), E1-product for synthetic lethality (0.707), and E1-concat for TF targets (0.670). Pair evaluation is strictly gene-disjoint in every outer fold. See [RESULTS.md](./RESULTS.md) for all conditions, coverage, interpretation, and limitations.

A static-only extension evaluated the locally available BulkFormer-50M and BulkFormer-147M checkpoints using the same supplied GO/OMIM training folds and holdouts. Mean AUROC was 0.6961/0.6540 (GO/OMIM) for BulkFormer-50M and 0.7146/0.6406 for BulkFormer-147M. BulkFormer pair tasks and contextual representations were not run. Because BulkFormer and Bridge use different available vocabularies, these values are contextual rather than a formal common-universe ranking.

## Layout

- `pipeline/`: reproducible scripts for mapping, representation prep, and benchmarks.
- `coverage/`: mapping and retention audit outputs.
- `single_gene/`, `gene_pair/`, `controls/`, `figures/`: reserved for later phases.
- `references/`: cached external benchmark resources.
- `logs/`: flushed run logs.
- `results/final_benchmark_provenance.json`: frozen machine-readable provenance and audit record.
- `results/final_benchmark_summary.csv`: manuscript-oriented single- and paired-gene table.
- `figures/final_*.{png,pdf}`: final publication-oriented figures.

## Run Phase 0/1 mapping + coverage

```bash
python3 benchmarks/zhong_gene_biology/pipeline/phase1_mapping_coverage.py
```

To force canonical shared assets from the main checkout:

```bash
python3 benchmarks/zhong_gene_biology/pipeline/phase1_mapping_coverage.py \
  --shared-root /home/walt/bridge-rna
```

Optional refresh flags:

```bash
python3 benchmarks/zhong_gene_biology/pipeline/phase1_mapping_coverage.py \
  --force-refresh-zhong --force-refresh-gene-info
```

Monitor:

```bash
tail -f benchmarks/zhong_gene_biology/logs/phase1_mapping_coverage.log
```

## Build E2 real-context representation (GTEx-200 L12)

```bash
python3 benchmarks/zhong_gene_biology/pipeline/prepare_e2_gtex200_representation.py \
  --shared-root /home/walt/bridge-rna --device cuda
```

Monitor:

```bash
tail -f benchmarks/zhong_gene_biology/logs/prepare_e2_gtex200.log
```

## Run Zhong GO/OMIM single-gene ablation for E0/E1/E2

```bash
python3 benchmarks/zhong_gene_biology/pipeline/run_single_gene_ablation.py --seed 42
```

Monitor:

```bash
tail -f benchmarks/zhong_gene_biology/logs/run_single_gene_ablation.log
```

## Run gene-pair ablation

The runner writes one checkpoint per task/representation/joining-operation condition and resumes by default:

```bash
python3 benchmarks/zhong_gene_biology/pipeline/run_gene_pair_ablation.py --skip-controls --n-jobs 3
```

Monitor:

```bash
tail -f benchmarks/zhong_gene_biology/logs/run_gene_pair_ablation.log
```

The control runners are retained for historical reproducibility but were not run for the frozen benchmark and should not be launched as part of this final protocol.

## Build and execute the report notebook

```bash
/home/walt/bridge-rna/.venv/bin/python \
  benchmarks/zhong_gene_biology/pipeline/build_notebook.py
/home/walt/bridge-rna/.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  benchmarks/zhong_gene_biology/zhong_gene_biology.ipynb
```

## Main outputs

- `bridge_gene_mapping.tsv`
- `coverage/phase0_bridge_embedding_audit.json`
- `coverage/coverage_summary.csv`
- `coverage/term_coverage.csv`
- `coverage/pair_coverage.csv`
- `coverage/phase1_report.json`

Additional outputs:

- `representations/E2_GTEx200_L12/`
- `coverage/representation_qc_e0_e1_e2.json`
- `results/E0_GO.csv`, `results/E1_GO.csv`, `results/E2_GO.csv`
- `results/E0_OMIM.csv`, `results/E1_OMIM.csv`, `results/E2_OMIM.csv`
- `results/comparison_summary_AUROC.csv`, `results/comparison_summary_AUPRC.csv`
- `results/representation_summary.csv`
- `results/single_gene_zhong_matched_summary.csv` (unweighted mean/median/SD across final held-out term metrics)
- `results/zhong_literature_comparison.csv` (reported Zhong reference values plus Bridge means)
- `results/final_pair_leakage_qc.csv` (deduplicated gene-disjoint outer-fold audit)
- `results/gtex200_samples.tsv` (exact frozen E2 panel)

Recalculate the Zhong-matched summary from completed per-term tables only (no inference or classifier fitting):

```bash
python3 benchmarks/zhong_gene_biology/pipeline/summarize_existing_single_gene.py
python3 benchmarks/zhong_gene_biology/pipeline/finalize_saved_benchmark.py
```

Run the static-only BulkFormer GO/OMIM extension (no inference or pair tasks):

```bash
python3 benchmarks/zhong_gene_biology/pipeline/run_bulkformer_static_single_gene.py
```

The extension writes raw per-term tables, its mapping audit, summary, log, and provenance under `results/bulkformer_static/`.

## Mapping policy

- Source vocabulary: `data/ensembl/canonical_genes.csv`.
- Entrez mapping uses NCBI `Homo_sapiens.gene_info.gz`.
- Mapping keeps the original Bridge gene symbol and token order.
- Ambiguous and duplicate Entrez mappings are explicitly flagged (never silently merged).
- Shared assets are read-only; this benchmark does not copy or rewrite checkpoint/vocab/cache files from `/home/walt/bridge-rna`.

## Freeze policy

The benchmark is frozen at the results documented in [RESULTS.md](./RESULTS.md). Finalization reads saved result tables only. It must not trigger Bridge inference, classifier fitting, cross-validation, split regeneration, or new controls. Large representations and the upstream clone remain ignored; compact manifests, provenance, tables, scripts, the executed notebook, and final figures are versioned.
