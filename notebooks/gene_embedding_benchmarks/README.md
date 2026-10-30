# Gene embedding benchmarks

This directory is the public, cached-first release of the validated GO, OMIM,
and gene-pair benchmark. The main report is
[`gene_embedding_benchmarks.ipynb`](./gene_embedding_benchmarks.ipynb).

The original research implementation and complete audit trail remain under
[`benchmarks/zhong_gene_biology/`](../../benchmarks/zhong_gene_biology/). Nothing
in this public directory replaces or mutates that research record.

## Cached execution

Cached mode performs no model inference, SVM fitting, cross-validation, split
generation, or controls. It recomputes every displayed aggregate from committed
per-term or per-fold outputs, verifies those aggregates against the frozen
research summaries, checks pair-fold gene disjointness, and regenerates the PNG
and PDF figures.

From the repository root:

```bash
python notebooks/gene_embedding_benchmarks/build_notebook.py
jupyter nbconvert --to notebook --execute --inplace \
  notebooks/gene_embedding_benchmarks/gene_embedding_benchmarks.ipynb
```

Expected cached-mode outputs:

```text
notebooks/gene_embedding_benchmarks/
├── gene_embedding_benchmarks.ipynb
├── cached/
│   ├── manifest.json
│   ├── per-term GO/OMIM tables
│   ├── pair outer-fold results
│   └── frozen expected summaries and provenance
└── figures/
    ├── single_gene_go_omim.{png,pdf}
    └── gene_pair_best.{png,pdf}
```

Use `prepare_cached_artifacts.py` only when intentionally refreshing this public
release from a newly validated research benchmark:

```bash
python notebooks/gene_embedding_benchmarks/prepare_cached_artifacts.py
```

## Full recomputation

Full recomputation is intentionally not triggered by the public notebook. It
requires the model checkpoints, large ignored representation arrays, GTEx data,
Zhong split assets, and the repository's research environment.

Run from the repository root in this order:

1. Audit the Bridge vocabulary, mappings, and Zhong assets:

   ```bash
   python benchmarks/zhong_gene_biology/pipeline/phase1_mapping_coverage.py \
     --shared-root /home/walt/bridge-rna
   ```

2. Extract Bridge E0 and E1:

   ```bash
   python benchmarks/zhong_gene_biology/pipeline/prepare_e0_e1_representations.py \
     --shared-root /home/walt/bridge-rna
   ```

3. Build the fixed GTEx-200 E2 representation. This performs model inference:

   ```bash
   python benchmarks/zhong_gene_biology/pipeline/prepare_e2_gtex200_representation.py \
     --shared-root /home/walt/bridge-rna --device cuda
   ```

4. Fit the GO and OMIM probes for Bridge:

   ```bash
   python benchmarks/zhong_gene_biology/pipeline/run_single_gene_ablation.py --seed 42
   ```

5. Optionally evaluate the locally available BulkFormer static tables. This
   performs no BulkFormer inference and no pair tasks:

   ```bash
   python benchmarks/zhong_gene_biology/pipeline/run_bulkformer_static_single_gene.py --seed 42
   ```

6. Fit the Bridge gene-pair probes. This is the expensive CPU step and resumes
   completed task/representation/operation conditions:

   ```bash
   python benchmarks/zhong_gene_biology/pipeline/run_gene_pair_ablation.py \
     --seed 42 --n-jobs 3 --skip-controls
   ```

7. Rebuild the research summaries, provenance, and notebook:

   ```bash
   python benchmarks/zhong_gene_biology/pipeline/summarize_existing_single_gene.py
   python benchmarks/zhong_gene_biology/pipeline/finalize_saved_benchmark.py
   python benchmarks/zhong_gene_biology/pipeline/build_notebook.py
   jupyter nbconvert --to notebook --execute --inplace \
     benchmarks/zhong_gene_biology/zhong_gene_biology.ipynb
   ```

8. Review the research outputs and only then refresh this public cache with
   `prepare_cached_artifacts.py` and execute the public notebook.

Do not overwrite or delete prior validated results when changing the cohort,
vocabulary, representation definition, split protocol, or model checkpoint.

## Interpretation boundaries

- Zhong literature values were not rerun locally.
- Bridge and BulkFormer use their own available vocabularies, so their table is
  contextual rather than a strict common-universe model ranking.
- BulkFormer was evaluated only for static single-gene GO/OMIM tasks.
- Bridge pair results use three strictly gene-disjoint outer folds; BulkFormer
  pair performance was not evaluated.
- Probe performance establishes decodability, not causal or mechanistic
  understanding, batch correction, or universal superiority.
