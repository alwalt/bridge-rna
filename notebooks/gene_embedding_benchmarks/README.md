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

The public cache also includes the original published Zhong common-intersection
results for the three methods classified by `embed_meta.csv` as
`gene expression (bulk)`: `GENE2VEC`, `TCGA-EMBEDDING`, and `FROGS-ARCHS4`.
Single-cell, protein-language, literature, PPI, and mixed-input methods are not
included in the bulk-expression comparison.

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
│   ├── frozen expected summaries and provenance
│   └── zhong/
│       ├── embed_meta.csv
│       ├── published GO/OMIM and pair TSVs
│       └── provenance.json
└── figures/
    ├── single_gene_go_omim.{png,pdf}
    ├── gene_pair_best.{png,pdf}
    ├── bulk_rnaseq_single_gene_comparison.{png,pdf}
    └── bulk_rnaseq_pair_comparison.{png,pdf}
```

Use `prepare_cached_artifacts.py` only when intentionally refreshing this public
release from a newly validated research benchmark:

```bash
python notebooks/gene_embedding_benchmarks/prepare_cached_artifacts.py
```

That command refreshes the local validated artifacts and downloads only the
compact published Zhong tables—never pretrained embedding matrices—from the
pinned revision below.

## Published Zhong source

- Repository: <https://github.com/ylaboratory/gene-embedding-benchmarks>
- Pinned commit: [`d1320026a2a4ee033d49517f91e2d1c2ccc8df1e`](https://github.com/ylaboratory/gene-embedding-benchmarks/commit/d1320026a2a4ee033d49517f91e2d1c2ccc8df1e)
- Results directory: <https://github.com/ylaboratory/gene-embedding-benchmarks/tree/d1320026a2a4ee033d49517f91e2d1c2ccc8df1e/results/tsvs>
- Metadata: <https://github.com/ylaboratory/gene-embedding-benchmarks/blob/d1320026a2a4ee033d49517f91e2d1c2ccc8df1e/data/embed_meta.csv>
- Download date for the committed cache: 2026-10-29

The cache preserves the upstream filenames. It contains the intersection
`single_gene_go_AUROC.tsv` and `single_gene_omim_AUROC.tsv` tables plus, for NG,
SL, and TF and each of concat/product/sum, both the published
`outerloop_avgs.tsv` summary and `outerloop_AUC.tsv` fold table. The latter lets
cached execution verify every published three-fold mean and derive its sample
SD. Source URLs, byte sizes, and SHA-256 hashes are recorded in
`cached/zhong/provenance.json` and the top-level cache manifest.

Zhong's metadata records Gene2vec as a skip-gram/CBOW method and TCGA-Embedding
and FRoGS-ARCHS4 as MLP methods, all trained from bulk gene expression. This is
also consistent with the benchmark paper's method descriptions. The notebook
uses the original metadata category as the eligibility rule rather than
classifying models from their names.

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
- The three cached Zhong bulk-expression results use the authors' common-gene
  intersection. Bridge and BulkFormer local adaptations use their own mapped
  vocabularies. They share tasks, split definitions, metrics, and aggregation
  logic, but not identical evaluated genes, so they are shown in separate
  table/figure sections and must not be treated as one ranking.
- Published Zhong single-gene TSVs contain per-term AUROC but not AUPRC. Pair
  summary TSVs contain mean outer-fold AUROC, AUPRC, and precision@10; the
  paired fold TSVs provide fold-level AUROC only.
- Bridge and BulkFormer use their own available vocabularies, so their table is
  contextual rather than a strict common-universe model ranking.
- BulkFormer was evaluated only for static single-gene GO/OMIM tasks.
- Bridge pair results use three strictly gene-disjoint outer folds; BulkFormer
  pair performance was not evaluated.
- Probe performance establishes decodability, not causal or mechanistic
  understanding, batch correction, or universal superiority.
