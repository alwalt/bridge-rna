# Static gene-embedding benchmark

This benchmark inspects the frozen BridgeRNA gene-identity embedding before any
sample-dependent contextualization. It extracts the checkpoint tensor, computes
an arithmetic mean TPM over the locally available subset of the model's training
split, embeds genes with a
deterministic t-SNE, clusters the original 512-dimensional vectors with k-means
(`k=10`), and tests every cluster for GO Biological Process and KEGG
enrichment. The canonical 15,165 model genes are the enrichment universe.

Cluster identifiers are the deterministic scikit-learn k-means labels plus one
(`1` through `10`). They are identifiers, not an ordering or biological rank.
The requested comparisons are GO clusters 6/9 and KEGG clusters 2/7.

## Run

From the repository root:

```bash
.venv/bin/python benchmarks/static_gene_embeddings/pipeline/run_static_embeddings.py
.venv/bin/python benchmarks/static_gene_embeddings/pipeline/build_notebook.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  benchmarks/static_gene_embeddings/static_gene_embeddings.ipynb \
  --ExecutePreprocessor.timeout=600
```

The default uses the exact intersection between the existing 40,000-sample
canonical expression cache and the checkpoint's 640,000-sample training
manifest. This local cache is human-only. The resulting cohort is frozen in `training_expression_cohort.csv`
(with a local Parquet copy for efficient reuse).
This is a training-set reference subset, not the mean over all 640,000 training
samples. Add `--exact-full-training` to reconstruct the full mean from the raw
compressed H5 sources; this is much slower because the normalized training
shards are no longer local. Progress is written to
`results/static_embeddings/run.log`:

```bash
tail -f benchmarks/static_gene_embeddings/results/static_embeddings/run.log
```

For a raw-H5 end-to-end validation, combine `--exact-full-training
--max-training-samples 200` with a separate output directory.

## Outputs

- `results/static_embeddings/gene_embedding_analysis.csv`: gene identifiers,
  mean training TPM, t-SNE coordinates, and cluster assignments.
- `results/static_embeddings/cluster_summary.csv`: cluster sizes and expression
  summaries.
- `results/static_embeddings/{go,kegg}_enrichment.csv`: complete over-
  representation results, including non-significant terms.
- `results/static_embeddings/figures/`: PNG and PDF figures used by the notebook.
- `results/static_embeddings/provenance.json`: frozen inputs, hashes, parameters,
  mapping coverage, and limitations.

The full 512-dimensional embedding is a regenerable checkpoint-derived cache
under `work/` and is intentionally not versioned.

## Extended geometry characterization

For fast hypothesis screening across Raw, L2/Cosine, and post-hoc LayerNorm
representations:

```bash
.venv/bin/python benchmarks/static_gene_embeddings/pipeline/characterize_static_geometry.py --exploratory
```

The exploratory profile uses three seeds, three K values, three K-means
initializations, 10 annotation nulls, and one shuffled-vector null. It is suitable
for deciding whether a larger confirmatory run is warranted, but not for precise
tail probabilities. Omit `--exploratory` for the confirmatory defaults.

The lightweight contextual-depth follow-up is run with:

```bash
.venv/bin/python benchmarks/static_gene_embeddings/pipeline/contextual_depth_followup.py
```

It retains individual gene tokens for eight tissue-diverse TCGA tumors and tests
L0, L1, L6, and L12 using within-sample neighbors. It never mean-pools genes.

The balanced normal-tissue replication and matched tumor comparison is run with:

```bash
.venv/bin/python benchmarks/static_gene_embeddings/pipeline/matched_gtex_tcga.py
```

This freezes 10 contexts × 20 samples in each dataset. GTEx and TCGA share the
same canonical `log1p(TPM)` contract, frozen model, layers, neighbor definitions,
clustering parameters, null controls, visualization gene sample, and seeds.

Cohort-level contextual gene modules and expression controls are generated with:

```bash
.venv/bin/python benchmarks/static_gene_embeddings/pipeline/contextual_gene_modules.py
.venv/bin/python benchmarks/static_gene_embeddings/pipeline/contextual_module_expression_controls.py
```

These cluster the original 512-D cohort-mean gene tokens, not a 2-D projection.
The second command compares expression-decile modules and repeats clustering after
linearly residualizing cohort mean `log1p(TPM)` from every embedding dimension.
