# Contextual Gene Network Utility

This standalone benchmark tests whether frozen BridgeRNA layer-12 contextual
gene relationships provide scientist-facing utility beyond conventional
coexpression and globally pooled sample embeddings.

## Scope

- Fixed nodes: the canonical 15,165 genes in token order.
- Primary graph: exact contextual-cosine directed kNN (`k=10`), union
  symmetrized with contextual cosine edge weights.
- Prespecified sensitivity: `k=5` and `k=20`.
- Frozen checkpoint `r7hnr92k`; existing graph caches are reused, so no new
  BridgeRNA inference is performed.
- Existing tissue, exercise, OSDR liver, and controlled T-cell cohorts only.

## Run

```bash
set -o pipefail
.venv/bin/python benchmarks/contextual_gene_network_utility/pipeline/run_benchmark.py \
  2>&1 | tee benchmarks/contextual_gene_network_utility/results/run.log
.venv/bin/python benchmarks/contextual_gene_network_utility/pipeline/build_notebook.py
```

Monitor a detached run with:

```bash
tail -f benchmarks/contextual_gene_network_utility/results/run.log
```

## Outputs

- `results/relationship_recovery/`: pathway co-membership prediction versus raw
  expression correlation using expression/degree-matched negatives.
- `results/perturbation_rewiring/`: gained/lost relationships and rewired genes.
- `results/cross_study/`: pairwise contextual-network response similarities.
- `results/cross_species/`: human-mouse exercise network conservation.
- `results/attribution_validation/`: overlap with existing IG/edgeR results.
- `results/figures/` and `results/summary/`: final figures and provenance.

## Limitations

The Top-20 cache contains exact similarities only for contextual nearest
neighbors. Relationship recovery therefore uses mean Top-10 edge weight, with
absent edges scored zero; this is a graph-recovery rather than dense all-pairs
cosine test. Edges are contextual relationships, not causal or regulatory
interactions. Communities are secondary because their prior technical-replicate
agreement was poor. No authoritative local PPI or TF-target resource was found,
so those sources are not silently replaced. Graph-guided deletion requires new
inference and is not claimed unless a compatible cache exists.
