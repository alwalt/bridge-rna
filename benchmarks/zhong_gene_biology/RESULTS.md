# Zhong gene-biology benchmark: final results

## Objective

This benchmark tests whether frozen Bridge gene representations encode established biological information at two scales: individual gene attributes and gene–gene relationships. It adapts the fixed splits and probe design from Zhong et al., *Benchmarking gene embeddings from sequence, expression, network, and text models for functional prediction tasks*, to the Bridge vocabulary. All numbers below are frozen outputs; the finalization step only audited and summarized saved results.

## Representations

- **E0 — static gene embedding:** `model_state_dict['gene_embedding.weight']`, before expression rotary encoding or transformer processing. Shape: 15,165 × 512.
- **E1 — null-context L12:** final layer-12 hidden state when every gene receives `log1p(TPM) = 0`. Expression is uniform across genes; E1 is not a biological or average context. Shape: 15,165 × 512.
- **E2 — GTEx-200 real-context L12:** unnormalized arithmetic mean of final layer-12 gene states across a frozen, tissue-balanced panel of 200 real GTEx transcriptomes (10 tissues × 20 samples), using natural `log1p(TPM)`. Shape: 15,165 × 512.

All representations use checkpoint SHA256 `f3491beaa697a0408dc1daeb6f8d648566bd46bafc6e44938d723e7ffc325541`, the canonical token order, a 512-dimensional hidden state, mask value `-10`, and no species embedding. Full details and file hashes are recorded in `results/final_benchmark_provenance.json`.

## Single-gene results

Zhong reports mean AUROC across terms, so the headline AUROC is the unweighted mean of one final held-out value per biological term. Medians are retained for reference. AUPRC is supplementary because no corresponding Zhong statistic was supplied.

| Task | Representation | Terms | Mean AUROC | Median AUROC | Mean AUPRC | Median AUPRC |
|---|---|---:|---:|---:|---:|---:|
| GO | E0 | 56 | 0.5177 | 0.5156 | 0.1321 | 0.1173 |
| GO | E1 | 56 | 0.7722 | 0.7736 | 0.3668 | 0.3347 |
| GO | E2 | 56 | 0.7811 | 0.7876 | 0.4019 | 0.3781 |
| OMIM | E0 | 103 | 0.4891 | 0.4950 | 0.1609 | 0.1111 |
| OMIM | E1 | 103 | 0.7354 | 0.7438 | 0.3879 | 0.3251 |
| OMIM | E2 | 103 | 0.7596 | 0.7536 | 0.4294 | 0.3999 |

The dominant transition is E0 to E1. E2 produces a smaller additional improvement for both single-gene tasks.

## Paired-gene results

Values are mean ± sample SD across three outer folds. Every outer fold is gene-disjoint: train/test gene overlap is zero in all nine task-fold records.

| Task | Rep. | Operation | AUROC | AUPRC |
|---|---|---|---:|---:|
| Negative genetic | E0 | concat | 0.546 ± 0.052 | 0.119 ± 0.052 |
| Negative genetic | E0 | product | 0.517 ± 0.015 | 0.168 ± 0.012 |
| Negative genetic | E0 | sum | 0.569 ± 0.039 | 0.172 ± 0.037 |
| Negative genetic | E1 | concat | 0.566 ± 0.040 | 0.129 ± 0.026 |
| Negative genetic | E1 | product | 0.565 ± 0.049 | 0.145 ± 0.018 |
| Negative genetic | E1 | sum | 0.564 ± 0.039 | 0.151 ± 0.044 |
| Negative genetic | E2 | concat | **0.580 ± 0.028** | 0.133 ± 0.024 |
| Negative genetic | E2 | product | 0.575 ± 0.072 | 0.160 ± 0.039 |
| Negative genetic | E2 | sum | 0.578 ± 0.033 | 0.143 ± 0.018 |
| Synthetic lethal | E0 | concat | 0.417 ± 0.145 | 0.081 ± 0.010 |
| Synthetic lethal | E0 | product | 0.470 ± 0.027 | 0.098 ± 0.042 |
| Synthetic lethal | E0 | sum | 0.472 ± 0.067 | 0.087 ± 0.026 |
| Synthetic lethal | E1 | concat | 0.608 ± 0.122 | 0.153 ± 0.090 |
| Synthetic lethal | E1 | product | **0.707 ± 0.122** | **0.228 ± 0.137** |
| Synthetic lethal | E1 | sum | 0.611 ± 0.139 | 0.157 ± 0.105 |
| Synthetic lethal | E2 | concat | 0.477 ± 0.136 | 0.103 ± 0.049 |
| Synthetic lethal | E2 | product | 0.589 ± 0.054 | 0.126 ± 0.057 |
| Synthetic lethal | E2 | sum | 0.456 ± 0.131 | 0.096 ± 0.040 |
| TF-target | E0 | concat | 0.516 ± 0.007 | 0.098 ± 0.016 |
| TF-target | E0 | product | 0.523 ± 0.013 | 0.098 ± 0.016 |
| TF-target | E0 | sum | 0.511 ± 0.013 | 0.098 ± 0.007 |
| TF-target | E1 | concat | **0.670 ± 0.040** | 0.186 ± 0.014 |
| TF-target | E1 | product | 0.661 ± 0.055 | 0.176 ± 0.029 |
| TF-target | E1 | sum | 0.668 ± 0.036 | 0.174 ± 0.009 |
| TF-target | E2 | concat | 0.666 ± 0.031 | **0.194 ± 0.008** |
| TF-target | E2 | product | 0.623 ± 0.028 | 0.156 ± 0.007 |
| TF-target | E2 | sum | 0.650 ± 0.035 | 0.172 ± 0.005 |

The highest observed AUROC conditions are E2-concat for negative genetic interactions, E1-product for synthetic lethality, and E1-concat for TF targets. Negative genetic interaction prediction was comparatively weak, indicating that Bridge does not capture all forms of gene–gene biology equally well.

## Coverage and split rigor

- Bridge vocabulary: 15,165 genes; 15,150 unambiguously mapped to unique Entrez identifiers; 11 unmapped and 4 ambiguous mappings excluded.
- Benchmark union: 16,968 genes; 13,913 overlap the mapped Bridge vocabulary.
- Retained positive pairs: synthetic lethal 1,860/2,124 (87.57%); negative genetic 9,054/10,568 (85.67%); TF-target 7,603/10,000 (76.03%).
- Across every pair task and outer fold, train/test gene overlap is exactly zero. The exact train/test gene and pair counts are in `results/final_pair_leakage_qc.csv`.

## Zhong context

Reported Zhong et al. literature values range from GO mean AUROC 0.787 for scGPT-human to 0.926 for GenePT-Model3, and from OMIM 0.685 for ESMB1 to 0.877 for GenePT-Model3 among the selected references. Bridge E1/E2 operate in a similar general range to the reported expression-derived scGPT-human result. These comparisons are contextual only: non-Bridge models were not rerun locally under the Bridge gene universe, so this is not a formal head-to-head ranking or evidence of superiority.

## Interpretation

Bridge's transformer-derived gene representations encode established biological information across gene function, disease association, and selected gene–gene interaction tasks. Functional information is substantially more decodable from transformer-derived representations than from the static gene embedding table. Aggregation across real transcriptomic contexts produces smaller and task-dependent changes.

These results do not demonstrate causal or mechanistic understanding, universal gene-function understanding, batch correction, or superiority to all existing gene embedding models. No additional controls or classifier runs were performed during finalization.
