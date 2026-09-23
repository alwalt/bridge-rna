# Final BulkFormer gene-essentiality protocol audit

Audit date: 2026-09-22

## Decision (historical-reproduction gate)

**STOP: the exact final evaluation that produces mean PCC = 0.186 cannot be
determined with reasonable confidence from the final public artifacts.**

No model training, BridgeRNA inference, split generation, or baseline
evaluation was performed under the historical-reproduction protocol. In particular, this benchmark does not import the
2025 preprint protocol into the 2026 evaluation. Doing so would silently choose
the correlation axis, folds, readout, missing-value policy, and hyperparameters
that the final release does not specify publicly.

## Subsequent local-benchmark decision

The benchmark was subsequently authorized to continue under a new explicitly
local protocol rather than reverse-engineering the final 0.186. The final
released matrices remain authoritative inputs, while all missing choices are
prespecified in `config.json` and applied identically to locally evaluated
representations. The final-publication 0.186 and preprint 0.931 are retained
only as external literature context and are not implementation targets.

## Sources inspected

1. Kang et al., *BulkFormer: A large-scale foundation model for bulk
   transcriptomes*, Cell Systems 17 (2026), article 101657,
   [DOI](https://doi.org/10.1016/j.cels.2026.101657),
   [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2405471226001390),
   and [PubMed 42385705](https://pubmed.ncbi.nlm.nih.gov/42385705/).
   The public article landing page and bibliographic/text-mining endpoints were
   inspected. The full article text was not openly retrievable from those
   endpoints during this audit.
2. The article's only discoverable supplemental file,
   [`1-s2.0-S2405471226001390-mmc1.pdf`](https://ars.els-cdn.com/content/image/1-s2.0-S2405471226001390-mmc1.pdf).
   It is five pages and contains only Figures S1-S4 plus captions. It contains
   no gene-essentiality methods or tables.
3. The [official BulkFormer repository](https://github.com/KangBoming/BulkFormer)
   at commit
   [`5bcf5b99e799cd55f41e4d6fe8200b2cdf3d9589`](https://github.com/KangBoming/BulkFormer/commit/5bcf5b99e799cd55f41e4d6fe8200b2cdf3d9589),
   including every filename in every reachable commit. The complete history
   contains no gene-essentiality/downstream evaluation script, fixed split,
   target manifest, or `gene_essentiality_data.h5ad`.
4. The final repository [README result table](https://github.com/KangBoming/BulkFormer/blob/5bcf5b99e799cd55f41e4d6fe8200b2cdf3d9589/README.md),
   which reports BulkFormer 0.186 and the seven comparator values as **mean
   PCC**, but does not define the mean.
5. The final [Zenodo record 15744294](https://doi.org/10.5281/zenodo.15744294),
   updated 2025-12-12. Its complete file manifest was inspected through the
   Zenodo API. For essentiality it contains only:

   - `gene_essentiality_expr_data.pkl`, 166,614,826 bytes,
     MD5 `46414f0652f51aa0e77c08eed1b27744`;
   - `gene_essentiality_score.pkl`, 159,091,770 bytes,
     MD5 `1e454ef311f288f1665f43c7b3092f8c`.

   It does **not** contain `gene_essentiality_data.h5ad`, folds, a cell-line
   manifest with DepMap release metadata, trained prediction heads, or
   essentiality code.
6. The 2025 preprint, [bioRxiv 2025.06.11.659222](https://doi.org/10.1101/2025.06.11.659222),
   and the historical official repository result table at commit
   [`379a3f284331f66e204d9aa3e602716c2c4ea59d`](https://github.com/KangBoming/BulkFormer/commit/379a3f284331f66e204d9aa3e602716c2c4ea59d).
   These describe/report the older 0.931 result and are not treated as evidence
   for missing final-release choices.
7. The official repository's December 2025 release notice, which says that all
   previous code, data, and weights were "comprehensively updated" and directs
   users to the latest release.

## Recoverable final data facts

The exact Zenodo pickles already present in the shared main checkout were read
without modifying or copying them. Their MD5 checksums match Zenodo.

| Object | Audited value |
|---|---:|
| Expression matrix | 1,108 cell lines x 18,757 Ensembl genes |
| Dependency-score matrix | 1,108 cell lines x 17,910 Ensembl genes |
| Cell-line indices/order | identical between matrices |
| Genes shared by the two matrices | 17,465 |
| Missing dependency values | 143,614 (0.7237048%) |
| Genes with at least one missing dependency | 837 |
| Cell lines with at least one missing dependency | 907 |
| Expression range | 0 to 17.095489743300227 |
| Dependency range | -5.648909018661887 to 4.436477945322404 |

These facts constrain a future reproduction, but they do not define the final
training examples or metric. The expression range is consistent with a
log-transformed DepMap expression product, but neither the release identifier
nor transformation is encoded in the pickles; it is therefore not promoted to
a fact.

## Requested protocol fields

| # | Field | Final-release finding | Confidence / evidence |
|---:|---|---|---|
| 1 | DepMap release | **Unknown.** | No release/version metadata accompanies the two pickles. Inferring a release from dimensions or values is not exact provenance. |
| 2 | Cell lines | **1,108 in both released matrices.** The number entering each fold/evaluation after missing-data handling is unknown. | Direct pickle inspection. |
| 3 | Target genes | **17,910 score columns.** Whether all columns enter the reported 0.186 is unknown. | Direct pickle inspection; 837 columns contain missing values. |
| 4 | Expression preprocessing | **Unknown.** Values span 0-17.0955; the precise source matrix, transformation, and any model-specific normalization are not documented with the resource. | Zenodo pickle plus repository extraction notebook; range alone is insufficient provenance. |
| 5 | Gene filtering | Released expression and target dimensions are known; the criteria and any post-release evaluation filtering are **unknown**. | No target/filter manifest or evaluation code. |
| 6 | Contextual embedding extraction | The older preprint describes final-layer, sample-conditioned gene representations. The final repository exposes generic feature extraction for the updated model, but no essentiality-specific layer/tensor selection or comparator harmonization code. Therefore the exact final extraction is **unknown**. | Preprint is supporting, not authoritative final-protocol evidence. |
| 7 | Training-example shape | Plausibly one sample-gene representation per observed dependency value, but **not established**. It could instead batch/output genes jointly or use another construction. | No downstream code or methods statement sufficient to distinguish these cases. |
| 8 | MLP architecture | **Unknown** (layer widths, activations, dropout, normalization, scalar versus vector output). | Not in final repository, Zenodo, or supplement. |
| 9 | Split/CV procedure | **Unknown.** The old preprint mentions 10-fold CV, but no final fold count, split axis, assignments, validation split, grouping, or refit procedure is distributed. | Reusing the preprint would violate the final-release authority requirement. |
| 10 | Loss | **Unknown.** | No downstream code/method. |
| 11 | Optimizer | **Unknown.** | No downstream code/method. |
| 12 | Hyperparameters | **Unknown** (learning rate, batch size, epochs, weight decay, dropout, scheduler, patience, initialization). | No downstream code/method. |
| 13 | Seeds/repetitions | **Unknown.** | No seed list, repetition count, or fold file. |
| 14 | Definition of mean PCC | **Unknown.** | The README labels the metric only as "mean PCC". |
| 15 | PCC aggregation axis | **Unknown:** per gene across held-out cell lines, per cell line across genes, per fold, per run/seed, or another aggregation cannot be distinguished. | No predictions or metric code are released. |
| 16 | Production of 0.186 | **Not reproducible from public artifacts without guessing.** | The reported scalar and input/target matrices are available, but the prediction-generating and aggregation procedures are not. |

## Why 0.931 changed to 0.186

The exact methodological cause is **not publicly established**.

What can be stated:

- The old public result table reported PCC 0.931; the final table reports mean
  PCC 0.186.
- The old description used 1,103 cell lines and 17,862 protein-coding genes;
  the final released matrices contain 1,108 cell lines, 18,757 expression
  genes, and 17,910 dependency genes.
- The authors explicitly state that the release comprehensively updated code,
  data, and model weights.
- The metric label changed from `PCC` to `mean PCC`.

What cannot be claimed from the evidence is that any one of those changes
caused the decrease. In particular, saying that the final score is a per-gene
mean whereas 0.931 was a global or per-cell-line correlation is an attractive
hypothesis, not a documented result. This audit does not present it as fact.

## Information required to reopen the gate

Any authoritative combination that resolves the following would be sufficient:

1. final essentiality evaluation/training source at the published release;
2. exact fold assignments and split axis;
3. exact training-example construction and missing-target policy;
4. prediction-head architecture, loss, optimizer, hyperparameters, stopping,
   and seeds/repetitions;
5. executable PCC aggregation code (including NaN/constant-vector policy);
6. DepMap release and exact expression preprocessing/filtering provenance; and
7. the claimed `gene_essentiality_data.h5ad`, if it is an author-provided final
   artifact, with its source URL and checksum.

Until these are available, a Bridge result could be a useful new benchmark but
could not honestly be described as reproducing the final 0.186 evaluation "as
closely as possible" under the requested no-guessing rule.
