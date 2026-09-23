# Bridge Drug-Response Prediction Benchmark

## 1. Scientific Question

> Does a frozen Bridge representation of baseline transcriptomic state contain information that predicts drug sensitivity in held-out cell lines?

This is a pharmacogenomic drug-response benchmark. It predicts experimentally measured IC50 from baseline expression representations. It does not test unseen-drug generalization, post-treatment expression prediction, transcriptional drug reversal, drug repurposing, or therapeutic efficacy.

## 2. Motivation

Drug-response prediction is one of the downstream applications evaluated by BulkFormer, making it a useful test of whether Bridge transfers to a pharmacologically relevant phenotype. The benchmark asks whether a frozen transcriptomic representation captures cellular state relevant to differential drug sensitivity without fine-tuning the representation model.

## 3. Dataset

The benchmark uses the final processed data released with BulkFormer:

| Dataset property | Final count |
|---|---:|
| DepMap cell lines | 700 |
| Drug-response observations | 212,299 |
| GDSC drug IDs | 295 |
| Unique PubChem compounds | 255 |
| Dataset-version-specific drug screens | 355 |
| Expression genes | 19,098 |

The endpoint is the released IC50 value. The expression table supplies one baseline profile per cell line and is not a post-treatment expression dataset.

GDSC1 and GDSC2 are treated as distinct dataset versions. A `(cell line, drug ID)` key is not unique across versions: 60 drug IDs occur in both GDSC1 and GDSC2, producing 65,546 rows that participate in repeated cross-version `(cell line, drug ID)` keys. By contrast, every `(cell line, dataset version, drug ID)` observation is unique. The evaluation therefore treats each dataset-version-specific drug screen as an independent response target, giving 355 screens with 134–697 observations each (median 658).

Of the 19,098 released expression genes, 15,130 overlap Bridge's ordered 15,165-gene vocabulary, and 18,916 overlap the 20,007 unique symbols in the released BulkFormer vocabulary table. Detailed counts and checks appear in [`dataset_audit.md`](dataset_audit.md) and [`results/dataset_audit.json`](results/dataset_audit.json).

## 4. Experimental Design

The common local protocol uses five deterministic folds. Complete cell lines are held out: no response measurement from a test cell line enters downstream readout training. Thus the tested generalization axis is:

> **known drug → unseen cell line**

It is not an unseen-drug benchmark. Each dataset-version-specific screen has a separate linear response output head. This representation of drug identity supports interpolation for drugs with training observations, not extrapolation to a new drug.

Bridge and both BulkFormer encoders remain frozen. All representations use the same response observations, folds, IC50 endpoint, train-only response completion rule, and dimension-scaled multi-output ridge readout rule. PCA is fitted separately using only the training cell lines in each fold. Predictions are assembled out of fold. PCC, SCC, and RMSE are then calculated within each screen over its out-of-fold cell-line predictions. The primary metric is the unweighted mean of the 355 per-screen PCC values.

Here, PCC measures agreement between predicted and measured variation in IC50 across held-out cell lines within one screen. A higher PCC means that the readout more accurately predicts and orders differential sensitivity among held-out cell lines for that screen.

## 5. Representations Compared

- **Bridge-45.6M:** frozen standard 512-dimensional Bridge sample representation.
- **BulkFormer-50M:** frozen mean-pooled final contextual gene-token representation.
- **BulkFormer-147M:** frozen mean-pooled final contextual gene-token representation.
- **Raw expression:** all 19,098 released expression features with a regularized linear readout.
- **PCA-128:** training-fold PCA with at most 128 components.
- **PCA-512:** training-fold PCA with rank-limited dimensionality up to 512 components.
- **PCA-128 + Bridge:** concatenation of the training-fold PCA-128 features and frozen Bridge representation.

All frozen-model comparisons use identical observations, folds, endpoint, and downstream evaluation protocol.

## 6. Primary Results

| Representation | Mean PCC | Median PCC | Mean SCC | RMSE |
|---|---:|---:|---:|---:|
| Bridge-45.6M | **0.448** | **0.461** | 0.424 | **1.235** |
| BulkFormer-50M | 0.438 | 0.450 | 0.414 | 1.238 |
| BulkFormer-147M | 0.436 | 0.452 | 0.416 | 1.244 |
| Raw expression | 0.196 | 0.156 | **0.440** | 2.257 |
| PCA-128 | 0.088 | 0.065 | 0.424 | 3.499 |
| PCA-512 | 0.031 | 0.021 | 0.348 | 4.002 |
| PCA-128 + Bridge | 0.284 | 0.273 | 0.432 | 1.587 |

All 355 screens were evaluable for every representation. Unrounded values are preserved in [`results/primary_results.csv`](results/primary_results.csv) and [`results/per_drug_metrics.csv`](results/per_drug_metrics.csv).

Bridge achieved the highest mean per-screen PCC, median per-screen PCC, and lowest mean per-screen RMSE under the common local protocol. Raw expression had the highest mean SCC, illustrating that metric choice and calibration matter: its rank association was stronger than its linear correlation and absolute prediction error.

## 7. Paired Model Comparisons

Comparisons are paired by dataset-version-specific screen. The confidence intervals resample the 355 screens with replacement.

| Comparison | Mean ΔPCC | Median ΔPCC | 95% bootstrap CI | Fraction Bridge higher |
|---|---:|---:|---:|---:|
| Bridge − BulkFormer-50M | +0.0102 | +0.0088 | [+0.0083, +0.0122] | 71.8% |
| Bridge − BulkFormer-147M | +0.0120 | +0.0117 | [+0.0092, +0.0147] | 74.6% |
| Bridge − PCA-512 | +0.4176 | +0.4291 | [+0.4042, +0.4308] | 100.0% |

These values come from [`results/paired_comparisons.csv`](results/paired_comparisons.csv). That file contains bootstrap confidence intervals but no paired p-values, so no p-values are reported or inferred here.

The differences from the locally evaluated BulkFormer representations are small but consistent. The supported interpretation is that Bridge has a modest advantage under this particular common protocol, not that the checkpoints differ dramatically in pharmacogenomic utility.

## 8. Raw Expression and PCA Controls

Bridge substantially exceeds raw expression, PCA-128, and PCA-512 in mean PCC under this fixed linear-readout protocol. This difference is not the primary evidence for Bridge superiority because high-dimensional raw/PCA performance is sensitive to feature dimensionality, regularization, and the undocumented processing of the released expression matrix.

The complementarity comparison gives:

- PCA-128 + Bridge: mean PCC 0.284
- PCA-128: mean PCC 0.088
- Bridge: mean PCC 0.448

PCA-128 + Bridge exceeds PCA-128 by a paired mean ΔPCC of +0.1958, with a 95% bootstrap interval of [+0.1840, +0.2075], and is higher on 99.4% of screens. Bridge therefore contributes substantial predictive information beyond PCA-128. However, PCA-128 + Bridge remains below Bridge alone, so PCA does not improve prediction beyond Bridge in this experiment.

The association checks in [`results/variance_controls.csv`](results/variance_controls.csv) do not show that Bridge's advantage over BulkFormer-147M is concentrated in screens with higher response variance or more observations. The corresponding associations for other comparisons are small to moderate and remain descriptive rather than causal.

## 9. Relationship to the Published BulkFormer Result

The final BulkFormer publication reports **mean PCC = 0.373**. This value is verified in the final paper's Table S1 and official repository, and is retained in [`results/published_results.csv`](results/published_results.csv) as literature context.

The local values cannot be numerically compared directly with 0.373 because the final publication and release do not expose enough information to reconstruct the exact train/test folds, downstream prediction head, drug grouping, or PCC aggregation procedure. The valid controlled comparison is therefore:

- Bridge-45.6M: 0.448
- locally evaluated BulkFormer-50M: 0.438
- locally evaluated BulkFormer-147M: 0.436

The published 0.373 is not used as evidence that Bridge outperforms published BulkFormer. The distinction between the final release and the materially different 2025 preprint protocol is documented in [`bulkformer_drug_response_protocol_audit.md`](bulkformer_drug_response_protocol_audit.md).

## 10. Scientific Interpretation

> Frozen Bridge representations encode baseline cellular state that is informative for predicting drug sensitivity in previously held-out cell lines.

Bridge performs comparably to, and slightly better than, the two locally evaluated BulkFormer checkpoints under the common protocol. This demonstrates transfer of a frozen Bridge representation to a pharmacogenomic phenotype without fine-tuning the Bridge encoder.

The result is biologically plausible because baseline transcription reflects lineage, pathway activity, signaling state, metabolism, stress programs, and other cellular properties associated with differential drug sensitivity. Bridge compresses this information into a 512-dimensional sample representation that remains predictive under a simple downstream readout.

This benchmark establishes predictive information, not causal drug-response biology. It does not identify mechanisms, validate biomarkers, or establish that changing an encoded transcriptional program would change drug response.

## 11. Relationship to Other Bridge Drug Benchmarks

These drug-related tasks answer distinct questions and their metrics must not be combined.

### Drug-response prediction

> Given baseline cellular state and a known drug screen, can Bridge predict IC50 in an unseen cell line?

This report addresses that question.

### Drug perturbation prediction

> Can a model predict the transcriptional change caused by a drug?

That separate task predicts a post-treatment expression change, not drug sensitivity.

### Drug discovery / target mapping

> Do Bridge-derived condition programs identify pharmacologically targetable genes or drugs?

That separate task concerns targetability and candidate mapping rather than measured cell-line IC50.

### Drug reversal

> Does an experimentally observed drug perturbation oppose a disease- or condition-associated transcriptional program?

That direction-aware task compares transcriptional signatures rather than predicting pharmacological response from baseline state.

## 12. Limitations

1. **No unseen-drug generalization.** Each drug screen has training observations and its own downstream output head. The benchmark does not establish performance for a new compound.
2. **BulkFormer protocol uncertainty.** The exact final published BulkFormer downstream protocol cannot be reconstructed, so local results are not a reproduction of published mean PCC 0.373.
3. **Expression preprocessing uncertainty.** The released expression matrix contains 1,782,260 negative processed values and does not provide an invertible raw-TPM transformation. Frozen-model inputs therefore use a documented approximation—negative values clipped to zero, remaining values treated as `log2(TPM+1)`, and converted to natural-log units—rather than exact natural `log1p(TPM)`.
4. **Raw/PCA sensitivity.** High-dimensional controls depend materially on regularization, dimensionality, and readout specification. Their gap from Bridge should not be interpreted independently of those choices.
5. **ARCHS4 exposure.** Exact Bridge pretraining exposure is unresolved because the released dataset lacks GEO/SRA sample and study accessions. Held-out cell lines guarantee downstream split isolation, not pretraining novelty.
6. **Representation benchmark.** This experiment measures predictive information in frozen representations. It is not an optimized state-of-the-art pharmacogenomic prediction system and does not exhaust downstream architectures or hyperparameter searches.

## 13. Final Conclusion

> Under a common held-out-cell-line evaluation, frozen Bridge representations achieved the highest mean per-screen drug-response correlation (PCC = 0.448), compared with 0.438 for BulkFormer-50M and 0.436 for BulkFormer-147M. Bridge was higher than the two BulkFormer representations on 71.8% and 74.6% of screens, respectively. These results support the conclusion that Bridge encodes transferable cellular-state information relevant to pharmacological response. Because the final published BulkFormer downstream protocol cannot be reconstructed exactly, the result should be interpreted as a controlled local comparison rather than a reproduction or direct improvement over the published BulkFormer PCC.

## 14. Reproducibility

The definitive benchmark files are:

- Executed report notebook: [`drug_response.ipynb`](drug_response.ipynb)
- Frozen configuration: [`config.json`](config.json)
- Dataset audit: [`dataset_audit.md`](dataset_audit.md), [`results/dataset_audit.json`](results/dataset_audit.json)
- BulkFormer protocol audit: [`bulkformer_drug_response_protocol_audit.md`](bulkformer_drug_response_protocol_audit.md)
- Asset manifest: [`asset_manifest.md`](asset_manifest.md), [`results/asset_manifest.json`](results/asset_manifest.json)
- Pretraining-exposure audit: [`pretraining_exposure_audit.md`](pretraining_exposure_audit.md)
- Primary results: [`results/primary_results.csv`](results/primary_results.csv)
- Published result reference: [`results/published_results.csv`](results/published_results.csv)
- Paired comparisons: [`results/paired_comparisons.csv`](results/paired_comparisons.csv)
- Per-screen metrics: [`results/per_drug_metrics.csv`](results/per_drug_metrics.csv)
- Variance controls: [`results/variance_controls.csv`](results/variance_controls.csv)
- Cell-line split definitions: [`results/cell_line_splits.csv`](results/cell_line_splits.csv)
- Observation manifest: [`results/observation_manifest.parquet`](results/observation_manifest.parquet)
- Cell-line manifest: [`results/cell_line_manifest.csv`](results/cell_line_manifest.csv)
- Gene-overlap table: [`results/gene_overlap.csv`](results/gene_overlap.csv)
- Fold diagnostics and provenance: [`results/fold_diagnostics.csv`](results/fold_diagnostics.csv), [`results/evaluation_provenance.json`](results/evaluation_provenance.json), [`results/input_preprocessing.json`](results/input_preprocessing.json)
- Pipeline source: [`pipeline/audit_data.py`](pipeline/audit_data.py), [`pipeline/prepare_inputs.py`](pipeline/prepare_inputs.py), [`pipeline/make_splits.py`](pipeline/make_splits.py), [`pipeline/extract_embeddings.py`](pipeline/extract_embeddings.py), [`pipeline/evaluate.py`](pipeline/evaluate.py), and [`pipeline/make_figures.py`](pipeline/make_figures.py)

Large released inputs, checkpoints, model-aligned matrices, embeddings, and out-of-fold prediction arrays remain outside version control or under the ignored [`work/`](work/) directory. Exact external asset paths and checksums are recorded in the asset manifest.
