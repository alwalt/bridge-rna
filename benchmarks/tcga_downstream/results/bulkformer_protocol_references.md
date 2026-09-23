# BulkFormer protocol references

Accessed 2026-09-22.

## Final release (authoritative)

- Kang B, Fan R, Yi M, Cui C, Cui Q. “BulkFormer: A large-scale foundation model for bulk transcriptomes.” *Cell Systems* 17(7), 101657 (2026). DOI: <https://doi.org/10.1016/j.cels.2026.101657>; PMID: <https://pubmed.ncbi.nlm.nih.gov/42385705/>.
- Official repository: <https://github.com/KangBoming/BulkFormer>, commit `5bcf5b99e799cd55f41e4d6fe8200b2cdf3d9589` (2026-07-03). The final README reports TCGA classification weighted F1 = 0.907. The repository includes inference/model code but no downstream classifier, fold, or evaluation implementation.
- Official released data: <https://doi.org/10.5281/zenodo.15744294>. `TCGA_survival.h5ad` SHA-256: `3ae462695aafc5c8399b77ff24ff79e209d7e52f2e51e9452642548c1d5b4414`.

The released TCGA object contains 10,429 sample rows, 9,679 unique 12-character TCGA patient barcodes, 33 cancer labels, 750 extra rows from repeated patients, 709 sample-type-code-11 normal aliquots, and explicit `OS` and `OS.time` fields. Thus, the final released cohort is not a one-row-per-patient primary-tumor cohort.

The exact released labels are ACC, BLCA, BRCA, CESC, CHOL, COAD, DLBC, ESCA, GBM, HNSC, KICH, KIRC, KIRP, LAML, LGG, LIHC, LUAD, LUSC, MESO, OV, PAAD, PCPG, PRAD, READ, SARC, SKCM, STAD, TGCT, THCA, THYM, UCEC, UCS, and UVM. GBM and LGG are separate classes. Released sample-row counts by class are 77, 425, 1,210, 308, 45, 328, 47, 195, 165, 563, 89, 603, 320, 161, 521, 420, 565, 542, 86, 425, 183, 185, 546, 102, 264, 455, 442, 137, 571, 120, 193, 57, and 79, respectively.

The final repository's extraction notebook documents raw-count to natural `log1p(TPM)` preprocessing, a native 20,010-gene Ensembl vocabulary, −10 for missing genes, and sample pooling choices. Its worked sample-level example uses mean pooling. It does not identify which pooling option produced the final 0.907 value.

The final article is closed-access through the available programmatic endpoints. Its metadata, abstract, final repository, release history, and released downstream object were inspected. The final supplementary methods and downstream training code were not publicly retrievable from those endpoints. Consequently, classifier hyperparameters, PCA dimensionality, fold assignments, random seed, patient grouping, and the final prognosis headline values remain undocumented in the accessible final release and are not guessed here.

## Preprint (historical only)

- Kang et al. “A large-scale foundation model for bulk transcriptomes.” bioRxiv (2025), DOI: <https://doi.org/10.1101/2025.06.11.659222>.

The 2025 preprint described 33-class TCGA classification and alive/dead prognosis using frozen sample embeddings, PCA, a random forest, and 10-fold cross-validation. It reported weighted F1 = 0.833 for classification and AUROC/AUPRC = 0.747/0.549 for prognosis. Its methods stated max pooling. These are not treated as final values: the final README changes classification to 0.907, the final notebook demonstrates mean pooling, and the final release changed model architecture/data/code.

## Local reconstruction policy

The local parity analyses retain only one eligible specimen per patient: primary tumors for solid cancers and peripheral-blood primary cancer for LAML. They use all 33 labels, frozen native-vocabulary representations, mean pooling for BulkFormer, training-fold-only PCA-128, scikit-learn's default random forest with a fixed per-fold seed, and 10 reproducible patient-level stratified folds. This is the closest conservative reconstruction supported by the accessible final and historical evidence, not a claim of exact reproduction of the authors' 0.907.
