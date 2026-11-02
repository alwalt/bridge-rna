# Scientific report: tissue and disease annotation

## Scientific question

How linearly accessible are tissue and disease identities from frozen Bridge sample representations?

## Datasets and cohorts

The primary GTEx v8 task contains 8,883 profiles from 549 donors and 51 detailed tissue classes under donor-grouped five-fold evaluation. Cell-derived labels were excluded, and one deterministic row was retained per biospecimen. The supplementary 30-class broad-tissue task uses the same 8,883 profiles and donor-grouped folds. TCGA contains one specimen per patient for 9,942 unique patients and all 33 cancer types under five-fold stratified evaluation; LAML is represented by primary peripheral-blood cancer samples.

## Representations and design

Frozen Bridge L12 final-token mean embeddings (512 dimensions), fold-local PCA-512, and the canonical 15,165-gene natural `log1p(TPM)` matrix were evaluated with the same samples, folds, and multinomial logistic regression. Bridge compresses the raw transcriptome by approximately 29.6-fold. Five fixed folds were used. GTEx folds are donor-grouped; TCGA folds are patient-level stratified. Macro F1 is primary.

## Results

| task | representation | folds | accuracy_mean | accuracy_sd | macro_f1_mean | macro_f1_sd | weighted_f1_mean | weighted_f1_sd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gtex_broad | bridge | 5 | 0.9779 | 0.0044 | 0.9265 | 0.0134 | 0.9773 | 0.0043 |
| gtex_broad | pca | 5 | 0.9907 | 0.0044 | 0.9623 | 0.0244 | 0.9904 | 0.0042 |
| gtex_broad | raw | 5 | 0.9890 | 0.0042 | 0.9595 | 0.0244 | 0.9887 | 0.0040 |
| gtex_detailed | bridge | 5 | 0.9326 | 0.0047 | 0.8772 | 0.0230 | 0.9315 | 0.0049 |
| gtex_detailed | pca | 5 | 0.9632 | 0.0045 | 0.9248 | 0.0283 | 0.9627 | 0.0047 |
| gtex_detailed | raw | 5 | 0.9622 | 0.0024 | 0.9240 | 0.0217 | 0.9616 | 0.0024 |
| tcga_cancer | bridge | 5 | 0.9314 | 0.0063 | 0.9099 | 0.0047 | 0.9299 | 0.0059 |
| tcga_cancer | pca | 5 | 0.9544 | 0.0054 | 0.9378 | 0.0092 | 0.9543 | 0.0054 |
| tcga_cancer | raw | 5 | 0.9592 | 0.0068 | 0.9424 | 0.0098 | 0.9587 | 0.0070 |

Bridge was strongly linearly separable, reaching mean macro F1 0.8772 for 51-class GTEx and 0.9099 for 33-class TCGA. It did not lead either primary task: PCA reached 0.9248 and 0.9378, while raw expression reached 0.9240 and 0.9424, respectively. The Bridge deficit was present in every paired fold. The supplementary 30-class GTEx task produced Bridge macro F1 0.9265, versus 0.9623 for PCA and 0.9595 for raw expression.

### Paired fold differences

| task | fold | bridge | pca | raw | bridge_minus_pca | bridge_minus_raw |
| --- | --- | --- | --- | --- | --- | --- |
| gtex_broad | 0 | 0.9291 | 0.9616 | 0.9533 | -0.0324 | -0.0242 |
| gtex_broad | 1 | 0.9429 | 0.9874 | 0.9821 | -0.0445 | -0.0392 |
| gtex_broad | 2 | 0.9293 | 0.9697 | 0.9671 | -0.0404 | -0.0378 |
| gtex_broad | 3 | 0.9254 | 0.9710 | 0.9748 | -0.0456 | -0.0495 |
| gtex_broad | 4 | 0.9057 | 0.9220 | 0.9203 | -0.0163 | -0.0146 |
| gtex_detailed | 0 | 0.8716 | 0.9200 | 0.9171 | -0.0485 | -0.0455 |
| gtex_detailed | 1 | 0.9109 | 0.9671 | 0.9503 | -0.0563 | -0.0394 |
| gtex_detailed | 2 | 0.8888 | 0.9368 | 0.9406 | -0.0481 | -0.0518 |
| gtex_detailed | 3 | 0.8532 | 0.9007 | 0.9166 | -0.0475 | -0.0634 |
| gtex_detailed | 4 | 0.8615 | 0.8991 | 0.8953 | -0.0376 | -0.0338 |
| tcga_cancer | 0 | 0.9080 | 0.9339 | 0.9402 | -0.0259 | -0.0323 |
| tcga_cancer | 1 | 0.9029 | 0.9239 | 0.9273 | -0.0210 | -0.0243 |
| tcga_cancer | 2 | 0.9132 | 0.9449 | 0.9522 | -0.0317 | -0.0390 |
| tcga_cancer | 3 | 0.9147 | 0.9468 | 0.9498 | -0.0321 | -0.0351 |
| tcga_cancer | 4 | 0.9108 | 0.9392 | 0.9426 | -0.0284 | -0.0317 |

Mean paired differences:

| task | bridge_minus_pca_mean | bridge_minus_raw_mean |
| --- | --- | --- |
| gtex_broad | -0.0359 | -0.0331 |
| gtex_detailed | -0.0476 | -0.0468 |
| tcga_cancer | -0.0278 | -0.0325 |

### Lowest Bridge class-level F1 estimates

| task | class | f1_mean | f1_sd | support |
| --- | --- | --- | --- | --- |
| gtex_broad | Cervix Uteri | 0.3333 | 0.3118 | 11 |
| gtex_broad | Fallopian Tube | 0.4000 | 0.5477 | 7 |
| gtex_broad | Bladder | 0.8800 | 0.2683 | 11 |
| gtex_broad | Small Intestine | 0.8960 | 0.0739 | 104 |
| gtex_broad | Breast | 0.8982 | 0.0336 | 217 |
| gtex_broad | Salivary Gland | 0.9145 | 0.0640 | 69 |
| gtex_broad | Vagina | 0.9165 | 0.0508 | 97 |
| gtex_broad | Uterus | 0.9383 | 0.0136 | 89 |
| gtex_detailed | Cervix - Ectocervix | 0.0000 | 0.0000 | 6 |
| gtex_detailed | Cervix - Endocervix | 0.4000 | 0.5477 | 5 |
| gtex_detailed | Fallopian Tube | 0.4000 | 0.5477 | 7 |
| gtex_detailed | Esophagus - Gastroesophageal Junction | 0.6772 | 0.0670 | 175 |
| gtex_detailed | Brain - Amygdala | 0.7668 | 0.1066 | 80 |
| gtex_detailed | Brain - Anterior cingulate cortex (BA24) | 0.7686 | 0.0610 | 95 |
| gtex_detailed | Brain - Putamen (basal ganglia) | 0.7756 | 0.0523 | 103 |
| gtex_detailed | Brain - Substantia nigra | 0.7756 | 0.0486 | 71 |
| tcga_cancer | READ | 0.4090 | 0.1104 | 166 |
| tcga_cancer | CHOL | 0.6555 | 0.1352 | 36 |
| tcga_cancer | ESCA | 0.7383 | 0.0483 | 184 |
| tcga_cancer | UCS | 0.8011 | 0.0752 | 57 |
| tcga_cancer | COAD | 0.8257 | 0.0233 | 458 |
| tcga_cancer | KICH | 0.8650 | 0.0314 | 66 |
| tcga_cancer | LUSC | 0.8771 | 0.0278 | 501 |
| tcga_cancer | STAD | 0.8846 | 0.0267 | 416 |

The three rarest highlighted GTEx classes require particular caution: Cervix - Ectocervix has N=6 and Bridge F1=0, Cervix - Endocervix has N=5, and Fallopian Tube has N=7. With only approximately one observation per held-out fold, these estimates are unstable; ectocervix F1=0 is not evidence that the representation contains no cervical information. CHOL is likewise small (N=36) and has variable fold-level F1.

### Notable Bridge confusions

- `gtex_detailed`: Esophagus - Gastroesophageal Junction → Esophagus - Muscularis: 53 (30.3%); Skin - Not Sun Exposed (Suprapubic) → Skin - Sun Exposed (Lower leg): 31 (11.5%); Colon - Transverse → Colon - Sigmoid: 30 (14.8%).
- `tcga_cancer`: READ → COAD: 106 (63.9%); COAD → READ: 46 (10.0%); ESCA → STAD: 41 (22.3%).

The dominant TCGA error is READ → COAD (106/166, 63.9%), with the reverse COAD → READ error occurring for 46/458 samples (10.0%). Descriptively, Bridge often places rectal and colon adenocarcinoma in the same transcriptional neighborhood, substantially reducing READ F1. ESCA → STAD occurs for 41/184 samples (22.3%). These errors are descriptive annotation patterns, not mechanistic findings.

## Leakage controls

PCA and every scaler were fitted independently on each training fold. Identical held-out samples and labels were used across representations. GTEx donors never cross folds; TCGA contains one row per patient; repeated GTEx biospecimens were removed. Bridge stayed frozen, and annotation labels never entered representation construction. Sixty canonical genes unavailable in the common source matrix were zero-filled before natural `log1p(TPM)`; 15,105 genes were observed.

## Interpretation and limitations

Frozen Bridge representations retain substantial tissue and cancer identity in a compact 512-dimensional representation: a fixed linear readout achieves mean macro F1 0.877 across 51 detailed GTEx tissues and 0.910 across all 33 TCGA cancer types. This is 94.9% of PCA macro-F1 performance for detailed GTEx and 97.0% for TCGA, reported only as descriptive performance ratios. Fold-local PCA and raw expression remain consistently more linearly separable, by about 0.048 and 0.047 macro-F1 points in GTEx and 0.028 and 0.032 in TCGA. Bridge therefore preserves much—but not all—of the tissue- and disease-identifying information accessible from the original transcriptome; dimensional compression alone does not establish representation superiority. Errors concentrate in biologically adjacent labels, including colorectal, upper gastrointestinal, lung, skin, and related brain regions, as well as extremely small GTEx classes. Rare GTEx tissues have as few as five donors, so their fold-level estimates are noisy. ARCHS4 pretraining used label-free masked-expression reconstruction rather than tissue or cancer labels, but public GTEx and/or TCGA expression may overlap the pretraining compendium. This benchmark therefore demonstrates biological linear separability of frozen Bridge representations, not guaranteed sample-unseen external generalization. GTEx-versus-TCGA normal/tumor classification is deliberately out of scope because dataset and cohort effects would be inseparable from disease state.

## Figures and artifacts

- Main GTEx detailed and TCGA performance panels: [PNG](results/figures/main_annotation_macro_f1.png), [PDF](results/figures/main_annotation_macro_f1.pdf).
- GTEx detailed normalized Bridge confusion matrix: [PNG](results/figures/gtex_detailed_bridge_confusion_normalized.png), [PDF](results/figures/gtex_detailed_bridge_confusion_normalized.pdf).
- TCGA normalized Bridge confusion matrix: [PNG](results/figures/tcga_cancer_bridge_confusion_normalized.png), [PDF](results/figures/tcga_cancer_bridge_confusion_normalized.pdf).
- Supplementary GTEx broad confusion matrix: [PNG](results/figures/gtex_broad_bridge_confusion_normalized.png), with broad/detailed and TCGA per-class plots under `results/figures/`.
- Exact manifests, folds, predictions, per-class metrics, fold metrics, provenance, configuration, asset hashes, and QC outputs are under `results/`.

## Reproducibility

Exact manifests, folds, predictions, fold metrics, per-class tables, normalized confusion matrices, asset hashes, and QC assertions are stored alongside this report. Large derived arrays remain git-ignored under `work/`.
