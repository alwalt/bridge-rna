# Complete five-fold OOF TCGA survival evaluation

Source: [Publication review](PUBLICATION_REVIEW.md).

## Methods

- Evaluate whether a frozen Bridge representation retains censoring-aware prognostic information relative to PCA-128 and full raw expression.
- Use 9,668 eligible TCGA patients across 32 cancers with 2,774 PanCanAtlas overall-survival events; require positive `OS.time`, a valid binary `OS` indicator, and one deterministic eligible specimen per patient.
- Use frozen Bridge 45.6M mean-pooled embeddings (512 dimensions; ordered 15,165-gene vocabulary; natural `log1p(TPM)`), training-fold-only PCA-128, and full expression (`log1p(CPM)`, 25,150 features).
- Assign every patient to exactly one of five mutually exclusive test folds balanced within cancer × event strata. Fit all learned preprocessing on training patients only and keep Bridge frozen.
- Train both a bias-free linear Cox head and Cox MLP `[512,256]` with Cox partial likelihood for every representation on identical folds and outcomes.
- Aggregate concordant and comparable pairs within held-out folds. Estimate within-cancer Harrell C-indices and pointwise 95% intervals using 1,000 patient-level bootstrap draws that retain fold membership; display events/N and stability flags.
- Compare representations using paired cancer-level sign-flip and Wilcoxon tests with Benjamini–Hochberg correction. Treat the L1–L12 depth analysis as exploratory and retain prespecified L12 as primary.

## Results

- Every eligible patient received exactly one held-out prediction per representation and head; all six combinations used 9,668 patients and 3,068,664 fold-valid comparable pairs.
- Cox-MLP C-indices were 0.7580 for Bridge, 0.7638 for PCA-128, and 0.7590 for full expression. Linear-Cox C-indices were 0.7589, 0.7615, and 0.7533, respectively.
- Bridge was nearly insensitive to head family. No representation or head contrast remained significant after Benjamini–Hochberg correction across seven paired cancer-level tests.
- Bridge's median within-cancer C-index was 0.6239 and its patient-weighted mean was 0.6247, versus medians of 0.6360 for PCA-128 and 0.6008 for full expression.
- Bridge exceeded PCA-128 in 15/32 cancers and full expression in 15/32. Median paired differences were −0.0024 versus PCA and −0.0030 versus full expression; sign-flip/Wilcoxon p-values were 0.761/0.733 and 0.995/0.963.
- Bridge pointwise intervals lay wholly above 0.5 in 17 cancers. Supported estimates with useful event counts included KIRP (0.858; 44 events), ACC (0.821; 28), LGG (0.775; 125), KIRC (0.687; 174), UCEC (0.686; 90), CESC (0.662; 71), LIHC (0.659; 130), and LUAD (0.654; 182).
- CHOL, DLBC, KICH, PCPG, PRAD, TGCT, THCA, and THYM remained difficult to interpret because of sparse events. L12 had the highest overall and median within-cancer Bridge performance; no earlier layer differed systematically after multiplicity correction.

## Discussion

- Frozen Bridge embeddings preserve substantial survival-ranking information while reducing the ordered 15,165-gene input to a reusable 512-dimensional representation.
- Similar linear- and nonlinear-head performance suggests that prognostic information is readily accessible in the frozen Bridge space rather than created primarily by a complex downstream head.
- Bridge performed in the same broad range as PCA-128 and full expression, without a systematic cancer-level advantage. This supports information preservation rather than superior survival prediction.
- Complete OOF coverage, matched folds, fold-aware concordance, event support, and patient-bootstrap uncertainty make the cancer-specific comparison auditable.
- Published frozen bulk-RNA embedding studies reported pooled Cox C-indices of approximately 0.750–0.765, placing Bridge's 0.7580–0.7589 in the same numerical range. These are literature-only comparisons because cohorts, preprocessing, pretraining exposure, and split designs differ ([Gélard et al.](https://proceedings.mlr.press/v259/gelard25a.html); [multimodal survival study](https://arxiv.org/abs/2505.07683)).
- A separate study applying scFoundation to bulk TCGA data found that embeddings were comparable to raw expression under Ridge–Cox (mean cancer-level C-index 0.680 ± 0.086 versus 0.686 ± 0.089; P = 0.211) but better under Lasso–Cox (0.670 ± 0.081 versus 0.603 ± 0.120; P = 2.17 × 10⁻⁵). This reinforces that representation conclusions can depend on downstream regularization and supports matched-head evaluation ([Liu et al.](https://pubmed.ncbi.nlm.nih.gov/41909608/)).
- Complete coverage cannot overcome intrinsically sparse events, and pointwise cancer-specific intervals should not be interpreted as multiplicity-adjusted discoveries.
- Retrospective TCGA evaluation, absent clinical covariates and external validation, heterogeneous event support, and untested practical benefits of compression limit clinical interpretation.

## Suggested Results paragraph

In complete five-fold out-of-fold evaluation of 9,668 TCGA patients with 2,774 overall-survival events, frozen Bridge representations achieved fold-aware C-indices of 0.7580 with a Cox MLP and 0.7589 with linear Cox. Corresponding values were 0.7638/0.7615 for PCA-128 and 0.7590/0.7533 for full expression. Bridge confidence intervals were wholly above chance in 17 cancers, although sparse events limited interpretation in several cohorts. Paired cancer-level analyses did not demonstrate a systematic Bridge advantage over PCA or full expression, indicating that the 512-dimensional frozen representation preserves substantial prognostic information without establishing superior survival prediction. Bridge's pooled discrimination also lies within the approximately 0.750–0.765 range reported for other frozen bulk-RNA embeddings, while an independent scFoundation study similarly found embedding performance comparable to raw expression under Ridge–Cox but dependent on regularization; these cross-study values are contextual rather than directly paired.
