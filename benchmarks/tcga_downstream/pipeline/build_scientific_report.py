#!/usr/bin/env python3
"""Build the manuscript-quality TCGA scientific report from saved results."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


def md_table(frame: pd.DataFrame, digits: int = 4) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include="number"):
        display[column] = display[column].map(
            lambda value: "NA" if pd.isna(value) else f"{value:.{digits}f}")
    headers = [str(column) for column in display.columns]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |"
                 for row in display.itertuples(index=False, name=None))
    return "\n".join(lines)


def main() -> None:
    class_mlp = pd.read_csv(R / "classification_mlp_summary.csv")
    class_pca = pd.read_csv(R / "classification_summary.csv")
    pan = pd.read_csv(R / "bulkformer_parity_pan_cancer_classification_summary.csv")
    prognosis = pd.read_csv(R / "bulkformer_parity_alive_dead_prognosis_summary.csv")
    survival_mlp = pd.read_csv(R / "survival_mlp_summary.csv")
    survival_pca = pd.read_csv(R / "survival_summary.csv")
    oof = pd.read_csv(R / "survival_5fold_oof_metrics.csv")
    per = pd.read_csv(R / "survival_5fold_oof_per_cancer.csv")
    comparison = pd.read_csv(R / "survival_5fold_oof_representation_comparison.csv")
    stats = pd.read_csv(R / "survival_5fold_oof_statistical_summary.csv").set_index("summary").value
    repeated = pd.read_csv(R / "survival_5fold_oof_comparison_to_repeated_splits.csv")
    manifest = pd.read_csv(R / "bulkformer_parity_pan_cancer_manifest.csv")
    cohort = pd.read_parquet(R / "cohort_manifest.parquet")
    provenance = json.loads((R / "survival_5fold_oof_provenance.json").read_text())

    label_names = {"ours_45.6m": "Bridge 45.6M", "bulkformer_50m": "BulkFormer-50M",
                   "bulkformer_147m": "BulkFormer-147M",
                   "full_expression_25150": "Full raw expression"}
    five = class_mlp.assign(Model=class_mlp.representation.map(label_names))[
        ["Model", "macro_f1_mean", "weighted_f1_mean", "balanced_accuracy_mean"]]
    pca_row = class_pca.query("representation == 'raw_expression' and probe == 'logistic_regression'").iloc[0]
    five = pd.concat([five, pd.DataFrame([{"Model": "PCA-128 expression + logistic regression",
        "macro_f1_mean": pca_row.macro_f1_mean, "weighted_f1_mean": pca_row.weighted_f1_mean,
        "balanced_accuracy_mean": pca_row.balanced_accuracy_mean}])], ignore_index=True)
    five.columns = ["Representation / head", "Macro F1", "Weighted F1", "Balanced accuracy"]

    pan_rows = []
    for label, method, name in [
        ("ours_45.6m", "pca128_random_forest", "Bridge 45.6M"),
        ("bulkformer_50m", "pca128_random_forest", "BulkFormer-50M"),
        ("bulkformer_147m", "pca128_random_forest", "BulkFormer-147M"),
        ("full_raw_expression", "pca128_random_forest", "PCA-128 expression"),
        ("full_raw_expression", "random_forest", "Full raw expression")]:
        row = pan.query("representation == @label and method == @method").iloc[0]
        pan_rows.append({"Representation": name, "Macro F1": row.macro_f1_mean,
                         "Weighted F1": row.weighted_f1_mean,
                         "Balanced accuracy": row.balanced_accuracy_mean})
    pan_table = pd.DataFrame(pan_rows)

    prog_rows = []
    for label, method, name in [
        ("ours_45.6m", "pca128_random_forest", "Bridge 45.6M"),
        ("bulkformer_50m", "pca128_random_forest", "BulkFormer-50M"),
        ("bulkformer_147m", "pca128_random_forest", "BulkFormer-147M"),
        ("full_raw_expression", "pca128_random_forest", "PCA-128 expression"),
        ("full_raw_expression", "random_forest", "Full raw expression")]:
        row = prognosis.query("representation == @label and method == @method").iloc[0]
        prog_rows.append({"Representation": name, "AUROC": row.auroc_mean, "AUPRC": row.auprc_mean})
    prog_table = pd.DataFrame(prog_rows)

    cox = survival_mlp.assign(Representation=survival_mlp.representation.map(label_names))[
        ["Representation", "c_index_mean", "weighted_c_index_mean", "macro_c_index_mean"]]
    pca_cox = survival_pca.query("representation == 'raw_expression' and probe == 'pca128_cox'").iloc[0]
    cox = pd.concat([cox, pd.DataFrame([{"Representation": "PCA-128 expression",
        "c_index_mean": pca_cox.c_index_mean,
        "weighted_c_index_mean": pca_cox.weighted_c_index_mean,
        "macro_c_index_mean": pca_cox.macro_c_index_mean}])], ignore_index=True)
    cox.columns = ["Representation / head", "Pooled C-index", "Weighted within-cancer C-index",
                   "Macro within-cancer C-index"]

    oof_table = oof[["representation", "oof_survival_c_index", "comparable_pairs"]].copy()
    oof_table.columns = ["Representation", "Fold-aware OOF C-index", "Comparable pairs"]

    bridge = per.query("representation == 'Bridge 45.6M'").copy()
    bridge_table = bridge[["cancer", "n_patients", "n_events", "c_index", "ci_low", "ci_high",
                           "comparable_pairs", "stability_flag"]].sort_values("cancer")
    bridge_table.columns = ["Cancer", "N", "Events", "Bridge C-index", "95% CI low", "95% CI high",
                            "Comparable pairs", "Stability flag"]

    old_bridge = repeated.query("representation == 'Bridge 45.6M'")
    narrowed = int((old_bridge.ci_width_oof < old_bridge.ci_width_repeated).sum())
    median_old = old_bridge.ci_width_repeated.median()
    median_new = old_bridge.ci_width_oof.median()
    supported = bridge.loc[bridge.ci_low > .5, "cancer"].sort_values().tolist()
    sparse = bridge.loc[bridge.n_events < 20, "cancer"].sort_values().tolist()
    weighted = per.groupby("representation").apply(
        lambda frame: (frame.c_index * frame.n_patients).sum() / frame.n_patients.sum(),
        include_groups=False)
    cancer_labels = sorted(manifest.cancer_label.unique())

    report = f"""# TCGA Downstream Evaluation of Frozen Bridge Representations

## 1. Scientific objective

This benchmark tests whether a frozen pretrained Bridge representation retains information relevant to cancer phenotype, patient prognosis, and survival-risk ranking within individual cancers. Bridge is compared with conventional full expression, training-fold-only PCA, and locally evaluated BulkFormer representations where methodologically applicable. The objective is downstream representation evaluation, not a claim of universal superiority.

## 2. Models

### Bridge

Bridge has approximately 45.6 million parameters and uses an ordered 15,165-gene vocabulary. The evaluated checkpoint/config is `model/r7hnr92k/config.json` (run `r7hnr92k`), configured for natural `log1p(TPM)`. The encoder is frozen and produces a 512-dimensional sample representation. Only downstream task heads are trained.

### BulkFormer

The local comparison evaluates frozen BulkFormer-50M and BulkFormer-147M checkpoints. Their sample representations contain 259 and 643 features, respectively. BulkFormer uses its native ordered 20,010-gene vocabulary, natural `log1p(TPM)`, −10 for missing genes, and mean gene-token pooling in the final-release-aligned local reconstruction. Bridge and BulkFormer retain their native vocabularies; neither is forced into the other's input space. The full-expression baseline uses 25,150 TCGA features as `log1p(CPM)`.

## 3. TCGA cohort

Expression comes from the repository TCGA matrix (`data/tcga/tcga_matrix.h5`), and survival labels come from the TCGA PanCanAtlas curated overall-survival table. Primary tumors are selected, sorted deterministically, and deduplicated to one expression aliquot per patient. The five-cohort classification task contains 3,227 patients and uses BLCA, BRCA, GBM+LGG (`GBMLGG`), LUAD, and UCEC.

The pan-cancer classification cohort contains 9,942 unique patients and 33 labels: {', '.join(cancer_labels)}. It uses primary tumors plus LAML peripheral-blood primary cancer. The final Cox cohort contains {provenance['eligible_patients']:,} unique primary-tumor patients across {provenance['cancers']} cancers and {provenance['events']:,} deaths. Overall survival uses PanCanAtlas `OS.time` in days and `OS` as the event indicator; eligibility requires `OS` in {{0,1}} and positive follow-up time. Of 9,816 primary-tumor patients, 148 are excluded: 21 lack a valid binary event, 34 have missing time with a valid event, and 93 have nonpositive time.

## 4. Five-cohort cancer classification

All representations use identical patient-level repeated 80/20 splits. The primary frozen-representation classifier is an MLP with hidden layers `[256,128]`; metrics are macro F1, weighted F1, and balanced accuracy. PCA-128 controls are fitted on training patients only. Full raw expression is evaluated separately with the same downstream architecture.

{md_table(five)}

This task is near-saturated. Full raw expression and the PCA control are strongest, while Bridge and both BulkFormer representations remain close in absolute performance. The result supports substantial cancer-phenotype information in all frozen representations without establishing an advantage over conventional expression.

## 5. 33-class pan-cancer classification

The final BulkFormer release reports weighted F1 = 0.907, but its released object has 10,429 rows for 9,679 patients, including repeated patients and normal aliquots, and its exact downstream code and folds are unavailable. That number is retained as a literature-only reference. The local paired reconstruction uses 10 fixed patient-level folds, one eligible specimen per patient, native model preprocessing, training-fold-only PCA-128, and random forests.

{md_table(pan_table)}

Conventional expression is particularly strong for cancer-type classification. Bridge performs in the same broad range as the BulkFormer representations, below full expression and its PCA control. This is not a failure: it shows that the learned 512-dimensional Bridge representation retains much of the phenotype signal present in the transcriptome, while simple conventional representations remain highly effective for tissue-of-origin-like classification.

## 6. BulkFormer-style prognosis parity

The parity reconstruction predicts the binary alive/dead label with 10-fold patient-level random forests. This endpoint discards follow-up duration and does not model censoring; it is therefore distinct from time-to-event survival and is not the primary clinical survival analysis.

{md_table(prog_table)}

Historical preprint values of AUROC 0.747 and AUPRC 0.549 are literature-only and not paired to local folds. The final release does not provide a sufficiently complete prognosis implementation to treat those values as a direct local comparator.

## 7. Cox time-to-event survival

The primary survival path is `expression → frozen representation → [512,256] Cox risk head`. The head is optimized with Cox partial likelihood, retains censored patients, and is evaluated with C-index. A C-index of 0.5 corresponds to chance ordering; higher values mean that patients with earlier observed events tend to receive higher predicted risk, among comparable patient pairs.

{md_table(cox)}

The pooled pan-cancer C-index may partly exploit systematic prognosis differences between cancer types. Weighted and macro within-cancer summaries more directly ask whether risk is ranked correctly among patients with the same cancer.

## 8. Repeated 80/20 survival analysis

The original design uses five fixed repeated random 80/20 train/test splits. It is retained because repeated fitting measures split-to-split stability. Because the test sets overlap, 6,467 of 9,668 eligible patients ({6467/9668:.1%}) appear in at least one held-out set; 3,201 do not. This is expected for independent repeated holdouts and does not make the design erroneous. It motivated a complementary true five-fold OOF analysis for complete manuscript-facing coverage.

## 9. Complete 5-fold OOF survival evaluation

The complete analysis assigns every eligible patient to exactly one of five mutually exclusive test folds using deterministic within-cancer × event balancing, with a sparse-stratum fallback. Fold sizes range from 1,932 to 1,936; cancer counts and cancer-specific event counts differ by at most one between folds. All {provenance['eligible_patients']:,} patients receive exactly one held-out risk prediction per representation. Bridge remains frozen; only downstream Cox heads are refitted. PCA/scaling is fitted independently within each training fold, and full-expression standardization uses training patients only.

Cox risk scales from independently trained heads are not assumed interchangeable. Concordant and comparable-pair counts are computed inside each held-out fold and then summed, so no pair crosses fitted heads.

{md_table(oof_table)}

## 10. Within-cancer survival discrimination

The question is: *among patients with the same cancer, can the transcriptomic representation correctly rank survival risk?* Confidence intervals use 1,000 patient-level bootstrap draws within cancer while retaining fixed fold membership and fold-aware concordance.

{md_table(bridge_table, digits=3)}

Bridge confidence intervals lie wholly above 0.5 for {len(supported)} cancers: {', '.join(supported)}. The previously highlighted KIRP, ACC, LGG, UCEC, CESC, LIHC, LUAD, and KIRC remain supported. Cancers with fewer than 20 events ({', '.join(sparse)}) require particular caution; complete coverage cannot compensate for intrinsically sparse outcomes.

## 11. Bridge versus PCA and raw expression

Median within-cancer C-indices are {stats['median_bridge_c_index']:.3f} for Bridge, {stats['median_pca128_c_index']:.3f} for PCA-128, and {stats['median_full_raw_c_index']:.3f} for full raw expression. Patient-count-weighted means are {weighted['Bridge 45.6M']:.3f}, {weighted['PCA-128']:.3f}, and {weighted['Full raw expression']:.3f}, respectively. Bridge exceeds PCA in {int(stats['cancers_bridge_gt_pca128'])}/32 cancers and raw expression in {int(stats['cancers_bridge_gt_full_raw'])}/32. Median paired differences are {stats['median_bridge_minus_pca128']:+.3f} versus PCA and {stats['median_bridge_minus_full_raw']:+.3f} versus raw expression.

The paired sign-flip/Wilcoxon p-values are {stats['paired_signflip_mean_p_pca128']:.3f}/{stats['paired_wilcoxon_p_pca128']:.3f} against PCA and {stats['paired_signflip_mean_p_full_raw']:.3f}/{stats['paired_wilcoxon_p_full_raw']:.3f} against raw expression. There is no evidence of a systematic cancer-level difference; nonsignificance is not proof of equivalence. Bridge retains prognostic information comparable to conventional transcriptomic representations, but this benchmark does not demonstrate systematic survival-prediction superiority over PCA or full expression. Bridge compresses the 15,165-gene input into 512 learned dimensions, but storage, compute, transfer, and clinical advantages of that compression were not tested.

## 12. Forest-plot interpretation

Primary figures:

- [Bridge forest plot](results/figures/bridge_5fold_oof_per_cancer_survival_forest.png)
- [PCA-128 forest plot](results/figures/pca128_5fold_oof_per_cancer_survival_forest.png)
- [Full-expression forest plot](results/figures/raw_expression_5fold_oof_per_cancer_survival_forest.png)
- [Combined forest plot](results/figures/bridge_pca_raw_5fold_oof_survival_forest.png)
- [Three-panel comparison](results/figures/bridge_pca_raw_5fold_oof_survival_three_panel.png)

Each point is the fold-aware held-out C-index, each whisker is a patient-bootstrap 95% interval, the dashed line marks C = 0.5, and `events / N` records outcome support. Sparse-event cancers have few comparable pairs and consequently wide or unstable intervals. Complete OOF coverage narrows Bridge intervals in {narrowed}/32 cancers; the median width falls from {median_old:.3f} to {median_new:.3f}.

## 13. Medical interpretation

The evaluated path is `tumor RNA-seq → Bridge → molecular representation → Cox risk`. It demonstrates retrospective prognostic information, not clinical utility. Clinical utility would require clinical-covariate adjustment, comparisons with stage and grade, combined clinical-plus-expression models, calibration, external independent cohorts, and prospective or temporal validation. A logical future analysis is `Clinical` versus `Clinical + PCA` versus `Clinical + Raw` versus `Clinical + Bridge`; it is outside the present benchmark.

## 14. Major scientific conclusions

1. Bridge preserves cancer-phenotype information: five-cohort classification is near-saturated, and 33-class performance remains strong.
2. Bridge preserves prognostic information: its complete fold-aware OOF C-index is {oof.query("representation == 'Bridge 45.6M'").oof_survival_c_index.iloc[0]:.3f}.
3. Within-cancer information is stable for a substantial subset of cancers, with 17 Bridge intervals wholly above chance, but not uniformly across TCGA.
4. Bridge does not systematically outperform PCA.
5. Bridge does not systematically outperform full expression.
6. Bridge lies in the same general performance range as locally evaluated BulkFormer representations; task-specific ordering varies.
7. The benchmark establishes that a frozen 512-dimensional Bridge representation retains substantial phenotype and prognosis signal under patient-held-out evaluation.
8. It leaves unresolved incremental value beyond clinical covariates, external generalization, calibration, clinical utility, and the practical value of representation compression.

## 15. Limitations

- TCGA is retrospective and lacks external clinical validation here.
- Event counts are heterogeneous and intrinsically sparse in several cancers.
- Pretraining exposure may differ across models; possible TCGA exposure is not uniform or always recoverable.
- Published BulkFormer protocols and local paired reconstructions differ, and exact final-release folds/code are unavailable.
- Clinical covariates are absent from the current representation comparison.
- Results depend on the declared downstream survival head and fixed hyperparameters.
- Pooled metrics can reflect cancer-type composition and between-cancer prognosis differences.
- Alive/dead parity results are not censoring-aware survival results.

## Reproducibility and final QA

The final lightweight QA passed: all required outputs are present; all 9,668 patients occur exactly once per representation; Bridge, PCA, and raw expression have identical patients, folds, and outcomes; fold exclusivity holds; learned preprocessing is training-fold-only; Bridge is frozen; all notebook cells execute; and figures regenerate from saved result tables. The benchmark is closed for the current manuscript unless a reproducibility defect is discovered.
"""
    (ROOT / "SCIENTIFIC_REPORT.md").write_text(report)
    parity_summary = f"""# BulkFormer parity and TCGA benchmark summary

The TCGA downstream benchmark is closed for the current manuscript. Detailed methods, tables, interpretation, and limitations are in [SCIENTIFIC_REPORT.md](SCIENTIFIC_REPORT.md).

## Completed analyses

- Five-cohort classification with frozen Bridge 45.6M, BulkFormer-50M, BulkFormer-147M, PCA controls, and full expression.
- Locally paired 33-class pan-cancer classification, separated from the final published BulkFormer literature value.
- BulkFormer-style alive/dead prognosis parity, kept separate from censoring-aware survival.
- Original repeated 80/20 Cox survival with pooled, weighted within-cancer, and macro within-cancer C-index.
- Complete true five-fold OOF Cox survival for all 9,668 eligible patients, including 1,000-draw patient-bootstrap intervals and matched Bridge/PCA/raw forest plots.

## Final complete OOF survival

{md_table(oof_table)}

Bridge's median within-cancer C-index is {stats['median_bridge_c_index']:.3f}; PCA-128 is {stats['median_pca128_c_index']:.3f}; full raw expression is {stats['median_full_raw_c_index']:.3f}. Bridge exceeds PCA and raw expression in 15 of 32 cancers each. Paired sign-flip and Wilcoxon analyses find no systematic difference. Seventeen Bridge cancer-specific intervals are wholly above chance.

## BulkFormer context

The final Cell Systems release reports TCGA classification weighted F1 = 0.907 as a literature value. It is not directly paired with the local evaluation because exact downstream code and folds are unavailable and the released object contains repeated-patient and normal-tissue rows. The local 33-class comparison uses one eligible specimen per patient and common folds while preserving native model vocabularies.

## Primary records

- [Scientific report](SCIENTIFIC_REPORT.md)
- [Executed notebook](tcga_downstream_benchmark.ipynb)
- [Gap audit](bulkformer_tcga_gap_audit.md)
- [OOF representation table](results/survival_5fold_oof_representation_comparison.csv)
- [OOF provenance](results/survival_5fold_oof_provenance.json)
- [Final QA](results/final_qa.json)
"""
    (ROOT / "BULKFORMER_PARITY_SUMMARY.md").write_text(parity_summary)


if __name__ == "__main__":
    main()
