#!/usr/bin/env python3
from pathlib import Path
import nbformat as nbf
HERE=Path(__file__).resolve().parents[1]
nb=nbf.v4.new_notebook(); nb.cells=[
 nbf.v4.new_markdown_cell("# Bridge frozen representations and Hallmark biology\n\nThis is the human-readable report for the reproducible pipeline in `pipeline/`. It never reimplements analysis logic."),
 nbf.v4.new_code_cell("from pathlib import Path\nimport json, pandas as pd\nR=Path('results')\nmanifest=json.loads((R/'provenance_manifest.json').read_text())\nmanifest['model'], manifest['cohorts']"),
 nbf.v4.new_markdown_cell("## Q1 — sample-space geometry"),
 nbf.v4.new_code_cell("g=json.loads((R/'gtex_geometry_summary.json').read_text())\ng"),
 nbf.v4.new_code_cell("from IPython.display import Image, display\ndisplay(Image(filename=R/'gtex_geometry.png'))"),
 nbf.v4.new_markdown_cell("## Hallmark targets and distributed probes\nTargets are independent within-sample rank-mean Hallmark scores from expression. Feature selection and alpha selection occur inside training folds; GTEx folds are donor-disjoint."),
 nbf.v4.new_code_cell("m=pd.read_csv(R/'gtex_nested_hallmark_probe_metrics.csv')\nm.groupby(['control','features'])[['pearson','spearman','r2']].mean()"),
 nbf.v4.new_markdown_cell("## Contextual gene decoder\nThe decoder is a new linear multi-label implementation. Its primary evaluation holds out gene identities. GTEx profiles below cover all 4,260 frozen samples and are validated against independent expression-derived targets."),
 nbf.v4.new_code_cell("decoder=json.loads((R/'decoder_summary.json').read_text())\nvalidation=pd.read_csv(R/'gtex_mean_profile_validation.csv')\nrecovery=pd.read_csv(R/'gtex_mean_profile_from_sample_embedding.csv')\ndecoder, validation[['pearson','spearman']].describe(), recovery[['pearson','spearman']].describe()"),
 nbf.v4.new_markdown_cell("Mean-member scores are the usable contextual-decoder profile. The top-k sensitivity is a negative result because many probabilities saturate. TCGA below uses newly regenerated, provenance-compatible embeddings."),
 nbf.v4.new_markdown_cell("## TCGA replication and zero-shot transfer"),
 nbf.v4.new_code_cell("tcga=json.loads((R/'tcga_geometry_summary.json').read_text())\ntransfer=pd.read_csv(R/'gtex_tcga_probe_transfer_summary.csv')\ntcga, transfer[['gtex_within_pearson','tcga_within_pearson','pearson','coefficient_cosine_gtex_tcga']].describe()"),
 nbf.v4.new_code_cell("display(Image(filename=R/'gtex_tcga_hallmark_probe_performance.png')); display(Image(filename=R/'gtex_to_tcga_zero_shot_performance.png'))"),
 nbf.v4.new_markdown_cell("## Frozen-probe Hallmark landscapes\nHeatmaps are descriptive and use a common pathway order and color scale."),
 nbf.v4.new_code_cell("display(Image(filename=R/'gtex_tissue_mean_hallmark_heatmap.png')); display(Image(filename=R/'tcga_cancer_mean_hallmark_heatmap.png'))"),
 nbf.v4.new_markdown_cell("## Conventional DE/GSEA and vocabulary control"),
 nbf.v4.new_code_cell("agreement=pd.read_csv(R/'gsea_universe_and_bridge_agreement.csv')\nagreement"),
 nbf.v4.new_code_cell("display(Image(filename=R/'LUAD_vs_LUSC_gsea_whole_vs_vocab.png')); display(Image(filename=R/'KIRC_vs_KIRP_bridge_vs_vocab_gsea.png'))"),
 nbf.v4.new_markdown_cell("## Numerically validated sample-specific Integrated Gradients\nDecoder scores measure Hallmark-like information in a contextual gene representation. IG measures an input gene's contribution to one sample's frozen Bridge→Hallmark prediction relative to the all-zero `log1p(TPM)` reference. It is not cancer-pair attribution. The first-order right-endpoint audit is preserved as a failed numerical control."),
 nbf.v4.new_code_cell("ig_steps=pd.read_csv(R/'ig_validation/step_selection_summary.csv')\nig=pd.read_csv(R/'ig_validation/decoder_ig_comparison_metrics.csv')\nig_steps, ig.describe()"),
 nbf.v4.new_code_cell("display(Image(filename=R/'ig_validation/ig_step_convergence.png')); display(Image(filename=R/'ig_validation/decoder_vs_validated_ig.png'))"),
 nbf.v4.new_markdown_cell("## Pan-cancer Bridge versus conventional GSEA\nSeven patient-disjoint primary-tumor subtype/site contrasts use identical samples, Welch statistics, Hallmark sets, and GSEA settings for the whole and vocabulary-matched universes. Adjacent-normal comparisons are not run because normals are absent from the frozen Bridge-profile cohort."),
 nbf.v4.new_code_cell("pan=pd.read_csv(R/'pan_cancer_gsea/pan_contrast_summary.csv')\nper_contrast=pd.read_csv(R/'pan_cancer_gsea/per_contrast_agreement.csv')\npan, per_contrast"),
 nbf.v4.new_code_cell("display(Image(filename=R/'pan_cancer_gsea/pan_contrast_agreement_summary.png')); display(Image(filename=R/'pan_cancer_gsea/bridge_vs_whole_all_contrasts.png')); display(Image(filename=R/'pan_cancer_gsea/bridge_vs_vocab_all_contrasts.png'))"),
 nbf.v4.new_code_cell("display(Image(filename=R/'pan_cancer_gsea/bridge_delta_heatmap.png')); display(Image(filename=R/'pan_cancer_gsea/whole_gsea_nes_heatmap.png')); display(Image(filename=R/'pan_cancer_gsea/vocab_gsea_nes_heatmap.png'))"),
 nbf.v4.new_markdown_cell("## Exploratory matched organs\nThese cross-cohort displacements are not pure tumor effects."),
 nbf.v4.new_code_cell("display(Image(filename=R/'exploratory_gtex_tcga_matched_organ_heatmap.png'))"),
 nbf.v4.new_markdown_cell("## Limitations\nGTEx/TCGA subtraction is cross-cohort and cannot identify pure tumor effects. Unexplained coordinates are not evidence of novel biology. Heatmap appearance is never treated as validation.")]
nb.metadata.kernelspec={"display_name":"Python 3","language":"python","name":"python3"}
nbf.write(nb,HERE/'bridge_hallmark_representation.ipynb')
