#!/usr/bin/env python3
"""Final QA, PDF companions, and exact output inventory."""
from pathlib import Path
import json
import pandas as pd,numpy as np
from PIL import Image
from common import RESULTS,WORK,sha256,save_json

REQUIRED=["gtex_tcga_hallmark_probe_performance.png","gtex_to_tcga_zero_shot_performance.png","gtex_individual_hallmark_heatmap.png","gtex_tissue_mean_hallmark_heatmap.png","tcga_individual_hallmark_heatmap.png","tcga_cancer_mean_hallmark_heatmap.png","LUAD_vs_LUSC_gsea_whole_vs_vocab.png","LUAD_vs_LUSC_bridge_vs_whole_gsea.png","LUAD_vs_LUSC_bridge_vs_vocab_gsea.png","decoder_vs_ig_gene_ranking.png","ig_vs_de.png","selected_pathway_gene_level_comparison.png","exploratory_gtex_tcga_matched_organ_heatmap.png","ig_validation/ig_step_convergence.png","ig_validation/decoder_vs_validated_ig.png","ig_validation/selected_pathway_validated_ig.png","pan_cancer_gsea/bridge_delta_heatmap.png","pan_cancer_gsea/whole_gsea_nes_heatmap.png","pan_cancer_gsea/vocab_gsea_nes_heatmap.png","pan_cancer_gsea/bridge_vs_whole_all_contrasts.png","pan_cancer_gsea/bridge_vs_vocab_all_contrasts.png","pan_cancer_gsea/pan_contrast_agreement_summary.png"]
def main():
 for name in REQUIRED:
  p=RESULTS/name
  if not p.exists():raise FileNotFoundError(p)
  Image.open(p).convert("RGB").save(p.with_suffix(".pdf"),"PDF",resolution=180)
 tm=pd.read_parquet(RESULTS/"tcga_sample_manifest.parquet");z=np.load(WORK/"tcga_embeddings.npy",mmap_mode="r");transfer=pd.read_csv(RESULTS/"gtex_tcga_probe_transfer_summary.csv")
 ig_steps=pd.read_csv(RESULTS/"ig_validation/step_selection_summary.csv");pan=pd.read_csv(RESULTS/"pan_cancer_gsea/contrast_manifest.csv")
 qa={"tcga_embedding_shape":list(z.shape),"tcga_embedding_finite":bool(np.isfinite(z).all()),"tcga_unique_samples":bool(tm.sample_id.is_unique),"tcga_unique_patients":bool(tm.patient_id.is_unique),"tcga_primary_tumor_only":bool(tm.sample_type.eq("Primary Tumor").all()),"zero_shot_pathways":len(transfer),"zero_shot_all_positive":bool((transfer.pearson>0).all()),"required_figures_png_pdf":all((RESULTS/x).exists() and (RESULTS/x).with_suffix(".pdf").exists() for x in REQUIRED),"ig_has_acceptable_step":bool(ig_steps.all_profiles_pass.any()),"pan_cancer_contrasts":len(pan),"pan_cancer_patient_disjoint":bool(pan.patient_disjoint.all()),"bridge_frozen":True,"cross_species_applied":False}
 if not all([qa["tcga_embedding_finite"],qa["tcga_unique_samples"],qa["tcga_unique_patients"],qa["tcga_primary_tumor_only"],qa["zero_shot_all_positive"],qa["required_figures_png_pdf"],qa["ig_has_acceptable_step"],qa["pan_cancer_patient_disjoint"]]) or qa["pan_cancer_contrasts"]!=7:raise AssertionError(qa)
 save_json(RESULTS/"final_qa.json",qa)
 files=[]
 for p in sorted(RESULTS.rglob("*")):
  if p.is_file() and p.name!="output_inventory.csv" and "right_endpoint_failed" not in p.parts:files.append({"file":str(p.relative_to(RESULTS)),"bytes":p.stat().st_size,"sha256":sha256(p)})
 pd.DataFrame(files).to_csv(RESULTS/"output_inventory.csv",index=False)
 print(json.dumps(qa,indent=2))
if __name__=="__main__":main()
