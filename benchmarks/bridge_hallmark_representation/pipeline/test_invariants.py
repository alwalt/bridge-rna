#!/usr/bin/env python3
import numpy as np,pandas as pd
from common import RESULTS,WORK,read_vocab,read_gmt,bh_fdr

def main():
 genes=read_vocab(); sets=read_gmt()
 assert len(genes)==15165 and len(set(genes))==15165
 assert len(sets)==50 and all(len(x)>0 for x in sets.values())
 q=bh_fdr(np.array([.04,.01,.03,.002])); assert np.all((q>=0)&(q<=1))
 assert np.all(q[np.argsort([.04,.01,.03,.002])][:-1] <= q[np.argsort([.04,.01,.03,.002])][1:])
 if (RESULTS/"tcga_sample_manifest.parquet").exists():
  m=pd.read_parquet(RESULTS/"tcga_sample_manifest.parquet");z=np.load(WORK/"tcga_embeddings.npy",mmap_mode="r")
  assert z.shape==(9816,512) and np.isfinite(z).all() and m.sample_id.is_unique and m.patient_id.is_unique
  p=pd.read_parquet(RESULTS/"tcga_zero_shot_hallmark_profiles.parquet");assert p.shape==(9816,51) and p.sample_id.tolist()==m.sample_id.tolist()
 t=pd.read_csv(RESULTS/"gtex_tcga_probe_transfer_summary.csv");assert len(t)==50 and (t.pearson>0).all()
 if (RESULTS/"ig_validation/step_convergence_and_completeness.csv").exists():
  c=pd.read_csv(RESULTS/"ig_validation/step_convergence_and_completeness.csv")
  assert set(c.steps)=={8,16,32,64,128} and len(c)==80
  assert np.isfinite(c[["score_x_minus_score_zero","attribution_sum","completeness_residual","relative_completeness_error"]]).all().all()
  assert c.groupby(["sample_id","hallmark"]).size().eq(5).all()
 if (RESULTS/"pan_cancer_gsea/pathway_comparison.csv").exists():
  p=pd.read_csv(RESULTS/"pan_cancer_gsea/pathway_comparison.csv")
  cm=pd.read_csv(RESULTS/"pan_cancer_gsea/contrast_manifest.csv")
  assert len(cm)==7 and cm.patient_disjoint.all() and len(p)==7*50
  assert p.groupby("contrast").hallmark.nunique().eq(50).all()
  assert np.isfinite(p[["bridge_delta","whole_gsea_nes","vocab_gsea_nes"]]).all().all()
 print("invariants: PASS")
if __name__=="__main__": main()
