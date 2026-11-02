#!/usr/bin/env python3
"""Freeze hashes and versions only after GTEx/TCGA acceptance criteria pass."""
import json,platform,sys,shutil
import numpy as np,pandas as pd,torch,sklearn,scipy
from common import CONFIG,RESULTS,WORK,resolve,read_vocab,sha256,save_json

def main():
 s=pd.read_csv(RESULTS/"gtex_tcga_probe_transfer_summary.csv")
 criteria={"median_tcga_within_pearson_positive":float(s.tcga_within_pearson.median())>0,
           "median_zero_shot_pearson_positive":float(s.pearson.median())>0,
           "at_least_half_zero_shot_positive":int((s.pearson>0).sum())>=25}
 status="validated_for_frozen_later_use" if all(criteria.values()) else "not_validated_do_not_apply"
 shutil.copyfile(WORK/"frozen_gtex_hallmark_probe.npz",RESULTS/"frozen_gtex_hallmark_probe.npz");shutil.copyfile(WORK/"hallmark_linear_decoder.npz",RESULTS/"frozen_contextual_hallmark_decoder.npz")
 pd.DataFrame({"model_index":range(len(read_vocab())),"gene":read_vocab()}).to_csv(RESULTS/"frozen_canonical_genes.csv",index=False)
 pd.DataFrame({"pathway_index":range(len(s)),"hallmark":np.load(WORK/"frozen_gtex_hallmark_probe.npz")["hallmarks"].astype(str)}).to_csv(RESULTS/"frozen_hallmark_order.csv",index=False)
 files=[RESULTS/"frozen_gtex_hallmark_probe.npz",RESULTS/"frozen_contextual_hallmark_decoder.npz",RESULTS/"frozen_canonical_genes.csv",RESULTS/"frozen_hallmark_order.csv",resolve(CONFIG["checkpoint"]),resolve(CONFIG["model_config"]),resolve(CONFIG["hallmark_gmt"])]
 save_json(RESULTS/"frozen_pathway_system_manifest.json",{"status":status,"criteria":criteria,"artifacts":[{"path":str(p),"sha256":sha256(p)} for p in files],"pathway_order":np.load(WORK/"frozen_gtex_hallmark_probe.npz")["hallmarks"].astype(str).tolist(),"canonical_gene_count":len(read_vocab()),"preprocessing":"natural log1p(TPM); exact canonical order","pooling":"unweighted final-layer mean","versions":{"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"pandas":pd.__version__,"torch":torch.__version__,"sklearn":sklearn.__version__,"scipy":scipy.__version__},"cross_species_application_performed":False})
 print(status)
if __name__=="__main__":main()
