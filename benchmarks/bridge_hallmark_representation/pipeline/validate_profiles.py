#!/usr/bin/env python3
"""Part IV-V: independent profile validation and bidirectional linear recovery."""
from __future__ import annotations
import argparse
import numpy as np,pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from common import CONFIG,RESULTS,WORK,bh_fdr

def main():
 p=argparse.ArgumentParser(); p.add_argument("--dataset",required=True); p.add_argument("--profiles",required=True); p.add_argument("--targets",required=True); p.add_argument("--embeddings",required=True); p.add_argument("--metadata",required=True); a=p.parse_args()
 P=pd.read_parquet(a.profiles); T=pd.read_parquet(a.targets); M=pd.read_parquet(a.metadata); names=[c for c in P if c!="sample_id"]
 if P.sample_id.tolist()!=T.sample_id.tolist() or P.sample_id.tolist()!=M.sample_id.tolist(): raise AssertionError("Sample order mismatch")
 rows=[]
 for name in names:
  pear=stats.pearsonr(P[name],T[name]); spear=stats.spearmanr(P[name],T[name]); rows.append({"hallmark":name,"pearson":pear.statistic,"pearson_p":pear.pvalue,"spearman":spear.statistic,"spearman_p":spear.pvalue})
 out=pd.DataFrame(rows); out["pearson_fdr"]=bh_fdr(out.pearson_p); out["spearman_fdr"]=bh_fdr(out.spearman_p); out.to_csv(RESULTS/f"{a.dataset}_profile_validation.csv",index=False)
 Z=np.load(a.embeddings); groups=np.asarray(M.donor_id if "donor_id" in M else M.patient_id); cv=GroupKFold(n_splits=CONFIG["outer_folds"]); recovery=[]
 Y=P[names].to_numpy(); pred=np.empty_like(Y)
 # Multi-output fitting gives exactly the same fixed-alpha linear objective while avoiding 250 redundant decompositions.
 for tr,te in cv.split(Z,groups=groups):
  scaler=StandardScaler().fit(Z[tr]); pred[te]=Ridge(alpha=10.0).fit(scaler.transform(Z[tr]),Y[tr]).predict(scaler.transform(Z[te]))
 for k,name in enumerate(names):
  recovery.append({"hallmark":name,"pearson":stats.pearsonr(pred[:,k],Y[:,k]).statistic,"spearman":stats.spearmanr(pred[:,k],Y[:,k]).statistic,"alpha":10.0})
 pd.DataFrame(recovery).to_csv(RESULTS/f"{a.dataset}_profile_from_sample_embedding.csv",index=False)
if __name__=="__main__": main()
