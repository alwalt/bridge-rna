#!/usr/bin/env python3
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parents[1];OUT=HERE/'results';SRC=Path('/home/walt/bridge-rna.worktrees/attachment-pasted-text-1-30bae6d6/benchmarks/drug_perturbation_prediction')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 hand=json.loads((OUT/'benchmark6_handoff_audit.json').read_text());bad=[]
 for rel,expected in hand['asset_sha256'].items():
  actual=sha(SRC/rel)
  if actual!=expected:bad.append({'file':rel,'expected':expected,'actual':actual})
 # Confirm the prior LINCS benchmark remains byte-identical to its frozen validation record.
 lv=json.loads((ROOT/'benchmarks/direction_aware_reversal/results/validation_summary.json').read_text());lbad=[]
 for rel,expected in lv['output_sha256'].items():
  p=(ROOT/'benchmarks/direction_aware_reversal/results'/rel).resolve()
  if p.exists() and sha(p)!=expected:lbad.append(str(p))
 e=pd.read_parquet(OUT/'expression_reversal_by_contrast_cell.parquet');l=pd.read_parquet(OUT/'latent_reversal_by_contrast_cell.parquet');u=pd.read_csv(OUT/'unified_primary_summary.csv');a=pd.read_csv(OUT/'all_primary_candidates.csv')
 assert len(u)==6 and e.drug.nunique()==l.drug.nunique()==372 and e.cell_line.nunique()==l.cell_line.nunique()==2
 assert (e.fdr<.05).sum()==0 and (l.fdr<.05).sum()==0
 assert u['Strict cross-study expression'].sum()==u['Strict cross-study latent'].sum()==0
 eq=json.loads((OUT/'encoder_equivalence.json').read_text());assert eq['max_abs_difference']==0
 files=['benchmark6_handoff_audit.json','expression_reversal_by_contrast_cell.parquet','latent_reversal_by_contrast_cell.parquet','integrated_candidate_evidence.csv','unified_primary_summary.csv','../GSE264130_PRIMARY_REVERSAL_REPORT.md','../figures/gse264130_primary_reversal_summary.png']
 result={'valid':not bad and not lbad,'benchmark6_input_hash_mismatches':bad,'frozen_lincs_hash_mismatches':lbad,'drugs':372,'cells':2,'expression_rows':len(e),'latent_rows':len(l),'candidate_rows':len(a),'bh_significant_expression':0,'bh_significant_latent':0,'strict_cross_study_candidates':0,'encoder_control_max_abs_difference':0.0,'output_sha256':{x:sha((OUT/x).resolve()) for x in files}}
 (OUT/'validation_summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['valid']
if __name__=='__main__':main()
