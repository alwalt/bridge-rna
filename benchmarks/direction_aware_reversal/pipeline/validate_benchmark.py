#!/usr/bin/env python3
"""Validate frozen inputs, key output invariants, and artifact hashes."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parents[1];OUT=HERE/'results'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 prov=json.loads((OUT/'condition_signature_provenance.json').read_text());bad=[]
 for rel,expected in prov['input_hashes'].items():
  actual=sha(ROOT/rel)
  if actual!=expected:bad.append({'file':rel,'expected':expected,'actual':actual})
 ext=json.loads((OUT/'lincs_extraction_provenance.json').read_text())
 matrix=ROOT/ext['extracted_matrix'];genes=pd.read_csv(OUT/'lincs_exemplar_genes.csv');meta=pd.read_parquet(OUT/'lincs_exemplar_contexts.parquet')
 arr=np.load(matrix,mmap_mode='r');assert arr.shape==(len(meta),len(genes))==(112495,8831)
 defs=pd.read_csv(OUT/'reversal_signature_definitions.csv');scores=np.load(HERE/'work/lincs/GSE92742/reversal_context_scores.npy',mmap_mode='r');assert scores.shape==(len(meta),len(defs))
 unified=pd.read_csv(OUT/'unified_six_condition_table.csv');assert len(unified)==6
 pub=pd.read_csv(OUT/'published_drug_direction_audit.csv');assert pub.requested_drug.nunique()==15
 outputs=['frozen_condition_signatures.parquet','drug_reversal_by_contrast.parquet','space_condition_consensus.csv.gz',
          'published_drug_direction_audit.csv','prioritized_candidate_nulls.csv','target_reversal_integration.parquet',
          'unified_six_condition_table.csv','../DIRECTION_AWARE_REVERSAL_REPORT.md','../figures/direction_aware_reversal_summary.png']
 hashes={x:sha((OUT/x).resolve()) for x in outputs}
 result={'valid':not bad,'frozen_input_hash_mismatches':bad,'contexts':len(meta),'genes':len(genes),'definitions':len(defs),
         'conditions':len(unified),'published_drugs_audited':pub.requested_drug.nunique(),'output_sha256':hashes}
 (OUT/'validation_summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
 if bad:raise SystemExit(1)
if __name__=='__main__':main()
