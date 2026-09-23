#!/usr/bin/env python3
"""Audit benchmark #6 observed GSE264130 assets without modifying them."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parents[1];OUT=HERE/'results';OUT.mkdir(parents=True,exist_ok=True)
SRC=Path('/home/walt/bridge-rna.worktrees/attachment-pasted-text-1-30bae6d6/benchmarks/drug_perturbation_prediction')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 manifest=pd.read_parquet(SRC/'results/perturbation_vectors/observed_expression_delta_manifest.parquet')
 genes=pd.read_parquet(SRC/'results/perturbation_vectors/gene_order.parquet')
 delta=np.load(SRC/'work/prepared/observed_condition_delta.npy',mmap_mode='r')
 samples=pd.read_parquet(SRC/'work/prepared/sample_manifest.parquet')
 expr=np.load(SRC/'work/prepared/sample_log1p_tpm.npy',mmap_mode='r')
 canonical=pd.read_csv('/home/walt/bridge-rna/data/ensembl/canonical_genes.csv')
 assert delta.shape==(len(manifest),len(genes))==(744,15028)
 assert expr.shape==(len(samples),len(genes))==(1536,15028)
 assert manifest.drug.nunique()==372 and set(manifest.cell_line)=={'DIPG6','SF8628'}
 assert manifest.groupby(['drug','cell_line']).size().eq(1).all()
 assert manifest.n_plate_replicates.eq(2).all() and manifest.time_hours.eq(24).all()
 pos=genes.token_id.astype(int).to_numpy()-1
 assert np.array_equal(canonical.iloc[pos].gene_symbol.to_numpy(),genes.bridge_symbol.to_numpy())
 files=['results/perturbation_vectors/observed_expression_delta_manifest.parquet','results/perturbation_vectors/gene_order.parquet',
        'results/perturbation_vectors/provenance.json','results/preprocessing_provenance.json','results/bridge_representation_provenance.json',
        'work/prepared/observed_condition_delta.npy','work/prepared/sample_log1p_tpm.npy','work/prepared/sample_manifest.parquet']
 audit={'source_benchmark':str(SRC),'source_commit':'5d23557','dataset':'GSE264130','assay':'PLATE-Seq 3-prime RNA-seq',
  'observed_drugs':372,'observed_drug_cell_contexts':744,'cell_lines':['DIPG6','SF8628'],'time_hours':[24],
  'dose_design':'one study-selected dose per drug/cell line; doses retained in manifest','plate_replicates_per_context':2,
  'dmso_wells':int(samples.drug.astype(str).str.upper().eq('DMSO').sum()),'mock_wells_excluded_from_export':24,
  'matched_control':'six DMSO wells matched by cell line, plate, 24 h, before averaging two plate-level deltas',
  'normalization':'Entrez counts -> RPK using released gene lengths -> TPM -> natural log1p',
  'source_genes':25122,'observed_bridge_genes':len(genes),'canonical_bridge_genes':len(canonical),
  'missing_canonical_genes':len(canonical)-len(genes),'gene_order_verified_by_token_id':True,
  'observed_expression_vectors_available':True,'observed_expression_delta_shape':list(delta.shape),
  'observed_sample_expression_available':True,'observed_sample_expression_shape':list(expr.shape),
  'observed_treated_embeddings_available_in_handoff':False,
  'observed_control_embeddings_available_in_handoff':True,
  'prediction_assets_permitted':False,
  'asset_sha256':{f:sha(SRC/f) for f in files}}
 (OUT/'benchmark6_handoff_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
 pd.DataFrame([audit]).drop(columns=['asset_sha256']).to_csv(OUT/'benchmark6_handoff_audit.csv',index=False)
 print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
