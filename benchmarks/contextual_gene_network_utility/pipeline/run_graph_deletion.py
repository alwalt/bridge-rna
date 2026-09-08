#!/usr/bin/env python3
"""Graph-ranked deletion for frozen exercise axes and Task-3 modes."""
from __future__ import annotations
import json,sys,time
from pathlib import Path
import numpy as np,pandas as pd,torch

HERE=Path(__file__).resolve().parents[1];REPO=HERE.parents[1];OUT=HERE/'results/final_decision';SIZES=[25,50,100,250,500,1000]
EX=REPO/'benchmarks/cross_species_exercise_response';T3=REPO/'benchmarks/osdr_batch_effect_representation'
sys.path.insert(0,str(REPO/'benchmarks/tcga_downstream/pipeline'));sys.path.insert(0,str(REPO))
from run_attention_pooling import load_frozen_encoder
from src.fm_embed.vocab import load_canonical_genes

def encode(model,x,mask,device,batch=4):
 out=[]
 with torch.inference_mode():
  for s in range(0,len(x),batch):
   z=torch.from_numpy(np.array(x[s:s+batch],copy=True)).to(device);z[:,mask]=-10.;out.append(model._encode_hidden(z).mean(1).float().cpu().numpy())
 return np.concatenate(out)
def responses(z,members,ids,treat='treatment'):
 out=[]
 for cid in ids:
  q=members[members.contrast_id.eq(cid)];a=q[q.role.eq(treat)].sample_index.to_numpy(int);b=q[q.role.ne(treat)].sample_index.to_numpy(int);out.append(z[a].mean(0)-z[b].mean(0))
 return np.stack(out)
def main():
 if not torch.cuda.is_available():raise RuntimeError('CUDA required')
 genes=np.array(load_canonical_genes(REPO/'data/ensembl/canonical_genes.csv'));gi={g:i for i,g in enumerate(genes)};metric=pd.read_parquet(OUT/'per_gene_metrics.parquet');device=torch.device('cuda:0');model=load_frozen_encoder(device)
 targets=[]
 # Exercise: score existing fixed Axis A/B directions on all eight established contrasts.
 x=np.load(EX/'work/matched_log1p_tpm_corrected.npy',mmap_mode='r');man=pd.read_parquet(EX/'results/matched_manifest.parquet').reset_index(names='sample_index');mem=pd.read_parquet(EX/'results/contrast_members.parquet').merge(man[['GSM','sample_index']],on='GSM');mem['role']=np.where(mem.role.str.contains('post|exercise',case=False,regex=True),'treatment','control');ids=pd.read_csv(EX/'results/response_contrasts.csv').contrast_id.tolist();base=np.load(EX/'work/response_effects_bridgerna.npy');direction=np.load(EX/'results/latent_axis_attribution/axis_directions.npz')
 axes={'exercise_axis_a':(['human_GSE108643','human_GSE86931','mouse_GSE126962','mouse_GSE132520'],direction['Axis_A']),'exercise_axis_b':(['human_GSE71972','human_GSE87748','mouse_GSE97718'],direction['Axis_B'])}
 for name,(assigned,d) in axes.items():targets.append((name,x,mem,ids,assigned,d))
 # Task3: fixed mode directions and all 14 contrasts.
 tx=np.load(T3/'work/bridgerna_log1p_tpm_inputs.npy',mmap_mode='r');tm=pd.read_csv(T3/'results/sample_manifest.csv').reset_index(names='sample_index');mm=pd.read_csv(T3/'results/task3b_contrast_sample_membership.csv').merge(tm[['sample_id','sample_index']],on='sample_id');mm['role']=np.where(mm.condition.eq('FLT'),'treatment','control');cl=pd.read_csv(T3/'results/task3c_cluster_assignments.csv');dirs=np.load(T3/'results/task3d_mode_ig/mode_response_directions.npz')
 for mode in [1,2]:
  key=next(k for k in dirs.files if str(mode) in k);assigned=cl[cl.geometry_cluster.eq(mode)].contrast_id.tolist();targets.append((f'task3_mode_{mode}',tx,mm,cl.contrast_id.tolist(),assigned,dirs[key]))
 rows=[];panels=[];started=time.time();total=len(targets)*len(SIZES)
 for ti,(name,matrix,members,allids,assigned,d) in enumerate(targets):
  relevant=[x for x in assigned if x in set(metric.contrast_id)];q=metric[metric.contrast_id.isin(relevant)].groupby('gene').graph_neighborhood_turnover.mean().reindex(genes).fillna(0);order=np.argsort(-q.to_numpy());original=responses(encode(model,matrix,np.array([],int),device),members,allids)@d;sel=np.array([allids.index(x) for x in assigned if x in allids]);original_signal=original[sel].mean()
  for si,size in enumerate(SIZES):
   panel=order[:size];changed=responses(encode(model,matrix,panel,device),members,allids)@d;masked=changed[sel].mean();rows.append({'target':name,'ranking':'graph_turnover','genes_masked':size,'original_signal':original_signal,'masked_signal':masked,'fraction_signal_remaining':masked/original_signal});panels.extend({'target':name,'genes_masked':size,'rank':r+1,'gene':genes[g]} for r,g in enumerate(panel));print(f'[progress] {ti*len(SIZES)+si+1}/{total} {name} size={size} elapsed={(time.time()-started)/60:.1f}m',flush=True)
 pd.DataFrame(rows).to_csv(OUT/'graph_ranked_deletion.csv',index=False);pd.DataFrame(panels).to_parquet(OUT/'graph_ranked_deletion_panels.parquet',index=False)
 (OUT/'graph_deletion_provenance.json').write_text(json.dumps({'encoder':'frozen','mask_value':-10.0,'sizes':SIZES,'ranking':'mean per-gene incident absolute contextual-edge response within frozen target grouping','targets':['exercise Axis A','exercise Axis B','Task3 Mode 1','Task3 Mode 2'],'random_IG_DE_comparators':'reused from existing Task2/Task3 deletion outputs'},indent=2)+'\n');print('[complete] graph-ranked deletion',flush=True)
if __name__=='__main__':main()
