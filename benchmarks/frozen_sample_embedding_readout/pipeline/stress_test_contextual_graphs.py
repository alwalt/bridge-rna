#!/usr/bin/env python3
"""Held-out RR1/RR3 graph-response stress tests after general validation."""
from pathlib import Path
import json,time
import numpy as np,pandas as pd
from scipy.sparse import csr_matrix
from validate_contextual_graphs import REPO,OUT as GENERAL,WORK,G,graph,load

HERE=Path(__file__).resolve().parents[1];OUT=HERE/'results/graph_fingerprint/rr1_rr3_stress';T3=REPO/'benchmarks/osdr_batch_effect_representation';READ=HERE/'results/task_agnostic/response_geometry_summary.csv'
SPECS={'RR1_original':('C14__OSD-48__RR1-NASA__37-day',None),'RR1_remeasurement':('C04__OSD-168__RR1-NASA__37-day',None),'RR3_39_original':('C01__OSD-137__RR3__39-day',None),'RR3_39_remeasurement':('C05__OSD-168__RR3__39-day',None),'RR3_40_original':('C02__OSD-137__RR3__40-day','strict'),'RR3_40_remeasurement':('C06__OSD-168__RR3__40-day',None)}
def group_vector(indices,n,w,k,kind):
 cols=[];vals=[]
 for i in indices:
  c,v=graph(n[i],w[i],k,kind=='mutual');cols.extend(c.tolist());vals.extend(v.tolist())
 return csr_matrix((vals,([0]*len(cols),cols)),shape=(1,G*G))/len(indices)
def cosine_sparse(a,b):
 d=np.sqrt(a.multiply(a).sum()*b.multiply(b).sum());return float(a.multiply(b).sum()/d) if d else np.nan
def corr_sparse(a,b):
 # Correlation over the union of nonzero response edges.
 ix=np.union1d(a.indices,b.indices);x=np.asarray(a[:,ix].todense()).ravel();y=np.asarray(b[:,ix].todense()).ravel();return float(np.corrcoef(x,y)[0,1])
def main():
 marker=GENERAL/'technical_retrieval_summary.csv'
 if not marker.exists():raise RuntimeError('General validation must finish before RR1/RR3 stress testing')
 OUT.mkdir(parents=True,exist_ok=True);m,n,w=load('task3');mem=pd.read_csv(T3/'results/task3b_contrast_sample_membership.csv');ix=dict(zip(m.sample_id.astype(str),range(len(m))));rows=[]
 for k in [5,10,20]:
  for kind in ['union','mutual']:
   effects={}
   for name,(cid,strict) in SPECS.items():
    q=mem[mem.contrast_id.eq(cid)].copy()
    if name=='RR1_original':q=q[~q.sample_id.str.endswith('_M27')]
    if name=='RR1_remeasurement':q=q[~q.sample_id.str.endswith('_M29')]
    if strict:q=q[~q.sample_id.str.endswith('_F5')]
    z={c:q[q.condition.eq(c)].sample_id.map(ix).to_numpy(int) for c in ['FLT','GC']};effects[name]=group_vector(z['FLT'],n,w,k,kind)-group_vector(z['GC'],n,w,k,kind)
   pairs={'RR1':('RR1_original','RR1_remeasurement'),'RR3-39':('RR3_39_original','RR3_39_remeasurement'),'RR3-40':('RR3_40_original','RR3_40_remeasurement'),'RR1↔RR3-39 false friend':('RR1_original','RR3_39_original')}
   for label,(a,b) in pairs.items():rows.append({'k':k,'graph':kind,'comparison':label,'signed_edge_cosine':cosine_sparse(effects[a],effects[b]),'signed_edge_correlation':corr_sparse(effects[a],effects[b]),'edges_A':effects[a].nnz,'edges_B':effects[b].nnz})
 d=pd.DataFrame(rows);d.to_csv(OUT/'graph_response_stress_metrics.csv',index=False)
 vector=pd.read_csv(READ);vector.to_csv(OUT/'vector_readout_reference.csv',index=False)
 primary=d[(d.k==10)&(d.graph=='union')].copy();order=primary.set_index('comparison').signed_edge_cosine.rank(ascending=False);primary['discrimination_rank']=primary.comparison.map(order);primary.to_csv(OUT/'primary_rank_discrimination.csv',index=False)
 (OUT/'provenance.json').write_text(json.dumps({'parameters_frozen_from_general_validation':True,'primary':'union k10','sensitivities':['union k5','union k20','mutual k5/10/20'],'response_graph':'mean FLT weighted adjacency minus mean GC weighted adjacency','RR1_RR3_used_for_parameter_selection':False},indent=2)+'\n');print(primary.to_string(index=False));print('[complete]',OUT)
if __name__=='__main__':main()
