#!/usr/bin/env python3
"""Prespecified null and stability controls for general graph validation."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from validate_contextual_graphs import OUT,WORK,G,SEED,REPO,load,graph,pair

def main():
 rng=np.random.default_rng(SEED);OUT.mkdir(parents=True,exist_ok=True);m,n,w=load('tcell');rows=[]
 # Independent within-sample gene permutations emulate shuffled contextual-token identity.
 for k in [5,10,20]:
  gs=[]
  for i in range(len(m)):
   c,v=graph(n[i],w[i],k);perm=np.random.default_rng(SEED+i+k*100).permutation(G);a=perm[c//G];b=perm[c%G];code=np.minimum(a,b)*G+np.maximum(a,b);order=np.argsort(code);gs.append((code[order],v[order]))
  for i in range(len(m)):
   cand=np.flatnonzero(m.library_prep.to_numpy()!=m.library_prep.iloc[i]);target=cand[m.pair_id.iloc[cand].to_numpy()==m.pair_id.iloc[i]][0];scores=np.array([pair(*gs[i],*gs[j])[0] for j in cand]);rank=int(np.flatnonzero(cand[np.argsort(-scores)]==target)[0])+1;rows.append({'k':k,'query':m.sample_id.iloc[i],'rank':rank,'R1':rank==1,'R5':rank<=5})
 pd.DataFrame(rows).to_csv(OUT/'embedding_identity_shuffle_retrieval.csv',index=False)
 # Bootstrap donor pairs for metric stability.
 q=pd.read_parquet(OUT/'technical_retrieval_queries.parquet');q=q[q['query'].str.endswith('_mRNA')].copy();boot=[]
 for keys,z in q.groupby(['k','graph','metric']):
  values=z.similarity.to_numpy();ranks=z['rank'].to_numpy()
  for b in range(1000):
   ix=rng.integers(0,len(values),len(values));boot.append({'k':keys[0],'graph':keys[1],'metric':keys[2],'bootstrap':b,'mean_similarity':np.nanmean(values[ix]),'R@1':np.mean(ranks[ix]<=1),'R@5':np.mean(ranks[ix]<=5)})
 pd.DataFrame(boot).to_parquet(OUT/'technical_pair_bootstrap.parquet',index=False)
 # Degree/expression-bin-matched biological-pair null on 50 tissue graphs.
 tm,tn,tw=load('tissue');base=REPO/'benchmarks/cross_species_exercise_response/work/hallmark_readout/archs4_log1p_tpm.float32.mmap';X=np.memmap(base,mode='r',dtype='float32',shape=(40000,G));expr=np.asarray(X[tm.matrix_row.to_numpy()]).mean(0);erank=pd.qcut(pd.Series(expr).rank(method='first'),10,labels=False).to_numpy();fun=[];genes=pd.read_csv(REPO/'data/ensembl/canonical_genes.csv').sort_values('token_id').gene_symbol.astype(str).str.upper().tolist();sets=[set(str(x).upper() for x in v) for v in json.loads((REPO/'data/gsea/hallmark_gene_sets.json').read_text()).values()];membership={g:{i for i,s in enumerate(sets) if g in s} for g in genes}
 for si in rng.choice(len(tm),50,replace=False):
  c,v=graph(tn[si],tw[si],10);a=c//G;b=c%G;deg=np.bincount(np.r_[a,b],minlength=G);drank=pd.qcut(pd.Series(deg).rank(method='first'),10,labels=False).to_numpy();pick=rng.choice(len(c),min(10000,len(c)),replace=False);same=[]
  for j in pick:
   pool=np.flatnonzero((drank==drank[b[j]])&(erank==erank[b[j]]));same.append(rng.choice(pool))
  observed=np.mean([bool(membership[genes[i]]&membership[genes[j]]) for i,j in zip(a[pick],b[pick])]);random=np.mean([bool(membership[genes[i]]&membership[genes[j]]) for i,j in zip(a[pick],same)]);fun.append({'sample':si,'observed_shared_hallmark':observed,'degree_expression_matched_random_shared_hallmark':random})
 pd.DataFrame(fun).to_csv(OUT/'degree_expression_matched_graph_null.csv',index=False);print('[complete]',OUT)
if __name__=='__main__':main()
