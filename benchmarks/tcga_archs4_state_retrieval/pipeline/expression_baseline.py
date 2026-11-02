#!/usr/bin/env python3
"""Centroid-only expression retrieval on the authoritative 40k canonical cache."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

HERE=Path(__file__).resolve().parents[1]; RESULTS=HERE/"results"
sys.path.insert(0,str(Path(__file__).parent)); from run_benchmark import Log, apply_probe, exact_topk, overlap_table, resolve_paths

def main():
 cfg=json.loads((HERE/"config.json").read_text()); p=resolve_paths(cfg); log=Log(RESULTS/"expression_baseline.log")
 tm=pd.read_parquet(RESULTS/"tcga_query_manifest.parquet").sort_values("matrix_row"); tx=np.load(p["tcga_expression"],mmap_mode="r")
 cm=pd.read_parquet(p["expression_cache_manifest"]).sort_values("matrix_row").reset_index(drop=True); n=len(cm); g=15165
 assert tx.shape==(9816,g) and n==40000 and cm.gsm.is_unique and cm.matrix_row.tolist()==list(range(n))
 ex=np.memmap(p["expression_cache"],mode="r",dtype="float32",shape=(n,g)); assert np.isfinite(ex).all()
 cancers=sorted(tm.cohort.unique()); groups=[np.flatnonzero(tm.cohort.to_numpy()==c) for c in cancers]
 q=np.vstack([np.asarray(tx[ix],np.float32).mean(0) for ix in groups]); idx,sim=exact_topk(q,ex,100,"cuda:0",1,log,"expression centroid")
 rows=[]
 for qi,c in enumerate(cancers):
  for rank,j in enumerate(idx[qi],1):
   r=cm.iloc[j]; rows.append({"query_index":qi,"query_cancer":c,"rank":rank,"cache_row":int(j),"gsm":r.gsm,"series_id":getattr(r,"gse",None),"similarity":float(sim[qi,rank-1]),"method":"expression15165_cosine","query_mode":"centroid"})
 out=pd.DataFrame(rows); out.to_parquet(RESULTS/"top100_centroid_expression15165_cosine.parquet",index=False)
 # Fair three-way comparison on the identical 40k candidate universe.
 adb=pd.read_parquet(RESULTS/"archs4_query_database_manifest.parquet"); lookup=dict(zip(adb.gsm,adb.matrix_row)); missing=set(cm.gsm)-lookup.keys(); assert not missing
 rr=np.array([lookup[x] for x in cm.gsm],int); az=np.memmap('/home/walt/bridge-rna/embeddings/archs4/sample_embeddings.float16.mmap',mode='r',dtype='float16',shape=(940455,512))[adb.iloc[rr].global_index.to_numpy(int)]
 ah=np.load(HERE/"work/archs4_hallmark_scores.float32.npy",mmap_mode="r")[rr]
 tz=np.load(p["tcga_embeddings"],mmap_mode="r"); probe=np.load(p["frozen_hallmark_head"]); th=apply_probe(tz,probe)
 zc=np.vstack([np.asarray(tz[ix],np.float32).mean(0) for ix in groups]); hc=np.vstack([th[ix].mean(0) for ix in groups])
 method_frames=[]
 for method,qq,dd in [("bridge512_cosine",zc,az),("hallmark50_cosine",hc,ah)]:
  ii,ss=exact_topk(qq,dd,100,"cuda:0",len(cancers),log,method+" 40k")
  rec=[]
  for qi,c in enumerate(cancers):
   for rank,j in enumerate(ii[qi],1): rec.append({"query_index":qi,"query_cancer":c,"rank":rank,"gsm":cm.iloc[j].gsm,"similarity":float(ss[qi,rank-1]),"method":method})
  method_frames.append(pd.DataFrame(rec))
 pairs=[]
 for label,a,b in [("bridge_vs_hallmark",method_frames[0],method_frames[1]),("bridge_vs_expression",method_frames[0],out),("hallmark_vs_expression",method_frames[1],out)]:
  x=overlap_table(a,b); x["comparison"]=label; x["cancer_type"]=[cancers[i] for i in x.query_index]; pairs.append(x)
 pd.concat(pairs).to_csv(RESULTS/"expression_subset_method_overlap.csv",index=False)
 manifest={"status":"complete","scope":"cancer centroids only; deterministic existing 40,000-sample cache","candidate_samples":n,"genes":g,"representation":"natural log1p(TPM), exact canonical Bridge order","same_candidate_universe_for_method_overlap":True,"reason_for_scope":"full 9,816 x 510,709 x 15,165 expression retrieval is unnecessarily expensive and explicitly permitted to be centroid/subset scoped"}
 (RESULTS/"expression_baseline_provenance.json").write_text(json.dumps(manifest,indent=2)); log("Expression baseline complete")

if __name__=="__main__": main()
