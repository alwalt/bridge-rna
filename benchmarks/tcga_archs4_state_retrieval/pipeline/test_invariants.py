#!/usr/bin/env python3
from pathlib import Path
import json, numpy as np, pandas as pd
H=Path(__file__).resolve().parents[1]; R=H/"results"; W=H/"work"
qa=json.loads((R/"qa.json").read_text()); assert qa["passed"]
assert qa["dimensions"]=={"archs4_bridge":[510709,512],"tcga_bridge":[9816,512],"archs4_hallmark":[510709,50],"tcga_hallmark":[9816,50]}
am=pd.read_parquet(R/"archs4_query_database_manifest.parquet"); tm=pd.read_parquet(R/"tcga_query_manifest.parquet")
assert len(am)==510709 and am.gsm.is_unique and am.matrix_row.tolist()==list(range(len(am)))
assert len(tm)==9816 and tm.patient_id.is_unique and tm.matrix_row.tolist()==list(range(len(tm)))
assert np.load(W/"archs4_hallmark_scores.float32.npy",mmap_mode="r").shape==(510709,50)
assert np.load(W/"tcga_hallmark_scores.float32.npy",mmap_mode="r").shape==(9816,50)
for method in ["bridge512_cosine","hallmark50_cosine"]:
 x=pd.read_parquet(R/f"top100_individual_{method}.parquet")
 assert len(x)==981600 and x.groupby("query_index").size().eq(100).all() and x.groupby("query_index")["rank"].apply(lambda s:s.tolist()==list(range(1,101))).all()
 assert np.isfinite(x.similarity).all() and x.gsm.isin(am.gsm).all()
for name in ["individual_method_overlap.csv","centroid_method_overlap.csv","metadata_coherence.csv","retrieval_hubs.csv","pathway_contributions.csv","representative_functional_matches.csv"]: assert (R/name).stat().st_size>0
prov=json.loads((R/"provenance.json").read_text()); assert prov["bridge_frozen"] and prov["hallmark_head_frozen"] and len(prov["hallmarks"])==50
print("PASS: full benchmark invariants")

