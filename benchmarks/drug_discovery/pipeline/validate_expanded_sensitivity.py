#!/usr/bin/env python3
"""Validate the new sensitivity layer and frozen-input invariants."""

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "results/expanded_chembl_sensitivity"

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    return h.hexdigest()

prov=json.loads((OUT/"provenance.json").read_text())
modules=HERE/"results/evaluation/gene_modules.parquet"
assert sha(modules)==prov["frozen_module_sha256"]
assert not any(prov["protected_steps_rerun"].values())
enr=pd.read_parquet(OUT/"expanded_drug_enrichment.parquet")
assert len(enr)==9*2*366
parts=pd.read_csv(OUT/"gene_partitions.csv.gz")
sizes=parts.groupby(["dataset","partition"]).size().unstack()
assert ((sizes["Bridge-only"]+sizes["Shared"])==500).all()
assert ((sizes["DE-only"]+sizes["Shared"])==500).all()
deletion=pd.read_csv(OUT/"partition_deletion_summary.csv")
assert len(deletion)==27 and deletion.dataset.nunique()==9
paths=pd.read_csv(OUT/"partition_pathway_summary.csv")
assert len(paths)==30
effects=pd.read_csv(OUT/"expanded_primary_effects.csv")
assert len(effects)==3
old=pd.read_csv(HERE/"results/evaluation/primary_effects.csv").set_index("condition")
expected={"Radiation":(0.155354,0.041667),"Bone loss":(0.142308,0.180556),"Muscle atrophy":(0.714286,0.182299)}
for c,(b,d) in expected.items():
    assert np.isclose(old.loc[c,"bridge_mean_pairwise_top25_jaccard"],b,atol=1e-6)
    assert np.isclose(old.loc[c,"de_mean_pairwise_top25_jaccard"],d,atol=1e-6)
summary={"valid":True,"expanded_enrichment_rows":len(enr),"eligible_drugs":366,
         "datasets":9,"partitions":27,"pathway_queries":30,
         "protected_steps_rerun":prov["protected_steps_rerun"],"original_phase2_values_unchanged":True}
(OUT/"validation_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
