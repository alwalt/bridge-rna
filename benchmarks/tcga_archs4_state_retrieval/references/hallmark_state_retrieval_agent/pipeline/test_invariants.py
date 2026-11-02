#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np, pandas as pd
HERE=Path(__file__).resolve().parents[1]; R=HERE/"results"; W=HERE/"work"
p=np.load(R/"frozen/frozen_gtex_hallmark_probe.npz",allow_pickle=False); order=pd.read_csv(R/"frozen/frozen_hallmark_order.csv")
assert p["weight"].shape==(50,512) and p["mean"].shape==p["scale"].shape==(512,)
assert order.hallmark.tolist()==p["hallmarks"].astype(str).tolist()
m=pd.read_parquet(R/"archs4/sample_metadata.parquet"); h=np.load(W/"archs4_hallmark_signatures.float32.npy",mmap_mode="r"); i=np.load(W/"archs4_hallmark_cosine_index.float32.npy",mmap_mode="r")
assert len(m)==len(h)==len(i)==510709 and h.shape==i.shape==(510709,50) and m.sample_id.is_unique
assert np.allclose(np.linalg.norm(i[::10000],axis=1),1,atol=2e-6)
t=pd.read_parquet(R/"tcga/hallmark_signatures.parquet"); o=pd.read_parquet(R/"osdr/hallmark_signatures.parquet")
assert t.shape==(9816,52) and o.shape==(112,51) and t.sample_id.is_unique and o.sample_id.is_unique
q=json.loads((R/"query_provenance.json").read_text()); assert q["tcga_head_fit"] is False and q["tcga_tuning"] is False
a=pd.read_csv(R/"similarity_scaling_audit/similarity_method_comparison.csv").set_index("method")
assert np.isclose(a.loc["Raw cosine","metadata_precision_at_10"],a.loc["GTEx-target-centered cosine","metadata_precision_at_10"])
assert np.isclose(a.loc["Raw cosine","liver_precision_at_10"],a.loc["GTEx-target-z-scored cosine","liver_precision_at_10"])
assert a.loc["Pearson correlation","liver_precision_at_10"] < a.loc["Bridge-512 cosine","liver_precision_at_10"]
ap=json.loads((R/"similarity_scaling_audit/provenance.json").read_text()); assert ap["frozen_outputs_only"] and not ap["bridge_inference"] and not ap["head_retrained"]
pan=pd.read_csv(R/"tcga33_retrieval/tcga33_cancer_comparison.csv")
assert len(pan)==33 and pan.cancer_type.nunique()==33
long=pd.read_csv(R/"tcga33_retrieval/per_cancer_metrics_long.csv")
assert set(long["query_type"])=={"individual_patient","cancer_centroid"}
assert set(long["k"])=={10,50,100} and set(long["method"])=={"Hallmark-50 cosine","Bridge-512 cosine"}
assert set(long["criterion"])=={"exact_disease","tissue_organ","malignancy","same_tissue_related_cancer","disease_or_same_tissue_cancer"}
cent=pd.read_csv(R/"tcga33_retrieval/centroid_top100_hits.csv")
assert len(cent)==33*2*100 and cent.groupby(["cancer_type","method"]).size().eq(100).all()
tp=json.loads((R/"tcga33_retrieval/provenance.json").read_text())
assert tp["patients"]==9942 and tp["cancer_types"]==33 and tp["frozen_outputs_only"] and not tp["bridge_inference"] and not tp["head_retrained"]
sm=pd.read_csv(R/"signal_matching_audit/macro_method_comparison.csv")
expected_methods={"whole_vector_cosine","signed_absolute_rank","signed_top5_overlap","signed_top10_overlap","query_weighted_top10","asymmetric_containment_top10","Bridge-512 reference"}
assert set(sm.method)==expected_methods and set(sm.k)=={5,10,50}
deep=pd.read_csv(R/"signal_matching_audit/deep_inspection_top5.csv")
assert set(deep.cancer_type)=={"UVM","CHOL","LUAD","PCPG","OV","THCA","KICH"}
assert deep.groupby(["cancer_type","method"]).size().eq(5).all()
null=pd.read_csv(R/"signal_matching_audit/null_score_distributions.csv")
assert null.groupby(["cancer_type","method"]).size().eq(6).all()
sp=json.loads((R/"signal_matching_audit/provenance.json").read_text())
assert sp["reference_samples"]==510709 and sp["background_samples_per_query"]==510709
assert sp["frozen_outputs_only"] and not sp["bridge_inference"] and not sp["head_retrained"]
oc=json.loads((R/"osdr_atlas/pre_inference_census.json").read_text())
assert oc["eligible_studies"]==64 and oc["eligible_samples"]==1471 and oc["eligible_matched_groups"]==88
om=pd.read_parquet(R/"osdr_atlas/eligible_sample_manifest.parquet")
assert len(om)==1471 and om.group_id.nunique()==88 and set(om.condition)=={"FLT","GC"}
assert not om.duplicated(["study","sample_id"]).any()
op=json.loads((R/"osdr_atlas/representation_provenance.json").read_text())
assert op["cached_rows_reused"]==112 and op["new_rows_inferred"]==1359 and op["cache_input_compatibility_max_abs_difference"]<1e-10
orp=json.loads((R/"osdr_atlas/retrieval_provenance.json").read_text())
assert orp["individual_queries"]==1471 and orp["centroid_queries"]==176 and orp["reference_samples"]==510709
oa=pd.read_csv(R/"osdr_atlas/centroid_top5_biological_audit_scored.csv")
assert len(oa)==88*2*3*5 and oa.groupby(["group_id","condition","method"]).size().eq(5).all()
orc=pd.read_csv(R/"osdr_atlas/replicate_consistency_complete.csv")
assert set(orc.k)=={10,50,100} and orc.group_id.nunique()==88
print("all invariants passed")

# Final benchmark-closure invariants.
import hashlib
def digest(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  while b:=f.read(8<<20): h.update(b)
 return h.hexdigest()
frozen=json.loads((R/"frozen_head_compatibility_provenance.json").read_text())["artifacts"]
assert digest(R/"frozen/frozen_gtex_hallmark_probe.npz")==frozen["probe"]
assert digest(R/"frozen/frozen_hallmark_order.csv")==frozen["order"]
assert digest(Path("/home/walt/bridge-rna/model/r7hnr92k/best_model.pt"))==frozen["checkpoint"]
assert digest(Path("/home/walt/bridge-rna/data/ensembl/canonical_genes.csv"))==frozen["canonical_source"]
multi=json.loads((R/"archs4_multispecies/atlas_provenance.json").read_text())
assert (multi["human_samples"],multi["mouse_samples"],multi["combined_samples"])==(510709,429746,940455)
assert not multi["bridge_inference_rerun"] and not multi["hallmark_head_retrained"]
cm=pd.read_parquet(R/"archs4_multispecies/combined_row_manifest.parquet")
assert len(cm)==940455 and cm.species.value_counts().to_dict()=={"human":510709,"mouse":429746}
tm=pd.read_parquet("/home/walt/bridge-rna/benchmarks/tcga_downstream/results/cohort_manifest.parquet")
assert len(tm)==9816 and tm.cohort.eq("LUAD").sum()==516

leak=pd.read_csv(R/"luad_focused_audit/luad_centroid_top100_leakage_audit.csv")
assert len(leak)==200 and leak.gsm.nunique()==181
assert leak.groupby("method").size().eq(100).all()
assert leak.groupby("method")["rank"].apply(lambda x:set(x)==set(range(1,101))).all()
assert not leak.audit_status.eq("TCGA-derived/reused").any()
assert not leak.exact_tcga_query_identifier_duplicates.fillna("").astype(str).ne("").any()

cross=pd.read_csv(R/"luad_focused_audit/luad_union_top100_cross_space_scores.csv")
assert cross.groupby("query_type").size().to_dict()=={"all_patient_centroid":181,"representative_patient":172}
assert cross.groupby("query_type").bridge_top100.sum().eq(100).all()
assert cross.groupby("query_type").hallmark_top100.sum().eq(100).all()
cent_luad=cent[cent.cancer_type.eq("LUAD")]
for method,flag,score in [("Bridge-512 cosine","bridge_top100","bridge_similarity"),("Hallmark-50 cosine","hallmark_top100","hallmark_similarity")]:
 got=cross[cross.query_type.eq("all_patient_centroid")&cross[flag]].sort_values(score,ascending=False)
 exp=cent_luad[cent_luad.method.eq(method)].sort_values("rank")
 assert got.gsm.tolist()==exp.gsm.tolist()
 assert np.allclose(got[score],exp.cosine_similarity,atol=2e-6)
ov=pd.read_csv(R/"luad_focused_audit/luad_rank_overlap.csv")
assert set(ov.k)=={5,10,50,100} and len(ov)==8
assert ov.query("query_type=='all_patient_centroid' and k==5").intersection.item()==0
assert json.loads((R/"tcga_pan_cancer_positive_control/bridge512_visualization_only_provenance.json").read_text())["visualization_only"]
for f in [R/"luad_focused_audit/luad_centroid_bridge_vs_hallmark_scatter.png",
          R/"luad_focused_audit/luad_hallmark_terms_bridge512_vs_hallmark50.png",
          R/"tcga_pan_cancer_positive_control/figures/cancer_similarity_matrices_final.png"]:
 assert f.exists() and f.stat().st_size>10000
print("final closure invariants passed")
