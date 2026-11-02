#!/usr/bin/env python3
"""Frozen-output-only all-patient retrieval audit for all 33 TCGA cancer types."""
from __future__ import annotations
import json,re,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from common import CONFIG,RESULTS,WORK,apply_probe,l2_rows,save_json,sha256

OUT=RESULTS/"tcga33_retrieval"; KS=(10,50,100); TOP=100
PARITY=Path("/home/walt/bridge-rna/benchmarks/tcga_downstream")
RULES={
"ACC":("adrenocortical|adrenal cortical", "adrenal"),"BLCA":("bladder.*(cancer|carcinoma|tumou?r)|urothelial carcinoma", "bladder|urothel"),
"BRCA":("breast.*(cancer|carcinoma|tumou?r)|mammary carcinoma", "breast|mammary"),"CESC":("cervi(cal|x).*(cancer|carcinoma|tumou?r)", "cervi(cal|x)"),
"CHOL":("cholangiocarcinoma|bile duct.*(cancer|carcinoma|tumou?r)", "bile duct|cholangi|biliary"),"COAD":("colon.*(cancer|carcinoma|adenocarcinoma|tumou?r)", "colon|colonic"),
"DLBC":("diffuse large b.?cell|dlbcl|b.?cell lymphoma", "lymph|b.?cell"),"ESCA":("esophag.*(cancer|carcinoma|adenocarcinoma|tumou?r)", "esophag"),
"GBM":("glioblastoma|gbm", "brain|cerebr|glioma|glioblastoma"),"HNSC":("head.?and.?neck.*(cancer|carcinoma)|oral.*(cancer|carcinoma)|tongue.*(cancer|carcinoma)", "head.?and.?neck|oral|tongue|laryn|pharyn"),
"KICH":("chromophobe.*(renal|kidney|rcc)|(renal|kidney|rcc).*chromophobe", "kidney|renal|\\brcc\\b"),"KIRC":("clear.?cell.*(renal|kidney|rcc)|(renal|kidney|rcc).*clear.?cell", "kidney|renal|\\brcc\\b"),
"KIRP":("papillary.*(renal|kidney|rcc)|(renal|kidney|rcc).*papillary", "kidney|renal|\\brcc\\b"),"LAML":("acute myeloid|acute myelogenous|\\baml\\b", "blood|bone marrow|myeloid|leuk|\\baml\\b"),
"LGG":("low.?grade glioma|oligodendroglioma|astrocytoma", "brain|cerebr|glioma"),"LIHC":("hepatocellular|liver.*(cancer|carcinoma|tumou?r)|\\bhcc\\b", "liver|hepatic|hepatocyte|\\bhcc\\b"),
"LUAD":("lung adenocarcinoma|pulmonary adenocarcinoma", "lung|pulmonary|bronch"),"LUSC":("lung.*squamous|pulmonary.*squamous", "lung|pulmonary|bronch"),
"MESO":("mesothelioma", "mesothel|pleura"),"OV":("ovari(an|y).*(cancer|carcinoma|tumou?r)", "ovari(an|y)|fallopian"),
"PAAD":("pancrea.*(cancer|carcinoma|adenocarcinoma|tumou?r)", "pancrea"),"PCPG":("pheochromocytoma|paraganglioma", "adrenal|paragangli|pheochromo"),
"PRAD":("prostate.*(cancer|carcinoma|adenocarcinoma|tumou?r)", "prostate"),"READ":("rect(al|um).*(cancer|carcinoma|adenocarcinoma|tumou?r)", "rect(al|um)|colorectal"),
"SARC":("sarcoma", "soft tissue|connective|muscle|bone"),"SKCM":("melanoma", "skin|cutaneous|melanocyte"),
"STAD":("gastric.*(cancer|carcinoma|adenocarcinoma|tumou?r)|stomach.*(cancer|carcinoma|tumou?r)", "gastric|stomach"),"TGCT":("testicular.*(cancer|tumou?r)|germ.?cell tumou?r", "testi(cle|cular)|germ.?cell"),
"THCA":("thyroid.*(cancer|carcinoma|tumou?r)", "thyroid"),"THYM":("thymoma|thymic.*(cancer|carcinoma|tumou?r)", "thymus|thymic"),
"UCEC":("endometri.*(cancer|carcinoma|tumou?r)|uterine.*(cancer|carcinoma)", "uter|endometri"),"UCS":("uterine carcinosarcoma|malignant mixed mullerian", "uter"),
"UVM":("uveal melanoma|ocular melanoma|choroidal melanoma", "uvea|ocular|choroid|eye")}
MALIGNANT=r"cancer|carcinoma|adenocarcinoma|tumou?r|neoplasm|malignan|leukemia|lymphoma|sarcoma|melanoma|glioma|glioblastoma|mesothelioma|myeloma|\\brcc\\b|\\bhcc\\b|\\bdlbcl\\b|\\baml\\b"
NONMALIGNANT=r"healthy|normal tissue|non.?tumou?r|non.?cancer|adjacent normal|control donor|benign"

def log(msg):
    stamp=pd.Timestamp.now(tz="UTC").isoformat(); print(f"[{stamp}] {msg}",flush=True)
    with (OUT/"run.log").open("a") as f:f.write(f"[{stamp}] {msg}\n")

def load_queries(probe,names):
    base_m=pd.read_parquet(CONFIG["tcga_manifest"]).sort_values("matrix_row").reset_index(drop=True); base_b=np.load(CONFIG["tcga_embeddings"],mmap_mode="r"); base_h=pd.read_parquet(RESULTS/"tcga/hallmark_signatures.parquet")
    assert len(base_m)==len(base_b)==len(base_h)==9816 and base_h.sample_id.astype(str).tolist()==base_m.sample_id.astype(str).tolist()
    pm=pd.read_csv(PARITY/"results/bulkformer_parity_pan_cancer_manifest.csv"); lm=pm[pm.cancer_label.eq("LAML")].reset_index(drop=True); lb=np.load(PARITY/"work/bulkformer_parity_laml_ours_45.6m_embeddings.npy",mmap_mode="r")
    assert len(lm)==len(lb)==126 and lm.patient_id.is_unique
    lh=apply_probe(lb,probe); meta=base_m[["sample_id","patient_id","cohort"]].rename(columns={"cohort":"cancer_type"}).copy(); meta["source_cache"]="frozen_primary_tumor"
    lmeta=pd.DataFrame({"sample_id":lm.sample_id.astype(str),"patient_id":lm.patient_id.astype(str),"cancer_type":"LAML","source_cache":"frozen_laml_parity"})
    meta=pd.concat([meta,lmeta],ignore_index=True); b=np.vstack([np.asarray(base_b,np.float32),np.asarray(lb,np.float32)]); h=np.vstack([base_h[names].to_numpy(np.float32),lh])
    assert len(meta)==len(b)==len(h)==9942 and meta.cancer_type.nunique()==33 and meta.patient_id.is_unique
    return meta,b,h,lm

def annotate(meta):
    text=meta[["title","source_name_ch1","characteristics_ch1"]].fillna("").agg(" ".join,axis=1).str.lower().str.replace(r"\s+"," ",regex=True)
    any_tissue=np.zeros(len(meta),bool); any_exact=np.zeros(len(meta),bool); exact={}; tissue={}
    for cancer,(ep,tp) in RULES.items():
        exact[cancer]=text.str.contains(ep,regex=True).to_numpy(); tissue[cancer]=text.str.contains(tp,regex=True).to_numpy(); any_exact|=exact[cancer]; any_tissue|=tissue[cancer]
    malignant=text.str.contains(MALIGNANT,regex=True).to_numpy(); nonmalignant=text.str.contains(NONMALIGNANT,regex=True).to_numpy()
    return text,exact,tissue,any_exact,any_tissue,malignant,nonmalignant

def retrieve_gpu(q,ref_path,dim,label,device="cuda:0",batch=64):
    started=time.time(); ref=np.load(ref_path,mmap_mode="r"); assert ref.shape==(510709,dim)
    dev=torch.device(device if torch.cuda.is_available() else "cpu"); log(f"{label} exact top-{TOP}: queries={len(q)} refs={len(ref)} dim={dim} device={dev}; estimate 2-8 minutes for 512-D, <1 minute for 50-D")
    # Index is already row-normalized. float32 preserves the saved exact-cosine definition.
    rt=torch.from_numpy(np.asarray(ref)).to(dev); out_i=np.empty((len(q),TOP),np.int32); out_s=np.empty((len(q),TOP),np.float32)
    for left in range(0,len(q),batch):
        right=min(left+batch,len(q)); qt=torch.from_numpy(l2_rows(q[left:right])).to(dev); score=qt@rt.T; val,idx=torch.topk(score,TOP,dim=1,sorted=True); out_i[left:right]=idx.cpu().numpy(); out_s[left:right]=val.cpu().numpy()
        if right==len(q) or right%512==0: log(f"{label} {right}/{len(q)} elapsed={time.time()-started:.1f}s rate={right/(time.time()-started):.1f} queries/s")
    del rt; torch.cuda.empty_cache() if dev.type=="cuda" else None
    return out_i,out_s

def metrics(indices,qmeta,exact,tissue,any_exact,any_tissue,malignant,nonmalignant,query_type):
    rows=[]
    for qi,cancer in enumerate(qmeta.cancer_type):
      for k in KS:
        ix=indices[qi,:k]; ex=exact[cancer][ix]; ti=tissue[cancer][ix]; mal=malignant[ix]
        # Require an identifiable cancer annotation for a related-cancer hit.
        related=ti&any_exact[ix]&~ex; program=ex|related
        evals={"exact_disease":any_exact[ix],"tissue_organ":any_tissue[ix],"malignancy":mal|nonmalignant[ix],"same_tissue_related_cancer":any_exact[ix]&ti,"disease_or_same_tissue_cancer":any_exact[ix]|(ti&(mal|nonmalignant[ix]))}
        vals={"exact_disease":ex,"tissue_organ":ti,"malignancy":mal,"same_tissue_related_cancer":related,"disease_or_same_tissue_cancer":program}
        for criterion,v in vals.items():
          ev=evals[criterion]; rows.append({"query_index":qi,"patient_id":qmeta.iloc[qi].patient_id,"cancer_type":cancer,"query_type":query_type,"k":k,"criterion":criterion,"confirmed_precision":v.mean(),"annotation_coverage":ev.mean(),"conditional_precision":v[ev].mean() if ev.any() else np.nan,"confirmed_hits":int(v.sum()),"evaluable_hits":int(ev.sum())})
    return pd.DataFrame(rows)

def main():
    OUT.mkdir(parents=True,exist_ok=True); (OUT/"figures").mkdir(exist_ok=True); started=time.time(); (OUT/"run.log").write_text("")
    probe=np.load(RESULTS/"frozen/frozen_gtex_hallmark_probe.npz",allow_pickle=False); names=probe["hallmarks"].astype(str).tolist(); qmeta,b,h,lm=load_queries(probe,names)
    refmeta=pd.read_parquet(RESULTS/"archs4/sample_metadata.parquet"); text,exact,tissue,any_exact,any_tissue,malignant,nonmalignant=annotate(refmeta)
    save_json(OUT/"metadata_annotation_rules.json",{"exact_and_tissue_regex":{k:{"exact":v[0],"tissue":v[1]} for k,v in RULES.items()},"malignancy_regex":MALIGNANT,"explicit_nonmalignant_regex":NONMALIGNANT,"missing_policy":"confirmed precision uses positives/k; conditional precision excludes unrecognized/ambiguous metadata and is always accompanied by annotation coverage"})
    # Append 33 centroids after all individual patients so both query types share one retrieval pass.
    cancers=sorted(RULES); centmeta=pd.DataFrame({"sample_id":[None]*33,"patient_id":[f"{c}_CENTROID" for c in cancers],"cancer_type":cancers,"source_cache":"centroid"})
    cb=np.vstack([b[qmeta.cancer_type.eq(c)].mean(0) for c in cancers]); ch=np.vstack([h[qmeta.cancer_type.eq(c)].mean(0) for c in cancers]); allmeta=pd.concat([qmeta,centmeta],ignore_index=True); allb=np.vstack([b,cb]); allh=np.vstack([h,ch])
    hi,hs=retrieve_gpu(allh,WORK/"archs4_hallmark_cosine_index.float32.npy",50,"Hallmark-50"); bi,bs=retrieve_gpu(allb,WORK/"archs4_bridge_cosine_index.float32.npy",512,"Bridge-512")
    # Large, deterministic ranking arrays are regenerable work products; compact
    # annotated hit and metric tables remain under results/.
    np.savez_compressed(WORK/"tcga33_top100_indices_scores.npz",hallmark_indices=hi,hallmark_scores=hs,bridge_indices=bi,bridge_scores=bs)
    parts=[]
    for method,idx in [("Hallmark-50 cosine",hi),("Bridge-512 cosine",bi)]:
      a=metrics(idx[:len(qmeta)],qmeta,exact,tissue,any_exact,any_tissue,malignant,nonmalignant,"individual_patient"); a["method"]=method; parts.append(a)
      c=metrics(idx[len(qmeta):],centmeta,exact,tissue,any_exact,any_tissue,malignant,nonmalignant,"cancer_centroid"); c["method"]=method; parts.append(c)
    perq=pd.concat(parts,ignore_index=True); perq.to_parquet(OUT/"per_query_metrics.parquet",index=False)
    summary=perq.groupby(["cancer_type","query_type","method","criterion","k"],as_index=False).agg(confirmed_precision=("confirmed_precision","mean"),annotation_coverage=("annotation_coverage","mean"),conditional_precision=("conditional_precision","mean"),queries=("query_index","size"))
    summary.to_csv(OUT/"per_cancer_metrics_long.csv",index=False)
    # Required compact 33-cancer table uses disease/program P@K, with every separate criterion retained long-form.
    primary=summary[(summary.query_type.eq("individual_patient"))&(summary.criterion.eq("disease_or_same_tissue_cancer"))].pivot(index="cancer_type",columns=["method","k"],values="confirmed_precision")
    table=pd.DataFrame(index=primary.index)
    for k in KS:
      table[f"hallmark_p@{k}"]=primary[("Hallmark-50 cosine",k)]; table[f"bridge_p@{k}"]=primary[("Bridge-512 cosine",k)]; table[f"delta_p@{k}"]=table[f"hallmark_p@{k}"]-table[f"bridge_p@{k}"]
    table=table.reset_index(); table["classification"]=np.select([table["delta_p@50"]>=.10,table["delta_p@50"]<=-.10],["clearly_improves","clearly_worsens"],default="mixed_or_similar"); table.to_csv(OUT/"tcga33_cancer_comparison.csv",index=False)
    # Centroid hit table with contribution decomposition for Hallmark hits.
    rows=[]; href=np.load(WORK/"archs4_hallmark_cosine_index.float32.npy",mmap_mode="r"); qhn=l2_rows(ch)
    for method,idx,score in [("Hallmark-50 cosine",hi[-33:],hs[-33:]),("Bridge-512 cosine",bi[-33:],bs[-33:])]:
      for qi,cancer in enumerate(cancers):
       for rank,(ri,sim) in enumerate(zip(idx[qi],score[qi]),1):
        r=refmeta.iloc[int(ri)]; rec={"cancer_type":cancer,"method":method,"rank":rank,"gsm":r.sample_id,"gse":r.series_id,"cosine_similarity":float(sim),"title":r.title,"source_name":r.source_name_ch1,"characteristics":r.characteristics_ch1,"exact_disease":bool(exact[cancer][ri]),"tissue_organ":bool(tissue[cancer][ri]),"malignancy":bool(malignant[ri]),"same_tissue_related_cancer":bool(tissue[cancer][ri] and any_exact[ri] and not exact[cancer][ri])}
        if method.startswith("Hallmark"):
          cc=qhn[qi]*href[int(ri)]; order=np.argsort(cc)[::-1][:8]; rec["top_hallmark_contributions"]="; ".join(f"{names[j]}={cc[j]:.4f}" for j in order)
        else:rec["top_hallmark_contributions"]=None
        rows.append(rec)
    cent=pd.DataFrame(rows); cent.to_csv(OUT/"centroid_top100_hits.csv",index=False); cent[cent["rank"].le(10)].to_csv(OUT/"centroid_top10_hits.csv",index=False)
    # Hypothesis: compare method delta for disease/program versus tissue identification.
    macro=summary[summary.query_type.eq("individual_patient")].groupby(["method","criterion","k"],as_index=False).agg(macro_precision=("confirmed_precision","mean"),macro_conditional_precision=("conditional_precision","mean"),macro_coverage=("annotation_coverage","mean"))
    macro.to_csv(OUT/"macro_hypothesis_metrics.csv",index=False)
    hp=summary[(summary.query_type.eq("individual_patient"))&(summary.k.eq(50))].pivot(index=["cancer_type","criterion"],columns="method",values="confirmed_precision").reset_index(); hp["hallmark_minus_bridge"]=hp["Hallmark-50 cosine"]-hp["Bridge-512 cosine"]; hp.to_csv(OUT/"per_cancer_criterion_differences_p50.csv",index=False)
    # Figures.
    import matplotlib.pyplot as plt
    t=table.sort_values("delta_p@50"); y=np.arange(len(t)); fig,ax=plt.subplots(figsize=(10,11)); ax.barh(y,t["delta_p@50"],color=np.where(t["delta_p@50"]>=0,"#2a9d8f","#e76f51")); ax.axvline(0,color="black",lw=1); ax.set_yticks(y,t.cancer_type); ax.set(xlabel="Hallmark-50 − Bridge-512 disease/program P@50",title="TCGA individual-patient retrieval by cancer type"); fig.tight_layout(); fig.savefig(OUT/"figures/per_cancer_method_difference.png",dpi=240); fig.savefig(OUT/"figures/per_cancer_method_difference.pdf"); plt.close(fig)
    criteria=["exact_disease","tissue_organ","malignancy","same_tissue_related_cancer","disease_or_same_tissue_cancer"]; heat=hp.pivot(index="cancer_type",columns="criterion",values="hallmark_minus_bridge").reindex(index=cancers,columns=criteria); fig,ax=plt.subplots(figsize=(12,12)); im=ax.imshow(heat,cmap="coolwarm",vmin=-max(abs(np.nanmin(heat)),abs(np.nanmax(heat))),vmax=max(abs(np.nanmin(heat)),abs(np.nanmax(heat))),aspect="auto"); ax.set_xticks(range(len(criteria)),criteria,rotation=30,ha="right"); ax.set_yticks(range(33),cancers); fig.colorbar(im,ax=ax,label="Hallmark − Bridge P@50"); ax.set_title("Which TCGA cancer types benefit from functional retrieval?"); fig.tight_layout(); fig.savefig(OUT/"figures/per_cancer_criterion_heatmap.png",dpi=240); fig.savefig(OUT/"figures/per_cancer_criterion_heatmap.pdf"); plt.close(fig)
    save_json(OUT/"provenance.json",{"frozen_outputs_only":True,"bridge_inference":False,"head_retrained":False,"patients":len(qmeta),"cancer_types":33,"solid_primary_patients":9816,"laml_peripheral_blood_patients":126,"neighbors":list(KS),"individual_queries":"all one-specimen-per-patient frozen TCGA queries","centroids":"arithmetic mean within cancer type before L2 normalization","runtime_seconds":time.time()-started,"inputs":{"frozen_compatibility_manifest":str(RESULTS/"frozen_head_compatibility_provenance.json"),"frozen_probe_sha256":sha256(RESULTS/"frozen/frozen_gtex_hallmark_probe.npz"),"laml_embeddings":str(PARITY/"work/bulkformer_parity_laml_ours_45.6m_embeddings.npy"),"laml_embeddings_sha256":sha256(PARITY/"work/bulkformer_parity_laml_ours_45.6m_embeddings.npy"),"hallmark_index":str(WORK/"archs4_hallmark_cosine_index.float32.npy"),"bridge_index":str(WORK/"archs4_bridge_cosine_index.float32.npy")},"metadata_limitations":["Rule-based free-text annotations are conservative and imperfect.","Confirmed precision uses only explicit positive matches; annotation coverage and conditional precision prevent unknown metadata from being mislabeled as confirmed negative.","LAML uses the existing compatible peripheral-blood parity cache because it is absent from the primary-tumor cohort."]})
    log(f"COMPLETE runtime={time.time()-started:.1f}s")
if __name__=="__main__":main()
