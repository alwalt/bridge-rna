#!/usr/bin/env python3
"""Run TCGA and OSDR queries against 50-D and 512-D ARCHS4 databases."""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np, pandas as pd
from common import CONFIG, RESULTS, WORK, apply_probe, l2_rows, save_json, sha256

TCGA_TERMS={
 "BRCA":["breast","mammary","carcinoma"],"COAD":["colon","colorectal"],"GBM":["glioblastoma","brain","glioma"],
 "KIRC":["kidney","renal","clear cell"],"KIRP":["kidney","renal","papillary"],"LIHC":["liver","hepatic","hepatocellular"],
 "LUAD":["lung","adenocarcinoma"],"LUSC":["lung","squamous"],"PRAD":["prostate"],"THCA":["thyroid"],"UCEC":["uterus","uterine","endometrial"]}

def top_hits(q:np.ndarray, ref:np.ndarray, k:int)->tuple[np.ndarray,np.ndarray]:
    q=l2_rows(np.atleast_2d(q)); best_i=[]; best_s=[]
    for v in q:
        scores=np.asarray(ref@v,dtype=np.float32); take=np.argpartition(scores,-k)[-k:]; take=take[np.argsort(scores[take])[::-1]]
        best_i.append(take); best_s.append(scores[take])
    return np.stack(best_i),np.stack(best_s)

def contributions(q,d,names,n=5):
    c=q*d/(np.linalg.norm(q)*np.linalg.norm(d)); ix=np.argsort(c)[::-1][:n]
    return "; ".join(f"{names[i]}={c[i]:.4f}" for i in ix)

def query_table(qmeta,qh,qb,refh,refb,refmeta,names,k=10):
    hi,hs=top_hits(qh,refh,k); bi,bs=top_hits(qb,refb,k); rows=[]
    for method,ii,ss,Q,R in [("Hallmark-50 cosine",hi,hs,qh,refh),("Bridge-512 cosine",bi,bs,qb,refb)]:
      for qi in range(len(qmeta)):
       for rank,(ri,sim) in enumerate(zip(ii[qi],ss[qi]),1):
        r=refmeta.iloc[int(ri)]; rec={**qmeta.iloc[qi].to_dict(),"method":method,"rank":rank,"archs4_sample_id":r.sample_id,"series_id":r.get("series_id"),"similarity":float(sim),"title":r.get("title"),"source_name":r.get("source_name_ch1"),"characteristics":r.get("characteristics_ch1")}
        rec["top_hallmark_contributions"]=(contributions(Q[qi],R[int(ri)],names) if method.startswith("Hallmark") else None); rows.append(rec)
    return pd.DataFrame(rows)

def text_coherence(frame):
    text=frame[["title","source_name","characteristics"]].fillna("").agg(" ".join,axis=1).str.lower()
    frame["metadata_keyword_match"]=[any(t in s for t in TCGA_TERMS.get(str(c),[])) for c,s in zip(frame.get("cohort",pd.Series("",index=frame.index)),text)]
    return frame

def main():
    started=time.time(); probe=np.load(RESULTS/"frozen/frozen_gtex_hallmark_probe.npz",allow_pickle=False); names=probe["hallmarks"].astype(str).tolist(); k=int(CONFIG["top_k"])
    refmeta=pd.read_parquet(RESULTS/"archs4/sample_metadata.parquet"); refh=np.load(WORK/"archs4_hallmark_cosine_index.float32.npy",mmap_mode="r")
    spec=json.loads((Path(CONFIG["archs4_embedding_dir"])/"embedding_manifest.json").read_text()); raw=np.memmap(Path(CONFIG["archs4_embedding_dir"])/"sample_embeddings.float16.mmap",mode="r",dtype=np.float16,shape=(spec["total_samples"],512))
    global_rows=refmeta.global_index.to_numpy(int); refb_path=WORK/"archs4_bridge_cosine_index.float32.npy"
    refb=np.lib.format.open_memmap(refb_path,mode="w+",dtype=np.float32,shape=(len(refmeta),512))
    for left in range(0,len(refmeta),32768):
        right=min(left+32768,len(refmeta)); refb[left:right]=l2_rows(np.asarray(raw[global_rows[left:right]],np.float32)); refb.flush()
    # TCGA sample queries: deterministic Hallmark medoid per representative cohort; cohort centroids are separate queries.
    tm=pd.read_parquet(CONFIG["tcga_manifest"]).sort_values("matrix_row").reset_index(drop=True); tb=np.load(CONFIG["tcga_embeddings"],mmap_mode="r")
    assert len(tm)==len(tb)==9816 and np.array_equal(tm.matrix_row.to_numpy(),np.arange(len(tm)))
    th=apply_probe(tb,probe); existing=pd.read_parquet(CONFIG["tcga_hallmark_profiles"]); assert existing.sample_id.astype(str).tolist()==tm.sample_id.astype(str).tolist()
    assert np.allclose(th,existing[names].to_numpy(),rtol=2e-5,atol=2e-5)
    cohorts=[x for x in ["BRCA","COAD","GBM","KIRC","KIRP","LIHC","LUAD","LUSC","PRAD","THCA","UCEC"] if x in set(tm.cohort)]
    qrows=[]; qh=[]; qb=[]
    for c in cohorts:
        ix=np.flatnonzero(tm.cohort.eq(c)); hc=th[ix].mean(0); bc=np.asarray(tb[ix],np.float32).mean(0); med=ix[np.argmax(l2_rows(th[ix])@l2_rows(hc[None])[0])]
        qrows += [{"query_id":f"TCGA_{c}_representative","query_type":"sample_medoid","cohort":c,"sample_id":tm.iloc[med].sample_id},{"query_id":f"TCGA_{c}_centroid","query_type":"cohort_centroid","cohort":c,"sample_id":None}]
        qh += [th[med],hc]; qb += [np.asarray(tb[med],np.float32),bc]
    qt=pd.DataFrame(qrows); qh=np.asarray(qh,np.float32); qb=np.asarray(qb,np.float32)
    tcga=text_coherence(query_table(qt,qh,qb,refh,refb,refmeta,names,k)); (RESULTS/"tcga").mkdir(parents=True,exist_ok=True); tcga.to_parquet(RESULTS/"tcga/retrieval_results.parquet",index=False); tcga.to_csv(RESULTS/"tcga/retrieval_results.csv",index=False)
    comp=tcga.groupby(["query_id","query_type","cohort","method"],as_index=False).agg(top10_keyword_precision=("metadata_keyword_match","mean"),mean_top10_similarity=("similarity","mean"),unique_studies=("series_id","nunique"))
    comp.to_csv(RESULTS/"tcga/hallmark50_vs_bridge512.csv",index=False)
    # OSDR: exact existing 112 liver samples. These are spaceflight (FLT) and ground-control (GC), not assumed NAFLD cases.
    om=pd.read_csv(CONFIG["osdr_manifest"]); ob=np.load(CONFIG["osdr_embeddings"],mmap_mode="r"); ox=np.load(CONFIG["osdr_expression"],mmap_mode="r")
    assert len(om)==len(ob)==len(ox)==112 and ox.shape[1]==15165 and om.sample_id.is_unique
    oh=apply_probe(ob,probe); qrows=[]; qh=[]; qb=[]
    for (mission,condition),g in om.groupby(["mission","condition"],sort=True):
        ix=g.index.to_numpy(); hc=oh[ix].mean(0); bc=np.asarray(ob[ix],np.float32).mean(0); med=ix[np.argmax(l2_rows(oh[ix])@l2_rows(hc[None])[0])]
        qrows += [{"query_id":f"OSDR_{mission}_{condition}_representative","query_type":"sample_medoid","mission":mission,"condition":condition,"sample_id":om.iloc[med].sample_id,"OSD":om.iloc[med].OSD},{"query_id":f"OSDR_{mission}_{condition}_centroid","query_type":"group_centroid","mission":mission,"condition":condition,"sample_id":None,"OSD":";".join(sorted(g.OSD.unique()))}]
        qh += [oh[med],hc]; qb += [np.asarray(ob[med],np.float32),bc]
    oq=pd.DataFrame(qrows); osdr=query_table(oq,np.asarray(qh),np.asarray(qb),refh,refb,refmeta,names,k); (RESULTS/"osdr").mkdir(parents=True,exist_ok=True)
    osdr.to_parquet(RESULTS/"osdr/retrieval_results.parquet",index=False); osdr.to_csv(RESULTS/"osdr/retrieval_results.csv",index=False)
    osdr.groupby(["query_id","query_type","mission","condition","method"],as_index=False).agg(mean_top10_similarity=("similarity","mean"),unique_studies=("series_id","nunique")).to_csv(RESULTS/"osdr/hallmark50_vs_bridge512.csv",index=False)
    pd.DataFrame(oh,columns=names).assign(sample_id=om.sample_id).to_parquet(RESULTS/"osdr/hallmark_signatures.parquet",index=False)
    pd.DataFrame(th,columns=names).assign(sample_id=tm.sample_id,cohort=tm.cohort).to_parquet(RESULTS/"tcga/hallmark_signatures.parquet",index=False)
    save_json(RESULTS/"query_provenance.json",{"tcga_samples":len(tm),"osdr_samples":len(om),"osdr_identity":"112 repository-frozen mouse liver RNA-seq samples spanning FLT and GC; no samples relabeled as fatty-liver disease","tcga_head_fit":False,"tcga_tuning":False,"cross_species_caveat":"OSDR is mouse; the frozen model has no species embedding and uses its one-to-one ordered vocabulary, but the GTEx probe was trained on human samples.","baseline":"same unmodified 512-D Bridge embeddings, raw cosine","runtime_seconds":time.time()-started,"input_hashes":{"tcga_embeddings":sha256(CONFIG["tcga_embeddings"]),"osdr_embeddings":sha256(CONFIG["osdr_embeddings"]),"osdr_expression":sha256(CONFIG["osdr_expression"])}})
if __name__=="__main__": main()

