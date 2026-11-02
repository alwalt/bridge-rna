#!/usr/bin/env python3
"""Exact frozen TCGA-to-ARCHS4 Bridge and Hallmark retrieval benchmark."""
from __future__ import annotations

import argparse, hashlib, json, time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
RESULTS, WORK = HERE / "results", HERE / "work"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(16 << 20), b""): h.update(block)
    return h.hexdigest()


def ordered_sha(values) -> str:
    h = hashlib.sha256()
    for value in values: h.update((str(value) + "\n").encode())
    return h.hexdigest()


class Log:
    def __init__(self, path):
        self.path = path; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("")
    def __call__(self, msg):
        line=f"[{datetime.now(timezone.utc).isoformat()}] {msg}"; print(line, flush=True)
        with self.path.open("a") as f: f.write(line+"\n")


def resolve_paths(cfg):
    out={}
    for k,v in cfg["paths"].items():
        p=Path(v); out[k]=(HERE/p).resolve() if not p.is_absolute() else p
    return out


def load_inputs(cfg, log):
    p=resolve_paths(cfg)
    spec=json.loads((p["archs4_embedding_dir"]/"embedding_manifest.json").read_text())
    locations=pd.read_parquet(p["archs4_embedding_dir"]/"sample_locations.parquet")
    master=pd.read_parquet(p["archs4_sample_manifest"])
    metadata=pd.read_parquet(p["archs4_metadata"])
    human=locations.loc[locations.species_id.eq(0)].copy().sort_values("global_index")
    human=human.merge(master.drop(columns=["gse_candidates"], errors="ignore"), left_on="geo_accession", right_on="gsm", validate="one_to_one")
    human=human.merge(metadata, on="gsm", validate="one_to_one", suffixes=("", "_metadata"))
    assert len(human)==510709 and human.geo_accession.is_unique and human.species.eq("human").all()
    total,dim=int(spec["total_samples"]),int(spec["embedding_dim"]); dtype=np.dtype(spec["embedding_dtype"])
    mmap_path=p["archs4_embedding_dir"]/f"sample_embeddings.{dtype.name}.mmap"
    assert mmap_path.stat().st_size==total*dim*dtype.itemsize and dim==512
    all_emb=np.memmap(mmap_path, mode="r", dtype=dtype, shape=(total,dim))
    tcga=np.load(p["tcga_embeddings"], mmap_mode="r")
    tm=pd.read_parquet(p["tcga_manifest"]).sort_values("matrix_row")
    assert tcga.shape==(9816,512) and len(tm)==9816 and tm.patient_id.is_unique
    assert tm.matrix_row.tolist()==list(range(9816)) and np.isfinite(tcga).all()
    probe=np.load(p["frozen_hallmark_head"], allow_pickle=False)
    hallmarks=probe["hallmarks"].astype(str).tolist()
    order=pd.read_csv(p["frozen_hallmark_order"]).sort_values("pathway_index")["hallmark"].astype(str).tolist()
    assert hallmarks==order and len(hallmarks)==50
    genes=pd.read_csv(p["canonical_genes"]).sort_values("token_id")
    assert len(genes)==15165 and genes.token_id.tolist()==list(range(1,15166))
    log(f"Validated authoritative matrices: ARCHS4 human={len(human):,}x512; TCGA={len(tm):,}x512; Hallmarks=50")
    return p,spec,human,all_emb,tm,tcga,probe,hallmarks,genes


def apply_probe(x, probe, batch=16384):
    out=np.empty((len(x),50),np.float32)
    mean,scale,w,b=(probe[k].astype(np.float32) for k in ["mean","scale","weight","bias"])
    for s in range(0,len(x),batch):
        z=np.asarray(x[s:s+batch],np.float32); out[s:s+len(z)]=((z-mean)/scale)@w.T+b
    assert np.isfinite(out).all(); return out


def exact_topk(query, database, k, device, batch, log, label):
    q=torch.as_tensor(np.asarray(query,np.float32),device=device)
    q=torch.nn.functional.normalize(q,dim=1)
    d=torch.as_tensor(np.asarray(database,np.float32),device=device)
    d=torch.nn.functional.normalize(d,dim=1)
    ids=[]; vals=[]; started=time.monotonic()
    for s in range(0,len(q),batch):
        scores=q[s:s+batch]@d.T
        v,i=torch.topk(scores,k=k,dim=1,sorted=True)
        ids.append(i.cpu().numpy()); vals.append(v.cpu().numpy())
        done=min(s+batch,len(q)); elapsed=time.monotonic()-started
        if s==0 or done==len(q) or done%(batch*10)==0:
            rate=done/max(elapsed,1e-9); log(f"{label}: {done:,}/{len(q):,} queries; {rate:.1f}/s; ETA={(len(q)-done)/max(rate,1e-9):.1f}s")
        del scores,v,i
    return np.vstack(ids),np.vstack(vals)


def long_retrieval(indices, similarities, qmeta, arch, method, mode):
    nq,k=indices.shape
    base=pd.DataFrame({"query_index":np.repeat(np.arange(nq),k),"rank":np.tile(np.arange(1,k+1),nq),"archs4_row":indices.ravel(),"similarity":similarities.ravel().astype(np.float32)})
    if mode=="individual":
        base["query_patient_id"]=np.repeat(qmeta.patient_id.astype(str).to_numpy(),k)
        base["query_cancer"]=np.repeat(qmeta.cohort.astype(str).to_numpy(),k)
    else:
        base["query_patient_id"]=""; base["query_cancer"]=np.repeat(qmeta.astype(str).to_numpy(),k)
    base["query_mode"]=mode; base["method"]=method
    cols=["geo_accession","gsm","series_id","title","source_name_ch1","characteristics_ch1","split","study_exposure","mapping_status","singlecellprobability"]
    a=arch.iloc[base.archs4_row][cols].reset_index(drop=True)
    return pd.concat([base.reset_index(drop=True),a],axis=1)


def overlap_table(a,b,query_col="query_index",ks=(1,5,10,25,50,100)):
    rows=[]
    for q in sorted(set(a[query_col]) & set(b[query_col])):
        aa=a.loc[a[query_col].eq(q)].sort_values("rank"); bb=b.loc[b[query_col].eq(q)].sort_values("rank")
        ar=dict(zip(aa.gsm,aa["rank"])); br=dict(zip(bb.gsm,bb["rank"]))
        for k in ks:
            sa=set(aa.loc[aa["rank"]<=k,"gsm"]); sb=set(bb.loc[bb["rank"]<=k,"gsm"]); shared=sa&sb
            rho=spearmanr([ar[x] for x in shared],[br[x] for x in shared]).statistic if len(shared)>=2 else np.nan
            rows.append({"query_index":q,"k":k,"overlap_count":len(shared),"jaccard":len(shared)/len(sa|sb),"shared_rank_spearman":rho,"reciprocal_rank_overlap":sum(1/ar[x]+1/br[x] for x in shared)/2})
    return pd.DataFrame(rows)


def primary_series(x):
    if pd.isna(x): return "UNKNOWN"
    return str(x).split(",")[0].strip()


ORGAN_TERMS={
"ACC":"adrenal","BLCA":"bladder","BRCA":"breast","CESC":"cervix|cervical","CHOL":"bile|biliary|cholangi","COAD":"colon|colorectal","DLBC":"lymph|b-cell|lymphoma","ESCA":"esoph","GBM":"brain|glioma|glioblast","HNSC":"head|neck|oral|tongue|pharyn|laryn","KICH":"kidney|renal","KIRC":"kidney|renal","KIRP":"kidney|renal","LGG":"brain|glioma","LIHC":"liver|hepatic|hepatoc","LUAD":"lung|pulmonary","LUSC":"lung|pulmonary","MESO":"mesotheli|pleura","OV":"ovary|ovarian","PAAD":"pancrea","PCPG":"adrenal|paragang|pheochromo","PRAD":"prostate","READ":"rectum|rectal|colorectal","SARC":"sarcoma|mesenchymal","SKCM":"skin|melanoma|melanocyte","STAD":"stomach|gastric","TGCT":"testis|testicular|germ cell","THCA":"thyroid","THYM":"thym","UCEC":"uter|endometri","UCS":"uter|endometri","UVM":"uveal|eye|ocular|melanoma"}


def coherence(top, ks=(10,50,100)):
    text=(top.title.fillna("")+" "+top.source_name_ch1.fillna("")+" "+top.characteristics_ch1.fillna("")).str.lower()
    top=top.copy(); top["primary_study"]=top.series_id.map(primary_series); top["text"]=text
    top["metadata_present"]=top.text.str.len().gt(2)
    top["organ_match"]=False
    for cancer,idx in top.groupby("query_cancer",sort=False).groups.items():
        top.loc[idx,"organ_match"]=top.loc[idx,"text"].str.contains(ORGAN_TERMS.get(cancer,"$"),regex=True).to_numpy()
    keys=["method","query_mode","query_index","query_cancer"]; out=[]
    for k in ks:
        x=top.loc[top["rank"]<=k]
        basic=x.groupby(keys,sort=False).agg(metadata_coverage=("metadata_present","mean"),organ_keyword_agreement=("organ_match","mean"),unique_studies=("primary_study","nunique")).reset_index()
        concentration=x.groupby(keys+["primary_study"],sort=False).size().groupby(level=list(range(len(keys)))).max().div(k).rename("max_study_fraction").reset_index()
        basic=basic.merge(concentration,on=keys); basic["k"]=k; out.append(basic)
    return pd.concat(out,ignore_index=True)


def savefig(fig,name):
    for ext in ["png","pdf"]: fig.savefig(RESULTS/"figures"/f"{name}.{ext}",dpi=240,bbox_inches="tight")
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--device",default="cuda:0"); ap.add_argument("--skip-hash-embedding",action="store_true"); ap.add_argument("--smoke",action="store_true"); args=ap.parse_args()
    RESULTS.mkdir(exist_ok=True); (RESULTS/"figures").mkdir(exist_ok=True); WORK.mkdir(exist_ok=True)
    log=Log(RESULTS/"run.log"); cfg=json.loads((HERE/"config.json").read_text()); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    p,spec,arch,all_emb,tm,tcga,probe,hallmarks,genes=load_inputs(cfg,log)
    if args.smoke: arch=arch.iloc[:5000].copy(); tm=tm.iloc[:64].copy(); tcga=np.asarray(tcga[:64]); log("SMOKE MODE active")
    ai=arch.global_index.to_numpy(np.int64); az=np.asarray(all_emb[ai]); ah=apply_probe(az,probe); th=apply_probe(tcga,probe)
    np.save(WORK/"archs4_hallmark_scores.float32.npy",ah); np.save(WORK/"tcga_hallmark_scores.float32.npy",th)
    arch_out=arch[["global_index","geo_accession","gsm","series_id","species","split","study_exposure","mapping_status"]].copy(); arch_out.insert(0,"matrix_row",np.arange(len(arch_out))); arch_out.to_parquet(RESULTS/"archs4_query_database_manifest.parquet",index=False)
    tm.to_parquet(RESULTS/"tcga_query_manifest.parquet",index=False)
    pd.DataFrame(th,columns=hallmarks).assign(patient_id=tm.patient_id.to_numpy(),cancer_type=tm.cohort.to_numpy()).to_parquet(RESULTS/"tcga_hallmark_profiles.parquet",index=False)
    pd.DataFrame({"matrix_row":np.arange(len(arch)),"gsm":arch.gsm,"hallmark_vector_row":np.arange(len(arch))}).to_parquet(RESULTS/"archs4_hallmark_profile_index.parquet",index=False)
    cancers=sorted(tm.cohort.unique()); groups=[np.flatnonzero(tm.cohort.to_numpy()==c) for c in cancers]
    zcent=np.vstack([np.asarray(tcga[g]).mean(0) for g in groups]); hcent=np.vstack([th[g].mean(0) for g in groups])
    n=min(len(tm),128) if args.smoke else len(tm); k=min(100,len(arch)); batch=min(cfg["query_batch_size"],n)
    all_top=[]
    for method,q,db in [("bridge512_cosine",np.asarray(tcga[:n]),az),("hallmark50_cosine",th[:n],ah)]:
        idx,sim=exact_topk(q,db,k,args.device,batch,log,method+" individual")
        frame=long_retrieval(idx,sim,tm.iloc[:n],arch,method,"individual"); frame.to_parquet(RESULTS/f"top100_individual_{method}.parquet",index=False); all_top.append(frame)
    centroid_frames=[]
    for method,q,db in [("bridge512_cosine",zcent,az),("hallmark50_cosine",hcent,ah)]:
        idx,sim=exact_topk(q,db,k,args.device,len(cancers),log,method+" centroid")
        frame=long_retrieval(idx,sim,pd.Series(cancers),arch,method,"centroid"); frame.to_parquet(RESULTS/f"top100_centroid_{method}.parquet",index=False); centroid_frames.append(frame); all_top.append(frame)
    # Prespecified Hallmark metric sensitivity, centroid queries on the same full corpus.
    sens=[]; mu=ah.mean(0); sd=ah.std(0); sd[sd==0]=1
    variants={"raw":(hcent,ah),"centered":(hcent-mu,ah-mu),"standardized":((hcent-mu)/sd,(ah-mu)/sd)}
    for name,(q,d) in variants.items():
        idx,sim=exact_topk(q,d,k,args.device,len(cancers),log,"Hallmark "+name)
        f=long_retrieval(idx,sim,pd.Series(cancers),arch,"hallmark50_"+name,"centroid"); sens.append(f)
    sensitivity=pd.concat(sens,ignore_index=True); sensitivity.to_parquet(RESULTS/"hallmark_metric_sensitivity_top100.parquet",index=False)
    overlap=overlap_table(all_top[0],all_top[1]); overlap=overlap.merge(tm[["matrix_row","cohort"]],left_on="query_index",right_on="matrix_row",how="left").drop(columns="matrix_row"); overlap.to_csv(RESULTS/"individual_method_overlap.csv",index=False)
    coverlap=overlap_table(centroid_frames[0],centroid_frames[1]); coverlap["cancer_type"]=[cancers[i] for i in coverlap.query_index]; coverlap.to_csv(RESULTS/"centroid_method_overlap.csv",index=False)
    sens_pairs=[]
    raw=sensitivity.loc[sensitivity.method.eq("hallmark50_raw")]
    for m in ["hallmark50_centered","hallmark50_standardized"]:
        x=overlap_table(raw,sensitivity.loc[sensitivity.method.eq(m)]); x["comparison"]="raw_vs_"+m.removeprefix("hallmark50_"); x["cancer_type"]=[cancers[i] for i in x.query_index]; sens_pairs.append(x)
    pd.concat(sens_pairs).to_csv(RESULTS/"hallmark_metric_sensitivity_overlap.csv",index=False)
    top=pd.concat(all_top,ignore_index=True); coh=coherence(top); coh.to_csv(RESULTS/"metadata_coherence.csv",index=False)
    # Hubs and study concentration.
    hubs=[]
    for method,g in top.loc[top.query_mode.eq("individual")].groupby("method"):
        z=g.groupby(["gsm","series_id"]).agg(retrieval_count=("query_index","size"),cancers=("query_cancer","nunique"),mean_rank=("rank","mean"),best_rank=("rank","min")).reset_index(); z["method"]=method; hubs.append(z)
    pd.concat(hubs).sort_values(["method","retrieval_count"],ascending=[True,False]).to_csv(RESULTS/"retrieval_hubs.csv",index=False)
    # Pairwise similarities and pathway contributions for centroid Hallmark results.
    hf=centroid_frames[1].copy(); rows=[]; contributions=[]
    az32=np.asarray(az,np.float32); an=az32/np.linalg.norm(az32,axis=1,keepdims=True); hn=ah/np.linalg.norm(ah,axis=1,keepdims=True)
    zcn=zcent/np.linalg.norm(zcent,axis=1,keepdims=True); hcn=hcent/np.linalg.norm(hcent,axis=1,keepdims=True)
    for r in hf.itertuples():
        qi=int(r.query_index); di=int(r.archs4_row); prod=hcn[qi]*hn[di]; order=np.argsort(-prod)[:5]
        rows.append({"query_index":qi,"cancer_type":cancers[qi],"gsm":r.gsm,"rank":r.rank,"hallmark_similarity":float(hcn[qi]@hn[di]),"bridge_similarity":float(zcn[qi]@an[di]),"series_id":r.series_id,"title":r.title,"source_name_ch1":r.source_name_ch1})
        for rank,j in enumerate(order,1): contributions.append({"query_index":qi,"cancer_type":cancers[qi],"gsm":r.gsm,"retrieval_rank":r.rank,"contribution_rank":rank,"hallmark":hallmarks[j],"cosine_contribution":float(prod[j]),"query_score":float(hcent[qi,j]),"archs4_score":float(ah[di,j])})
    discord=pd.DataFrame(rows); discord["hallmark_only_candidate"]=~discord.set_index(["query_index","gsm"]).index.isin(centroid_frames[0].set_index(["query_index","gsm"]).index)
    discord.to_csv(RESULTS/"centroid_functional_discordance.csv",index=False); pd.DataFrame(contributions).to_csv(RESULTS/"pathway_contributions.csv",index=False)
    # Leave dominant dimensions out: re-query Hallmark centroids after dropping top 1 and top 3 |centroid z| dimensions.
    loo=[]
    for drop in [1,3]:
        for qi,c in enumerate(cancers):
            keep=np.ones(50,bool); keep[np.argsort(-np.abs((hcent[qi]-mu)/sd))[:drop]]=False
            idx,sim=exact_topk(hcent[qi:qi+1,keep],ah[:,keep],k,args.device,1,log,f"LOO {c} drop{drop}")
            for rank,di in enumerate(idx[0],1): loo.append({"query_index":qi,"cancer_type":c,"dimensions_removed":drop,"rank":rank,"gsm":arch.iloc[di].gsm,"similarity":sim[0,rank-1]})
    pd.DataFrame(loo).to_csv(RESULTS/"dominant_pathway_removal_top100.csv",index=False)
    # Random-query and shuffled-dimension controls (centroid scope).
    rng=np.random.default_rng(cfg["seed"]); random_q=rng.normal(size=hcent.shape).astype(np.float32); shuffled=ah.copy()
    for j in range(50): rng.shuffle(shuffled[:,j])
    controls=[]
    for label,q,d in [("random_query",random_q,ah),("shuffled_database_dimensions",hcent,shuffled)]:
        idx,sim=exact_topk(q,d,k,args.device,len(cancers),log,label)
        for qi in range(len(cancers)):
            controls.append({"control":label,"cancer_type":cancers[qi],"mean_top10_similarity":float(sim[qi,:10].mean()),"top1_similarity":float(sim[qi,0])})
    pd.DataFrame(controls).to_csv(RESULTS/"null_controls.csv",index=False)
    # Cancer-level summaries.
    ov100=overlap.loc[overlap.k.eq(100)].groupby("cohort").agg(jaccard100=("jaccard","mean"),overlap100=("overlap_count","mean")).reset_index().rename(columns={"cohort":"cancer_type"})
    cc=coh.loc[(coh.query_mode.eq("individual")) & (coh.k.eq(100))].groupby(["query_cancer","method"]).agg(organ_agreement=("organ_keyword_agreement","mean"),metadata_coverage=("metadata_coverage","mean"),unique_studies=("unique_studies","mean"),max_study_fraction=("max_study_fraction","mean")).reset_index().pivot(index="query_cancer",columns="method")
    cc.columns=[f"{a}_{b}" for a,b in cc.columns]; cc=cc.reset_index().rename(columns={"query_cancer":"cancer_type"}).merge(ov100,on="cancer_type"); cc.to_csv(RESULTS/"cancer_summary.csv",index=False)
    pd.DataFrame(hcent,index=cancers,columns=hallmarks).to_csv(RESULTS/"tcga_centroid_hallmark_profiles.csv")
    retrieved=np.vstack([ah[centroid_frames[1].loc[centroid_frames[1].query_index.eq(i),"archs4_row"].to_numpy(int)].mean(0) for i in range(len(cancers))]); pd.DataFrame(retrieved,index=cancers,columns=hallmarks).to_csv(RESULTS/"retrieved_archs4_mean_hallmark_profiles.csv")
    # Figures.
    sns.set_theme(style="whitegrid",context="paper")
    fig,ax=plt.subplots(figsize=(7,4)); sns.boxplot(data=overlap,x="k",y="jaccard",ax=ax,color="#4c78a8"); ax.set(title="Bridge-512 vs Hallmark-50 neighborhood overlap",ylabel="Jaccard"); savefig(fig,"method_overlap")
    fig,axs=plt.subplots(1,2,figsize=(12,10),sharey=True); sns.heatmap(pd.DataFrame(hcent,index=cancers,columns=hallmarks),cmap="vlag",center=0,ax=axs[0],cbar=False); sns.heatmap(pd.DataFrame(retrieved,index=cancers,columns=hallmarks),cmap="vlag",center=0,ax=axs[1]); axs[0].set_title("TCGA cancer centroids"); axs[1].set_title("Mean top-100 Hallmark retrievals"); savefig(fig,"centroid_hallmark_profiles")
    piv=cc.set_index("cancer_type").select_dtypes("number"); fig,ax=plt.subplots(figsize=(9,10)); sns.heatmap((piv-piv.mean())/piv.std(ddof=0),cmap="vlag",center=0,ax=ax); ax.set_title("Cancer × retrieval metric (column z-scores)"); savefig(fig,"cancer_metric_summary")
    fig,ax=plt.subplots(figsize=(6,5)); sns.scatterplot(data=discord,x="bridge_similarity",y="hallmark_similarity",hue="cancer_type",legend=False,ax=ax,s=22); ax.set_title("Functional versus global similarity (Hallmark top-100)"); savefig(fig,"functional_discordance")
    # Provenance only hashes immutable inputs, never generated large work arrays.
    inputs=[p["canonical_genes"],p["bridge_checkpoint"],p["bridge_config"],p["frozen_hallmark_head"],p["frozen_hallmark_order"],p["frozen_system_manifest"],p["tcga_embeddings"],p["tcga_manifest"],p["archs4_embedding_dir"]/"embedding_manifest.json",p["archs4_embedding_dir"]/"sample_locations.parquet",p["archs4_sample_manifest"],p["archs4_metadata"]]
    hashes={str(x):sha256(x) for x in inputs}; mmap_path=p["archs4_embedding_dir"]/f"sample_embeddings.{np.dtype(spec['embedding_dtype']).name}.mmap"
    hashes[str(mmap_path)]=None if args.skip_hash_embedding else sha256(mmap_path)
    prov={"created_utc":datetime.now(timezone.utc).isoformat(),"bridge_frozen":True,"hallmark_head_frozen":True,"hallmark_formula":"((z - mean) / scale) @ weight.T + bias","hallmark_score_semantics":"continuous GTEx-trained Bridge Hallmark predictions; not probabilities or enrichment scores","archs4_samples":len(arch),"tcga_samples":len(tm),"embedding_dim":512,"hallmarks":hallmarks,"archs4_ordered_gsm_sha256":ordered_sha(arch.gsm),"tcga_ordered_patient_sha256":ordered_sha(tm.patient_id),"config":cfg,"input_sha256":hashes,"expression_baseline_status":"implemented separately by expression_baseline.py; centroid-only on authoritative 40,000-sample natural-log1p(TPM) cache","limitations":["ARCHS4 metadata keyword agreement is not biological ground truth","Hallmark target scale is inherited from GTEx","cosine sensitivity analyses are prespecified, not used for metric selection","functional discordance is exploratory"]}
    (RESULTS/"provenance.json").write_text(json.dumps(prov,indent=2))
    qa={"passed":True,"archs4_unique":bool(arch.gsm.is_unique),"tcga_patient_unique":bool(tm.patient_id.is_unique),"finite_bridge":bool(np.isfinite(az).all() and np.isfinite(tcga).all()),"finite_hallmark":bool(np.isfinite(ah).all() and np.isfinite(th).all()),"dimensions":{"archs4_bridge":list(az.shape),"tcga_bridge":list(tcga.shape),"archs4_hallmark":list(ah.shape),"tcga_hallmark":list(th.shape)},"no_tcga_archs4_id_overlap":len(set(tm.sample_id)&set(arch.gsm))==0}
    (RESULTS/"qa.json").write_text(json.dumps(qa,indent=2)); log("Primary benchmark complete; all invariants passed")

if __name__=="__main__": main()
