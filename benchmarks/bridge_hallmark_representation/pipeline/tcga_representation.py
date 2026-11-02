#!/usr/bin/env python3
"""Authoritative TCGA preparation, frozen inference, and final-layer replication."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from common import CONFIG, REPO, RESULTS, WORK, resolve, read_vocab, read_gmt, sha256, save_json, assert_matrix
from sample_space import ssgsea_rank_score, geometry, categorical_association, hallmark_associations

sys.path.insert(0,str(REPO/"src"))
from fm_embed.model import load_expression_performer

def prepare():
    meta=pd.read_parquet(resolve(CONFIG["tcga_manifest"])).copy().sort_values("matrix_row").reset_index(drop=True)
    X=np.load(resolve(CONFIG["tcga_bridge_expression"]),mmap_mode="r")
    assert_matrix(X,len(meta),15165,"TCGA log1p TPM")
    assert meta.patient_id.is_unique and meta.sample_id.is_unique
    assert meta.sample_type.eq("Primary Tumor").all()
    assert np.array_equal(meta.matrix_row.to_numpy(),np.arange(len(meta)))
    meta["organ_group"]=meta.cohort.map({"LUAD":"lung","LUSC":"lung","KIRC":"kidney","KIRP":"kidney","KICH":"kidney","BRCA":"breast","THCA":"thyroid"}).fillna(meta.cohort)
    meta.to_parquet(RESULTS/"tcga_sample_manifest.parquet",index=False)
    assets={k:{"path":str(resolve(CONFIG[k])),"sha256":sha256(resolve(CONFIG[k]))} for k in
            ["checkpoint","model_config","canonical_genes","tcga_manifest","tcga_bridge_expression","tcga_full_expression","tcga_full_h5"]}
    cfg=json.loads(resolve(CONFIG["model_config"]).read_text())
    save_json(RESULTS/"tcga_representation_provenance.json",{"samples":len(meta),"patients":meta.patient_id.nunique(),
      "cancers":meta.cohort.nunique(),"eligibility":"Primary Tumor; sorted sample UUID; one aliquot per patient",
      "preprocessing":"raw counts -> GENCODE v49 exon-length TPM -> natural log1p; canonical 15,165 order",
      "pooling":"unweighted arithmetic mean over final-layer 15,165 contextual gene states","bridge_frozen":True,
      "checkpoint_run_id":cfg["run_id"],"architecture":cfg["architecture"],"assets":assets})
    print(f"prepared {len(meta)} unique TCGA patients",flush=True)

def embed(args):
    X=np.load(resolve(CONFIG["tcga_bridge_expression"]),mmap_mode="r"); end=min(args.end or len(X),len(X)); start=args.start
    model,device=load_expression_performer(resolve(CONFIG["checkpoint"]),resolve(CONFIG["model_config"]),15165,args.device)
    out=WORK/f"tcga_embeddings_{start}_{end}.npy"; mm=np.lib.format.open_memmap(out,mode="w+",dtype=np.float32,shape=(end-start,512)); t0=time.time()
    for left in range(start,end,args.batch_size):
        right=min(left+args.batch_size,end); xb=torch.as_tensor(np.array(X[left:right],copy=True),device=device)
        with torch.inference_mode(),torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"):
            mm[left-start:right-start]=model.encode(xb,normalize=False).float().cpu().numpy()
        mm.flush(); done=right-start; elapsed=time.time()-t0; rate=done/elapsed
        if done==right-start or done%100<args.batch_size: print(f"[embed {start}:{end}] {done}/{end-start} elapsed={elapsed:.1f}s rate={rate:.2f}/s eta={(end-right)/max(rate,1e-9):.1f}s",flush=True)
    save_json(out.with_suffix(".json"),{"start":start,"end":end,"shape":list(mm.shape),"sample_ids":str(RESULTS/"tcga_sample_manifest.parquet"),"checkpoint_sha256":sha256(resolve(CONFIG["checkpoint"]))})

def combine(args):
    meta=pd.read_parquet(RESULTS/"tcga_sample_manifest.parquet"); parts=[]
    for p in args.parts: parts.append(np.load(p))
    Z=np.concatenate(parts); assert_matrix(Z,len(meta),512,"combined TCGA embedding")
    np.save(WORK/"tcga_embeddings.npy",Z.astype(np.float32))
    save_json(RESULTS/"tcga_embedding_qc.json",{"shape":list(Z.shape),"finite":bool(np.isfinite(Z).all()),"duplicate_samples":int(meta.sample_id.duplicated().sum()),"duplicate_patients":int(meta.patient_id.duplicated().sum()),"embedding_sha256":sha256(WORK/"tcga_embeddings.npy")})

def ci_r(r,n):
    if not np.isfinite(r) or n<4:return (np.nan,np.nan)
    z=np.arctanh(np.clip(r,-.999999,.999999)); d=1.96/np.sqrt(n-3); return tuple(np.tanh([z-d,z+d]))

def nested(Z,H,names,groups):
    outer=GroupKFold(n_splits=CONFIG["outer_folds"]); rows=[]; coefficients=[]; rng=np.random.default_rng(CONFIG["seed"])
    for fold,(tr,te) in enumerate(outer.split(Z,groups=groups)):
      sx=StandardScaler().fit(Z[tr]); a,b=sx.transform(Z[tr]),sx.transform(Z[te]); utr=np.unique(groups[tr]); iv=np.isin(groups[tr],utr[fold::5])
      for k,name in enumerate(names):
       y,yte=H[tr,k],H[te,k]; corr=np.abs([stats.pearsonr(a[:,j],y).statistic for j in range(512)]); order=np.nan_to_num(corr).argsort()[::-1]
       for nf in CONFIG["dimension_subsets"]:
        ids=order[:nf]; candidates=[]
        for alpha in CONFIG["ridge_alphas"]:
         m=Ridge(alpha=alpha).fit(a[~iv][:,ids],y[~iv]); candidates.append((np.mean((m.predict(a[iv][:,ids])-y[iv])**2),alpha))
        alpha=min(candidates)[1]; m=Ridge(alpha=alpha).fit(a[:,ids],y); pred=m.predict(b[:,ids]); pr=stats.pearsonr(pred,yte).statistic; lo,hi=ci_r(pr,len(te))
        rows.append({"dataset":"tcga","fold":fold,"hallmark":name,"features":nf,"alpha":alpha,"pearson":pr,"pearson_ci_low":lo,"pearson_ci_high":hi,
          "spearman":stats.spearmanr(pred,yte).statistic,"r2":r2_score(yte,pred),"mae":mean_absolute_error(yte,pred),"n_test":len(te),"control":"observed"})
       null=Ridge(alpha=10).fit(a,y[rng.permutation(len(y))]).predict(b)
       rows.append({"dataset":"tcga","fold":fold,"hallmark":name,"features":512,"alpha":10,"pearson":stats.pearsonr(null,yte).statistic,"spearman":stats.spearmanr(null,yte).statistic,"r2":r2_score(yte,null),"mae":mean_absolute_error(yte,null),"n_test":len(te),"control":"permuted_label"})
    pd.DataFrame(rows).to_csv(RESULTS/"tcga_nested_hallmark_probe_metrics.csv",index=False)

def analyze():
    meta=pd.read_parquet(RESULTS/"tcga_sample_manifest.parquet"); X=np.load(resolve(CONFIG["tcga_bridge_expression"]),mmap_mode="r"); Z=np.load(WORK/"tcga_embeddings.npy")
    genes,sets=read_vocab(),read_gmt(); H,names=ssgsea_rank_score(X,genes,sets); np.save(WORK/"tcga_hallmark_targets.npy",H)
    pd.DataFrame(H,columns=names).assign(sample_id=meta.sample_id).to_parquet(RESULTS/"tcga_expression_hallmark_scores.parquet",index=False)
    geometry(Z,"tcga"); categorical_association(Z,meta.cohort,"tcga","cancer_type"); categorical_association(Z,meta.organ_group,"tcga","organ_group"); hallmark_associations(Z,H,names,"tcga")
    nested(Z,H,names,meta.patient_id.to_numpy()); print("TCGA replication complete",flush=True)

def main():
 p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="cmd",required=True); sub.add_parser("prepare")
 e=sub.add_parser("embed");e.add_argument("--start",type=int,required=True);e.add_argument("--end",type=int);e.add_argument("--batch-size",type=int,default=4);e.add_argument("--device",default="cuda:0")
 c=sub.add_parser("combine");c.add_argument("parts",nargs="+");sub.add_parser("analyze")
 a=p.parse_args(); {"prepare":lambda _:prepare(),"embed":embed,"combine":combine,"analyze":lambda _:analyze()}[a.cmd](a)
if __name__=="__main__":main()
