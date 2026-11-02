#!/usr/bin/env python3
"""Parts I-II: geometry, label associations, ssGSEA-like targets, and nested probes."""
from __future__ import annotations
import argparse, time
import numpy as np, pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from common import CONFIG, RESULTS, WORK, resolve, read_vocab, read_gmt, bh_fdr, assert_matrix, save_json

def ssgsea_rank_score(X, genes, sets):
    """Deterministic rank-mean z score: a documented GSVA-equivalent single-sample rank statistic."""
    index={g:i for i,g in enumerate(genes)}; names=list(sets); out=np.empty((len(X),len(names)),np.float32)
    for start in range(0,len(X),128):
        chunk=np.asarray(X[start:start+128]); ranks=stats.rankdata(chunk,axis=1)/chunk.shape[1]
        for k,name in enumerate(names):
            ids=[index[g] for g in sets[name] if g in index]
            out[start:start+len(chunk),k]=ranks[:,ids].mean(1)
    out=(out-out.mean(0))/np.maximum(out.std(0),1e-8)
    return out,names

def geometry(Z,dataset):
    centered=Z-Z.mean(0); cov=np.cov(centered,rowvar=False); eig=np.linalg.eigvalsh(cov)[::-1]
    ratio=eig/np.maximum(eig.sum(),1e-20); cumulative=np.cumsum(ratio)
    erank=float(np.exp(-np.sum(ratio[ratio>0]*np.log(ratio[ratio>0])))); pr=float(eig.sum()**2/np.sum(eig**2))
    dims=pd.DataFrame({"dimension":np.arange(512),"mean":Z.mean(0),"variance":Z.var(0,ddof=1)})
    dims.to_csv(RESULTS/f"{dataset}_dimension_summary.csv",index=False)
    pd.DataFrame({"component":np.arange(1,513),"eigenvalue":eig,"explained_variance_ratio":ratio,
                  "cumulative_explained_variance":cumulative}).to_csv(RESULTS/f"{dataset}_eigen_spectrum.csv",index=False)
    np.save(WORK/f"{dataset}_dimension_correlation.npy",np.corrcoef(Z,rowvar=False).astype(np.float32))
    thresholds={str(t):int(np.searchsorted(cumulative,t)+1) for t in [.5,.8,.9,.95,.99]}
    save_json(RESULTS/f"{dataset}_geometry_summary.json",{"samples":len(Z),"effective_rank_entropy":erank,
        "effective_rank_participation":pr,"pcs_for_variance":thresholds})
    fig,ax=plt.subplots(2,2,figsize=(12,9)); ax[0,0].plot(dims.dimension,dims.variance); ax[0,0].set(title="Dimension variance",xlabel="dimension")
    im=ax[0,1].imshow(np.load(WORK/f"{dataset}_dimension_correlation.npy"),cmap="coolwarm",vmin=-1,vmax=1,aspect="auto"); fig.colorbar(im,ax=ax[0,1]); ax[0,1].set_title("Dimension correlation")
    ax[1,0].semilogy(np.arange(1,513),np.maximum(eig,1e-12)); ax[1,0].set(title="Covariance spectrum",xlabel="component")
    ax[1,1].plot(np.arange(1,513),cumulative); ax[1,1].axhline(.9,color="k",ls="--"); ax[1,1].set(title="Cumulative variance",xlabel="component",ylim=(0,1.01))
    fig.tight_layout(); fig.savefig(RESULTS/f"{dataset}_geometry.png",dpi=180); plt.close(fig)

def categorical_association(Z,labels,dataset,label_name):
    labels=np.asarray(labels); valid=pd.notna(labels); groups=np.unique(labels[valid]); rows=[]
    for j in range(Z.shape[1]):
        vals=Z[valid,j]; overall=vals.mean(); between=sum(np.sum(labels[valid]==g)*(vals[labels[valid]==g].mean()-overall)**2 for g in groups)
        total=np.sum((vals-overall)**2); eta=between/total if total else 0
        samples=[vals[labels[valid]==g] for g in groups]
        stat,p=stats.kruskal(*samples) if all(len(x)>0 for x in samples) else (np.nan,np.nan)
        rows.append((j,eta,stat,p))
    out=pd.DataFrame(rows,columns=["dimension","eta_squared","kruskal_h","p_value"]); out["fdr"]=bh_fdr(out.p_value)
    out.sort_values("eta_squared",ascending=False).to_csv(RESULTS/f"{dataset}_{label_name}_dimension_associations.csv",index=False)

def hallmark_associations(Z,H,names,dataset):
    r=np.corrcoef(Z.T,H.T)[:512,512:]; n=len(Z); t=r*np.sqrt((n-2)/np.maximum(1-r*r,1e-12)); p=2*stats.t.sf(np.abs(t),n-2); q=bh_fdr(p)
    pd.DataFrame(r,index=[f"z_{i}" for i in range(512)],columns=names).to_csv(RESULTS/f"{dataset}_latent_hallmark_pearson.csv")
    pd.DataFrame(q,index=[f"z_{i}" for i in range(512)],columns=names).to_csv(RESULTS/f"{dataset}_latent_hallmark_fdr.csv")
    rows=[]
    for k,name in enumerate(names):
        for j in np.argsort(-np.abs(r[:,k]))[:10]: rows.append({"hallmark":name,"dimension":j,"pearson":r[j,k],"fdr":q[j,k]})
    pd.DataFrame(rows).to_csv(RESULTS/f"{dataset}_strongest_dimensions_per_hallmark.csv",index=False)
    plt.figure(figsize=(14,10)); plt.imshow(r.T,cmap="coolwarm",vmin=-1,vmax=1,aspect="auto"); plt.colorbar(label="Pearson r"); plt.yticks(np.arange(len(names)),names,fontsize=5); plt.xlabel("Bridge dimension"); plt.tight_layout(); plt.savefig(RESULTS/f"{dataset}_latent_hallmark_heatmap.png",dpi=180); plt.close()

def nested_probes(Z,H,names,groups,dataset):
    outer=GroupKFold(n_splits=min(CONFIG["outer_folds"],len(np.unique(groups)))); rows=[]
    rng=np.random.default_rng(CONFIG["seed"])
    for fold,(tr,te) in enumerate(outer.split(Z,groups=groups)):
      sx=StandardScaler().fit(Z[tr]); xtr,xte=sx.transform(Z[tr]),sx.transform(Z[te])
      for k,name in enumerate(names):
        ytr,yte=H[tr,k],H[te,k]; corr=np.abs(np.array([stats.pearsonr(xtr[:,j],ytr).statistic for j in range(512)])); order=np.nan_to_num(corr).argsort()[::-1]
        for nfeat in CONFIG["dimension_subsets"]:
          ids=order[:nfeat]; best=None
          # alpha selection only on outer training data using deterministic grouped inner validation
          utr=np.unique(groups[tr]); inner_test=np.isin(groups[tr],utr[fold::5]); arows=[]
          for alpha in CONFIG["ridge_alphas"]:
            m=Ridge(alpha=alpha).fit(xtr[~inner_test][:,ids],ytr[~inner_test]); arows.append((np.mean((m.predict(xtr[inner_test][:,ids])-ytr[inner_test])**2),alpha))
          alpha=min(arows)[1]; pred=Ridge(alpha=alpha).fit(xtr[:,ids],ytr).predict(xte[:,ids])
          rows.append({"dataset":dataset,"fold":fold,"hallmark":name,"features":nfeat,"alpha":alpha,"pearson":stats.pearsonr(pred,yte).statistic,
            "spearman":stats.spearmanr(pred,yte).statistic,"r2":r2_score(yte,pred),"mae":mean_absolute_error(yte,pred),"n_test":len(te),"control":"observed"})
        null=Ridge(alpha=1).fit(xtr,ytr[rng.permutation(len(ytr))]).predict(xte)
        rows.append({"dataset":dataset,"fold":fold,"hallmark":name,"features":512,"alpha":1,"pearson":stats.pearsonr(null,yte).statistic,
          "spearman":stats.spearmanr(null,yte).statistic,"r2":r2_score(yte,null),"mae":mean_absolute_error(yte,null),"n_test":len(te),"control":"permuted_label"})
    pd.DataFrame(rows).to_csv(RESULTS/f"{dataset}_nested_hallmark_probe_metrics.csv",index=False)

def main():
    RESULTS.mkdir(parents=True,exist_ok=True); WORK.mkdir(parents=True,exist_ok=True)
    genes,sets=read_vocab(),read_gmt(); X=np.load(resolve(CONFIG["gtex_expression"]),mmap_mode="r"); Z=np.load(resolve(CONFIG["gtex_embeddings"])); meta=pd.read_parquet(RESULTS/"gtex_sample_manifest.parquet")
    assert_matrix(Z,len(meta),512,"GTEx embeddings"); assert_matrix(X,len(meta),15165,"GTEx expression")
    geometry(Z,"gtex"); categorical_association(Z,meta.source_tissue,"gtex","detailed_tissue"); categorical_association(Z,meta.tissue,"gtex","broad_tissue")
    H,names=ssgsea_rank_score(X,genes,sets); np.save(WORK/"gtex_hallmark_targets.npy",H); pd.DataFrame(H,columns=names).assign(sample_id=meta.sample_id.values).to_parquet(RESULTS/"gtex_expression_hallmark_scores.parquet",index=False)
    hallmark_associations(Z,H,names,"gtex"); nested_probes(Z,H,names,meta.donor_id.to_numpy(),"gtex")
    print("GTEx Parts I-II complete",flush=True)
if __name__=="__main__": main()
