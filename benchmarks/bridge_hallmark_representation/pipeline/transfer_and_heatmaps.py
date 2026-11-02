#!/usr/bin/env python3
"""Freeze GTEx-only Hallmark probes, apply zero-shot to TCGA, and plot profiles."""
from __future__ import annotations
import json
import numpy as np,pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from common import CONFIG,RESULTS,WORK,resolve,save_json,sha256

def tune_alpha(X,Y,groups):
    cv=GroupKFold(n_splits=CONFIG["outer_folds"]); scores={a:[] for a in CONFIG["ridge_alphas"]}
    # multi-output validation makes selection global and prevents pathway-wise TCGA adaptation
    for tr,va in cv.split(X,groups=groups):
        s=StandardScaler().fit(X[tr]); a,b=s.transform(X[tr]),s.transform(X[va])
        for alpha in scores: scores[alpha].append(np.mean((Ridge(alpha=alpha).fit(a,Y[tr]).predict(b)-Y[va])**2))
    return min(scores,key=lambda x:np.mean(scores[x])),scores

def correlations(pred,Y,names,label):
    rows=[]
    for k,name in enumerate(names):
        p=stats.pearsonr(pred[:,k],Y[:,k]);s=stats.spearmanr(pred[:,k],Y[:,k]);
        z=np.arctanh(np.clip(p.statistic,-.999999,.999999));d=1.96/np.sqrt(len(Y)-3);lo,hi=np.tanh([z-d,z+d])
        rows.append({"hallmark":name,"evaluation":label,"pearson":p.statistic,"pearson_ci_low":lo,"pearson_ci_high":hi,"pearson_p":p.pvalue,"spearman":s.statistic,"spearman_p":s.pvalue})
    return pd.DataFrame(rows)

def fit():
    gm=pd.read_parquet(RESULTS/"gtex_sample_manifest.parquet"); tm=pd.read_parquet(RESULTS/"tcga_sample_manifest.parquet")
    gz=np.load(resolve(CONFIG["gtex_embeddings"]));tz=np.load(WORK/"tcga_embeddings.npy")
    gh=pd.read_parquet(RESULTS/"gtex_expression_hallmark_scores.parquet");th=pd.read_parquet(RESULTS/"tcga_expression_hallmark_scores.parquet")
    names=[c for c in gh if c!="sample_id"];GY=gh[names].to_numpy();TY=th[names].to_numpy()
    alpha,cv=tune_alpha(gz,GY,gm.donor_id.to_numpy()); scaler=StandardScaler().fit(gz); model=Ridge(alpha=alpha).fit(scaler.transform(gz),GY)
    gp=model.predict(scaler.transform(gz));tp=model.predict(scaler.transform(tz))
    np.savez(WORK/"frozen_gtex_hallmark_probe.npz",weight=model.coef_.astype(np.float32),bias=model.intercept_.astype(np.float32),mean=scaler.mean_.astype(np.float32),scale=scaler.scale_.astype(np.float32),hallmarks=np.array(names),alpha=alpha)
    pd.DataFrame(gp,columns=names).assign(sample_id=gm.sample_id).to_parquet(RESULTS/"gtex_frozen_probe_hallmark_profiles.parquet",index=False)
    pd.DataFrame(tp,columns=names).assign(sample_id=tm.sample_id).to_parquet(RESULTS/"tcga_zero_shot_hallmark_profiles.parquet",index=False)
    zero=correlations(tp,TY,names,"gtex_to_tcga_zero_shot")
    # Fit TCGA coefficients only for direction comparison, never for zero-shot predictions.
    talpha,_=tune_alpha(tz,TY,tm.patient_id.to_numpy());ts=StandardScaler().fit(tz);tmodel=Ridge(alpha=talpha).fit(ts.transform(tz),TY)
    cos=np.sum(model.coef_*tmodel.coef_,axis=1)/(np.linalg.norm(model.coef_,axis=1)*np.linalg.norm(tmodel.coef_,axis=1)+1e-12)
    zero["coefficient_cosine_gtex_tcga"]=cos
    within=pd.read_csv(RESULTS/"tcga_nested_hallmark_probe_metrics.csv");within=within[(within.control=="observed")&(within.features==512)].groupby("hallmark",as_index=False).agg(tcga_within_pearson=("pearson","mean"),tcga_within_pearson_sd=("pearson","std"),tcga_within_spearman=("spearman","mean"),tcga_within_r2=("r2","mean"));within["tcga_within_pearson_ci_low"]=within.tcga_within_pearson-2.776*within.tcga_within_pearson_sd/np.sqrt(5);within["tcga_within_pearson_ci_high"]=within.tcga_within_pearson+2.776*within.tcga_within_pearson_sd/np.sqrt(5)
    gwithin=pd.read_csv(RESULTS/"gtex_nested_hallmark_probe_metrics.csv");gwithin=gwithin[(gwithin.control=="observed")&(gwithin.features==512)].groupby("hallmark",as_index=False).agg(gtex_within_pearson=("pearson","mean"),gtex_within_pearson_sd=("pearson","std"),gtex_within_spearman=("spearman","mean"),gtex_within_r2=("r2","mean"));gwithin["gtex_within_pearson_ci_low"]=gwithin.gtex_within_pearson-2.776*gwithin.gtex_within_pearson_sd/np.sqrt(5);gwithin["gtex_within_pearson_ci_high"]=gwithin.gtex_within_pearson+2.776*gwithin.gtex_within_pearson_sd/np.sqrt(5)
    summary=zero.merge(gwithin,on="hallmark").merge(within,on="hallmark");summary.to_csv(RESULTS/"gtex_tcga_probe_transfer_summary.csv",index=False)
    save_json(RESULTS/"frozen_probe_manifest.json",{"training_dataset":"GTEx only","alpha":alpha,"alpha_cv_mse":cv,"features":512,"normalization":"GTEx training mean/SD","tcga_information_used_for_fit":False,"artifact_sha256":sha256(WORK/"frozen_gtex_hallmark_probe.npz")})

def heatmap(data,meta,row_label,out,individual=True):
    names=[c for c in data if c!="sample_id"]; merged=meta.merge(data,on="sample_id",validate="one_to_one")
    if individual:
        merged=merged.sort_values(row_label); matrix=merged[names].to_numpy(); labels=None; size=(15,12)
    else:
        grouped=merged.groupby(row_label)[names].mean().sort_index();matrix=grouped.to_numpy();labels=grouped.index;size=(15,max(6,len(grouped)*.22))
    fig,ax=plt.subplots(figsize=size);im=ax.imshow(matrix,aspect="auto",cmap="coolwarm",vmin=-2.5,vmax=2.5,interpolation="nearest");ax.set_xticks(range(50),names,rotation=90,fontsize=5)
    if labels is not None:ax.set_yticks(range(len(labels)),labels,fontsize=6)
    else:ax.set_yticks([])
    ax.set_ylabel(row_label);fig.colorbar(im,ax=ax,label="Frozen GTEx-probe score");fig.tight_layout();fig.savefig(RESULTS/out,dpi=180);plt.close(fig)

def plots():
    gm=pd.read_parquet(RESULTS/"gtex_sample_manifest.parquet");tm=pd.read_parquet(RESULTS/"tcga_sample_manifest.parquet");g=pd.read_parquet(RESULTS/"gtex_frozen_probe_hallmark_profiles.parquet");t=pd.read_parquet(RESULTS/"tcga_zero_shot_hallmark_profiles.parquet")
    heatmap(g,gm,"source_tissue","gtex_individual_hallmark_heatmap.png",True);heatmap(g,gm,"source_tissue","gtex_tissue_mean_hallmark_heatmap.png",False)
    heatmap(t,tm,"cohort","tcga_individual_hallmark_heatmap.png",True);heatmap(t,tm,"cohort","tcga_cancer_mean_hallmark_heatmap.png",False)
    s=pd.read_csv(RESULTS/"gtex_tcga_probe_transfer_summary.csv");fig,ax=plt.subplots(figsize=(8,7));ax.scatter(s.gtex_within_pearson,s.tcga_within_pearson,label="TCGA within",alpha=.8);ax.scatter(s.gtex_within_pearson,s.pearson,label="GTEx→TCGA zero-shot",alpha=.8);ax.axline((0,0),slope=1,color="k",ls="--");ax.set(xlabel="GTEx held-out Pearson",ylabel="TCGA Pearson");ax.legend();fig.tight_layout();fig.savefig(RESULTS/"gtex_tcga_hallmark_probe_performance.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,6));ax.scatter(s.tcga_within_pearson,s.pearson,c=s.coefficient_cosine_gtex_tcga,cmap="viridis");ax.axline((0,0),slope=1,color="k",ls="--");ax.set(xlabel="TCGA within-cohort Pearson",ylabel="GTEx→TCGA zero-shot Pearson");fig.colorbar(ax.collections[0],ax=ax,label="coefficient cosine");fig.tight_layout();fig.savefig(RESULTS/"gtex_to_tcga_zero_shot_performance.png",dpi=180);plt.close(fig)

if __name__=="__main__":fit();plots()
