#!/usr/bin/env python3
"""Study-disjoint biological-label generalization: raw expression, PCA, BridgeRNA."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import bootstrap
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit,
                                     StratifiedShuffleSplit)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
OUT = HERE / "results/task4_pca_vs_bridgerna_generalization"
FIG = OUT / "figures"
HALL = REPO / "benchmarks/cross_species_exercise_response"
WORK = HALL / "work/hallmark_readout"
SEED = 20260906
DIMS = [10, 25, 50, 100, 256, 512]

# Deliberately conservative exact source-name mappings. The output manifest keeps
# the original value and mapping rule, making label inclusion fully auditable.
TISSUES = {
    "blood": r"^(whole |peripheral )?blood$|^pbmcs?$",
    "liver": r"^liver( tissue)?$", "lung": r"^lung( tissue)?$",
    "skin": r"^skin( tissue)?$", "brain": r"^brain( tissue)?$|^cerebral cortex$|^cortex$",
    "breast": r"^(breast|mammary gland)( tissue)?$", "colon": r"^(colon|colonic mucosa)( tissue)?$",
    "kidney": r"^kidney( tissue)?$|^renal cortex$", "prostate": r"^prostate( tissue)?$",
    "pancreas": r"^pancreas( tissue)?$", "adipose": r"^(adipose|adipose tissue|subcutaneous adipose tissue)$",
    "ovary": r"^ovary( tissue)?$", "heart": r"^heart( tissue)?$|^left ventricle$",
    "skeletal_muscle": r"^(skeletal )?muscle( tissue)?$", "spleen": r"^spleen( tissue)?$",
    "testis": r"^(testis|testes)( tissue)?$",
}


def load_cohort():
    manifest = pd.read_parquet(HALL / "results/hallmark_readout/sample_manifest.parquet")
    meta = pd.read_parquet(REPO / "data/manifests/archs4_sample_metadata_v2.5.parquet")
    d = manifest.merge(meta, on="gsm", how="left", validate="one_to_one")
    source = d.source_name_ch1.fillna("").str.strip().str.lower()
    d["tissue"] = pd.NA; d["tissue_mapping_rule"] = pd.NA
    for label, pattern in TISSUES.items():
        hit = source.str.match(pattern)
        d.loc[hit, "tissue"] = label; d.loc[hit, "tissue_mapping_rule"] = pattern
    d = d.dropna(subset=["tissue", "gse"]).copy()
    # Require tissue labels to span enough independent studies for group evaluation.
    keep = d.groupby("tissue").agg(samples=("gsm", "size"), studies=("gse", "nunique"))
    keep = keep[(keep.samples >= 40) & (keep.studies >= 20)].index
    d = d[d.tissue.isin(keep)].sort_values("matrix_row").reset_index(drop=True)
    d.to_parquet(OUT / "sample_manifest.parquet", index=False)
    d.to_csv(OUT / "sample_manifest.csv", index=False)
    d.groupby("tissue").agg(samples=("gsm", "size"), studies=("gse", "nunique")).reset_index().to_csv(OUT / "cohort_summary.csv", index=False)
    Xall = np.memmap(WORK / "archs4_log1p_tpm.float32.mmap", mode="r", dtype="float32", shape=(40000, 15165))
    Zall = np.load(WORK / "archs4_bridgerna_embeddings.npy", mmap_mode="r")
    rows = d.matrix_row.to_numpy(int)
    return d, np.asarray(Xall[rows]), np.asarray(Zall[rows])


def scores(y, pred):
    return {"accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y, pred, average="weighted", zero_division=0)}


def classifier(Xtr, ytr, C=1.0):
    sc = StandardScaler().fit(Xtr)
    model = LogisticRegression(C=C, max_iter=600, solver="lbfgs", class_weight="balanced", n_jobs=-1, random_state=SEED).fit(sc.transform(Xtr), ytr)
    return sc, model


def fit_eval(Xtr, ytr, Xte, yte, C=1.0):
    sc, model = classifier(Xtr, ytr, C)
    pred = model.predict(sc.transform(Xte))
    return pred, scores(yte, pred)


def choose_pca(Xtr, ytr, groups):
    inner = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED)
    a, b = next(inner.split(Xtr, ytr, groups))
    maxd = min(512, len(a)-1, Xtr.shape[1])
    pca = PCA(n_components=maxd, svd_solver="randomized", random_state=SEED).fit(Xtr[a])
    A, B = pca.transform(Xtr[a]), pca.transform(Xtr[b]); rows=[]
    for dim in DIMS:
        if dim > maxd: continue
        for C in [.1, 1., 10.]:
            _, m = fit_eval(A[:, :dim], ytr[a], B[:, :dim], ytr[b], C)
            rows.append({"dimension": dim, "C": C, **m})
    q = pd.DataFrame(rows).sort_values(["macro_f1", "dimension"], ascending=[False, True]).iloc[0]
    return int(q.dimension), float(q.C), rows


def evaluate(d, X, Z):
    y = d.tissue.to_numpy(str); groups = d.gse.to_numpy(str); labels = sorted(np.unique(y))
    fold_rows=[]; class_rows=[]; curves=[]; split_rows=[]
    cv = GroupKFold(5)
    for fold, (tr, te) in enumerate(cv.split(X, y, groups)):
        split_rows.extend([{"scheme":"group5","fold":fold,"matrix_row":int(d.matrix_row.iloc[i]),"partition":"train" if i in set(tr) else "test"} for i in np.r_[tr,te]])
        dim, Cpca, inner = choose_pca(X[tr], y[tr], groups[tr])
        maxd=min(512,len(tr)-1,X.shape[1]); pca=PCA(n_components=maxd,svd_solver="randomized",random_state=SEED+fold).fit(X[tr]); Ptr=pca.transform(X[tr]);Pte=pca.transform(X[te])
        # Same C grid selected using training-only group split for raw and BridgeRNA.
        ia,iv=next(GroupShuffleSplit(1,test_size=.2,random_state=SEED).split(X[tr],y[tr],groups[tr]))
        selected={}
        for name,A in [("raw15165",X[tr]),("bridgerna512",Z[tr])]:
            cand=[]
            for C in [.1,1.,10.]:
                _,m=fit_eval(A[ia],y[tr][ia],A[iv],y[tr][iv],C);cand.append((m["macro_f1"],C))
            selected[name]=max(cand)[1]
        reps={"raw15165":(X[tr],X[te],15165,selected['raw15165']),
              "bridgerna512":(Z[tr],Z[te],512,selected['bridgerna512']),
              "pca15165_selected":(Ptr[:,:dim],Pte[:,:dim],dim,Cpca),
              "pca15165_512":(Ptr[:,:maxd],Pte[:,:maxd],maxd,Cpca)}
        for name,(A,B,nd,C) in reps.items():
            pred,m=fit_eval(A,y[tr],B,y[te],C);fold_rows.append({"scheme":"study_disjoint_group5","fold":fold,"representation":name,"dimensions":nd,"C":C,"train_samples":len(tr),"test_samples":len(te),**m})
            pr,re,f1,sup=precision_recall_fscore_support(y[te],pred,labels=labels,zero_division=0)
            class_rows.extend({"scheme":"study_disjoint_group5","fold":fold,"representation":name,"tissue":lab,"precision":pr[j],"recall":re[j],"f1":f1[j],"support":sup[j]} for j,lab in enumerate(labels))
            pd.DataFrame(confusion_matrix(y[te],pred,labels=labels),index=labels,columns=labels).to_csv(OUT/f"confusion_group5_fold{fold}_{name}.csv")
        for q in inner:
            q.update({"fold":fold,"scheme":"inner_validation"})
        curves.extend(inner)
    # Random split sanity check; PCA still fitted on training only.
    tr,te=next(StratifiedShuffleSplit(1,test_size=.2,random_state=SEED).split(X,y));maxd=min(512,len(tr)-1);pca=PCA(maxd,svd_solver='randomized',random_state=SEED).fit(X[tr]);Ptr=pca.transform(X[tr]);Pte=pca.transform(X[te])
    for name,A,B,nd in [("raw15165",X[tr],X[te],15165),("bridgerna512",Z[tr],Z[te],512),("pca15165_512",Ptr,Pte,maxd)]:
        pred,m=fit_eval(A,y[tr],B,y[te]);fold_rows.append({"scheme":"random_split","fold":0,"representation":name,"dimensions":nd,"C":1.,"train_samples":len(tr),"test_samples":len(te),**m})
    return pd.DataFrame(fold_rows),pd.DataFrame(class_rows),pd.DataFrame(curves),pd.DataFrame(split_rows)


def neighborhoods(d, X, Z):
    # PCA fitted on the defined study-disjoint ARCHS4 reference, not this label cohort.
    import joblib
    pca=joblib.load(WORK/'pca512.pkl'); P=pca.transform(X); rows=[]
    for name,A in [('raw15165',X),('pca15165_512',P),('bridgerna512',Z)]:
        nn=NearestNeighbors(n_neighbors=51,metric='cosine',n_jobs=-1).fit(A).kneighbors(return_distance=False)
        for k in [5,10,25,50]:
            tissue=np.mean([np.mean(d.tissue.to_numpy()[nn[i,:k]]==d.tissue.iloc[i]) for i in range(len(d))]);study=np.mean([np.mean(d.gse.to_numpy()[nn[i,:k]]==d.gse.iloc[i]) for i in range(len(d))])
            rows.append({'representation':name,'k':k,'tissue_purity':tissue,'study_purity':study,'tissue_to_study_ratio':tissue/max(study,1e-12)})
    return pd.DataFrame(rows)


def learning_curves(d,X,Z):
    y=d.tissue.to_numpy(str);g=d.gse.to_numpy(str);tr,te=next(GroupKFold(5).split(X,y,g));rng=np.random.default_rng(SEED);rows=[]
    import joblib
    # Reference PCA was fit without its held-out GSEs; use only samples assigned to its train split.
    pca=joblib.load(WORK/'pca512.pkl');P=pca.transform(X)
    studies=np.unique(g[tr])
    for frac in [.1,.25,.5,.75,1.]:
        chosen=rng.choice(studies,max(1,int(round(frac*len(studies)))),replace=False);sub=tr[np.isin(g[tr],chosen)]
        for name,A in [('raw15165',X),('pca15165_512',P),('bridgerna512',Z)]:
            pred,m=fit_eval(A[sub],y[sub],A[te],y[te]);rows.append({'training_study_fraction':frac,'training_studies':len(chosen),'training_samples':len(sub),'test_samples':len(te),'representation':name,**m})
    return pd.DataFrame(rows)


def bridge_pc_sensitivity(d,Z):
    y=d.tissue.to_numpy(str);g=d.gse.to_numpy(str);rows=[]
    tr,te=next(GroupKFold(5).split(Z,y,g));mu=Z[tr].mean(0);_,_,V=np.linalg.svd(Z[tr]-mu,full_matrices=False)
    for k in [0,1,2,5]:
        def remove(A):
            C=A-mu
            return C if k==0 else C-(C@V[:k].T)@V[:k]
        pred,m=fit_eval(remove(Z[tr]),y[tr],remove(Z[te]),y[te]);rows.append({'removed_bridge_pcs':k,**m})
    return pd.DataFrame(rows)


def summarize(folds, neighborhood, learning):
    g=folds.groupby(['scheme','representation']).agg(macro_f1_mean=('macro_f1','mean'),macro_f1_sd=('macro_f1','std'),balanced_accuracy_mean=('balanced_accuracy','mean'),accuracy_mean=('accuracy','mean'),folds=('fold','nunique')).reset_index()
    g.to_csv(OUT/'summary_metrics.csv',index=False)
    primary=g[g.scheme.eq('study_disjoint_group5')].set_index('representation'); rand=g[g.scheme.eq('random_split')].set_index('representation')
    rows=[]
    def add(endpoint,source,metric='macro_f1_mean'):
        rows.append({'endpoint':endpoint,'RAW_15165':source.loc['raw15165',metric] if 'raw15165' in source.index else np.nan,'PCA_15165':source.loc['pca15165_selected' if 'pca15165_selected' in source.index else 'pca15165_512',metric],'PCA_FULL':np.nan,'BridgeRNA_512':source.loc['bridgerna512',metric]})
    add('Random-split tissue macro F1',rand);add('Study-disjoint tissue macro F1',primary)
    n=neighborhood[neighborhood.k.eq(10)].set_index('representation');rows.append({'endpoint':'Tissue 10-NN purity','RAW_15165':n.loc['raw15165','tissue_purity'],'PCA_15165':n.loc['pca15165_512','tissue_purity'],'PCA_FULL':np.nan,'BridgeRNA_512':n.loc['bridgerna512','tissue_purity']});rows.append({'endpoint':'Study 10-NN purity','RAW_15165':n.loc['raw15165','study_purity'],'PCA_15165':n.loc['pca15165_512','study_purity'],'PCA_FULL':np.nan,'BridgeRNA_512':n.loc['bridgerna512','study_purity']})
    q=pd.DataFrame(rows);q['Delta_BridgeRNA_PCA']=q.BridgeRNA_512-q.PCA_15165;q.to_csv(OUT/'primary_summary_table.csv',index=False);return g,q


def figures(summary,neigh,learn,pcs):
    plt.style.use('seaborn-v0_8-whitegrid');FIG.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(10,5));q=summary[summary.scheme.eq('study_disjoint_group5')];ax.bar(q.representation,q.macro_f1_mean,yerr=q.macro_f1_sd,capsize=4);ax.set(ylabel='Macro F1',title='Study-disjoint tissue classification');ax.tick_params(axis='x',rotation=20);fig.tight_layout()
    for e in ['png','pdf']:fig.savefig(FIG/f'study_disjoint.{e}',dpi=400);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for n,q in neigh.groupby('representation'):axes[0].plot(q.k,q.tissue_purity,marker='o',label=n);axes[1].plot(q.k,q.study_purity,marker='o',label=n)
    axes[0].set(title='Tissue neighborhood purity',xlabel='k',ylabel='Fraction');axes[1].set(title='Study neighborhood purity',xlabel='k',ylabel='Fraction');axes[0].legend();axes[1].legend();fig.tight_layout()
    for e in ['png','pdf']:fig.savefig(FIG/f'neighborhood_purity.{e}',dpi=400);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for n,q in learn.groupby('representation'):axes[0].plot(q.training_study_fraction,q.macro_f1,marker='o',label=n)
    axes[1].plot(pcs.removed_bridge_pcs,pcs.macro_f1,marker='o');axes[0].set(title='Low-label learning curve',xlabel='Training-study fraction',ylabel='Macro F1');axes[1].set(title='BridgeRNA dominant-PC sensitivity',xlabel='Top PCs removed',ylabel='Macro F1');axes[0].legend();fig.tight_layout()
    for e in ['png','pdf']:fig.savefig(FIG/f'learning_and_pc_sensitivity.{e}',dpi=400);plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True);d,X,Z=load_cohort();print(f'[cohort] samples={len(d):,} studies={d.gse.nunique():,} tissues={d.tissue.nunique()}')
    folds,classes,curves,splits=evaluate(d,X,Z);folds.to_csv(OUT/'fold_metrics.csv',index=False);classes.to_csv(OUT/'per_class_metrics.csv',index=False);curves.to_csv(OUT/'pca_dimension_validation_curve.csv',index=False);splits.to_csv(OUT/'exact_splits.csv',index=False)
    neigh=neighborhoods(d,X,Z);neigh.to_csv(OUT/'neighborhood_results.csv',index=False);learn=learning_curves(d,X,Z);learn.to_csv(OUT/'learning_curves.csv',index=False);pcs=bridge_pc_sensitivity(d,Z);pcs.to_csv(OUT/'bridge_pc_removal_sensitivity.csv',index=False);summary,primary=summarize(folds,neigh,learn);figures(summary,neigh,learn,pcs)
    limitations={'PCA_FULL':'Unavailable for same 40k ARCHS4 samples; not forced.','leave_one_study_out':'Not run: the 40k reference was capped at at most two samples/GSE, making individual held-out-study estimates 1-2-sample unstable and repeated fold-specific PCA disproportionate. GroupKFold is primary.','study_prediction':'Not estimable responsibly: every labeled GSE has at most two samples in this preselected reference.','dataset_disjoint':'Existing balanced GTEx-human/ENCODE-mouse analysis reused as a qualified dataset-plus-species-disjoint secondary result in notebook.','normalization':'Existing per-sample log1p(TPM); no cohort normalization.','tissue_labels':'Conservative exact mapping of ARCHS4 free-text source_name_ch1.'}
    (OUT/'limitations.json').write_text(json.dumps(limitations,indent=2)+'\n');print(summary.to_string(index=False));print('\n',primary.to_string(index=False));print('[complete]',OUT)

if __name__=='__main__': main()
