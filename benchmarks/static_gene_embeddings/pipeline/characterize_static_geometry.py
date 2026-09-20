#!/usr/bin/env python3
"""Rigorous geometry and functional-topology tests for static gene embeddings."""
from __future__ import annotations
import argparse, json, time
from itertools import combinations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
from scipy import sparse
from scipy.stats import hypergeom
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import normalize
from statsmodels.stats.multitest import multipletests
from run_static_embeddings import ROOT, HERE, CHECKPOINT, VOCAB, GMTS, enrich, load_gmt, Logger

OUT = HERE / "results/static_characterization"
WORK = HERE / "work/static_characterization"
REPS = ["raw", "l2_cosine", "layernorm"]
LABELS = {"raw":"Raw", "l2_cosine":"L2/Cosine", "layernorm":"LayerNorm"}
SEEDS = [42,43,44,45,46]
KS = [5,10,15,20]
NN_K = [10,25,50]
N_INIT = 10
SILHOUETTE_SAMPLE = 3000

def representations(x):
    l2 = normalize(x, norm="l2").astype("float32")
    ln = x - x.mean(1, keepdims=True); ln /= np.sqrt(ln.var(1, keepdims=True)+1e-5)
    return {"raw":x.astype("float32"), "l2_cosine":l2, "layernorm":ln.astype("float32")}

def pca_analysis(name, x, genes, out, log):
    p=PCA(n_components=x.shape[1], svd_solver="full").fit(x); cum=np.cumsum(p.explained_variance_ratio_)
    pd.DataFrame({"representation":name,"pc":np.arange(1,len(cum)+1),"explained_variance_ratio":p.explained_variance_ratio_,"cumulative_explained_variance":cum}).to_csv(out/f"pca_variance_{name}.csv",index=False)
    row={"representation":name,"participation_ratio":float(1/np.square(p.explained_variance_ratio_).sum()),"pc1_variance":float(p.explained_variance_ratio_[0])}
    for q in [.5,.8,.9,.95]: row[f"pcs_for_{int(q*100)}pct"]=int(np.searchsorted(cum,q)+1)
    xy=p.transform(x)[:,:2].astype("float32"); np.save(WORK/f"pca2_{name}.npy",xy)
    if name=="raw": pd.DataFrame({"gene":genes,"embedding_norm":np.linalg.norm(x,axis=1)}).to_csv(out/"raw_embedding_norms.csv",index=False)
    log(f"PCA complete: {name}"); return row

def cluster_analysis(name,x,out,log):
    rows=[]; labels_by={}
    metric="cosine" if name=="l2_cosine" else "euclidean"
    for k in KS:
        labs=[]
        for seed in SEEDS:
            y=KMeans(k,random_state=seed,n_init=N_INIT,algorithm="lloyd").fit_predict(x); labs.append(y)
            rows.append({"representation":name,"k":k,"seed":seed,"silhouette":silhouette_score(x,y,metric=metric,sample_size=SILHOUETTE_SAMPLE,random_state=seed),"calinski_harabasz":calinski_harabasz_score(x,y),"min_cluster_size":int(np.bincount(y).min()),"max_cluster_size":int(np.bincount(y).max())})
        ari=[adjusted_rand_score(a,b) for a,b in combinations(labs,2)]
        for r in rows[-len(SEEDS):]: r["mean_pairwise_ari_for_k"]=float(np.mean(ari))
        labels_by[k]=labs[0]+1
    pd.DataFrame(rows).to_csv(out/f"clustering_metrics_{name}.csv",index=False)
    np.save(WORK/f"clusters_k10_{name}.npy",labels_by[10]); log(f"Clustering complete: {name}")
    return labels_by[10],pd.DataFrame(rows)

def shuffled_vector_null(name,x,reps,out,log):
    rng=np.random.default_rng(519); rows=[]; metric="cosine" if name=="l2_cosine" else "euclidean"
    for rep in range(reps):
        z=np.empty_like(x)
        for j in range(x.shape[1]): z[:,j]=x[rng.permutation(len(x)),j]
        if name=="l2_cosine": z=normalize(z,norm="l2").astype("float32")
        a=KMeans(10,random_state=42,n_init=5).fit_predict(z); b=KMeans(10,random_state=43,n_init=5).fit_predict(z)
        rows.append({"representation":name,"null_rep":rep,"silhouette":silhouette_score(z,a,metric=metric,sample_size=2000,random_state=rep),"calinski_harabasz":calinski_harabasz_score(z,a),"stability_ari":adjusted_rand_score(a,b)})
    q=pd.DataFrame(rows);q.to_csv(out/f"clustering_random_vector_null_{name}.csv",index=False);log(f"Random-vector null complete: {name}");return q

def exact_neighbors(name,x,max_k=50,batch=256):
    dev=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); allx=torch.as_tensor(x,device=dev); result=np.empty((len(x),max_k),np.int32)
    cosine=name=="l2_cosine"; norms=(allx*allx).sum(1)
    for start in range(0,len(x),batch):
        q=allx[start:start+batch]; scores=q@allx.T
        if cosine: dist=-scores
        else: dist=(q*q).sum(1,keepdim=True)+norms[None,:]-2*scores
        r=torch.arange(len(q),device=dev);dist[r,torch.arange(start,start+len(q),device=dev)]=torch.inf
        result[start:start+len(q)]=torch.topk(dist,max_k,largest=False).indices.cpu().numpy()
    return result

def annotation_bits(genes, gmt):
    terms=load_gmt(gmt); universe=set(genes); terms={n:(s&universe) for n,s in terms.items()};terms={n:s for n,s in terms.items() if 5<=len(s)<=2000}
    lookup={g:i for i,g in enumerate(genes)}; bits=[0]*len(genes)
    for ti,members in enumerate(terms.values()):
        flag=1<<ti
        for g in members: bits[lookup[g]]|=flag
    return bits,terms

def overlap_rate(bits, neighbors, k):
    shared=eligible=0; per=[]
    for i,row in enumerate(neighbors[:,:k]):
        if not bits[i]: continue
        valid=[j for j in row if bits[j]]
        if valid:
            hits=sum(bool(bits[i]&bits[j]) for j in valid);shared+=hits;eligible+=len(valid);per.append(hits/len(valid))
    return shared/eligible, float(np.mean(per)),eligible,len(per)

def neighborhood_tests(all_neighbors,genes,out,null_reps,log):
    rng=np.random.default_rng(812); observed=[]; null=[]
    for library,gmt in GMTS.items():
        bits,terms=annotation_bits(genes,gmt); annotated=np.array([i for i,b in enumerate(bits) if b],int)
        for k in NN_K:
            for name,nbr in all_neighbors.items():
                rate,per,pairs,qgenes=overlap_rate(bits,nbr,k);observed.append({"representation":name,"library":library.upper(),"k":k,"shared_pair_fraction":rate,"mean_gene_shared_fraction":per,"eligible_pairs":pairs,"annotated_query_genes":qgenes})
            for rep in range(null_reps):
                random_nbr=rng.choice(annotated,size=(len(genes),k),replace=True)
                rate,per,pairs,qgenes=overlap_rate(bits,random_nbr,k);null.append({"library":library.upper(),"k":k,"null_rep":rep,"shared_pair_fraction":rate,"mean_gene_shared_fraction":per})
        log(f"Neighborhood annotation tests complete: {library}")
    obs=pd.DataFrame(observed);nul=pd.DataFrame(null)
    for i,r in obs.iterrows():
        q=nul[(nul.library==r.library)&(nul.k==r.k)].shared_pair_fraction
        obs.loc[i,"null_mean"]=q.mean();obs.loc[i,"effect_difference"]=r.shared_pair_fraction-q.mean();obs.loc[i,"fold_over_null"]=r.shared_pair_fraction/q.mean();obs.loc[i,"empirical_p"]=(1+(q>=r.shared_pair_fraction).sum())/(len(q)+1)
    obs.to_csv(out/"neighborhood_functional_overlap.csv",index=False);nul.to_csv(out/"neighborhood_random_null.csv",index=False);return obs,nul

def neighborhood_agreement(neighbors,out):
    rows=[]
    for a,b in combinations(REPS,2):
        for k in NN_K:
            vals=[]
            for x,y in zip(neighbors[a][:,:k],neighbors[b][:,:k]): vals.append(len(set(x)&set(y))/len(set(x)|set(y)))
            rows.append({"representation_a":a,"representation_b":b,"k":k,"mean_neighbor_jaccard":np.mean(vals),"sd_neighbor_jaccard":np.std(vals)})
    pd.DataFrame(rows).to_csv(out/"neighborhood_representation_agreement.csv",index=False)

def enrichment_null(labels,genes,gmt,reps=100):
    terms=load_gmt(gmt); universe=set(genes);members=[s&universe for s in terms.values()];members=[s for s in members if 5<=len(s)<=2000]; annotated=universe&set().union(*members); gi={g:i for i,g in enumerate(genes)}
    rows=[];cols=[]
    for j,s in enumerate(members):
        for g in s: rows.append(gi[g]);cols.append(j)
    mat=sparse.csr_matrix((np.ones(len(rows),np.int8),(rows,cols)),shape=(len(genes),len(members))); term_sizes=np.asarray(mat.sum(0)).ravel();ann=np.array([g in annotated for g in genes]);rng=np.random.default_rng(901);out=[]
    for rep in range(reps):
        y=rng.permutation(labels);sig=0
        for c in range(1,11):
            q=(y==c)&ann;n=q.sum();hits=np.asarray(mat[q].sum(0)).ravel();p=hypergeom.sf(hits-1,ann.sum(),term_sizes,n);sig+=int((multipletests(p,method="fdr_bh")[1]<.05).sum())
        out.append({"null_rep":rep,"significant_terms":sig})
    return pd.DataFrame(out)

def projections(name,x,clusters,out,seed,log):
    p50=PCA(50,random_state=seed).fit_transform(x); ts=TSNE(2,perplexity=30,init="pca",learning_rate="auto",max_iter=1500,random_state=seed).fit_transform(p50); um=umap.UMAP(n_neighbors=30,min_dist=.1,n_components=2,metric="cosine" if name=="l2_cosine" else "euclidean",random_state=seed).fit_transform(p50)
    pc=np.load(WORK/f"pca2_{name}.npy");q=pd.DataFrame({"gene":pd.read_csv(VOCAB).sort_values('token_id').gene_symbol,"cluster":clusters,"pca_1":pc[:,0],"pca_2":pc[:,1],"tsne_1":ts[:,0],"tsne_2":ts[:,1],"umap_1":um[:,0],"umap_2":um[:,1]});q.to_csv(out/f"projections_{name}.csv",index=False);log(f"Visual projections complete: {name}")

def plot_outputs(out):
    colors="tab10"
    fig,axes=plt.subplots(3,3,figsize=(15,14),constrained_layout=True)
    for i,name in enumerate(REPS):
        q=pd.read_csv(out/f"projections_{name}.csv")
        for j,(a,b,title) in enumerate([("pca_1","pca_2","PCA"),("tsne_1","tsne_2","t-SNE"),("umap_1","umap_2","UMAP")]):
            axes[i,j].scatter(q[a],q[b],c=q.cluster,s=2,cmap=colors,vmin=.5,vmax=10.5,rasterized=True);axes[i,j].set_title(f"{LABELS[name]} — {title}");axes[i,j].set(xticks=[],yticks=[])
    for ext in ["png","pdf"]:fig.savefig(out/f"geometry_projections.{ext}",dpi=180,bbox_inches="tight")
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    for ax,name in zip(axes,REPS):
        q=pd.read_csv(out/f"pca_variance_{name}.csv");ax.plot(q.pc,q.cumulative_explained_variance);[ax.axhline(v,color="gray",lw=.7,ls="--") for v in [.5,.8,.9,.95]];ax.set(title=LABELS[name],xlabel="PC",ylabel="Cumulative explained variance",xlim=(1,512),ylim=(0,1.01))
    for ext in ["png","pdf"]:fig.savefig(out/f"pca_variance_spectra.{ext}",dpi=300,bbox_inches="tight")
    plt.close(fig)

def main():
    global SEEDS,KS,N_INIT,SILHOUETTE_SAMPLE
    ap=argparse.ArgumentParser();ap.add_argument("--null-reps",type=int,default=100);ap.add_argument("--vector-null-reps",type=int,default=10);ap.add_argument("--exploratory",action="store_true",help="Fast screening profile; not confirmatory");a=ap.parse_args()
    if a.exploratory:
        SEEDS=[42,43,44];KS=[5,10,20];N_INIT=3;SILHOUETTE_SAMPLE=1000;a.null_reps=min(a.null_reps,10);a.vector_null_reps=min(a.vector_null_reps,1)
    OUT.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True);log=Logger(OUT/"run.log");started=time.time();log(f"Profile={'exploratory' if a.exploratory else 'confirmatory'}; seeds={SEEDS}; K={KS}; n_init={N_INIT}; annotation_nulls={a.null_reps}; vector_nulls={a.vector_null_reps}")
    vocab=pd.read_csv(VOCAB).sort_values("token_id");genes=vocab.gene_symbol.astype(str).str.upper().tolist();state=torch.load(CHECKPOINT,map_location="cpu",weights_only=False)["model_state_dict"];x=state["gene_embedding.weight"].numpy();reps=representations(x);pca_rows=[];cluster_frames=[];neighbors={};labels={};null_frames=[]
    for name,z in reps.items():
        pca_rows.append(pca_analysis(name,z,genes,OUT,log));y,m=cluster_analysis(name,z,OUT,log);labels[name]=y;cluster_frames.append(m);null_frames.append(shuffled_vector_null(name,z,a.vector_null_reps,OUT,log));neighbors[name]=exact_neighbors(name,z);np.save(WORK/f"neighbors_top50_{name}.npy",neighbors[name]);projections(name,z,y,OUT,42,log)
        assignment=pd.DataFrame({"token_id":vocab.token_id,"gene":genes,"cluster":y});assignment.to_csv(OUT/f"cluster_assignments_{name}.csv",index=False)
        for lib,gmt in GMTS.items():
            e,meta=enrich(assignment,lib.upper(),gmt);e.to_csv(OUT/f"{lib}_enrichment_{name}.csv",index=False);n=enrichment_null(y,genes,gmt,a.null_reps);n.insert(0,"representation",name);n.insert(1,"library",lib.upper());n.to_csv(OUT/f"{lib}_cluster_enrichment_null_{name}.csv",index=False)
    pca=pd.DataFrame(pca_rows);pca.to_csv(OUT/"pca_summary.csv",index=False);cl=pd.concat(cluster_frames,ignore_index=True);cl.to_csv(OUT/"clustering_metrics_all.csv",index=False);neigh,_=neighborhood_tests(neighbors,genes,OUT,a.null_reps,log);neighborhood_agreement(neighbors,OUT);plot_outputs(OUT)
    summary=[]
    for name in REPS:
        primary=cl[(cl.representation==name)&(cl.k==10)];go=pd.read_csv(OUT/f"go_enrichment_{name}.csv");ke=pd.read_csv(OUT/f"kegg_enrichment_{name}.csv");pn=pca[pca.representation==name].iloc[0];cn=pd.read_csv(OUT/f"clustering_random_vector_null_{name}.csv")
        row={"representation":name,"pca_participation_ratio":pn.participation_ratio,"pcs_for_90pct":pn.pcs_for_90pct,"silhouette":primary.silhouette.mean(),"calinski_harabasz":primary.calinski_harabasz.mean(),"cluster_stability_ari":primary.mean_pairwise_ari_for_k.iloc[0],"significant_go_terms":int(go.significant.sum()),"significant_kegg_terms":int(ke.significant.sum()),"silhouette_null_mean":cn.silhouette.mean(),"silhouette_empirical_p":(1+(cn.silhouette>=primary.silhouette.mean()).sum())/(len(cn)+1),"ch_null_mean":cn.calinski_harabasz.mean(),"stability_null_mean":cn.stability_ari.mean()}
        for lib in ["GO","KEGG"]:
            for k in NN_K:
                q=neigh[(neigh.representation==name)&(neigh.library==lib)&(neigh.k==k)].iloc[0];row[f"{lib.lower()}_neighbor_fold_k{k}"]=q.fold_over_null;row[f"{lib.lower()}_neighbor_p_k{k}"]=q.empirical_p
        summary.append(row)
    pd.DataFrame(summary).to_csv(OUT/"representation_summary.csv",index=False)
    prov={"status":"complete","analysis_profile":"exploratory" if a.exploratory else "confirmatory","confirmatory_inference":not a.exploratory,"genes":len(genes),"representations":{"raw":"checkpoint tensor unchanged","l2_cosine":"row-wise unit L2 norm","layernorm":"post-hoc row-wise mean zero/unit variance with eps=1e-5; not a retrained model"},"primary_k":10,"secondary_k":KS,"seeds":SEEDS,"kmeans_n_init":N_INIT,"silhouette_sample_size":SILHOUETTE_SAMPLE,"nearest_neighbor_k":NN_K,"nearest_neighbors":"exact GPU/CPU matrix geometry","nulls":{"annotation_random_neighbor_reps":a.null_reps,"cluster_label_permutation_reps":a.null_reps,"featurewise_shuffled_vector_reps":a.vector_null_reps},"visualization":"PCA; t-SNE and UMAP exploratory only","limitations":["exploratory profile uses too few null replicates for precise tail probabilities"] if a.exploratory else [],"elapsed_seconds":time.time()-started};(OUT/"provenance.json").write_text(json.dumps(prov,indent=2)+"\n");log(f"Complete elapsed={(time.time()-started)/60:.1f}m")
if __name__=="__main__":main()
