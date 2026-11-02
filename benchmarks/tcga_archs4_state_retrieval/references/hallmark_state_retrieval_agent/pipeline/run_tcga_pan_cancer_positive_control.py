#!/usr/bin/env python3
"""Matched TCGA positive control using frozen saved expression/Hallmark/Bridge outputs."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np,pandas as pd,seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage,leaves_list,fcluster
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import adjusted_rand_score
from common import RESULTS,save_json

SRC=Path('/home/walt/bridge-rna.worktrees/create-a-new-self-contained-benchmark-investigat/benchmarks/bridge_hallmark_representation');OUT=RESULTS/'tcga_pan_cancer_positive_control';FIG=OUT/'figures'
def corrmat(x): return np.corrcoef(x)
def main():
 OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(exist_ok=True)
 m=pd.read_parquet(SRC/'results/tcga_sample_manifest.parquet').sort_values('matrix_row').reset_index(drop=True);e=pd.read_parquet(SRC/'results/tcga_expression_hallmark_scores.parquet').set_index('sample_id').loc[m.sample_id];p=pd.read_parquet(SRC/'results/tcga_zero_shot_hallmark_profiles.parquet').set_index('sample_id').loc[m.sample_id];z=np.load(SRC/'work/tcga_embeddings.npy');names=e.columns.tolist();assert p.columns.tolist()==names and len(z)==len(m)
 cohorts=sorted(m.cohort.unique());em=np.vstack([e.loc[m.cohort.values==c].mean(0) for c in cohorts]);pm=np.vstack([p.loc[m.cohort.values==c].mean(0) for c in cohorts]);zm=np.vstack([z[m.cohort.values==c].mean(0) for c in cohorts])
 emz=(em-em.mean(0))/em.std(0);pmz=(pm-pm.mean(0))/pm.std(0)
 profile=[]
 for i,c in enumerate(cohorts): profile.append({'cancer':c,'n':int((m.cohort==c).sum()),'pearson_conventional_vs_bridge_hallmark':pearsonr(emz[i],pmz[i]).statistic,'spearman_conventional_vs_bridge_hallmark':spearmanr(emz[i],pmz[i]).statistic})
 pd.DataFrame(profile).to_csv(OUT/'cancer_profile_concordance.csv',index=False)
 se,sp,sz=corrmat(emz),corrmat(pmz),corrmat(zm);tri=np.triu_indices(len(cohorts),1)
 nearest=[]
 for i,c in enumerate(cohorts):
  def nn(a): q=a[i].copy();q[i]=-np.inf;return cohorts[int(np.argmax(q))]
  nearest.append({'cancer':c,'conventional_nearest':nn(se),'bridge_hallmark_nearest':nn(sp),'bridge512_nearest':nn(sz),'hallmark_agrees':nn(se)==nn(sp)})
 pd.DataFrame(nearest).to_csv(OUT/'nearest_cancer_agreement.csv',index=False)
 pw=[{'hallmark':n,'cancer_ranking_spearman':spearmanr(emz[:,j],pmz[:,j]).statistic} for j,n in enumerate(names)];pd.DataFrame(pw).to_csv(OUT/'pathway_cancer_ranking_concordance.csv',index=False)
 clusters=[]
 for k in [4,5,6,8,10]:
  ce=fcluster(linkage(emz,method='average',metric='correlation'),k,criterion='maxclust');cp=fcluster(linkage(pmz,method='average',metric='correlation'),k,criterion='maxclust');cz=fcluster(linkage(zm,method='average',metric='cosine'),k,criterion='maxclust');clusters.append({'k':k,'ARI_conventional_vs_bridge_hallmark':adjusted_rand_score(ce,cp),'ARI_conventional_vs_bridge512':adjusted_rand_score(ce,cz),'ARI_bridge_hallmark_vs_bridge512':adjusted_rand_score(cp,cz)})
 pd.DataFrame(clusters).to_csv(OUT/'clustering_agreement.csv',index=False)
 pairs=[]
 for i,j in zip(*tri): pairs.append({'cancer_a':cohorts[i],'cancer_b':cohorts[j],'conventional_hallmark_similarity':se[i,j],'bridge_hallmark_similarity':sp[i,j],'bridge512_similarity':sz[i,j]})
 pairs=pd.DataFrame(pairs);pairs['hallmark_minus_bridge512']=pairs.bridge_hallmark_similarity-pairs.bridge512_similarity
 pairs['bridge_hallmark_similarity_rank']=pairs.bridge_hallmark_similarity.rank(ascending=False,method='min');pairs['bridge512_similarity_rank']=pairs.bridge512_similarity.rank(ascending=False,method='min');pairs['functional_rank_gain_vs_bridge512']=pairs.bridge512_similarity_rank-pairs.bridge_hallmark_similarity_rank
 pairs.sort_values(['functional_rank_gain_vs_bridge512','bridge_hallmark_similarity'],ascending=False).to_csv(OUT/'cross_tissue_functional_pairs.csv',index=False)
 for label,mat in [('conventional_hallmark',se),('bridge_hallmark',sp),('bridge512',sz)]: pd.DataFrame(mat,index=cohorts,columns=cohorts).to_csv(OUT/f'{label}_cancer_similarity.csv')
 order=leaves_list(linkage(emz,method='average',metric='correlation'));fig,ax=plt.subplots(figsize=(15,10));sns.heatmap(pd.DataFrame(emz,index=cohorts,columns=names).iloc[order],cmap='RdBu_r',center=0,ax=ax,xticklabels=True,yticklabels=True);ax.set_title('TCGA cancer × Hallmark: conventional expression-derived rank score');ax.tick_params(axis='x',labelsize=6,rotation=75);fig.tight_layout();fig.savefig(FIG/'cancer_hallmark_heatmap.png',dpi=300);fig.savefig(FIG/'cancer_hallmark_heatmap.pdf');plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(20,6));
 for ax,(title,mat) in zip(axes,[('Conventional Hallmark',se),('Bridge Hallmark-50',sp),('Bridge-512',sz)]): sns.heatmap(mat[np.ix_(order,order)],cmap='RdBu_r',center=0,vmin=-1,vmax=1,ax=ax,xticklabels=np.array(cohorts)[order],yticklabels=np.array(cohorts)[order]);ax.set_title(title);ax.tick_params(labelsize=5)
 fig.tight_layout();fig.savefig(FIG/'cancer_similarity_matrices.png',dpi=300);fig.savefig(FIG/'cancer_similarity_matrices.pdf');plt.close(fig)
 summary={'paper':'McClure et al., Landscape of Genetic Alterations Underlying Hallmark Signature Changes in Cancer Reveals TP53 Aneuploidy-driven Metabolic Reprogramming, Cancer Research Communications (2023)','paper_method_frozen_before_bridge_analysis':{'samples':8603,'tumor_types':31,'signatures_started':50,'signatures_analyzed':48,'excluded':['KRAS Signaling Dn','Spermatogenesis'],'ssGSEA2_parameters':{'normalization':'rank','weight':0.75,'statistic':'area under RES','output':'NES','permutations':1000,'min_overlap':5,'correlation_type':'z score'},'major_relationship':'TP53 and aneuploidy associated with high proliferation and glycolysis, especially squamous cancers and basal/basal-squamous BRCA/BLCA subtypes'},'local_reconstruction':{'matched_samples':len(m),'cancer_types':len(cohorts),'deviation':'available conventional profiles are deterministic rank-mean ssGSEA-like scores on the frozen 15,165-gene vocabulary, not exact ssGSEA 2.0 NES; subtype/mutation reconstruction not claimed'},'metrics':{'mean_cancer_profile_pearson':float(np.mean([x['pearson_conventional_vs_bridge_hallmark'] for x in profile])),'similarity_matrix_spearman_conventional_vs_bridge_hallmark':float(spearmanr(se[tri],sp[tri]).statistic),'similarity_matrix_spearman_conventional_vs_bridge512':float(spearmanr(se[tri],sz[tri]).statistic),'nearest_cancer_agreement_fraction':float(np.mean([x['hallmark_agrees'] for x in nearest])),'median_pathway_ranking_spearman':float(pd.DataFrame(pw).cancer_ranking_spearman.median())}}
 save_json(OUT/'provenance_and_summary.json',summary)
 pd.DataFrame(em,index=cohorts,columns=names).to_csv(OUT/'conventional_cancer_hallmark_profiles.csv');pd.DataFrame(pm,index=cohorts,columns=names).to_csv(OUT/'bridge_cancer_hallmark_profiles.csv')
if __name__=='__main__':main()
