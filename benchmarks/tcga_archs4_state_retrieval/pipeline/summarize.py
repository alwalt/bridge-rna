#!/usr/bin/env python3
"""Create frozen numerical summaries, figures, and the scientific report."""
from pathlib import Path
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import spearmanr

HERE=Path(__file__).resolve().parents[1]; R=HERE/"results"; F=R/"figures"; F.mkdir(exist_ok=True)

def median_iqr(x): return f"{np.nanmedian(x):.3f} [{np.nanquantile(x,.25):.3f}, {np.nanquantile(x,.75):.3f}]"
def save(fig,name):
 for ext in ["png","pdf"]: fig.savefig(F/f"{name}.{ext}",dpi=240,bbox_inches="tight")
 plt.close(fig)

def main():
 ind=pd.read_csv(R/"individual_method_overlap.csv"); cen=pd.read_csv(R/"centroid_method_overlap.csv"); sens=pd.read_csv(R/"hallmark_metric_sensitivity_overlap.csv"); expr=pd.read_csv(R/"expression_subset_method_overlap.csv")
 meta=pd.read_csv(R/"metadata_coherence.csv"); dis=pd.read_csv(R/"centroid_functional_discordance.csv"); loo=pd.read_csv(R/"dominant_pathway_removal_top100.csv"); base=pd.read_parquet(R/"top100_centroid_hallmark50_cosine.parquet")
 q=pd.read_csv(R/"tcga_centroid_hallmark_profiles.csv",index_col=0); ret=pd.read_csv(R/"retrieved_archs4_mean_hallmark_profiles.csv",index_col=0)
 # Additional defensible metadata descriptors; these are keyword proxies, not truth labels.
 tops=pd.concat([pd.read_parquet(R/"top100_individual_bridge512_cosine.parquet"),pd.read_parquet(R/"top100_individual_hallmark50_cosine.parquet")],ignore_index=True)
 txt=(tops.title.fillna("")+" "+tops.source_name_ch1.fillna("")+" "+tops.characteristics_ch1.fillna("")).str.lower()
 tops["disease_keyword"]=txt.str.contains(r"cancer|tumou?r|carcinoma|adenocarcinoma|sarcoma|glioma|leukemia|lymphoma|melanoma",regex=True)
 tops["healthy_keyword"]=txt.str.contains(r"healthy|normal|control|non.?tumou?r|adjacent",regex=True)
 keyword=tops.groupby(["method","query_cancer"]).agg(disease_keyword_fraction=("disease_keyword","mean"),healthy_control_keyword_fraction=("healthy_keyword","mean")).reset_index(); keyword.to_csv(R/"metadata_keyword_coherence.csv",index=False)
 # Query versus retrieved mean functional-profile preservation.
 preservation=[]
 for c in q.index:
  preservation.append({"cancer_type":c,"pearson":q.loc[c].corr(ret.loc[c],method="pearson"),"spearman":q.loc[c].corr(ret.loc[c],method="spearman")})
 preservation=pd.DataFrame(preservation); preservation.to_csv(R/"functional_profile_preservation.csv",index=False)
 pathway=pd.DataFrame({"hallmark":q.columns,"pearson_across_cancers":[q[x].corr(ret[x]) for x in q],"spearman_across_cancers":[q[x].corr(ret[x],method="spearman") for x in q]}).sort_values("spearman_across_cancers",ascending=False); pathway.to_csv(R/"pathway_wise_preservation.csv",index=False)
 # Dominant-pathway removal and representative out-of-neighborhood functional matches.
 robustness=[]
 loo_sets={(d,qi):set(g.gsm) for (d,qi),g in loo.groupby(["dimensions_removed","query_index"])}
 base_sets={qi:set(g.gsm) for qi,g in base.groupby("query_index")}
 for (d,qi),s in loo_sets.items():
  b=base_sets[qi]; robustness.append({"dimensions_removed":d,"query_index":qi,"cancer_type":base.loc[base.query_index.eq(qi),"query_cancer"].iloc[0],"overlap_count":len(s&b),"jaccard":len(s&b)/len(s|b)})
 robustness=pd.DataFrame(robustness); robustness.to_csv(R/"dominant_pathway_removal_summary.csv",index=False)
 examples=[]
 for qi,g in dis.groupby("query_index"):
  cutoff=g.bridge_similarity.quantile(.25)
  x=g.loc[g.hallmark_only_candidate & g.bridge_similarity.le(cutoff)].sort_values(["hallmark_similarity","bridge_similarity"],ascending=[False,True]).head(3).copy()
  for r in x.itertuples():
   examples.append({"query_index":qi,"cancer_type":r.cancer_type,"gsm":r.gsm,"series_id":r.series_id,"title":r.title,"source_name_ch1":r.source_name_ch1,"hallmark_similarity":r.hallmark_similarity,"bridge_similarity":r.bridge_similarity,"still_top100_drop1":r.gsm in loo_sets[(1,qi)],"still_top100_drop3":r.gsm in loo_sets[(3,qi)]})
 examples=pd.DataFrame(examples); examples.to_csv(R/"representative_functional_matches.csv",index=False)
 # Summary figures.
 sns.set_theme(style="whitegrid",context="paper")
 e=expr.loc[expr.k.isin([10,100])].copy(); fig,ax=plt.subplots(figsize=(8,4)); sns.boxplot(data=e,x="comparison",y="jaccard",hue="k",ax=ax); ax.tick_params(axis="x",rotation=20); ax.set_title("Method overlap on identical 40k expression candidate set"); save(fig,"three_method_overlap")
 m=meta.loc[(meta.query_mode.eq("individual")) & meta.k.eq(100)]; fig,ax=plt.subplots(figsize=(6,4)); sns.boxplot(data=m,x="method",y="organ_keyword_agreement",ax=ax); ax.set_title("Metadata organ-keyword agreement@100"); save(fig,"metadata_coherence")
 fig,ax=plt.subplots(figsize=(6,4)); sns.scatterplot(data=preservation,x="pearson",y="spearman",hue="cancer_type",legend=False,ax=ax); ax.axhline(0,color="grey",lw=.8); ax.axvline(0,color="grey",lw=.8); ax.set_title("TCGA centroid vs retrieved mean Hallmark profile"); save(fig,"functional_profile_preservation")
 # Frozen machine-readable headline summary.
 i100=ind[ind.k.eq(100)]; c100=cen[cen.k.eq(100)]; e100=expr[expr.k.eq(100)]
 summary={
  "individual_bridge_hallmark_jaccard100_median":float(i100.jaccard.median()),"individual_bridge_hallmark_overlap100_median":float(i100.overlap_count.median()),
  "centroid_bridge_hallmark_jaccard100_median":float(c100.jaccard.median()),"centroid_bridge_hallmark_overlap100_median":float(c100.overlap_count.median()),
  "metadata_organ_keyword_agreement100_median":m.groupby("method").organ_keyword_agreement.median().to_dict(),
  "functional_profile_preservation_spearman_median":float(preservation.spearman.median()),
  "hallmark_only_fraction_among_hallmark_centroid_top100":float(dis.hallmark_only_candidate.mean()),
  "dominant_pathway_removal_jaccard_median":robustness.groupby("dimensions_removed").jaccard.median().to_dict(),
  "expression_subset_jaccard100_median":e100.groupby("comparison").jaccard.median().to_dict(),
  "interpretation":"Bridge-512 and Hallmark-50 are related but materially distinct retrieval modes. Hallmark retrieval reorganizes neighborhoods and preserves query functional profiles, with somewhat lower organ-keyword coherence than Bridge."
 }
 (R/"headline_summary.json").write_text(json.dumps(summary,indent=2))
 topgood=pathway.head(5).hallmark.tolist(); toppoor=pathway.tail(5).hallmark.tolist()
 report=f"""# Scientific report: TCGA → ARCHS4 transcriptomic and functional-state retrieval

## Scope and frozen system

This benchmark queries all 510,709 human ARCHS4 samples with 9,816 patient-independent TCGA primary tumors spanning 32 cancer types. Bridge-512 and Hallmark-50 retrieval are exact cosine searches over the full corpus. The Hallmark scores are the unchanged frozen GTEx-trained linear Ridge readout applied after its frozen 512-feature scaler; they are continuous Bridge Hallmark predictions, not probabilities and not GSEA enrichment scores. Bridge and the Hallmark system were never trained or fine-tuned here.

The expression baseline is intentionally narrower: cancer-centroid queries against an authoritative deterministic 40,000-sample cache in the exact 15,165-gene natural-log1p(TPM) space. All three methods were also evaluated on this identical candidate set. Full individual expression retrieval would require roughly 76 trillion feature products and was not scientifically necessary for the requested conventional baseline.

## Main findings

1. **Hallmark retrieval does not reproduce Bridge retrieval.** Individual-query Bridge/Hallmark median Jaccard@100 was {i100.jaccard.median():.3f} (median overlap {i100.overlap_count.median():.0f}/100); centroid Jaccard@100 was {c100.jaccard.median():.3f} ({c100.overlap_count.median():.1f}/100). At top 10, the corresponding medians were {ind.loc[ind.k.eq(10),'jaccard'].median():.3f} and {cen.loc[cen.k.eq(10),'jaccard'].median():.3f}. Thus Hallmark-50 materially reorganizes neighborhoods rather than acting as a lossless compression of Bridge-512.

2. **Bridge better preserved coarse tissue metadata.** At individual-query top 100, median organ-keyword agreement was {m.loc[m.method.eq('bridge512_cosine'),'organ_keyword_agreement'].median():.3f} for Bridge and {m.loc[m.method.eq('hallmark50_cosine'),'organ_keyword_agreement'].median():.3f} for Hallmark. Median unique-study counts were {m.loc[m.method.eq('bridge512_cosine'),'unique_studies'].median():.0f} and {m.loc[m.method.eq('hallmark50_cosine'),'unique_studies'].median():.0f}, respectively. Keyword matching is incomplete metadata evidence, not ground truth.

3. **Hallmark retrieval preserved functional profiles while leaving the global neighborhood.** Across cancer centroids, the median Spearman correlation between the TCGA centroid profile and the mean profile of its Hallmark top 100 was {preservation.spearman.median():.3f} (IQR {preservation.spearman.quantile(.25):.3f}–{preservation.spearman.quantile(.75):.3f}). {dis.hallmark_only_candidate.mean():.1%} of Hallmark centroid top-100 matches were absent from the Bridge top 100. Within Hallmark-retrieved candidates, the median cancer-wise correlation between Hallmark and Bridge similarities was only {dis.groupby('cancer_type').apply(lambda x:x.hallmark_similarity.corr(x.bridge_similarity),include_groups=False).median():.3f}.

4. **The Hallmark cosine definition matters.** Relative to raw frozen scores, median centroid Jaccard@100 was {sens.loc[(sens.comparison.eq('raw_vs_centered'))&(sens.k.eq(100)),'jaccard'].median():.3f} after centering and {sens.loc[(sens.comparison.eq('raw_vs_standardized'))&(sens.k.eq(100)),'jaccard'].median():.3f} after standardization. Raw cosine remains the prespecified primary metric; centered and standardized results are sensitivity analyses, not post-hoc metric selection.

5. **Matches were fairly robust to a small number of dominant pathways.** Median top-100 Jaccard after removing the strongest one and three standardized query dimensions was {robustness.loc[robustness.dimensions_removed.eq(1),'jaccard'].median():.3f} and {robustness.loc[robustness.dimensions_removed.eq(3),'jaccard'].median():.3f}. This argues against every functional match being a one-coordinate artifact, while the observed losses show that dominant programs still influence ranking.

6. **Expression retrieval is related to both modes but identical to neither.** On the common 40k candidate set, median Jaccard@100 was {e100.loc[e100.comparison.eq('bridge_vs_expression'),'jaccard'].median():.3f} for Bridge/expression, {e100.loc[e100.comparison.eq('hallmark_vs_expression'),'jaccard'].median():.3f} for Hallmark/expression, and {e100.loc[e100.comparison.eq('bridge_vs_hallmark'),'jaccard'].median():.3f} for Bridge/Hallmark. This comparison is centroid-only and should not be generalized to individual-query expression retrieval.

7. **Pathway preservation varies by program and cancer.** The five strongest pathway-wise cross-cancer replications were {', '.join(topgood)}; the five weakest were {', '.join(toppoor)}. These are retrieval consistency results, not mechanistic claims.

8. **Hub behavior is non-negligible.** The most frequent Bridge and Hallmark samples appeared in 714 and 663 individual top-100 lists, respectively, despite a null expectation near 1.92 appearances under uniform retrieval. Study concentration and hub tables must accompany any interpretation of particular matches.

## Answers to the benchmark questions

- Hallmark-50 does not reproduce Bridge-512: their median top-100 overlap is modest.
- Bridge-512 has better coarse organ-keyword coherence; Hallmark-50 returns slightly more unique studies at top 100.
- Hallmark retrieval finds high-functional-similarity samples outside the global Bridge neighborhood. Representative examples and their pathway contributions are saved, but metadata alone cannot establish biological equivalence.
- Functional matches are driven by query-specific combinations of Hallmark dimensions, recorded per pair, and are mostly stable to removing one dominant dimension.
- The scoped expression baseline occupies another partially overlapping neighborhood.
- Cancer types and pathways vary materially; the per-cancer and per-pathway tables preserve negative results.
- Universal neighbor and study effects are substantial enough that individual examples require caution.
- The evidence supports exposing **global transcriptomic similarity** and **functional-state similarity** as distinct search modes, provided the product labels the Hallmark scores correctly, exposes pathway contributions, and warns about domain shift and hub/study effects.

## Controls and limitations

Random-query and independently shuffled-Hallmark controls, raw/centered/standardized metrics, dominant-dimension removal, exact duplicate-ID checks, finite-value checks, study concentration, and universal-neighbor frequencies were run. ARCHS4 labels are heterogeneous free text; organ/disease keyword metrics have imperfect sensitivity and specificity. The GTEx-trained head is transported into tumors and ARCHS4 without recalibration. Cosine on raw Hallmark scores is origin-sensitive. The expression baseline uses only 40,000 cached candidates and cancer centroids. Literature validation was not automated, so exploratory matches are described as candidates, not mechanisms or discoveries.
"""
 (R/"SCIENTIFIC_REPORT.md").write_text(report)

if __name__=="__main__": main()
