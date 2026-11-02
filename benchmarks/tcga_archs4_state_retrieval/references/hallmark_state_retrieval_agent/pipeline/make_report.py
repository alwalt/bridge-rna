#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from common import RESULTS, save_json

def main():
    tc=pd.read_csv(RESULTS/"tcga/hallmark50_vs_bridge512.csv"); hits=pd.read_csv(RESULTS/"tcga/retrieval_results.csv"); oc=pd.read_csv(RESULTS/"osdr/hallmark50_vs_bridge512.csv"); ohits=pd.read_csv(RESULTS/"osdr/retrieval_results.csv")
    figdir=RESULTS/"figures"; figdir.mkdir(parents=True,exist_ok=True)
    p=tc.pivot_table(index="cohort",columns="method",values="top10_keyword_precision",aggfunc="mean").sort_index(); ax=p.plot.bar(figsize=(11,5),color=["#7a5195","#ef5675"]); ax.set(ylabel="Top-10 metadata keyword precision",xlabel="TCGA cohort",ylim=(0,1),title="TCGA → ARCHS4 biological metadata coherence"); ax.legend(title="Retrieval representation",frameon=False); plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(figdir/"tcga_metadata_coherence.png",dpi=240); plt.savefig(figdir/"tcga_metadata_coherence.pdf"); plt.close()
    s=tc.groupby("method").top10_keyword_precision.mean(); fig,ax=plt.subplots(figsize=(6,5)); ax.bar(s.index,s.values,color=["#7a5195","#ef5675"]); ax.set(ylabel="Mean cohort keyword precision@10",ylim=(0,1),title="50-D versus 512-D retrieval"); ax.tick_params(axis="x",rotation=15); fig.tight_layout(); fig.savefig(figdir/"method_comparison.png",dpi=240); fig.savefig(figdir/"method_comparison.pdf"); plt.close(fig)
    delta=(tc.pivot_table(index=["query_id","cohort"],columns="method",values="top10_keyword_precision").dropna()); delta["delta_50_minus_512"]=delta["Hallmark-50 cosine"]-delta["Bridge-512 cosine"]; delta.reset_index().to_csv(RESULTS/"tcga/paired_method_differences.csv",index=False)
    examples=hits[(hits.method.eq("Hallmark-50 cosine"))&(hits["rank"].le(3))].copy(); examples.to_csv(RESULTS/"tcga/qualitative_top3_examples.csv",index=False)
    otext=ohits[["title","source_name","characteristics"]].fillna("").agg(" ".join,axis=1).str.lower(); ohits["liver_metadata_match"]=otext.str.contains(r"liver|hepatic|hepatocyte",regex=True)
    osum=ohits.groupby("method",as_index=False).agg(liver_precision_at_10=("liver_metadata_match","mean"),mean_similarity=("similarity","mean"),unique_studies=("series_id","nunique")); osum.to_csv(RESULTS/"osdr/liver_coherence_summary.csv",index=False)
    readme=f"""# Bridge Hallmark-state retrieval

This benchmark applies the frozen GTEx-only 512-D → 50-Hallmark ridge probe to the compatible human ARCHS4 Bridge embedding cache, then compares exact cosine retrieval in Hallmark and raw Bridge space.

## Reproduce

```bash
cd {RESULTS.parent}
/home/walt/bridge-rna/.venv/bin/python pipeline/audit_and_build.py
/home/walt/bridge-rna/.venv/bin/python pipeline/run_queries.py
/home/walt/bridge-rna/.venv/bin/python pipeline/make_report.py
```

For a new sample, construct natural `log1p(TPM)` in the exact bundled 15,165-gene order, obtain the unnormalized final-layer arithmetic-mean Bridge embedding with checkpoint `r7hnr92k`, then apply `((z - mean) / scale) @ weight.T + bias` from `results/frozen/frozen_gtex_hallmark_probe.npz`. L2-normalize the resulting 50-vector and multiply it by `work/archs4_hallmark_cosine_index.float32.npy.T`.

```bash
/home/walt/bridge-rna/.venv/bin/python pipeline/query_new_samples.py \\
  --expression /absolute/path/new_samples_log1p_tpm.npy \\
  --sample-ids /absolute/path/sample_ids.txt \\
  --output results/new_query/retrieval.csv --device cuda:0 --top-k 10
```

The expression array must be finite, nonnegative, natural `log1p(TPM)`, and ordered exactly as `results/frozen/frozen_canonical_genes.csv`.

The database has 510,709 human ARCHS4 samples. Large database/index files are deliberately ignored by Git but reproducible from shared read-only assets.
"""; (RESULTS.parent/"README.md").write_text(readme)
    overall=s.to_dict(); d=delta.delta_50_minus_512
    oval=osum.set_index("method").liver_precision_at_10.to_dict(); report=f"""# Final report: Bridge Hallmark functional-state retrieval

## Frozen system and database

The benchmark reused, without retraining, the validated GTEx-only ridge probe and exactly compatible ARCHS4 embeddings. The database contains 510,709 human samples × 50 ordered Hallmark scores. The 512-D cache passed a 40,000-row bitwise compatibility witness. TCGA was evaluation-only.

## TCGA validation

Across representative samples and cohort-centroid queries, mean metadata keyword precision@10 was {overall.get('Hallmark-50 cosine',float('nan')):.3f} in Hallmark space and {overall.get('Bridge-512 cosine',float('nan')):.3f} in Bridge space. The paired mean difference was {d.mean():.3f} (median {d.median():.3f}; Hallmark better for {(d>0).sum()}/{len(d)} queries, tied for {(d==0).sum()}/{len(d)}). This keyword metric is deliberately external to Hallmark labels, but is coarse because ARCHS4 metadata are free text.

## OSDR liver test

The repository-defined cohort contains 112 mouse liver RNA-seq samples (spaceflight FLT and ground-control GC across missions). These were not relabeled as human fatty-liver cases. Across 28 representative/group queries, liver-keyword precision@10 was {oval.get('Hallmark-50 cosine',float('nan')):.3f} for Hallmark retrieval versus {oval.get('Bridge-512 cosine',float('nan')):.3f} for global Bridge retrieval. The Hallmark top hits were predominantly adipose samples, driven chiefly by Unfolded Protein Response, Wnt-beta Catenin Signaling, mTORC1 Signaling, KRAS Signaling Dn, Protein Secretion, and/or Cholesterol Homeostasis; the 512-D baseline overwhelmingly retrieved liver/hepatocyte samples and included a MAFLD liver sample for one RR6-LAR ground-control representative. This is evidence of cross-species/domain-transfer failure for this frozen Hallmark retrieval, not evidence of improved functional matching. Applying the human GTEx probe to mouse is exploratory despite the shared one-to-one vocabulary and species-agnostic checkpoint.

## Scientific conclusion

These results test retrieval behavior, not Hallmark-label recovery. The aggregate answer is negative: 50-D retrieval did not outperform 512-D global retrieval on TCGA and failed the OSDR liver-coherence check. There are narrow TCGA examples where functional retrieval improved metadata coherence—KIRC (+0.30 precision@10), BRCA (+0.10), and LUAD/LUSC cohort centroids (+0.10)—but these heterogeneous qualitative wins do not offset the overall result and require independent biological review. Nothing here establishes batch correction, causal program equivalence, or clinical validity.

## Limitations

- ARCHS4 titles/source/characteristics are incomplete and noisy; keyword coherence undercounts implicit biology.
- The Hallmark scores are frozen linear predictions of GTEx-derived ssGSEA-like targets, not direct pathway measurements.
- Cosine contributions are coordinate-wise decompositions of similarity, not gene-level attribution or mechanism.
- TCGA and OSDR were not used to fit or tune the head, queries, or scoring rule.
"""; (RESULTS.parent/"FINAL_REPORT.md").write_text(report)
    save_json(RESULTS/"final_summary.json",{"tcga_mean_keyword_precision_by_method":overall,"paired_delta_mean":float(d.mean()),"paired_delta_median":float(d.median()),"queries":len(d),"osdr_query_groups":int(oc.query_id.nunique()),"osdr_liver_precision_at_10_by_method":oval})
if __name__=="__main__": main()
