#!/usr/bin/env python3
"""Animal-matched STS-135 liver metabolomics analysis (OSD-173 ↔ OSD-108)."""
from __future__ import annotations
import json, re
from datetime import datetime, timezone
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; RESULTS = ROOT / "results"
OUT = RESULTS / "task3_sts135_metabolomics"; WORK = ROOT / "work" / "task3_sts135_metabolomics" / "OSD-108"
OUT.mkdir(parents=True, exist_ok=True)

def animal_code(text):
    m = re.search(r"(?:^|_)([FG]\d+)(?:$|_)", str(text)); return m.group(1) if m else ""

def build_linkage():
    s = pd.read_csv(WORK / "isa/s_OSD-108.txt", sep="\t", dtype=str).fillna("")
    a = pd.read_csv(WORK / "isa/a_OSD-108_metabolite-profiling_mass-spectrometry_Thermo-Finnigan Mass Spectrometer.txt", sep="\t", dtype=str).fillna("")
    lookup = s.merge(a[["Sample Name", "Extract Name", "MS Assay Name"]], on="Sample Name", how="left")
    parts = lookup["Source Name"].str.extract(r"(Flight|AEM Control)\s+(\d+)")
    lookup["animal"] = [("F" if x == "Flight" else "G") + y if x and y else "" for x,y in parts.fillna("").itertuples(index=False)]
    m = pd.read_csv(RESULTS / "task3b_contrast_sample_membership.csv"); m = m[m.OSD.eq("OSD-173")].copy(); m["animal"] = m.sample_id.map(animal_code)
    rows=[]
    for x in m.itertuples(index=False):
        hit=lookup[lookup.animal.eq(x.animal)]
        rows.append({"contrast_id":x.contrast_id,"rna_sample_id":x.sample_id,"rna_animal":x.animal,"rna_condition":x.condition,"metabolomics_OSD":"OSD-108",
          "metabolomics_source_name":hit["Source Name"].iloc[0] if len(hit) else "","metabolomics_sample_name":hit["Sample Name"].iloc[0] if len(hit) else "",
          "metabolomics_extract_name":hit["Extract Name"].iloc[0] if len(hit) else "","exact_animal_match":bool(len(hit)),
          "evidence_level":"A_exact_animal" if len(hit) else "B_same_cohort_only","included_primary":bool(len(hit)),
          "linkage_basis":"OSD-173 animal token equals OSD-108 ISA Source Name animal number" if len(hit) else "No identical animal number in OSD-108 ISA source table"})
    out=pd.DataFrame(rows); out.to_csv(OUT/"rna_metabolomics_animal_matching.csv",index=False); lookup.to_csv(OUT/"osd108_authoritative_sample_lookup.csv",index=False); return out

def load_metabolites():
    path=WORK/"GLDS-108_metabolomics_ScaledImpData_LIVER.csv"; h=pd.read_csv(path,encoding="cp1252",header=None,nrows=8)
    mapping=dict(zip(h.iloc[2,12:].tolist(),h.iloc[1,12:].tolist()))
    mapping.update({"G"+k[1:]:v for k,v in list(mapping.items()) if str(k).startswith("A")})
    names=["pathway_sort_order","metabolite","super_pathway","sub_pathway","compound_id","platform","ri","mass","cas","pubchem","kegg","hmdb"]
    d=pd.read_csv(path,encoding="cp1252",skiprows=8,header=None,names=names+h.iloc[1,12:].tolist())
    return d,mapping

def process_label(x):
    text=f"{x.metabolite} {x.super_pathway} {x.sub_pathway}".lower()
    if "corticosterone" in text:return "Corticosterone"
    if "bile" in str(x.sub_pathway).lower():return "Bile acids"
    if re.search(r"glutathione|oxidative|redox|tocopherol",text):return "Glutathione / oxidative stress"
    if re.search(r"retinol|retinoid|vitamin a",text):return "Retinol-related metabolism"
    if re.search(r"glycol|gluconeo|carbohydrate|tca cycle|krebs|pentose|fructose|galactose",text):return "Energy / carbohydrate metabolism"
    if str(x.super_pathway).lower()=="lipid" or re.search(r"fatty acid|lipid|acyl|glycerol|phospholipid",text):return "Fatty-acid / lipid metabolism"
    return "Other measured metabolism"

def calculate_effects(linkage):
    d,mapping=load_metabolites(); exact=linkage[linkage.exact_animal_match]
    ids=exact.groupby("rna_condition").rna_animal.apply(list).to_dict(); primary={k:[mapping[a] for a in v] for k,v in ids.items()}
    cohort={"FLT":[mapping[x] for x in ["F52","F58","F60","F64","F66","F74"]],"GC":[mapping[x] for x in ["A18","A28","A30","A32","A36","A44"]]}
    rows=[]
    for analysis,groups in [("exact_animal_primary",primary),("same_cohort_sensitivity",cohort)]:
      for _,x in d.iterrows():
        f=pd.to_numeric(x[groups["FLT"]],errors="coerce").to_numpy(float); g=pd.to_numeric(x[groups["GC"]],errors="coerce").to_numpy(float); mf,mg=np.nanmean(f),np.nanmean(g)
        rows.append({"analysis":analysis,"metabolite":x.metabolite,"super_pathway":x.super_pathway,"sub_pathway":x.sub_pathway,"platform":x.platform,"kegg":x.kegg,"hmdb":x.hmdb,
          "process":process_label(x),"n_FLT":np.isfinite(f).sum(),"n_GC":np.isfinite(g).sum(),"mean_FLT":mf,"mean_GC":mg,"FLT_minus_GC":mf-mg,
          "fold_change_FLT_over_GC":mf/mg if mg>0 else np.nan,"log2_fold_change":np.log2(mf/mg) if mf>0 and mg>0 else np.nan,
          "FLT_individual_values":" | ".join(f"{v:.8g}" for v in f),"GC_individual_values":" | ".join(f"{v:.8g}" for v in g)})
    e=pd.DataFrame(rows); e["abs_log2_fold_change_rank"]=e.groupby("analysis").log2_fold_change.transform(lambda s:s.abs().rank(method="min",ascending=False))
    e.to_csv(OUT/"metabolite_effects.csv",index=False); e.to_parquet(OUT/"metabolite_effects.parquet",index=False)
    s=(e.groupby(["analysis","process"],sort=False).agg(metabolites=("metabolite","size"),median_log2FC=("log2_fold_change","median"),mean_log2FC=("log2_fold_change","mean"),
      increased=("log2_fold_change",lambda z:int((z>0).sum())),decreased=("log2_fold_change",lambda z:int((z<0).sum()))).reset_index())
    s.to_csv(OUT/"metabolic_process_summary.csv",index=False); return e,s

def bridge_concordance(summary):
    d=pd.read_parquet(RESULTS/"task3d_mode_ig/signed_expression/signed_pathway_changes_by_contrast.parquet"); sts=d[d.contrast_id.str.startswith("C07")]
    pats={"Fatty-acid / lipid metabolism":r"fatty acid|lipid|cholesterol","Bile acids":r"bile","Glutathione / oxidative stress":r"toxic|peroxisome|oxidative|detox|glutathione",
      "Retinol-related metabolism":r"retinol|retinoid","Energy / carbohydrate metabolism":r"catabolic|small molecule|oxoacid|carbon|energy|carbohydrate","Corticosterone":r"corticosterone|steroid"}
    p=summary[summary.analysis.eq("exact_animal_primary")].set_index("process"); rows=[]
    for process,pat in pats.items():
      h=sts[sts.term.str.contains(pat,case=False,regex=True,na=False)]; mv=p.loc[process,"median_log2FC"] if process in p.index else np.nan; bv=h.median_member_expression_change.median() if len(h) else np.nan
      same=np.isfinite(mv) and np.isfinite(bv) and np.sign(mv)==np.sign(bv)
      rows.append({"process":process,"metabolomics_metabolites":int(p.loc[process,"metabolites"]) if process in p.index else 0,"metabolomics_median_log2FC":mv,
       "BridgeRNA_supporting_terms":" | ".join(h.term),"BridgeRNA_median_member_expression_change":bv,"direction_compatible":same if np.isfinite(bv) else pd.NA,
       "interpretation":"both measured modalities support same direction" if same else ("both implicated, directions differ" if len(h) else "no direct saved BridgeRNA counterpart")})
    out=pd.DataFrame(rows);out.to_csv(OUT/"bridgerna_metabolomics_concordance.csv",index=False);sts.to_csv(OUT/"sts135_bridgerna_pathway_reference.csv",index=False);return out

def figures(e,s):
    p=e[e.analysis.eq("exact_animal_primary")].nsmallest(25,"abs_log2_fold_change_rank")
    p=p.sort_values("log2_fold_change");fig,ax=plt.subplots(figsize=(9,8),layout="constrained");ax.barh(p.metabolite,p.log2_fold_change,color=np.where(p.log2_fold_change>=0,"#b2182b","#2166ac"));ax.axvline(0,color="black",lw=.8)
    ax.set(xlabel="log2(mean FLT / mean GC)",title="STS-135 liver metabolomics: strongest exact-animal effects\nFLT F58/F64 versus GC G44")
    for ext in ["png","pdf"]:fig.savefig(OUT/f"strongest_metabolite_changes.{ext}",dpi=400,bbox_inches="tight")
    plt.close(fig)
    p=s[s.analysis.eq("exact_animal_primary")].sort_values("median_log2FC");fig,ax=plt.subplots(figsize=(8.5,4.8),layout="constrained");ax.barh(p.process,p.median_log2FC,color=np.where(p.median_log2FC>=0,"#b2182b","#2166ac"));ax.axvline(0,color="black",lw=.8)
    ax.set(xlabel="Median metabolite log2 fold change",title="Descriptive metabolic-process response")
    for ext in ["png","pdf"]:fig.savefig(OUT/f"metabolic_process_summary.{ext}",dpi=400,bbox_inches="tight")
    plt.close(fig)

def main():
    linkage=build_linkage();effects,summary=calculate_effects(linkage);concordance=bridge_concordance(summary);figures(effects,summary)
    a=effects[effects.analysis.eq("exact_animal_primary")][["metabolite","log2_fold_change"]];b=effects[effects.analysis.eq("same_cohort_sensitivity")][["metabolite","log2_fold_change"]]
    rho=a.merge(b,on="metabolite").iloc[:,1:].corr(method="spearman").iloc[0,1]
    prov={"created_utc":datetime.now(timezone.utc).isoformat(),"RNA_OSD":"OSD-173","metabolomics_OSD":"OSD-108","primary_FLT_animals":["F58","F64"],"primary_GC_animals":["G44"],
      "unmatched_RNA_animal":["G40"],"metabolites":int(len(a)),"data":"Metabolon median-scaled, minimum-imputed released table","primary_inference":"descriptive; n=2 FLT vs n=1 GC",
      "same_cohort_sensitivity_Spearman_with_primary":rho,"classification":"C_inconclusive_due_to_small_exact_matched_sample_size"}
    (OUT/"provenance.json").write_text(json.dumps(prov,indent=2));print(linkage[["rna_animal","rna_condition","metabolomics_sample_name","exact_animal_match","evidence_level"]].to_string(index=False))
    print("\n",summary[summary.analysis.eq("exact_animal_primary")].to_string(index=False));print("\n",concordance.to_string(index=False));print(f"\nPrimary/cohort rho={rho:.3f}\nConclusion: C — inconclusive")
if __name__=="__main__":main()
