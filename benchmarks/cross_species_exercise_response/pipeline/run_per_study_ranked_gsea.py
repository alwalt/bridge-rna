#!/usr/bin/env python3
"""Preranked GSEA of existing per-study edgeR results (no DE/model inference)."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.fm_embed.species import load_mouse_to_human_symbol_map

OUT = HERE / "results/per_study_ranked_gsea"
OUT.mkdir(parents=True, exist_ok=True)
ORDER = ["GSE108643", "GSE86931", "GSE126962", "GSE132520", "GSE151066", "GSE71972", "GSE87748", "GSE97718"]
PATTERN = {**{g: "A" for g in ORDER[:4]}, "GSE151066": "Intermediate", **{g: "B" for g in ORDER[5:]}}
SPECIES = {g: ("human" if g in {"GSE108643", "GSE86931", "GSE151066", "GSE71972", "GSE87748"} else "mouse") for g in ORDER}
LIBRARIES = {
    "GO:BP": "GO_Biological_Process_2026",
    "KEGG": "KEGG_2026",
    "REAC": "Reactome_Pathways_2024",
}


def edge_rank(gse: str) -> pd.DataFrame:
    species = SPECIES[gse]
    path = HERE / f"results/full_transcriptome_de/{species}_{gse}_full_de.parquet"
    x = pd.read_parquet(path).query("tested").copy()
    # Signed sqrt(QLF) preserves edgeR direction while using its formal test statistic.
    x["rank_score"] = np.sign(x.log2_fold_change) * np.sqrt(x.quasi_likelihood_f.clip(lower=0))
    if species == "mouse":
        mapping = load_mouse_to_human_symbol_map(ROOT / "data/ensembl/orthologs_one2one.txt")
        x["gsea_gene"] = x.gene_id.astype(str).str.split(".").str[0].map(mapping)
    else:
        x["gsea_gene"] = x.gene_symbol
        bad = x.gsea_gene.astype(str).str.match(r"^ENSG\d+") | x.gsea_gene.isna()
        x.loc[bad, "gsea_gene"] = np.nan
    x = x.dropna(subset=["gsea_gene", "rank_score"])
    x = x.sort_values("rank_score", key=lambda s: s.abs(), ascending=False).drop_duplicates("gsea_gene")
    return x[["gsea_gene", "rank_score"]].sort_values("rank_score", ascending=False)


def get_libraries() -> tuple[dict[str, dict[str, list[str]]], list[dict]]:
    sets, provenance = {}, []
    for source, name in LIBRARIES.items():
        gene_sets = gp.get_library(name=name, organism="Human")
        sets[source] = gene_sets
        payload = "\n".join(f"{term}\t" + "\t".join(sorted(genes)) for term, genes in sorted(gene_sets.items()))
        (OUT / f"{name}.gmt").write_text(payload + "\n")
        provenance.append({"source": source, "library": name, "terms": len(gene_sets), "sha256": hashlib.sha256(payload.encode()).hexdigest()})
    return sets, provenance


def run_gsea() -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    libraries, library_provenance = get_libraries()
    results, ranks = [], []
    total = len(ORDER) * len(libraries)
    done = 0
    for gse in ORDER:
        rank = edge_rank(gse)
        ranks.append({"GSE": gse, "species": SPECIES[gse], "pattern": PATTERN[gse], "tested_genes": len(rank)})
        for source, gene_sets in libraries.items():
            done += 1
            print(f"[GSEA {done}/{total}] {gse} {source}: {len(rank):,} ranked genes", flush=True)
            pre = gp.prerank(rnk=rank, gene_sets=gene_sets, min_size=10, max_size=500,
                             permutation_num=1000, weight=1.0, ascending=False,
                             threads=8, seed=20260903, outdir=None, verbose=False)
            z = pre.res2d.rename(columns={"NOM p-val": "nominal_p", "FDR q-val": "fdr", "FWER p-val": "fwer_p", "Lead_genes": "leading_edge"})
            z = z.rename(columns={"Term": "pathway", "ES": "es", "NES": "nes"})
            z["GSE"], z["species"], z["pattern"], z["source"] = gse, SPECIES[gse], PATTERN[gse], source
            results.append(z[["GSE", "species", "pattern", "source", "pathway", "es", "nes", "nominal_p", "fdr", "fwer_p", "leading_edge"]])
    return pd.concat(results, ignore_index=True), pd.DataFrame(ranks), library_provenance


def recurrence_table(result: pd.DataFrame) -> pd.DataFrame:
    sig = result[result.fdr < .05].copy()
    groups = {
        "human": [g for g in ORDER if SPECIES[g] == "human"],
        "mouse": [g for g in ORDER if SPECIES[g] == "mouse"],
        "Pattern A": [g for g in ORDER if PATTERN[g] == "A"],
        "Pattern B": [g for g in ORDER if PATTERN[g] == "B"],
    }
    rows = []
    for group, studies in groups.items():
        for (source, pathway), z in sig[sig.GSE.isin(studies)].groupby(["source", "pathway"]):
            if z.GSE.nunique() >= 2:
                rows.append({"group": group, "source": source, "pathway": pathway, "significant_studies": z.GSE.nunique(),
                             "eligible_studies": len(studies), "median_nes": z.nes.median(), "direction_consistent": (z.nes.gt(0).all() or z.nes.lt(0).all()),
                             "studies": ";".join(g for g in studies if g in set(z.GSE))})
    # Cross-species recurrence means at least one significant human and one significant mouse study.
    for (source, pathway), z in sig.groupby(["source", "pathway"]):
        hs, ms = z[z.species == "human"], z[z.species == "mouse"]
        if not hs.empty and not ms.empty:
            rows.append({"group": "human_and_mouse", "source": source, "pathway": pathway, "significant_studies": z.GSE.nunique(),
                         "eligible_studies": 8, "median_nes": z.nes.median(), "direction_consistent": (z.nes.gt(0).all() or z.nes.lt(0).all()),
                         "studies": ";".join(g for g in ORDER if g in set(z.GSE))})
    return pd.DataFrame(rows).sort_values(["group", "significant_studies", "source", "pathway"], ascending=[True, False, True, True])


def select_heatmap_terms(result: pd.DataFrame, recurring: pd.DataFrame, cap: int = 42) -> list[tuple[str, str]]:
    keys = set(map(tuple, recurring[["source", "pathway"]].drop_duplicates().to_numpy()))
    sig = result[result.fdr < .05]
    # Ensure study-specific conclusions remain visible even when not recurrent.
    for _, z in sig.groupby("GSE"):
        keys.update(map(tuple, z.nsmallest(2, "fdr")[["source", "pathway"]].to_numpy()))
    score = (sig.assign(key=list(zip(sig.source, sig.pathway)))
             .groupby("key").agg(studies=("GSE", "nunique"), strength=("nes", lambda x: x.abs().max())).reset_index())
    ordered = score[score.key.isin(keys)].sort_values(["studies", "strength"], ascending=False).key.tolist()[:cap]
    return ordered


def plot_heatmap(result: pd.DataFrame, keys: list[tuple[str, str]]) -> None:
    label = {k: k[0] + " | " + re.sub(r" \(GO:\d+\)$", "", k[1]) for k in keys}
    z = result.assign(key=list(zip(result.source, result.pathway)))
    nes = z[z.key.isin(keys)].pivot(index="GSE", columns="key", values="nes").reindex(index=ORDER, columns=keys)
    fdr = z[z.key.isin(keys)].pivot(index="GSE", columns="key", values="fdr").reindex(index=ORDER, columns=keys)
    annot = np.where(fdr.to_numpy() < .05, "●", "")
    fig, ax = plt.subplots(figsize=(max(15, .43 * len(keys)), 6.4), layout="constrained")
    im = ax.imshow(nes.to_numpy(float), cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
    ax.set_xticks(range(len(keys)))
    ax.set_yticks(range(len(ORDER)))
    for i in range(len(ORDER)):
        for j in range(len(keys)):
            if annot[i, j]: ax.text(j, i, annot[i, j], ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=.8, pad=.015, label="Normalized enrichment score (NES)")
    ax.set_xticklabels([label[k] for k in keys], rotation=62, ha="right", fontsize=7)
    ax.set_yticklabels([f"{g}  [{PATTERN[g]}; {SPECIES[g]}]" for g in ORDER], rotation=0)
    ax.set(xlabel="Pathway (● = FDR < 0.05)", ylabel="Study", title="Per-study edgeR preranked GSEA: Pattern A → intermediate → Pattern B")
    fig.savefig(OUT / "per_study_ranked_gsea_nes_heatmap.png", dpi=400, bbox_inches="tight")
    fig.savefig(OUT / "per_study_ranked_gsea_nes_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def compare_ig(result: pd.DataFrame) -> pd.DataFrame:
    ig = pd.read_csv(HERE / "results/final_synthesis/synthesis_enrichment_terms.csv")
    ig = ig[ig.column.isin(["Human IG", "Mouse IG", "Conserved H-M IG"])].copy()
    rows = []
    for pattern, axis in [("A", "Pattern A"), ("B", "Pattern B")]:
        g = result[(result.pattern == pattern) & (result.fdr < .05)]
        for column, q in ig[ig.pattern == axis].groupby("column"):
            gkeys = set(zip(g.source, g.pathway)); ikeys = set(zip(q.source, q.name))
            rows.append({"pattern": pattern, "IG_gene_set": column, "significant_GSEA_terms": len(gkeys), "significant_IG_ORA_terms": len(ikeys),
                         "exact_shared_terms": len(gkeys & ikeys), "shared_terms": "; ".join(sorted(name for _, name in gkeys & ikeys))})
    return pd.DataFrame(rows)


def main() -> None:
    result, ranks, libraries = run_gsea()
    result["significant_fdr_0_05"] = result.fdr < .05
    result.to_csv(OUT / "per_study_ranked_gsea_full.csv", index=False)
    ranks.to_csv(OUT / "ranking_input_summary.csv", index=False)
    recurring = recurrence_table(result)
    recurring.to_csv(OUT / "recurring_pathways.csv", index=False)
    selected = select_heatmap_terms(result, recurring)
    plot_heatmap(result, selected)
    heatmap_table = result.assign(key=list(zip(result.source, result.pathway)))
    heatmap_table = heatmap_table[heatmap_table.key.isin(selected)].drop(columns="key")
    heatmap_table.to_csv(OUT / "heatmap_pathway_values.csv", index=False)
    comparison = compare_ig(result)
    comparison.to_csv(OUT / "gsea_vs_ig_exact_term_overlap.csv", index=False)
    summary = (result.groupby(["GSE", "species", "pattern", "source"], as_index=False)
               .agg(tested_pathways=("pathway", "size"), significant_pathways=("significant_fdr_0_05", "sum"),
                    positive_significant=("nes", lambda x: int(((x > 0) & (result.loc[x.index, "fdr"] < .05)).sum())),
                    negative_significant=("nes", lambda x: int(((x < 0) & (result.loc[x.index, "fdr"] < .05)).sum()))))
    summary.to_csv(OUT / "per_study_gsea_summary.csv", index=False)
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(), "edgeR_rerun": False,
                  "ranking": "sign(log2_fold_change) * sqrt(quasi_likelihood_f)", "genes": "all edgeR-tested genes with a unique usable human symbol; mouse genes mapped through one-to-one orthology",
                  "gsea": {"implementation": f"gseapy {gp.__version__}", "permutations": 1000, "seed": 20260903, "min_size": 10, "max_size": 500, "weight": 1.0, "fdr_threshold": .05},
                  "libraries": libraries, "study_order": ORDER, "patterns": PATTERN,
                  "recurrence": "at least two FDR-significant studies per named group; human_and_mouse requires >=1 significant study of each species",
                  "IG_comparison": "exact native pathway-name overlap with existing ORA; GSEA and top-gene ORA answer different questions"}
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("\nPer-study summary\n", summary.to_string(index=False))
    print("\nRecurring pathways\n", recurring.head(50).to_string(index=False))
    print("\nExact-term comparison with IG ORA\n", comparison.to_string(index=False))


if __name__ == "__main__":
    main()
