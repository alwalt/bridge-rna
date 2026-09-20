#!/usr/bin/env python3
"""Native-transcriptome sensitivity for matched RR1/RR3 technical replications.

Conventional raw-count edgeR/GSEA only. BridgeRNA defines one comparison
universe but no embeddings or attribution are loaded or recomputed.
"""
from __future__ import annotations

import gzip
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "benchmarks/rr1_rr3_robust_response_comparison"
OUT = BENCH / "results/native_transcriptome_sensitivity"
WORK = BENCH / "work/native_transcriptome_sensitivity"
T4 = ROOT / "benchmarks/library_prep_disentanglement"
MAP = T4 / "results/task4_rr1_rr3_paired_technical_replication/animal_mapping.csv"
EDGER = T4 / "pipeline/run_independent_biological_replication_edger.R"
GMT_ROOT = ROOT / "benchmarks/cross_species_exercise_response/results/per_study_ranked_gsea"
GMT = {"GO:BP": "GO_Biological_Process_2026.gmt", "KEGG": "KEGG_2026.gmt", "REAC": "Reactome_Pathways_2024.gmt"}
COUNT_FILES = {
    "OSD-48": ROOT / "data/osdr/raw/replaced_star_supplementary/GLDS-48_rna_seq_STAR_Unnormalized_Counts_GLbulkRNAseq.csv",
    "OSD-137": ROOT / "data/osdr/raw/GLDS-137_rna_seq_Unnormalized_Counts.csv",
    "OSD-168": ROOT / "data/osdr/raw/GLDS-168_rna_seq_Unnormalized_Counts.csv",
}
COHORTS = ["RR1", "RR3-39", "RR3-40"]
SEED = 43019
N_PERM = 1000


def mouse_symbols() -> dict[str, str]:
    result = {}
    path = ROOT / "data/gencode/gencode.vM38.basic.annotation.gtf.gz"
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            f = line.rstrip().split("\t")
            if len(f) < 9 or f[2] != "gene":
                continue
            gid = re.search(r'gene_id "([^"]+)', f[8]); name = re.search(r'gene_name "([^"]+)', f[8])
            if gid and name:
                result[gid.group(1).split(".")[0]] = name.group(1).upper()
    return result


def prepare() -> tuple[Path, Path, Path]:
    OUT.mkdir(parents=True, exist_ok=True); WORK.mkdir(parents=True, exist_ok=True)
    mapping = pd.read_csv(MAP)
    mapping["contrast_id"] = mapping.cohort.str.replace("-", "_", regex=False) + "__" + mapping.measurement
    mapping[["contrast_id", "sample_id", "condition", "OSD", "cohort", "measurement", "library_preparation"]].to_csv(OUT / "matched_membership.csv", index=False)
    gmap = mouse_symbols(); matrices = []; audits = []
    for osd, group in mapping.groupby("OSD", sort=False):
        raw = pd.read_csv(COUNT_FILES[osd], index_col=0)
        missing = set(group.sample_id) - set(raw.columns)
        if missing:
            raise ValueError(f"{osd}: missing {sorted(missing)}")
        raw = raw[group.sample_id.tolist()]
        ids = raw.index.astype(str).str.split(".").str[0]; symbols = ids.map(gmap)
        audits.append({"OSD": osd, "source_rows": len(raw), "mapped_rows": int(symbols.notna().sum()),
                       "native_unique_symbols": int(symbols.dropna().nunique()), "samples": len(raw.columns),
                       "source_file": str(COUNT_FILES[osd].relative_to(ROOT))})
        raw = raw.loc[symbols.notna()].copy(); raw.index = symbols[symbols.notna()]
        matrices.append(raw.groupby(level=0).sum())
    counts = pd.concat(matrices, axis=1).fillna(0).astype(np.int64)
    full = WORK / "native_symbol_counts.csv.gz"; counts.to_csv(full, compression="gzip")
    vocab = set(pd.read_csv(ROOT / "data/ensembl/canonical_genes.csv").gene_symbol.str.upper())
    restricted = WORK / "bridge_vocab_counts.csv.gz"; counts.loc[counts.index.intersection(sorted(vocab))].to_csv(restricted, compression="gzip")
    pd.DataFrame(audits).to_csv(OUT / "native_gene_audit.csv", index=False)
    memberships = WORK / "memberships.csv"; mapping[["contrast_id", "sample_id", "condition", "OSD"]].to_csv(memberships, index=False)
    return full, restricted, memberships


def run_edger(counts: Path, membership: Path, output: Path) -> pd.DataFrame:
    if not output.exists():
        subprocess.run(["Rscript", str(EDGER), str(counts), str(membership), str(output)], check=True)
    return pd.read_csv(output)


def run_gsea(edge: pd.DataFrame, universe: str, output: Path) -> pd.DataFrame:
    if output.exists():
        return pd.read_parquet(output)
    rows = []; groups = list(edge[edge.tested].groupby("contrast_id", sort=False)); total = len(groups) * len(GMT); done = 0; started = time.time()
    for cid, q in groups:
        rank = q[["gene_symbol", "signed_statistic"]].dropna().sort_values("signed_statistic", ascending=False)
        for source, filename in GMT.items():
            z = gp.prerank(rnk=rank, gene_sets=str(GMT_ROOT / filename), min_size=10, max_size=500,
                           permutation_num=N_PERM, threads=8, seed=SEED, outdir=None, verbose=False).res2d
            z = z.rename(columns={"Term": "pathway", "ES": "es", "NES": "nes", "NOM p-val": "nominal_p", "FDR q-val": "fdr", "Lead_genes": "leading_edge"})
            z["contrast_id"] = cid; z["gene_universe"] = universe; z["source"] = source
            rows.append(z[["contrast_id", "gene_universe", "source", "pathway", "es", "nes", "nominal_p", "fdr", "leading_edge"]])
            done += 1; elapsed = time.time() - started
            print(f"[GSEA heartbeat] {universe} {done}/{total} elapsed={elapsed/60:.1f}m eta={elapsed/max(done,1)*(total-done)/60:.1f}m", flush=True)
    result = pd.concat(rows, ignore_index=True); result.to_parquet(output, index=False); return result


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den else np.nan


def summarize(full_e: pd.DataFrame, vocab_e: pd.DataFrame, full_g: pd.DataFrame, vocab_g: pd.DataFrame) -> None:
    all_e = pd.concat([full_e.assign(gene_universe="native"), vocab_e.assign(gene_universe="bridge_vocab")], ignore_index=True)
    all_g = pd.concat([full_g, vocab_g], ignore_index=True); all_e.to_csv(OUT / "edger_detailed.csv.gz", index=False, compression="gzip"); all_g.to_parquet(OUT / "gsea_detailed.parquet", index=False)
    response = []; pathway = []; status = []
    for universe, e, g in [("native", full_e, full_g), ("bridge_vocab", vocab_e, vocab_g)]:
        for cohort in COHORTS:
            a = e[(e.contrast_id == f"{cohort.replace('-', '_')}__original") & e.tested].set_index("gene_symbol").logFC.dropna()
            b = e[(e.contrast_id == f"{cohort.replace('-', '_')}__remeasure") & e.tested].set_index("gene_symbol").logFC.dropna()
            genes = a.index.intersection(b.index); x = a.loc[genes].to_numpy(); y = b.loc[genes].to_numpy()
            response.append({"cohort": cohort, "gene_universe": universe, "common_tested_genes": len(genes), "cosine": cosine(x, y), "spearman": spearmanr(x, y).statistic,
                             "direction_agreement": float((np.sign(x) == np.sign(y)).mean())})
            ga = g[g.contrast_id == f"{cohort.replace('-', '_')}__original"][["source", "pathway", "nes", "fdr"]]
            gb = g[g.contrast_id == f"{cohort.replace('-', '_')}__remeasure"][["source", "pathway", "nes", "fdr"]]
            m = ga.merge(gb, on=["source", "pathway"], suffixes=("_original", "_remeasure")); robust = (m.fdr_original < .05) & (m.fdr_remeasure < .05) & (np.sign(m.nes_original) == np.sign(m.nes_remeasure)); reversing = (m.fdr_original < .05) & (m.fdr_remeasure < .05) & (np.sign(m.nes_original) != np.sign(m.nes_remeasure))
            pathway.append({"cohort": cohort, "gene_universe": universe, "pathways_compared": len(m), "NES_spearman": spearmanr(m.nes_original, m.nes_remeasure).statistic,
                            "robust_pathways": int(robust.sum()), "reversing_pathways": int(reversing.sum()), "significant_original": int((m.fdr_original < .05).sum()), "significant_remeasurement": int((m.fdr_remeasure < .05).sum())})
            for _, r in m.iterrows():
                status.append({"cohort": cohort, "gene_universe": universe, **r.to_dict(), "protocol_robust": bool((r.fdr_original < .05) and (r.fdr_remeasure < .05) and np.sign(r.nes_original) == np.sign(r.nes_remeasure)), "direction_reversing": bool((r.fdr_original < .05) and (r.fdr_remeasure < .05) and np.sign(r.nes_original) != np.sign(r.nes_remeasure))})
    response = pd.DataFrame(response); pathways = pd.DataFrame(pathway); statuses = pd.DataFrame(status)
    response.to_csv(OUT / "expression_response_reproducibility.csv", index=False); pathways.to_csv(OUT / "pathway_reproducibility_summary.csv", index=False); statuses.to_csv(OUT / "pathway_replication_detailed.csv.gz", index=False, compression="gzip")
    # Validate the 22 primary signed-IG pathways against both conventional universes.
    core = pd.read_csv(BENCH / "results/pathways/independent_conserved_core_full.csv")[["source", "pathway", "module"]].drop_duplicates()
    core["source_key"] = core.source.replace({"Reactome": "REAC"}); core["pathway_key"] = core.pathway.str.casefold(); checks = []
    for _, t in core.iterrows():
        row = {"module": t.module, "source": t.source, "pathway": t.pathway}
        for universe in ["native", "bridge_vocab"]:
            for cohort in COHORTS:
                q = all_g[(all_g.gene_universe == universe) & (all_g.contrast_id.str.startswith(cohort.replace('-', '_'))) & (all_g.source == t.source_key) & (all_g.pathway.str.casefold() == t.pathway_key)]
                for measurement in ["original", "remeasure"]:
                    z = q[q.contrast_id.str.endswith(measurement)]; prefix = f"{universe}_{cohort.replace('-', '_')}_{measurement}"
                    row[f"{prefix}_tested"] = bool(len(z)); row[f"{prefix}_NES"] = z.nes.iloc[0] if len(z) else np.nan; row[f"{prefix}_FDR"] = z.fdr.iloc[0] if len(z) else np.nan
        checks.append(row)
    pd.DataFrame(checks).to_csv(OUT / "signed_ig_22_conventional_support.csv", index=False)
    # Native-only significant pathways, relative to restricted analysis, per measurement.
    native_only = []
    for cohort in COHORTS:
        for measurement in ["original", "remeasure"]:
            cid = f"{cohort.replace('-', '_')}__{measurement}"; n = full_g[(full_g.contrast_id == cid) & (full_g.fdr < .05)]; v = vocab_g[vocab_g.contrast_id == cid]
            merged = n.merge(v[["source", "pathway", "fdr", "nes"]], on=["source", "pathway"], how="left", suffixes=("_native", "_bridge_vocab"))
            q = merged[merged.fdr_bridge_vocab.isna() | merged.fdr_bridge_vocab.ge(.05)].copy(); q.insert(0, "cohort", cohort); q.insert(1, "measurement", measurement); native_only.append(q)
    pd.concat(native_only, ignore_index=True).to_csv(OUT / "native_only_significant_pathways.csv.gz", index=False, compression="gzip")
    # Direct universe-to-universe audit. A "GSEA hit" in the primary compact
    # table is protocol-robust: FDR<0.05 in both measurements with concordant NES.
    universe_rows = []; change_rows = []
    core_keys = set(zip(core.source.replace({"Reactome": "REAC"}), core.pathway.str.casefold()))
    for cohort in COHORTS:
        ck = cohort.replace('-', '_')
        e_by = {"native": full_e, "bridge_vocab": vocab_e}; g_by = {"native": full_g, "bridge_vocab": vocab_g}
        robust_sets = {}; cohort_row = {"Cohort": cohort}
        for universe in ["native", "bridge_vocab"]:
            e = e_by[universe]; g = g_by[universe]
            eo = e[(e.contrast_id == f"{ck}__original") & e.tested]; er = e[(e.contrast_id == f"{ck}__remeasure") & e.tested]
            go = g[g.contrast_id == f"{ck}__original"][["source", "pathway", "nes", "fdr"]]
            gr = g[g.contrast_id == f"{ck}__remeasure"][["source", "pathway", "nes", "fdr"]]
            gm = go.merge(gr, on=["source", "pathway"], suffixes=("_original", "_remeasure"))
            mask = gm.fdr_original.lt(.05) & gm.fdr_remeasure.lt(.05) & (np.sign(gm.nes_original) == np.sign(gm.nes_remeasure))
            robust_sets[universe] = set(zip(gm.loc[mask, "source"], gm.loc[mask, "pathway"].str.casefold()))
            label = "Native" if universe == "native" else "Bridge_vocab"
            cohort_row[f"{label}_genes"] = int(len(set(eo.gene_symbol) & set(er.gene_symbol)))
            cohort_row[f"{label}_DE_genes_original"] = int((eo.FDR < .05).sum())
            cohort_row[f"{label}_DE_genes_remeasurement"] = int((er.FDR < .05).sum())
            cohort_row[f"{label}_GSEA_significant_original"] = int((go.fdr < .05).sum())
            cohort_row[f"{label}_GSEA_significant_remeasurement"] = int((gr.fdr < .05).sum())
            cohort_row[f"{label}_GSEA_hits"] = int(mask.sum())
        native = robust_sets["native"]; restricted = robust_sets["bridge_vocab"]
        cohort_row["Shared_hits"] = len(native & restricted); cohort_row["Gained_by_restriction"] = len(restricted - native); cohort_row["Lost_by_restriction"] = len(native - restricted)
        cohort_row["Pathway_Jaccard"] = len(native & restricted) / len(native | restricted) if native | restricted else np.nan
        cohort_row["Native_IG_core_overlap"] = len(native & core_keys); cohort_row["Bridge_vocab_IG_core_overlap"] = len(restricted & core_keys)
        # Per-measurement correlations and sign reversals between universes.
        reversals = 0
        for measurement in ["original", "remeasure"]:
            a = full_g[full_g.contrast_id == f"{ck}__{measurement}"][["source", "pathway", "nes", "fdr"]]
            b = vocab_g[vocab_g.contrast_id == f"{ck}__{measurement}"][["source", "pathway", "nes", "fdr"]]
            m = a.merge(b, on=["source", "pathway"], suffixes=("_native", "_bridge_vocab"))
            cohort_row[f"NES_Spearman_{measurement}"] = spearmanr(m.nes_native, m.nes_bridge_vocab).statistic
            cohort_row[f"FDR_Spearman_{measurement}"] = spearmanr(m.fdr_native, m.fdr_bridge_vocab).statistic
            reversals += int(((m.fdr_native < .05) & (m.fdr_bridge_vocab < .05) & (np.sign(m.nes_native) != np.sign(m.nes_bridge_vocab))).sum())
        cohort_row["Direction_reversals_between_universes"] = reversals
        universe_rows.append(cohort_row)
        for status_name, keys in [("gained_by_restriction", restricted-native), ("lost_by_restriction", native-restricted), ("shared", native&restricted)]:
            for source, pathway_key in sorted(keys):
                display_name = pd.concat([full_g, vocab_g]).loc[lambda x: (x.source == source) & (x.pathway.str.casefold() == pathway_key), "pathway"].iloc[0]
                change_rows.append({"cohort": cohort, "status": status_name, "source": source, "pathway": display_name, "in_IG_core_22": (source, pathway_key) in core_keys})
    universe_summary = pd.DataFrame(universe_rows)
    universe_summary.to_csv(OUT / "gene_universe_summary.csv", index=False)
    pd.DataFrame(change_rows).to_csv(OUT / "pathway_gained_lost_shared.csv", index=False)
    # Compact plots.
    plt.style.use("seaborn-v0_8-whitegrid"); fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for ax, metric, title in zip(axes, ["cosine", "spearman"], ["FLT−GC response cosine", "FLT−GC response Spearman"]):
        p = response.pivot(index="cohort", columns="gene_universe", values=metric).reindex(COHORTS); p[["native", "bridge_vocab"]].plot.bar(ax=ax, color=["#4C78A8", "#F58518"]); ax.set(title=title, xlabel="", ylabel=metric.capitalize(), ylim=(-1, 1)); ax.tick_params(axis="x", rotation=0); ax.axhline(0, color="#555", lw=.8)
    fig.tight_layout(); fig.savefig(OUT / "expression_reproducibility_native_vs_vocab.png", dpi=300, bbox_inches="tight"); fig.savefig(OUT / "expression_reproducibility_native_vs_vocab.pdf", bbox_inches="tight"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for ax, metric, title in zip(axes, ["robust_pathways", "reversing_pathways"], ["Conventionally robust pathways", "Direction-reversing pathways"]):
        p = pathways.pivot(index="cohort", columns="gene_universe", values=metric).reindex(COHORTS); p[["native", "bridge_vocab"]].plot.bar(ax=ax, color=["#4C78A8", "#F58518"]); ax.set(title=title, xlabel="", ylabel="Pathways"); ax.tick_params(axis="x", rotation=0)
    fig.tight_layout(); fig.savefig(OUT / "pathway_reproducibility_native_vs_vocab.png", dpi=300, bbox_inches="tight"); fig.savefig(OUT / "pathway_reproducibility_native_vs_vocab.pdf", bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9.2, 4.8)); x = np.arange(len(COHORTS)); width = .36
    ax.bar(x-width/2, universe_summary.Native_GSEA_hits, width, label="Native", color="#4C78A8")
    ax.bar(x+width/2, universe_summary.Bridge_vocab_GSEA_hits, width, label="Bridge vocabulary", color="#F58518")
    for i, row in universe_summary.iterrows():
        ax.text(i-width/2, row.Native_GSEA_hits+max(universe_summary.Native_GSEA_hits.max(),1)*.015, str(row.Native_GSEA_hits), ha="center", fontsize=9)
        ax.text(i+width/2, row.Bridge_vocab_GSEA_hits+max(universe_summary.Bridge_vocab_GSEA_hits.max(),1)*.015, str(row.Bridge_vocab_GSEA_hits), ha="center", fontsize=9)
    ax.set(xticks=x, xticklabels=COHORTS, ylabel="Protocol-robust GSEA pathways", title="Pathway recovery is not uniformly increased by vocabulary restriction"); ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(OUT / "robust_gsea_hits_by_gene_universe.png", dpi=300, bbox_inches="tight"); fig.savefig(OUT / "robust_gsea_hits_by_gene_universe.pdf", bbox_inches="tight"); plt.close(fig)
    summary = response.merge(pathways, on=["cohort", "gene_universe"])
    summary.to_csv(OUT / "primary_summary.csv", index=False)
    (OUT / "provenance.json").write_text(json.dumps({"analysis": "matched original/technical-remeasurement conventional native versus Bridge-vocabulary raw-count edgeR and ranked GSEA", "BridgeRNA_or_IG_rerun": False, "membership": str(MAP.relative_to(ROOT)), "GSEA_permutations": N_PERM, "GSEA_seed": SEED, "pathway_resources": GMT}, indent=2) + "\n")
    print(summary.to_string(index=False)); print(f"[complete] {OUT}", flush=True)


def main() -> None:
    full, restricted, membership = prepare()
    full_e = run_edger(full, membership, OUT / "native_edger.csv.gz"); vocab_e = run_edger(restricted, membership, OUT / "bridge_vocab_edger.csv.gz")
    full_g = run_gsea(full_e, "native", OUT / "native_gsea.parquet"); vocab_g = run_gsea(vocab_e, "bridge_vocab", OUT / "bridge_vocab_gsea.parquet")
    summarize(full_e, vocab_e, full_g, vocab_g)


if __name__ == "__main__":
    main()
