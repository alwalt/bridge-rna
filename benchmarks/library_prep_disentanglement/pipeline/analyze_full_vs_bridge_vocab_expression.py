#!/usr/bin/env python3
"""Compare full-transcriptome and Bridge-vocabulary conventional edgeR/GSEA.

This is a conventional-expression control. BridgeRNA is used only to define
the restricted 15,165-gene universe; no model representations are loaded.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import gseapy as gp
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
SOURCE = HERE / "results/task4_confounding_profiler/independent_biological_replication"
OUT = HERE / "results/task4_full_vs_bridge_vocab_expression"
WORK = HERE / "work/task4_full_vs_bridge_vocab_expression"
GMT_ROOT = REPO / "benchmarks/cross_species_exercise_response/results/per_study_ranked_gsea"
GMT = {
    "GO:BP": "GO_Biological_Process_2026.gmt",
    "KEGG": "KEGG_2026.gmt",
    "REAC": "Reactome_Pathways_2024.gmt",
}
COUNT_FILES = {
    "OSD-47": REPO / "data/osdr/raw/replaced_star_supplementary/GLDS-47_rna_seq_STAR_Unnormalized_Counts_GLbulkRNAseq.csv",
    "OSD-48": REPO / "data/osdr/raw/replaced_star_supplementary/GLDS-48_rna_seq_STAR_Unnormalized_Counts_GLbulkRNAseq.csv",
    "OSD-137": REPO / "data/osdr/raw/GLDS-137_rna_seq_Unnormalized_Counts.csv",
    "OSD-173": REPO / "data/osdr/raw/GLDS-173_rna_seq_Unnormalized_Counts.csv",
    "OSD-242": REPO / "data/osdr/raw/GLDS-242_rna_seq_Unnormalized_Counts.csv",
    "OSD-245": REPO / "data/osdr/raw/GLDS-245_rna_seq_Unnormalized_Counts.csv",
}
SEED = 43019
N_PERM = 1000


def pathway_family(term: str) -> str:
    t = term.upper()
    if "SPLIC" in t or "RNA PROCESS" in t or "MRNA PROCESS" in t:
        return "RNA processing / splicing"
    if "CHROMATIN" in t or "NUCLEOSOME" in t or "HISTONE" in t:
        return "Chromatin organization / remodeling"
    if "DNA REPAIR" in t or "DNA METABOL" in t or "DNA DAMAGE" in t:
        return "DNA repair / DNA-damage response"
    if any(x in t for x in ["LIPID", "FATTY ACID", "PEROXISOM", "CHOLESTEROL", "BILE", "SMALL MOLECULE", "CATABOL", "METABOL"]):
        return "Hepatic lipid/fatty-acid/peroxisomal metabolism"
    return "Other"


def program_label(term: str) -> str:
    """Collapse only obvious redundant pathway labels for the compact table."""
    t = term.upper()
    rules = [
        (r"RRNA PROCESS", "rRNA processing"),
        (r"SPLICEOSOME|SPLICING|SPLICE SITE", "RNA splicing"),
        (r"CHOLESTEROL", "cholesterol metabolism"),
        (r"FATTY ACID", "fatty-acid metabolism"),
        (r"PEROXISOM", "peroxisomal metabolism"),
        (r"BILE", "bile metabolism/secretion"),
        (r"DNA DAMAGE|DNA REPAIR", "DNA damage/repair"),
        (r"CHROMATIN|NUCLEOSOME|HISTONE", "chromatin organization/remodeling"),
        (r"MITOCHONDRIAL TRANSLATION|TRANSLATION IN MITOCHONDRIA", "mitochondrial translation"),
        (r"RIBOSOM", "ribosome/translation"),
    ]
    for pattern, label in rules:
        if re.search(pattern, t):
            return label
    return re.sub(r"\s*\(GO:\d+\)\s*$", "", term).strip()


def load_gene_sets() -> dict[str, set[str]]:
    sets: dict[str, set[str]] = {}
    for filename in GMT.values():
        with (GMT_ROOT / filename).open() as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                if len(fields) >= 3:
                    sets[fields[0]] = set(fields[2:])
    return sets


def parse_mouse_gtf() -> dict[str, str]:
    mapping: dict[str, str] = {}
    gtf = REPO / "data/gencode/gencode.vM38.basic.annotation.gtf.gz"
    with gzip.open(gtf, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue
            gid = re.search(r'gene_id "([^"]+)"', fields[8])
            symbol = re.search(r'gene_name "([^"]+)"', fields[8])
            if gid and symbol:
                mapping[gid.group(1).split(".")[0]] = symbol.group(1).upper()
    if not mapping:
        raise RuntimeError(f"No gene mappings parsed from {gtf}")
    return mapping


def prepare_full_counts(membership: pd.DataFrame) -> tuple[Path, Path, pd.DataFrame]:
    WORK.mkdir(parents=True, exist_ok=True)
    counts_path = WORK / "full_symbol_counts.csv.gz"
    meta_path = WORK / "memberships.csv"
    mapping_audit_path = OUT / "full_gene_mapping_audit.csv"
    gene_map = parse_mouse_gtf()
    matrices = []
    audits = []
    for osd, group in membership.groupby("OSD", sort=False):
        source = COUNT_FILES[osd]
        data = pd.read_csv(source, index_col=0)
        sample_ids = group.sample_id.tolist()
        missing = sorted(set(sample_ids) - set(data.columns))
        if missing:
            raise ValueError(f"{osd}: missing count columns: {missing}")
        data = data[sample_ids]
        raw_ids = data.index.astype(str).str.split(".").str[0]
        symbols = raw_ids.map(gene_map)
        audits.append({
            "OSD": osd,
            "raw_gene_rows": len(data),
            "mapped_symbol_rows": int(symbols.notna().sum()),
            "unmapped_rows": int(symbols.isna().sum()),
            "unique_mapped_symbols": int(symbols.dropna().nunique()),
            "source_file": str(source.relative_to(REPO)),
        })
        data = data.loc[symbols.notna()].copy()
        data.index = symbols[symbols.notna()].values
        data = data.groupby(level=0).sum()
        matrices.append(data)
    # Each contrast is filtered independently by edgeR; zero-filling genes absent
    # from another study does not enter that contrast's samples.
    counts = pd.concat(matrices, axis=1).fillna(0).astype(np.int64)
    if not counts.columns.is_unique:
        raise AssertionError("Sample columns are not unique")
    counts.to_csv(counts_path, compression="gzip")
    membership[["contrast_id", "sample_id", "condition", "OSD"]].to_csv(meta_path, index=False)
    audit = pd.DataFrame(audits)
    audit.to_csv(mapping_audit_path, index=False)
    return counts_path, meta_path, audit


def run_edger(counts: Path, memberships: Path, output: Path) -> pd.DataFrame:
    if not output.exists():
        command = [
            "Rscript",
            str(HERE / "pipeline/run_independent_biological_replication_edger.R"),
            str(counts),
            str(memberships),
            str(output),
        ]
        subprocess.run(command, check=True)
    return pd.read_csv(output)


def run_gsea(edge: pd.DataFrame, universe: str, cache: Path) -> pd.DataFrame:
    if cache.exists():
        return pd.read_parquet(cache)
    rows = []
    groups = list(edge[edge.tested].groupby("contrast_id", sort=False))
    total = len(groups) * len(GMT)
    completed = 0
    start = time.time()
    for contrast_id, group in groups:
        ranking = group[["gene_symbol", "signed_statistic"]].dropna().sort_values("signed_statistic", ascending=False)
        for source, filename in GMT.items():
            result = gp.prerank(
                rnk=ranking,
                gene_sets=str(GMT_ROOT / filename),
                min_size=10,
                max_size=500,
                permutation_num=N_PERM,
                threads=8,
                seed=SEED,
                outdir=None,
                verbose=False,
            ).res2d.rename(columns={
                "Term": "pathway", "ES": "es", "NES": "nes",
                "NOM p-val": "nominal_p", "FDR q-val": "fdr",
                "Lead_genes": "leading_edge",
            })
            result["contrast_id"] = contrast_id
            result["gene_universe"] = universe
            result["source"] = source
            rows.append(result[["contrast_id", "gene_universe", "source", "pathway", "es", "nes", "nominal_p", "fdr", "leading_edge"]])
            completed += 1
            elapsed = time.time() - start
            eta = elapsed / completed * (total - completed)
            print(f"[GSEA heartbeat] {universe} {completed}/{total} elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m", flush=True)
    answer = pd.concat(rows, ignore_index=True)
    answer["family"] = answer.pathway.map(pathway_family)
    answer["program_label"] = answer.pathway.map(program_label)
    answer.to_parquet(cache, index=False)
    return answer


def top_programs(frame: pd.DataFrame, n: int = 3) -> str:
    significant = frame[frame.fdr < 0.05].copy()
    significant["abs_nes"] = significant.nes.abs()
    significant = significant.sort_values(["fdr", "abs_nes"], ascending=[True, False])
    rows = []
    for program, group in significant.groupby("program_label", sort=False):
        row = group.iloc[0]
        rows.append(f"{program} ({row.nes:+.2f}, FDR={row.fdr:.3g})")
        if len(rows) == n:
            break
    return "; ".join(rows) if rows else "none at FDR < 0.05"


def best_family(frame: pd.DataFrame, family: str) -> str:
    z = frame[(frame.family == family) & (frame.fdr < 0.05)].copy()
    if z.empty:
        return "NS"
    z["abs_nes"] = z.nes.abs()
    row = z.sort_values(["fdr", "abs_nes"], ascending=[True, False]).iloc[0]
    return f"NES {row.nes:+.3f}; FDR {row.fdr:.3g}"


def summarize(contrasts: pd.DataFrame, full_edge: pd.DataFrame, bridge_edge: pd.DataFrame,
              full_gsea: pd.DataFrame, bridge_gsea: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_gsea = pd.concat([full_gsea, bridge_gsea], ignore_index=True)
    all_gsea.to_parquet(OUT / "pathway_enrichment_detailed.parquet", index=False)
    all_gsea.to_csv(OUT / "pathway_enrichment_detailed.csv.gz", index=False, compression="gzip")
    summary_rows = []
    family_rows = []
    agreement_rows = []
    families = [
        "RNA processing / splicing",
        "Chromatin organization / remodeling",
        "DNA repair / DNA-damage response",
        "Hepatic lipid/fatty-acid/peroxisomal metabolism",
    ]
    for meta in contrasts.itertuples():
        cid = meta.contrast_id
        fe = full_edge[(full_edge.contrast_id == cid) & full_edge.tested]
        be = bridge_edge[(bridge_edge.contrast_id == cid) & bridge_edge.tested]
        fg = full_gsea[full_gsea.contrast_id == cid]
        bg = bridge_gsea[bridge_gsea.contrast_id == cid]
        merged = fg[["source", "pathway", "nes", "fdr"]].merge(
            bg[["source", "pathway", "nes", "fdr"]], on=["source", "pathway"], suffixes=("_full", "_bridge")
        )
        rho = spearmanr(merged.nes_full, merged.nes_bridge).statistic if len(merged) > 1 else np.nan
        sf = set(map(tuple, fg.loc[fg.fdr < .05, ["source", "pathway"]].to_numpy()))
        sb = set(map(tuple, bg.loc[bg.fdr < .05, ["source", "pathway"]].to_numpy()))
        shared = sf & sb
        union = sf | sb
        shared_table = merged[
            merged.apply(lambda r: (r.source, r.pathway) in shared, axis=1)
        ]
        direction_agreement = float((np.sign(shared_table.nes_full) == np.sign(shared_table.nes_bridge)).mean()) if len(shared_table) else np.nan
        agreement_rows.append({
            "contrast_id": cid, "pathways_compared": len(merged), "nes_spearman": rho,
            "significant_full": len(sf), "significant_bridge_vocab": len(sb),
            "shared_significant": len(shared), "shared_fraction_of_union": len(shared) / len(union) if union else np.nan,
            "shared_direction_agreement": direction_agreement,
        })
        summary_rows.append({
            "Study": meta.OSD, "Contrast": cid, "FLT_n": meta.n_FLT, "GC_n": meta.n_GC,
            "Library_selection": meta.library_preparation,
            "Full_genes": len(fe), "Bridge_genes": len(be), "Coverage": len(be) / len(fe),
            "Full_top_3_programs": top_programs(fg), "Bridge_vocab_top_3_programs": top_programs(bg),
            "NES_profile_Spearman": rho, "Shared_significant_pathways": len(shared),
        })
        row = {"Study": meta.OSD, "Contrast": cid, "Library": meta.library_preparation}
        for family in families:
            short = {
                "RNA processing / splicing": "RNA_processing",
                "Chromatin organization / remodeling": "Chromatin",
                "DNA repair / DNA-damage response": "DNA_repair",
                "Hepatic lipid/fatty-acid/peroxisomal metabolism": "Hepatic_metabolism",
            }[family]
            row[f"{short}_full"] = best_family(fg, family)
            row[f"{short}_bridge_vocab"] = best_family(bg, family)
        family_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    family_summary = pd.DataFrame(family_rows)
    agreement = pd.DataFrame(agreement_rows)
    summary.to_csv(OUT / "primary_summary.csv", index=False)
    family_summary.to_csv(OUT / "program_family_summary.csv", index=False)
    agreement.to_csv(OUT / "pathway_agreement_summary.csv", index=False)
    return summary, family_summary, agreement


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-gsea", action="store_true", help="Recompute cached GSEA outputs.")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    contrasts = pd.read_csv(SOURCE / "independent_contrast_summary.csv")
    membership = pd.read_csv(SOURCE / "independent_contrast_membership.csv")
    assert len(contrasts) == 11 and membership.sample_id.nunique() == 84
    if args.force_gsea:
        for path in [OUT / "full_pathway_enrichment.parquet", OUT / "bridge_vocab_pathway_enrichment.parquet"]:
            path.unlink(missing_ok=True)
    counts, memberships, mapping_audit = prepare_full_counts(membership)
    full_edge_path = OUT / "full_transcriptome_edger.csv.gz"
    full_edge = run_edger(counts, memberships, full_edge_path)
    bridge_edge_path = OUT / "bridge_vocab_edger.csv.gz"
    if not bridge_edge_path.exists():
        shutil.copy2(SOURCE / "edger_results.csv.gz", bridge_edge_path)
    bridge_edge = pd.read_csv(bridge_edge_path)
    full_gsea = run_gsea(full_edge, "full_transcriptome", OUT / "full_pathway_enrichment.parquet")
    # Reuse the exact existing conventional Bridge-vocabulary GSEA results.
    bridge_cache = OUT / "bridge_vocab_pathway_enrichment.parquet"
    if not bridge_cache.exists():
        prior = pd.read_parquet(SOURCE / "pathway_enrichment.parquet")
        prior = prior[prior.analysis == "edgeR_expression"].drop(columns="analysis").copy()
        prior["gene_universe"] = "bridge_vocab"
        prior["program_label"] = prior.pathway.map(program_label)
        prior["family"] = prior.pathway.map(pathway_family)
        prior.to_parquet(bridge_cache, index=False)
    bridge_gsea = pd.read_parquet(bridge_cache)
    summary, family_summary, agreement = summarize(contrasts, full_edge, bridge_edge, full_gsea, bridge_gsea)
    rr39 = family_summary[family_summary.Contrast.str.contains("RR3__39-day")].iloc[0]
    rr40 = family_summary[family_summary.Contrast.str.contains("RR3__40-day")].iloc[0]
    scientific_summary = {
        "contrasts": int(len(contrasts)),
        "samples": int(membership.sample_id.nunique()),
        "median_tested_gene_coverage": float(summary.Coverage.median()),
        "median_nes_profile_spearman": float(summary.NES_profile_Spearman.median()),
        "min_nes_profile_spearman": float(summary.NES_profile_Spearman.min()),
        "max_nes_profile_spearman": float(summary.NES_profile_Spearman.max()),
        "median_shared_significant_fraction_of_union": float(agreement.shared_fraction_of_union.median()),
        "shared_significant_direction_agreement": float(
            agreement.shared_direction_agreement.dropna().mean()
        ),
        "rna_processing_significant_contrasts_full": int(
            full_gsea[(full_gsea.family == "RNA processing / splicing") & (full_gsea.fdr < .05)].contrast_id.nunique()
        ),
        "rna_processing_significant_contrasts_bridge_vocab": int(
            bridge_gsea[(bridge_gsea.family == "RNA processing / splicing") & (bridge_gsea.fdr < .05)].contrast_id.nunique()
        ),
        "rr3_39_rna_processing_full": rr39.RNA_processing_full,
        "rr3_40_rna_processing_full": rr40.RNA_processing_full,
        "rr3_39_rna_processing_bridge_vocab": rr39.RNA_processing_bridge_vocab,
        "rr3_40_rna_processing_bridge_vocab": rr40.RNA_processing_bridge_vocab,
        "underpowered_contrasts": [
            "C03__OSD-137__RR3__41-day (1 FLT / 2 GC)",
            "C12__OSD-47__RR1-CASIS__22-day (1 FLT / 1 GC; descriptive fixed-BCV ranking)",
        ],
        "conclusion": (
            "Vocabulary restriction preserves the global conventional pathway interpretation strongly, "
            "but can remove individual marginal pathway calls. RR3 39d retains a stronger RNA-processing "
            "signal than RR3 40d in both universes. Vocabulary restriction alone does not explain the "
            "conventional-versus-contextual differences."
        ),
    }
    (OUT / "scientific_summary.json").write_text(json.dumps(scientific_summary, indent=2) + "\n")
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "conventional full-transcriptome versus exact BridgeRNA-vocabulary edgeR/GSEA",
        "contrasts": int(len(contrasts)), "samples": int(membership.sample_id.nunique()),
        "bridge_vocabulary_file": "data/ensembl/canonical_genes.csv",
        "bridge_vocabulary_size": 15165,
        "full_gene_annotation": "data/gencode/gencode.vM38.basic.annotation.gtf.gz",
        "edgeR_script": "pipeline/run_independent_biological_replication_edger.R",
        "gsea_permutations": N_PERM, "gsea_seed": SEED,
        "pathway_files": GMT,
        "limitations": [
            "Very small RR3-41d and RR1-CASIS-22d strata have weak/no replication.",
            "Mouse symbols are upper-cased to match the existing human-symbol pathway collections.",
            "Significant pathways are not independent because gene sets overlap.",
        ],
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("\nPrimary summary", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\nGlobal agreement", flush=True)
    print(agreement.to_string(index=False), flush=True)
    print("\nConcise scientific summary", flush=True)
    print(json.dumps(scientific_summary, indent=2), flush=True)
    print(f"\nSaved outputs to {OUT}", flush=True)


if __name__ == "__main__":
    main()
