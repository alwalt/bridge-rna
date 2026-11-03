#!/usr/bin/env python3
"""Audit public LINCS Level-5 releases before selecting a perturbation matrix."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
META = HERE / "work/lincs_metadata"
OUT = HERE / "results"
CANONICAL = Path("/home/walt/bridge-rna/data/ensembl/canonical_genes.csv")
CHEMBL = HERE.parents[0] / "deweerd_replication/results/evaluation2_expanded/chembl37_broad_drug_target_counts.csv"

PUBLISHED = {
    "MS": ["muromonab", "ibrutinib", "daclizumab", "zanubrutinib", "alemtuzumab"],
    "Crohn": ["zinc", "zinc acetate", "dilmapimod", "glucosamine", "vx-702"],
    "SLE": ["gemcitabine", "enzastaurin", "sunitinib", "fostamatinib", "cladribine"],
}
RADIATION = ["pentoxifylline", "sitagliptin", "dipyridamole", "omipalisib", "dactolisib",
             "rg-547", "roniciclib", "afatinib", "trastuzumab", "molibresib", "jq1",
             "ocriplasmin", "collagenase clostridium histolyticum", "luteolin", "resveratrol"]


def norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def path_for(accession, kind):
    matches = list((META / accession).glob(f"*{kind}*.gz"))
    if len(matches) != 1:
        raise RuntimeError((accession, kind, matches))
    return matches[0]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    canonical = set(pd.read_csv(CANONICAL).gene_symbol.astype(str))
    chembl = pd.read_csv(CHEMBL)
    chembl_names = set(chembl.loc[chembl.target_count.gt(10), "drug_name"].map(norm))
    audit_rows, coverage_rows, context_rows = [], [], []
    hashes = {}
    for accession in ["GSE70138", "GSE92742"]:
        gene_path, sig_path, inst_path, pert_path, cell_path = [path_for(accession, x) for x in
                                                                ["gene_info", "sig_info", "inst_info", "pert_info", "cell_info"]]
        genes = pd.read_csv(gene_path, sep="\t", low_memory=False)
        sig = pd.read_csv(sig_path, sep="\t", low_memory=False)
        inst = pd.read_csv(inst_path, sep="\t", low_memory=False)
        pert = pd.read_csv(pert_path, sep="\t", low_memory=False)
        cells = pd.read_csv(cell_path, sep="\t", low_memory=False)
        compounds = sig[sig.pert_type.eq("trt_cp")].copy()
        controls = sig[sig.pert_type.str.startswith("ctl", na=False)].copy()
        compounds["normalized_name"] = compounds.pert_iname.map(norm)
        available = set(compounds.normalized_name)
        distil_reps = compounds.distil_id.fillna("").map(lambda x: len(str(x).split("|")) if x else 0)
        measured = set(genes.loc[genes.pr_is_lm.eq(1), "pr_gene_symbol"].astype(str))
        bing = set(genes.loc[genes.pr_is_bing.eq(1), "pr_gene_symbol"].astype(str))
        audit_rows.append({
            "accession": accession, "level5_signatures": len(sig), "compound_signatures": len(compounds),
            "unique_compounds": compounds.pert_id.nunique(), "unique_compound_names": compounds.pert_iname.nunique(),
            "compound_cell_lines": compounds.cell_id.nunique(), "all_cell_metadata_rows": len(cells),
            "compound_doses": compounds.pert_idose.nunique(), "compound_timepoints": compounds.pert_itime.nunique(),
            "median_distil_replicates": distil_reps.median(), "min_distil_replicates": distil_reps.min(),
            "max_distil_replicates": distil_reps.max(), "vehicle_control_signatures": len(controls),
            "matrix_genes": len(genes), "landmark_genes": len(measured), "best_inferred_genes": len(bing),
            "canonical_overlap_all": len(set(genes.pr_gene_symbol.astype(str)) & canonical),
            "canonical_overlap_landmark": len(measured & canonical), "canonical_overlap_bing": len(bing & canonical),
            "expanded_chembl_eligible_name_overlap": len(available & chembl_names),
        })
        for family, names in {**PUBLISHED, "radiation_hypotheses": RADIATION}.items():
            for name in names:
                matches = compounds[compounds.normalized_name.eq(norm(name))]
                coverage_rows.append({
                    "accession": accession, "family": family, "requested_drug": name,
                    "represented_by_exact_normalized_name": len(matches) > 0,
                    "lincs_names": ";".join(sorted(matches.pert_iname.unique())),
                    "signature_count": len(matches), "cell_count": matches.cell_id.nunique(),
                    "dose_count": matches.pert_idose.nunique(), "timepoint_count": matches.pert_itime.nunique(),
                })
        for (cell, dose, time), group in compounds.groupby(["cell_id", "pert_idose", "pert_itime"]):
            context_rows.append({"accession": accession, "cell_id": cell, "dose": dose, "timepoint": time,
                                 "compound_signatures": len(group), "unique_compounds": group.pert_id.nunique()})
        for path in [gene_path, sig_path, inst_path, pert_path, cell_path]:
            hashes[str(path.relative_to(HERE))] = sha(path)

    audit = pd.DataFrame(audit_rows)
    coverage = pd.DataFrame(coverage_rows)
    audit.to_csv(OUT / "lincs_resource_audit.csv", index=False)
    coverage.to_csv(OUT / "prespecified_drug_coverage_by_release.csv", index=False)
    pd.DataFrame(context_rows).to_csv(OUT / "lincs_context_inventory.csv.gz", index=False, compression="gzip")
    selected = "GSE92742"
    provenance = {
        "selected_release": selected,
        "selection_rule": "Choose the public release with broader compound/cell coverage and greater prespecified-drug coverage before reversal outcomes are examined; report that GSE70138 has slightly greater exact-name overlap with eligible expanded-ChEMBL drugs.",
        "selected_matrix": "GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz",
        "selected_matrix_size_from_geo": "20 GB compressed",
        "data_level": "Level 5 replicate-collapsed moderated z-score signatures",
        "control_interpretation": "Level 5 signatures are already standardized to plate controls; vehicle signatures exist in metadata but raw treatment-minus-control reconstruction requires lower data levels.",
        "latent_bridge_compatibility": False,
        "latent_bridge_reason": "L1000 Level 5 values are perturbational z-scores for 978 measured plus inferred genes, not the validated canonical log1p(TPM) model input; missing canonical genes and inferred values make a Bridge encoder pass invalid.",
        "hashes": hashes,
    }
    (OUT / "lincs_resource_audit_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(audit.to_string(index=False))
    print("\nCoverage totals")
    print(coverage.groupby(["accession", "family"]).represented_by_exact_normalized_name.agg(["sum", "count"]).to_string())


if __name__ == "__main__":
    main()
