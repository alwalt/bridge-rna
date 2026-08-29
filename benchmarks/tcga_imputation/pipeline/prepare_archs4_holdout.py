#!/usr/bin/env python3
"""Freeze and preprocess 1,000 unused ARCHS4 samples per species."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pandas as pd

from common import REPO_ROOT, RESULTS, WORK
from prepare_tcga import HGNC, OUR_LENGTHS, build_hgnc_crosswalk, norm_gene, tpm_log1p


MANIFEST = REPO_ROOT / "data/manifests/sample_manifest.parquet"
ORTHOLOGS = REPO_ROOT / "data/ensembl/orthologs_one2one.txt"
MOUSE_LENGTHS = REPO_ROOT / "data/gencode/gencode_v49_mouse_gene_exon_lengths.csv"
H5 = {
    "archs4_holdout_human": REPO_ROOT / "data/archs4/human_gene_v2.5.h5",
    "archs4_holdout_mouse": REPO_ROOT / "data/archs4/mouse_gene_v2.5.h5",
}
SELECTION_SEED = 20260828
SAMPLES_PER_SPECIES = 1_000


def stable_id(value: object) -> str:
    return str(value).strip().split(".", 1)[0].upper()


def hgnc_ensembl_to_symbol() -> dict[str, str]:
    frame = pd.read_csv(HGNC, sep="\t", dtype=str)
    grouped = frame.dropna(subset=["ensembl_gene_id"]).groupby(
        frame.ensembl_gene_id.map(stable_id)
    )["symbol"].agg(lambda values: sorted(set(map(norm_gene, values))))
    return {gene_id: values[0] for gene_id, values in grouped.items() if len(values) == 1}


def select_samples() -> pd.DataFrame:
    manifest = pd.read_parquet(MANIFEST)
    pool = manifest.loc[
        manifest["split"].eq("unseen")
        & manifest.study_exposure.eq("unseen_study")
        & manifest.mapping_status.eq("mapped_single")
    ].copy()
    selected = []
    for species in ("human", "mouse"):
        species_pool = pool.loc[pool.species.eq(species)].drop_duplicates("sample_id")
        chosen = species_pool.sample(SAMPLES_PER_SPECIES, random_state=SELECTION_SEED).copy()
        chosen["dataset"] = f"archs4_holdout_{species}"
        selected.append(chosen)
    result = pd.concat(selected, ignore_index=True)
    result.to_parquet(RESULTS / "archs4_holdout_selected_samples.parquet", index=False)
    return result


def source_rows(dataset: str, handle: h5py.File,
                model_approved: list[str | None]) -> tuple[np.ndarray, np.ndarray]:
    symbols = [norm_gene(x.decode() if isinstance(x, bytes) else x)
               for x in handle["meta/genes/symbol"][:]]
    ensembl = [stable_id(x.decode() if isinstance(x, bytes) else x)
               for x in handle["meta/genes/ensembl_gene"][:]]
    if dataset.endswith("human"):
        _, source_to_approved = build_hgnc_crosswalk(symbols)
        approved_to_row = {source_to_approved[source]: row for row, source in enumerate(symbols)
                           if source in source_to_approved}
        lengths = pd.read_csv(OUR_LENGTHS).set_index("gene_symbol")["exon_length"]
        rows = np.asarray([approved_to_row.get(gene, -1) if gene else -1
                           for gene in model_approved], dtype=int)
        model_lengths = lengths.reindex(pd.read_parquet(WORK / "ours_genes.parquet").gene).to_numpy(float)
        return rows, model_lengths

    orthologs = pd.read_csv(ORTHOLOGS, sep="\t", dtype=str)
    orthologs = orthologs.loc[
        orthologs["Human homology type"].eq("ortholog_one2one")
        & orthologs["Human orthology confidence [0 low, 1 high]"].eq("1")
    ].copy()
    hgnc_by_ensembl = hgnc_ensembl_to_symbol()
    orthologs["approved"] = orthologs["Human gene stable ID"].map(stable_id).map(hgnc_by_ensembl)
    orthologs["mouse_ensembl"] = orthologs["Gene stable ID"].map(stable_id)
    pairs = orthologs.dropna(subset=["approved"]).drop_duplicates("approved", keep=False)
    approved_to_mouse = dict(zip(pairs.approved, pairs.mouse_ensembl))
    ensembl_to_row = {gene_id: row for row, gene_id in enumerate(ensembl)}
    mouse_lengths = pd.read_csv(MOUSE_LENGTHS)
    mouse_lengths.index = mouse_lengths["gene_symbol"].map(norm_gene)
    mouse_length_lookup = mouse_lengths["exon_length"]
    rows = np.asarray([ensembl_to_row.get(approved_to_mouse.get(gene, ""), -1)
                       if gene else -1 for gene in model_approved], dtype=int)
    model_lengths = np.asarray([
        mouse_length_lookup.get(symbols[row], np.nan) if row >= 0 else np.nan for row in rows
    ], dtype=float)
    rows[~np.isfinite(model_lengths)] = -1
    model_lengths[~np.isfinite(model_lengths)] = 1.0
    return rows, model_lengths


def prepare_dataset(dataset: str, selected: pd.DataFrame, model_approved: list[str | None]) -> dict:
    with h5py.File(H5[dataset], "r") as handle:
        accessions = [x.decode() if isinstance(x, bytes) else str(x)
                      for x in handle["meta/samples/geo_accession"][:]]
        accession_to_col = {gsm: col for col, gsm in enumerate(accessions)}
        sample_ids = selected.loc[selected.dataset.eq(dataset), "sample_id"].astype(str).tolist()
        rows, lengths = source_rows(dataset, handle, model_approved)
        observed = rows >= 0
        matrix = np.zeros((len(sample_ids), len(rows)), dtype=np.float32)
        expression = handle["data/expression"]
        for out_row, gsm in enumerate(sample_ids):
            if gsm not in accession_to_col:
                raise KeyError(f"{gsm} absent from {H5[dataset]}")
            raw = expression[:, accession_to_col[gsm]]
            matrix[out_row, observed] = raw[rows[observed]]
            if (out_row + 1) % 250 == 0:
                print(f"[{dataset}] extracted {out_row + 1:,}/{len(sample_ids):,}", flush=True)
    matrix = tpm_log1p(matrix, lengths)
    np.save(WORK / f"{dataset}_log1p_tpm.npy", matrix)
    pd.DataFrame({"gene_index": np.arange(len(rows)), "observed": observed}).to_parquet(
        WORK / f"{dataset}_genes.parquet", index=False
    )
    return {"dataset": dataset, "samples": len(sample_ids), "observed_genes": int(observed.sum())}


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    selected = select_samples()
    model_table = pd.read_parquet(WORK / "ours_genes.parquet")
    model_approved = model_table.approved_symbol.where(model_table.approved_symbol.notna(), None).tolist()
    summaries = [prepare_dataset(dataset, selected, model_approved) for dataset in H5]
    manifest = {
        "selection_seed": SELECTION_SEED,
        "source_manifest": str(MANIFEST),
        "criteria": {
            "split": "unseen", "study_exposure": "unseen_study",
            "mapping_status": "mapped_single",
        },
        "datasets": summaries,
    }
    (RESULTS / "archs4_holdout_preparation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
