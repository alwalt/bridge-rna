#!/usr/bin/env python3
"""Construct strict within-study/stratum FLT-minus-GC response vectors."""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
RESULTS = HERE / "results"
WORK = HERE / "work"
MISSION_COLORS = {
    "RR1_CASIS": "#F8766D", "RR1_NASA": "#B79F00", "RR3": "#00BA38",
    "RR6": "#00BFC4", "RR9": "#619CFF", "STS_135": "#F564E3",
}

RAW_META_COLUMNS = {
    "study.parameter value.duration": "flight_duration",
    "study.parameter value.carcass preservation method": "carcass_preservation",
    "study.parameter value.sample preservation method": "sample_preservation",
}
STRATIFY = [
    "OSD", "mission", "strain", "age_at_launch", "sex", "material",
    "flight_duration", "library_preparation", "sequencing_facility",
    "sequencing_parameters", "carcass_preservation", "sample_preservation",
]


def clean(value: object) -> str:
    if pd.isna(value) or str(value).strip() in {"", "nan", "NaN"}:
        return "not_reported"
    return str(value).strip()


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-") or "NA"


def load_annotated_manifest() -> pd.DataFrame:
    manifest = pd.read_csv(RESULTS / "sample_manifest.csv")
    raw = pd.read_csv(ROOT / "data/osdr/metadata/selected_sample_metadata.tsv", sep="\t", low_memory=False)
    raw = raw.rename(columns={"id.accession": "OSD", "id.sample name": "sample_id", **RAW_META_COLUMNS})
    extra = ["OSD", "sample_id", *RAW_META_COLUMNS.values()]
    raw = raw[extra].drop_duplicates(["OSD", "sample_id"])
    out = manifest.merge(raw, on=["OSD", "sample_id"], how="left", validate="one_to_one")
    for column in STRATIFY:
        out[column] = out[column].map(clean)
    assert len(out) == 112 and out.sample_id.is_unique
    return out


def assign_contrasts(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records, memberships = [], []
    grouped = manifest.groupby(STRATIFY, sort=True, dropna=False)
    for number, (key, frame) in enumerate(grouped, 1):
        values = dict(zip(STRATIFY, key))
        counts = frame.condition.value_counts()
        n_flt, n_gc = int(counts.get("FLT", 0)), int(counts.get("GC", 0))
        if not n_flt or not n_gc:
            continue
        contrast_id = f"C{number:02d}__{slug(values['OSD'])}__{slug(values['mission'])}__{slug(values['flight_duration'])}"
        available_preservation = [
            f"carcass={values['carcass_preservation']}"
            for _ in [0] if values["carcass_preservation"] not in {"not_reported", "{Not Available}"}
        ] + [
            f"sample={values['sample_preservation']}"
            for _ in [0] if values["sample_preservation"] not in {"not_reported", "{Not Available}"}
        ]
        preservation = "; ".join(available_preservation) or "not_reported"
        records.append({"contrast_id": contrast_id, **values, "preservation": preservation,
                        "n_FLT": n_flt, "n_GC": n_gc, "n_total": len(frame)})
        memberships.extend({"contrast_id": contrast_id, "sample_id": sample,
                            "OSD": values["OSD"], "condition": condition}
                           for sample, condition in frame[["sample_id", "condition"]].itertuples(index=False))
    contrasts = pd.DataFrame(records)
    members = pd.DataFrame(memberships)
    if members.sample_id.duplicated().any():
        raise RuntimeError("A sample was assigned to multiple contrasts")
    missing = set(manifest.sample_id) - set(members.sample_id)
    if missing:
        raise RuntimeError(f"Samples lack matched controls within their strata: {sorted(missing)}")
    return contrasts, members


def response_matrix(matrix: np.ndarray, sample_ids: list[str], contrasts: pd.DataFrame,
                    members: pd.DataFrame) -> np.ndarray:
    positions = {sample: i for i, sample in enumerate(sample_ids)}
    responses = []
    for contrast_id in contrasts.contrast_id:
        z = members[members.contrast_id == contrast_id]
        flt = [positions[x] for x in z.loc[z.condition == "FLT", "sample_id"]]
        gc = [positions[x] for x in z.loc[z.condition == "GC", "sample_id"]]
        responses.append(matrix[flt].mean(axis=0) - matrix[gc].mean(axis=0))
    return np.asarray(responses)


def plot_arrows(manifest: pd.DataFrame, contrasts: pd.DataFrame, members: pd.DataFrame,
                bridge_pca: pd.DataFrame) -> pd.DataFrame:
    frame = manifest[["sample_id", "mission", "condition"]].merge(bridge_pca, on="sample_id", validate="one_to_one")
    rows = []
    fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
    ax.scatter(frame.PC1, frame.PC2, s=14, color="#C9C9C9", alpha=.28, zorder=1, label="individual sample")
    for row in contrasts.itertuples(index=False):
        ids = members.loc[members.contrast_id == row.contrast_id, "sample_id"]
        z = frame[frame.sample_id.isin(ids)]
        gc = z[z.condition == "GC"][["PC1", "PC2"]].mean().to_numpy()
        flt = z[z.condition == "FLT"][["PC1", "PC2"]].mean().to_numpy()
        color = MISSION_COLORS[row.mission]
        ax.scatter(*gc, s=52, marker="o", color=color, edgecolor="black", linewidth=.4, zorder=3)
        ax.scatter(*flt, s=62, marker="^", color=color, edgecolor="black", linewidth=.4, zorder=3)
        ax.annotate("", xy=flt, xytext=gc, arrowprops={"arrowstyle": "->", "color": color, "lw": 1.8}, zorder=2)
        ax.text(flt[0], flt[1], row.contrast_id.split("__")[0], fontsize=7, color="#333333")
        rows.append({"contrast_id": row.contrast_id, "GC_PC1": gc[0], "GC_PC2": gc[1],
                     "FLT_PC1": flt[0], "FLT_PC2": flt[1],
                     "delta_PC1": flt[0] - gc[0], "delta_PC2": flt[1] - gc[1]})
    ax.set(xlabel="BridgeRNA PC1", ylabel="BridgeRNA PC2",
           title="Within-stratum spaceflight displacement in frozen BridgeRNA space\narrow: GC centroid → FLT centroid")
    ax.grid(alpha=.15)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=color, label=mission)
               for mission, color in MISSION_COLORS.items()]
    handles += [plt.Line2D([], [], marker="o", linestyle="", color="#666666", label="GC centroid"),
                plt.Line2D([], [], marker="^", linestyle="", color="#666666", label="FLT centroid")]
    ax.legend(handles=handles, frameon=False, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.savefig(RESULTS / "task3b_bridgerna_gc_to_flt_arrows.png", dpi=400, bbox_inches="tight")
    fig.savefig(RESULTS / "task3b_bridgerna_gc_to_flt_arrows.pdf", bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def main() -> None:
    manifest = load_annotated_manifest()
    contrasts, members = assign_contrasts(manifest)
    sample_ids = manifest.sample_id.tolist()
    expression = np.load(WORK / "deseq2_normalized_log2p1.npy")
    bridge = np.load(WORK / "bridgerna_embeddings.npy")
    expression_pca = pd.read_csv(RESULTS / "expression_pca_coordinates.csv").set_index("sample_id").loc[sample_ids]
    bridge_pca = pd.read_csv(RESULTS / "bridgerna_pca_coordinates.csv").set_index("sample_id").loc[sample_ids]
    genes = pd.read_csv(WORK / "intersected_gene_ids.csv")["ensembl_gene_id"].astype(str).to_numpy()
    assert expression.shape == (112, len(genes)) and bridge.shape == (112, 512)

    delta_x = response_matrix(expression, sample_ids, contrasts, members)
    delta_pc = response_matrix(expression_pca.to_numpy(), sample_ids, contrasts, members)
    delta_z = response_matrix(bridge, sample_ids, contrasts, members)
    np.savez_compressed(RESULTS / "task3b_expression_response_vectors.npz",
                        contrast_id=contrasts.contrast_id.to_numpy(), gene_id=genes, delta_X=delta_x)
    np.savez_compressed(RESULTS / "task3b_bridgerna_response_vectors.npz",
                        contrast_id=contrasts.contrast_id.to_numpy(), dimension=np.arange(1, 513), delta_z=delta_z)
    pc_table = pd.concat([contrasts[["contrast_id"]],
                          pd.DataFrame(delta_pc, columns=[f"delta_PC{i}" for i in range(1, delta_pc.shape[1] + 1)])], axis=1)
    pc_table.to_csv(RESULTS / "task3b_pca_response_vectors.csv", index=False)
    contrasts.to_csv(RESULTS / "task3b_contrast_summary.csv", index=False)
    members.to_csv(RESULTS / "task3b_contrast_sample_membership.csv", index=False)
    centroids = plot_arrows(manifest, contrasts, members, bridge_pca.reset_index())
    centroids.to_csv(RESULTS / "task3b_bridgerna_centroid_displacements.csv", index=False)
    print(f"Valid contrasts: {len(contrasts)}; samples represented: {len(members)}/112")
    print(contrasts[["contrast_id", "OSD", "mission", "strain", "flight_duration",
                     "preservation", "library_preparation", "sequencing_parameters",
                     "n_FLT", "n_GC"]].to_string(index=False))


if __name__ == "__main__":
    main()
