#!/usr/bin/env python3
"""Run Task 2 exploratory GEPREP/ARCHS4 exercise-response analysis."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data_audit.check_exposure import load_manifest, rows_for_gse
from src.fm_embed.vocab import load_canonical_genes

HERE = Path(__file__).resolve().parents[1]
RESULTS, WORK = HERE / "results", HERE / "work"
FIGURES = RESULTS / "figures"
GSM_RE = re.compile(r"GSM\d+", re.IGNORECASE)
GSE_RE = re.compile(r"GSE\d+", re.IGNORECASE)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def normalized(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower()


def load_geprep(path: Path, species: str) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", low_memory=False)
    frame = frame[normalized(frame["tissue"]).eq("skeletal muscle")].copy()
    frame["GSM"] = frame["GSM"].astype("string").str.extract(r"(GSM\d+)", expand=False).str.upper()
    frame["GSE"] = frame["datasets"].astype("string").str.extract(r"(GSE\d+)", expand=False).str.upper()
    frame["species"] = species
    frame["subject_id"] = frame["subject id(or sample id)"].astype("string")
    return frame


def training_gses(manifest: pd.DataFrame) -> set[str]:
    found: set[str] = set()
    for value in manifest.loc[manifest.split.eq("train"), "gse_candidates_str"].dropna().astype(str):
        found.update(x.upper() for x in GSE_RE.findall(value))
    return found


def annotate_exposure(frame: pd.DataFrame, manifest: pd.DataFrame) -> pd.Series:
    train_gsm = set(manifest.loc[manifest.split.eq("train"), "gsm"].dropna().astype(str))
    train_studies = training_gses(manifest)
    # Validate the cached study set against the existing lookup helper.
    for gse in frame.GSE.dropna().unique():
        helper_seen = rows_for_gse(gse, manifest).split.eq("train").any()
        if helper_seen != (gse in train_studies):
            raise ValueError(f"Exposure helper disagreement for {gse}")
    return pd.Series(
        np.where(frame.GSM.isin(train_gsm), "exact_sample_seen",
                 np.where(frame.GSE.isin(train_studies), "same_study_seen", "fully_unseen")),
        index=frame.index,
    )


def build_manifest(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    locations = pd.read_parquet(resolve(cfg["embedding_directory"]) / "sample_locations.parquet")
    if locations.geo_accession.duplicated().any():
        raise ValueError("ARCHS4 embedding locations contain duplicate GSMs")
    source = pd.concat([
        load_geprep(resolve(cfg["human_geprep"]), "human"),
        load_geprep(resolve(cfg["mouse_geprep"]), "mouse"),
    ], ignore_index=True, sort=False)
    source["found_in_archs4"] = source.GSM.isin(set(locations.geo_accession))
    manifest = source.merge(locations, left_on="GSM", right_on="geo_accession", how="left", validate="one_to_one")
    exposure_manifest = load_manifest(resolve(cfg["sample_manifest"]))
    manifest["exposure_status"] = annotate_exposure(manifest, exposure_manifest)
    manifest["ARCHS4_index"] = manifest["global_index"].astype("Int64")
    manifest["embedding_index"] = manifest["global_index"].astype("Int64")
    manifest["exercise_type"] = manifest["exercise type"]
    manifest["biopsy_timepoint"] = manifest["biopsy timepoint"]
    manifest["sampling_site"] = manifest["sampling site"]
    manifest["embedding_available"] = manifest.embedding_index.notna()
    required = [
        "species", "GSE", "GSM", "ARCHS4_index", "embedding_index", "tissue", "sampling_site",
        "classification", "exercise_type", "protocol", "biopsy_timepoint", "group", "subject_id",
        "exposure_status", "found_in_archs4", "embedding_available", "shard_idx", "shard_file",
        "row_in_shard", "sample title in GEO", "treatment", "duration", "frequency", "intensity",
        "health status", "mus background", "mus genotype",
    ]
    for column in required:
        if column not in manifest: manifest[column] = pd.NA
    return source, manifest[required].sort_values(["species", "GSE", "GSM"]).reset_index(drop=True)


def count_table(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    work = frame.copy()
    work[column] = work[column].fillna("<missing>").astype(str)
    return work.groupby(["species", column], as_index=False).agg(
        samples=("GSM", "nunique"), studies=("GSE", "nunique")
    ).sort_values(["species", "samples", column], ascending=[True, False, True])


def save_descriptive_tables(source: pd.DataFrame, manifest: pd.DataFrame) -> None:
    coverage = manifest.groupby("species", as_index=False).agg(
        geprep_skeletal_muscle_samples=("GSM", "nunique"),
        found_in_archs4=("found_in_archs4", "sum"),
        bridge_embeddings=("embedding_available", "sum"),
        unique_gse_studies=("GSE", "nunique"),
    )
    coverage["archs4_percent"] = coverage.found_in_archs4 / coverage.geprep_skeletal_muscle_samples * 100
    coverage.to_csv(RESULTS / "coverage_summary.csv", index=False)
    matched = manifest[manifest.found_in_archs4].copy()
    for column, filename in [
        ("classification", "classification_summary.csv"), ("exercise_type", "exercise_type_summary.csv"),
        ("protocol", "protocol_summary.csv"), ("biopsy_timepoint", "biopsy_timepoint_summary.csv"),
        ("group", "group_summary.csv"),
    ]:
        count_table(matched, column).to_csv(RESULTS / filename, index=False)
    exposure = matched.groupby(["species", "exposure_status"], as_index=False).agg(
        samples=("GSM", "nunique"), studies=("GSE", "nunique")
    )
    exposure.to_csv(RESULTS / "exposure_summary.csv", index=False)
    overlap_rows = []
    for column in ["classification", "exercise_type", "protocol"]:
        sets = {species: set(matched.loc[matched.species.eq(species), column].dropna().astype(str))
                for species in ["human", "mouse"]}
        for value in sorted(sets["human"] & sets["mouse"]):
            overlap_rows.append({"metadata_field": column, "shared_value": value,
                                 "human_samples": int((matched.species.eq("human") & matched[column].eq(value)).sum()),
                                 "mouse_samples": int((matched.species.eq("mouse") & matched[column].eq(value)).sum())})
    pd.DataFrame(overlap_rows).to_csv(RESULTS / "cross_species_metadata_overlap.csv", index=False)

    comparison_rows = []
    for (species, gse), study in matched.groupby(["species", "GSE"], dropna=False):
        control = normalized(study.classification).eq("inactivity")
        response = normalized(study.classification).isin(["acute exercise", "longterm training"])
        control_subjects = set(study.loc[control, "subject_id"].dropna())
        response_subjects = set(study.loc[response, "subject_id"].dropna())
        comparison_rows.append({
            "species": species, "GSE": gse, "matched_samples": study.GSM.nunique(),
            "control_or_inactivity_samples": int(control.sum()), "exercise_response_samples": int(response.sum()),
            "subjects_in_both": len(control_subjects & response_subjects),
            "has_within_study_comparison": bool(control.any() and response.any()),
            "classifications": "; ".join(sorted(study.classification.dropna().astype(str).unique())),
            "exercise_types": "; ".join(sorted(study.exercise_type.dropna().astype(str).unique())),
            "protocols": "; ".join(sorted(study.protocol.dropna().astype(str).unique())),
            "biopsy_timepoints": "; ".join(sorted(study.biopsy_timepoint.dropna().astype(str).unique())),
        })
    pd.DataFrame(comparison_rows).sort_values(
        ["has_within_study_comparison", "species", "GSE"], ascending=[False, True, True]
    ).to_csv(RESULTS / "within_study_comparison_inventory.csv", index=False)


def post_hour(value: object) -> float:
    text = str(value).lower()
    if "post" not in text: return np.inf
    found = re.search(r"post\((\d+(?:\.\d+)?)h\)", text)
    if found: return float(found.group(1))
    return 0.0 if text.strip() == "post" else np.inf


def build_contrasts(matched: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select one transparent acute-aerobic contrast per usable study."""
    focused = matched[
        normalized(matched.classification).eq("acute exercise")
        & normalized(matched.exercise_type).eq("aerobic exercise")
    ]
    candidate_studies = set(focused.GSE.dropna())
    eligibility = []
    members = []
    for species in ["human", "mouse"]:
        for gse, study in matched[(matched.species.eq(species)) & matched.GSE.isin(candidate_studies)].groupby("GSE"):
            exercise = normalized(study.classification).eq("acute exercise") & normalized(study.exercise_type).eq("aerobic exercise")
            control = normalized(study.classification).eq("inactivity")
            subject_overlap = set(study.loc[exercise, "subject_id"].dropna()) & set(study.loc[control, "subject_id"].dropna())
            eligibility.append({"species": species, "GSE": gse, "matched_samples": len(study),
                                "exercise_samples": int(exercise.sum()), "control_or_pre_samples": int(control.sum()),
                                "subjects_in_both": len(subject_overlap), "candidate_acute_aerobic": True})

            if species == "human":
                # GSE71972 also contains a pharmacologic histamine-blockade arm;
                # use its unblocked control-exercise arm to isolate exercise.
                eligible_arm = (normalized(study.group).eq("control exercise")
                                if gse == "GSE71972" else pd.Series(True, index=study.index))
                times = sorted(study.loc[exercise, "biopsy_timepoint"].dropna().unique(), key=post_hour)
                chosen = next((t for t in times if post_hour(t) < np.inf), None)
                post = exercise & eligible_arm & study.biopsy_timepoint.eq(chosen) & study.subject_id.isin(subject_overlap)
                pre = control & eligible_arm & study.subject_id.isin(set(study.loc[post, "subject_id"]))
                # Enforce complete subject pairing at the chosen time point.
                paired = set(study.loc[post, "subject_id"]) & set(study.loc[pre, "subject_id"])
                post &= study.subject_id.isin(paired); pre &= study.subject_id.isin(paired)
                if study.loc[post, "subject_id"].duplicated().any() or study.loc[pre, "subject_id"].duplicated().any():
                    raise ValueError(f"{gse}: selected human contrast is not one row per subject and role")
                rule = f"paired earliest acute post ({chosen}) minus pre/baseline"
                stratum = str(chosen)
            elif gse == "GSE126962":
                phase = study.group.astype(str).str.startswith("Active phase")
                timepoint = study.biopsy_timepoint.eq("post(4h)")
                post, pre = exercise & phase & timepoint, control & phase & timepoint
                rule, stratum = "active-phase treadmill exercise minus phase/time-matched sham", "active phase; post(4h)"
            elif gse == "GSE132520":
                wild = normalized(study["mus genotype"]).eq("wild-type")
                post, pre = exercise & wild, control & wild
                rule, stratum = "wild-type treadmill exercise minus wild-type sedentary", "wild-type; post(3h)"
            elif gse == "GSE97718":
                con = study.group.astype(str).str.lower().str.contains("control diet|con-")
                post, pre = exercise & con, control & con
                rule, stratum = "control-diet acute treadmill post minus control-diet pre", "control diet; post(3h)"
            else:
                continue
            if post.sum() == 0 or pre.sum() == 0:
                continue
            contrast_id = f"{species}_{gse}"
            for role, mask in [("post_exercise", post), ("pre_control", pre)]:
                for row in study[mask].itertuples():
                    members.append({"contrast_id": contrast_id, "species": species, "GSE": gse,
                                    "GSM": row.GSM, "role": role, "rule": rule, "stratum": stratum})
            eligibility[-1].update({"selected_primary_contrast": True, "contrast_id": contrast_id,
                                    "selected_exercise": int(post.sum()), "selected_control": int(pre.sum()),
                                    "contrast_rule": rule, "contrast_stratum": stratum})
    eligible = pd.DataFrame(eligibility)
    eligible["selected_primary_contrast"] = eligible["selected_primary_contrast"].fillna(False)
    return eligible, pd.DataFrame(members)


def extract_shard_expression(manifest: pd.DataFrame, genes: list[str], shard_dir: Path) -> np.ndarray:
    """Extract ARCHS4 TPM values in the canonical BridgeRNA gene order."""
    matrix = np.empty((len(manifest), len(genes)), dtype=np.float32)
    manifest = manifest.copy(); manifest["matrix_row"] = np.arange(len(manifest))
    for shard_file, shard_rows in manifest.groupby("shard_file", sort=True):
        path = shard_dir / shard_file
        if not path.is_file(): raise FileNotFoundError(path)
        parquet = pq.ParquetFile(path)
        available = parquet.schema_arrow.names
        if available[:-1] != genes or available[-1] != "geo_accession":
            raise ValueError(f"Gene vocabulary/order mismatch in {path}")
        shard_rows = shard_rows.assign(row_group=shard_rows.row_in_shard.astype(int) // 2048)
        for row_group, selected in shard_rows.groupby("row_group"):
            table = parquet.read_row_group(int(row_group), columns=[*genes, "geo_accession"])
            block = table.to_pandas(ignore_metadata=True)
            start = int(row_group) * 2048
            offsets = selected.row_in_shard.astype(int).to_numpy() - start
            observed = block.iloc[offsets]["geo_accession"].astype(str).to_numpy()
            if not np.array_equal(observed, selected.GSM.astype(str).to_numpy()):
                raise ValueError(f"GSM/row mismatch in {shard_file} row group {row_group}")
            matrix[selected.matrix_row.to_numpy(int)] = block.iloc[offsets][genes].to_numpy(np.float32)
        log(f"Expression extracted: {shard_file} ({len(shard_rows)} matched rows)")
    return matrix


def validate_tpm(matrix: np.ndarray, expected_shape: tuple[int, int]) -> None:
    """Guard against mislabeled or transformed ARCHS4 expression caches."""
    if matrix.shape != expected_shape:
        raise ValueError(f"TPM matrix shape mismatch: expected {expected_shape}, got {matrix.shape}")
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("TPM matrix contains non-finite or negative values")
    row_sums = matrix.sum(axis=1, dtype=np.float64)
    if not np.allclose(row_sums, 1_000_000.0, rtol=5e-4, atol=250.0):
        raise ValueError(
            "ARCHS4 expression cache is not TPM: sample sums are not approximately 1,000,000"
        )


def load_embeddings(manifest: pd.DataFrame, embedding_dir: Path) -> np.ndarray:
    spec = json.loads((embedding_dir / "embedding_manifest.json").read_text())
    total, dim = int(spec["total_samples"]), int(spec["embedding_dim"])
    dtype = np.dtype(spec["embedding_dtype"])
    mmap = np.memmap(embedding_dir / f"sample_embeddings.{dtype.name}.mmap", dtype=dtype, mode="r", shape=(total, dim))
    return np.asarray(mmap[manifest.embedding_index.to_numpy(int)], dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return a @ b.T


def response_effects(
    members: pd.DataFrame, manifest: pd.DataFrame, representations: dict[str, np.ndarray]
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    lookup = manifest.reset_index().set_index("GSM")["index"]
    metadata_rows, effects = [], {key: [] for key in representations}
    for contrast_id, rows in members.groupby("contrast_id", sort=True):
        post = lookup.loc[rows.loc[rows.role.eq("post_exercise"), "GSM"]].to_numpy(int)
        pre = lookup.loc[rows.loc[rows.role.eq("pre_control"), "GSM"]].to_numpy(int)
        first = rows.iloc[0]
        metadata_rows.append({"contrast_id": contrast_id, "species": first.species, "GSE": first.GSE,
                              "exercise_samples": len(post), "control_samples": len(pre),
                              "rule": first.rule, "stratum": first.stratum,
                              "exercise_type": "aerobic exercise"})
        for name, matrix in representations.items():
            effects[name].append(matrix[post].mean(axis=0) - matrix[pre].mean(axis=0))
    metadata = pd.DataFrame(metadata_rows)
    return metadata, {name: np.stack(values) for name, values in effects.items()}


def plot_heatmap(values: np.ndarray, humans: list[str], mice: list[str], representation: str) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    image = ax.imshow(values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(mice)), mice, rotation=35, ha="right")
    ax.set_yticks(range(len(humans)), humans)
    ax.set(xlabel="Mouse study contrast", ylabel="Human study contrast", title=f"{representation}: acute aerobic response cosine")
    for i in range(len(humans)):
        for j in range(len(mice)): ax.text(j, i, f"{values[i,j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, label="Cosine similarity")
    fig.tight_layout()
    fig.savefig(FIGURES / f"response_similarity_{representation}.png", dpi=320, bbox_inches="tight")
    fig.savefig(FIGURES / f"response_similarity_{representation}.pdf", bbox_inches="tight")
    plt.close(fig)


def analyze(cfg: dict, args: argparse.Namespace) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True); WORK.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
    source, all_manifest = build_manifest(cfg)
    all_manifest.to_parquet(RESULTS / "geprep_skeletal_muscle_manifest.parquet", index=False)
    all_manifest.to_csv(RESULTS / "geprep_skeletal_muscle_manifest.csv", index=False)
    save_descriptive_tables(source, all_manifest)
    matched = all_manifest[all_manifest.found_in_archs4].reset_index(drop=True)
    matched.to_parquet(RESULTS / "matched_manifest.parquet", index=False)
    eligibility, members = build_contrasts(matched)
    eligibility.to_csv(RESULTS / "contrast_eligibility.csv", index=False)
    members.to_parquet(RESULTS / "contrast_members.parquet", index=False)
    log(f"Matched skeletal muscle: {len(matched):,}; selected contrasts: {members.contrast_id.nunique()}")

    genes = load_canonical_genes(resolve(cfg["canonical_genes"]))
    tpm_path = WORK / "matched_tpm.npy"
    legacy_mislabeled_tpm_path = WORK / "matched_log1p_tpm.npy"
    expression_path = WORK / "matched_log1p_tpm_corrected.npy"
    embedding_path, pca_path = WORK / "matched_embeddings.npy", WORK / "matched_pca_log1p_tpm.npy"
    expected_shape = (len(matched), len(genes))

    # The original Task 2 run saved untransformed TPM under the misleading
    # matched_log1p_tpm.npy name. Reuse it only after proving it is TPM.
    if args.reuse_prepared and tpm_path.exists():
        tpm = np.load(tpm_path)
        validate_tpm(tpm, expected_shape)
        log("Reusing validated TPM cache")
    elif args.reuse_prepared and legacy_mislabeled_tpm_path.exists():
        tpm = np.load(legacy_mislabeled_tpm_path)
        validate_tpm(tpm, expected_shape)
        np.save(tpm_path, tpm)
        log("Migrated and validated legacy TPM cache; no shard re-extraction needed")
    else:
        tpm = extract_shard_expression(matched, genes, resolve(cfg["archs4_shard_directory"]))
        validate_tpm(tpm, expected_shape)
        np.save(tpm_path, tpm)

    expression = np.log1p(tpm).astype(np.float32)
    np.save(expression_path, expression)

    if args.reuse_prepared and embedding_path.exists():
        embeddings = np.load(embedding_path)
        log("Reusing frozen BridgeRNA embedding cache")
    else:
        embeddings = load_embeddings(matched, resolve(cfg["embedding_directory"]))
        np.save(embedding_path, embeddings)

    # PCA must always correspond to the corrected log1p(TPM) matrix. Its
    # distinct filename prevents reuse of the earlier PCA fitted to TPM.
    if args.reuse_prepared and pca_path.exists():
        pca_scores = np.load(pca_path)
        log("Reusing corrected log1p(TPM) PCA cache")
    else:
        n_components = min(int(cfg["pca_components"]), len(matched) - 1, len(genes))
        pca = PCA(n_components=n_components, svd_solver="randomized", random_state=int(cfg["seed"]))
        pca_scores = pca.fit_transform(expression).astype(np.float32)
        np.save(pca_path, pca_scores)
        np.save(WORK / "pca_components_log1p_tpm.npy", pca.components_.astype(np.float32))
        np.save(WORK / "pca_mean_log1p_tpm.npy", pca.mean_.astype(np.float32))
        pd.DataFrame({"pc": np.arange(1, n_components + 1), "variance_explained": pca.explained_variance_ratio_,
                      "cumulative_variance": np.cumsum(pca.explained_variance_ratio_)}).to_csv(RESULTS / "pca_variance.csv", index=False)
    if expression.shape != (len(matched), 15165) or embeddings.shape != (len(matched), 512):
        raise ValueError("Prepared representation shape mismatch")
    metadata, effects = response_effects(members, matched, {"raw_expression": expression, "joint_pca": pca_scores, "bridgerna": embeddings})
    metadata.to_csv(RESULTS / "response_contrasts.csv", index=False)
    ranking_rows, similarity_rows = [], []
    for name, vectors in effects.items():
        np.save(WORK / f"response_effects_{name}.npy", vectors)
        human_mask = metadata.species.eq("human").to_numpy(); mouse_mask = metadata.species.eq("mouse").to_numpy()
        similarities = cosine(vectors[human_mask], vectors[mouse_mask])
        human_ids = metadata.loc[human_mask, "GSE"].tolist(); mouse_ids = metadata.loc[mouse_mask, "GSE"].tolist()
        pd.DataFrame(similarities, index=human_ids, columns=mouse_ids).to_csv(RESULTS / f"response_similarity_{name}.csv")
        plot_heatmap(similarities, human_ids, mouse_ids, name)
        for i, human in enumerate(human_ids):
            order = np.argsort(-similarities[i])
            for rank, j in enumerate(order, 1):
                ranking_rows.append({"representation": name, "human_GSE": human, "mouse_GSE": mouse_ids[j],
                                     "rank": rank, "cosine_similarity": float(similarities[i,j]),
                                     "same_exercise_type": True,
                                     "human_protocol": metadata.loc[human_mask].iloc[i].rule,
                                     "mouse_protocol": metadata.loc[mouse_mask].iloc[j].rule,
                                     "human_timepoint": metadata.loc[human_mask].iloc[i].stratum,
                                     "mouse_timepoint": metadata.loc[mouse_mask].iloc[j].stratum})
                similarity_rows.append({"representation": name, "human_GSE": human, "mouse_GSE": mouse_ids[j],
                                        "cosine_similarity": float(similarities[i,j])})
    pd.DataFrame(ranking_rows).to_csv(RESULTS / "cross_species_response_rankings.csv", index=False)
    pd.DataFrame(similarity_rows).to_csv(RESULTS / "cross_species_response_similarities.csv", index=False)
    provenance = {"genes": len(genes), "expression": "natural log1p of existing ARCHS4 shard TPM",
                  "pca": f"joint PCA over {len(matched)} matched skeletal-muscle profiles",
                  "embeddings": "existing frozen 512-D BridgeRNA memmap", "exposure_filter": "none; informational only",
                  "response_focus": "acute aerobic exercise", "alignment": "none", "seed": int(cfg["seed"])}
    (RESULTS / "provenance.json").write_text(json.dumps(provenance, indent=2))
    log("Task 2 exploratory analysis complete")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-prepared", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    config = json.loads((HERE / "config.json").read_text())
    analyze(config, parse_args())
