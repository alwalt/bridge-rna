#!/usr/bin/env python3
"""Compare Task 3B FLT-minus-GC response geometry across representations."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"
REPRESENTATIONS = ("expression", "pca", "bridgerna")
FACTORS = ("mission", "library_preparation", "sequencing_parameters", "sequencing_facility", "strain")


def load_inputs() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    meta = pd.read_csv(RESULTS / "task3b_contrast_summary.csv")
    expression = np.load(RESULTS / "task3b_expression_response_vectors.npz", allow_pickle=True)
    bridge = np.load(RESULTS / "task3b_bridgerna_response_vectors.npz", allow_pickle=True)
    pca = pd.read_csv(RESULTS / "task3b_pca_response_vectors.csv")
    ids = meta.contrast_id.astype(str).to_numpy()
    if not (np.array_equal(ids, expression["contrast_id"].astype(str)) and
            np.array_equal(ids, bridge["contrast_id"].astype(str)) and
            np.array_equal(ids, pca.contrast_id.astype(str).to_numpy())):
        raise RuntimeError("Task 3B contrast order differs among response-vector files")
    matrices = {
        "expression": expression["delta_X"],
        "pca": pca.filter(regex=r"^delta_PC").to_numpy(),
        "bridgerna": bridge["delta_z"],
    }
    if any(matrix.shape[0] != len(meta) for matrix in matrices.values()):
        raise RuntimeError("A response matrix does not contain one row per contrast")
    return meta, matrices


def short_ids(meta: pd.DataFrame) -> list[str]:
    return [f"C{i + 1:02d} | {row.OSD} | {row.mission}" for i, row in meta.iterrows()]


def make_pair_table(meta: pd.DataFrame, similarities: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for i in range(len(meta)):
        for j in range(i + 1, len(meta)):
            row = {
                "contrast_i": meta.loc[i, "contrast_id"], "contrast_j": meta.loc[j, "contrast_id"],
                "OSD_i": meta.loc[i, "OSD"], "OSD_j": meta.loc[j, "OSD"],
                "different_OSD": meta.loc[i, "OSD"] != meta.loc[j, "OSD"],
            }
            for factor in FACTORS:
                row[f"same_{factor}"] = meta.loc[i, factor] == meta.loc[j, factor]
            for representation, matrix in similarities.items():
                row[f"cosine_{representation}"] = matrix[i, j]
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_overall(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes = {"all_contrast_pairs": pairs, "different_OSD": pairs[pairs.different_OSD]}
    for scope, frame in scopes.items():
        for representation in REPRESENTATIONS:
            values = frame[f"cosine_{representation}"].to_numpy()
            rows.append({"pair_scope": scope, "representation": representation,
                         "n_pairs": len(values), "mean_cosine": values.mean(),
                         "sd_cosine": values.std(ddof=1), "median_cosine": np.median(values),
                         "q25_cosine": np.quantile(values, .25), "q75_cosine": np.quantile(values, .75),
                         "positive_fraction": np.mean(values > 0)})
    return pd.DataFrame(rows)


def permutation_pvalue(values: np.ndarray, labels: np.ndarray, osd_labels: np.ndarray, observed: float,
                       rng: np.random.Generator, repetitions: int = 9999) -> float:
    null = np.empty(repetitions)
    for index in range(repetitions):
        shuffled = rng.permutation(labels)
        same = shuffled[:, None] == shuffled[None, :]
        upper = np.triu(np.ones_like(same, dtype=bool), 1)
        cross_study = osd_labels[:, None] != osd_labels[None, :]
        valid_same, valid_diff = same & upper & cross_study, ~same & upper & cross_study
        null[index] = values[valid_same].mean() - values[valid_diff].mean()
    return (np.count_nonzero(np.abs(null) >= abs(observed)) + 1) / (repetitions + 1)


def summarize_boundaries(meta: pd.DataFrame, similarities: dict[str, np.ndarray], pairs: pd.DataFrame) -> pd.DataFrame:
    # Primary boundary comparisons omit pairs from the same OSD.
    cross = pairs[pairs.different_OSD].copy()
    rng = np.random.default_rng(3407)
    rows = []
    for factor in FACTORS:
        labels = meta[factor].fillna("not_reported").astype(str).to_numpy()
        for representation, similarity in similarities.items():
            same = cross[f"same_{factor}"].to_numpy(dtype=bool)
            values = cross[f"cosine_{representation}"].to_numpy()
            same_values, different_values = values[same], values[~same]
            difference = same_values.mean() - different_values.mean() if len(same_values) and len(different_values) else np.nan
            pvalue = (permutation_pvalue(similarity, labels, meta.OSD.astype(str).to_numpy(), difference, rng)
                      if np.isfinite(difference) and len(np.unique(labels)) > 1 else np.nan)
            rows.append({"representation": representation, "factor": factor,
                         "n_same_pairs": len(same_values), "n_different_pairs": len(different_values),
                         "same_mean_cosine": same_values.mean() if len(same_values) else np.nan,
                         "same_sd_cosine": same_values.std(ddof=1) if len(same_values) > 1 else np.nan,
                         "different_mean_cosine": different_values.mean() if len(different_values) else np.nan,
                         "different_sd_cosine": different_values.std(ddof=1) if len(different_values) > 1 else np.nan,
                         "same_minus_different": difference, "permutation_pvalue_exploratory": pvalue})
    return pd.DataFrame(rows)


def cluster_bridge(similarity: np.ndarray, meta: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    distance = np.clip(1 - similarity, 0, 2)
    np.fill_diagonal(distance, 0)
    tree = linkage(squareform(distance, checks=False), method="average")
    order = leaves_list(tree)
    scans = []
    assignments = {}
    for k in range(2, min(7, len(meta))):
        labels = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average").fit_predict(distance)
        score = silhouette_score(distance, labels, metric="precomputed")
        scans.append({"n_clusters": k, "silhouette_cosine": score})
        assignments[k] = labels
    scan = pd.DataFrame(scans)
    selected_k = int(scan.loc[scan.silhouette_cosine.idxmax(), "n_clusters"])
    labels = assignments[selected_k]
    result = meta.copy()
    result.insert(1, "geometry_cluster", labels + 1)
    result.insert(2, "heatmap_order", np.argsort(order))
    return order, scan, result.sort_values("heatmap_order")


def summarize_clusters(similarities: dict[str, np.ndarray], clusters: pd.DataFrame,
                       meta: pd.DataFrame) -> pd.DataFrame:
    labels_by_id = clusters.set_index("contrast_id").geometry_cluster
    labels = meta.contrast_id.map(labels_by_id).to_numpy()
    cross_study = meta.OSD.to_numpy()[:, None] != meta.OSD.to_numpy()[None, :]
    upper = np.triu(np.ones((len(meta), len(meta)), dtype=bool), 1)
    rows = []
    for representation, matrix in similarities.items():
        for relation, mask in {
            "same_geometry_cluster": (labels[:, None] == labels[None, :]),
            "different_geometry_cluster": (labels[:, None] != labels[None, :]),
        }.items():
            values = matrix[mask & cross_study & upper]
            rows.append({"representation": representation, "pair_relation": relation,
                         "n_different_OSD_pairs": len(values), "mean_cosine": values.mean(),
                         "sd_cosine": values.std(ddof=1), "median_cosine": np.median(values)})
    return pd.DataFrame(rows)


def save_matrices(meta: pd.DataFrame, similarities: dict[str, np.ndarray]) -> None:
    ids = meta.contrast_id.tolist()
    for name, matrix in similarities.items():
        pd.DataFrame(matrix, index=ids, columns=ids).to_csv(RESULTS / f"task3c_cosine_{name}.csv")


def plot_heatmaps(meta: pd.DataFrame, similarities: dict[str, np.ndarray], order: np.ndarray) -> None:
    labels = np.asarray(short_ids(meta))[order]
    fig, axes = plt.subplots(1, 3, figsize=(24, 8.5), layout="constrained")
    titles = {"expression": "Conventional expression", "pca": "Expression PCA (20 PCs)",
              "bridgerna": "BridgeRNA (512-D)"}
    image = None
    for ax, name in zip(axes, REPRESENTATIONS):
        image = ax.imshow(similarities[name][np.ix_(order, order)], cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(titles[name], weight="bold")
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
        ax.set_yticks(range(len(labels)), labels if ax is axes[0] else [], fontsize=7)
    fig.colorbar(image, ax=axes, shrink=.72, label="Cosine similarity of FLT − GC response vectors")
    fig.suptitle("Task 3C: matched spaceflight-response geometry\nOne BridgeRNA-derived ordering and common color scale", fontsize=15)
    for suffix, kwargs in (("png", {"dpi": 400}), ("pdf", {})):
        fig.savefig(RESULTS / f"task3c_response_cosine_heatmaps.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def plot_boundaries(summary: pd.DataFrame) -> None:
    labels = {"mission": "Mission", "library_preparation": "Library prep",
              "sequencing_parameters": "Sequencing config", "sequencing_facility": "Facility", "strain": "Strain"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.3), sharey=True, layout="constrained")
    y = np.arange(len(FACTORS))
    for ax, representation in zip(axes, REPRESENTATIONS):
        frame = summary.set_index(["representation", "factor"]).loc[representation].loc[list(FACTORS)]
        ax.barh(y + .18, frame.same_mean_cosine, height=.34, color="#31688e", label="Same factor")
        ax.barh(y - .18, frame.different_mean_cosine, height=.34, color="#f28e2b", label="Different factor")
        ax.axvline(0, color="#333333", lw=.7)
        ax.set_title(representation.replace("bridgerna", "BridgeRNA").title())
        ax.set_xlabel("Mean cosine (different-OSD pairs)")
        ax.set_yticks(y, [labels[x] for x in FACTORS])
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Response concordance across known technical and biological boundaries", fontsize=14)
    for suffix, kwargs in (("png", {"dpi": 400}), ("pdf", {})):
        fig.savefig(RESULTS / f"task3c_batch_boundary_concordance.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def main() -> None:
    meta, matrices = load_inputs()
    similarities = {name: cosine_similarity(matrix) for name, matrix in matrices.items()}
    save_matrices(meta, similarities)
    pairs = make_pair_table(meta, similarities)
    overall = summarize_overall(pairs)
    boundaries = summarize_boundaries(meta, similarities, pairs)
    order, scan, clusters = cluster_bridge(similarities["bridgerna"], meta)
    cluster_summary = summarize_clusters(similarities, clusters, meta)
    annotations = meta[["contrast_id", "OSD", "mission", "library_preparation", "sequencing_parameters",
                        "sequencing_facility", "strain", "flight_duration", "n_FLT", "n_GC"]].copy()
    pairs.to_csv(RESULTS / "task3c_pairwise_comparisons.csv", index=False)
    overall.to_csv(RESULTS / "task3c_overall_concordance.csv", index=False)
    boundaries.to_csv(RESULTS / "task3c_batch_boundary_summary.csv", index=False)
    scan.to_csv(RESULTS / "task3c_cluster_scan.csv", index=False)
    clusters.to_csv(RESULTS / "task3c_cluster_assignments.csv", index=False)
    cluster_summary.to_csv(RESULTS / "task3c_cluster_concordance.csv", index=False)
    annotations.to_csv(RESULTS / "task3c_contrast_annotations.csv", index=False)
    plot_heatmaps(meta, similarities, order)
    plot_boundaries(boundaries)
    print("Task 3C complete")
    print(f"Contrasts: {len(meta)}; all pairs: {len(pairs)}; different-OSD pairs: {pairs.different_OSD.sum()}")
    print("\nDifferent-OSD concordance:")
    print(overall[overall.pair_scope == "different_OSD"].to_string(index=False))
    print("\nBridgeRNA boundary comparisons:")
    print(boundaries[boundaries.representation == "bridgerna"].to_string(index=False))
    print("\nBridgeRNA exploratory cluster scan:")
    print(scan.to_string(index=False))


if __name__ == "__main__":
    main()
