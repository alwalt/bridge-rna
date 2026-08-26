"""Presentation-ready 3D UMAP of ARCHS4 sample embeddings.

Produces interactive HTML and/or a rotating MP4/GIF, coloured by species (human vs
mouse) and by inferred tissue type. MP4 is the best option for PowerPoint.

Examples
--------
    python viz/archs4_map.py                       # HTML + MP4, 10k points
    python viz/archs4_map.py --export gif          # GIF only
    python viz/archs4_map.py --color-by tissue --drop-unknown --export mp4 gif

    # The two commands used for the 100k presentation movies:
    MPLCONFIGDIR=/tmp/bridge-rna-mpl .venv/bin/python viz/archs4_map.py --n-samples 100000 --color-by species --export mp4 --video-duration 12 --fps 15
    MPLCONFIGDIR=/tmp/bridge-rna-mpl .venv/bin/python viz/archs4_map.py --n-samples 100000 --color-by tissue --export mp4 --video-duration 12 --fps 15
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA

REPO = Path(__file__).resolve().parent.parent
EMB_DIR = REPO / "embeddings" / "archs4"
MANIFEST = EMB_DIR / "embedding_manifest.json"
MMAP = EMB_DIR / "sample_embeddings.float16.mmap"
META = EMB_DIR / "sample_locations.parquet"
ARCHS4_H5 = {
    "human": REPO / "data" / "archs4" / "human_gene_v2.5.h5",
    "mouse": REPO / "data" / "archs4" / "mouse_gene_v2.5.h5",
}
CACHE_DIR = REPO / "viz" / "cache"
OUT_DIR = REPO / "viz" / "output"

SPECIES_COLORS = {"human": "#38BDF8", "mouse": "#FB7185"}
UNKNOWN_COLOR = "#3F4A5A"
UNKNOWN_LABEL = "unassigned"

# Ordered keyword rules: first match wins, so put specific terms before generic ones.
TISSUE_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("blood / immune", ("pbmc", "whole blood", "peripheral blood", "leukocyte", "lymphocyte",
                        "monocyte", "macrophage", "neutrophil", "t cell", "t-cell", "b cell",
                        "b-cell", "nk cell", "dendritic cell", "cd4", "cd8", "thymus", "spleen",
                        "bone marrow", "erythroid", "megakaryocyte")),
    ("brain / neural", ("brain", "cortex", "cortical", "hippocamp", "cerebell", "neuron",
                        "neural", "glia", "astrocyte", "microglia", "oligodendrocyte",
                        "striatum", "hypothalam", "spinal cord", "retina", "dorsal root")),
    ("liver", ("liver", "hepato", "hepatic")),
    ("kidney", ("kidney", "renal", "nephron", "podocyte")),
    ("lung / airway", ("lung", "pulmonary", "alveolar", "bronchial", "airway", "trachea")),
    ("heart / vessel", ("heart", "cardiac", "cardiomyocyte", "myocard", "aorta", "endothelial",
                        "vascular", "artery", "vein")),
    ("muscle", ("skeletal muscle", "myoblast", "myotube", "muscle", "gastrocnemius", "soleus",
                "quadriceps", "tibialis")),
    ("intestine / gut", ("intestin", "colon", "colorect", "ileum", "jejunum", "duodenum",
                         "caecum", "cecum", "gut", "organoid intestinal", "rectum")),
    ("stomach / esophagus", ("stomach", "gastric", "esophag", "oesophag")),
    ("pancreas", ("pancrea", "islet", "beta cell")),
    ("skin", ("skin", "epiderm", "keratinocyte", "melanocyte", "dermal", "fibroblast dermal",
              "hair follicle")),
    ("breast", ("breast", "mammary", "mcf-7", "mcf7")),
    ("prostate", ("prostate", "lncap", "pc-3", "pc3")),
    ("ovary / uterus", ("ovary", "ovarian", "uterus", "uterine", "endometri", "placenta",
                        "cervix", "cervical", "oocyte")),
    ("testis", ("testis", "testicular", "sperm", "spermato")),
    ("adipose", ("adipose", "adipocyte", "fat pad", "white adipose", "brown adipose")),
    ("bone / cartilage", ("bone", "osteo", "chondro", "cartilage")),
    ("stem cell / iPSC", ("ipsc", "embryonic stem", "esc line", " hesc", "mesc", "pluripotent",
                          "stem cell", "embryoid", "blastocyst", "embryo")),
    ("cancer cell line", ("cell line", "hela", "hek293", "k562", "a549", "hct116", "u2os",
                          "jurkat", "sh-sy5y", "huh7", "hepg2", "tumor cell line",
                          "carcinoma cell", "xenograft")),
    ("tumor / biopsy", ("tumor", "tumour", "carcinoma", "adenocarcinoma", "melanoma", "glioma",
                        "glioblastoma", "lymphoma", "leukemia", "leukaemia", "sarcoma",
                        "metasta", "biopsy")),
    ("bacteria / microbe", ("e. coli", "escherichia", "yeast", "saccharomyces", "bacteri")),
    ("yeast/other model", ("drosophila", "zebrafish", "c. elegans", "arabidopsis")),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-samples", type=int, default=10_000,
                   help="Number of embeddings to subsample for the map (0 = all).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--color-by", choices=["species", "tissue", "both"], default="both")
    p.add_argument("--n-neighbors", type=int, default=30)
    p.add_argument("--min-dist", type=float, default=0.0)
    p.add_argument("--pca-dim", type=int, default=50,
                   help="PCA dimensions fed to UMAP (0 disables PCA).")
    p.add_argument("--drop-unknown", action="store_true",
                   help="Hide samples whose tissue could not be inferred.")
    p.add_argument("--max-tissues", type=int, default=18,
                   help="Keep only the N most frequent tissue labels.")
    p.add_argument("--rotation-period", type=float, default=60.0,
                   help="Seconds per full revolution on screen.")
    p.add_argument("--export", nargs="+", choices=["html", "mp4", "gif"],
                   default=["html", "mp4"],
                   help="Output formats (default: html mp4).")
    p.add_argument("--video-duration", type=float, default=12.0,
                   help="Length of exported MP4/GIF in seconds.")
    p.add_argument("--fps", type=int, default=30,
                   help="MP4 frame rate. GIF is capped at 15 fps to limit file size.")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--point-size", type=float, default=1.8)
    p.add_argument("--opacity", type=float, default=0.72)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--recompute", action="store_true", help="Ignore cached UMAP coordinates.")
    return p.parse_args()


def load_embeddings(idx: np.ndarray) -> np.ndarray:
    manifest = json.loads(MANIFEST.read_text())
    n, dim = manifest["total_samples"], manifest["embedding_dim"]
    mm = np.memmap(MMAP, dtype=np.float16, mode="r", shape=(n, dim))
    # Sorted fancy indexing keeps the memmap read close to sequential.
    return np.ascontiguousarray(mm[np.sort(idx)]).astype(np.float32)


def _decode(raw: np.ndarray) -> pd.Series:
    return pd.Series(raw).map(
        lambda v: v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v))


def load_archs4_metadata() -> pd.DataFrame:
    """geo_accession -> organism + free-text description, from the ARCHS4 h5 files."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / "archs4_sample_text.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    frames = []
    for organism, path in ARCHS4_H5.items():
        if not path.exists():
            print(f"  [warn] missing {path.name}; tissue labels for {organism} unavailable")
            continue
        print(f"  parsing sample metadata from {path.name} (cached after first run)")
        with h5py.File(path, "r") as f:
            g = f["meta"]["samples"]
            frames.append(pd.DataFrame({
                "geo_accession": _decode(g["geo_accession"][:]),
                "organism": organism,
                "text": (_decode(g["source_name_ch1"][:]) + " | "
                         + _decode(g["characteristics_ch1"][:]) + " | "
                         + _decode(g["title"][:])).str.lower(),
            }))
    if not frames:
        return pd.DataFrame(columns=["geo_accession", "organism", "text"])
    out = pd.concat(frames, ignore_index=True).drop_duplicates("geo_accession")
    out.to_parquet(cache, index=False)
    return out


def infer_tissue(text: pd.Series) -> pd.Series:
    labels = pd.Series(UNKNOWN_LABEL, index=text.index, dtype=object)
    unassigned = pd.Series(True, index=text.index)
    clean = text.fillna("")
    for label, keywords in TISSUE_RULES:
        pattern = "|".join(re.escape(k) for k in keywords)
        hit = unassigned & clean.str.contains(pattern, regex=True, na=False)
        labels[hit] = label
        unassigned &= ~hit
    return labels


def build_table(args: argparse.Namespace) -> pd.DataFrame:
    meta = pd.read_parquet(META, columns=["global_index", "geo_accession", "species_id"])
    rng = np.random.default_rng(args.seed)
    if args.n_samples and args.n_samples < len(meta):
        keep = rng.choice(len(meta), size=args.n_samples, replace=False)
        meta = meta.iloc[np.sort(keep)].reset_index(drop=True)

    archs4 = load_archs4_metadata()
    meta = meta.merge(archs4, on="geo_accession", how="left")
    meta["organism"] = meta["organism"].fillna(
        meta["species_id"].map({0: "human", 1: "mouse"}))
    meta["tissue"] = infer_tissue(meta["text"])

    counts = meta.loc[meta.tissue != UNKNOWN_LABEL, "tissue"].value_counts()
    keep_labels = set(counts.head(args.max_tissues).index)
    meta.loc[~meta.tissue.isin(keep_labels), "tissue"] = UNKNOWN_LABEL
    return meta.drop(columns=["text"])


def umap_3d(emb: np.ndarray, args: argparse.Namespace, cache_key: str) -> np.ndarray:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"umap3d_{cache_key}.npy"
    if cache.exists() and not args.recompute:
        print(f"  reusing cached coordinates: {cache.name}")
        return np.load(cache)

    x = emb
    if args.pca_dim and args.pca_dim < emb.shape[1]:
        print(f"  PCA {emb.shape[1]} -> {args.pca_dim}")
        x = PCA(n_components=args.pca_dim, random_state=args.seed).fit_transform(emb)

    import umap  # slow import, only needed on cache miss

    print(f"  UMAP on {x.shape[0]:,} points (this takes a while)")
    coords = umap.UMAP(
        n_components=3,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric="cosine",
        random_state=args.seed,
        verbose=True,
    ).fit_transform(x)
    coords = np.asarray(coords, dtype=np.float32)
    np.save(cache, coords)
    return coords


def normalize_coords(coords: np.ndarray) -> np.ndarray:
    """Centre and scale so the bulk of the cloud fills the cube (outliers may exceed it)."""
    coords = coords - np.median(coords, axis=0)
    scale = np.percentile(np.linalg.norm(coords, axis=1), 99.0)
    return coords / max(scale, 1e-6)


def palette(labels: list[str]) -> dict[str, str]:
    base = ["#38BDF8", "#FB7185", "#FACC15", "#4ADE80", "#C084FC", "#F97316", "#22D3EE",
            "#F472B6", "#A3E635", "#60A5FA", "#FDBA74", "#2DD4BF", "#E879F9", "#FCA5A5",
            "#93C5FD", "#BEF264", "#FDE68A", "#5EEAD4", "#D8B4FE", "#FF8FA3"]
    return {lab: base[i % len(base)] for i, lab in enumerate(labels)}


def make_figure(coords: np.ndarray, labels: pd.Series, colors: dict[str, str],
                title: str, subtitle: str, args: argparse.Namespace) -> go.Figure:
    fig = go.Figure()
    order = [lab for lab in colors if lab != UNKNOWN_LABEL]
    if (labels == UNKNOWN_LABEL).any():
        order = [UNKNOWN_LABEL] + order  # draw grey background points first
    for lab in order:
        mask = (labels == lab).to_numpy()
        if not mask.any():
            continue
        is_unknown = lab == UNKNOWN_LABEL
        fig.add_trace(go.Scatter3d(
            x=coords[mask, 0], y=coords[mask, 1], z=coords[mask, 2],
            mode="markers",
            name=f"{lab}  ({mask.sum():,})",
            marker=dict(
                size=args.point_size * (0.8 if is_unknown else 1.0),
                color=colors[lab],
                opacity=args.opacity * (0.45 if is_unknown else 1.0),
                line=dict(width=0),
            ),
            hoverinfo="skip",
            showlegend=True,
        ))

    axis = dict(showgrid=False, zeroline=False, showticklabels=False, showbackground=False,
                title="", visible=False)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#05070D",
        plot_bgcolor="#05070D",
        title=dict(
            text=f"<b>{title}</b><br><span style='font-size:18px;color:#94A3B8'>{subtitle}</span>",
            x=0.02, y=0.95, font=dict(size=34, color="#E2E8F0"),
        ),
        legend=dict(
            itemsizing="constant", font=dict(size=16, color="#E2E8F0"),
            bgcolor="rgba(5,7,13,0.55)", bordercolor="#1E293B", borderwidth=1,
            xanchor="right", x=0.995, yanchor="middle", y=0.5, tracegroupgap=2,
        ),
        scene=dict(
            xaxis=axis, yaxis=axis, zaxis=axis,
            bgcolor="#05070D", aspectmode="cube",
            camera=dict(eye=dict(x=1.05, y=1.05, z=0.55), up=dict(x=0, y=0, z=1)),
            dragmode="orbit",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
    )
    return fig


ROTATION_JS = """
<script>
(function () {
  const gd = document.getElementById("%(div_id)s");
  const RADIUS = %(radius).4f, HEIGHT = %(height).4f;
  const STEP = 2 * Math.PI / (%(period).3f * 30);  // 30 fps
  let angle = Math.atan2(%(eye_y).4f, %(eye_x).4f);
  let paused = false, resumeTimer = null, busy = false;

  function fit() { Plotly.Plots.resize(gd); }
  fit();
  window.addEventListener("resize", fit);

  function pause() {
    paused = true;
    clearTimeout(resumeTimer);
    resumeTimer = setTimeout(function () {
      const cam = (gd.layout.scene && gd.layout.scene.camera) || null;
      if (cam) angle = Math.atan2(cam.eye.y, cam.eye.x);
      paused = false;
    }, 15000);
  }
  gd.addEventListener("mousedown", pause);
  gd.addEventListener("wheel", pause);
  gd.addEventListener("touchstart", pause);

  setInterval(function () {
    if (paused || busy || document.hidden) return;
    angle += STEP;
    busy = true;
    Plotly.relayout(gd, {
      "scene.camera.eye": {
        x: RADIUS * Math.cos(angle),
        y: RADIUS * Math.sin(angle),
        z: HEIGHT,
      },
    }).then(function () { busy = false; }, function () { busy = false; });
  }, 1000 / 30);

  window.addEventListener("keydown", function (e) {
    if (e.key === "f" || e.key === "F") {
      if (document.fullscreenElement) document.exitFullscreen();
      else document.documentElement.requestFullscreen();
    }
  });
})();
</script>
"""


def write_html(fig: go.Figure, path: Path, args: argparse.Namespace, page_title: str) -> None:
    div_id = "archs4-map"
    eye = fig.layout.scene.camera.eye
    body = fig.to_html(
        include_plotlyjs="cdn", full_html=False, div_id=div_id,
        config={"displayModeBar": False, "responsive": True, "scrollZoom": True},
    )
    script = ROTATION_JS % {
        "div_id": div_id,
        "radius": float(np.hypot(eye.x, eye.y)),
        "height": float(eye.z),
        "period": args.rotation_period,
        "eye_x": float(eye.x),
        "eye_y": float(eye.y),
    }
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(page_title)}</title>
<style>
  html, body {{ margin: 0; padding: 0; height: 100%; background: #05070D;
                font-family: "Inter", "Helvetica Neue", Arial, sans-serif; overflow: hidden; }}
  #{div_id} {{ width: 100vw !important; height: 100vh !important; }}
</style>
</head>
<body>
{body}
{script}
</body>
</html>
"""
    path.write_text(page)
    print(f"  wrote {path}")


def write_animation(coords: np.ndarray, labels: pd.Series, colors: dict[str, str],
                    title: str, subtitle: str, stem: Path,
                    args: argparse.Namespace) -> None:
    """Render a clean 16:9 orbit without requiring a browser or Kaleido."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
    from matplotlib.lines import Line2D

    bg = "#05070D"
    fg = "#E2E8F0"
    fig = plt.figure(figsize=(args.width / args.dpi, args.height / args.dpi),
                     dpi=args.dpi, facecolor=bg)
    ax = fig.add_axes((0.0, 0.0, 0.82, 1.0), projection="3d", facecolor=bg)
    order = [lab for lab in colors if lab != UNKNOWN_LABEL]
    if (labels == UNKNOWN_LABEL).any():
        order = [UNKNOWN_LABEL] + order

    handles = []
    for lab in order:
        mask = (labels == lab).to_numpy()
        if not mask.any():
            continue
        unknown = lab == UNKNOWN_LABEL
        size = (args.point_size * (0.8 if unknown else 1.0)) ** 2
        alpha = args.opacity * (0.35 if unknown else 1.0)
        ax.scatter(coords[mask, 0], coords[mask, 1], coords[mask, 2],
                   s=size, c=colors[lab], alpha=alpha, edgecolors="none",
                   depthshade=False, rasterized=True)
        handles.append(Line2D([], [], linestyle="", marker="o", markersize=7,
                              markerfacecolor=colors[lab], markeredgewidth=0,
                              label=f"{lab}  ({mask.sum():,})"))

    # Use the occupied range rather than the normalization sphere's full diameter;
    # this keeps the cloud visually prominent in a widescreen slide.
    limit = max(0.55, float(np.quantile(np.abs(coords), 0.999)) * 1.08)
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit))
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=20, azim=35)
    fig.text(0.035, 0.93, title, color=fg, fontsize=28, weight="bold", va="top")
    fig.text(0.035, 0.875, subtitle, color="#94A3B8", fontsize=15, va="top")
    fig.legend(handles=handles, loc="center right", bbox_to_anchor=(0.985, 0.5),
               frameon=False, labelcolor=fg, fontsize=11, markerscale=1.0)

    def rotate(frame: int, frames: int) -> tuple:
        ax.view_init(elev=20, azim=35 + 360.0 * frame / frames)
        return (ax,)

    for fmt in args.export:
        if fmt not in ("mp4", "gif"):
            continue
        fps = args.fps if fmt == "mp4" else min(args.fps, 15)
        frames = max(2, round(args.video_duration * fps))
        animation = FuncAnimation(fig, lambda i: rotate(i, frames), frames=frames,
                                  interval=1000 / fps, blit=False)
        path = stem.with_suffix(f".{fmt}")
        print(f"  rendering {path.name}: {frames} frames at {fps} fps")
        if fmt == "mp4":
            writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=8_000,
                                  extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
        else:
            writer = PillowWriter(fps=fps)
        animation.save(path, writer=writer, dpi=args.dpi)
        print(f"  wrote {path}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading metadata ...")
    table = build_table(args)
    print(f"  {len(table):,} samples | "
          f"{table.organism.value_counts().to_dict()} | "
          f"{table.tissue.nunique()} tissue labels")

    print("Loading embeddings ...")
    emb = load_embeddings(table["global_index"].to_numpy())

    key = f"n{len(table)}_s{args.seed}_nn{args.n_neighbors}_md{args.min_dist}_p{args.pca_dim}"
    print("Computing 3D UMAP ...")
    coords = normalize_coords(umap_3d(emb, args, key))

    n_total = len(table)
    if args.color_by in ("species", "both"):
        labels = table["organism"].fillna(UNKNOWN_LABEL)
        colors = dict(SPECIES_COLORS, **{UNKNOWN_LABEL: UNKNOWN_COLOR})
        title = "ARCHS4 sample embeddings"
        subtitle = f"3D UMAP of {n_total:,} RNA-seq samples · coloured by species"
        if "html" in args.export:
            fig = make_figure(coords, labels, colors, title, subtitle, args)
            write_html(fig, args.out_dir / "archs4_umap3d_species.html", args,
                       "ARCHS4 embedding map - species")
        if {"mp4", "gif"}.intersection(args.export):
            write_animation(coords, labels, colors, title, subtitle,
                            args.out_dir / "archs4_umap3d_species", args)

    if args.color_by in ("tissue", "both"):
        sub = table
        sub_coords = coords
        if args.drop_unknown:
            mask = (table.tissue != UNKNOWN_LABEL).to_numpy()
            sub, sub_coords = table[mask].reset_index(drop=True), coords[mask]
        labels = sub["tissue"]
        ordered = list(labels.value_counts().index)
        ordered = [l for l in ordered if l != UNKNOWN_LABEL]
        colors = palette(ordered)
        colors[UNKNOWN_LABEL] = UNKNOWN_COLOR
        title = "ARCHS4 sample embeddings"
        subtitle = (f"3D UMAP of {len(sub):,} RNA-seq samples · "
                    "coloured by tissue / sample type")
        if "html" in args.export:
            fig = make_figure(sub_coords, labels, colors, title, subtitle, args)
            write_html(fig, args.out_dir / "archs4_umap3d_tissue.html", args,
                       "ARCHS4 embedding map - tissue")
        if {"mp4", "gif"}.intersection(args.export):
            write_animation(sub_coords, labels, colors, title, subtitle,
                            args.out_dir / "archs4_umap3d_tissue", args)


if __name__ == "__main__":
    main()
