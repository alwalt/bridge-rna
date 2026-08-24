#!/usr/bin/env python3
"""Generate 512-D ExpressionPerformer embeddings for TCGA, GTEx, OSDR, or GEO.

Uses the exact same canonical gene vocabulary alignment and log1p(TPM)
normalization as the ARCHS4 training/embedding pipeline, so embeddings from
different sources stay comparable.

Examples:
    python -m fm_embed.generate_embeddings --source tcga --output-dir prepared_data/tcga_embeddings
    python -m fm_embed.generate_embeddings --source gtex --output-dir prepared_data/gtex_embeddings
    python -m fm_embed.generate_embeddings --source osdr \\
        --osdr-counts-csv data/osdr/raw/GLDS-100_rna_seq_Unnormalized_Counts.csv \\
        --output-dir prepared_data/osdr_embeddings
    # GEO series already indexed by ARCHS4: just look up precomputed embeddings.
    python -m fm_embed.generate_embeddings --source geo --geo-accessions-file gsms.txt \\
        --output-dir prepared_data/geo_embeddings
    # GEO series NOT in ARCHS4: process a raw counts matrix (human or mouse).
    python -m fm_embed.generate_embeddings --source geo --geo-counts-csv GSExxxxx_counts.csv \\
        --output-dir prepared_data/geo_embeddings
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .vocab import load_canonical_genes
from .model import load_expression_performer
from .transform import align_to_vocab, apply_preprocessing
from .encode import encode_matrix
from .sources.tcga import DEFAULT_TCGA_TPM, load_tcga_matrix
from .sources.gtex import DEFAULT_GTEX_PARQUET, load_gtex_matrix
from .sources.osdr import load_osdr_matrix
from .sources.geo import DEFAULT_ARCHS4_EMBEDDING_DIR, lookup_archs4_embeddings, load_geo_counts_matrix
from .species import DEFAULT_ORTHOLOGS, DEFAULT_MOUSE_EXON_LENGTHS, DEFAULT_HUMAN_EXON_LENGTHS

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_id_list(path: Optional[Path]) -> Optional[List[str]]:
    if path is None:
        return None
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["tcga", "gtex", "osdr", "geo"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--preprocessing",
        choices=["auto", "raw", "cpm", "tpm", "log1p_raw", "log1p_cpm", "log1p_tpm"],
        default="auto",
        help="auto picks log1p_tpm for already-TPM sources (TCGA/OSDR) and log1p_cpm for raw-count sources (GTEx).",
    )
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "r7hnr92k" / "best_model.pt")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "r7hnr92k" / "config.json")
    parser.add_argument("--canonical-genes", type=Path, default=REPO_ROOT / "data/archs4/train_orthologs/canonical_genes.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sample-ids-file", type=Path, default=None, help="Optional newline-delimited file restricting which samples to embed.")

    parser.add_argument("--tcga-tpm-parquet", type=Path, default=DEFAULT_TCGA_TPM)

    parser.add_argument("--gtex-parquet", type=Path, default=DEFAULT_GTEX_PARQUET)

    parser.add_argument("--osdr-counts-csv", type=Path, default=None)
    parser.add_argument("--osdr-orthologs", type=Path, default=REPO_ROOT / DEFAULT_ORTHOLOGS)
    parser.add_argument("--osdr-mouse-exon-lengths", type=Path, default=REPO_ROOT / DEFAULT_MOUSE_EXON_LENGTHS)

    parser.add_argument(
        "--geo-accessions-file",
        type=Path,
        default=None,
        help="Newline-delimited GEO accessions already indexed by ARCHS4; skips reprocessing and reads precomputed embeddings.",
    )
    parser.add_argument("--archs4-embedding-dir", type=Path, default=REPO_ROOT / DEFAULT_ARCHS4_EMBEDDING_DIR)
    parser.add_argument("--geo-counts-csv", type=Path, default=None, help="Raw counts matrix for a GEO series not indexed by ARCHS4.")
    parser.add_argument("--geo-species", choices=["human", "mouse"], default=None, help="Required with --geo-counts-csv; never guessed.")
    parser.add_argument("--geo-orthologs", type=Path, default=REPO_ROOT / DEFAULT_ORTHOLOGS)
    parser.add_argument("--geo-mouse-exon-lengths", type=Path, default=REPO_ROOT / DEFAULT_MOUSE_EXON_LENGTHS)
    parser.add_argument("--geo-human-exon-lengths", type=Path, default=REPO_ROOT / DEFAULT_HUMAN_EXON_LENGTHS)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_ids = _read_id_list(args.sample_ids_file)

    # GEO accessions already indexed by ARCHS4 skip preprocessing entirely:
    # read their precomputed embeddings straight from the ARCHS4 archive.
    if args.source == "geo" and args.geo_accessions_file is not None:
        geo_accessions = _read_id_list(args.geo_accessions_file)
        out_df = lookup_archs4_embeddings(geo_accessions, args.archs4_embedding_dir)
        out_dir = args.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        emb_parquet = out_dir / "geo_embeddings_512d.parquet"
        out_df.to_parquet(emb_parquet, index=False)
        found = len(out_df)
        print(f"Found {found}/{len(geo_accessions)} accessions already in the ARCHS4 archive.")
        missing = sorted(set(geo_accessions) - set(out_df["geo_accession"]))
        if missing:
            print(f"Not indexed by ARCHS4 (use --geo-counts-csv instead): {missing[:10]}{'...' if len(missing) > 10 else ''}")
        print(f"Saved embeddings: {emb_parquet}")
        return

    if args.source == "tcga":
        matrix, already_tpm = load_tcga_matrix(args.tcga_tpm_parquet, sample_ids)
        source_path = str(args.tcga_tpm_parquet)
    elif args.source == "gtex":
        matrix, already_tpm = load_gtex_matrix(args.gtex_parquet, sample_ids)
        source_path = str(args.gtex_parquet)
    elif args.source == "osdr":
        if args.osdr_counts_csv is None:
            raise ValueError("--osdr-counts-csv is required when --source osdr")
        matrix, already_tpm = load_osdr_matrix(
            args.osdr_counts_csv, sample_ids, args.osdr_orthologs, args.osdr_mouse_exon_lengths
        )
        source_path = str(args.osdr_counts_csv)
    else:
        if args.geo_counts_csv is None:
            raise ValueError("--geo-counts-csv or --geo-accessions-file is required when --source geo")
        if args.geo_species is None:
            raise ValueError("--geo-species human|mouse is required with --geo-counts-csv")
        matrix, already_tpm = load_geo_counts_matrix(
            args.geo_counts_csv, args.geo_species, sample_ids,
            args.geo_orthologs, args.geo_human_exon_lengths, args.geo_mouse_exon_lengths,
        )
        source_path = str(args.geo_counts_csv)

    preprocessing = args.preprocessing
    if preprocessing == "auto":
        preprocessing = "log1p_tpm" if already_tpm else "log1p_cpm"

    canonical_genes = load_canonical_genes(args.canonical_genes)
    aligned = align_to_vocab(matrix, canonical_genes, genes_are_columns=True)
    preprocessed = apply_preprocessing(aligned, mode=preprocessing)

    model, device = load_expression_performer(args.checkpoint, args.config, num_genes=len(canonical_genes), device=args.device)
    embeddings = encode_matrix(model, device, preprocessed, batch_size=args.batch_size, label=f"{args.source}_encoding")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_df = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(embeddings.shape[1])])
    emb_df.insert(0, "sample_id", matrix.index.astype(str).to_numpy())

    emb_parquet = out_dir / f"{args.source}_embeddings_512d.parquet"
    emb_npy = out_dir / f"{args.source}_embeddings_512d.npy"
    manifest_path = out_dir / "embedding_manifest.json"

    emb_df.to_parquet(emb_parquet, index=False)
    np.save(emb_npy, embeddings)
    manifest_path.write_text(json.dumps({
        "source": args.source,
        "source_path": source_path,
        "num_samples": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "preprocessing": preprocessing,
        "canonical_gene_count": len(canonical_genes),
        "checkpoint": str(args.checkpoint),
    }, indent=2))

    print(f"Saved embeddings: {emb_parquet}")
    print(f"Saved embeddings: {emb_npy}")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
