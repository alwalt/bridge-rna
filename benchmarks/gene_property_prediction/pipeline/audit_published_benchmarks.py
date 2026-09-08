#!/usr/bin/env python3
"""Audit published gene-property benchmarks without running BridgeRNA inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import urllib.request
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "benchmarks" / "gene_property_prediction"
GENEFORMER_LABEL_URL = (
    "https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/resolve/main/"
    "example_input_files/gene_classification/dosage_sensitive_tfs/"
    "dosage_sensitivity_TFs.pickle"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_exact_geneformer_labels(source_dir: Path) -> Path:
    target = source_dir / "geneformer_dosage_sensitivity_TFs.pickle"
    if not target.exists():
        urllib.request.urlretrieve(GENEFORMER_LABEL_URL, target)
    return target


def load_hgnc_mapping() -> dict[str, str]:
    path = ROOT / "data/annotations/hgnc/hgnc_complete_set_2026-08-27.tsv"
    hgnc = pd.read_csv(path, sep="\t", low_memory=False)
    valid = hgnc["ensembl_gene_id"].notna() & hgnc["symbol"].notna()
    return dict(
        zip(
            hgnc.loc[valid, "ensembl_gene_id"].str.replace(r"\..*$", "", regex=True),
            hgnc.loc[valid, "symbol"].str.upper(),
        )
    )


def process_dosage_labels(label_path: Path, processed_dir: Path) -> pd.DataFrame:
    with label_path.open("rb") as handle:
        labels = pickle.load(handle)
    canonical = pd.read_csv(ROOT / "data/ensembl/canonical_genes.csv")
    canonical_symbols = set(canonical["gene_symbol"].str.upper())
    ensembl_to_symbol = load_hgnc_mapping()
    rows = []
    class_map = {"Dosage-sensitive TFs": 1, "Dosage-insensitive TFs": 0}
    for published_class, identifiers in labels.items():
        for gene_id in identifiers:
            clean_id = gene_id.split(".")[0]
            symbol = ensembl_to_symbol.get(clean_id)
            rows.append(
                {
                    "gene_id": clean_id,
                    "gene_symbol": symbol,
                    "label": class_map[published_class],
                    "published_class": published_class,
                    "source": "Geneformer Genecorpus-30M exact published label pickle",
                    "published_split": "5-fold stratified CV; no fixed fold assignment distributed",
                    "maps_to_bridgerna": bool(symbol in canonical_symbols if symbol else False),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(processed_dir / "geneformer_dosage_sensitivity_labels.csv", index=False)
    return frame


def checkpoint_inventory() -> dict:
    checkpoint_path = ROOT / "model/r7hnr92k/best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    tensor = state["gene_embedding.weight"]
    return {
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "static_tensor_key": "model_state_dict.gene_embedding.weight",
        "static_shape": list(tensor.shape),
        "static_dtype": str(tensor.dtype),
        "static_output_bytes_float32": tensor.numel() * 4,
    }


def build_audit(dosage: pd.DataFrame, source_path: Path) -> tuple[pd.DataFrame, dict]:
    matrix_audit_path = BENCH / "results/audit/bulkformer_matrix_audit.json"
    matrix_audit = json.loads(matrix_audit_path.read_text()) if matrix_audit_path.exists() else None
    bulk_overlap = (
        f"{matrix_audit['score_bridge_mapped']}/{matrix_audit['score_shape'][1]} target genes; "
        f"{matrix_audit['expression_bridge_mapped']}/{matrix_audit['expression_shape'][1]} input genes"
        if matrix_audit
        else "UNKNOWN until the expression/score pair is downloaded and inspected"
    )
    sensitive = dosage[dosage.label.eq(1)]
    insensitive = dosage[dosage.label.eq(0)]
    rows = [
        {
            "benchmark": "BulkFormer gene essentiality",
            "task_type": "sample-conditioned continuous per-gene dependency-score prediction (not fixed-gene classification)",
            "label_resource": "gene_essentiality_score.pkl (159.1 MB, Zenodo record 15744294)",
            "input_resource": "gene_essentiality_expr_data.pkl (166.6 MB, same record)",
            "reported_metric": "mean Pearson correlation coefficient (0.186 for BulkFormer)",
            "exact_labels_obtainable": True,
            "exact_protocol_status": "PARTIAL: exact matrices recovered; preprint says contextual final-layer gene embedding + MLP and 10-fold CV, but final split/readout details are not distributed",
            "bridge_overlap": bulk_overlap,
        },
        {
            "benchmark": "Geneformer dosage sensitivity",
            "task_type": "binary classification of dosage-sensitive vs dosage-insensitive transcription factors",
            "label_resource": "dosage_sensitivity_TFs.pickle",
            "input_resource": "10,000 random cells from a 50,000-cell Genecorpus-30M example dataset",
            "reported_metric": "ROC AUC (0.91 in paper)",
            "exact_labels_obtainable": True,
            "exact_protocol_status": "RECOVERED: 5-fold stratified gene CV, shuffled random_state=0; token-classification fine-tuning",
            "bridge_overlap": f"{int(dosage.maps_to_bridgerna.sum())}/490 total; {int(sensitive.maps_to_bridgerna.sum())}/122 sensitive; {int(insensitive.maps_to_bridgerna.sum())}/368 insensitive",
        },
        {
            "benchmark": "GeneCompass dosage sensitivity",
            "task_type": "same binary TF dosage-sensitivity task, following Geneformer",
            "label_resource": "No separate labels distributed in current official repository; paper states Geneformer protocol",
            "input_resource": "10,000 random single-cell transcriptomes",
            "reported_metric": "ROC AUC (~0.95 in paper figure/text)",
            "exact_labels_obtainable": True,
            "exact_protocol_status": "PARTIAL: paper specifies Geneformer protocol/10,000 cells; exact split and training implementation absent from official repo",
            "bridge_overlap": "Same exact Geneformer labels if protocol identity is confirmed; otherwise UNKNOWN",
        },
    ]
    details = {
        "geneformer_label_url": GENEFORMER_LABEL_URL,
        "geneformer_label_sha256": sha256(source_path),
        "geneformer_original_counts": {"positive": len(sensitive), "negative": len(insensitive)},
        "geneformer_bridge_counts": {
            "positive": int(sensitive.maps_to_bridgerna.sum()),
            "negative": int(insensitive.maps_to_bridgerna.sum()),
            "total": int(dosage.maps_to_bridgerna.sum()),
        },
        "bridgerna": checkpoint_inventory(),
        "bulkformer_matrix_audit": matrix_audit,
        "context_average": {
            "compatible_complete_cache_found": False,
            "existing_reference_expression": "40,000 x 15,165 ARCHS4 log1p(TPM) memmap",
            "existing_sample_embeddings_are_not_gene_embeddings": True,
            "streaming_output_bytes_float32": 15165 * 512 * 4,
            "full_hidden_cache_bytes_per_sample_float16": 15165 * 512 * 2,
            "estimated_inference_rate_samples_per_second": "7.5-8.7 based on existing reconstruction logs; contextual export may be slower",
            "estimated_1000_sample_wall_time": "~2-10 minutes inference plus I/O on one RTX 3090",
            "estimated_5000_sample_wall_time": "~10-50 minutes inference plus I/O on one RTX 3090",
            "recommendation": "stream a running per-gene mean; never save all sample x gene x 512 tensors",
        },
    }
    return pd.DataFrame(rows), details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-download", action="store_true", help="Require the exact Geneformer label file to exist already.")
    args = parser.parse_args()
    source_dir = BENCH / "data/source"
    processed_dir = BENCH / "data/processed"
    output_dir = BENCH / "results/audit"
    for directory in (source_dir, processed_dir, output_dir):
        directory.mkdir(parents=True, exist_ok=True)
    label_path = source_dir / "geneformer_dosage_sensitivity_TFs.pickle"
    if not label_path.exists() and args.no_download:
        raise FileNotFoundError(label_path)
    if not label_path.exists():
        label_path = download_exact_geneformer_labels(source_dir)
    dosage = process_dosage_labels(label_path, processed_dir)
    audit, details = build_audit(dosage, label_path)
    audit.to_csv(output_dir / "published_benchmark_audit.csv", index=False)
    (output_dir / "audit_details.json").write_text(json.dumps(details, indent=2) + "\n")
    print(audit.to_string(index=False))
    print("\nNo BridgeRNA inference or downstream evaluation was run.")


if __name__ == "__main__":
    main()
