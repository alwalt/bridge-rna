"""TCGA source adapter.

The TCGA TPM parquet in this repo is already TPM-normalized: index is
sample/file id, columns are gene symbols.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd

DEFAULT_TCGA_TPM = Path("/home/walt/Attention/data/tcga/tcga_tpm_unstranded_matrix.parquet")
DEFAULT_TCGA_H5 = Path("data/tcga/tcga_matrix.h5")


def load_tcga_matrix(
    tpm_parquet_path: Path = DEFAULT_TCGA_TPM,
    sample_ids: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, bool]:
    """Return (matrix indexed by sample id with gene columns, already_tpm=True)."""
    matrix = pd.read_parquet(tpm_parquet_path)
    matrix.index = matrix.index.astype(str)
    if sample_ids is not None:
        matrix = matrix.reindex([str(s) for s in sample_ids])
    return matrix, True


def _decode(values: np.ndarray) -> list[str]:
    return [value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
            for value in values]


def load_tcga_h5_counts(
    h5_path: Path | str = DEFAULT_TCGA_H5,
    sample_ids: Optional[Sequence[str]] = None,
    metadata_fields: Sequence[str] = (
        "meta/gdc_cases.samples.submitter_id", "meta/cancertype",
        "meta/gdc_cases.project.project_id", "meta/gdc_cases.samples.sample_type",
    ),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load selected raw-count TCGA samples without reading the full H5 matrix.

    The returned expression frame is sample-by-gene and indexed by the stable
    ``meta/sampleid`` UUID. Metadata uses the same index. HDF5 requires sorted
    row indices for fancy indexing, so requested samples are restored to the
    caller's order after reading.
    """
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as handle:
        all_ids = _decode(handle["meta/sampleid"][:])
        id_to_row = {sample_id: row for row, sample_id in enumerate(all_ids)}
        if sample_ids is None:
            requested = all_ids
        else:
            requested = [str(sample_id) for sample_id in sample_ids]
            missing = [sample_id for sample_id in requested if sample_id not in id_to_row]
            if missing:
                raise KeyError(f"TCGA H5 is missing {len(missing)} requested IDs: {missing[:5]}")
        requested_rows = np.asarray([id_to_row[sample_id] for sample_id in requested], dtype=np.int64)
        sort_order = np.argsort(requested_rows)
        sorted_rows = requested_rows[sort_order]
        counts_sorted = handle["data/expression"][sorted_rows, :]
        restore = np.argsort(sort_order)
        counts = counts_sorted[restore]
        genes = _decode(handle["meta/genes"][:])
        metadata = {"sample_id": requested}
        for field in metadata_fields:
            if field in handle:
                key = field.removeprefix("meta/").replace(".", "_")
                metadata[key] = np.asarray(_decode(handle[field][sorted_rows]))[restore]
    return (
        pd.DataFrame(counts, index=pd.Index(requested, name="sample_id"), columns=genes),
        pd.DataFrame(metadata).set_index("sample_id"),
    )
