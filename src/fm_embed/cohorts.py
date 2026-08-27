"""Reproducible cohort selection and precomputed ARCHS4 embedding access.

Purpose
-------
The sample manifest is the source of truth for *which* samples belong to a
benchmark. The ARCHS4 embedding memmap is the source of truth for their frozen
512-D model representations. This module joins those artifacts by GSM and keeps
cohort definitions out of notebooks and one-off analysis scripts.

No expression preprocessing or model inference happens here. In particular,
this module does not perform CPM, TPM, or any other normalization; it only reads
the already-generated ARCHS4 embeddings.

Typical use
-----------
    from fm_embed.cohorts import load_archs4_cohort

    cohort = load_archs4_cohort("strict_unseen_single_gse", species="human")
    metadata = cohort.metadata
    embeddings = cohort.load_embeddings()  # float32 ndarray, when it fits RAM

For larger analyses, avoid materializing the entire cohort:

    for metadata_batch, embedding_batch in cohort.iter_batches(batch_size=4096):
        run_analysis(metadata_batch, embedding_batch)

To freeze the exact ordered sample list used by an experiment without copying
the embedding matrix:

    cohort.write_index("data/cohorts/my_benchmark.parquet")

The exported index includes ``global_index``. That value is the row position in
the ARCHS4 memmap and must be used instead of the manifest row number.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE_MANIFEST = REPO_ROOT / "data" / "manifests" / "sample_manifest.parquet"
DEFAULT_EMBEDDING_DIR = REPO_ROOT / "embeddings" / "archs4"

REQUIRED_MANIFEST_COLUMNS = {
    "sample_id", "gsm", "split", "study_exposure", "mapping_status", "species",
}
REQUIRED_LOCATION_COLUMNS = {"global_index", "geo_accession"}
MAPPED_STATUSES = {"mapped_single", "mapped_multiple"}

COHORT_DESCRIPTIONS = {
    "train": "Samples used to fit the foundation model.",
    "validation": "Samples used for model validation (split='val').",
    "unseen": "All processed samples excluded from train and validation.",
    "unseen_sample_seen_study": (
        "Held-out GSMs for which at least one candidate GSE occurred in training."
    ),
    "unseen_sample_unseen_study": (
        "Held-out GSMs with a mapping and no candidate GSE represented in training."
    ),
    "strict_unseen_single_gse": (
        "Study-unseen samples with exactly one unambiguous GSE; recommended primary benchmark."
    ),
    "strict_unseen_multiple_gse": (
        "Study-unseen samples with multiple candidate GSEs; useful sensitivity cohort."
    ),
}


def available_cohorts() -> dict[str, str]:
    """Return stable named cohort definitions and their human-readable purpose."""
    return dict(COHORT_DESCRIPTIONS)


def _cohort_mask(manifest: pd.DataFrame, name: str) -> pd.Series:
    if name not in COHORT_DESCRIPTIONS:
        choices = ", ".join(COHORT_DESCRIPTIONS)
        raise ValueError(f"Unknown cohort {name!r}. Choose one of: {choices}")

    split = manifest["split"].astype("string")
    exposure = manifest["study_exposure"].astype("string")
    status = manifest["mapping_status"].astype("string")
    mapped = status.isin(MAPPED_STATUSES)

    masks = {
        "train": split.eq("train"),
        "validation": split.eq("val"),
        "unseen": split.eq("unseen"),
        "unseen_sample_seen_study": split.eq("unseen") & exposure.eq("seen_study"),
        "unseen_sample_unseen_study": (
            split.eq("unseen") & exposure.eq("unseen_study") & mapped
        ),
        "strict_unseen_single_gse": (
            split.eq("unseen") & exposure.eq("unseen_study")
            & status.eq("mapped_single")
        ),
        "strict_unseen_multiple_gse": (
            split.eq("unseen") & exposure.eq("unseen_study")
            & status.eq("mapped_multiple")
        ),
    }
    return masks[name].fillna(False)


def select_manifest_cohort(
    manifest: pd.DataFrame,
    name: str,
    species: str | None = None,
) -> pd.DataFrame:
    """Select a named cohort from an already-loaded sample manifest.

    The returned rows are copied. Ordering is finalized only after joining the
    embedding locations, where rows are sorted by ``global_index``.
    """
    missing = REQUIRED_MANIFEST_COLUMNS - set(manifest.columns)
    if missing:
        raise ValueError(f"Sample manifest is missing required columns: {sorted(missing)}")
    mask = _cohort_mask(manifest, name)
    if species is not None:
        normalized_species = species.strip().lower()
        if normalized_species not in {"human", "mouse"}:
            raise ValueError("species must be 'human', 'mouse', or None")
        mask &= manifest["species"].astype("string").str.lower().eq(normalized_species)
    return manifest.loc[mask].copy()


def _load_embedding_spec(embedding_dir: Path) -> tuple[Path, int, int, np.dtype]:
    metadata_path = embedding_dir / "embedding_manifest.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing embedding manifest: {metadata_path}")
    spec = json.loads(metadata_path.read_text())
    total = int(spec["total_samples"])
    dimension = int(spec["embedding_dim"])
    dtype = np.dtype(spec.get("embedding_dtype", "float16"))
    embedding_path = embedding_dir / f"sample_embeddings.{dtype.name}.mmap"
    if not embedding_path.is_file():
        raise FileNotFoundError(f"Missing embedding memmap: {embedding_path}")
    expected_bytes = total * dimension * dtype.itemsize
    actual_bytes = embedding_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"Embedding memmap size mismatch: expected {expected_bytes:,} bytes, "
            f"found {actual_bytes:,} bytes"
        )
    return embedding_path, total, dimension, dtype


def _ordered_id_sha256(metadata: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for gsm in metadata["gsm"].astype(str):
        digest.update(gsm.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True)
class Archs4Cohort:
    """A validated metadata selection with lazy access to memmapped embeddings."""

    name: str
    species: str | None
    metadata: pd.DataFrame
    embedding_path: Path
    total_embeddings: int
    embedding_dim: int
    embedding_dtype: np.dtype
    sample_manifest_path: Path
    locations_path: Path
    archs4_metadata_path: Path | None = None

    def __len__(self) -> int:
        return len(self.metadata)

    @property
    def ordered_gsm_sha256(self) -> str:
        """Checksum of ordered GSMs for experiment provenance."""
        return _ordered_id_sha256(self.metadata)

    def _memmap(self) -> np.memmap:
        return np.memmap(
            self.embedding_path,
            dtype=self.embedding_dtype,
            mode="r",
            shape=(self.total_embeddings, self.embedding_dim),
        )

    def load_embeddings(self, dtype: np.dtype | type = np.float32) -> np.ndarray:
        """Materialize the cohort embeddings in deterministic metadata order."""
        indices = self.metadata["global_index"].to_numpy(dtype=np.int64)
        return np.asarray(self._memmap()[indices], dtype=dtype)

    def iter_batches(
        self, batch_size: int = 4096, dtype: np.dtype | type = np.float32
    ) -> Iterator[tuple[pd.DataFrame, np.ndarray]]:
        """Yield aligned metadata/embedding batches without loading the cohort at once."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        vectors = self._memmap()
        for start in range(0, len(self), batch_size):
            stop = min(start + batch_size, len(self))
            metadata_batch = self.metadata.iloc[start:stop].copy()
            indices = metadata_batch["global_index"].to_numpy(dtype=np.int64)
            embedding_batch = np.asarray(vectors[indices], dtype=dtype)
            yield metadata_batch, embedding_batch

    def write_index(self, path: Path | str) -> Path:
        """Write the ordered cohort metadata/index, not a copy of its embeddings."""
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        self.metadata.to_parquet(output, index=False)
        return output

    def provenance(self) -> dict[str, object]:
        """Return serializable facts that should accompany downstream results."""
        return {
            "cohort": self.name,
            "cohort_description": COHORT_DESCRIPTIONS[self.name],
            "species_filter": self.species,
            "num_samples": len(self),
            "ordered_gsm_sha256": self.ordered_gsm_sha256,
            "sample_manifest": str(self.sample_manifest_path),
            "sample_locations": str(self.locations_path),
            "archs4_metadata": (
                str(self.archs4_metadata_path) if self.archs4_metadata_path else None
            ),
            "embedding_path": str(self.embedding_path),
            "embedding_dim": self.embedding_dim,
            "embedding_dtype": self.embedding_dtype.name,
        }


def load_archs4_cohort(
    name: str,
    species: str | None = None,
    sample_manifest_path: Path | str = DEFAULT_SAMPLE_MANIFEST,
    embedding_dir: Path | str = DEFAULT_EMBEDDING_DIR,
    include_archs4_metadata: bool = False,
    archs4_metadata_path: Path | str | None = None,
    metadata_columns: list[str] | None = None,
) -> Archs4Cohort:
    """Load, validate, and join one named cohort to precomputed ARCHS4 rows.

    Every selected GSM must have exactly one location. Set
    ``include_archs4_metadata=True`` to join a versioned snapshot produced by
    ``scripts/data_audit/build_archs4_sample_metadata.py``. Pass an explicit
    ``archs4_metadata_path`` when multiple versions exist. ``metadata_columns``
    can restrict the raw fields joined (``gsm`` is always included).

    Returned rows are sorted by ``global_index`` for deterministic,
    near-sequential memmap reads.
    """
    manifest_path = Path(sample_manifest_path).expanduser().resolve()
    directory = Path(embedding_dir).expanduser().resolve()
    locations_path = directory / "sample_locations.parquet"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing sample manifest: {manifest_path}")
    if not locations_path.is_file():
        raise FileNotFoundError(f"Missing sample locations: {locations_path}")

    manifest = pd.read_parquet(manifest_path)
    selected = select_manifest_cohort(manifest, name, species)
    if selected["gsm"].isna().any() or selected["gsm"].duplicated().any():
        raise ValueError(f"Cohort {name!r} contains missing or duplicate GSMs")

    locations = pd.read_parquet(locations_path)
    missing_location_columns = REQUIRED_LOCATION_COLUMNS - set(locations.columns)
    if missing_location_columns:
        raise ValueError(
            f"Sample locations are missing columns: {sorted(missing_location_columns)}"
        )
    if locations["geo_accession"].isna().any() or locations["geo_accession"].duplicated().any():
        raise ValueError("Sample locations contain missing or duplicate GEO accessions")

    joined = selected.merge(
        locations,
        left_on="gsm",
        right_on="geo_accession",
        how="left",
        validate="one_to_one",
    )
    if joined["global_index"].isna().any():
        examples = joined.loc[joined["global_index"].isna(), "gsm"].head(10).tolist()
        raise ValueError(
            f"{joined['global_index'].isna().sum():,} cohort GSMs lack embeddings; "
            f"examples: {examples}"
        )

    embedding_path, total, dimension, dtype = _load_embedding_spec(directory)
    indices = joined["global_index"].astype(np.int64)
    if indices.lt(0).any() or indices.ge(total).any():
        raise ValueError("Cohort contains global_index values outside the embedding memmap")
    if indices.duplicated().any():
        raise ValueError("Cohort contains duplicate global_index values")
    joined["global_index"] = indices

    resolved_metadata_path: Path | None = None
    if include_archs4_metadata:
        if archs4_metadata_path is None:
            candidates = sorted(
                (REPO_ROOT / "data" / "manifests").glob(
                    "archs4_sample_metadata_v*.parquet"
                )
            )
            if not candidates:
                raise FileNotFoundError(
                    "No versioned ARCHS4 metadata snapshot found. Run "
                    "scripts/data_audit/build_archs4_sample_metadata.py first."
                )
            if len(candidates) > 1:
                raise ValueError(
                    "Multiple ARCHS4 metadata snapshots exist; pass "
                    f"archs4_metadata_path explicitly: {candidates}"
                )
            resolved_metadata_path = candidates[0].resolve()
        else:
            resolved_metadata_path = Path(archs4_metadata_path).expanduser().resolve()
        if not resolved_metadata_path.is_file():
            raise FileNotFoundError(
                f"Missing ARCHS4 metadata snapshot: {resolved_metadata_path}"
            )

        available_columns = pq.ParquetFile(resolved_metadata_path).schema_arrow.names
        if "gsm" not in available_columns:
            raise ValueError("ARCHS4 metadata snapshot is missing the 'gsm' column")
        requested_columns = ["gsm"]
        if metadata_columns is None:
            requested_columns.extend(column for column in available_columns if column != "gsm")
        else:
            missing_metadata_columns = set(metadata_columns) - set(available_columns)
            if missing_metadata_columns:
                raise ValueError(
                    "ARCHS4 metadata snapshot lacks requested columns: "
                    f"{sorted(missing_metadata_columns)}"
                )
            requested_columns.extend(column for column in metadata_columns if column != "gsm")
        raw_metadata = pd.read_parquet(
            resolved_metadata_path, columns=requested_columns
        )
        if raw_metadata["gsm"].isna().any() or raw_metadata["gsm"].duplicated().any():
            raise ValueError("ARCHS4 metadata snapshot contains missing or duplicate GSMs")
        overlap = (set(joined.columns) & set(raw_metadata.columns)) - {"gsm"}
        if overlap:
            raise ValueError(
                f"ARCHS4 metadata columns collide with cohort columns: {sorted(overlap)}"
            )
        joined = joined.merge(raw_metadata, on="gsm", how="left", validate="one_to_one")
        added_columns = [column for column in requested_columns if column != "gsm"]
        if added_columns and joined[added_columns].isna().all(axis=1).any():
            examples = joined.loc[
                joined[added_columns].isna().all(axis=1), "gsm"
            ].head(10).tolist()
            raise ValueError(
                "ARCHS4 metadata snapshot lacks cohort rows; examples: "
                f"{examples}"
            )

    joined = joined.sort_values("global_index").reset_index(drop=True)

    normalized_species = species.strip().lower() if species is not None else None
    return Archs4Cohort(
        name=name,
        species=normalized_species,
        metadata=joined,
        embedding_path=embedding_path,
        total_embeddings=total,
        embedding_dim=dimension,
        embedding_dtype=dtype,
        sample_manifest_path=manifest_path,
        locations_path=locations_path,
        archs4_metadata_path=resolved_metadata_path,
    )
