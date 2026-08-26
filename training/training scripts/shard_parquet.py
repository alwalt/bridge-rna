#!/usr/bin/env python3
"""Shard a large sample-major parquet into many DDP-friendly parquet files.

Input options:
  - --input-parquet: path to a single parquet file (for example expression.parquet)
  - --input-dir: directory containing expression.parquet and optional metadata files

Output layout:
  <output_dir>/batch_files/batch_00000.parquet
  <output_dir>/batch_files/batch_00001.parquet
  ...
  <output_dir>/batch_manifest.json
  <output_dir>/metadata.csv            (copied if present in input dir)
  <output_dir>/samples.json            (copied if present in input dir)
  <output_dir>/genes.json              (copied if present in input dir)
  <output_dir>/canonical_genes.csv     (copied if present in input dir)

The manifest maps each batch parquet filename to ordered sample IDs, which
train_single.py uses to build species-aware sample indices.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

INDEX_CANDIDATES = ("geo_accession", "__index_level_0__")
SIDECAR_FILES = ("metadata.csv", "samples.json", "genes.json", "canonical_genes.csv")


def _find_index_column(columns: list[str]) -> str | None:
    for c in INDEX_CANDIDATES:
        if c in columns:
            return c
    return None


def _resolve_input(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.input_parquet is not None:
        input_parquet = args.input_parquet
        input_dir = input_parquet.parent
    else:
        input_dir = args.input_dir
        input_parquet = input_dir / "expression.parquet"

    if not input_parquet.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_parquet}")

    return input_parquet, input_dir


def _load_species_map(metadata_csv: Path) -> dict[str, str]:
    if not metadata_csv.exists():
        raise FileNotFoundError(
            f"metadata.csv not found at {metadata_csv}. Stratified mode requires species metadata."
        )

    sample_to_species: dict[str, str] = {}
    with open(metadata_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = str(row.get("geo_accession") or row.get("id") or "").strip()
            sp = str(row.get("species") or "").strip().lower()
            if sid and sp:
                sample_to_species[sid] = sp
    if not sample_to_species:
        raise ValueError(f"No usable (sample_id, species) rows found in {metadata_csv}")
    return sample_to_species


def _norm_species(x: str) -> str:
    s = str(x).strip().lower()
    if s.startswith("human"):
        return "human"
    if s.startswith("mouse"):
        return "mouse"
    return "other"


class _BatchCursor:
    """Incrementally consumes rows from a parquet batch iterator."""

    def __init__(self, pf: pq.ParquetFile, read_batch_size: int):
        self._it = pf.iter_batches(batch_size=read_batch_size, use_threads=True)
        self._batch: pa.RecordBatch | None = None
        self._offset = 0

    def take(self, n: int) -> tuple[list[pa.RecordBatch], int]:
        chunks: list[pa.RecordBatch] = []
        taken = 0
        need = max(0, int(n))
        while need > 0:
            if self._batch is None or self._offset >= self._batch.num_rows:
                try:
                    self._batch = next(self._it)
                    self._offset = 0
                except StopIteration:
                    break
            avail = self._batch.num_rows - self._offset
            use = min(avail, need)
            if use > 0:
                chunks.append(self._batch.slice(self._offset, use))
                self._offset += use
                taken += use
                need -= use
        return chunks, taken


def _build_shard_from_cursors(
    rows_per_shard: int,
    target_ratio_human: float,
    human_cur: _BatchCursor,
    mouse_cur: _BatchCursor,
    other_cur: _BatchCursor,
) -> tuple[list[pa.RecordBatch], int]:
    target_h = int(round(rows_per_shard * target_ratio_human))
    target_h = max(0, min(rows_per_shard, target_h))
    target_m = rows_per_shard - target_h

    chunks: list[pa.RecordBatch] = []
    total = 0

    c, t = human_cur.take(target_h)
    chunks.extend(c)
    total += t

    c, t = mouse_cur.take(target_m)
    chunks.extend(c)
    total += t

    # Fill any remaining slots from whichever species still has rows.
    remaining = rows_per_shard - total
    if remaining > 0:
        for cur in (human_cur, mouse_cur, other_cur):
            if remaining <= 0:
                break
            c, t = cur.take(remaining)
            chunks.extend(c)
            total += t
            remaining -= t

    return chunks, total


def _split_input_by_species(
    input_parquet: Path,
    temp_dir: Path,
    index_col: str,
    sample_to_species: dict[str, str],
    read_batch_size: int,
    compression: str,
) -> tuple[Path, Path, Path, dict[str, int]]:
    pf = pq.ParquetFile(str(input_parquet))
    schema = pf.schema_arrow

    human_path = temp_dir / "human.parquet"
    mouse_path = temp_dir / "mouse.parquet"
    other_path = temp_dir / "other.parquet"

    wh = pq.ParquetWriter(human_path, schema, compression=compression, use_dictionary=True)
    wm = pq.ParquetWriter(mouse_path, schema, compression=compression, use_dictionary=True)
    wo = pq.ParquetWriter(other_path, schema, compression=compression, use_dictionary=True)

    counts = {"human": 0, "mouse": 0, "other": 0}
    try:
        for rb in pf.iter_batches(batch_size=read_batch_size, use_threads=True):
            ids = [str(x) for x in rb.column(rb.schema.get_field_index(index_col)).to_pylist()]
            sp = [_norm_species(sample_to_species.get(i, "other")) for i in ids]

            mh = pa.array([v == "human" for v in sp], type=pa.bool_())
            mm = pa.array([v == "mouse" for v in sp], type=pa.bool_())
            mo = pa.array([v not in ("human", "mouse") for v in sp], type=pa.bool_())

            t = pa.Table.from_batches([rb])
            th = t.filter(mh)
            tm = t.filter(mm)
            to = t.filter(mo)

            if th.num_rows:
                wh.write_table(th)
                counts["human"] += th.num_rows
            if tm.num_rows:
                wm.write_table(tm)
                counts["mouse"] += tm.num_rows
            if to.num_rows:
                wo.write_table(to)
                counts["other"] += to.num_rows
    finally:
        wh.close()
        wm.close()
        wo.close()

    return human_path, mouse_path, other_path, counts


def _shard_parquet_stratified(
    input_parquet: Path,
    output_dir: Path,
    index_col: str,
    rows_per_shard: int,
    read_batch_size: int,
    row_group_size: int,
    compression: str,
    sample_to_species: dict[str, str],
    target_ratio_human: float,
    keep_temp_files: bool,
) -> tuple[int, int, dict[str, list[str]]]:
    temp_dir = output_dir / "_tmp_species"
    temp_dir.mkdir(parents=True, exist_ok=True)

    human_path, mouse_path, other_path, counts = _split_input_by_species(
        input_parquet=input_parquet,
        temp_dir=temp_dir,
        index_col=index_col,
        sample_to_species=sample_to_species,
        read_batch_size=read_batch_size,
        compression=compression,
    )

    print(
        f"[SHARD] Species split counts: human={counts['human']:,}, mouse={counts['mouse']:,}, other={counts['other']:,}",
        flush=True,
    )

    hp = pq.ParquetFile(str(human_path))
    mp = pq.ParquetFile(str(mouse_path))
    op = pq.ParquetFile(str(other_path))

    hcur = _BatchCursor(hp, read_batch_size)
    mcur = _BatchCursor(mp, read_batch_size)
    ocur = _BatchCursor(op, read_batch_size)

    batch_dir = output_dir / "batch_files"
    manifest: dict[str, list[str]] = {}
    shard_idx = 0
    rows_processed = 0
    running_row_index = 0

    while True:
        chunks, nrows = _build_shard_from_cursors(
            rows_per_shard=rows_per_shard,
            target_ratio_human=target_ratio_human,
            human_cur=hcur,
            mouse_cur=mcur,
            other_cur=ocur,
        )
        if nrows <= 0:
            break

        wrote = _write_shard(
            batches=chunks,
            shard_idx=shard_idx,
            batch_dir=batch_dir,
            compression=compression,
            row_group_size=row_group_size,
            index_col=index_col,
            running_row_index=running_row_index,
            manifest=manifest,
        )
        rows_processed += wrote
        running_row_index += wrote
        shard_idx += 1

        if shard_idx % 10 == 0:
            print(
                f"[SHARD] Wrote {shard_idx:,} shards | rows processed: {rows_processed:,}",
                flush=True,
            )

    if not keep_temp_files:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return shard_idx, rows_processed, manifest


def _write_shard(
    batches: list[pa.RecordBatch],
    shard_idx: int,
    batch_dir: Path,
    compression: str,
    row_group_size: int,
    index_col: str | None,
    running_row_index: int,
    manifest: dict[str, list[str]],
) -> int:
    table = pa.Table.from_batches(batches)
    shard_name = f"batch_{shard_idx:05d}.parquet"
    shard_path = batch_dir / shard_name

    pq.write_table(
        table,
        shard_path,
        compression=compression,
        use_dictionary=True,
        row_group_size=min(max(1, int(row_group_size)), table.num_rows),
    )

    if index_col is not None:
        sample_ids = [str(x) for x in table.column(index_col).to_pylist()]
    else:
        # Fallback to synthetic row IDs if no explicit sample-id column exists.
        sample_ids = [str(running_row_index + i) for i in range(table.num_rows)]

    manifest[shard_name] = sample_ids
    return table.num_rows


def shard_parquet(
    input_parquet: Path,
    input_dir: Path,
    output_dir: Path,
    rows_per_shard: int,
    read_batch_size: int,
    row_group_size: int,
    compression: str,
    stratified_by_species: bool = False,
    metadata_csv: Path | None = None,
    target_ratio_human: float = 0.5,
    keep_temp_files: bool = False,
) -> None:
    if rows_per_shard <= 0:
        raise ValueError("rows_per_shard must be > 0")
    if read_batch_size <= 0:
        raise ValueError("read_batch_size must be > 0")

    output_dir.mkdir(parents=True, exist_ok=True)
    batch_dir = output_dir / "batch_files"
    batch_dir.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(str(input_parquet))
    index_col = _find_index_column(pf.schema_arrow.names)

    print(f"[SHARD] Input file: {input_parquet}")
    print(f"[SHARD] Total rows: {pf.metadata.num_rows:,}")
    print(f"[SHARD] Total row groups: {pf.metadata.num_row_groups:,}")
    print(f"[SHARD] Index column: {index_col}")
    print(f"[SHARD] Rows per shard: {rows_per_shard:,}")

    if stratified_by_species:
        if index_col is None:
            raise ValueError("Stratified mode requires an index column (geo_accession or __index_level_0__).")
        md = metadata_csv if metadata_csv is not None else (input_dir / "metadata.csv")
        sample_to_species = _load_species_map(md)
        print(f"[SHARD] Stratified mode enabled using metadata: {md}", flush=True)
        print(f"[SHARD] Target human ratio per shard: {target_ratio_human:.3f}", flush=True)
        shard_idx, rows_processed, manifest = _shard_parquet_stratified(
            input_parquet=input_parquet,
            output_dir=output_dir,
            index_col=index_col,
            rows_per_shard=rows_per_shard,
            read_batch_size=read_batch_size,
            row_group_size=row_group_size,
            compression=compression,
            sample_to_species=sample_to_species,
            target_ratio_human=target_ratio_human,
            keep_temp_files=keep_temp_files,
        )
    else:
        manifest: dict[str, list[str]] = {}
        buffer_batches: list[pa.RecordBatch] = []
        buffer_rows = 0
        shard_idx = 0
        running_row_index = 0
        rows_processed = 0

        for rb in pf.iter_batches(batch_size=read_batch_size, use_threads=True):
            rb_offset = 0
            while rb_offset < rb.num_rows:
                remaining = rows_per_shard - buffer_rows
                take = min(remaining, rb.num_rows - rb_offset)
                chunk = rb.slice(rb_offset, take)
                buffer_batches.append(chunk)
                buffer_rows += take
                rb_offset += take

                if buffer_rows >= rows_per_shard:
                    wrote = _write_shard(
                        batches=buffer_batches,
                        shard_idx=shard_idx,
                        batch_dir=batch_dir,
                        compression=compression,
                        row_group_size=row_group_size,
                        index_col=index_col,
                        running_row_index=running_row_index,
                        manifest=manifest,
                    )
                    rows_processed += wrote
                    running_row_index += wrote
                    shard_idx += 1
                    buffer_batches = []
                    buffer_rows = 0

                    if shard_idx % 10 == 0:
                        print(
                            f"[SHARD] Wrote {shard_idx:,} shards | rows processed: {rows_processed:,}",
                            flush=True,
                        )

        if buffer_batches:
            wrote = _write_shard(
                batches=buffer_batches,
                shard_idx=shard_idx,
                batch_dir=batch_dir,
                compression=compression,
                row_group_size=row_group_size,
                index_col=index_col,
                running_row_index=running_row_index,
                manifest=manifest,
            )
            rows_processed += wrote
            running_row_index += wrote
            shard_idx += 1

    manifest_path = output_dir / "batch_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    copied = []
    for name in SIDECAR_FILES:
        src = input_dir / name
        if src.exists():
            dst = output_dir / name
            shutil.copy2(src, dst)
            copied.append(name)

    print(f"[DONE] Wrote {shard_idx:,} shard files to {batch_dir}")
    print(f"[DONE] Rows processed: {rows_processed:,}")
    print(f"[DONE] Manifest: {manifest_path}")
    if copied:
        print(f"[DONE] Copied sidecar files: {', '.join(copied)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shard a single parquet into many training shards")
    group = p.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--input-parquet",
        type=Path,
        default=None,
        help="Path to source parquet (default with --input-dir: <input-dir>/expression.parquet)",
    )
    group.add_argument(
        "--input-dir",
        type=Path,
        default=Path("./data/archs4/train_orthologs_merged"),
        help="Directory containing expression.parquet and optional sidecar metadata",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./data/archs4/train_orthologs_sharded"),
        help="Directory to write batch_files/ and manifest",
    )
    p.add_argument(
        "--rows-per-shard",
        type=int,
        default=20000,
        help="Target samples per output parquet shard",
    )
    p.add_argument(
        "--read-batch-size",
        type=int,
        default=8192,
        help="Record-batch size used while scanning input parquet",
    )
    p.add_argument(
        "--row-group-size",
        type=int,
        default=2048,
        help="Row-group size used when writing each shard",
    )
    p.add_argument(
        "--compression",
        default="zstd",
        help="Parquet compression codec (default: zstd)",
    )
    p.add_argument(
        "--stratified-by-species",
        action="store_true",
        help="Interleave human/mouse samples into mixed shards using metadata.csv",
    )
    p.add_argument(
        "--metadata-csv",
        type=Path,
        default=None,
        help="Optional explicit path to metadata.csv with geo_accession/species columns",
    )
    p.add_argument(
        "--target-ratio-human",
        type=float,
        default=0.5,
        help="Target human fraction per shard in stratified mode (default: 0.5)",
    )
    p.add_argument(
        "--keep-temp-files",
        action="store_true",
        help="Keep temporary species parquet files used by stratified mode",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_parquet, input_dir = _resolve_input(args)
    shard_parquet(
        input_parquet=input_parquet,
        input_dir=input_dir,
        output_dir=args.output_dir,
        rows_per_shard=int(args.rows_per_shard),
        read_batch_size=int(args.read_batch_size),
        row_group_size=int(args.row_group_size),
        compression=args.compression,
        stratified_by_species=bool(args.stratified_by_species),
        metadata_csv=args.metadata_csv,
        target_ratio_human=float(args.target_ratio_human),
        keep_temp_files=bool(args.keep_temp_files),
    )


if __name__ == "__main__":
    main()
