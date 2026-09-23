#!/usr/bin/env bash
set -euo pipefail

benchmark_dir="$(cd "$(dirname "$0")/.." && pwd)"
archive="$benchmark_dir/work/chembl/chembl_37_sqlite.tar.gz"
url="https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_37/chembl_37_sqlite.tar.gz"
total_bytes=5764252857
connections=12
prefix_bytes="$(stat -c %s "$archive")"
chunk_dir="$benchmark_dir/work/chembl/archive_chunks"
mkdir -p "$chunk_dir"

remaining=$((total_bytes-prefix_bytes))
chunk_bytes=$(((remaining+connections-1)/connections))
pids=()
for index in $(seq 0 $((connections-1))); do
  start=$((prefix_bytes+index*chunk_bytes))
  end=$((start+chunk_bytes-1))
  if (( start >= total_bytes )); then break; fi
  if (( end >= total_bytes )); then end=$((total_bytes-1)); fi
  part="$chunk_dir/part_${index}_${start}_${end}"
  expected=$((end-start+1))
  if [[ -f "$part" ]] && [[ "$(stat -c %s "$part")" -eq "$expected" ]]; then continue; fi
  curl -sS -L --fail --retry 8 --retry-delay 5 --range "$start-$end" -o "$part" "$url" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

assembled="$archive.assembled"
cp "$archive" "$assembled"
for index in $(seq 0 $((connections-1))); do
  start=$((prefix_bytes+index*chunk_bytes))
  end=$((start+chunk_bytes-1))
  if (( start >= total_bytes )); then break; fi
  if (( end >= total_bytes )); then end=$((total_bytes-1)); fi
  part="$chunk_dir/part_${index}_${start}_${end}"
  expected=$((end-start+1))
  actual="$(stat -c %s "$part")"
  if [[ "$actual" -ne "$expected" ]]; then
    echo "range size mismatch for $part: $actual != $expected" >&2
    exit 1
  fi
  dd if="$part" of="$assembled" bs=8M oflag=append conv=notrunc status=none
done
if [[ "$(stat -c %s "$assembled")" -ne "$total_bytes" ]]; then
  echo "assembled archive has incorrect size" >&2
  exit 1
fi
mv "$archive" "$archive.partial"
mv "$assembled" "$archive"
python3 "$benchmark_dir/pipeline/verify_chembl_archive.py"
