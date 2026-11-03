#!/usr/bin/env bash
set -euo pipefail

benchmark_dir="$(cd "$(dirname "$0")/.." && pwd)"
chunk_dir="$benchmark_dir/work/chembl/archive_chunks"
url="https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_37/chembl_37_sqlite.tar.gz"
pids=()

for part in "$chunk_dir"/part_*; do
  name="$(basename "$part")"; range="${name#part_*_}"; start="${range%_*}"; end="${range##*_}"
  actual="$(stat -c %s "$part")"; expected=$((end-start+1)); remaining=$((expected-actual))
  if (( remaining == 0 )); then continue; fi
  tail_start=$((start+actual)); segment=$(((remaining+3)/4))
  for index in 0 1 2 3; do
    sub_start=$((tail_start+index*segment)); sub_end=$((sub_start+segment-1))
    if (( sub_start > end )); then break; fi
    if (( sub_end > end )); then sub_end=$end; fi
    tail="$part.tail_${index}_${sub_start}_${sub_end}"
    curl -sS -L --fail --retry 8 --retry-delay 5 --range "$sub_start-$sub_end" -o "$tail" "$url" &
    pids+=("$!")
  done
done
for pid in "${pids[@]}"; do wait "$pid"; done

for part in "$chunk_dir"/part_*; do
  [[ "$part" == *.tail_* ]] && continue
  name="$(basename "$part")"; range="${name#part_*_}"; start="${range%_*}"; end="${range##*_}"
  for tail in "$part".tail_*; do
    [[ -e "$tail" ]] || continue
    tail_range="${tail##*.tail_}"; tail_range="${tail_range#*_}"; tail_start="${tail_range%_*}"; tail_end="${tail_range##*_}"
    if [[ "$(stat -c %s "$tail")" -ne $((tail_end-tail_start+1)) ]]; then echo "bad tail $tail" >&2; exit 1; fi
    dd if="$tail" of="$part" bs=8M oflag=append conv=notrunc status=none
  done
  if [[ "$(stat -c %s "$part")" -ne $((end-start+1)) ]]; then echo "bad completed part $part" >&2; exit 1; fi
done

bash "$benchmark_dir/pipeline/acquire_chembl_archive.sh"
