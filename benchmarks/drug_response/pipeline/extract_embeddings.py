#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import sys
import time

import numpy as np
import torch

from common import SHARED, WORK


def load_models_module():
    path = SHARED / "benchmarks/tcga_imputation/pipeline"
    sys.path.insert(0, str(path))
    import model_adapters
    return model_adapters


@torch.inference_mode()
def extract(name: str, device: torch.device) -> None:
    output = WORK / f"{name}_embeddings.npy"
    if output.exists():
        print(f"reusing {output}", flush=True)
        return
    adapters = load_models_module()
    if name == "bridge_45.6m":
        matrix = np.load(WORK / "bridge_log1p_tpm_approx.npy", mmap_mode="r")
        model = adapters.load_ours(device).eval()
        batch_size = 8
    else:
        matrix = np.load(WORK / "bulkformer_log1p_tpm_approx.npy", mmap_mode="r")
        model = adapters.load_bulkformer(name, device).eval()
        batch_size = 1 if name == "bulkformer_147m" else 4
    vectors, started = [], time.monotonic()
    print(f"startup: {name}, {len(matrix)} samples, batch={batch_size}, device={device}", flush=True)
    for start in range(0, len(matrix), batch_size):
        values = torch.as_tensor(matrix[start:start + batch_size], device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            if name == "bridge_45.6m":
                vector = model.encode(values, normalize=False)
            else:
                vector = model(values, mask_prob=0.0, output_expr=False).mean(dim=1)
        vectors.append(vector.float().cpu().numpy())
        done = min(start + batch_size, len(matrix))
        if done == len(matrix) or done % max(batch_size, 50) < batch_size:
            elapsed = time.monotonic() - started
            print(f"heartbeat {name}: {done}/{len(matrix)} elapsed={elapsed/60:.1f}m "
                  f"rate={done/max(elapsed, 1e-9):.2f}/s", flush=True)
    np.save(output, np.concatenate(vectors))
    print(f"saved {output}", flush=True)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["bridge_45.6m", "bulkformer_50m", "bulkformer_147m"])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    extract(args.model, torch.device(args.device if torch.cuda.is_available() else "cpu"))


if __name__ == "__main__":
    main()
