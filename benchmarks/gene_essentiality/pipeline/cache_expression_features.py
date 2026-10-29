#!/usr/bin/env python3
"""Create the scalar target-expression feature cache for the common head."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
CONFIG = json.loads((BENCH / "config.json").read_text())
WORK = Path(CONFIG["shared_work_root"])
RESULTS = BENCH / "results"


def main() -> None:
    expression = np.load(WORK / "expression_released_log2_tpm_plus_1.float32.npy", mmap_mode="r")
    targets = pd.read_csv(RESULTS / "gene_overlap.csv")
    selected = targets[targets.bridge_representable].copy().reset_index(drop=True)
    output_dir = WORK / "target_expression_contextual"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        output_dir / "tokens.float16.npy", mode="w+", dtype=np.float16,
        shape=(len(expression), len(selected), 1),
    )
    present = selected.present_in_expression.to_numpy(dtype=bool)
    output[:, :, 0] = -10.0
    output[:, present, 0] = expression[:, selected.loc[present, "expression_column_index"].astype(int)]
    output.flush()
    np.save(output_dir / "completed.npy", np.ones(len(expression), dtype=bool))
    selected.to_csv(output_dir / "target_manifest.csv", index=False)
    (output_dir / "provenance.json").write_text(json.dumps({
        "representation": "target_expression", "shape": list(output.shape), "dtype": "float16",
        "feature": "released target-gene log2(TPM+1) value only", "encoder": None,
        "all_method_common_genes_have_expression": bool(selected.loc[selected.all_method_common, "present_in_expression"].all()),
    }, indent=2) + "\n")
    print(f"cached {output.shape}")


if __name__ == "__main__":
    main()
