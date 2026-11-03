#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from common import CONFIG, RESULTS, WORK


def main() -> None:
    samples = pd.read_parquet(WORK / "sample_metadata.parquet")
    strata = samples[CONFIG["split"]["stratify_column"]].fillna("unknown").astype(str)
    rare = strata.map(strata.value_counts()).lt(CONFIG["split"]["folds"])
    strata = strata.mask(rare, "rare_or_unknown")
    fold = np.full(len(samples), -1, dtype=int)
    splitter = StratifiedKFold(n_splits=CONFIG["split"]["folds"], shuffle=True,
                              random_state=CONFIG["split"]["seed"])
    for number, (_, test) in enumerate(splitter.split(samples, strata)):
        fold[test] = number
    assert (fold >= 0).all() and samples.depmap_id.is_unique
    output = samples.assign(fold=fold, stratum=strata)
    output.to_csv(RESULTS / "cell_line_splits.csv", index=False)
    print(output.groupby("fold").size())


if __name__ == "__main__":
    main()
