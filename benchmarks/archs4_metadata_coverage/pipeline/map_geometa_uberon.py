#!/usr/bin/env python3
"""Create conservative exact-label UBERON mappings for observed GEOMeta organs."""

from __future__ import annotations

import glob
import time
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parents[1] / "data"


def main() -> None:
    organs = set()
    for path in glob.glob(str(DATA_ROOT / "geometa" / "*.csv")):
        values = pd.read_csv(path, usecols=["Organ"], dtype=str)["Organ"].dropna()
        organs.update(x.strip() for x in values if x.strip())
    rows = []
    for number, label in enumerate(sorted(organs, key=str.casefold), 1):
        response = requests.get("https://www.ebi.ac.uk/ols4/api/search",
            params={"q":label,"ontology":"uberon","exact":"true","rows":20}, timeout=30,
            headers={"User-Agent":"BridgeRNA metadata audit"})
        response.raise_for_status()
        docs = response.json().get("response",{}).get("docs",[])
        exact = [x for x in docs if str(x.get("label","")).casefold() == label.casefold()
                 and x.get("ontology_name") == "uberon" and x.get("type") == "class"]
        unique = {x.get("obo_id"):x for x in exact if x.get("obo_id")}
        if len(unique) == 1:
            doc = next(iter(unique.values()))
            rows.append({"field":"organ","canonical_label":label,"ontology_id":doc["obo_id"],
                         "ontology_label":doc["label"],"ontology_name":"UBERON",
                         "mapping_method":"OLS4 exact label","mapping_status":"reliable"})
        else:
            rows.append({"field":"organ","canonical_label":label,"ontology_id":None,
                         "ontology_label":None,"ontology_name":"UBERON",
                         "mapping_method":"OLS4 exact label",
                         "mapping_status":"ambiguous" if unique else "unmapped"})
        if number % 10 == 0: print(f"UBERON {number}/{len(organs)}",flush=True)
        time.sleep(0.1)
    pd.DataFrame(rows).to_csv(ROOT / "results/uberon_mappings.csv",index=False)


if __name__ == "__main__":
    main()
