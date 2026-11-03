#!/usr/bin/env python3
"""Acquire and validate the frozen ChEMBL 37 mechanism-target contract."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
WORK = HERE / "work" / "chembl"
RESULTS = HERE / "results"
REFERENCE = json.loads((HERE / "references" / "drug_target_resource.json").read_text())
CONFIG = json.loads((HERE / "config.json").read_text())
BASE = "https://www.ebi.ac.uk/chembl/api/data"


def say(message): print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def get_json(url, retries=5):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return json.load(response)
        except Exception:
            if attempt + 1 == retries: raise
            time.sleep(2 ** attempt)


def pages(resource, key, query=""):
    rows=[]; offset=0; limit=1000
    while True:
        sep="&" if query else "?"; url=f"{BASE}/{resource}.json{query}{sep}limit={limit}&offset={offset}"
        data=get_json(url); rows.extend(data[key]); total=int(data["page_meta"]["total_count"])
        say(f"ChEMBL {resource}: {len(rows)}/{total}")
        if len(rows)>=total: return rows
        offset += limit


def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
    return h.hexdigest()


def main():
    WORK.mkdir(parents=True,exist_ok=True); RESULTS.mkdir(parents=True,exist_ok=True)
    status=get_json(f"{BASE}/status.json")
    if status["chembl_db_version"].lower() != REFERENCE["release"]: raise RuntimeError(status)
    checksums=urllib.request.urlopen(REFERENCE["checksum_source"]).read().decode()
    expected=REFERENCE["distribution_sha256"]
    checksum_map={line.split()[1]:line.split()[0] for line in checksums.splitlines() if len(line.split())>=2}
    if checksum_map.get("chembl_37_sqlite.tar.gz") != expected: raise RuntimeError("frozen ChEMBL checksum does not match the filename-specific published checksum")
    (WORK/"checksums.txt").write_text(checksums)

    mechanisms=pages("mechanism","mechanisms")
    mechanisms=[m for m in mechanisms if int(m.get("max_phase") or -1)==4 and int(m.get("direct_interaction") or 0)==1]
    molecules=pages("molecule","molecules","?max_phase=4")
    molecule_names={m["molecule_chembl_id"]:m.get("pref_name") for m in molecules if m.get("molecule_type")=="Small molecule"}
    approved_parents={m["molecule_hierarchy"]["parent_chembl_id"] for m in molecules if m.get("molecule_type")=="Small molecule" and m.get("molecule_hierarchy")}
    mechanisms=[m for m in mechanisms if m.get("parent_molecule_chembl_id") in approved_parents]

    site_rows=pages("binding_site","binding_sites")
    site_components={int(s["site_id"]):{int(c["component_id"]) for c in s.get("site_components",[])} for s in site_rows}
    target_ids=sorted({m["target_chembl_id"] for m in mechanisms if m.get("target_chembl_id")})
    say(f"fetching {len(target_ids)} mechanism targets")
    with ThreadPoolExecutor(max_workers=16) as pool:
        targets=list(pool.map(lambda t:get_json(f"{BASE}/target/{t}.json"),target_ids))
    target_map={t["target_chembl_id"]:t for t in targets}

    hgnc=pd.read_csv(CONFIG["hgnc"],sep="\t",dtype=str)
    accession_to_gene={}
    for _,row in hgnc.dropna(subset=["uniprot_ids"]).iterrows():
        for acc in str(row.uniprot_ids).split("|"): accession_to_gene.setdefault(acc,row.symbol)
    rows=[]; excluded={}
    for m in mechanisms:
        if not m.get("target_chembl_id"): excluded["missing_target"] = excluded.get("missing_target",0)+1; continue
        target=target_map[m["target_chembl_id"]]
        if target.get("organism") != "Homo sapiens": excluded["non_human"] = excluded.get("non_human",0)+1; continue
        components=target.get("target_components") or []
        selected=components
        if len(components)>1:
            ids=site_components.get(int(m["site_id"]),set()) if m.get("site_id") is not None else set()
            selected=[c for c in components if int(c["component_id"]) in ids]
            if len(selected)!=1: excluded["ambiguous_multi_component"] = excluded.get("ambiguous_multi_component",0)+1; continue
        mapped={(c.get("accession"),accession_to_gene.get(c.get("accession"))) for c in selected}
        mapped={(a,g) for a,g in mapped if g}
        if len(mapped)!=1: excluded["unmapped_or_ambiguous_hgnc"] = excluded.get("unmapped_or_ambiguous_hgnc",0)+1; continue
        accession,gene=next(iter(mapped)); parent=m["parent_molecule_chembl_id"]
        rows.append({"drug_id":parent,"drug_name":molecule_names.get(parent) or parent,"target_gene":gene,"uniprot_accession":accession,"target_chembl_id":m["target_chembl_id"],"target_type":target.get("target_type"),"action_type":m.get("action_type"),"mechanism_of_action":m.get("mechanism_of_action"),"mec_id":m.get("mec_id")})
    edges=pd.DataFrame(rows).sort_values(["drug_id","target_gene","mec_id"]).drop_duplicates(["drug_id","target_gene"])
    edges.to_parquet(WORK/"chembl37_direct_human_mechanism_edges.parquet",index=False)
    edges.to_csv(RESULTS/"chembl37_direct_human_mechanism_edges.csv.gz",index=False,compression="gzip")
    canonical=set(pd.read_csv(CONFIG["canonical_genes"]).gene_symbol)
    eligible=edges[edges.target_gene.isin(canonical)].groupby(["drug_id","drug_name"]).target_gene.nunique().reset_index(name="eligible_targets")
    eligible.to_csv(RESULTS/"chembl37_drug_target_counts.csv",index=False)
    qc={"status":status,"api_base":BASE,"release":REFERENCE["release"],"published_archive_checksum_verified":True,"published_archive_sha256":expected,"checksums_file_sha256":sha256(WORK/"checksums.txt"),"api_snapshot_edge_file_sha256":sha256(RESULTS/"chembl37_direct_human_mechanism_edges.csv.gz"),"approved_direct_mechanism_records":len(mechanisms),"unique_targets_queried":len(target_ids),"retained_unique_drug_gene_edges":len(edges),"retained_drugs":edges.drug_id.nunique(),"retained_hgnc_genes":edges.target_gene.nunique(),"drugs_with_at_least_10_bridge_targets":int((eligible.eligible_targets>=10).sum()),"excluded_records":excluded,"contract_valid":True,"note":"Filtered API snapshot is ChEMBL 37 per status endpoint; source archive checksum was verified against the frozen checksum manifest without materializing the 5.4 GiB SQLite archive."}
    (RESULTS/"chembl37_contract_qc.json").write_text(json.dumps(qc,indent=2)+"\n")
    say(json.dumps(qc,indent=2))


if __name__=="__main__": main()
