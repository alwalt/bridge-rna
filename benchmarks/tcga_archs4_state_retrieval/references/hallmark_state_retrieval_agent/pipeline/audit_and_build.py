#!/usr/bin/env python3
"""Audit frozen assets and build the human ARCHS4 Hallmark retrieval database."""
from __future__ import annotations
import json, shutil, time
from pathlib import Path
import numpy as np, pandas as pd
from common import CONFIG, RESULTS, WORK, apply_probe, l2_rows, save_json, sha256

EXPECTED = {
 "probe":"f3f6c2ddc75d872b1e78af189f9286ff2be64ee43e90047916fca5ffe0538931",
 "order":"7975818ac52f21c9ed0ed50ef3406ba6512ab5715dc6673c6cc9fc39dfd3cc8d",
 "checkpoint":"f3491beaa697a0408dc1daeb6f8d648566bd46bafc6e44938d723e7ffc325541",
 "model_config":"1cccc2a0dc09da29f69b574e96f1fa2c5e70dc99d20a8b79d921d1e4fa49b805",
 "canonical_source":"1dbe9a5753e5982b3ddb383f35e92ce5795c21fc054fff685514831c8dadbb8d",
}

def main():
    started=time.time(); source=Path(CONFIG["source_hallmark_benchmark"]); frozen=RESULTS/"frozen"
    frozen.mkdir(parents=True,exist_ok=True); (RESULTS/"archs4").mkdir(parents=True,exist_ok=True); WORK.mkdir(exist_ok=True)
    source_files={"probe":source/"results/frozen_gtex_hallmark_probe.npz","order":source/"results/frozen_hallmark_order.csv","system_manifest":source/"results/frozen_pathway_system_manifest.json","probe_manifest":source/"results/frozen_probe_manifest.json","canonical_frozen":source/"results/frozen_canonical_genes.csv"}
    for role,p in source_files.items():
        if not p.exists(): raise FileNotFoundError(p)
        shutil.copy2(p,frozen/p.name)
    shutil.copy2(CONFIG["model_config"],frozen/"bridge_model_config.json")
    checks={k:sha256(v) for k,v in {"probe":source_files["probe"],"order":source_files["order"],"checkpoint":CONFIG["checkpoint"],"model_config":CONFIG["model_config"],"canonical_source":CONFIG["canonical_genes"]}.items()}
    assert checks==EXPECTED, (checks,EXPECTED)
    system=json.loads(source_files["system_manifest"].read_text()); assert system["status"]=="validated_for_frozen_later_use"
    probe=np.load(frozen/"frozen_gtex_hallmark_probe.npz",allow_pickle=False)
    order=pd.read_csv(frozen/"frozen_hallmark_order.csv")
    assert probe["weight"].shape==(50,512) and probe["mean"].shape==probe["scale"].shape==(512,)
    assert order.hallmark.astype(str).tolist()==probe["hallmarks"].astype(str).tolist()
    canonical=pd.read_csv(CONFIG["canonical_genes"]).sort_values("token_id")
    cf=pd.read_csv(frozen/"frozen_canonical_genes.csv")
    assert len(canonical)==15165 and canonical.gene_symbol.astype(str).tolist()==cf.gene.astype(str).tolist()

    embdir=Path(CONFIG["archs4_embedding_dir"]); spec=json.loads((embdir/"embedding_manifest.json").read_text())
    loc=pd.read_parquet(embdir/"sample_locations.parquet"); sample=pd.read_parquet(CONFIG["archs4_sample_manifest"]); meta=pd.read_parquet(CONFIG["archs4_metadata"])
    assert len(loc)==spec["total_samples"]==940455 and spec["embedding_dim"]==512 and spec["normalization"]=="log1p_tpm" and not spec["l2_normalize"]
    raw=np.memmap(embdir/"sample_embeddings.float16.mmap",mode="r",dtype=np.float16,shape=(spec["total_samples"],512))
    human=loc.loc[loc.species_id.eq(0)].sort_values("global_index").reset_index(drop=True)
    assert len(human)==510709 and human.geo_accession.is_unique
    # Independent cache produced from the same selected full-cache rows is a bitwise compatibility witness.
    witness_m=Path("/home/walt/bridge-rna/benchmarks/cross_species_exercise_response/results/hallmark_readout/sample_manifest.parquet")
    witness_z=Path("/home/walt/bridge-rna/benchmarks/cross_species_exercise_response/work/hallmark_readout/archs4_bridgerna_embeddings.npy")
    wm=pd.read_parquet(witness_m); wz=np.load(witness_z,mmap_mode="r")
    observed=np.asarray(raw[wm.global_index.to_numpy(int)],np.float32)
    assert np.array_equal(observed,np.asarray(wz)), "ARCHS4 embedding compatibility witness failed"

    n=len(human); hall_path=WORK/"archs4_hallmark_signatures.float32.npy"; idx_path=WORK/"archs4_hallmark_cosine_index.float32.npy"
    hall=np.lib.format.open_memmap(hall_path,mode="w+",dtype=np.float32,shape=(n,50)); index=np.lib.format.open_memmap(idx_path,mode="w+",dtype=np.float32,shape=(n,50))
    rows=human.global_index.to_numpy(int)
    for left in range(0,n,32768):
        right=min(left+32768,n); scores=apply_probe(np.asarray(raw[rows[left:right]],np.float32),probe)
        hall[left:right]=scores; index[left:right]=l2_rows(scores)
        hall.flush(); index.flush(); print(f"ARCHS4 Hallmark {right:,}/{n:,}",flush=True)
    joined=human[["global_index","geo_accession"]].rename(columns={"geo_accession":"sample_id"})
    joined=joined.merge(sample.drop(columns=["sample_id"]).drop_duplicates("gsm"),left_on="sample_id",right_on="gsm",how="left",validate="one_to_one").merge(meta.drop_duplicates("gsm"),on="gsm",how="left",validate="one_to_one")
    assert joined.sample_id.is_unique and joined.gsm.notna().all()
    joined.to_parquet(RESULTS/"archs4/sample_metadata.parquet",index=False)
    wide=pd.concat([joined[["sample_id","global_index","series_id","title","source_name_ch1","characteristics_ch1"]].reset_index(drop=True),pd.DataFrame(np.asarray(hall),columns=probe["hallmarks"].astype(str))],axis=1)
    wide.to_parquet(RESULTS/"archs4/hallmark_signatures.parquet",index=False,compression="zstd")
    # Stable convenience links/copies keep the requested database/index discoverable while large arrays remain ignored.
    shutil.copy2(hall_path,RESULTS/"archs4/hallmark_signatures.float32.npy"); shutil.copy2(idx_path,RESULTS/"archs4/hallmark_cosine_index.float32.npy")
    pd.DataFrame({"row":np.arange(n),"sample_id":joined.sample_id,"global_index":joined.global_index}).to_parquet(RESULTS/"archs4/index_rows.parquet",index=False)
    manifest={"decision":"reuse_frozen_head_and_compatible_embeddings; no retraining and no Bridge inference","compatibility_witness":{"samples":len(wm),"max_abs_difference":0.0,"bitwise_equal":True},"sample_count":n,"dimensions":50,"preprocessing":"natural log1p(TPM), exact canonical 15,165-gene order; final-layer unweighted mean Bridge embedding","head_transform":"((embedding - GTEx_training_mean) / GTEx_training_SD) @ weight.T + bias","cosine_index":"row-wise L2-normalized float32 NumPy matrix; exact inner product equals cosine","artifacts":checks,"source_manifest_sha256":sha256(embdir/"embedding_manifest.json"),"locations":{"database":str((RESULTS/"archs4/hallmark_signatures.parquet").resolve()),"matrix":str((WORK/"archs4_hallmark_signatures.float32.npy").resolve()),"index":str((WORK/"archs4_hallmark_cosine_index.float32.npy").resolve())},"runtime_seconds":time.time()-started,"limitations":["ARCHS4 metadata are free text and incomplete.","The linear head predicts GTEx-trained ssGSEA-like scores; values are not calibrated probabilities."]}
    save_json(RESULTS/"frozen_head_compatibility_provenance.json",manifest)
    save_json(frozen/"preprocessing_contract.json",{"input":"finite sample-by-15165 matrix","gene_order_file":"frozen_canonical_genes.csv (model_index 0..15164)","expression_units":"species-correct TPM","transform":"natural log1p","bridge_checkpoint":CONFIG["checkpoint"],"bridge_checkpoint_sha256":checks["checkpoint"],"model_config_file":"bridge_model_config.json","pooling":"unweighted arithmetic mean of final-layer contextual gene states","embedding_l2_normalized":False,"hallmark_transform":"((z - mean) / scale) @ weight.T + bias","hallmark_order_file":"frozen_hallmark_order.csv","query_similarity":"cosine after row-wise L2 normalization of the 50-D signature"})
    print(json.dumps(manifest,indent=2),flush=True)
if __name__=="__main__": main()
