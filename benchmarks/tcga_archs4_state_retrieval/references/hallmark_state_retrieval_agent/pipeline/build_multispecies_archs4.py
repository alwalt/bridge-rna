#!/usr/bin/env python3
"""Build mouse and logical combined ARCHS4 indexes from frozen cached embeddings."""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np, pandas as pd
from common import CONFIG, RESULTS, WORK, apply_probe, l2_rows, save_json, sha256

OUT=RESULTS/'archs4_multispecies'; MW=WORK/'archs4_multispecies'; EMB=Path(CONFIG['archs4_embedding_dir'])

def main():
    OUT.mkdir(parents=True,exist_ok=True); MW.mkdir(parents=True,exist_ok=True); started=time.time()
    spec=json.loads((EMB/'embedding_manifest.json').read_text()); loc=pd.read_parquet(EMB/'sample_locations.parquet')
    raw=np.memmap(EMB/'sample_embeddings.float16.mmap',mode='r',dtype=np.float16,shape=(spec['total_samples'],512))
    sample=pd.read_parquet(CONFIG['archs4_sample_manifest']); meta=pd.read_parquet(CONFIG['archs4_metadata'])
    probe=np.load(RESULTS/'frozen/frozen_gtex_hallmark_probe.npz',allow_pickle=False); names=probe['hallmarks'].astype(str)
    mouse=loc[loc.species_id.eq(1)].sort_values('global_index').reset_index(drop=True); assert len(mouse)==429746
    hall=np.lib.format.open_memmap(MW/'mouse_hallmark_signatures.float32.npy',mode='w+',dtype=np.float32,shape=(len(mouse),50))
    hidx=np.lib.format.open_memmap(MW/'mouse_hallmark_cosine_index.float32.npy',mode='w+',dtype=np.float32,shape=(len(mouse),50))
    bidx=np.lib.format.open_memmap(MW/'mouse_bridge_cosine_index.float32.npy',mode='w+',dtype=np.float32,shape=(len(mouse),512))
    rows=mouse.global_index.to_numpy(int)
    for left in range(0,len(mouse),32768):
        right=min(left+32768,len(mouse)); z=np.asarray(raw[rows[left:right]],np.float32); s=apply_probe(z,probe)
        hall[left:right]=s; hidx[left:right]=l2_rows(s); bidx[left:right]=l2_rows(z)
        hall.flush();hidx.flush();bidx.flush();print(f'mouse atlas {right:,}/{len(mouse):,}',flush=True)
    joined=mouse[['global_index','geo_accession']].rename(columns={'geo_accession':'sample_id'})
    joined=joined.merge(sample.drop(columns=['sample_id']).drop_duplicates('gsm'),left_on='sample_id',right_on='gsm',how='left',validate='one_to_one')
    joined=joined.merge(meta.drop_duplicates('gsm'),on='gsm',how='left',validate='one_to_one',suffixes=('','_meta'))
    assert joined.gsm.notna().all() and joined.sample_id.is_unique
    joined.insert(0,'species_reference_row',np.arange(len(joined),dtype=np.int64)); joined['species']='mouse'
    joined.to_parquet(OUT/'mouse_sample_metadata.parquet',index=False)
    wide=pd.concat([joined[['sample_id','global_index','series_id','species','title','source_name_ch1','characteristics_ch1']].reset_index(drop=True),pd.DataFrame(np.asarray(hall),columns=names)],axis=1)
    wide.to_parquet(OUT/'mouse_hallmark_signatures.parquet',index=False,compression='zstd')
    human=pd.read_parquet(RESULTS/'archs4/sample_metadata.parquet').copy(); human['species']='human'; human['species_reference_row']=np.arange(len(human),dtype=np.int64)
    hrows=human[['sample_id','global_index','series_id','species','species_reference_row']].copy(); hrows.insert(0,'combined_row',np.arange(len(hrows),dtype=np.int64))
    mrows=joined[['sample_id','global_index','series_id','species','species_reference_row']].copy(); mrows.insert(0,'combined_row',np.arange(len(hrows),len(hrows)+len(mrows),dtype=np.int64))
    combined=pd.concat([hrows,mrows],ignore_index=True); assert len(combined)==940455 and combined.sample_id.is_unique
    combined.to_parquet(OUT/'combined_row_manifest.parquet',index=False)
    save_json(OUT/'atlas_provenance.json',{
      'human_samples':len(human),'mouse_samples':len(mouse),'combined_samples':len(combined),'combined_order':'human rows in existing order, followed by mouse rows sorted by frozen global_index',
      'bridge_inference_rerun':False,'hallmark_head_retrained':False,'mouse_hallmark_label':'zero-shot cross-species predictions from the human GTEx-trained head',
      'preprocessing':'frozen natural log1p(TPM), shared ordered 15,165 one-to-one orthologs','checkpoint_sha256':sha256(CONFIG['checkpoint']),
      'probe_sha256':sha256(RESULTS/'frozen/frozen_gtex_hallmark_probe.npz'),'embedding_manifest_sha256':sha256(EMB/'embedding_manifest.json'),
      'search_sets':{'human':'existing human arrays','mouse':'mouse arrays in work/archs4_multispecies','combined':'exact top-k merge across human and mouse arrays'},'runtime_seconds':time.time()-started})
    print(json.dumps({'human':len(human),'mouse':len(mouse),'combined':len(combined),'seconds':time.time()-started},indent=2))
if __name__=='__main__': main()
