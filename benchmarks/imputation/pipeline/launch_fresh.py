#!/usr/bin/env python3
"""Run balanced fresh inference on two GPUs; finalize only after both succeed."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from common import HERE, RESULTS, WORK, REPO_ROOT, CONFIG, sha256


def worker(gpu,names):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MPLBACKEND='Agg')
    for name in names:
        log=RESULTS/f'run_{name}.log'
        with log.open('a',buffering=1) as h:
            subprocess.run([sys.executable,'-u',str(HERE/'pipeline/fresh_run.py'),'--model',name],env=env,stdout=h,stderr=subprocess.STDOUT,check=True)


if __name__=='__main__':
    RESULTS.mkdir(exist_ok=True)
    print('Fresh 54-condition inference: six ratios, three seeds, native vocabulary only.',flush=True)
    assert json.loads((RESULTS/'protocol_validation.json').read_text())['status']=='passed'
    inputs={str(p.relative_to(REPO_ROOT)):sha256(p) for p in sorted(WORK.iterdir()) if p.is_file()}
    checkpoints=['model/r7hnr92k/best_model.pt','model/BulkFormer/BulkFormer_50M.pt','model/BulkFormer/model/BulkFormer_147M.pt']
    provenance={'status':'running','started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'config':CONFIG,
        'input_sha256':inputs,'checkpoint_sha256':{p:sha256(REPO_ROOT/p) for p in checkpoints},
        'resource_sha256':{p:sha256(REPO_ROOT/p) for p in ['model/r7hnr92k/config.json','model/BulkFormer/utils/BulkFormer.py','model/BulkFormer/data/G_tcga.pt','model/BulkFormer/data/G_tcga_weight.pt','data/ensembl/canonical_genes.csv','data/gencode/gencode_v49_gene_exon_lengths.csv','model/BulkFormer/data/bulkformer_gene_info.csv']},
        'pipeline_sha256':{p.name:sha256(p) for p in sorted((HERE/'pipeline').glob('*.py'))},
        'torch_version':__import__('torch').__version__,'fresh_model_inference':True,
        'prepared_matrices_and_frozen_training_profiles_reused':True,
        'limitations':['Different native target preprocessing: cross-model MSE is not common-target MSE.',
                       'Native-only evaluation: gene panels and exact masks differ across model vocabularies.',
                       'TCGA pretraining exposure not established; no patient-level uncertainty estimated.']}
    path=RESULTS/'run_provenance.json';path.write_text(json.dumps(provenance,indent=2)+'\n')
    start=time.monotonic()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(worker,0,['ours_45.6m','bulkformer_50m']),pool.submit(worker,1,['bulkformer_147m'])]
            for f in futures:f.result()
        subprocess.run([sys.executable,str(HERE/'pipeline/fresh_run.py'),'--aggregate'],check=True)
        provenance.update(status='complete',elapsed_seconds=time.monotonic()-start)
        path.write_text(json.dumps(provenance,indent=2)+'\n')
        subprocess.run([sys.executable,str(HERE/'pipeline/build_notebook.py')],check=True,env=dict(os.environ,MPLBACKEND='Agg'))
    except Exception as exc:
        provenance.update(status='failed',error=str(exc),elapsed_seconds=time.monotonic()-start)
        path.write_text(json.dumps(provenance,indent=2)+'\n');raise
