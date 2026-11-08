#!/usr/bin/env python3
"""Fresh, balanced-seed TCGA inference; independent condition caches and explicit QC."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch

from common import CONFIG, HERE, REPO_ROOT, RESULTS, WORK, sha256
import model_adapters as adapters
from evaluation import (MODELS, MODEL_SPACE, load_inputs, condition_masks,
                            apply_mask, score_prediction, load_frozen_baselines)


def deterministic(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def check_inputs():
    samples, ours, bulk, og, bg, shared = load_inputs()
    ids=samples.sample_id.astype(str).tolist()
    assert len(ids)==1000 and len(set(ids))==1000
    assert ours.shape==(1000,len(og)) and bulk.shape==(1000,len(bg))
    assert len(shared)==15101 and shared.gene.is_unique
    assert np.isfinite(ours).all() and np.isfinite(bulk).all()
    canonical=pd.read_csv(REPO_ROOT/'data/ensembl/canonical_genes.csv')
    assert og.gene.tolist()==canonical.gene_symbol.tolist()
    for ratio in CONFIG['mask_ratios']:
        a,_=condition_masks('shared_vocab',MODELS[0],ids,ratio,0,og,bg,shared)
        b,_=condition_masks('shared_vocab',MODELS[1],ids,ratio,0,og,bg,shared)
        assert np.array_equal(a[:,shared.ours_index], b[:,shared.bulkformer_index])
        assert np.all(a.sum(1)==max(1,round(len(shared)*ratio)))
    return samples, ours, bulk, og, bg, shared


def cache_model(name):
    device=adapters.resolved_device()
    model=adapters.load_ours(device) if name==MODELS[0] else adapters.load_bulkformer(name,device)
    # Existing adapter and checkpoint behavior are preserved; retain the model across conditions.
    adapters.load_ours=lambda device:model
    adapters.load_bulkformer=lambda variant,device:model
    return model,device


def run(name,pilot=False):
    deterministic(0)
    samples,ours,bulk,og,bg,shared=check_inputs()
    ids=samples.sample_id.astype(str).tolist()
    space=MODEL_SPACE[name]; truth=ours if space=='ours' else bulk
    start=time.monotonic(); model,device=cache_model(name)
    load_seconds=time.monotonic()-start
    outdir=WORK/'fresh_conditions';outdir.mkdir(exist_ok=True)
    if pilot:
        timings=[]
        for ratio in [.15,.90,1.0]:
            mask,_=condition_masks('shared_vocab',name,ids,ratio,0,og,bg,shared)
            x=apply_mask(truth[:32],mask[:32]);deterministic(0)
            begin=time.monotonic()
            pred=adapters.predict(name,x,float((x==CONFIG['mask_token']).mean()),device,CONFIG['batch_sizes'][name])
            assert pred.shape==x.shape and np.isfinite(pred).all()
            scores=score_prediction(ids[:32],truth[:32],pred,mask[:32])
            assert np.isfinite(scores.mse).all()
            elapsed=time.monotonic()-begin
            timings.append({'mask_ratio':ratio,'samples':32,'seconds':elapsed,
                            'sample_rate':32/elapsed,'pearson_defined':int(scores.pearson.notna().sum())})
        report={'model':name,'device':str(device),'load_seconds':load_seconds,'timings':timings,
                'estimated_120_conditions_hours':float(np.mean([r['seconds'] for r in timings])*1000/32*120/3600),
                'status':'passed'}
        (RESULTS/f'pilot_{name}.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True);return
    assert json.loads((RESULTS/f'pilot_{name}.json').read_text())['status']=='passed'
    total=len(CONFIG['benchmark_modes'])*len(CONFIG['mask_ratios'])*len(CONFIG['mask_seeds']);done=0
    for mode in CONFIG['benchmark_modes']:
        for ratio in CONFIG['mask_ratios']:
            for seed in CONFIG['mask_seeds']:
                output=outdir/f'{mode}__{name}__r{round(ratio*100):03d}__s{seed:02d}.parquet'
                if not output.exists():
                    deterministic(seed)
                    mask,eligible=condition_masks(mode,name,ids,ratio,seed,og,bg,shared)
                    x=apply_mask(truth,mask)
                    pred=adapters.predict(name,x,float((x==CONFIG['mask_token']).mean()),device,CONFIG['batch_sizes'][name])
                    assert pred.shape==truth.shape and np.isfinite(pred).all()
                    frame=score_prediction(ids,truth,pred,mask,benchmark=mode,method=name,
                        target_space=space,mask_ratio=ratio,mask_seed=seed,
                        native_genes=truth.shape[1],evaluated_genes=int(eligible.sum()),
                        mask_sha256=hashlib.sha256(np.packbits(mask[:,shared.ours_index.to_numpy(int) if space=='ours' else shared.bulkformer_index.to_numpy(int)] if mode=='shared_vocab' else mask[:,eligible]).tobytes()).hexdigest())
                    assert np.isfinite(frame.mse).all()
                    temp=output.with_suffix('.tmp');frame.to_parquet(temp,index=False);temp.replace(output)
                done+=1;elapsed=time.monotonic()-start
                print(f'[progress] model={name} completed={done}/{total} elapsed={elapsed/3600:.2f}h '
                      f'rate={done/elapsed:.5f} conditions/s ETA={(total-done)*elapsed/done/3600:.2f}h',flush=True)
    (RESULTS/f'complete_{name}.json').write_text(json.dumps({'status':'complete','conditions':total,'seconds':time.monotonic()-start},indent=2))


def aggregate():
    samples,ours,bulk,og,bg,shared=check_inputs();ids=samples.sample_id.astype(str).tolist()
    frames=[];centers=load_frozen_baselines()
    for mode in CONFIG['benchmark_modes']:
        for ratio in CONFIG['mask_ratios']:
            for seed in CONFIG['mask_seeds']:
                for name in MODELS:
                    path=WORK/'fresh_conditions'/f'{mode}__{name}__r{round(ratio*100):03d}__s{seed:02d}.parquet'
                    frame=pd.read_parquet(path);assert frame.sample_id.tolist()==ids
                    frames.append(frame)
                if mode=='shared_vocab':
                    hashes=[frames[-i].mask_sha256.iloc[0] for i in range(1,4)]
                    assert len(set(hashes))==1, 'Model masks differ'
                for space,name,truth in [('ours',MODELS[0],ours),('bulkformer',MODELS[1],bulk)]:
                    mask,eligible=condition_masks(mode,name,ids,ratio,seed,og,bg,shared)
                    for stat in ['mean','median']:
                        frames.append(score_prediction(ids,truth,np.broadcast_to(centers[f'{space}_{stat}'],truth.shape),mask,
                            benchmark=mode,method=f'train_gene_{stat}',target_space=space,mask_ratio=ratio,
                            mask_seed=seed,native_genes=truth.shape[1],evaluated_genes=int(eligible.sum())))
    data=pd.concat(frames,ignore_index=True)
    keys=['benchmark','method','target_space','mask_ratio','mask_seed']
    assert not data.duplicated(keys+['sample_id']).any()
    per_seed=data.groupby(keys,as_index=False).agg(pearson=('pearson','mean'),spearman=('spearman','mean'),mse=('mse','mean'),
        samples=('sample_id','size'),pearson_defined=('pearson','count'),spearman_defined=('spearman','count'),evaluated_genes=('evaluated_genes','first'))
    assert per_seed.samples.eq(1000).all()
    summary=per_seed.groupby(keys[:-1],as_index=False).agg(
        **{f'{metric}_{stat}':(metric,stat) for metric in ['pearson','spearman','mse'] for stat in ['mean','std']},
        seeds=('mask_seed','size'),pearson_defined_min=('pearson_defined','min'),spearman_defined_min=('spearman_defined','min'))
    assert summary.seeds.eq(len(CONFIG['mask_seeds'])).all()
    data.to_parquet(RESULTS/'per_sample_results.parquet',index=False)
    per_seed.to_csv(RESULTS/'per_seed_results.csv',index=False);summary.to_csv(RESULTS/'summary_results.csv',index=False)
    qc=per_seed[keys+['samples','pearson_defined','spearman_defined']]
    qc.to_csv(RESULTS/'correlation_qc.csv',index=False)
    cancer=samples[['sample_id','cancer_type']]
    by_cancer=data.merge(cancer,on='sample_id',validate='many_to_one').groupby(keys+['cancer_type'],as_index=False).agg(
        pearson=('pearson','mean'),spearman=('spearman','mean'),mse=('mse','mean'),samples=('sample_id','size'))
    by_cancer.to_csv(RESULTS/'per_cancer_results.csv',index=False)
    by_cancer.groupby(keys,as_index=False)[['pearson','spearman','mse']].mean().to_csv(RESULTS/'cancer_macro_per_seed.csv',index=False)
    pd.crosstab(samples.cancer_type,columns='samples').to_csv(RESULTS/'cancer_counts.csv')
    print(f"All {len(MODELS)*len(CONFIG['benchmark_modes'])*len(CONFIG['mask_ratios'])*len(CONFIG['mask_seeds'])} model conditions and {4*len(CONFIG['benchmark_modes'])*len(CONFIG['mask_ratios'])*len(CONFIG['mask_seeds'])} baseline conditions validated.",flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=MODELS);p.add_argument('--pilot',action='store_true');p.add_argument('--aggregate',action='store_true')
    a=p.parse_args()
    if a.aggregate:aggregate()
    else:run(a.model,a.pilot)
