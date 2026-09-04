#!/usr/bin/env python3
"""Apply frozen Task 4 FE/RE to the independent Task 3 RR1/RR3 challenge."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

HERE=Path(__file__).resolve().parents[1]; ROOT=HERE.parents[1]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from task4_model import Disentangler
T3=ROOT/'benchmarks/osdr_batch_effect_representation'

def condition(s):
    return 'FLT' if '_FLT_' in s else ('GC' if '_GC_' in s else 'other')

def cosine(a,b):
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))

def responses(x, index, design):
    out={}
    for row in design.itertuples():
        ids=str(row.samples).split(' | '); f=[index[s] for s in ids if condition(s)=='FLT']; g=[index[s] for s in ids if condition(s)=='GC']
        if not f or not g: raise ValueError(f'Invalid FLT/GC design: {row.representation}')
        out[row.representation]=x[f].mean(0)-x[g].mean(0)
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--checkpoint',type=Path,default=HERE/'results/task4_disentanglement/full.pt'); p.add_argument('--output',type=Path,default=HERE/'results/task4g_task3_challenge'); p.add_argument('--device',default='cuda:0'); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    manifest=pd.read_csv(T3/'results/sample_manifest.csv'); z=np.load(T3/'work/bridgerna_embeddings.npy').astype(np.float32)
    if len(manifest)!=len(z): raise ValueError('Task 3 manifest/embedding mismatch')
    index={s:i for i,s in enumerate(manifest.sample_id)}
    design=pd.read_csv(T3/'results/task3_osd168_technical_replication/technical_response_design.csv')
    state=torch.load(a.checkpoint,map_location='cpu'); cfg=state['config']; model=Disentangler(hidden_dim=cfg['hidden_dim'],latent_dim=cfg['fe_dim'],dropout=cfg['dropout']); model.load_state_dict(state['model_state_dict']); device=torch.device(a.device if torch.cuda.is_available() else 'cpu'); model.to(device).eval()
    with torch.no_grad():
        out=model(torch.from_numpy(z).to(device)); fe=out['fe'].cpu().numpy(); re=out['re'].cpu().numpy()
    residual_path=a.checkpoint.parent/'linear_residualizer.joblib'
    reps={'Bridge':z,'Linear baseline':joblib.load(residual_path).transform(z),'FE':fe,'RE':re}; rows=[]
    comparisons=[('RR1','RR1_OSD48_original_matched','RR1_OSD168_no-ERCC'),('RR1 ERCC','RR1_OSD48_original_matched','RR1_OSD168_all_ERCC'),('RR3-39','C01_OSD137_original_matched','C01_OSD168_all_ERCC'),('RR3-40','C02_OSD137_original_matched','C02_OSD168_all_ERCC'),('ERCC/no-ERCC','RR1_OSD168_no-ERCC','RR1_OSD168_all_ERCC')]
    for name,x in reps.items():
        rv=responses(x,index,design)
        for label,left,right in comparisons:
            rows.append({'representation':name,'comparison':label,'left':left,'right':right,'cosine':cosine(rv[left],rv[right]),'spearman':spearmanr(rv[left],rv[right]).statistic,'left_norm':np.linalg.norm(rv[left]),'right_norm':np.linalg.norm(rv[right])})
    result=pd.DataFrame(rows); result.to_csv(a.output/'task3_challenge_metrics.csv',index=False)
    np.savez_compressed(a.output/'task3_fe_re_sample_embeddings.npz',sample_id=manifest.sample_id.to_numpy(dtype=object),FE=fe,RE=re)
    (a.output/'provenance.json').write_text(json.dumps({'task4_checkpoint':str(a.checkpoint),'task3_samples':len(manifest),'task3_used_for_training_or_selection':False,'mode_assignments_changed':False},indent=2)+'\n')
    print(result.to_string(index=False)); print(f'[complete] {a.output}',flush=True)
if __name__=='__main__': main()
