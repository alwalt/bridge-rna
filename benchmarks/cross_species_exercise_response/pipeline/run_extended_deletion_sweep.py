#!/usr/bin/env python3
"""Two-GPU IG/edgeR/random deletion sweep for fixed Task 2 latent axes."""
from __future__ import annotations
import argparse, json, multiprocessing as mp, queue, sys, time
from pathlib import Path
import matplotlib.pyplot as plt, numpy as np, pandas as pd, torch

HERE=Path(__file__).resolve().parents[1]; ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'benchmarks/tcga_downstream/pipeline')); sys.path.insert(0,str(ROOT))
from run_attention_pooling import load_frozen_encoder
from src.fm_embed.vocab import load_canonical_genes
from attribute_latent_axes import axis_directions, sample_score
OUT=HERE/'results/latent_axis_attribution'; WORK=HERE/'work/latent_axis_attribution/extended_deletion'; FIG=OUT/'figures'
AXES={'Axis A':['GSE108643','GSE86931','GSE126962','GSE132520'],'Axis B':['GSE71972','GSE87748','GSE97718']}
SIZES=[25,50,100,250,500,1000]; TOTAL=2*(len(SIZES)*2+len(SIZES)*5); REUSED=6

def score(model,matrix,direction,mask,device,batch=4):
    out=[]; d=torch.from_numpy(direction).to(device).unsqueeze(0)
    with torch.inference_mode():
      for start in range(0,len(matrix),batch):
        x=torch.from_numpy(np.array(matrix[start:start+batch],copy=True)).to(device); x[:,mask]=-10.0
        out.append(sample_score(model,x,d).cpu().numpy())
    return np.concatenate(out)

def worker(axis,device_name,events,seed):
    try:
      genes=load_canonical_genes(ROOT/'data/ensembl/canonical_genes.csv'); manifest=pd.read_parquet(HERE/'results/matched_manifest.parquet'); members=pd.read_parquet(HERE/'results/contrast_members.parquet'); meta=pd.read_csv(HERE/'results/response_contrasts.csv').sort_values('contrast_id').reset_index(drop=True)
      expression=np.load(HERE/'work/matched_log1p_tpm_corrected.npy',mmap_mode='r'); effects=np.load(HERE/'work/response_effects_bridgerna.npy'); directions=axis_directions(meta,effects); ig=np.load(HERE/'work/latent_axis_attribution/study_integrated_gradient_changes.npy')
      meta_index={g:i for i,g in enumerate(meta.GSE)}; ig_score=ig[[meta_index[g] for g in AXES[axis]]].mean(0); ig_order=np.argsort(-np.abs(ig_score))
      de=pd.read_parquet(OUT/'de_ig_enrichment/axis_consensus_de_rankings.parquet').query('axis == @axis').sort_values('consensus_de_rank'); gene_index={g:i for i,g in enumerate(genes)}; de_order=np.array([gene_index[g] for g in de.gene])
      lookup=manifest.reset_index().set_index('GSM')['index']; profiles=[]; profile_keys=[]
      for gse in AXES[axis]:
        group=members[members.GSE.eq(gse)]
        for role in ['post_exercise','pre_control']:
          idx=lookup.loc[group.loc[group.role.eq(role),'GSM']].to_numpy(int); profiles.append(np.asarray(expression[idx]).mean(0)); profile_keys.append((gse,role))
      matrix=np.stack(profiles).astype(np.float32); role_lookup={x:i for i,x in enumerate(profile_keys)}; device=torch.device(device_name); model=load_frozen_encoder(device); direction=directions[axis]
      original=score(model,matrix,direction,np.array([],dtype=int),device); original_effect={g:original[role_lookup[(g,'post_exercise')]]-original[role_lookup[(g,'pre_control')]] for g in AXES[axis]}
      prior=pd.read_parquet(OUT/'deletion_test_results.parquet'); reused=prior.query("axis == @axis and panel_type == 'top' and genes_masked in [25,50,100]").copy(); reused['panel_type']='ig_ranked'; rows=reused.to_dict('records'); panels=[]
      conditions=[]
      for size in SIZES:
        if size not in [25,50,100]: conditions.append(('ig_ranked',0,size,ig_order[:size]))
        conditions.append(('de_ranked',0,size,de_order[:size]))
        for rep in range(5):
          rng=np.random.default_rng(np.random.SeedSequence([seed,0 if axis=='Axis A' else 1,size,rep])); conditions.append(('random',rep,size,rng.choice(len(genes),size,replace=False)))
      for panel_type,rep,size,panel in conditions:
        panels.extend({'axis':axis,'panel_type':panel_type,'replicate':rep,'genes_masked':size,'model_index':int(i),'gene':genes[i]} for i in panel)
        changed=score(model,matrix,direction,np.asarray(panel,dtype=int),device)
        for gse in AXES[axis]:
          value=changed[role_lookup[(gse,'post_exercise')]]-changed[role_lookup[(gse,'pre_control')]]; orig=original_effect[gse]
          rows.append({'axis':axis,'GSE':gse,'species':meta.loc[meta.GSE.eq(gse),'species'].iloc[0],'genes_masked':size,'panel_type':panel_type,'replicate':rep,'original_axis_effect':orig,'masked_axis_effect':value,'absolute_score_change':abs(value-orig),'fraction_signal_remaining':value/orig if abs(orig)>1e-9 else np.nan})
        events.put(('done',axis,panel_type,rep,size))
      pd.DataFrame(rows).to_parquet(WORK/f"{axis.replace(' ','_').lower()}_results.parquet",index=False); pd.DataFrame(panels).to_parquet(WORK/f"{axis.replace(' ','_').lower()}_panels.parquet",index=False); events.put(('finished',axis))
    except Exception as exc: events.put(('error',axis,repr(exc))); raise

def main():
    p=argparse.ArgumentParser(); p.add_argument('--devices',nargs=2,default=['cuda:0','cuda:1']); p.add_argument('--seed',type=int,default=20260903); p.add_argument('--heartbeat-seconds',type=int,default=30); a=p.parse_args()
    if torch.cuda.device_count()<2: raise RuntimeError('Two CUDA GPUs are required for this requested run')
    OUT.mkdir(parents=True,exist_ok=True); WORK.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
    print(f'[estimate] conditions={TOTAL} reused={REUSED} new={TOTAL-REUSED} GPUs=2 approximate_runtime=5-10m',flush=True)
    ctx=mp.get_context('spawn'); events=ctx.Queue(); procs=[ctx.Process(target=worker,args=(axis,device,events,a.seed)) for axis,device in zip(AXES,a.devices)]; started=time.monotonic(); completed=REUSED; finished=0; last=started
    for proc in procs: proc.start()
    while finished<len(procs):
      try: event=events.get(timeout=a.heartbeat_seconds)
      except queue.Empty:
        elapsed=time.monotonic()-started; rate=max(0,completed-REUSED)/elapsed; eta=(TOTAL-completed)/rate if rate else float('nan'); print(f'[heartbeat] completed={completed}/{TOTAL} elapsed={elapsed/60:.1f}m ETA={eta/60:.1f}m workers_finished={finished}/2',flush=True); continue
      if event[0]=='done':
        completed+=1; elapsed=time.monotonic()-started; rate=(completed-REUSED)/elapsed; eta=(TOTAL-completed)/rate
        print(f'[progress] completed={completed}/{TOTAL} axis={event[1]} panel={event[2]} size={event[4]} replicate={event[3]} elapsed={elapsed/60:.1f}m ETA={eta/60:.1f}m',flush=True)
      elif event[0]=='finished': finished+=1; print(f'[worker] {event[1]} finished ({finished}/2)',flush=True)
      else: raise RuntimeError(f'{event[1]} worker failed: {event[2]}')
    for proc in procs: proc.join()
    if any(p.exitcode for p in procs): raise RuntimeError(f'Worker exit codes {[p.exitcode for p in procs]}')
    result=pd.concat([pd.read_parquet(WORK/f'{x}_results.parquet') for x in ['axis_a','axis_b']],ignore_index=True); panels=pd.concat([pd.read_parquet(WORK/f'{x}_panels.parquet') for x in ['axis_a','axis_b']],ignore_index=True)
    result.to_parquet(OUT/'extended_deletion_sweep.parquet',index=False); panels.to_parquet(OUT/'extended_deletion_panels.parquet',index=False)
    summary=result.groupby(['axis','genes_masked','panel_type'],as_index=False).agg(fraction_remaining_mean=('fraction_signal_remaining','mean'),fraction_remaining_sd=('fraction_signal_remaining','std'),absolute_change_mean=('absolute_score_change','mean'),observations=('GSE','size'),studies=('GSE','nunique'),replicates=('replicate','nunique')); summary.to_csv(OUT/'extended_deletion_sweep_summary.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=True); colors={'ig_ranked':'#4C72B0','de_ranked':'#C44E52','random':'#999999'}
    for ax,axis in zip(axes,AXES):
      for panel in ['ig_ranked','de_ranked','random']:
        z=summary.query('axis == @axis and panel_type == @panel'); label={'ig_ranked':'IG-ranked','de_ranked':'edgeR DE-ranked','random':'Random'}[panel]; ax.plot(z.genes_masked,z.fraction_remaining_mean,marker='o',label=label,color=colors[panel]); ax.fill_between(z.genes_masked,z.fraction_remaining_mean-z.fraction_remaining_sd,z.fraction_remaining_mean+z.fraction_remaining_sd,color=colors[panel],alpha=.15)
      ax.axhline(1,color='black',ls='--',lw=1); ax.set(title=axis,xlabel='Genes deleted',ylabel='Fraction of original latent response remaining'); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG/'extended_deletion_sweep.png',dpi=350,bbox_inches='tight'); fig.savefig(FIG/'extended_deletion_sweep.pdf',bbox_inches='tight'); plt.close(fig)
    provenance={'sizes':SIZES,'panels':['IG-ranked','edgeR DE-ranked within 15,165','5 deterministic random panels'],'seed':a.seed,'mask_value':-10.0,'outcome':'post-minus-pre latent-axis projection after deletion divided by original projection','IG_recomputed':False,'model_retrained':False,'reused':'IG-ranked 25/50/100 results from deletion_test_results.parquet','devices':a.devices}
    (OUT/'extended_deletion_sweep_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n'); print(f'[complete] completed={TOTAL}/{TOTAL} elapsed={(time.monotonic()-started)/60:.1f}m',flush=True)
if __name__=='__main__': main()
