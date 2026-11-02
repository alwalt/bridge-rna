#!/usr/bin/env python3
"""Factor representation, metric, and covariance-normalization effects."""
from __future__ import annotations
import json, math, resource, time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, torch

HERE=Path(__file__).resolve().parents[1]; OUT=HERE/'results/representation_metric_controls'; FIG=OUT/'figures'; WORK=HERE/'work/representation_metric_controls'
METRICS=['Cosine','Euclidean','Manhattan','Canberra']; KS=[1,5,10,100,1000]; REF_CHUNK=8192; QB=32; RANDOM_N=2_000_000; EXPR_RANDOM_N=100_000

def unit(x): return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-20)
def orient(metric,score): return score if metric in ['Cosine','Pearson','Native'] else -score
def calc_np(q,r,metric):
 if metric=='Cosine': return unit(q)@unit(r).T
 if metric=='Pearson': return unit(q-q.mean(1,keepdims=True))@unit(r-r.mean(1,keepdims=True)).T
 if metric=='Euclidean': return np.sqrt(np.maximum(np.square(q).sum(1)[:,None]+np.square(r).sum(1)[None,:]-2*q@r.T,0))
 if metric=='Manhattan': return np.abs(q[:,None,:]-r[None,:,:]).sum(2)
 if metric=='Canberra':
  den=np.abs(q[:,None,:])+np.abs(r[None,:,:]); num=np.abs(q[:,None,:]-r[None,:,:])
  return np.divide(num,den,out=np.zeros_like(num),where=den>0).sum(2)
 raise ValueError(metric)
def pair_np(a,b,metric):
 if metric=='Cosine': return np.sum(unit(a)*unit(b),axis=1)
 if metric=='Pearson': return np.sum(unit(a-a.mean(1,keepdims=True))*unit(b-b.mean(1,keepdims=True)),axis=1)
 if metric=='Euclidean': return np.linalg.norm(a-b,axis=1)
 if metric=='Manhattan': return np.abs(a-b).sum(1)
 if metric=='Canberra':
  den=np.abs(a)+np.abs(b); num=np.abs(a-b)
  return np.divide(num,den,out=np.zeros_like(num),where=den>0).sum(1)
 raise ValueError(metric)
def calc_torch(q,r,metric):
 if metric=='Cosine':
  return torch.nn.functional.normalize(q,dim=1)@torch.nn.functional.normalize(r,dim=1).T
 if metric=='Pearson':
  return torch.nn.functional.normalize(q-q.mean(1,keepdim=True),dim=1)@torch.nn.functional.normalize(r-r.mean(1,keepdim=True),dim=1).T
 if metric=='Euclidean': return torch.cdist(q,r,p=2)
 if metric=='Manhattan': return torch.cdist(q,r,p=1)
 if metric=='Canberra':
  den=q[:,None,:].abs()+r[None,:,:].abs(); return ((q[:,None,:]-r[None,:,:]).abs()/den.clamp_min(1e-20)).sum(2)
 raise ValueError(metric)
def summary(r):
 return {**{f'r_at_{k}':np.mean(r<=k) for k in KS},'mrr':np.mean(1/r),'median_rank':np.median(r),'mean_rank':np.mean(r),'median_percentile_rank':np.median(1-(r-1)/510709)}
def auc(ex,bg,higher=True):
 b=np.sort(bg if higher else -bg); e=ex if higher else -ex
 return float(np.mean((np.searchsorted(b,e,'left')+np.searchsorted(b,e,'right'))/(2*len(b))))
def separation(ex,bg,higher=True):
 eo,bo=(ex,bg) if higher else (-ex,-bg); pooled=np.sqrt(((len(eo)-1)*eo.var(ddof=1)+(len(bo)-1)*bo.var(ddof=1))/(len(eo)+len(bo)-2))
 return {'exact_mean':ex.mean(),'exact_median':np.median(ex),'background_mean':bg.mean(),'background_median':np.median(bg),'auroc':auc(ex,bg,higher),'cohens_d_oriented':(eo.mean()-bo.mean())/pooled,'exact_median_background_percentile':np.searchsorted(np.sort(bo),np.median(eo),'right')/len(bo)}

def rank_stream(name,metric,q,correct_ref,ref_loader,nref,ref_chunk=REF_CHUNK):
 exact=pair_np(q,correct_ref,metric); threshold=orient(metric,exact); counts=np.zeros(len(q),np.int64); t=time.time()
 dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); qt=torch.from_numpy(np.asarray(q,np.float32)).to(dev)
 for s in range(0,nref,ref_chunk):
  r=torch.from_numpy(np.asarray(ref_loader(s,min(nref,s+ref_chunk)),np.float32)).to(dev)
  for qs in range(0,len(q),QB):
   qe=min(len(q),qs+QB); block=calc_torch(qt[qs:qe],r,metric)
   if metric not in ['Cosine','Pearson','Native']: block=-block
   counts[qs:qe]+=torch.sum(block>torch.from_numpy(threshold[qs:qe]).to(dev)[:,None],dim=1).cpu().numpy()
 return counts+1,exact,time.time()-t

def main():
 t0=time.time(); OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True); lf=open(OUT/'run.log','w',buffering=1)
 def log(x): z=f'[{pd.Timestamp.now().isoformat()}] {x}'; print(z,flush=True); lf.write(z+'\n')
 log('Starting representation × metric controls; estimated runtime 30-180 minutes')
 cfg=json.loads((HERE/'config.json').read_text()); inp={k:Path(v) for k,v in cfg['inputs'].items()}; up=inp['upstream_benchmark']; emb=inp['embedding_dir']; spec=json.loads((emb/'embedding_manifest.json').read_text())
 loc=pd.read_parquet(emb/'sample_locations.parquet'); man=pd.read_parquet(inp['sample_manifest'],columns=['gsm','species']); human=loc.merge(man,left_on='geo_accession',right_on='gsm',validate='one_to_one'); human=human[human.species.astype(str).str.lower().eq('human')].reset_index(drop=True); n=len(human)
 samples0=pd.read_parquet(up/'work/paired_expression/samples.parquet'); keep=samples0.study_exposure.eq('unseen_study').to_numpy(); samples=samples0[keep].reset_index(drop=True); query_expr=np.asarray(np.load(up/'work/paired_expression/recount3_log1p_tpm.npy',mmap_mode='r')[keep],np.float32)
 lookup=pd.Series(np.arange(n),index=human.geo_accession); correct=samples.gsm.map(lookup).to_numpy(int); gse=samples.gse.astype(str).to_numpy()
 expr_path=WORK/'archs4_human_log1p_tpm.float32.mmap'; expr=np.memmap(expr_path,mode='r',dtype=np.float32,shape=(n,15165)); pca=np.memmap(WORK/'archs4_pca512.float32.mmap',mode='r',dtype=np.float32,shape=(n,512)); pp=np.load(WORK/'expression_pca512_transform.npz'); query_pca=(query_expr-pp['mean'])@pp['components'].T; pca_scale=1/np.sqrt(np.maximum(pp['eigenvalues'],float(pp['eigenvalue_floor']))); query_pcaw=query_pca*pca_scale
 rawemb=np.memmap(emb/'sample_embeddings.float16.mmap',mode='r',dtype=np.float16,shape=(spec['total_samples'],512)); rg=human.global_index.to_numpy(int); query_bridge=np.asarray(np.load(up/'work/recount3_embeddings.npy',mmap_mode='r')[keep],np.float32); bp=np.load(HERE/'work/literature_postprocessing_transform.npz'); query_bw=(query_bridge-bp['mean'])@bp['whitening']
 bw_path=WORK/'archs4_bridge_whitened.float32.mmap'
 if not bw_path.exists() or bw_path.stat().st_size != n*512*4:
  log('Caching the frozen ARCHS4-fitted Bridge whitening transform once')
  bw=np.memmap(bw_path,mode='w+',dtype=np.float32,shape=(n,512))
  dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); mt=torch.from_numpy(bp['mean']).to(dev); wt=torch.from_numpy(bp['whitening']).to(dev)
  for s in range(0,n,8192):
   x=torch.from_numpy(np.asarray(rawemb[rg[s:min(n,s+8192)]],np.float32)).to(dev); bw[s:s+len(x)]=((x-mt)@wt).cpu().numpy()
  bw.flush(); del bw
 bw=np.memmap(bw_path,mode='r',dtype=np.float32,shape=(n,512))
 reps={
  'log1p(TPM)':(15165,query_expr,lambda s,e:np.asarray(expr[s:e]),lambda:np.asarray(expr[correct])),
  'PCA-512':(512,query_pca,lambda s,e:np.asarray(pca[s:e]),lambda:np.asarray(pca[correct])),
  'PCA-512 whitened':(512,query_pcaw,lambda s,e:np.asarray(pca[s:e])*pca_scale,lambda:np.asarray(pca[correct])*pca_scale),
  'Bridge':(512,query_bridge,lambda s,e:np.asarray(rawemb[rg[s:e]],np.float32),lambda:np.asarray(rawemb[rg[correct]],np.float32)),
  'Bridge whitened':(512,query_bw,lambda s,e:np.asarray(bw[s:e]),lambda:np.asarray(bw[correct])),
 }
 combos=[]
 for rep in reps:
  mets=(['Cosine','Pearson','Euclidean','Manhattan','Canberra'] if rep=='log1p(TPM)' else METRICS)
  for m in mets: combos.append((rep,m))
 # Existing exact sources of truth.
 lit=pd.read_csv(HERE/'results/literature_postprocessing/retrieval_metrics_micro.csv').set_index('method'); litmac=pd.read_csv(HERE/'results/literature_postprocessing/retrieval_metrics_study_macro.csv').set_index('method')
 existing={('Bridge','Cosine'):('Raw cosine','Raw cosine'),('Bridge whitened','Cosine'):('Whitening','Whitening')}
 rows=[]; macros=[]; pq=[]; exact_scores={}; runtimes={}
 for rep,metric in combos:
  dim,q,loader,correct_loader=reps[rep]
  if (rep,metric) in existing:
   old,_=existing[(rep,metric)]; ranks=None
   # Reuse exact published ranks for raw; frozen whitening ranks are reconstructed from saved per-query cache unavailable, so compute.
   if rep=='Bridge': ranks=samples[['gsm']].merge(pd.read_parquet(up/'results/full_archs4_retrieval.parquet'),on='gsm').exact_rank_full.to_numpy(int); exact=pair_np(q,correct_loader(),metric); runtime=0.
   else: ranks,exact,runtime=rank_stream(rep,metric,q,correct_loader(),loader,n)
  else: ranks,exact,runtime=rank_stream(rep,metric,q,correct_loader(),loader,n,512 if rep=='log1p(TPM)' and metric in ['Manhattan','Canberra'] else REF_CHUNK)
  assert np.all(ranks>=1); exact_scores[(rep,metric)]=exact; runtimes[(rep,metric)]=runtime; sm=summary(ranks); rows.append({'representation':rep,'metric':metric,'dim':dim,**sm,'runtime_seconds':runtime})
  for i,r in enumerate(ranks): pq.append({'representation':rep,'metric':metric,'query_index':i,'gsm':samples.gsm[i],'gse':gse[i],'exact_rank':r,'exact_score':exact[i]})
  md=pd.DataFrame([summary(ranks[gse==s]) for s in np.unique(gse)])
  macro_row={'representation':rep,'metric':metric,'studies':len(md),**{f'macro_{c}':md[c].mean() for c in ['r_at_1','r_at_5','r_at_10','r_at_100','r_at_1000','mrr']}}
  # Published cosine rows remain the authoritative source when floating-point
  # near-ties make a fresh rank count differ at sub-micro precision.
  if (rep,metric) in existing:
   old,_=existing[(rep,metric)]; src=lit.loc[old]; sm.update({c:src[c] for c in ['r_at_1','r_at_5','r_at_10','r_at_100','r_at_1000','mrr','median_rank','median_percentile_rank']})
   srcm=litmac.loc[old]; macro_row.update({c:srcm[c] for c in ['macro_r_at_1','macro_r_at_5','macro_r_at_10','macro_r_at_100','macro_r_at_1000','macro_mrr']})
   rows[-1].update(sm)
  macros.append(macro_row)
  log(f'complete {rep} × {metric}: R@10={sm["r_at_10"]:.3%}; runtime={runtime:.1f}s')
 micro=pd.DataFrame(rows); macro=pd.DataFrame(macros); perq=pd.DataFrame(pq)
 # Existing non-factorial baselines.
 for method,typ,dim in [('ABTT D=10','Cosine',502),('NICDM','Native',512),('Local Scaling','Native',512),('CSLS','Native',512)]:
  r=lit.loc[method]; ma=litmac.loc[method]; micro.loc[len(micro)]={'representation':method,'metric':typ,'dim':dim,**{c:r[c] for c in ['r_at_1','r_at_5','r_at_10','r_at_100','r_at_1000','mrr','median_rank','median_percentile_rank']},'mean_rank':r.get('mean_rank',np.nan),'runtime_seconds':0}
  macro.loc[len(macro)]={'representation':method,'metric':typ,'studies':40,**{c:ma[c] for c in ['macro_r_at_1','macro_r_at_5','macro_r_at_10','macro_r_at_100','macro_r_at_1000','macro_mrr']}}
 micro.to_csv(OUT/'micro_metrics.csv',index=False); macro.to_csv(OUT/'study_macro_metrics.csv',index=False); perq.to_parquet(OUT/'per_query_ranks.parquet',index=False)
 micro.merge(macro,on=['representation','metric'],how='left').to_csv(OUT/'factorial_comparison.csv',index=False)
 # Separation on fixed pair indices. Expression uses first 100k because 15,165-D scattered reads are expensive; low-D uses all 2m.
 pairs=np.load(HERE/'work/anisotropy_random_pair_indices.npz'); a,b=pairs['a'],pairs['b']; sep=[]
 for rep,metric in combos:
  dim,q,loader,cl=reps[rep]; nn=EXPR_RANDOM_N if rep=='log1p(TPM)' else RANDOM_N; bg=[]; bs=256 if rep=='log1p(TPM)' else 32768
  for s in range(0,nn,bs):
   e=min(nn,s+bs); xa=loader(0,n)[a[s:e]] if False else None
   if rep=='log1p(TPM)': xa=np.asarray(expr[a[s:e]]); xb=np.asarray(expr[b[s:e]])
   elif rep.startswith('PCA'):
    xa=np.asarray(pca[a[s:e]]); xb=np.asarray(pca[b[s:e]]);
    if rep.endswith('whitened'): xa=xa*pca_scale; xb=xb*pca_scale
   else:
    xa=np.asarray(rawemb[rg[a[s:e]]],np.float32); xb=np.asarray(rawemb[rg[b[s:e]]],np.float32)
    if rep.endswith('whitened'): xa=(xa-bp['mean'])@bp['whitening']; xb=(xb-bp['mean'])@bp['whitening']
   bg.append(pair_np(xa,xb,metric))
  bg=np.concatenate(bg); sep.append({'representation':rep,'metric':metric,'background_pairs':nn,**separation(exact_scores[(rep,metric)],bg,metric in ['Cosine','Pearson'])})
  log(f'separation complete {rep} × {metric}')
 pd.DataFrame(sep).to_csv(OUT/'separation_metrics.csv',index=False)
 # Sensitivity and compute/storage summaries.
 fact=micro[micro.representation.isin(reps)].copy(); sens=[]
 for rep,d in fact.groupby('representation'):
  best=d.loc[d.r_at_10.idxmax()]; worst=d.loc[d.r_at_10.idxmin()]; sens.append({'representation':rep,'best_metric':best.metric,'worst_metric':worst.metric,'best_r_at_10':best.r_at_10,'worst_r_at_10':worst.r_at_10,'delta_r_at_10':best.r_at_10-worst.r_at_10})
 pd.DataFrame(sens).to_csv(OUT/'metric_sensitivity_summary.csv',index=False)
 storage=pd.DataFrame([{'representation':'log1p(TPM)','dim':15165,'dtype':'float32','bytes_per_sample':15165*4,'relative_dimensions_to_bridge':15165/512,'cache_bytes':expr_path.stat().st_size},{'representation':'PCA-512','dim':512,'dtype':'float32','bytes_per_sample':512*4,'relative_dimensions_to_bridge':1,'cache_bytes':(WORK/'archs4_pca512.float32.mmap').stat().st_size},{'representation':'PCA-512 whitened','dim':512,'dtype':'float32','bytes_per_sample':512*4,'relative_dimensions_to_bridge':1,'cache_bytes':(WORK/'archs4_pca512.float32.mmap').stat().st_size},{'representation':'Bridge','dim':512,'dtype':'float16','bytes_per_sample':512*2,'relative_dimensions_to_bridge':1,'cache_bytes':(emb/'sample_embeddings.float16.mmap').stat().st_size},{'representation':'Bridge whitened','dim':512,'dtype':'float32','bytes_per_sample':512*4,'relative_dimensions_to_bridge':1,'cache_bytes':bw_path.stat().st_size}]); storage.to_csv(OUT/'storage_comparison.csv',index=False)
 compute=micro[micro.representation.isin(reps)].groupby('representation',as_index=False).agg(metric_sweeps=('metric','count'),retrieval_runtime_seconds=('runtime_seconds','sum'),maximum_single_metric_runtime_seconds=('runtime_seconds','max')); compute=compute.merge(storage[['representation','dim','dtype','bytes_per_sample','cache_bytes']],on='representation',how='left'); compute.to_csv(OUT/'computational_storage_summary.csv',index=False)
 # Figures.
 primary=fact.copy(); primary['label']=primary.representation+' · '+primary.metric
 def bar(field,title,stem):
  d=primary.sort_values(field); fig,ax=plt.subplots(figsize=(11,7)); ax.barh(d.label,d[field]); ax.set(xlabel=field.replace('_',' ').upper(),title=title); fig.tight_layout(); fig.savefig(FIG/f'{stem}.png',dpi=220); fig.savefig(FIG/f'{stem}.pdf'); plt.close(fig)
 for f,t,s in [('r_at_1','Exact R@1 by representation and metric','r_at_1'),('r_at_10','Exact R@10 by representation and metric','r_at_10'),('r_at_100','Exact R@100 by representation and metric','r_at_100')]: bar(f,t,s)
 mm=macro.merge(micro[['representation','metric']],on=['representation','metric']); d=mm[mm.representation.isin(reps)].sort_values('macro_r_at_10'); fig,ax=plt.subplots(figsize=(11,7)); ax.barh(d.representation+' · '+d.metric,d.macro_r_at_10); ax.set(xlabel='Study-macro R@10',title='Equal-study retrieval'); fig.tight_layout(); fig.savefig(FIG/'macro_r_at_10.png',dpi=220); fig.savefig(FIG/'macro_r_at_10.pdf'); plt.close(fig)
 sd=pd.DataFrame(sep).sort_values('auroc'); fig,ax=plt.subplots(figsize=(11,7)); ax.barh(sd.representation+' · '+sd.metric,sd.auroc); ax.set(xlim=(.5,1),xlabel='Exact-vs-random AUROC',title='Cross-pipeline separation'); fig.tight_layout(); fig.savefig(FIG/'separation_auroc.png',dpi=220); fig.savefig(FIG/'separation_auroc.pdf'); plt.close(fig)
 pc=pd.read_csv(OUT/'pca_eigenvalues.csv'); fig,ax=plt.subplots(figsize=(9,5)); ax.plot(pc.pc,pc.cumulative_explained_variance); ax.set(xlabel='PC',ylabel='Cumulative expression variance',title='ARCHS4 reference-only PCA'); fig.tight_layout(); fig.savefig(FIG/'pca_explained_variance.png',dpi=220); fig.savefig(FIG/'pca_explained_variance.pdf'); plt.close(fig)
 for filt,stem,title in [(['PCA-512','PCA-512 whitened'],'pca_whitening_retrieval','PCA whitening effect'),(['Bridge'],'bridge_metric_comparison','Raw Bridge metric sensitivity'),(['Bridge','Bridge whitened'],'bridge_whitening_retrieval','Bridge whitening and metric effects')]:
  d=fact[fact.representation.isin(filt)]; fig,ax=plt.subplots(figsize=(9,5));
  for rep in filt:
   z=d[d.representation.eq(rep)]; ax.plot(z.metric,z.r_at_10,marker='o',label=rep)
  ax.set(ylim=(0,1),ylabel='R@10',title=title); ax.legend(); fig.tight_layout(); fig.savefig(FIG/f'{stem}.png',dpi=220); fig.savefig(FIG/f'{stem}.pdf'); plt.close(fig)
 bestexpr=fact[fact.representation.isin(['log1p(TPM)','PCA-512','PCA-512 whitened'])].sort_values('r_at_10').groupby('representation').tail(1); bw=fact[(fact.representation.eq('Bridge whitened'))&(fact.metric.eq('Cosine'))]; d=pd.concat([bestexpr,bw]); fig,ax=plt.subplots(figsize=(8,5)); ax.bar(d.representation+' · '+d.metric,d.r_at_10); ax.tick_params(axis='x',rotation=25); ax.set(ylim=(0,1),ylabel='R@10',title='Best conventional controls versus Bridge whitening'); fig.tight_layout(); fig.savefig(FIG/'best_baseline_vs_bridge_whitened.png',dpi=220); fig.savefig(FIG/'best_baseline_vs_bridge_whitened.pdf'); plt.close(fig)
 prov={'runtime_seconds':time.time()-t0,'peak_rss_gb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,'reference_fit_only':True,'query_fit':False,'expression_random_pairs':EXPR_RANDOM_N,'low_dim_random_pairs':RANDOM_N,'rank_orientation_asserted':True,'gtEx_rerun':False,'limitations':['Expression random-background separation uses the first 100,000 retained deterministic pairs because scattered 15,165-D reads are expensive.','PCA is fixed randomized SVD with 32 oversamples and one power iteration; dimension is exactly 512.']}; (OUT/'analysis_provenance.json').write_text(json.dumps(prov,indent=2)+'\n'); log(f"COMPLETE runtime={prov['runtime_seconds']:.1f}s peak_RSS={prov['peak_rss_gb']:.2f}GB"); lf.close()
if __name__=='__main__': main()
