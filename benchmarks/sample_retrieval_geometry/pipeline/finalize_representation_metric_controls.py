#!/usr/bin/env python3
"""Finalize compact tables and figures from completed control retrievals."""
from pathlib import Path
import json, resource, time
import matplotlib.pyplot as plt
import pandas as pd

HERE=Path(__file__).resolve().parents[1]; OUT=HERE/'results/representation_metric_controls'; FIG=OUT/'figures'; WORK=HERE/'work/representation_metric_controls'
def main():
 t=time.time(); FIG.mkdir(parents=True,exist_ok=True)
 micro=pd.read_csv(OUT/'micro_metrics.csv'); macro=pd.read_csv(OUT/'study_macro_metrics.csv'); sep=pd.read_csv(OUT/'separation_metrics.csv')
 reps=['log1p(TPM)','PCA-512','PCA-512 whitened','Bridge','Bridge whitened']; fact=micro[micro.representation.isin(reps)].copy()
 sens=[]
 for rep,d in fact.groupby('representation'):
  best=d.loc[d.r_at_10.idxmax()]; worst=d.loc[d.r_at_10.idxmin()]; sens.append({'representation':rep,'best_metric':best.metric,'worst_metric':worst.metric,'best_r_at_10':best.r_at_10,'worst_r_at_10':worst.r_at_10,'delta_r_at_10':best.r_at_10-worst.r_at_10})
 pd.DataFrame(sens).to_csv(OUT/'metric_sensitivity_summary.csv',index=False)
 cfg=json.loads((HERE/'config.json').read_text()); emb=Path(cfg['inputs']['embedding_dir']); n=510709
 storage=pd.DataFrame([
  {'representation':'log1p(TPM)','dim':15165,'dtype':'float32','bytes_per_sample':15165*4,'relative_dimensions_to_bridge':15165/512,'cache_bytes':(WORK/'archs4_human_log1p_tpm.float32.mmap').stat().st_size},
  {'representation':'PCA-512','dim':512,'dtype':'float32','bytes_per_sample':512*4,'relative_dimensions_to_bridge':1,'cache_bytes':(WORK/'archs4_pca512.float32.mmap').stat().st_size},
  {'representation':'PCA-512 whitened','dim':512,'dtype':'float32','bytes_per_sample':512*4,'relative_dimensions_to_bridge':1,'cache_bytes':(WORK/'archs4_pca512.float32.mmap').stat().st_size},
  {'representation':'Bridge','dim':512,'dtype':'float16','bytes_per_sample':512*2,'relative_dimensions_to_bridge':1,'cache_bytes':(emb/'sample_embeddings.float16.mmap').stat().st_size},
  {'representation':'Bridge whitened','dim':512,'dtype':'float32','bytes_per_sample':512*4,'relative_dimensions_to_bridge':1,'cache_bytes':(WORK/'archs4_bridge_whitened.float32.mmap').stat().st_size}])
 storage.to_csv(OUT/'storage_comparison.csv',index=False)
 compute=fact.groupby('representation',as_index=False).agg(metric_sweeps=('metric','count'),retrieval_runtime_seconds=('runtime_seconds','sum'),maximum_single_metric_runtime_seconds=('runtime_seconds','max')).merge(storage,on='representation',how='left'); compute.to_csv(OUT/'computational_storage_summary.csv',index=False)
 micro.merge(macro,on=['representation','metric'],how='left').to_csv(OUT/'factorial_comparison.csv',index=False)
 fact['label']=fact.representation+' · '+fact.metric
 def bar(field,title,stem):
  d=fact.sort_values(field); fig,ax=plt.subplots(figsize=(11,7)); ax.barh(d.label,d[field]); ax.set(xlabel=field.replace('_',' ').upper(),title=title); fig.tight_layout(); fig.savefig(FIG/f'{stem}.png',dpi=220); fig.savefig(FIG/f'{stem}.pdf'); plt.close(fig)
 for f,tit,stem in [('r_at_1','Exact R@1 by representation and metric','r_at_1'),('r_at_10','Exact R@10 by representation and metric','r_at_10'),('r_at_100','Exact R@100 by representation and metric','r_at_100')]: bar(f,tit,stem)
 d=macro[macro.representation.isin(reps)].sort_values('macro_r_at_10'); fig,ax=plt.subplots(figsize=(11,7)); ax.barh(d.representation+' · '+d.metric,d.macro_r_at_10); ax.set(xlabel='Study-macro R@10',title='Equal-study retrieval'); fig.tight_layout(); fig.savefig(FIG/'macro_r_at_10.png',dpi=220); fig.savefig(FIG/'macro_r_at_10.pdf'); plt.close(fig)
 d=sep.sort_values('auroc'); fig,ax=plt.subplots(figsize=(11,7)); ax.barh(d.representation+' · '+d.metric,d.auroc); ax.set(xlim=(.5,1),xlabel='Exact-vs-random AUROC',title='Cross-pipeline separation'); fig.tight_layout(); fig.savefig(FIG/'separation_auroc.png',dpi=220); fig.savefig(FIG/'separation_auroc.pdf'); plt.close(fig)
 pc=pd.read_csv(OUT/'pca_eigenvalues.csv'); fig,ax=plt.subplots(figsize=(9,5)); ax.plot(pc.pc,pc.cumulative_explained_variance); ax.set(xlabel='PC',ylabel='Cumulative expression variance',title='ARCHS4 reference-only PCA'); fig.tight_layout(); fig.savefig(FIG/'pca_explained_variance.png',dpi=220); fig.savefig(FIG/'pca_explained_variance.pdf'); plt.close(fig)
 for filt,stem,title in [(['PCA-512','PCA-512 whitened'],'pca_whitening_retrieval','PCA whitening effect'),(['Bridge'],'bridge_metric_comparison','Raw Bridge metric sensitivity'),(['Bridge','Bridge whitened'],'bridge_whitening_retrieval','Bridge whitening and metric effects')]:
  d=fact[fact.representation.isin(filt)]; fig,ax=plt.subplots(figsize=(9,5))
  for rep in filt:
   z=d[d.representation.eq(rep)]; ax.plot(z.metric,z.r_at_10,marker='o',label=rep)
  ax.set(ylim=(0,1),ylabel='R@10',title=title); ax.legend(); fig.tight_layout(); fig.savefig(FIG/f'{stem}.png',dpi=220); fig.savefig(FIG/f'{stem}.pdf'); plt.close(fig)
 bestexpr=fact[fact.representation.isin(['log1p(TPM)','PCA-512','PCA-512 whitened'])].sort_values('r_at_10').groupby('representation').tail(1); bw=fact[(fact.representation.eq('Bridge whitened'))&(fact.metric.eq('Cosine'))]; d=pd.concat([bestexpr,bw]); fig,ax=plt.subplots(figsize=(8,5)); ax.bar(d.representation+' · '+d.metric,d.r_at_10); ax.tick_params(axis='x',rotation=25); ax.set(ylim=(0,1),ylabel='R@10',title='Best conventional controls versus Bridge whitening'); fig.tight_layout(); fig.savefig(FIG/'best_baseline_vs_bridge_whitened.png',dpi=220); fig.savefig(FIG/'best_baseline_vs_bridge_whitened.pdf'); plt.close(fig)
 prov={'finalization_runtime_seconds':time.time()-t,'retrieval_and_separation_runtime_seconds_approx':849,'peak_rss_gb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,'reference_fit_only':True,'query_fit':False,'expression_random_pairs':100000,'low_dim_random_pairs':2000000,'rank_orientation_asserted':True,'gtex_rerun':False,'retrieval_results_reused_from_completed_partial_run':True}; (OUT/'analysis_provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
 print('Finalized',OUT)
if __name__=='__main__': main()
