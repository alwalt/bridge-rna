#!/usr/bin/env python3
"""Expression-adjusted contextual sensitivity for the final Task 4 control."""
from __future__ import annotations
import json, time
from pathlib import Path
import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import hypergeom, rankdata, spearmanr
from statsmodels.nonparametric.smoothers_lowess import lowess

HERE=Path(__file__).resolve().parents[1];REPO=HERE.parents[1]
OUT=HERE/'results/task4_confounding_profiler/expression_adjusted_context';FIG=OUT/'figures'
BASE=HERE/'results/task4_confounding_profiler';CONV=BASE/'conventional_expression_baseline';CTRL=BASE/'controlled_gene_context';RR=BASE/'contextual_robustness'
GMT_ROOT=REPO/'benchmarks/cross_species_exercise_response/results/per_study_ranked_gsea';GMT={'GO:BP':'GO_Biological_Process_2026.gmt','KEGG':'KEGG_2026.gmt','REAC':'Reactome_Pathways_2024.gmt'}
SEED=42931;N_PERM=1000

def family(term):
 t=term.upper()
 if 'SPLIC' in t or 'RNA PROCESS' in t or 'MRNA' in t:return 'RNA processing / splicing'
 if 'CHROMATIN' in t:return 'Chromatin organization / remodeling'
 if 'DNA REPAIR' in t or 'DNA METABOL' in t:return 'DNA repair / DNA metabolism'
 return 'Other'

def residualize(x,y,frac=.20):
 order=np.argsort(x);fit_sorted=lowess(y[order],x[order],frac=frac,it=3,return_sorted=False);fit=np.empty_like(fit_sorted);fit[order]=fit_sorted
 resid=y-fit;return fit,resid,(resid-resid.mean())/resid.std(ddof=1)

def build_tables():
 t=pd.read_csv(CONV/'tcell_edger.csv').query('tested').copy();tc=pd.read_parquet(CTRL/'controlled_gene_sensitivity.parquet')
 a=t[['gene_symbol','logFC','F','PValue','FDR','instability_statistic']].merge(tc,on='gene_symbol',validate='one_to_one')
 a=a.rename(columns={'instability_statistic':'expression_statistic','sensitivity_score':'contextual_statistic','mean_displacement_magnitude':'contextual_displacement_magnitude','loo_directional_consistency':'contextual_directional_consistency','mean_log1p_tpm':'expression_abundance'})
 r=pd.read_csv(CONV/'rr1_edger.csv').query('tested').copy();rc=pd.read_parquet(RR/'gene_level_metric_audit.parquet').query("comparison=='RR1'")
 b=r[['gene_symbol','logFC','F','PValue','FDR','instability_statistic']].merge(rc,on='gene_symbol',validate='one_to_one')
 b=b.rename(columns={'instability_statistic':'expression_statistic','normalized_context_discrepancy':'contextual_statistic','context_discrepancy_norm':'contextual_discrepancy_magnitude','context_reproducibility':'contextual_reproducibility','combined_response_magnitude':'contextual_response_magnitude','mean_log1p_tpm':'expression_abundance'})
 for label,q in [('controlled_tcell',a),('RR1',b)]:
  x=np.log1p(q.expression_statistic.to_numpy(float));y0=q.contextual_statistic.to_numpy(float);y=np.log1p(y0) if label=='controlled_tcell' else y0
  fit,resid,z=residualize(x,y);q['expression_predictor']=x;q['contextual_response_transformed']=y;q['predicted_contextual_sensitivity']=fit;q['residual_contextual_sensitivity']=resid;q['standardized_residual']=z;q['residual_rank']=rankdata(-resid,method='ordinal')
  xr=rankdata(q.expression_statistic)/len(q);yr=rankdata(q.contextual_statistic)/len(q);rfit,rresid,rz=residualize(xr,yr);q['rank_predicted_contextual_sensitivity']=rfit;q['rank_residual_contextual_sensitivity']=rresid;q['rank_standardized_residual']=rz;q['rank_residual_rank']=rankdata(-rresid,method='ordinal')
  q.to_parquet(OUT/f'{label}_matched_gene_table.parquet',index=False);q.sort_values('residual_contextual_sensitivity',ascending=False).to_csv(OUT/f'{label}_context_excess_ranking.csv',index=False)
 return a,b

def audit(q,label):
 rows=[];base=q.residual_contextual_sensitivity
 magnitude='contextual_displacement_magnitude' if label=='controlled_tcell' else 'contextual_response_magnitude'
 for metric in ['expression_statistic','expression_abundance',magnitude]:rows.append({'analysis':label,'audit':f'residual_vs_{metric}','excluded_fraction':0,'genes':len(q),'spearman':spearmanr(base,q[metric]).statistic,'top500_overlap':500})
 for exclude in [.05,.10,.20]:
  keep=(q.expression_abundance>=q.expression_abundance.quantile(exclude))&(q[magnitude]>=q[magnitude].quantile(exclude));z=q[keep].copy();fit,resid,_=residualize(z.expression_predictor.to_numpy(),z.contextual_response_transformed.to_numpy());z['new']=resid
  top0=set(q.nlargest(500,'residual_contextual_sensitivity').gene_symbol);top=set(z.nlargest(500,'new').gene_symbol)
  rows.append({'analysis':label,'audit':'refit_after_expression_and_magnitude_filter','excluded_fraction':exclude,'genes':len(z),'spearman':spearmanr(z.residual_contextual_sensitivity,z.new).statistic,'top500_overlap':len(top0&top)})
 for tail in [100,250,500,1000]:
  z=q.nlargest(tail,'residual_contextual_sensitivity');rows.append({'analysis':label,'audit':'positive_tail_low_feature_prevalence','excluded_fraction':tail,'genes':tail,'spearman':(z.expression_abundance<q.expression_abundance.quantile(.1)).mean(),'top500_overlap':(z[magnitude]<q[magnitude].quantile(.1)).mean()})
 return pd.DataFrame(rows)

def categories(q,label):
 rows=[];M=len(q)
 for frac in [.05,.10,.20]:
  n=int(M*frac);e=set(q.nlargest(n,'expression_statistic').gene_symbol);c=set(q.nlargest(n,'contextual_statistic').gene_symbol)
  for name,s in [('strong_expression_strong_context',e&c),('strong_expression_weak_context',e-c),('weak_expression_strong_context',c-e),('weak_expression_weak_context',set(q.gene_symbol)-(e|c))]:rows.append({'analysis':label,'top_fraction':frac,'category':name,'genes':len(s),'gene_symbols':';'.join(sorted(s))})
 return pd.DataFrame(rows)

def gsea(a,b):
 cache=OUT/'residual_gsea.parquet'
 if cache.exists():return pd.read_parquet(cache)
 ranks={('controlled_tcell','lowess_residual'):a[['gene_symbol','residual_contextual_sensitivity']],('controlled_tcell','rank_residual'):a[['gene_symbol','rank_residual_contextual_sensitivity']],('RR1','lowess_residual'):b[['gene_symbol','residual_contextual_sensitivity']],('RR1','rank_residual'):b[['gene_symbol','rank_residual_contextual_sensitivity']]};out=[]
 for (analysis,ranking),rnk in ranks.items():
  for source,file in GMT.items():
   print(f'[GSEA] {analysis} {ranking} {source}',flush=True);pre=gp.prerank(rnk=rnk.sort_values(rnk.columns[1],ascending=False),gene_sets=str(GMT_ROOT/file),min_size=10,max_size=500,permutation_num=N_PERM,threads=8,seed=SEED,outdir=None,verbose=False)
   z=pre.res2d.rename(columns={'Term':'pathway','ES':'es','NES':'nes','NOM p-val':'nominal_p','FDR q-val':'fdr','Lead_genes':'leading_edge'});z['analysis']=analysis;z['ranking']=ranking;z['source']=source;out.append(z[['analysis','ranking','source','pathway','es','nes','nominal_p','fdr','leading_edge']])
 ans=pd.concat(out,ignore_index=True);ans.to_parquet(cache,index=False);return ans

def add_sizes(enr,universes):
 rows=[]
 for source,file in GMT.items():
  for term,members in gp.parser.read_gmt(path=str(GMT_ROOT/file)).items():
   for a,u in universes.items():rows.append({'analysis':a,'source':source,'pathway':term,'represented_genes':len(set(members)&u)})
 return enr.merge(pd.DataFrame(rows),on=['analysis','source','pathway'],how='left')

def cross_gene(a,b):
 m=a[['gene_symbol','residual_contextual_sensitivity']].merge(b[['gene_symbol','residual_contextual_sensitivity']],on='gene_symbol',suffixes=('_tcell','_rr1'));M=len(m);rows=[]
 for n in [100,250,500,1000]:
  x=set(m.nlargest(n,'residual_contextual_sensitivity_tcell').gene_symbol);y=set(m.nlargest(n,'residual_contextual_sensitivity_rr1').gene_symbol);k=len(x&y);rows.append({'top_n':n,'overlap':k,'expected':n*n/M,'fold_enrichment':k/(n*n/M),'hypergeom_p':hypergeom.sf(k-1,M,n,n),'genes':';'.join(sorted(x&y))})
 return m,pd.DataFrame(rows),spearmanr(m.residual_contextual_sensitivity_tcell,m.residual_contextual_sensitivity_rr1).statistic

def family_results(resid):
 # Reuse original conventional/contextual tables for the 3-way comparison.
 conv=pd.read_csv(CONV/'conventional_gsea.csv');ctrl=pd.read_csv(CTRL/'controlled_pathway_enrichment.csv');ctrl['analysis']='controlled_tcell';rr=pd.read_parquet(RR/'gsea_full_results.parquet');rr=rr[rr.comparison.eq('RR1')]
 rows=[]
 for exp in ['controlled_tcell','RR1']:
  sources=[('conventional',conv[conv.analysis.eq('tcell_conventional' if exp=='controlled_tcell' else 'rr1_conventional_instability')]),('original_contextual',ctrl if exp=='controlled_tcell' else rr),('expression_adjusted',resid[(resid.analysis==exp)&(resid.ranking=='lowess_residual')])]
  for fam in ['RNA processing / splicing','Chromatin organization / remodeling','DNA repair / DNA metabolism']:
   for typ,z in sources:
    q=z[z.pathway.map(family)==fam];sig=q[q.fdr<.05];rows.append({'experiment':exp,'family':fam,'signal':typ,'significant_terms':sig.pathway.nunique(),'best_positive_nes':sig.nes.max() if len(sig) else np.nan,'best_abs_nes':sig.nes.abs().max() if len(sig) else np.nan,'best_fdr':sig.fdr.min() if len(sig) else np.nan})
 result=pd.DataFrame(rows);dec=[]
 for fam in result.family.unique():
  q=resid[(resid.pathway.map(family)==fam)&(resid.fdr<.05)&(resid.nes>0)];n=q.groupby('analysis').ranking.nunique()
  for exp in ['controlled_tcell','RR1']:
   nr=int(n.get(exp,0));dec.append({'experiment':exp,'family':fam,'classification':'STRONGLY RETAINED AFTER EXPRESSION ADJUSTMENT' if nr==2 else 'PARTIALLY RETAINED AFTER EXPRESSION ADJUSTMENT' if nr==1 else 'EXPLAINED BY CONVENTIONAL EXPRESSION'})
 return result,pd.DataFrame(dec)

def leading_edges(enr,universes):
 rows=[]
 for fam in ['RNA processing / splicing','Chromatin organization / remodeling','DNA repair / DNA metabolism']:
  sets={}
  for exp in ['controlled_tcell','RR1']:
   q=enr[(enr.analysis==exp)&(enr.ranking=='lowess_residual')&(enr.pathway.map(family)==fam)&(enr.fdr<.05)&(enr.nes>0)].sort_values('nes',ascending=False)
   if len(q):r=q.iloc[0];sets[exp]=set(str(r.leading_edge).split(';'));path=r.pathway
   else:sets[exp]=set()
  x,y=sets['controlled_tcell'],sets['RR1'];M=len(set.intersection(*universes.values()));k=len(x&y);rows.append({'family':fam,'tcell_leading_edge':len(x),'rr1_leading_edge':len(y),'overlap':k,'expected':len(x)*len(y)/M,'hypergeom_p':hypergeom.sf(k-1,M,len(x),len(y)) if x and y else np.nan,'overlap_genes':';'.join(sorted(x&y))})
 return pd.DataFrame(rows)

def permutation_control(a,b,universes):
 # Competitive family score: mean percentile residual rank among the union of
 # pathway genes in each predefined family. Independent gene-label shuffles
 # preserve each residual distribution and pathway sizes.
 genes=sorted(set(a.gene_symbol)&set(b.gene_symbol));ia=a.set_index('gene_symbol').loc[genes].residual_contextual_sensitivity;ib=b.set_index('gene_symbol').loc[genes].residual_contextual_sensitivity
 pa=pd.Series(rankdata(ia)/len(ia),index=genes);pb=pd.Series(rankdata(ib)/len(ib),index=genes);families={f:set() for f in ['RNA processing / splicing','Chromatin organization / remodeling','DNA repair / DNA metabolism']}
 for source,file in GMT.items():
  for term,members in gp.parser.read_gmt(path=str(GMT_ROOT/file)).items():
   f=family(term)
   if f in families:families[f]|=set(members)&set(genes)
 observed={f:(pa.loc[list(s)].mean()+pb.loc[list(s)].mean())/2 for f,s in families.items()};rng=np.random.default_rng(SEED);null={f:[] for f in families}
 va=pa.to_numpy();vb=pb.to_numpy();index={g:i for i,g in enumerate(genes)};family_indices={f:np.fromiter((index[g] for g in s),int) for f,s in families.items()}
 for _ in range(1000):
  xa=rng.permutation(va);xb=rng.permutation(vb)
  for f,idx in family_indices.items():null[f].append((xa[idx].mean()+xb[idx].mean())/2)
 return pd.DataFrame([{'family':f,'observed_joint_mean_percentile':observed[f],'null_mean':np.mean(null[f]),'null_sd':np.std(null[f],ddof=1),'empirical_p':(1+np.sum(np.array(null[f])>=observed[f]))/1001} for f in families])

def plots(a,b,fam,ov,perm):
 fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
 for ax,(label,q) in zip(axes,[('Controlled T cells',a),('RR1',b)]):
  top=q.nlargest(250,'residual_contextual_sensitivity');ax.scatter(q.expression_predictor,q.contextual_response_transformed,s=3,alpha=.12,color='#777777');ax.scatter(top.expression_predictor,top.contextual_response_transformed,s=8,alpha=.65,color='#CC3311');order=np.argsort(q.expression_predictor);ax.plot(q.expression_predictor.iloc[order],q.predicted_contextual_sensitivity.iloc[order],color='#0077BB',lw=2);ax.set(xlabel='log1p conventional effect statistic',ylabel='Contextual sensitivity (transformed)',title=label)
 fig.savefig(FIG/'expression_vs_contextual_sensitivity.png',dpi=300);fig.savefig(FIG/'expression_vs_contextual_sensitivity.pdf');plt.close(fig)
 q=fam.copy();q['value']=q.best_positive_nes.fillna(0);fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained');colors={'conventional':'#999999','original_contextual':'#4477AA','expression_adjusted':'#CC3311'}
 for ax,(exp,z) in zip(axes,q.groupby('experiment',sort=False)):
  x=np.arange(3);w=.25
  for j,(sig,g) in enumerate(z.groupby('signal',sort=False)):ax.bar(x+(j-1)*w,g.value,w,label=sig,color=colors[sig])
  ax.set(xticks=x,xticklabels=z.family.drop_duplicates(),ylabel='Best positive NES (FDR < 0.05)',title=exp);ax.tick_params(axis='x',rotation=12);ax.legend(fontsize=8)
 fig.savefig(FIG/'residual_pathway_family_comparison.png',dpi=300);fig.savefig(FIG/'residual_pathway_family_comparison.pdf');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained');x=np.arange(len(ov));axes[0].bar(x-.18,ov.overlap,.36,label='Observed',color='#4477AA');axes[0].bar(x+.18,ov.expected,.36,label='Expected',color='#BBBBBB');axes[0].set(xticks=x,xticklabels=[f'Top {n}' for n in ov.top_n],ylabel='Overlap',title='Cross-context residual gene overlap');axes[0].legend();axes[1].barh(perm.family,-np.log10(perm.empirical_p),color='#228833');axes[1].axvline(-np.log10(.05),ls='--',color='black');axes[1].set(xlabel='-log10 empirical p',title='Residual pathway-convergence null');fig.savefig(FIG/'cross_context_residual_concordance.png',dpi=300);fig.savefig(FIG/'cross_context_residual_concordance.pdf');plt.close(fig)

def main():
 OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(exist_ok=True);started=time.time();a,b=build_tables();pd.concat([audit(a,'controlled_tcell'),audit(b,'RR1')]).to_csv(OUT/'residual_artifact_audit.csv',index=False);pd.concat([categories(a,'controlled_tcell'),categories(b,'RR1')]).to_csv(OUT/'context_categories.csv',index=False)
 enr=add_sizes(gsea(a,b),{'controlled_tcell':set(a.gene_symbol),'RR1':set(b.gene_symbol)});enr.to_csv(OUT/'residual_gsea.csv',index=False);m,ov,rho=cross_gene(a,b);ov.to_csv(OUT/'cross_context_gene_overlap.csv',index=False)
 fam,dec=family_results(enr);fam.to_csv(OUT/'pathway_family_comparison.csv',index=False);dec.to_csv(OUT/'pathway_family_decisions.csv',index=False);le=leading_edges(enr,{'controlled_tcell':set(a.gene_symbol),'RR1':set(b.gene_symbol)});le.to_csv(OUT/'residual_leading_edge_comparison.csv',index=False);perm=permutation_control(a,b,{'controlled_tcell':set(a.gene_symbol),'RR1':set(b.gene_symbol)});perm.to_csv(OUT/'permutation_pathway_control.csv',index=False)
 retained=dec[dec.classification.ne('EXPLAINED BY CONVENTIONAL EXPRESSION')].groupby('family').experiment.nunique();shared=[f for f,n in retained.items() if n==2];classification='STRONG ADDITIONAL CONTEXTUAL ORGANIZATION' if len(shared)==3 and all(perm.set_index('family').loc[shared].empirical_p<.05) else 'PARTIAL ADDITIONAL CONTEXTUAL ORGANIZATION' if shared else 'CONTEXTUAL SIGNAL LARGELY EXPLAINED BY EXPRESSION'
 summary={'tcell_genes':len(a),'rr1_genes':len(b),'tcell_expression_context_spearman':spearmanr(a.expression_statistic,a.contextual_statistic).statistic,'rr1_expression_context_spearman':spearmanr(b.expression_statistic,b.contextual_statistic).statistic,'cross_context_residual_spearman':rho,'shared_retained_pathway_families':shared,'classification':classification,'lowess':'robust LOWESS frac=0.20, it=3 on log1p conventional statistic; log1p contextual score for T cells and normalized discrepancy for RR1','rank_sensitivity':'LOWESS on within-experiment percentile ranks','permutation_control':'1,000 independent gene-label shuffles; joint family mean percentile competitive statistic','elapsed_minutes':(time.time()-started)/60}
 pd.DataFrame([summary|{'shared_retained_pathway_families':'; '.join(shared)}]).to_csv(OUT/'final_summary.csv',index=False);(OUT/'final_summary.json').write_text(json.dumps(summary,indent=2)+'\n');plots(a,b,fam,ov,perm);print(json.dumps(summary,indent=2));print('[complete]',OUT)
if __name__=='__main__':main()
