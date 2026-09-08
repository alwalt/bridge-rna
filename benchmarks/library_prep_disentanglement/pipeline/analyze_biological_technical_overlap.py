#!/usr/bin/env python3
"""Relate biological FLT-GC strength to a controlled PolyA/Ribo reference."""
from __future__ import annotations
import json
import re
from datetime import datetime, timezone
from pathlib import Path
import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE=Path(__file__).resolve().parents[1]; REPO=HERE.parents[1]
BASE=HERE/'results/task4_confounding_profiler'; PREV=BASE/'independent_biological_replication'
OUT=BASE/'biological_technical_overlap'; FIG=OUT/'figures'
T3=REPO/'benchmarks/osdr_batch_effect_representation/results'
CONTROL=HERE/'work/datasets/chen_2020_tcells'; GMT_ROOT=REPO/'benchmarks/cross_species_exercise_response/results/per_study_ranked_gsea'
GMT={'GO:BP':'GO_Biological_Process_2026.gmt','KEGG':'KEGG_2026.gmt','REAC':'Reactome_Pathways_2024.gmt'}
KS=(1,2,5); N_RANDOM=1000; SEED=43111

def family(term):
 t=term.upper()
 if 'SPLIC' in t or 'RNA PROCESS' in t or 'MRNA PROCESS' in t:return 'RNA processing / splicing'
 if 'CHROMATIN' in t or 'NUCLEOSOME' in t or 'HISTONE' in t:return 'Chromatin organization / remodeling'
 if 'DNA REPAIR' in t or 'DNA METABOL' in t or 'DNA DAMAGE' in t:return 'DNA repair / DNA metabolism'
 if any(x in t for x in ['LIPID','FATTY ACID','PEROXISOM','CHOLESTEROL','BILE','SMALL MOLECULE','CATABOL','METABOL']):return 'Hepatic lipid / metabolic'
 return 'Other'

def basis():
 m=pd.read_parquet(CONTROL/'manifest.parquet').reset_index(drop=True);z=np.load(CONTROL/'bridgerna_embeddings.npy').astype(float);D=[]
 for _,g in m.groupby('pair_id',sort=True):
  D.append(z[g.index[g.library_prep.eq('ribo')]].mean(0)-z[g.index[g.library_prep.eq('polyA')]].mean(0))
 D=np.stack(D);_,s,vt=np.linalg.svd(D,full_matrices=False)
 return vt,s*s/(s*s).sum()

def program_genes(universe):
 sets={f:set() for f in ['RNA processing / splicing','Chromatin organization / remodeling','DNA repair / DNA metabolism','Hepatic lipid / metabolic']}
 for _,file in GMT.items():
  for term,genes in gp.parser.read_gmt(path=str(GMT_ROOT/file)).items():
   f=family(term)
   if f in sets:sets[f]|=set(genes)&universe
 return sets

def response_table(vt):
 summary=pd.read_csv(PREV/'independent_contrast_summary.csv');edge=pd.read_csv(PREV/'edger_results.csv.gz');x=np.load(T3/'task3b_bridgerna_response_vectors.npz',allow_pickle=True);vectors=dict(zip(x['contrast_id'],x['delta_z'].astype(float)))
 universe=set(edge.gene_symbol);sets=program_genes(universe);rows=[]
 for r in summary.itertuples(index=False):
  dz=vectors[r.contrast_id];M=np.linalg.norm(dz);e=edge[(edge.contrast_id.eq(r.contrast_id))&edge.tested].dropna(subset=['logFC']);lfc=e.logFC.to_numpy(float)
  duration=re.search(r'([0-9.]+)',str(r.flight_duration));row={'contrast_id':r.contrast_id,'OSD':r.OSD,'mission':r.mission,'flight_duration':r.flight_duration,'flight_duration_days':float(duration.group(1)) if duration else np.nan,'n_FLT':r.n_FLT,'n_GC':r.n_GC,'underpowered':min(r.n_FLT,r.n_GC)<2,'bridge_total_magnitude':M,'rms_log2FC':np.sqrt(np.mean(lfc**2)),'median_abs_log2FC':np.median(abs(lfc)),'mean_abs_log2FC':np.mean(abs(lfc)),'genes_tested':len(e),'significant_DE':e.FDR.lt(.05).sum(),'significant_DE_fraction':e.FDR.lt(.05).mean(),'significant_DE_abs_logFC_ge_1':(e.FDR.lt(.05)&e.logFC.abs().ge(1)).sum(),'significant_DE_abs_logFC_ge_1_fraction':(e.FDR.lt(.05)&e.logFC.abs().ge(1)).mean()}
  for k in KS:
   p=(dz@vt[:k].T)@vt[:k];o=dz-p;row.update({f'aligned_magnitude_PC1_{k}':np.linalg.norm(p),f'orthogonal_magnitude_PC1_{k}':np.linalg.norm(o),f'aligned_fraction_PC1_{k}':np.dot(p,p)/np.dot(dz,dz),f'pythagorean_relative_error_PC1_{k}':abs(M*M-np.dot(p,p)-np.dot(o,o))/max(M*M,1e-12)})
  for fam,genes in sets.items():
   q=e[e.gene_symbol.isin(genes)].logFC.to_numpy(float);slug=fam.split(' / ')[0].replace(' ','_').lower();row[f'{slug}_rms_log2FC']=np.sqrt(np.mean(q*q));row[f'{slug}_genes']=len(q)
  rows.append(row)
 return pd.DataFrame(rows),vectors,sets

def correlations(tab,vt,vectors):
 pairs=[('rms_log2FC','bridge_total_magnitude')]
 for k in KS:
  pairs += [('rms_log2FC',f'aligned_magnitude_PC1_{k}'),('rms_log2FC',f'aligned_fraction_PC1_{k}'),('rms_log2FC',f'orthogonal_magnitude_PC1_{k}')]
 rows=[]
 for subset,q in [('all_11',tab),('adequately_replicated',tab[~tab.underpowered])]:
  for a,b in pairs:
   z=spearmanr(q[a],q[b]);rows.append({'subset':subset,'variable_x':a,'variable_y':b,'spearman_rho':z.statistic,'p_value':z.pvalue,'n_contrasts':len(q)})
 corr=pd.DataFrame(rows)
 duration=tab.dropna(subset=['flight_duration_days'])
 for target in ['rms_log2FC','aligned_magnitude_PC1_2','aligned_fraction_PC1_2']:
  z=spearmanr(duration.flight_duration_days,duration[target]);corr=pd.concat([corr,pd.DataFrame([{'subset':'duration_reported_only','variable_x':'flight_duration_days','variable_y':target,'spearman_rho':z.statistic,'p_value':z.pvalue,'n_contrasts':len(duration)}])],ignore_index=True)
 # Matched random 2D subspaces; compare absolute Spearman associations.
 rng=np.random.default_rng(SEED);D=np.stack([vectors[c] for c in tab.contrast_id]);strength=tab.rms_log2FC.to_numpy();null=[]
 for rep in range(N_RANDOM):
  Q,_=np.linalg.qr(rng.normal(size=(512,2)));P=D@Q;A=np.linalg.norm(P,axis=1);F=np.sum(P*P,axis=1)/np.sum(D*D,axis=1)
  null.append({'replicate':rep,'rho_aligned_magnitude':spearmanr(strength,A).statistic,'rho_aligned_fraction':spearmanr(strength,F).statistic})
 null=pd.DataFrame(null)
 for metric,col in [('aligned_magnitude_PC1_2','rho_aligned_magnitude'),('aligned_fraction_PC1_2','rho_aligned_fraction')]:
  obs=spearmanr(strength,tab[metric]).statistic;corr.loc[(corr.subset.eq('all_11'))&(corr.variable_x.eq('rms_log2FC'))&(corr.variable_y.eq(metric)),'random_subspace_empirical_p']=(1+(null[col].abs()>=abs(obs)).sum())/(N_RANDOM+1)
 return corr,null

def program_correlations(tab):
 rows=[]
 for col in [c for c in tab if c.endswith('_rms_log2FC') and c!='rms_log2FC']:
  for subset,q in [('all_11',tab),('adequately_replicated',tab[~tab.underpowered])]:
   for target in ['aligned_magnitude_PC1_2','aligned_fraction_PC1_2']:
    z=spearmanr(q[col],q[target]);rows.append({'program_metric':col,'subset':subset,'target':target,'spearman_rho':z.statistic,'p_value':z.pvalue,'n_contrasts':len(q)})
 return pd.DataFrame(rows)

def figures(tab,corr,null,pcorr):
 plt.style.use('seaborn-v0_8-whitegrid');labels=tab.OSD+'\n'+tab.mission.astype(str)
 for y,title,name in [('aligned_magnitude_PC1_2','Aligned magnitude','de_vs_aligned_magnitude'),('aligned_fraction_PC1_2','Aligned fraction','de_vs_aligned_fraction')]:
  fig,ax=plt.subplots(figsize=(8,6),layout='constrained');ax.scatter(tab.rms_log2FC,tab[y],s=75,c=np.where(tab.underpowered,'#CC3311','#4477AA'))
  for x0,y0,l in zip(tab.rms_log2FC,tab[y],labels):ax.annotate(l,(x0,y0),xytext=(4,4),textcoords='offset points',fontsize=7)
  rho=spearmanr(tab.rms_log2FC,tab[y]).statistic;ax.set(xlabel='Conventional response strength (RMS log2FC)',ylabel=title,title=f'DE strength vs PC1–2 {title.lower()} (ρ={rho:.3f})');fig.savefig(FIG/f'{name}.png',dpi=350);fig.savefig(FIG/f'{name}.pdf');plt.close(fig)
 fig,ax=plt.subplots(figsize=(11,6),layout='constrained');order=np.argsort(tab.rms_log2FC);x=np.arange(len(tab));q=tab.iloc[order];aligned=q.aligned_magnitude_PC1_2**2;orth=q.orthogonal_magnitude_PC1_2**2
 ax.bar(x,aligned,label='PC1–2 aligned squared magnitude',color='#CC6677');ax.bar(x,orth,bottom=aligned,label='Orthogonal squared magnitude',color='#4477AA');ax.set(xticks=x,xticklabels=(q.OSD+'\n'+q.mission),ylabel='Squared Bridge response magnitude',title='Orthogonal decomposition of each FLT−GC response');ax.legend(fontsize=9);fig.savefig(FIG/'per_contrast_decomposition.png',dpi=350);fig.savefig(FIG/'per_contrast_decomposition.pdf');plt.close(fig)
 z=pcorr[pcorr.subset.eq('all_11')].pivot(index='program_metric',columns='target',values='spearman_rho').reindex(columns=['aligned_magnitude_PC1_2','aligned_fraction_PC1_2']);fig,ax=plt.subplots(figsize=(8,4.8),layout='constrained');im=ax.imshow(z,vmin=-1,vmax=1,cmap='coolwarm',aspect='auto');ax.set(xticks=range(len(z.columns)),xticklabels=['Aligned magnitude','Aligned fraction'],yticks=range(len(z)),yticklabels=[x.replace('_rms_log2FC','').replace('_',' ') for x in z.index],xlabel='',ylabel='',title='Program-specific DE strength vs technical-reference alignment');
 for i in range(len(z)):
  for j in range(len(z.columns)):ax.text(j,i,f'{z.iloc[i,j]:.2f}',ha='center',va='center',color='white' if abs(z.iloc[i,j])>.55 else 'black')
 fig.colorbar(im,ax=ax,label='Spearman ρ');fig.savefig(FIG/'program_specific_associations.png',dpi=350);fig.savefig(FIG/'program_specific_associations.pdf');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
 for ax,col,metric,title in [(axes[0],'rho_aligned_magnitude','aligned_magnitude_PC1_2','Aligned magnitude'),(axes[1],'rho_aligned_fraction','aligned_fraction_PC1_2','Aligned fraction')]:
  obs=spearmanr(tab.rms_log2FC,tab[metric]).statistic;ax.hist(null[col],bins=35,color='#BBBBBB',edgecolor='white');ax.axvline(obs,color='#CC3311',lw=3,label=f'Observed ρ={obs:.3f}');ax.set(xlabel='Random-subspace Spearman ρ',ylabel='Random 2D subspaces',title=title);ax.legend(fontsize=9)
 fig.suptitle('Controlled PC1–2 association versus matched random subspaces');fig.savefig(FIG/'random_subspace_null.png',dpi=350);fig.savefig(FIG/'random_subspace_null.pdf');plt.close(fig)

def main():
 OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(exist_ok=True);vt,evr=basis();tab,vectors,sets=response_table(vt);corr,null=correlations(tab,vt,vectors);pcorr=program_correlations(tab)
 tab.to_csv(OUT/'per_contrast_overlap_metrics.csv',index=False);corr.to_csv(OUT/'correlation_summary.csv',index=False);pcorr.to_csv(OUT/'program_specific_correlations.csv',index=False);null.to_parquet(OUT/'random_subspace_null.parquet',index=False)
 pd.DataFrame([{'family':k,'genes_in_union':len(v)} for k,v in sets.items()]).to_csv(OUT/'program_gene_universes.csv',index=False);figures(tab,corr,null,pcorr)
 primary=corr[(corr.subset.eq('all_11'))&(corr.variable_x.eq('rms_log2FC'))].set_index('variable_y');A=primary.loc['aligned_magnitude_PC1_2'];F=primary.loc['aligned_fraction_PC1_2'];reduced=corr[(corr.subset.eq('adequately_replicated'))&(corr.variable_x.eq('rms_log2FC'))].set_index('variable_y')
 if A.p_value<.05 and F.p_value<.05: outcome='OUTCOME 2, qualified by random-subspace control'
 elif A.p_value<.05: outcome='OUTCOME 1 with OUTCOME 5 qualification'
 elif (pcorr.query("subset=='all_11'").p_value<.05).any(): outcome='OUTCOME 4'
 else: outcome='OUTCOME 3'
 result={'outcome':outcome,'primary_n':len(tab),'adequately_replicated_n':int((~tab.underpowered).sum()),'excluded_underpowered':tab.loc[tab.underpowered,'contrast_id'].tolist(),'PC1_variance_fraction':float(evr[0]),'PC1_2_cumulative_variance_fraction':float(evr[:2].sum()),'rms_vs_aligned_magnitude_rho':float(A.spearman_rho),'rms_vs_aligned_magnitude_p':float(A.p_value),'rms_vs_aligned_magnitude_random_p':float(A.random_subspace_empirical_p),'rms_vs_aligned_fraction_rho':float(F.spearman_rho),'rms_vs_aligned_fraction_p':float(F.p_value),'rms_vs_aligned_fraction_random_p':float(F.random_subspace_empirical_p),'adequate_rms_vs_aligned_magnitude_rho':float(reduced.loc['aligned_magnitude_PC1_2'].spearman_rho),'adequate_rms_vs_aligned_fraction_rho':float(reduced.loc['aligned_fraction_PC1_2'].spearman_rho),'max_pythagorean_relative_error':float(tab[[c for c in tab if c.startswith('pythagorean')]].to_numpy().max()),'interpretation':'Exploratory association only. Projection into the controlled PolyA/Ribo-associated reference does not establish technical causation or quantify removable batch.'}
 (OUT/'summary.json').write_text(json.dumps(result,indent=2));(OUT/'provenance.json').write_text(json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(),'embeddings_recomputed':False,'basis':'uncentered SVD of 40 cached same-RNA T-cell Ribo-minus-PolyA displacements','primary_basis':'PC1-2','sensitivity_bases':['PC1','PC1-5'],'random_subspaces':N_RANDOM,'random_seed':SEED,'DE_primary':'RMS edgeR log2FC among filterByExpr-tested genes','large_effect_threshold':'FDR < 0.05 and abs(log2FC) >= 1'},indent=2))
 print(pd.DataFrame([result]).to_string(index=False));print('[complete]',OUT)

if __name__=='__main__':main()
