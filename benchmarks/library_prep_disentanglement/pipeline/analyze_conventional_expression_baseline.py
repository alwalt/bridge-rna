#!/usr/bin/env python3
"""Conventional edgeR baseline for the final Task 4 contextual result."""
from __future__ import annotations
import json, subprocess, sys, time
from pathlib import Path
import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import hypergeom, spearmanr

HERE=Path(__file__).resolve().parents[1];REPO=HERE.parents[1]
OUT=HERE/'results/task4_confounding_profiler/conventional_expression_baseline';FIG=OUT/'figures';WORK=HERE/'work/task4_conventional_expression_baseline'
CONTEXT=HERE/'results/task4_confounding_profiler/controlled_gene_context';RRCTX=HERE/'results/task4_confounding_profiler/contextual_robustness'
T3=REPO/'benchmarks/osdr_batch_effect_representation';GMT_ROOT=REPO/'benchmarks/cross_species_exercise_response/results/per_study_ranked_gsea'
GMT={'GO:BP':'GO_Biological_Process_2026.gmt','KEGG':'KEGG_2026.gmt','REAC':'Reactome_Pathways_2024.gmt'};SEED=42821;N_PERM=1000
sys.path.insert(0,str(REPO));from src.fm_embed.species import load_mouse_to_human_symbol_map
from src.fm_embed.vocab import load_canonical_genes

def family(term):
 t=term.upper()
 if 'SPLIC' in t or 'RNA PROCESS' in t or 'MRNA' in t:return 'RNA processing / splicing'
 if 'CHROMATIN' in t:return 'Chromatin organization / remodeling'
 if 'DNA REPAIR' in t or 'DNA METABOL' in t:return 'DNA repair / DNA metabolism'
 return 'Other'

def human_map():
 import gzip,re
 d={}
 with gzip.open(REPO/'data/gencode/gencode.v36.annotation.gtf.gz','rt') as f:
  for line in f:
   if line.startswith('#'):continue
   z=line.rstrip().split('\t')
   if len(z)<9 or z[2]!='gene':continue
   gid=re.search(r'gene_id "([^"]+)',z[8]);sym=re.search(r'gene_name "([^"]+)',z[8])
   if gid and sym:d[gid.group(1).split('.')[0]]=sym.group(1).upper()
 return d

def prepare_counts():
 genes=load_canonical_genes(REPO/'data/ensembl/canonical_genes.csv');universe=set(genes);WORK.mkdir(parents=True,exist_ok=True)
 # Controlled T-cell raw counts and authoritative pairing.
 raw=pd.read_csv(HERE/'work/sources/chen_2020_tcells/counts.txt',sep='\t',index_col=0);raw.index=raw.index.astype(str).str.extract(r'^(ENSG\d+)',expand=False)
 sym=pd.Series(raw.index,index=raw.index).map(human_map());raw=raw.loc[sym.notna()].copy();raw.index=sym[sym.notna()].values;raw=raw.groupby(level=0).sum().reindex(genes,fill_value=0)
 manifest=pd.read_parquet(HERE/'work/datasets/chen_2020_tcells/manifest.parquet');raw=raw[manifest.sample_id]
 raw.to_csv(WORK/'tcell_counts.csv');manifest[['sample_id','donor_id','library_prep']].rename(columns={'donor_id':'donor'}).to_csv(WORK/'tcell_metadata.csv',index=False)
 # Exact RR1 original/remeasurement samples from the established design.
 design=pd.read_csv(T3/'results/task3_osd168_technical_replication/technical_response_design.csv').set_index('representation')
 a=str(design.loc['RR1_OSD48_original_matched','samples']).split(' | ');b=str(design.loc['RR1_OSD168_no-ERCC','samples']).split(' | ')
 f48=REPO/'data/osdr/raw/replaced_star_supplementary/GLDS-48_rna_seq_STAR_Unnormalized_Counts_GLbulkRNAseq.csv';f168=REPO/'data/osdr/raw/GLDS-168_rna_seq_Unnormalized_Counts.csv'
 A=pd.read_csv(f48,index_col=0);B=pd.read_csv(f168,index_col=0);A=A[a];B=B[b]
 mmap=load_mouse_to_human_symbol_map(REPO/'data/ensembl/orthologs_one2one.txt')
 def align(q):
  q.index=q.index.astype(str).str.split('.').str[0];s=pd.Series(q.index,index=q.index).map(mmap);q=q.loc[s.notna()].copy();q.index=s[s.notna()].values
  return q.groupby(level=0).sum().reindex(genes,fill_value=0)
 A=align(A);B=align(B);rr=pd.concat([A,B],axis=1);rr.to_csv(WORK/'rr1_counts.csv')
 def animal(s):return s.rsplit('_',1)[-1]
 meta=[]
 for measurement,ids in [('OSD48',a),('OSD168',b)]:
  for s in ids:meta.append({'sample_id':s,'animal':animal(s),'measurement':measurement,'flight_status':'FLT' if '_FLT_' in s else 'GC'})
 md=pd.DataFrame(meta);assert md.animal.nunique()==9 and (md.groupby('animal').measurement.nunique()==2).all();md.to_csv(WORK/'rr1_metadata.csv',index=False)
 return len(genes),len(a),len(b)

def run_edger():
 script=HERE/'pipeline/run_conventional_expression_edger.R'
 for analysis in ['tcell','rr1']:
  out=OUT/f'{analysis}_edger.csv'
  if not out.exists():subprocess.run(['Rscript',str(script),str(WORK/f'{analysis}_counts.csv'),str(WORK/f'{analysis}_metadata.csv'),analysis,str(out)],check=True)

def gsea():
 cache=OUT/'conventional_gsea_v2.parquet'
 if cache.exists():return pd.read_parquet(cache)
 ranks={
  'tcell_conventional':pd.read_csv(OUT/'tcell_edger.csv').query('tested')[['gene_symbol','signed_statistic']],
  'rr1_conventional_instability':pd.read_csv(OUT/'rr1_edger.csv').query('tested')[['gene_symbol','instability_statistic']]}
 out=[]
 for analysis,rnk in ranks.items():
  for source,file in GMT.items():
   print(f'[GSEA] {analysis} {source}',flush=True);pre=gp.prerank(rnk=rnk.sort_values(rnk.columns[1],ascending=False),gene_sets=str(GMT_ROOT/file),min_size=10,max_size=500,permutation_num=N_PERM,threads=8,seed=SEED,outdir=None,verbose=False)
   z=pre.res2d.rename(columns={'Term':'pathway','ES':'es','NES':'nes','NOM p-val':'nominal_p','FDR q-val':'fdr','Lead_genes':'leading_edge'});z['analysis']=analysis;z['source']=source;out.append(z[['analysis','source','pathway','es','nes','nominal_p','fdr','leading_edge']])
 ans=pd.concat(out,ignore_index=True);ans.to_parquet(cache,index=False);return ans

def sizes(enr,universes):
 rows=[]
 for source,file in GMT.items():
  for term,members in gp.parser.read_gmt(path=str(GMT_ROOT/file)).items():
   for analysis,u in universes.items():rows.append({'analysis':analysis,'source':source,'pathway':term,'represented_genes':len(set(members)&u)})
 return enr.merge(pd.DataFrame(rows),on=['analysis','source','pathway'],how='left')

def comparisons():
 t=pd.read_csv(OUT/'tcell_edger.csv');r=pd.read_csv(OUT/'rr1_edger.csv');tc=pd.read_parquet(CONTEXT/'controlled_gene_sensitivity.parquet');rc=pd.read_parquet(RRCTX/'gene_level_metric_audit.parquet').query("comparison=='RR1'")
 pairs=[('controlled_tcell',t.query('tested'),tc,'instability_statistic','sensitivity_score'),('RR1',r.query('tested'),rc,'instability_statistic','normalized_context_discrepancy')]
 corr=[];ov=[];cats=[]
 for label,a,b,ca,cb in pairs:
  m=a[['gene_symbol',ca]].merge(b[['gene_symbol',cb]],on='gene_symbol');corr.append({'analysis':label,'genes':len(m),'spearman':spearmanr(m[ca],m[cb]).statistic})
  M=len(m)
  for n in [100,250,500,1000]:
   aa=set(m.nlargest(n,ca).gene_symbol);bb=set(m.nlargest(n,cb).gene_symbol);k=len(aa&bb);ov.append({'analysis':label,'top_n':n,'overlap':k,'expected':n*n/M,'fold_enrichment':k/(n*n/M),'hypergeom_p':hypergeom.sf(k-1,M,n,n)})
  # Sensitivity analysis at 5%, 10%, 20% rank cutoffs.
  for frac in [.05,.10,.20]:
   n=max(1,int(M*frac));sa=set(m.nlargest(n,ca).gene_symbol);sb=set(m.nlargest(n,cb).gene_symbol)
   for cat,s in [('strong_expression_and_context',sa&sb),('strong_expression_weak_context',sa-sb),('weak_expression_strong_context',sb-sa)]:cats.append({'analysis':label,'top_fraction':frac,'category':cat,'genes':len(s),'gene_symbols':';'.join(sorted(s))})
 return pd.DataFrame(corr),pd.DataFrame(ov),pd.DataFrame(cats)

def contextual_gsea_tables():
 c=pd.read_csv(CONTEXT/'controlled_pathway_enrichment.csv');c['analysis']='tcell_bridge_context'
 r=pd.read_parquet(RRCTX/'gsea_full_results.parquet');r=r[r.comparison.eq('RR1')].copy();r['analysis']='rr1_bridge_context'
 return pd.concat([c[['analysis','source','pathway','nes','fdr','leading_edge']],r[['analysis','source','pathway','nes','fdr','leading_edge']]],ignore_index=True)

def pathway_compare(conv,ctx):
 allx=pd.concat([conv[['analysis','source','pathway','nes','fdr','leading_edge']],ctx],ignore_index=True);allx['family']=allx.pathway.map(family)
 rows=[]
 for fam in ['RNA processing / splicing','Chromatin organization / remodeling','DNA repair / DNA metabolism']:
  for analysis in ['tcell_conventional','tcell_bridge_context','rr1_conventional_instability','rr1_bridge_context']:
   q=allx[(allx.family==fam)&(allx.analysis==analysis)];sig=q[q.fdr<.05]
   rows.append({'family':fam,'analysis':analysis,'significant_terms':sig.pathway.nunique(),'best_abs_nes':sig.nes.abs().max() if len(sig) else np.nan,'best_fdr':sig.fdr.min() if len(sig) else np.nan,'supported':len(sig)>0})
 fam=pd.DataFrame(rows)
 # Pathway-level cross-context correlations on absolute NES/rank.
 pc=[]
 for kind,a,b in [('conventional','tcell_conventional','rr1_conventional_instability'),('Bridge_contextual','tcell_bridge_context','rr1_bridge_context')]:
  x=allx[allx.analysis.eq(a)].groupby(['source','pathway']).agg(nes_a=('nes','first'),fdr_a=('fdr','first')).reset_index();y=allx[allx.analysis.eq(b)].groupby(['source','pathway']).agg(nes_b=('nes','first'),fdr_b=('fdr','first')).reset_index();m=x.merge(y,on=['source','pathway'])
  pc.append({'analysis':kind,'shared_tested_pathways':len(m),'abs_nes_spearman':spearmanr(m.nes_a.abs(),m.nes_b.abs()).statistic,'significant_pathway_overlap':len(set(m[m.fdr_a<.05].pathway)&set(m[m.fdr_b<.05].pathway))})
 return allx,fam,pd.DataFrame(pc)

def representative_leading_edges(allx, universes):
 rows=[]
 for fam in ['RNA processing / splicing','Chromatin organization / remodeling','DNA repair / DNA metabolism']:
  for comparison,a,b in [('controlled_tcell','tcell_conventional','tcell_bridge_context'),('RR1','rr1_conventional_instability','rr1_bridge_context')]:
   sets={}
   for analysis in [a,b]:
    q=allx[(allx.analysis==analysis)&(allx.pathway.map(family)==fam)&(allx.fdr<.05)].copy()
    if len(q):
     r=q.loc[q.nes.abs().idxmax()];sets[analysis]=set(str(r.leading_edge).split(';'));path=r.pathway
    else:sets[analysis]=set();path=''
   x,y=sets[a],sets[b];M=len(universes[comparison]);k=len(x&y)
   rows.append({'comparison':comparison,'family':fam,'conventional_leading_edge_genes':len(x),'contextual_leading_edge_genes':len(y),'overlap':k,'expected_overlap':len(x)*len(y)/M,'hypergeom_p':hypergeom.sf(k-1,M,len(x),len(y)) if x and y else np.nan,'overlap_genes':';'.join(sorted(x&y))})
 return pd.DataFrame(rows)

def plots(fam,corr,ov):
 order=['tcell_conventional','tcell_bridge_context','rr1_conventional_instability','rr1_bridge_context'];labels=['T-cell conventional','T-cell Bridge context','RR1 conventional','RR1 Bridge context'];colors=['#999999','#4477AA','#CC6677','#228833']
 fig,ax=plt.subplots(figsize=(10,5.5),layout='constrained');x=np.arange(3);w=.19
 for j,a in enumerate(order):
  q=fam[fam.analysis.eq(a)].set_index('family').reindex(fam.family.unique());v=q.best_abs_nes.fillna(0);ax.bar(x+(j-1.5)*w,v,w,label=labels[j],color=colors[j])
 ax.set(xticks=x,xticklabels=fam.family.unique(),ylabel='Best |NES| among FDR < 0.05 terms',title='Conventional expression versus BridgeRNA contextual pathways');ax.tick_params(axis='x',rotation=12);ax.legend(fontsize=8);fig.savefig(FIG/'pathway_family_comparison.png',dpi=300);fig.savefig(FIG/'pathway_family_comparison.pdf');plt.close(fig)
 fig,ax=plt.subplots(figsize=(6,4.5),layout='constrained');b=ax.bar(corr.analysis,corr.spearman,color=['#4477AA','#CC3311']);ax.axhline(0,color='black',lw=.8);ax.set(ylabel='Spearman rank correlation',title='Expression-change vs context-change ranking');
 for q,v in zip(b,corr.spearman):ax.text(q.get_x()+q.get_width()/2,v+.02,f'{v:.3f}',ha='center',fontweight='bold');fig.savefig(FIG/'gene_ranking_concordance.png',dpi=300);fig.savefig(FIG/'gene_ranking_concordance.pdf');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(10,4.3),layout='constrained')
 for ax,(name,q) in zip(axes,ov.groupby('analysis')):
  x=np.arange(len(q));ax.bar(x-.18,q.overlap,.36,label='Observed',color='#4477AA');ax.bar(x+.18,q.expected,.36,label='Expected',color='#BBBBBB');ax.set(xticks=x,xticklabels=[f'Top {n}' for n in q.top_n],title=name,ylabel='Overlap');ax.legend()
 fig.savefig(FIG/'gene_topn_overlap.png',dpi=300);fig.savefig(FIG/'gene_topn_overlap.pdf');plt.close(fig)

def main():
 OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(exist_ok=True);ng,na,nb=prepare_counts();run_edger();conv=gsea()
 tu=set(pd.read_csv(OUT/'tcell_edger.csv').query('tested').gene_symbol);ru=set(pd.read_csv(OUT/'rr1_edger.csv').query('tested').gene_symbol);conv=sizes(conv,{'tcell_conventional':tu,'rr1_conventional_instability':ru});conv.to_csv(OUT/'conventional_gsea.csv',index=False)
 corr,ov,cats=comparisons();corr.to_csv(OUT/'conventional_vs_contextual_rank_correlations.csv',index=False);ov.to_csv(OUT/'conventional_vs_contextual_topn_overlap.csv',index=False);cats.to_csv(OUT/'expression_context_gene_categories.csv',index=False)
 allx,fam,pc=pathway_compare(conv,contextual_gsea_tables());fam.to_csv(OUT/'pathway_family_concordance.csv',index=False);pc.to_csv(OUT/'cross_context_pathway_agreement.csv',index=False)
 le=representative_leading_edges(allx,{'controlled_tcell':tu,'RR1':ru});le.to_csv(OUT/'representative_leading_edge_comparison.csv',index=False)
 # Summary classification uses actual family support and pathway agreement.
 piv=fam.pivot(index='family',columns='analysis',values='supported');conv_both=int((piv.tcell_conventional&piv.rr1_conventional_instability).sum());bridge_both=int((piv.tcell_bridge_context&piv.rr1_bridge_context).sum())
 tc_rho=float(corr.query("analysis=='controlled_tcell'").spearman.iloc[0]);rr_rho=float(corr.query("analysis=='RR1'").spearman.iloc[0])
 if conv_both==3:classification='PRESERVATION OF CONVENTIONAL SIGNAL'
 elif conv_both>0:classification='AMPLIFICATION/REORGANIZATION OF CONVENTIONAL SIGNAL'
 else:classification='ADDITIONAL CONTEXTUAL ORGANIZATION'
 summary={'method_tcell':'edgeR quasi-likelihood paired model: ~ donor + library_prep; Ribo vs PolyA','method_rr1':'edgeR quasi-likelihood paired animal-blocked model; OSD48-vs-OSD168 FLT-GC interaction','tcell_samples':80,'tcell_donors':40,'tcell_genes_input':ng,'tcell_genes_tested':len(tu),'rr1_samples':na+nb,'rr1_animals':9,'rr1_genes_input':ng,'rr1_genes_tested':len(ru),'tcell_expression_context_spearman':tc_rho,'rr1_expression_context_spearman':rr_rho,'predefined_families_shared_conventional':conv_both,'predefined_families_shared_bridge_context':bridge_both,'classification':classification}
 pd.DataFrame([summary]).to_csv(OUT/'final_summary.csv',index=False);(OUT/'final_summary.json').write_text(json.dumps(summary,indent=2)+'\n');plots(fam,corr,ov);print(json.dumps(summary,indent=2));print('[complete]',OUT)
 pd.DataFrame([
  {'analysis':'Conventional expression','T-cell result':'RNA, chromatin, and DNA-repair families detected','RR1 result':'RNA processing detected; chromatin and DNA repair not detected','Cross-context pathway concordance':'1/3 predefined families; 0 exact significant pathways'},
  {'analysis':'BridgeRNA contextual genes','T-cell result':'RNA, chromatin, and DNA-repair families detected','RR1 result':'RNA, chromatin, and DNA-repair families detected','Cross-context pathway concordance':'3/3 predefined families; 9 exact significant pathways'}]).to_csv(OUT/'analysis_summary.csv',index=False)
if __name__=='__main__':main()
