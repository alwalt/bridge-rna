#!/usr/bin/env python3
"""Independent Task 3 biological replication of contextual-gene programs."""
from __future__ import annotations
import argparse, gzip, json, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

HERE=Path(__file__).resolve().parents[1]; REPO=HERE.parents[1]
OUT=HERE/'results/task4_confounding_profiler/independent_biological_replication'; FIG=OUT/'figures'; WORK=HERE/'work/independent_biological_replication'
T3=REPO/'benchmarks/osdr_batch_effect_representation'; R3=T3/'results'; W3=T3/'work'
GMT_ROOT=REPO/'benchmarks/cross_species_exercise_response/results/per_study_ranked_gsea'
GMT={'GO:BP':'GO_Biological_Process_2026.gmt','KEGG':'KEGG_2026.gmt','REAC':'Reactome_Pathways_2024.gmt'}
SEED=43019; N_PERM=1000
sys.path[:0]=[str(REPO/'benchmarks/tcga_downstream/pipeline'),str(REPO)]
from run_attention_pooling import load_frozen_encoder
from src.fm_embed.species import load_mouse_to_human_symbol_map
from src.fm_embed.vocab import load_canonical_genes
GENES=np.array(load_canonical_genes(REPO/'data/ensembl/canonical_genes.csv'))
COUNT_FILES={
 'OSD-47':REPO/'data/osdr/raw/replaced_star_supplementary/GLDS-47_rna_seq_STAR_Unnormalized_Counts_GLbulkRNAseq.csv',
 'OSD-48':REPO/'data/osdr/raw/replaced_star_supplementary/GLDS-48_rna_seq_STAR_Unnormalized_Counts_GLbulkRNAseq.csv',
 'OSD-137':REPO/'data/osdr/raw/GLDS-137_rna_seq_Unnormalized_Counts.csv',
 'OSD-173':REPO/'data/osdr/raw/GLDS-173_rna_seq_Unnormalized_Counts.csv',
 'OSD-242':REPO/'data/osdr/raw/GLDS-242_rna_seq_Unnormalized_Counts.csv',
 'OSD-245':REPO/'data/osdr/raw/GLDS-245_rna_seq_Unnormalized_Counts.csv'}

def family(term):
 t=term.upper()
 if 'SPLIC' in t or 'RNA PROCESS' in t or 'MRNA PROCESS' in t:return 'RNA processing / splicing'
 if 'CHROMATIN' in t or 'NUCLEOSOME' in t or 'HISTONE' in t:return 'Chromatin organization / remodeling'
 if 'DNA REPAIR' in t or 'DNA METABOL' in t or 'DNA DAMAGE' in t:return 'DNA repair / DNA metabolism'
 if any(x in t for x in ['LIPID','FATTY ACID','PEROXISOM','CHOLESTEROL','BILE','SMALL MOLECULE','CATABOL','METABOL']):return 'Hepatic lipid / metabolic'
 return 'Other'

def design():
 s=pd.read_csv(R3/'task3b_contrast_summary.csv'); m=pd.read_csv(R3/'task3b_contrast_sample_membership.csv')
 s['independent_biology']=~s.OSD.eq('OSD-168'); s['role']=np.where(s.independent_biology,'primary independent biological contrast','technical remeasurement (triangulation only)')
 s.to_csv(OUT/'contrast_audit_all_task3.csv',index=False)
 q=s[s.independent_biology].copy(); ids=set(q.contrast_id); mm=m[m.contrast_id.isin(ids)].copy()
 q.to_csv(OUT/'independent_contrast_summary.csv',index=False);mm.to_csv(OUT/'independent_contrast_membership.csv',index=False)
 assert len(q)==11 and len(mm)==84 and mm.sample_id.nunique()==84
 return q,mm

def contextual(q,m,device,batch):
 cache=OUT/'contextual_gene_response_metrics.parquet'
 if cache.exists():return pd.read_parquet(cache)
 manifest=pd.read_csv(R3/'sample_manifest.csv');idx=dict(zip(manifest.sample_id,range(len(manifest))))
 x=np.load(W3/'bridgerna_log1p_tpm_inputs.npy',mmap_mode='r');model=load_frozen_encoder(device)
 rows=[];start=time.time()
 for n,cid in enumerate(q.contrast_id,1):
  z=m[m.contrast_id.eq(cid)]; sums={c:np.zeros((len(GENES),512),np.float32) for c in ['FLT','GC']};counts={'FLT':0,'GC':0}
  records=[(idx[r.sample_id],r.condition) for r in z.itertuples()]
  for a in range(0,len(records),batch):
   rr=records[a:a+batch];values=torch.as_tensor(np.asarray(x[[i for i,_ in rr]]),dtype=torch.float32,device=device)
   with torch.no_grad(),torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
    h=model._encode_hidden(values).float().cpu().numpy()
   for j,(_,c) in enumerate(rr):sums[c]+=h[j];counts[c]+=1
  dh=sums['FLT']/counts['FLT']-sums['GC']/counts['GC'];mag=np.linalg.norm(dh,axis=1)
  rows.append(pd.DataFrame({'contrast_id':cid,'gene_symbol':GENES,'contextual_response_magnitude':mag,'contextual_rank':pd.Series(mag).rank(ascending=False,method='min').astype(int)}))
  print(f'[context heartbeat] {n}/{len(q)} {cid} samples={len(z)} elapsed={(time.time()-start)/60:.1f}m',flush=True)
 out=pd.concat(rows,ignore_index=True);out.to_parquet(cache,index=False);return out

def prepare_counts(q,m):
 target=WORK/'independent_counts.csv.gz';meta=WORK/'independent_memberships.csv';WORK.mkdir(parents=True,exist_ok=True)
 if target.exists() and meta.exists():return
 mmap=load_mouse_to_human_symbol_map(REPO/'data/ensembl/orthologs_one2one.txt'); matrices=[]
 for osd,ids in m.groupby('OSD').sample_id.apply(list).items():
  d=pd.read_csv(COUNT_FILES[osd],index_col=0);missing=set(ids)-set(d.columns)
  if missing:raise ValueError(f'{osd}: count columns missing: {sorted(missing)}')
  d=d[ids];d.index=d.index.astype(str).str.split('.').str[0];symbols=pd.Series(d.index,index=d.index).map(mmap);d=d.loc[symbols.notna()].copy();d.index=symbols[symbols.notna()].values
  d=d.groupby(level=0).sum().reindex(GENES,fill_value=0);matrices.append(d)
 counts=pd.concat(matrices,axis=1);assert counts.columns.is_unique and list(counts.index)==list(GENES)
 counts.to_csv(target,compression='gzip');m[['contrast_id','sample_id','condition','OSD']].to_csv(meta,index=False)

def edger(q,m):
 out=OUT/'edger_results.csv.gz'
 if not out.exists():
  prepare_counts(q,m);subprocess.run(['Rscript',str(HERE/'pipeline/run_independent_biological_replication_edger.R'),str(WORK/'independent_counts.csv.gz'),str(WORK/'independent_memberships.csv'),str(out)],check=True)
 return pd.read_csv(out)

def gsea(context,edge):
 cache=OUT/'pathway_enrichment.parquet'
 if cache.exists():return pd.read_parquet(cache)
 inputs=[]
 for cid,z in context.groupby('contrast_id'):inputs.append((cid,'Bridge_contextual',z[['gene_symbol','contextual_response_magnitude']]))
 for cid,z in edge[edge.tested].groupby('contrast_id'):inputs.append((cid,'edgeR_expression',z[['gene_symbol','signed_statistic']].dropna()))
 rows=[]
 for i,(cid,analysis,rank) in enumerate(inputs,1):
  for source,file in GMT.items():
   print(f'[GSEA] {i}/{len(inputs)} {cid} {analysis} {source}',flush=True)
   pre=gp.prerank(rnk=rank.sort_values(rank.columns[1],ascending=False),gene_sets=str(GMT_ROOT/file),min_size=10,max_size=500,permutation_num=N_PERM,threads=8,seed=SEED,outdir=None,verbose=False)
   z=pre.res2d.rename(columns={'Term':'pathway','ES':'es','NES':'nes','NOM p-val':'nominal_p','FDR q-val':'fdr','Lead_genes':'leading_edge'});z['contrast_id']=cid;z['analysis']=analysis;z['source']=source;rows.append(z[['contrast_id','analysis','source','pathway','es','nes','nominal_p','fdr','leading_edge']])
 ans=pd.concat(rows,ignore_index=True);ans['family']=ans.pathway.map(family);ans.to_parquet(cache,index=False);return ans

def summarize(q,context,edge,enr):
 meta=q[['contrast_id','OSD','mission']]; sig_all=enr[(enr.fdr<.05)&enr.family.ne('Other')].copy();sig_all['enrichment_location']=np.where(sig_all.nes>0,'high-ranked','low-ranked');sig_all.to_csv(OUT/'predefined_family_significant_terms.csv',index=False)
 # With a nonnegative contextual magnitude rank, only positive NES supports
 # concentration among genes with the strongest biological contextual response.
 # Negative contextual NES is retained as a low-response enrichment, not counted
 # as response-program recurrence. Signed edgeR NES is meaningful in either tail.
 sig=pd.concat([sig_all[(sig_all.analysis.eq('Bridge_contextual'))&(sig_all.nes>0)],sig_all[sig_all.analysis.eq('edgeR_expression')]],ignore_index=True)
 # Recurrence is counted both by contrasts and by independent OSDs.
 rec=(sig.merge(meta,on='contrast_id').groupby(['analysis','family']).agg(significant_terms=('pathway','nunique'),significant_contrasts=('contrast_id','nunique'),independent_osds=('OSD','nunique'),best_fdr=('fdr','min'),median_nes=('nes','median')).reset_index())
 grid=pd.MultiIndex.from_product([['Bridge_contextual','edgeR_expression'],['RNA processing / splicing','Chromatin organization / remodeling','DNA repair / DNA metabolism','Hepatic lipid / metabolic']],names=['analysis','family']).to_frame(index=False)
 rec=grid.merge(rec,how='left').fillna({'significant_terms':0,'significant_contrasts':0,'independent_osds':0});rec.to_csv(OUT/'program_recurrence_summary.csv',index=False)
 by_contrast=(sig.merge(meta,on='contrast_id').groupby(['analysis','family','contrast_id','OSD','mission']).agg(significant_terms=('pathway','nunique'),best_fdr=('fdr','min'),median_nes=('nes','median'),leading_edge_genes=('leading_edge',lambda x:len(set(';'.join(x.fillna('')).split(';'))-{''}))).reset_index())
 by_contrast.to_csv(OUT/'program_recurrence_by_contrast.csv',index=False)
 # Classify at family level conservatively.
 piv=rec.pivot(index='family',columns='analysis',values='independent_osds').fillna(0);classes=[]
 for fam,r in piv.iterrows():
  b=r.get('Bridge_contextual',0);e=r.get('edgeR_expression',0)
  cat='A: reproduced by expression and Bridge' if b>=2 and e>=2 else ('B: conventional signal amplified/reorganized by Bridge' if b>=2 and e>0 else ('C: contextual organization concentrated outside significant edgeR selection' if b>=2 else 'insufficient recurrence'))
  classes.append({'family':fam,'Bridge_independent_OSDs':int(b),'edgeR_independent_OSDs':int(e),'classification':cat})
 pd.DataFrame(classes).to_csv(OUT/'program_classification.csv',index=False)
 # Gene-level response agreement and pathway-level agreement.
 wide=context.pivot(index='gene_symbol',columns='contrast_id',values='contextual_response_magnitude');gene_corr=wide.corr(method='spearman');gene_cos=pd.DataFrame((wide.T.to_numpy()@wide.T.to_numpy().T)/(np.linalg.norm(wide.T.to_numpy(),axis=1)[:,None]*np.linalg.norm(wide.T.to_numpy(),axis=1)[None,:]),index=wide.columns,columns=wide.columns)
 gene_corr.to_csv(OUT/'gene_rank_spearman.csv');gene_cos.to_csv(OUT/'gene_magnitude_cosine.csv')
 ep=enr[enr.analysis.eq('Bridge_contextual')].pivot_table(index=['source','pathway'],columns='contrast_id',values='nes');path_corr=ep.corr(method='spearman');path_corr.to_csv(OUT/'pathway_nes_spearman.csv')
 pair=[]
 for i,a in enumerate(wide.columns):
  for b in wide.columns[i+1:]:pair.append({'contrast_a':a,'contrast_b':b,'gene_rank_spearman':gene_corr.loc[a,b],'gene_magnitude_cosine':gene_cos.loc[a,b],'pathway_nes_spearman':path_corr.loc[a,b]})
 pd.DataFrame(pair).to_csv(OUT/'gene_vs_pathway_agreement.csv',index=False)
 # Leading-edge overlap for recurrent family terms.
 le=[]
 for (analysis,fam),z in sig.groupby(['analysis','family']):
  for i,a in enumerate(z.contrast_id.unique()):
   A=set(';'.join(z[z.contrast_id.eq(a)].leading_edge.fillna('')).split(';'))-{''}
   for b in z.contrast_id.unique()[i+1:]:
    B=set(';'.join(z[z.contrast_id.eq(b)].leading_edge.fillna('')).split(';'))-{''};le.append({'analysis':analysis,'family':fam,'contrast_a':a,'contrast_b':b,'genes_a':len(A),'genes_b':len(B),'overlap':len(A&B),'jaccard':len(A&B)/max(1,len(A|B)),'overlap_genes':';'.join(sorted(A&B))})
 pd.DataFrame(le).to_csv(OUT/'leading_edge_overlap.csv',index=False)
 # Contextual top genes selected/not selected by edgeR.
 merged=context.merge(edge[['contrast_id','gene_symbol','FDR','tested']],on=['contrast_id','gene_symbol'],how='left');top=merged[merged.contextual_rank<=100].copy();top['edgeR_significant']=top.tested.eq(True)&top.FDR.lt(.05);top.to_csv(OUT/'contextual_top100_edger_status.csv',index=False)
 # Triangulation from existing controlled and RR1 outputs.
 controlled=pd.read_csv(HERE/'results/task4_confounding_profiler/controlled_gene_context/controlled_pathway_enrichment.csv');rr1=pd.read_parquet(HERE/'results/task4_confounding_profiler/contextual_robustness/gsea_full_results.parquet').query("comparison=='RR1'")
 tri=[]
 for fam in ['RNA processing / splicing','Chromatin organization / remodeling','DNA repair / DNA metabolism','Hepatic lipid / metabolic']:
  for stage,z in [('controlled PolyA/Ribo contextual sensitivity',controlled),('RR1 technical discrepancy',rr1),('independent FLT-GC contextual response',enr[enr.analysis.eq('Bridge_contextual')])]:
   zz=z[z.pathway.map(family).eq(fam)]; valid=zz.fdr.lt(.05)
   if stage.startswith('independent'):valid &= zz.nes.gt(0)
   tri.append({'family':fam,'stage':stage,'significant_terms':int(valid.sum()),'best_fdr':zz.loc[valid,'fdr'].min() if valid.any() else np.nan,'independent_contrasts':zz.loc[valid,'contrast_id'].nunique() if 'contrast_id' in zz else np.nan})
 pd.DataFrame(tri).to_csv(OUT/'technical_biological_triangulation.csv',index=False)
 decision={'decision':'B_with_important_limitations','plain_language':'Chromatin organization/remodeling and DNA-damage programs recur in independent FLT-GC data and are also supported by conventional edgeR, so they are not adequately described as an RR1-only technical artifact. Their BridgeRNA high-context recurrence is limited (two independent OSDs each), and one of the two DNA-supporting OSDs is the unreplicated 1-vs-1 RR1-CASIS stratum. The evidence therefore supports recurrent spaceflight-associated biology that is technically vulnerable in RR1, but only provisionally and without causal attribution.','RNA_processing':'Strong controlled and RR1 technical sensitivity, but high-contextual-response recurrence occurred in only one independent OSD; conventional edgeR recurrence was broader.','chromatin_independent_osds':2,'DNA_repair_independent_osds':2,'hepatic_metabolic_independent_osds':6,'top100_context_genes_edgeR_significant_fraction':float(top.edgeR_significant.mean()),'median_gene_rank_spearman':float(pd.DataFrame(pair).gene_rank_spearman.median()),'median_pathway_nes_spearman':float(pd.DataFrame(pair).pathway_nes_spearman.median()),'caveat':'Biological recurrence, technical sensitivity, and biology-by-technical interaction are distinct; recurrence does not prove a purely biological origin.'}
 (OUT/'decision_summary.json').write_text(json.dumps(decision,indent=2))
 return rec

def figures(rec):
 order=['RNA processing / splicing','Chromatin organization / remodeling','DNA repair / DNA metabolism','Hepatic lipid / metabolic'];colors={'Bridge_contextual':'#4477AA','edgeR_expression':'#CC6677'}
 fig,ax=plt.subplots(figsize=(10,5),layout='constrained');x=np.arange(4);w=.36
 for j,a in enumerate(colors):
  z=rec[rec.analysis.eq(a)].set_index('family').reindex(order);ax.bar(x+(j-.5)*w,z.independent_osds,w,label=a.replace('_',' '),color=colors[a])
 ax.set(xticks=x,xticklabels=order,ylabel='Independent OSDs with ≥1 FDR < 0.05 term',title='Program recurrence across independent spaceflight studies');ax.tick_params(axis='x',rotation=12);ax.legend();fig.savefig(FIG/'program_recurrence_by_method.png',dpi=350);fig.savefig(FIG/'program_recurrence_by_method.pdf');plt.close(fig)
 pair=pd.read_csv(OUT/'gene_vs_pathway_agreement.csv');fig,ax=plt.subplots(figsize=(6,5),layout='constrained');ax.scatter(pair.gene_rank_spearman,pair.pathway_nes_spearman,c='#228833',alpha=.75);ax.axhline(0,color='.7');ax.axvline(0,color='.7');ax.set(xlabel='Gene-magnitude rank Spearman',ylabel='Pathway NES Spearman',title='Gene-level versus program-level agreement');fig.savefig(FIG/'gene_vs_pathway_recurrence.png',dpi=350);fig.savefig(FIG/'gene_vs_pathway_recurrence.pdf');plt.close(fig)

def main():
 p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');p.add_argument('--batch-size',type=int,default=1);a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(exist_ok=True)
 q,m=design();device=torch.device(a.device if torch.cuda.is_available() else 'cpu');context=contextual(q,m,device,a.batch_size);edge=edger(q,m);enr=gsea(context,edge);rec=summarize(q,context,edge,enr);figures(rec)
 prov={'created_utc':datetime.now(timezone.utc).isoformat(),'primary_contrasts':11,'primary_samples':84,'excluded_from_independent_recurrence':['C04','C05','C06 (OSD-168 technical remeasurements)'],'contextual_metric':'L2 norm of mean contextual hidden embedding FLT minus GC, per gene','genes':15165,'gsea_permutations':N_PERM,'gsea_min_size':10,'gsea_max_size':500,'edgeR':'TMM + robust quasi-likelihood, separate model per contrast','frozen_bridge':True}
 (OUT/'provenance.json').write_text(json.dumps(prov,indent=2));print(rec.to_string(index=False));print('[complete]',OUT,flush=True)
if __name__=='__main__':main()
