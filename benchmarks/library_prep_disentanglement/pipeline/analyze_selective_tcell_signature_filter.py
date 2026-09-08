#!/usr/bin/env python3
"""Selective response-level filtering using T-cell-only PolyA/Ribo gene signatures."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import hypergeom,pearsonr,spearmanr

ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[1]
OUT=ROOT/'results/task4_selective_tcell_signature_filter';FIG=OUT/'figures';WORK=ROOT/'work/task4_selective_tcell_signature_filter'
T3=REPO/'benchmarks/osdr_batch_effect_representation';R3=T3/'results';W3=T3/'work';CTRL=ROOT/'results/task4_confounding_profiler/controlled_gene_context'
GMTROOT=REPO/'benchmarks/cross_species_exercise_response/results/per_study_ranked_gsea';GMT={'GO:BP':'GO_Biological_Process_2026.gmt','KEGG':'KEGG_2026.gmt','Reactome':'Reactome_Pathways_2024.gmt'}
sys.path[:0]=[str(REPO/'benchmarks/tcga_downstream/pipeline'),str(REPO)]
from run_attention_pooling import load_frozen_encoder
from src.fm_embed.vocab import load_canonical_genes
GENES=np.asarray(load_canonical_genes(REPO/'data/ensembl/canonical_genes.csv'));G=len(GENES);SEED=94271

def cos(a,b):
 d=np.linalg.norm(a)*np.linalg.norm(b);return float(a@b/d) if d else np.nan
def metrics(a,b):return {'cosine':cos(a,b),'pearson':pearsonr(a,b).statistic,'spearman':spearmanr(a,b).statistic}
def basis():
 m=pd.read_parquet(ROOT/'work/datasets/chen_2020_tcells/manifest.parquet').reset_index(drop=True);z=np.load(ROOT/'work/datasets/chen_2020_tcells/bridgerna_embeddings.npy').astype(float);d=[]
 for _,g in m.groupby('pair_id',sort=True):d.append(z[g.index[g.library_prep.eq('ribo')][0]]-z[g.index[g.library_prep.eq('polyA')][0]])
 d=np.stack(d);_,_,vt=np.linalg.svd(d,full_matrices=False);return vt[:2],d

def signatures(B,D):
 ctx=np.memmap(ROOT/'work/task4_controlled_gene_context/controlled_context_displacements.float16.dat',dtype='float16',mode='r',shape=(40,G,512));mean=np.zeros((G,512),np.float32)
 for start in range(0,G,256):mean[start:start+256]=np.asarray(ctx[:,start:start+256],np.float32).mean(0)
 ctrl=pd.read_parquet(CTRL/'controlled_gene_sensitivity.parquet').set_index('gene_symbol').loc[GENES].reset_index();coords=mean@B.T;direction=D.mean(0);direction=(direction@B.T)@B;direction/=np.linalg.norm(direction)
 ctrl['signed_direction']=mean@direction;ctrl['PC1_contribution']=coords[:,0];ctrl['PC2_contribution']=coords[:,1];ctrl['PC1_2_magnitude']=np.linalg.norm(coords,axis=1)
 de=pd.read_csv(ROOT/'results/task4_confounding_profiler/conventional_expression_baseline/tcell_edger.csv').rename(columns={'gene_symbol':'gene_symbol','logFC':'paired_expression_logFC','FDR':'paired_expression_FDR'});ctrl=ctrl.merge(de[['gene_symbol','paired_expression_logFC','paired_expression_FDR']],on='gene_symbol',how='left')
 rna=pd.read_csv(ROOT/'results/task4_confounding_profiler/rna_processing_gene_analysis/rna_processing_pathway_membership.csv');rcol='gene' if 'gene' in rna else 'gene_symbol';ann=rna.groupby(rcol).pathway.apply(lambda x:';'.join(sorted(set(x)))).rename('RNA_pathways');ctrl=ctrl.merge(ann,left_on='gene_symbol',right_index=True,how='left')
 sets={f'top_{n}':set(ctrl.nsmallest(n,'sensitivity_rank').gene_symbol) for n in [100,250,500,1000]}
 sets['paired_DE_FDR05']=set(ctrl.loc[ctrl.paired_expression_FDR<.05,'gene_symbol'])
 enr=pd.read_parquet(CTRL/'controlled_pathway_enrichment_v2.parquet');sig=enr[enr.fdr<.05].copy();rna_terms=sig[sig.pathway.str.contains('RNA|SPLIC|RIBO|MRNA|RRNA',case=False,regex=True,na=False)]
 lead=set();
 for s in rna_terms.leading_edge.dropna():lead.update(str(s).replace(';',',').split(','))
 sets['RNA_leading_edge']=lead&set(GENES)
 ctrl['signature_memberships']='';
 for name,s in sets.items():ctrl.loc[ctrl.gene_symbol.isin(s),'signature_memberships']+=name+';'
 ctrl.to_parquet(OUT/'controlled_tcell_technical_signature.parquet',index=False)
 pd.DataFrame([{'signature':k,'genes':len(v),'definition':'T-cell only'} for k,v in sets.items()]).to_csv(OUT/'signature_definitions.csv',index=False)
 return ctrl,sets

def load_context(device):
 oldnames=['RR1 matched original','RR1 remeasurement','RR3-39','RR3-39 remeasurement','RR3-40','RR3-40 remeasurement'];old=np.memmap(ROOT/'work/task4_rna_processing_gene_analysis/matched_contextual_responses.float16.dat',dtype='float16',mode='r',shape=(6,G,512));arrays={n:np.asarray(old[i],np.float32) for i,n in enumerate(oldnames)}
 cache=WORK/'full_biological_contextual_responses.float16.dat';shape=(3,G,512);WORK.mkdir(parents=True,exist_ok=True)
 if not (cache.exists() and cache.stat().st_size==np.prod(shape)*2):
  manifest=pd.read_csv(R3/'sample_manifest.csv');membership=pd.read_csv(R3/'task3b_contrast_sample_membership.csv');x=np.load(W3/'bridgerna_log1p_tpm_inputs.npy',mmap_mode='r');idx=dict(zip(manifest.sample_id,range(len(manifest))));model=load_frozen_encoder(torch.device(device));mm=np.memmap(cache,dtype='float16',mode='w+',shape=shape)
  for oi,cid in enumerate(['C14__OSD-48__RR1-NASA__37-day','C13__OSD-48__RR1-NASA__37-day','C02__OSD-137__RR3__40-day']):
   g=membership[membership.contrast_id==cid];means={}
   for condition in ['FLT','GC']:
    ids=g[g.condition==condition].sample_id.map(idx).tolist();acc=np.zeros((G,512),np.float32)
    for i in ids:
     with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.float16):h=model._encode_hidden(torch.as_tensor(np.array(x[i],copy=True)[None],dtype=torch.float32,device=device))[0].float().cpu().numpy()
     acc+=h
    means[condition]=acc/len(ids)
   mm[oi]=(means['FLT']-means['GC']).astype(np.float16);mm.flush();print(f'[heartbeat] contextual cohorts={oi+1}/3',flush=True)
  del mm
 full=np.memmap(cache,dtype='float16',mode='r',shape=shape);arrays['RR1 carcass']=np.asarray(full[0],np.float32);arrays['RR1 upon-euthanasia']=np.asarray(full[1],np.float32);arrays['RR3-40']=np.asarray(full[2],np.float32)
 return arrays

def corrected(delta,B,selected):
 raw=delta.mean(0);p=np.einsum('gd,kd,kj->gj',delta[selected],B,B,optimize=True).sum(0)/G if len(selected) else np.zeros(512);return raw-p,p
def whole(delta,B):
 r=delta.mean(0);return r-(r@B.T)@B

def benchmark(arrays,B,sets):
 pairs=[('RR1 technical','RR1 matched original','RR1 remeasurement'),('RR1 carcass/RR3-39','RR1 carcass','RR3-39'),('RR1 carcass/RR3-40','RR1 carcass','RR3-40'),('RR1 euth/RR3-39','RR1 upon-euthanasia','RR3-39'),('RR1 euth/RR3-40','RR1 upon-euthanasia','RR3-40')];gidx={g:i for i,g in enumerate(GENES)};rows=[]
 for sig,genes in sets.items():
  sel=np.array([gidx[g] for g in genes if g in gidx],int)
  for label,a,b in pairs:
   variants={'raw':(arrays[a].mean(0),arrays[b].mean(0)),'whole_PC1_2_removed':(whole(arrays[a],B),whole(arrays[b],B)),'selective':(corrected(arrays[a],B,sel)[0],corrected(arrays[b],B,sel)[0])}
   rawcos=cos(*variants['raw'])
   for method,(x,y) in variants.items():
    z=metrics(x,y);rows.append({'signature':sig,'comparison':label,'method':method,'selected_genes':len(sel),**z,'absolute_cosine_change':z['cosine']-rawcos,'signed_relationship_preservation':z['cosine']/rawcos if abs(rawcos)>.1 else np.nan})
 return pd.DataFrame(rows)

def nulls(arrays,B,ctrl,sets,reps=200):
 rng=np.random.default_rng(SEED);rows=[];gidx={g:i for i,g in enumerate(GENES)};score=ctrl.sensitivity_score.rank(pct=True);expr=ctrl.mean_log1p_tpm.rank(pct=True);bins=(np.minimum((score*5).astype(int),4)*5+np.minimum((expr*5).astype(int),4)).to_numpy()
 for sig in ['top_100','top_250','top_500','top_1000']:
  actual=np.array([gidx[g] for g in sets[sig]],int);counts=pd.Series(bins[actual]).value_counts().to_dict();pool=np.arange(G)
  for rep in range(reps):
   chosen=[]
   for k,n in counts.items():
    q=np.setdiff1d(pool[bins==k],actual,assume_unique=False);chosen.extend(rng.choice(q,n,replace=len(q)<n))
   chosen=np.asarray(chosen,int)
   a=corrected(arrays['RR1 matched original'],B,chosen)[0];b=corrected(arrays['RR1 remeasurement'],B,chosen)[0]
   rows.append({'signature':sig,'replicate':rep,'genes':len(chosen),'rr1_technical_cosine':cos(a,b),'rr1_carcass_rr3_39_cosine':cos(corrected(arrays['RR1 carcass'],B,chosen)[0],corrected(arrays['RR3-39'],B,chosen)[0])})
 return pd.DataFrame(rows)

def overlap_tables(arrays,B,D,ctrl,sets):
 gidx={g:i for i,g in enumerate(GENES)};t_rank=ctrl.set_index('gene_symbol').sensitivity_score;rows=[]
 direction=(D.mean(0)@B.T)@B;direction/=np.linalg.norm(direction)
 for name,d in arrays.items():
  pc=np.einsum('gd,kd,kj->gj',d,B,B,optimize=True);score=np.linalg.norm(pc,axis=1);rank=pd.Series(score,index=GENES).rank(ascending=False)
  rho=spearmanr(t_rank.loc[GENES],score).statistic
  signed=pd.Series(pc@direction,index=GENES)
  for n in [100,250,500,1000]:
   a=set(t_rank.nlargest(n).index);r=set(rank.nsmallest(n).index);o=a&r;rows.append({'response':name,'top_n':n,'overlap':len(o),'jaccard':len(o)/len(a|r),'expected':n*n/G,'fold_enrichment':len(o)/(n*n/G),'hypergeom_p':hypergeom.sf(len(o)-1,G,n,n),'rank_spearman':rho,'direction_agreement':np.mean(np.sign(ctrl.set_index('gene_symbol').loc[list(o),'signed_direction'])==np.sign(signed.loc[list(o)])) if o else np.nan})
 return pd.DataFrame(rows)

def response_decomposition(arrays,B,sets,ctrl):
 """Partition the PC1-2 response into T-cell-signature and other-gene sums.

 The two gene sums can interfere constructively/destructively, so their squared
 norm ratios need not sum to one.  Reconstruction error documents the exact
 additive decomposition instead of presenting the ratios as causal fractions.
 """
 gidx={g:i for i,g in enumerate(GENES)};rows=[]
 for response,d in arrays.items():
  raw=d.mean(0);inside=(raw@B.T)@B;outside=raw-inside
  for signature,genes in sets.items():
   sel=np.array([gidx[g] for g in genes if g in gidx],int)
   sig=np.einsum('gd,kd,kj->j',d[sel],B,B,optimize=True)/G if len(sel) else np.zeros(512)
   nonsig=inside-sig;den=np.linalg.norm(inside)
   rows.append({'response':response,'signature':signature,'signature_genes':len(sel),
    'response_norm':np.linalg.norm(raw),'outside_PC1_2_norm':np.linalg.norm(outside),
    'inside_PC1_2_norm':den,'signature_inside_norm':np.linalg.norm(sig),
    'non_signature_inside_norm':np.linalg.norm(nonsig),
    'signature_to_inside_norm_ratio':np.linalg.norm(sig)/den if den else np.nan,
    'signature_to_inside_squared_norm_ratio':np.dot(sig,sig)/np.dot(inside,inside) if den else np.nan,
    'non_signature_to_inside_norm_ratio':np.linalg.norm(nonsig)/den if den else np.nan,
    'reconstruction_error':np.linalg.norm(raw-(outside+sig+nonsig))})
 return pd.DataFrame(rows)

def preserved_biology(arrays,B,sets,ctrl):
 """Describe independently shared RR1/RR3 genes retained by selective filtering."""
 pairs=[('RR1 carcass/RR3-39','RR1 carcass','RR3-39'),('RR1 carcass/RR3-40','RR1 carcass','RR3-40'),
        ('RR1 euth/RR3-39','RR1 upon-euthanasia','RR3-39'),('RR1 euth/RR3-40','RR1 upon-euthanasia','RR3-40')]
 projected={n:np.einsum('gd,kd,kj->gj',d,B,B,optimize=True) for n,d in arrays.items()}
 gene_rows=[]
 for comparison,a,b in pairs:
  sa=np.linalg.norm(projected[a],axis=1);sb=np.linalg.norm(projected[b],axis=1)
  shared=set(GENES[np.argsort(sa)[-500:]])&set(GENES[np.argsort(sb)[-500:]])
  for signature,selected in sets.items():
   for gene in shared:
    i=np.where(GENES==gene)[0][0]
    gene_rows.append({'comparison':comparison,'signature':signature,'gene_symbol':gene,
     'in_tcell_signature':gene in selected,'rr1_component_magnitude':sa[i],
     'rr3_component_magnitude':sb[i],'gene_vector_cosine':cos(projected[a][i],projected[b][i]),
     'rna_processing':pd.notna(ctrl.set_index('gene_symbol').loc[gene,'RNA_pathways'])})
 genes=pd.DataFrame(gene_rows)
 summary=(genes.groupby(['comparison','signature'],as_index=False).agg(
  shared_top500_genes=('gene_symbol','size'),tcell_signature_overlap=('in_tcell_signature','sum'),
  retained_non_signature_genes=('in_tcell_signature',lambda x:(~x).sum()),
  rna_processing_shared=('rna_processing','sum')))
 return genes,summary

def biology(arrays,B,sets):
 rows=[];universe=set(GENES);libs={s:gp.parser.read_gmt(path=str(GMTROOT/f)) for s,f in GMT.items()}
 for signature,genes in sets.items():
  selected=set(genes);N=len(selected)
  for source,terms in libs.items():
   for term,members in terms.items():
    members=set(members)&universe;n=len(members);k=len(selected&members)
    if N and 10<=n<=500 and k:rows.append({'signature':signature,'source':source,'pathway':term,'overlap':k,'p_value':hypergeom.sf(k-1,G,n,N),'genes':';'.join(sorted(selected&members))})
 d=pd.DataFrame(rows)
 if len(d):d['fdr']=d.groupby('signature').p_value.transform(lambda x:pd.Series(np.minimum.accumulate((np.sort(x)*len(x)/np.arange(1,len(x)+1))[::-1])[::-1],index=x.sort_values().index).reindex(x.index).values)
 return d

def plots(results,null):
 plt.style.use('seaborn-v0_8-whitegrid');q=results[results.signature=='top_500'];p=q.pivot(index='comparison',columns='method',values='cosine').reindex(columns=['raw','whole_PC1_2_removed','selective']);fig,ax=plt.subplots(figsize=(11,6));p.plot.bar(ax=ax,color=['#4C78A8','#E45756','#59A14F']);ax.axhline(0,color='k',lw=.8);ax.set(ylabel='Cosine similarity',title='Raw, whole-subspace, and T-cell Top-500 selective filtering');ax.tick_params(axis='x',rotation=20);fig.tight_layout()
 for e in ['png','pdf']:fig.savefig(FIG/f'three_way_benchmark.{e}',dpi=400)
 plt.close(fig);fig,ax=plt.subplots(figsize=(8,5));g=null[null.signature=='top_500'];ax.hist(g.rr1_technical_cosine,bins=25,color='#BBBBBB');obs=q[(q.comparison=='RR1 technical')&(q.method=='selective')].cosine.iloc[0];ax.axvline(obs,color='#D62728',lw=3,label='T-cell signature');ax.set(xlabel='RR1 technical-replication cosine',ylabel='Matched random sets',title='Expression/context-score matched random control');ax.legend();fig.tight_layout()
 for e in ['png','pdf']:fig.savefig(FIG/f'top500_random_control.{e}',dpi=400)
 plt.close(fig)

def main():
 a=argparse.ArgumentParser();a.add_argument('--device',default='cuda:0');a.add_argument('--random-replicates',type=int,default=200);x=a.parse_args();OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(exist_ok=True)
 B,D=basis();ctrl,sets=signatures(B,D);arrays=load_context(x.device);results=benchmark(arrays,B,sets);results.to_csv(OUT/'three_way_benchmark.csv',index=False);null=nulls(arrays,B,ctrl,sets,x.random_replicates);null.to_parquet(OUT/'matched_random_controls.parquet',index=False)
 ov=overlap_tables(arrays,B,D,ctrl,sets);ov.to_csv(OUT/'tcell_rr_gene_mechanism_overlap.csv',index=False);enr=biology(arrays,B,sets);enr.to_csv(OUT/'controlled_signature_pathway_enrichment.csv',index=False);plots(results,null)
 decomp=response_decomposition(arrays,B,sets,ctrl);decomp.to_csv(OUT/'response_signature_decomposition.csv',index=False)
 preserved,preserved_summary=preserved_biology(arrays,B,sets,ctrl);preserved.to_parquet(OUT/'preserved_biology_genes.parquet',index=False);preserved_summary.to_csv(OUT/'preserved_biology_summary.csv',index=False)
 summary=[]
 for sig in sets:
  q=results[(results.signature==sig)&(results.method=='selective')].set_index('comparison');n=null[null.signature==sig] if sig in null.signature.unique() else pd.DataFrame();summary.append({'signature':sig,'genes':len(sets[sig]),'RR1_technical_cosine':q.loc['RR1 technical','cosine'],'technical_gain':q.loc['RR1 technical','absolute_cosine_change'],'RR1_carcass_RR3_39':q.loc['RR1 carcass/RR3-39','cosine'],'RR1_carcass_RR3_40':q.loc['RR1 carcass/RR3-40','cosine'],'RR1_euth_RR3_39':q.loc['RR1 euth/RR3-39','cosine'],'RR1_euth_RR3_40':q.loc['RR1 euth/RR3-40','cosine'],'random_empirical_p':(1+(n.rr1_technical_cosine>=q.loc['RR1 technical','cosine']).sum())/(len(n)+1) if len(n) else np.nan})
 pd.DataFrame(summary).to_csv(OUT/'selective_filter_summary.csv',index=False)
 (OUT/'provenance.json').write_text(json.dumps({'reference':'controlled T-cell only; unchanged uncentered PC1-2','selective_formula':'delta_corrected = mean_g(delta_h_g) - sum_{g in signature} P_PC1-2(delta_h_g)/15165','new_contextual_inference':['RR1 full carcass','RR1 upon-euthanasia','RR3-40 full 3-FLT/2-GC cohort'],'sample_embeddings_recomputed':False,'random_replicates':x.random_replicates,'seed':SEED,'limitations':['Response-level experiment; not a deployable sample correction.','T-cell signature selection is independent of RR1/RR3.','Full-cohort contextual tensors were generated from the frozen model because signed vectors were not cached.']},indent=2)+'\n')
 print(pd.DataFrame(summary).to_string(index=False));print('[complete]',OUT)
if __name__=='__main__':main()
