#!/usr/bin/env python3
"""Decompose fixed T-cell PC1/PC2 geometry and contextual attribution."""
from __future__ import annotations
import json
from itertools import combinations
from pathlib import Path
import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE=Path(__file__).resolve().parents[1];REPO=HERE.parents[1]
OUT=HERE/'results/task4_pc1_pc2_specific_geometry';FIG=OUT/'figures'
T3=REPO/'benchmarks/osdr_batch_effect_representation';R3=T3/'results';W3=T3/'work'
NULL=HERE/'results/task4_pc12_empirical_biological_null';GLOB=T3.parent/'cross_species_exercise_response/work/hallmark_readout/archs4_bridgerna_embeddings.npy'
GMTROOT=REPO/'benchmarks/cross_species_exercise_response/results/per_study_ranked_gsea';GMTS={'GO:BP':'GO_Biological_Process_2026.gmt','KEGG':'KEGG_2026.gmt','Reactome':'Reactome_Pathways_2024.gmt'}
GENES=np.asarray(pd.read_csv(REPO/'data/ensembl/canonical_genes.csv').gene_symbol.astype(str));G=len(GENES);SEED=20260907

def cos(a,b):
 d=np.linalg.norm(a)*np.linalg.norm(b);return float(a@b/d) if d else np.nan
def basis():
 m=pd.read_parquet(HERE/'work/datasets/chen_2020_tcells/manifest.parquet').reset_index(drop=True);z=np.load(HERE/'work/datasets/chen_2020_tcells/bridgerna_embeddings.npy').astype(float);d=[]
 for _,q in m.groupby('pair_id',sort=True):d.append(z[q.index[q.library_prep.eq('ribo')]].mean(0)-z[q.index[q.library_prep.eq('polyA')]].mean(0))
 return np.linalg.svd(np.stack(d),full_matrices=False)[2][:2]

def sample_response(cid):
 mem=pd.read_csv(R3/'task3b_contrast_sample_membership.csv');m=pd.read_csv(R3/'sample_manifest.csv');z=np.load(W3/'bridgerna_embeddings.npy');ix=dict(zip(m.sample_id,range(len(m))));q=mem[mem.contrast_id.eq(cid)]
 return z[q[q.condition.eq('FLT')].sample_id.map(ix)].mean(0)-z[q[q.condition.eq('GC')].sample_id.map(ix)].mean(0)

def geometry(B):
 d=pd.read_csv(NULL/'biological_contrast_occupancy.csv');d['E_PC1']=d.PC1.pow(2)/d.response_norm.pow(2);d['E_PC2']=d.PC2.pow(2)/d.response_norm.pow(2)
 for pc in ['E_PC1','E_PC2']:d[f'{pc}_empirical_percentile']=d[pc].rank(method='average',pct=True)*100
 d['E_PC1_plus_E_PC2']=d.E_PC1+d.E_PC2;d.to_csv(OUT/'pc_specific_biological_contrasts.csv',index=False)
 # Add four-state and explicitly named RR reference values from existing vectors.
 four=pd.read_csv(HERE/'results/task4_rr3_39_40_four_state/four_state_response_geometry.csv');refs=[]
 for _,r in four[four.comparison.isin(['GC39→GC40','FLT39→FLT40'])].iterrows():refs.append({'reference':f'RR3 {r.comparison}','response_norm':r.full_norm,'a_PC1':r.PC1,'a_PC2':r.PC2,'E_PC1':r.PC1**2/r.full_norm**2,'E_PC2':r.PC2**2/r.full_norm**2})
 ids={'OSD-48 RR1 carcass FLT−GC':'C14__OSD-48__RR1-NASA__37-day','OSD-48 RR1 upon-euthanasia FLT−GC':'C13__OSD-48__RR1-NASA__37-day','OSD-137 RR3-40 FLT−GC':'C02__OSD-137__RR3__40-day','OSD-168 RR1 FLT−GC':'C04__OSD-168__RR1-NASA__37-day','OSD-168 RR3-40 FLT−GC':'C06__OSD-168__RR3__40-day'}
 for name,cid in ids.items():
  v=sample_response(cid);p=v@B.T;refs.append({'reference':name,'response_norm':np.linalg.norm(v),'a_PC1':p[0],'a_PC2':p[1],'E_PC1':p[0]**2/(v@v),'E_PC2':p[1]**2/(v@v)})
 r=pd.DataFrame(refs);r['E_PC1_plus_E_PC2']=r.E_PC1+r.E_PC2
 for pc in ['E_PC1','E_PC2']:
  base=d[d.primary_null][pc];r[f'{pc}_empirical_percentile']=[100*np.mean(base<x) for x in r[pc]]
 r.to_csv(OUT/'pc_specific_reference_contrasts.csv',index=False);return d,r

def global_geometry(B):
 z=np.load(GLOB,mmap_mode='r').astype(float);zc=z-z.mean(0);cov=zc.T@zc/(len(zc)-1);ev,V=np.linalg.eigh(cov);o=np.argsort(ev)[::-1];ev=ev[o];V=V[:,o]
 total=np.trace(cov);rows=[]
 rng=np.random.default_rng(SEED);iso=rng.normal(size=(10000,512));iso/=np.linalg.norm(iso,axis=1,keepdims=True);iso_var=np.einsum('ij,jk,ik->i',iso,cov,iso)/total
 weights=np.maximum(ev,0)/np.maximum(ev,0).sum();j=rng.choice(512,10000,p=weights);vm_var=ev[j]/total
 for i,name in enumerate(['PC1','PC2']):
  b=B[i];align=np.abs(V.T@b);variance=float(b@cov@b);row={'Tcell_PC':name,'ARCHS4_variance':variance,'total_variance_fraction':variance/total,'strongest_global_PC':int(np.argmax(align)+1),'strongest_global_PC_abs_cosine':align.max(),'isotropic_variance_percentile':100*np.mean(iso_var<variance/total),'variance_matched_PC_percentile':100*np.mean(vm_var<variance/total)}
  for k in [2,5,10,20,50]:row[f'energy_in_top{k}_global_PCs']=float(np.sum(align[:k]**2))
  rows.append(row)
 out=pd.DataFrame(rows);out.to_csv(OUT/'global_pc_specific_geometry.csv',index=False);pd.DataFrame({'global_PC':np.arange(1,513),'variance_fraction':ev/ev.sum()}).to_csv(OUT/'global_variance_spectrum.csv',index=False);return out

def contextual_responses():
 ctrl=np.memmap(HERE/'work/task4_controlled_gene_context/controlled_context_displacements.float16.dat',dtype='float16',mode='r',shape=(40,G,512));tc=np.empty((G,512),np.float32)
 for s in range(0,G,256):tc[s:s+256]=np.asarray(ctrl[:,s:s+256],np.float32).mean(0)
 old=np.memmap(HERE/'work/task4_rna_processing_gene_analysis/matched_contextual_responses.float16.dat',dtype='float16',mode='r',shape=(6,G,512));full=np.memmap(HERE/'work/task4_selective_tcell_signature_filter/full_biological_contextual_responses.float16.dat',dtype='float16',mode='r',shape=(3,G,512));rr=np.memmap(HERE/'work/task4_rr3_39_40_four_state/rr3_nine_animals_context.float16.dat',dtype='float16',mode='r',shape=(9,G,512))
 # four-state cache order follows animal_metadata_audit.csv.
 q=pd.read_csv(HERE/'results/task4_rr3_39_40_four_state/animal_metadata_audit.csv');lookup={a:np.asarray(rr[i],np.float32) for i,a in enumerate(q.animal_id)}
 groups={'GC39':['G1','G2'],'FLT39':['F1','F2'],'GC40':['G3','G5'],'FLT40':['F3','F4','F5']};state={k:np.stack([lookup[a] for a in aa]).mean(0) for k,aa in groups.items()}
 return {'T-cell PolyA→Ribo':tc,'RR1 OSD-48 carcass FLT−GC':np.asarray(full[0],np.float32),'RR1 OSD-168 FLT−GC':np.asarray(old[1],np.float32),'RR3-40 OSD-137 FLT−GC':np.asarray(full[2],np.float32),'RR3-40 OSD-168 FLT−GC':np.asarray(old[5],np.float32),'RR3 GC39→GC40':state['GC40']-state['GC39'],'RR3 FLT39→FLT40':state['FLT40']-state['FLT39']}

def attributions(ctx,B):
 rows=[];vectors={};latent={}
 for name,h in ctx.items():
  latent[name]=h.mean(0)
  for j,pc in enumerate(['PC1','PC2']):
   a=h@B[j]/G;vectors[(name,pc)]=a;rank=pd.Series(np.abs(a)).rank(ascending=False,method='min').astype(int)
   rows.append(pd.DataFrame({'context':name,'PC':pc,'gene_symbol':GENES,'signed_attribution':a,'absolute_attribution':np.abs(a),'rank':rank}))
 out=pd.concat(rows,ignore_index=True);out.to_parquet(OUT/'pc_specific_gene_attribution.parquet',index=False);out[out['rank']<=500].to_csv(OUT/'pc_specific_top500_genes.csv',index=False)
 pair=[]
 for pc in ['PC1','PC2']:
  for a,b in combinations(ctx,2):
   x,y=vectors[(a,pc)],vectors[(b,pc)];r={'PC':pc,'context_A':a,'context_B':b,'signed_geometry_product':(latent[a]@B[int(pc[-1])-1])*(latent[b]@B[int(pc[-1])-1]),'attribution_cosine':cos(x,y),'attribution_spearman':spearmanr(x,y).statistic}
   for n in [100,500]:
    sa=set(np.argpartition(np.abs(x),-n)[-n:]);sb=set(np.argpartition(np.abs(y),-n)[-n:]);ov=list(sa&sb);r[f'top{n}_overlap']=len(ov);r[f'top{n}_direction_agreement']=np.mean(np.sign(x[ov])==np.sign(y[ov])) if ov else np.nan
   pair.append(r)
 pairs=pd.DataFrame(pair);pairs.to_csv(OUT/'pc_specific_attribution_similarity.csv',index=False);return out,vectors,pairs

def pathways(profiles):
 terms=[]
 for source,file in GMTS.items():
  for term,members in gp.parser.read_gmt(path=str(GMTROOT/file)).items():
   ii=np.flatnonzero(np.isin(GENES,list(set(members))))
   if 10<=len(ii)<=500:terms.append((source,term,ii))
 rows=[]
 for (context,pc),q in profiles.groupby(['context','PC'],sort=False):
  x=q.sort_values('gene_symbol').set_index('gene_symbol').loc[GENES,'signed_attribution'].to_numpy();z=(x-x.mean())/(x.std() or 1)
  for source,term,ii in terms:rows.append({'context':context,'PC':pc,'source':source,'pathway':term,'pathway_attribution_z':z[ii].mean()*np.sqrt(len(ii)),'genes':len(ii)})
 d=pd.DataFrame(rows);d.to_parquet(OUT/'pc_specific_pathway_profiles.parquet',index=False);pair=[]
 for pc,g in d.groupby('PC'):
  w=g.pivot(index=['source','pathway'],columns='context',values='pathway_attribution_z')
  for a,b in combinations(w.columns,2):
   sa=set(w[a].abs().nlargest(100).index);sb=set(w[b].abs().nlargest(100).index);ov=sa&sb;pair.append({'PC':pc,'context_A':a,'context_B':b,'pathway_profile_pearson':w[a].corr(w[b]),'pathway_profile_spearman':w[a].corr(w[b],method='spearman'),'top100_pathway_overlap':len(ov),'top100_direction_agreement':np.mean([np.sign(w.loc[i,a])==np.sign(w.loc[i,b]) for i in ov]) if ov else np.nan})
 pd.DataFrame(pair).to_csv(OUT/'pc_specific_pathway_similarity.csv',index=False);return d,pd.DataFrame(pair)

def figures(geom,refs,glob,pairs,pathpairs,path):
 plt.style.use('seaborn-v0_8-whitegrid');FIG.mkdir(parents=True,exist_ok=True)
 fig,axes=plt.subplots(1,3,figsize=(18,5));axes[0].hist(geom.E_PC1,alpha=.65,label='PC1');axes[0].hist(geom.E_PC2,alpha=.65,label='PC2');axes[0].legend();axes[0].set(xlabel='Energy fraction',title='Biological occupancy')
 axes[1].scatter(geom.E_PC1,geom.E_PC2);axes[1].set(xlabel='E_PC1',ylabel='E_PC2',title='PC-specific energy')
 axes[2].scatter(geom.PC1,geom.PC2);axes[2].axhline(0,color='grey');axes[2].axvline(0,color='grey');axes[2].set(xlabel='Signed PC1',ylabel='Signed PC2',title='Signed response coordinates');fig.tight_layout()
 for e in ['png','pdf']:fig.savefig(FIG/f'pc_specific_geometry.{e}',dpi=400);plt.close(fig)
 fig,ax=plt.subplots(figsize=(10,6));g=refs.set_index('reference')[['E_PC1','E_PC2']];g.plot.bar(stacked=True,ax=ax,color=['#4C78A8','#F58518']);ax.set(ylabel='Response energy fraction',title='PC1 and PC2 contributions to reference responses');fig.tight_layout()
 for e in ['png','pdf']:fig.savefig(FIG/f'reference_pc_decomposition.{e}',dpi=400,bbox_inches='tight');plt.close(fig)
 names=list(dict.fromkeys(pairs.context_A.tolist()+pairs.context_B.tolist()))
 fig,axes=plt.subplots(1,2,figsize=(16,7))
 for ax,pc in zip(axes,['PC1','PC2']):
  m=np.eye(len(names));q=pairs[pairs.PC.eq(pc)]
  for _,r in q.iterrows():i=names.index(r.context_A);j=names.index(r.context_B);m[i,j]=m[j,i]=r.attribution_cosine
  im=ax.imshow(m,cmap='RdBu_r',vmin=-1,vmax=1);ax.set_xticks(range(len(names)),[x.replace(' FLT−GC','') for x in names],rotation=70,ha='right',fontsize=7);ax.set_yticks(range(len(names)),[x.replace(' FLT−GC','') for x in names],fontsize=7);ax.set_title(f'{pc} attribution cosine')
 fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.8);fig.subplots_adjust(bottom=.35,wspace=.45)
 for e in ['png','pdf']:fig.savefig(FIG/f'pc_attribution_similarity.{e}',dpi=400,bbox_inches='tight');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(17,10))
 for ax,pc in zip(axes,['PC1','PC2']):
  q=path[path.PC.eq(pc)];w=q.pivot(index=['source','pathway'],columns='context',values='pathway_attribution_z');sel=w.abs().max(axis=1).nlargest(25).index;w=w.loc[sel];lim=np.nanmax(np.abs(w.values));im=ax.imshow(w,cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto');ax.set_xticks(range(len(w.columns)),[x.replace(' FLT−GC','') for x in w.columns],rotation=70,ha='right',fontsize=7);ax.set_yticks(range(len(w)),[x[1][:55] for x in w.index],fontsize=6);ax.set_title(f'{pc}: strongest pathway-attribution profiles')
 fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.7,label='Attribution z');fig.subplots_adjust(bottom=.3,wspace=.75)
 for e in ['png','pdf']:fig.savefig(FIG/f'pc_specific_pathways.{e}',dpi=400,bbox_inches='tight');plt.close(fig)

def main():
 OUT.mkdir(parents=True,exist_ok=True);B=basis();geom,refs=geometry(B);glob=global_geometry(B);ctx=contextual_responses();profiles,vectors,pairs=attributions(ctx,B);path,pathpairs=pathways(profiles);figures(geom,refs,glob,pairs,pathpairs,path)
 summary={'primary_contrasts':int(geom.primary_null.sum()),'median_E_PC1':float(geom[geom.primary_null].E_PC1.median()),'median_E_PC2':float(geom[geom.primary_null].E_PC2.median()),'global_PC1_variance_fraction':float(glob.iloc[0].total_variance_fraction),'global_PC2_variance_fraction':float(glob.iloc[1].total_variance_fraction),'basis':'unchanged controlled T-cell uncentered SVD PC1/PC2','contextual_inference_recomputed':False}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(OUT/'provenance.json').write_text(json.dumps({'basis':'unchanged controlled T-cell uncentered SVD PC1/PC2','attribution':'existing cached contextual response times fixed PC divided by 15165','RR3_40_original_context':'full F3/F4/F5 vs G3/G5','RR3_40_remeasurement_context':'strict matched F3/F4 vs G3/G5','pathways':'descriptive standardized mean signed attribution times sqrt(set size); not causal enrichment','embeddings_recomputed':False},indent=2)+'\n');print('\nReferences\n',refs.to_string(index=False));print('\nGlobal\n',glob.to_string(index=False));print('\nSummary\n',json.dumps(summary,indent=2));print('[complete]',OUT)
if __name__=='__main__':main()
