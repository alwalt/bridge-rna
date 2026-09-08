#!/usr/bin/env python3
"""Compare response-specific gene mechanisms inside the fixed PolyA/Ribo-sensitive PC1-2."""
from __future__ import annotations
import json
from itertools import combinations
from pathlib import Path

import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import hypergeom, pearsonr, spearmanr

HERE=Path(__file__).resolve().parents[1];REPO=HERE.parents[1]
OUT=HERE/'results/task4_pc12_attribution_mechanisms';FIG=OUT/'figures'
T3=REPO/'benchmarks/osdr_batch_effect_representation';R3=T3/'results';W3=T3/'work'
GMTROOT=REPO/'benchmarks/cross_species_exercise_response/results/per_study_ranked_gsea'
GMTS={'GO:BP':'GO_Biological_Process_2026.gmt','KEGG':'KEGG_2026.gmt','Reactome':'Reactome_Pathways_2024.gmt'}
GENES=np.asarray(pd.read_csv(REPO/'data/ensembl/canonical_genes.csv').gene_symbol.astype(str));G=len(GENES)

def cos(a,b):
 d=np.linalg.norm(a)*np.linalg.norm(b);return float(a@b/d) if d else np.nan
def condition(s):return 'FLT' if '_FLT_' in s else 'GC' if '_GC_' in s else None

def fixed_basis():
 m=pd.read_parquet(HERE/'work/datasets/chen_2020_tcells/manifest.parquet').reset_index(drop=True)
 z=np.load(HERE/'work/datasets/chen_2020_tcells/bridgerna_embeddings.npy').astype(float);d=[]
 for _,q in m.groupby('pair_id',sort=True):
  d.append(z[q.index[q.library_prep.eq('ribo')]].mean(0)-z[q.index[q.library_prep.eq('polyA')]].mean(0))
 return np.linalg.svd(np.stack(d),full_matrices=False)[2][:2]

def contextual_responses():
 ctrl=np.memmap(HERE/'work/task4_controlled_gene_context/controlled_context_displacements.float16.dat',dtype='float16',mode='r',shape=(40,G,512))
 tc=np.empty((G,512),np.float32)
 for s in range(0,G,256):tc[s:s+256]=np.asarray(ctrl[:,s:s+256],np.float32).mean(0)
 old=np.memmap(HERE/'work/task4_rna_processing_gene_analysis/matched_contextual_responses.float16.dat',dtype='float16',mode='r',shape=(6,G,512))
 full=np.memmap(HERE/'work/task4_selective_tcell_signature_filter/full_biological_contextual_responses.float16.dat',dtype='float16',mode='r',shape=(3,G,512))
 return {'T-cell PolyA→Ribo':tc,'RR1 OSD-48 carcass':np.asarray(full[0],np.float32),
  'RR1 OSD-48 upon-euthanasia':np.asarray(full[1],np.float32),'RR1 OSD-168 no-ERCC':np.asarray(old[1],np.float32),
  'RR3-39':np.asarray(old[2],np.float32),'RR3-40':np.asarray(full[2],np.float32)}

def expression_responses():
 result={};m=pd.read_parquet(HERE/'work/datasets/chen_2020_tcells/manifest.parquet').reset_index(drop=True);x=np.load(HERE/'work/datasets/chen_2020_tcells/log1p_tpm.npy')
 result['T-cell PolyA→Ribo']=x[m.library_prep.eq('ribo')].mean(0)-x[m.library_prep.eq('polyA')].mean(0)
 sm=pd.read_csv(R3/'sample_manifest.csv');ix=dict(zip(sm.sample_id,range(len(sm))));x=np.load(W3/'bridgerna_log1p_tpm_inputs.npy',mmap_mode='r')
 membership=pd.read_csv(R3/'task3b_contrast_sample_membership.csv')
 specs={'RR1 OSD-48 carcass':'C14__OSD-48__RR1-NASA__37-day','RR1 OSD-48 upon-euthanasia':'C13__OSD-48__RR1-NASA__37-day','RR3-39':'C01__OSD-137__RR3__39-day','RR3-40':'C02__OSD-137__RR3__40-day'}
 for name,cid in specs.items():
  q=membership[membership.contrast_id.eq(cid)];ids={c:q[q.condition.eq(c)].sample_id.map(ix).tolist() for c in ['FLT','GC']};result[name]=np.asarray(x[ids['FLT']]).mean(0)-np.asarray(x[ids['GC']]).mean(0)
 design=pd.read_csv(R3/'task3_osd168_technical_replication/technical_response_design.csv').set_index('representation');samples=design.loc['RR1_OSD168_no-ERCC','samples'].split(' | ');ids={c:[ix[s] for s in samples if condition(s)==c] for c in ['FLT','GC']};result['RR1 OSD-168 no-ERCC']=np.asarray(x[ids['FLT']]).mean(0)-np.asarray(x[ids['GC']]).mean(0)
 return result

def profiles(ctx,expr,B):
 all_rows=[];vectors={};signed_vectors={};latent={}
 coords_by_response={name:h@B.T/G for name,h in ctx.items()}
 reference=coords_by_response['T-cell PolyA→Ribo'].sum(0);reference/=max(np.linalg.norm(reference),1e-12)
 for response,h in ctx.items():
  coords=coords_by_response[response];mag=np.linalg.norm(coords,axis=1);z=coords.sum(0);signed=coords@reference
  er=expr[response];df=pd.DataFrame({'response':response,'gene_symbol':GENES,'PC1_contribution':coords[:,0],'PC2_contribution':coords[:,1],
   'PC1_2_magnitude':mag,'signed_contribution':signed,'contextual_rank':pd.Series(mag).rank(ascending=False,method='min').astype(int),
   'expression_change':er,'expression_change_rank':pd.Series(np.abs(er)).rank(ascending=False,method='min').astype(int)})
  all_rows.append(df);vectors[response]=coords.reshape(-1);signed_vectors[response]=signed;latent[response]=z
 out=pd.concat(all_rows,ignore_index=True);out.to_parquet(OUT/'full_gene_attribution_profiles.parquet',index=False);return out,vectors,signed_vectors,latent

def pairwise(vectors,signed_vectors,latent,expr):
 rows=[]
 for a,b in combinations(vectors,2):
  va,vb=vectors[a],vectors[b];sa_vec,sb_vec=signed_vectors[a],signed_vectors[b]
  ma=np.linalg.norm(va.reshape(G,2),axis=1);mb=np.linalg.norm(vb.reshape(G,2),axis=1)
  base={'response_A':a,'response_B':b,'latent_PC1_2_cosine':cos(latent[a],latent[b]),'attribution_cosine':cos(va,vb),
   'attribution_pearson':pearsonr(va,vb).statistic,'attribution_spearman':spearmanr(va,vb).statistic,
   'signed_rank_agreement':np.mean(np.sign(sa_vec)==np.sign(sb_vec)),'expression_cosine':cos(expr[a],expr[b]),
   'expression_pearson':pearsonr(expr[a],expr[b]).statistic,'expression_spearman':spearmanr(expr[a],expr[b]).statistic}
  for n in [100,250,500,1000]:
   sa=set(np.argpartition(ma,-n)[-n:]);sb=set(np.argpartition(mb,-n)[-n:]);k=len(sa&sb)
   overlap=list(sa&sb);base.update({f'top{n}_overlap':k,f'top{n}_jaccard':k/(2*n-k),f'top{n}_hypergeom_p':hypergeom.sf(k-1,G,n,n),
                f'top{n}_direction_agreement':np.mean(np.sign(sa_vec[overlap])==np.sign(sb_vec[overlap])) if k else np.nan})
  rows.append(base)
 d=pd.DataFrame(rows);d.to_csv(OUT/'pairwise_attribution_similarity.csv',index=False);return d

def pathways(profile):
 terms=[]
 for source,file in GMTS.items():
  for term,members in gp.parser.read_gmt(path=str(GMTROOT/file)).items():
   idx=np.flatnonzero(np.isin(GENES,list(set(members))))
   if 10<=len(idx)<=500:terms.append((source,term,idx))
 rows=[]
 for response,q in profile.groupby('response',sort=False):
  score=q.set_index('gene_symbol').loc[GENES,'signed_contribution'].to_numpy();sd=score.std() or 1;z=(score-score.mean())/sd
  for source,term,idx in terms:rows.append({'response':response,'source':source,'pathway':term,'pathway_attribution_z':z[idx].mean()*np.sqrt(len(idx)),'genes':len(idx)})
 d=pd.DataFrame(rows);d.to_parquet(OUT/'pathway_attribution_profiles.parquet',index=False)
 wide=d.pivot(index=['source','pathway'],columns='response',values='pathway_attribution_z');comp=[]
 for a,b in combinations(wide.columns,2):
  row={'response_A':a,'response_B':b,'pathway_profile_pearson':wide[a].corr(wide[b]),'pathway_profile_spearman':wide[a].corr(wide[b],method='spearman')}
  for n in [50,100,250]:
   sa=set(wide[a].abs().nlargest(n).index);sb=set(wide[b].abs().nlargest(n).index);ov=sa&sb
   row.update({f'top{n}_pathway_overlap':len(ov),f'top{n}_pathway_jaccard':len(ov)/len(sa|sb),
               f'top{n}_pathway_direction_agreement':np.mean([np.sign(wide.loc[i,a])==np.sign(wide.loc[i,b]) for i in ov]) if ov else np.nan})
  comp.append(row)
 pd.DataFrame(comp).to_csv(OUT/'pairwise_pathway_similarity.csv',index=False)
 patt={'RNA processing/splicing':r'RNA|SPLIC|RIBO|MRNA|RRNA','hepatic lipid/PPAR/peroxisome':r'FATTY|LIPID|PEROX|PPAR|BILE|CHOLESTEROL','mitochondrial/small-molecule':r'MITOCH|SMALL MOLECULE|CATABOL'}
 fam=[]
 for response,g in d.groupby('response'):
  for family,pat in patt.items():
   q=g[g.pathway.str.contains(pat,case=False,regex=True,na=False)].copy();q=q.loc[q.pathway_attribution_z.abs().nlargest(min(5,len(q))).index]
   fam.append({'response':response,'family':family,'mean_top5_signed_z':q.pathway_attribution_z.mean(),'max_absolute_z':q.pathway_attribution_z.abs().max(),'representative_terms':' | '.join(q.sort_values('pathway_attribution_z',key=abs,ascending=False).pathway.head(3))})
 pd.DataFrame(fam).to_csv(OUT/'program_family_attribution.csv',index=False);return d,pd.DataFrame(comp)

def figures(pair,profile,path):
 plt.style.use('seaborn-v0_8-whitegrid');FIG.mkdir(exist_ok=True)
 fig,ax=plt.subplots(figsize=(9,7));ax.scatter(pair.latent_PC1_2_cosine,pair.attribution_cosine,s=60,color='#4C78A8')
 important=pair[(pair.response_A.str.contains('T-cell'))|((pair.response_A.str.contains('RR1'))&(pair.response_B.str.contains('RR3')))|((pair.response_A.str.contains('OSD-48 carcass'))&(pair.response_B.str.contains('OSD-168')))]
 for _,r in important.iterrows():ax.annotate(f"{r.response_A.replace('RR1 ','').replace('T-cell ','TC ')}\n↔ {r.response_B.replace('RR1 ','').replace('RR3-','R3-')}",(r.latent_PC1_2_cosine,r.attribution_cosine),fontsize=7,xytext=(3,3),textcoords='offset points')
 ax.axhline(0,color='grey',lw=.8);ax.axvline(0,color='grey',lw=.8);ax.set(xlabel='Latent PC1–2 cosine',ylabel='Signed contextual-attribution cosine',title='Latent direction versus gene-level mechanism');fig.tight_layout()
 for e in ['png','pdf']:fig.savefig(FIG/f'latent_vs_attribution_similarity.{e}',dpi=400)
 plt.close(fig)
 names=['T-cell PolyA→Ribo','RR1 OSD-48 carcass','RR1 OSD-168 no-ERCC','RR3-39','RR3-40'];w=path[path.response.isin(names)].pivot(index=['source','pathway'],columns='response',values='pathway_attribution_z');sel=w.abs().max(axis=1).nlargest(30).index;w=w.loc[sel,names]
 fig,ax=plt.subplots(figsize=(10,10));im=ax.imshow(w,cmap='RdBu_r',vmin=-np.nanmax(abs(w.values)),vmax=np.nanmax(abs(w.values)),aspect='auto');ax.set_xticks(range(len(names)),[n.replace(' OSD-','\nOSD-') for n in names],rotation=25,ha='right');ax.set_yticks(range(len(w)),[i[1][:70] for i in w.index],fontsize=7);fig.colorbar(im,ax=ax,label='Pathway attribution z');ax.set_title('Distinct program combinations within shared PC1–2 directions');fig.tight_layout()
 for e in ['png','pdf']:fig.savefig(FIG/f'pathway_attribution_heatmap.{e}',dpi=400)
 plt.close(fig)
 # Central measured synthesis: geometry/mechanism scatter plus predefined family profiles.
 family=pd.read_csv(OUT/'program_family_attribution.csv');fw=family.pivot(index='family',columns='response',values='mean_top5_signed_z')[names]
 fig,(ax1,ax2)=plt.subplots(1,2,figsize=(16,6),gridspec_kw={'width_ratios':[1,1.35]})
 ax1.scatter(pair.latent_PC1_2_cosine,pair.attribution_cosine,s=65,color='#4C78A8')
 for _,r in important.iterrows():ax1.annotate(f"{r.response_A.replace('T-cell PolyA→Ribo','T-cell')} ↔\n{r.response_B}",(r.latent_PC1_2_cosine,r.attribution_cosine),fontsize=6,xytext=(3,3),textcoords='offset points')
 ax1.axhline(0,color='grey',lw=.8);ax1.axvline(0,color='grey',lw=.8);ax1.set(xlabel='PC1–2 latent cosine',ylabel='PC1/PC2 gene-attribution cosine',title='Same latent geometry can use different gene profiles')
 lim=np.nanmax(abs(fw.values));im=ax2.imshow(fw,cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto');ax2.set_xticks(range(len(names)),[n.replace(' OSD-','\nOSD-') for n in names],rotation=25,ha='right');ax2.set_yticks(range(len(fw)),fw.index);ax2.set_title('Measured pathway-family attribution');fig.colorbar(im,ax=ax2,label='Mean signed z, five strongest terms');fig.suptitle('PolyA/Ribo-sensitive PC1–2: latent similarity versus contextual mechanism',fontweight='bold');fig.tight_layout()
 for e in ['png','pdf']:fig.savefig(FIG/f'central_mechanism_comparison.{e}',dpi=400,bbox_inches='tight')
 plt.close(fig)

def main():
 OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(exist_ok=True);B=fixed_basis();ctx=contextual_responses();expr=expression_responses();prof,vectors,signed_vectors,latent=profiles(ctx,expr,B);pair=pairwise(vectors,signed_vectors,latent,expr);path,pathpair=pathways(prof);figures(pair,prof,path)
 focus=pair[((pair.response_A.eq('RR1 OSD-48 carcass'))&(pair.response_B.eq('RR1 OSD-168 no-ERCC')))|((pair.response_A.str.contains('T-cell')))|((pair.response_A.str.contains('RR1'))&(pair.response_B.str.contains('RR3')))].copy();focus.to_csv(OUT/'central_comparisons.csv',index=False)
 # Simple reference positioning, explicitly descriptive rather than a trained classifier.
 refs=['T-cell PolyA→Ribo','RR3-39','RR3-40'];rows=[]
 for query in ['RR1 OSD-48 carcass','RR1 OSD-48 upon-euthanasia','RR1 OSD-168 no-ERCC']:
  for ref in refs:
   q=pair[((pair.response_A.eq(ref))&(pair.response_B.eq(query)))|((pair.response_A.eq(query))&(pair.response_B.eq(ref)))].iloc[0]
   rows.append({'query':query,'reference':ref,'attribution_cosine':q.attribution_cosine,'attribution_spearman':q.attribution_spearman,'latent_PC1_2_cosine':q.latent_PC1_2_cosine})
 pd.DataFrame(rows).to_csv(OUT/'reference_positioning.csv',index=False)
 (OUT/'provenance.json').write_text(json.dumps({'basis':'unchanged uncentered controlled T-cell PC1-2','gene_universe':G,'contextual_profiles':'cached full 15165x512 response tensors','attribution_comparison':'flattened per-gene PC1 and PC2 contribution coordinates; preserves fixed basis orientation','signed_contribution':'projection onto the independently oriented mean T-cell PolyA-to-Ribo direction','pathway_statistic':'mean standardized signed gene contribution times sqrt(set size); descriptive, not enrichment significance','correction_performed':False},indent=2)+'\n')
 print(focus[['response_A','response_B','latent_PC1_2_cosine','attribution_cosine','attribution_spearman','expression_cosine','top500_overlap']].to_string(index=False));print('[complete]',OUT)
if __name__=='__main__':main()
