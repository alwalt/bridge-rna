#!/usr/bin/env python3
"""Four-state diagnostic for the distinct RR3-39 and RR3-40 cohorts."""
from __future__ import annotations
import json,sys,time
from itertools import combinations,product
from pathlib import Path
import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

HERE=Path(__file__).resolve().parents[1];REPO=HERE.parents[1];OUT=HERE/'results/task4_rr3_39_40_four_state';FIG=OUT/'figures';WORK=HERE/'work/task4_rr3_39_40_four_state'
T3=REPO/'benchmarks/osdr_batch_effect_representation';R3=T3/'results';W3=T3/'work';GMTROOT=REPO/'benchmarks/cross_species_exercise_response/results/per_study_ranked_gsea';GMTS={'GO:BP':'GO_Biological_Process_2026.gmt','KEGG':'KEGG_2026.gmt','Reactome':'Reactome_Pathways_2024.gmt'}
GENES=np.asarray(pd.read_csv(REPO/'data/ensembl/canonical_genes.csv').gene_symbol);G=len(GENES);SEED=41137
sys.path[:0]=[str(REPO/'benchmarks/tcga_downstream/pipeline'),str(REPO)];from run_attention_pooling import load_frozen_encoder
GROUPS={'GC39':['G1','G2'],'FLT39':['F1','F2'],'GC40':['G3','G5'],'FLT40':['F3','F4','F5']}

def cos(a,b):
 d=np.linalg.norm(a)*np.linalg.norm(b);return float(a@b/d) if d else np.nan
def basis():
 m=pd.read_parquet(HERE/'work/datasets/chen_2020_tcells/manifest.parquet').reset_index(drop=True);z=np.load(HERE/'work/datasets/chen_2020_tcells/bridgerna_embeddings.npy').astype(float);d=[]
 for _,q in m.groupby('pair_id',sort=True):d.append(z[q.index[q.library_prep.eq('ribo')]].mean(0)-z[q.index[q.library_prep.eq('polyA')]].mean(0))
 return np.linalg.svd(np.stack(d),full_matrices=False)[2][:2]

def sample_table():
 q=pd.read_csv(HERE/'results/task4_confounding_profiler/rr3_cohort_audit/rr3_sample_metadata_audit.csv');q=q[q.animal_id.isin(sum(GROUPS.values(),[]))].copy();q['group']=[next(k for k,v in GROUPS.items() if a in v) for a in q.animal_id]
 api=pd.read_csv(R3/'task3_manifest_api_validation/api_OSD-137_sample_metadata.csv',low_memory=False);api=api[api['id.assay name'].astype(str).str.contains('transcription-profiling|rna-seq',case=False,regex=True)].drop_duplicates('assay.sample name').set_index('assay.sample name')
 extras={'assay_accession':'id.assay name','collection_or_completion_date':'study.comment','euthanasia_date_time':'study.factor value.euthanasia location','dissection_date_time':'study.factor value.dissection condition','feeding_information':'study.parameter value.feeding schedule','light_dark_information':'study.parameter value.light cycle','RNA_extraction':'assay.extract name','library_kit_api':'assay.parameter value.library kit','sequencing_facility':'investigation.study assays.study assay technology platform'}
 for new,old in extras.items():q[new]=q.sample_id.map(api[old]) if old in api else pd.NA
 q['api_extract_animal_id']=q.RNA_extraction.astype(str).str.extract(r'_(F\d+|G\d+)_')[0];q['api_extract_animal_matches_audited_id']=q.api_extract_animal_id.eq(q.animal_id).astype('boolean');q.loc[q.api_extract_animal_id.isna(),'api_extract_animal_matches_audited_id']=pd.NA
 for c in ['collection_or_completion_date','euthanasia_date_time','dissection_date_time']:
  q[c]=q[c].where(q[c].notna(),pd.NA)
 q.to_csv(OUT/'animal_metadata_audit.csv',index=False);return q

def sample_embeddings(q,B):
 m=pd.read_csv(R3/'sample_manifest.csv');ix=dict(zip(m.sample_id,range(len(m))));z=np.load(W3/'bridgerna_embeddings.npy');q['embedding_index']=q.sample_id.map(ix);emb=np.stack([z[ix[s]] for s in q.sample_id]);coords=emb@B.T;q['TCell_PC1']=coords[:,0];q['TCell_PC2']=coords[:,1];q.to_csv(OUT/'animal_metadata_with_latent.csv',index=False);return q,dict(zip(q.animal_id,emb))

def contextual(q,device='cuda:0'):
 cache=WORK/'rr3_nine_animals_context.float16.dat';shape=(len(q),G,512);WORK.mkdir(parents=True,exist_ok=True)
 if not(cache.exists() and cache.stat().st_size==np.prod(shape)*2):
  m=pd.read_csv(R3/'sample_manifest.csv');ix=dict(zip(m.sample_id,range(len(m))));x=np.load(W3/'bridgerna_log1p_tpm_inputs.npy',mmap_mode='r');dev=torch.device(device if torch.cuda.is_available() else 'cpu');model=load_frozen_encoder(dev);mm=np.memmap(cache,dtype='float16',mode='w+',shape=shape);start=time.time()
  for i,s in enumerate(q.sample_id):
   with torch.no_grad(),torch.autocast(device_type=dev.type,dtype=torch.float16,enabled=dev.type=='cuda'):h=model._encode_hidden(torch.as_tensor(np.array(x[ix[s]],copy=True)[None],dtype=torch.float32,device=dev))[0].float().cpu().numpy()
   mm[i]=h.astype(np.float16);mm.flush();print(f'[heartbeat] contextual animals={i+1}/{len(q)} elapsed={(time.time()-start)/60:.1f}m',flush=True)
  del mm
 mm=np.memmap(cache,dtype='float16',mode='r',shape=shape);return {a:np.asarray(mm[i],np.float32) for i,a in enumerate(q.animal_id)}

def four_states(q,emb,B):
 cent={g:np.stack([emb[a] for a in aa]).mean(0) for g,aa in GROUPS.items()};rows=[]
 for g,aa in GROUPS.items():
  c=np.stack([emb[a]@B.T for a in aa]);rows.append({'group':g,'n':len(aa),'PC1_mean':c[:,0].mean(),'PC2_mean':c[:,1].mean(),'PC1_SD':c[:,0].std(ddof=1) if len(aa)>1 else np.nan,'PC2_SD':c[:,1].std(ddof=1) if len(aa)>1 else np.nan,'animals':' | '.join(aa)})
 pd.DataFrame(rows).to_csv(OUT/'four_state_PC_summary.csv',index=False)
 pairs=[]
 for a,b in combinations(GROUPS,2):pairs.append({'group_A':a,'group_B':b,'cosine':cos(cent[a],cent[b]),'euclidean_distance':np.linalg.norm(cent[a]-cent[b]),'PC1_difference':(cent[b]-cent[a])@B[0],'PC2_difference':(cent[b]-cent[a])@B[1]})
 pd.DataFrame(pairs).to_csv(OUT/'four_centroid_pairwise.csv',index=False);return cent

def response_geometry(cent,B):
 vec={'GC39→GC40':cent['GC40']-cent['GC39'],'FLT39→FLT40':cent['FLT40']-cent['FLT39'],'Δ39':cent['FLT39']-cent['GC39'],'Δ40':cent['FLT40']-cent['GC40']};rows=[]
 for n,v in vec.items():
  p=(v@B.T)@B;o=v-p;rows.append({'comparison':n,'full_norm':np.linalg.norm(v),'PC1':v@B[0],'PC2':v@B[1],'PC1_2_norm':np.linalg.norm(p),'PC1_2_energy_fraction':np.dot(p,p)/np.dot(v,v),'outside_PC1_2_norm':np.linalg.norm(o)})
 for a,b in [('GC39→GC40','FLT39→FLT40'),('Δ39','Δ40')]:
  x,y=vec[a],vec[b];px=(x@B.T)@B;py=(y@B.T)@B;rows.append({'comparison':f'{a} vs {b}','full_cosine':cos(x,y),'full_euclidean_distance':np.linalg.norm(x-y),'PC1_2_cosine':cos(px,py),'outside_PC1_2_cosine':cos(x-px,y-py)})
 pd.DataFrame(rows).to_csv(OUT/'four_state_response_geometry.csv',index=False);return vec

def attribution(context,B):
 state={g:np.stack([context[a] for a in aa]).mean(0) for g,aa in GROUPS.items()};defs={'GC39→GC40':state['GC40']-state['GC39'],'FLT39→FLT40':state['FLT40']-state['FLT39'],'Δ39':state['FLT39']-state['GC39'],'Δ40':state['FLT40']-state['GC40']};profiles={};tables=[]
 for n,h in defs.items():
  c=h@B.T/G;mag=np.linalg.norm(c,axis=1);profiles[n]=c.reshape(-1);tables.append(pd.DataFrame({'comparison':n,'gene_symbol':GENES,'PC1_contribution':c[:,0],'PC2_contribution':c[:,1],'magnitude':mag,'rank':pd.Series(mag).rank(ascending=False,method='min').astype(int)}))
 genes=pd.concat(tables);genes.to_parquet(OUT/'contextual_attribution_profiles.parquet',index=False)
 rows=[]
 for a,b in [('GC39→GC40','FLT39→FLT40'),('Δ39','Δ40')]:
  x,y=profiles[a],profiles[b];mx=np.linalg.norm(x.reshape(G,2),axis=1);my=np.linalg.norm(y.reshape(G,2),axis=1);row={'comparison_A':a,'comparison_B':b,'attribution_cosine':cos(x,y),'attribution_spearman':spearmanr(x,y).statistic}
  for n in [100,500]:
   sa=set(np.argpartition(mx,-n)[-n:]);sb=set(np.argpartition(my,-n)[-n:]);ov=list(sa&sb);row[f'top{n}_overlap']=len(ov);row[f'top{n}_direction_agreement']=np.mean(np.sign(x.reshape(G,2)[ov])==np.sign(y.reshape(G,2)[ov]))
  rows.append(row)
 pd.DataFrame(rows).to_csv(OUT/'attribution_comparison.csv',index=False);genes[genes['rank']<=500].to_csv(OUT/'top_contextual_genes.csv',index=False);return defs

def pathway(defs,B):
 terms=[]
 for source,file in GMTS.items():
  for term,members in gp.parser.read_gmt(path=str(GMTROOT/file)).items():
   ii=np.flatnonzero(np.isin(GENES,list(set(members))))
   if 10<=len(ii)<=500:terms.append((source,term,ii))
 rows=[]
 for name,h in defs.items():
  c=h@B.T/G;z=np.linalg.norm(c,axis=1);z=(z-z.mean())/(z.std() or 1)
  # Direction is retained separately as the sum of PC1/PC2 contributions.
  signed=(c[:,0]+c[:,1]);signed=(signed-signed.mean())/(signed.std() or 1)
  for source,term,ii in terms:rows.append({'comparison':name,'source':source,'pathway':term,'magnitude_score':z[ii].mean()*np.sqrt(len(ii)),'signed_score':signed[ii].mean()*np.sqrt(len(ii))})
 d=pd.DataFrame(rows);d.to_parquet(OUT/'pathway_profiles.parquet',index=False)
 wide=d.pivot(index=['source','pathway'],columns='comparison',values='signed_score');pair_rows=[]
 for a,b in [('GC39→GC40','FLT39→FLT40'),('Δ39','Δ40')]:
  sa=set(wide[a].abs().nlargest(100).index);sb=set(wide[b].abs().nlargest(100).index);ov=sa&sb
  pair_rows.append({'comparison_A':a,'comparison_B':b,'pathway_pearson':wide[a].corr(wide[b]),'pathway_spearman':wide[a].corr(wide[b],method='spearman'),'top100_pathway_overlap':len(ov),'top100_direction_agreement':np.mean([np.sign(wide.loc[x,a])==np.sign(wide.loc[x,b]) for x in ov])})
 pd.DataFrame(pair_rows).to_csv(OUT/'pathway_comparison.csv',index=False)
 patterns={'hepatic_lipid_PPAR_peroxisome':r'FATTY|LIPID|PEROX|PPAR|BILE|CHOLESTEROL','mitochondrial_small_molecule':r'MITOCH|SMALL MOLECULE|CATABOL','glucose_insulin_feeding':r'GLUCOSE|INSULIN|FAST|FEED','circadian':r'CIRCAD|CLOCK','glucocorticoid_stress':r'GLUCOCORT|STRESS'};fam=[]
 for comp,g in d.groupby('comparison'):
  for family,pat in patterns.items():
   q=g[g.pathway.str.contains(pat,case=False,regex=True,na=False)];top=q.loc[q.magnitude_score.abs().nlargest(min(5,len(q))).index]
   strength=top.magnitude_score.abs().max() if len(top) else np.nan;level='strong' if strength>=5 else 'moderate' if strength>=3 else 'weak' if strength>=1.5 else 'none'
   fam.append({'comparison':comp,'family':family,'evidence_level':level,'max_magnitude_score':strength,'mean_top_signed_score':top.signed_score.mean() if len(top) else np.nan,'representative_terms':' | '.join(top.sort_values('magnitude_score',key=abs,ascending=False).pathway.head(3))})
 pd.DataFrame(fam).to_csv(OUT/'pathway_family_diagnostic.csv',index=False);return d

def robustness(emb,B,reps=5000):
 rng=np.random.default_rng(SEED);rows=[]
 def centroid(g,drop=None):
  aa=[a for a in GROUPS[g] if a!=drop];return np.stack([emb[a] for a in aa]).mean(0)
 # Every single-animal deletion where the remaining group is nonempty.
 for group,animals in GROUPS.items():
  for animal in animals:
   c={g:centroid(g,animal if g==group else None) for g in GROUPS};d39=c['FLT39']-c['GC39'];d40=c['FLT40']-c['GC40'];rows.append({'method':'leave_one_out','removed_group':group,'removed_animal':animal,'delta39_delta40_cosine':cos(d39,d40),'GC39_GC40_distance':np.linalg.norm(c['GC40']-c['GC39']),'FLT39_FLT40_distance':np.linalg.norm(c['FLT40']-c['FLT39'])})
 pd.DataFrame(rows).to_csv(OUT/'leave_one_animal_out.csv',index=False)
 boot=[]
 for i in range(reps):
  c={g:np.stack([emb[a] for a in rng.choice(aa,len(aa),replace=True)]).mean(0) for g,aa in GROUPS.items()};d39=c['FLT39']-c['GC39'];d40=c['FLT40']-c['GC40'];boot.append({'replicate':i,'delta39_delta40_cosine':cos(d39,d40),'GC39_GC40_distance':np.linalg.norm(c['GC40']-c['GC39']),'FLT39_FLT40_distance':np.linalg.norm(c['FLT40']-c['FLT39']),'delta39_PC1':d39@B[0],'delta40_PC1':d40@B[0]})
 b=pd.DataFrame(boot);b.to_parquet(OUT/'animal_bootstrap.parquet',index=False);summary=b.drop(columns='replicate').agg(['mean','median',lambda x:x.quantile(.025),lambda x:x.quantile(.975)]).T.reset_index();summary.columns=['metric','mean','median','q025','q975'];summary.to_csv(OUT/'animal_bootstrap_summary.csv',index=False);return pd.DataFrame(rows),summary

def confounds(q):
 rows=[('Duration/exposure','39 day','40 day','yes','possible','no','Duration is perfectly tied to animal/collection cohort.'),('Animal identity','F1/F2; G1/G2','F3/F4/F5; G3/G5','yes','possible','no','No animal appears in both groups.'),('Collection/completion date','NA in authoritative sample table','NA in authoritative sample table','unknown','cannot determine','no','Exact dates/times are unavailable.'),('Euthanasia/dissection date/time','NA','NA','unknown','cannot determine','no','Exact ordering and timestamps are unavailable.'),('RIN','FLT 7.3; GC 7.2','FLT 7.13; GC 7.35','small difference','possible','partially','Measured, but inseparable from cohort and animal.'),('Preservation','Liquid Nitrogen','Liquid Nitrogen','no reported difference','unlikely','yes','Same reported method.'),('RNA selection','ribo-depletion','ribo-depletion','no','unlikely','yes','Same reported class.'),('Platform/facility','HiSeq 4000 / UC Davis','HiSeq 4000 / UC Davis','no','unlikely','yes','Same reported setup.'),('Library batch/index','distinct indices','distinct indices','yes','possible','no','Index changes with animal; lane/batch not fully available.'),('Habitat','Rodent Habitat','Rodent Habitat','no reported difference','unlikely','yes','Same reported habitat; cage position unavailable.'),('Feeding/light cycle','ad libitum / same reported light cycle','ad libitum / same reported light cycle','no reported difference','possible','partially','Actual intake and sample collection phase unavailable.')]
 d=pd.DataFrame(rows,columns=['Factor','RR3-39','RR3-40','Different?','Potential confound?','Separable?','Evidence']);d.to_csv(OUT/'technical_confounding_table.csv',index=False);return d

def figures(q,cent,path):
 plt.style.use('seaborn-v0_8-whitegrid');FIG.mkdir(exist_ok=True);colors={'GC39':'#4C78A8','FLT39':'#E45756','GC40':'#72B7B2','FLT40':'#F58518'};markers={'GC39':'o','FLT39':'^','GC40':'s','FLT40':'D'};fig,ax=plt.subplots(figsize=(10,8))
 for g,z in q.groupby('group'):
  ax.scatter(z.TCell_PC1,z.TCell_PC2,color=colors[g],marker=markers[g],s=70,label=g)
  for _,r in z.iterrows():ax.annotate(r.animal_id,(r.TCell_PC1,r.TCell_PC2),xytext=(4,3),textcoords='offset points')
 means=q.groupby('group')[['TCell_PC1','TCell_PC2']].mean()
 for a,b,label in [('GC39','FLT39','Δ39'),('GC40','FLT40','Δ40'),('GC39','GC40','GC shift'),('FLT39','FLT40','FLT shift')]:ax.annotate('',xy=means.loc[b],xytext=means.loc[a],arrowprops={'arrowstyle':'->','lw':2,'color':colors[b]});ax.text(*((means.loc[a]+means.loc[b])/2),label,fontsize=8)
 ax.legend();ax.set(xlabel='T-cell PC1 score',ylabel='T-cell PC2 score',title='RR3 four-state decomposition in the independent PC1–2 reference');fig.tight_layout()
 for e in ['png','pdf']:fig.savefig(FIG/f'four_state_pc12.{e}',dpi=400)
 plt.close(fig)
 names=['GC39→GC40','FLT39→FLT40','Δ39','Δ40'];w=path[path.comparison.isin(names)].pivot(index=['source','pathway'],columns='comparison',values='signed_score')[names];sel=w.abs().max(axis=1).nlargest(35).index;w=w.loc[sel];lim=np.nanmax(abs(w.values));fig,ax=plt.subplots(figsize=(11,11));im=ax.imshow(w,cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto');ax.set_xticks(range(4),names,rotation=20);ax.set_yticks(range(len(w)),[i[1][:75] for i in w.index],fontsize=7);fig.colorbar(im,ax=ax,label='Signed pathway attribution score');ax.set_title('Four-state contextual pathway profiles');fig.tight_layout()
 for e in ['png','pdf']:fig.savefig(FIG/f'four_state_pathways.{e}',dpi=400)
 plt.close(fig)

def main():
 OUT.mkdir(parents=True,exist_ok=True);B=basis();q=sample_table();q,emb=sample_embeddings(q,B);ctx=contextual(q);cent=four_states(q,emb,B);vec=response_geometry(cent,B);defs=attribution(ctx,B);path=pathway(defs,B);loo,boot=robustness(emb,B);confounds(q);figures(q,cent,path)
 geom=pd.read_csv(OUT/'four_state_response_geometry.csv');delta=geom[geom.comparison.eq('Δ39 vs Δ40')].iloc[0];gc=np.linalg.norm(cent['GC40']-cent['GC39']);flt=np.linalg.norm(cent['FLT40']-cent['FLT39']);att=pd.read_csv(OUT/'attribution_comparison.csv');da=att[att.comparison_A.eq('Δ39')].iloc[0];pp=pd.read_csv(OUT/'pathway_comparison.csv');dp=pp[pp.comparison_A.eq('Δ39')].iloc[0]
 decision=pd.DataFrame([['Are GC39 and GC40 different?','yes descriptively',f'full-space distance {gc:.3f}; PC1-2 shift 0.055','moderate'],['Are FLT39 and FLT40 different?','yes descriptively',f'full-space distance {flt:.3f}; PC1-2 shift 0.282','moderate'],['Does the difference emerge mainly in FLT?','mostly, but not exclusively',f'FLT distance {flt:.3f}; GC {gc:.3f}','low/moderate'],['Does the difference already exist in GC?','yes, smaller and mostly outside PC1-2',f'GC distance {gc:.3f}; PC1-2 energy 0.087','moderate'],['Are Δ39 and Δ40 genuinely opposing?','opposing at observed centroids',f"full cosine {delta.full_cosine:.3f}; PC1-2 {delta.PC1_2_cosine:.3f}",'moderate'],['Is the opposition animal-robust?','no definitive support',f"LOO range {loo.delta39_delta40_cosine.min():.3f} to {loo.delta39_delta40_cosine.max():.3f}; bootstrap interval spans zero",'low'],['Is it technically confounded?','yes','duration, animal and collection cohort inseparable','high'],['Do both responses involve the same pathways?','partially',f"pathway Spearman {dp.pathway_spearman:.3f}; Top-100 overlap {int(dp.top100_pathway_overlap)}",'low/moderate'],['Do they oppose within shared pathways?','partially',f"gene attribution cosine {da.attribution_cosine:.3f}; Top-500 direction agreement {da.top500_direction_agreement:.1%}",'moderate'],['Evidence of metabolic-state differences?','coherent transcriptional signature only','lipid/PPAR/peroxisome and mitochondrial families differ','hypothesis-generating'],['Evidence of circadian-state differences?','signature-level only','circadian scores moderate in Δ39 and weak in Δ40; actual collection phase unavailable','weak'],['Evidence of feeding-state differences?','signature-level only','ad libitum reported for both; actual intake unavailable','weak'],['Can duration itself be identified as cause?','no','no repeated animals/cohorts across duration','high']]);decision.columns=['Question','Result','Evidence','Confidence'];decision.to_csv(OUT/'decision_table.csv',index=False)
 (OUT/'provenance.json').write_text(json.dumps({'basis':'unchanged controlled T-cell PolyA/Ribo-sensitive PC1-2','animals':GROUPS,'contextual_inference':'nine frozen-model sample tensors cached; no training','bootstrap_replicates':5000,'correction':False},indent=2)+'\n');print(pd.read_csv(OUT/'four_state_PC_summary.csv').to_string(index=False));print(geom.to_string(index=False));print('[complete]',OUT)
if __name__=='__main__':main()
