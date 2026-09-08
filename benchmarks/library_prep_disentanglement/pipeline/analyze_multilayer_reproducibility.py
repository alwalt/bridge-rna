#!/usr/bin/env python3
"""Multi-layer BridgeRNA response reproducibility benchmark from cached results."""
from __future__ import annotations
import json
from itertools import combinations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
import gseapy as gp

HERE=Path(__file__).resolve().parents[1];REPO=HERE.parents[1]
OUT=HERE/'results/task4_multilayer_reproducibility';FIG=OUT/'figures';SEED=20260908
GD=HERE/'results/task4_gene_attribution_diagnostic';T3=REPO/'benchmarks/osdr_batch_effect_representation';R3=T3/'results';W3=T3/'work'
GMTROOT=REPO/'benchmarks/cross_species_exercise_response/results/per_study_ranked_gsea';GMTS={'GO:BP':'GO_Biological_Process_2026.gmt','KEGG':'KEGG_2026.gmt','Reactome':'Reactome_Pathways_2024.gmt'}
NAMES=['RR1_original','RR1_remeasurement','RR3_39_original','RR3_39_remeasurement','RR3_40_original','RR3_40_remeasurement']
KEYS={'RR1_original':'RR1_OSD48_original_matched','RR1_remeasurement':'RR1_OSD168_no-ERCC','RR3_39_original':'C01_OSD137_original_matched','RR3_39_remeasurement':'C01_OSD168_all_ERCC','RR3_40_original':'C02_OSD137_original_matched','RR3_40_remeasurement':'C02_OSD168_all_ERCC'}
POS={frozenset(['RR1_original','RR1_remeasurement']):'A_same_biological_material_technical_remeasurement',frozenset(['RR3_39_original','RR3_39_remeasurement']):'A_same_biological_material_technical_remeasurement',frozenset(['RR3_40_original','RR3_40_remeasurement']):'A_same_biological_material_technical_remeasurement'}

def cos(a,b):
 d=np.linalg.norm(a)*np.linalg.norm(b);return float(a@b/d) if d else np.nan
def responses():
 m=pd.read_csv(R3/'sample_manifest.csv');mem=pd.read_csv(R3/'task3b_contrast_sample_membership.csv');z=np.load(W3/'bridgerna_embeddings.npy');ix=dict(zip(m.sample_id,range(len(m))))
 specs={'RR1_original':('C14__OSD-48__RR1-NASA__37-day',None),'RR1_remeasurement':('C04__OSD-168__RR1-NASA__37-day',None),'RR3_39_original':('C01__OSD-137__RR3__39-day',None),'RR3_39_remeasurement':('C05__OSD-168__RR3__39-day',None),'RR3_40_original':('C02__OSD-137__RR3__40-day','strict'),'RR3_40_remeasurement':('C06__OSD-168__RR3__40-day',None)}
 out={};meta=[]
 for name,(cid,strict) in specs.items():
  q=mem[mem.contrast_id.eq(cid)].copy()
  if name=='RR1_original':q=q[~q.sample_id.str.endswith('_M27')]
  if name=='RR1_remeasurement':q=q[~q.sample_id.str.endswith('_M29')]
  if strict:q=q[~q.sample_id.str.endswith('_F5')]
  ids={c:q[q.condition.eq(c)].sample_id.map(ix).tolist() for c in ['FLT','GC']};out[name]=z[ids['FLT']].mean(0)-z[ids['GC']].mean(0)
  mm=m[m.sample_id.isin(q.sample_id)];meta.append({'response':name,'study':'OSD-48' if name=='RR1_original' else 'OSD-137' if name.endswith('original') else 'OSD-168','species':'mouse','tissue':'liver','biological_contrast':'spaceflight FLT minus GC','condition_A':'FLT','condition_B':'GC','n_A':len(ids['FLT']),'n_B':len(ids['GC']),'samples_identical_or_matched':'matched same animals' if 'remeasurement' in name or 'original' in name else 'NA','library_preparation':' | '.join(sorted(mm.library_preparation.unique())),'technical_context':' | '.join(sorted(mm.sequencing_parameters.unique()))})
 return out,pd.DataFrame(meta)

def load_profiles():
 a=np.load(GD/'signed_gene_attributions.npz');x=np.load(GD/'expression_response_vectors.npz');genes=a['gene_symbol'].astype(str)
 attrs={n:a[KEYS[n]].astype(float) for n in NAMES};expr={n:x[KEYS[n]].astype(float) for n in NAMES};return genes,attrs,expr

def pathway_profiles(genes,attrs):
 terms=[]
 for source,file in GMTS.items():
  for term,members in gp.parser.read_gmt(path=str(GMTROOT/file)).items():
   ii=np.flatnonzero(np.isin(genes,list(set(members))))
   if 10<=len(ii)<=500:terms.append((source,term,ii))
 rows=[]
 for name,v in attrs.items():
  zz=(v-v.mean())/(v.std() or 1)
  for source,term,ii in terms:rows.append({'response':name,'source':source,'pathway':term,'score':zz[ii].mean()*np.sqrt(len(ii))})
 d=pd.DataFrame(rows);d.to_parquet(OUT/'pathway_profiles.parquet',index=False);return d

def pairs(resp,attrs,expr,path,meta):
 w=path.pivot(index=['source','pathway'],columns='response',values='score');rows=[]
 for a,b in combinations(NAMES,2):
  va,vb=resp[a],resp[b];aa,ab=attrs[a],attrs[b];ea,eb=expr[a],expr[b];true=frozenset([a,b]) in POS
  row={'response_A':a,'response_B':b,'is_replication':true,'replication_class':POS.get(frozenset([a,b]),'unrelated_null'),'negative_category':'NA' if true else ('high_latent_unrelated_candidate' if cos(va,vb)>=.75 else 'same_tissue_species_unrelated'),'latent_cosine':cos(va,vb),'latent_distance':np.linalg.norm(va-vb),'response_norm_A':np.linalg.norm(va),'response_norm_B':np.linalg.norm(vb),'response_magnitude_ratio':min(np.linalg.norm(va),np.linalg.norm(vb))/max(np.linalg.norm(va),np.linalg.norm(vb)),'attribution_cosine':cos(aa,ab),'attribution_spearman':spearmanr(aa,ab).statistic,'expression_pearson':np.corrcoef(ea,eb)[0,1],'expression_spearman':spearmanr(ea,eb).statistic,'pathway_pearson':w[a].corr(w[b]),'pathway_spearman':w[a].corr(w[b],method='spearman')}
  for n in [100,500]:
   sa=set(np.argpartition(np.abs(aa),-n)[-n:]);sb=set(np.argpartition(np.abs(ab),-n)[-n:]);ov=list(sa&sb);row[f'top{n}_overlap']=len(ov);row[f'top{n}_jaccard']=len(ov)/len(sa|sb);row[f'top{n}_signed_agreement']=np.mean(np.sign(aa[ov])==np.sign(ab[ov])) if ov else np.nan
   pa=set(w[a].abs().nlargest(n if n<=len(w) else len(w)).index);pb=set(w[b].abs().nlargest(n if n<=len(w) else len(w)).index);pov=pa&pb;row[f'top{n}_pathway_overlap']=len(pov);row[f'top{n}_pathway_direction_agreement']=np.mean([np.sign(w.loc[i,a])==np.sign(w.loc[i,b]) for i in pov]) if pov else np.nan
  rows.append(row)
 out=pd.DataFrame(rows);out['latent_rank_percentile']=out.latent_cosine.rank(pct=True);out['attribution_rank_percentile']=out.attribution_cosine.rank(pct=True);out['pathway_rank_percentile']=out.pathway_pearson.rank(pct=True);out['expression_rank_percentile']=out.expression_spearman.rank(pct=True)
 out.to_csv(OUT/'pairwise_benchmark.csv',index=False);out[out.is_replication].to_csv(OUT/'replication_pairs.csv',index=False);out[~out.is_replication].to_csv(OUT/'null_pairs.csv',index=False);return out

def scores(d):
 y=d.is_replication.astype(int).to_numpy();models={'geometry_only':['latent_cosine'],'attribution_only':['attribution_cosine','attribution_spearman','top500_jaccard'],'pathway_only':['pathway_pearson','pathway_spearman'],'geometry_plus_attribution':['latent_cosine','attribution_cosine','attribution_spearman','top500_jaccard'],'geometry_plus_attribution_plus_pathway':['latent_cosine','attribution_cosine','attribution_spearman','top500_jaccard','pathway_pearson','pathway_spearman'],'expression_only':['expression_pearson','expression_spearman']}
 result=[];rng=np.random.default_rng(SEED)
 for name,features in models.items():
  # Rank-average composite prevents scale dominance and avoids fitted weights.
  s=np.mean([d[f].rank(pct=True).to_numpy() for f in features],axis=0);auc=roc_auc_score(y,s);ap=average_precision_score(y,s);boots=[]
  pos=np.flatnonzero(y);neg=np.flatnonzero(~y)
  for _ in range(5000):
   jj=np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)];boots.append(roc_auc_score(y[jj],s[jj]))
  loo=[]
  for p in pos:
   jj=np.flatnonzero(np.arange(len(y))!=p);loo.append(roc_auc_score(y[jj],s[jj]))
  result.append({'model':name,'features':' | '.join(features),'ROC_AUC':auc,'PR_AUC':ap,'bootstrap_low':np.quantile(boots,.025),'bootstrap_high':np.quantile(boots,.975),'leave_one_positive_out_min':min(loo),'leave_one_positive_out_max':max(loo)})
 out=pd.DataFrame(result);out.to_csv(OUT/'composite_score_performance.csv',index=False);return out

def robustness():
 src=HERE/'results/task4_rr1_rr3_paired_technical_replication';b=pd.read_csv(src/'bootstrap_summary.csv');l=pd.read_csv(src/'leave_one_animal_out.csv');b.to_csv(OUT/'latent_bootstrap_summary_reused.csv',index=False);l.to_csv(OUT/'latent_leave_one_out_reused.csv',index=False)

def figures(d,score):
 plt.style.use('seaborn-v0_8-whitegrid');FIG.mkdir(parents=True,exist_ok=True);colors=np.where(d.is_replication,'#E45756','#4C78A8')
 fig,axes=plt.subplots(1,3,figsize=(17,5))
 for ax,col,title in zip(axes,['latent_cosine','attribution_cosine','pathway_pearson'],['Latent geometry','Gene attribution','Pathway profile']):
  vals=[d[d.is_replication][col],d[~d.is_replication][col]];ax.boxplot(vals,labels=['Replication','Unrelated'],showfliers=False);ax.scatter(np.repeat([1,2],[len(vals[0]),len(vals[1])]),np.r_[vals[0],vals[1]],alpha=.75);ax.set(ylabel=col,title=title)
 fig.tight_layout();
 for e in ['png','pdf']:fig.savefig(FIG/f'layer_distributions.{e}',dpi=400);plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(18,5));pairs=[('latent_cosine','attribution_cosine'),('latent_cosine','pathway_pearson'),('attribution_cosine','pathway_pearson')]
 for ax,(x,y) in zip(axes,pairs):
  ax.scatter(d[x],d[y],c=colors,s=75)
  for _,r in d.iterrows():
   if r.is_replication or r.latent_cosine>=.75:ax.annotate(f"{r.response_A.replace('_',' ')} ↔\n{r.response_B.replace('_',' ')}",(r[x],r[y]),fontsize=6)
  ax.set(xlabel=x,ylabel=y)
 fig.suptitle('Geometry, attribution, and pathways provide distinct evidence');fig.tight_layout()
 for e in ['png','pdf']:fig.savefig(FIG/f'layer_relationships.{e}',dpi=400,bbox_inches='tight');plt.close(fig)
 fig,ax=plt.subplots(figsize=(11,6));ax.barh(score.model,score.ROC_AUC,xerr=[score.ROC_AUC-score.bootstrap_low,score.bootstrap_high-score.ROC_AUC],color='#4C78A8');ax.axvline(.5,color='grey',ls='--');ax.set(xlim=(0,1.05),xlabel='ROC AUC (bootstrap 95% interval)',title='Descriptive discrimination of three technical replications');fig.tight_layout()
 for e in ['png','pdf']:fig.savefig(FIG/f'composite_performance.{e}',dpi=400,bbox_inches='tight');plt.close(fig)

def main():
 OUT.mkdir(parents=True,exist_ok=True);resp,meta=responses();meta.to_csv(OUT/'response_metadata.csv',index=False);genes,attrs,expr=load_profiles();path=pathway_profiles(genes,attrs);d=pairs(resp,attrs,expr,path,meta);score=scores(d);robustness();figures(d,score)
 false=d[(~d.is_replication)&(d.latent_cosine>=d[~d.is_replication].latent_cosine.quantile(.75))].sort_values('latent_cosine',ascending=False);false.to_csv(OUT/'geometric_false_friends.csv',index=False)
 summary={'responses':len(resp),'replication_pairs':int(d.is_replication.sum()),'replication_categories':{'A_same_biological_material_technical_remeasurement':int(d.is_replication.sum())},'independent_biological_replications':0,'null_pairs':int((~d.is_replication).sum()),'high_latent_unrelated_pairs':int(((~d.is_replication)&(d.latent_cosine>=.75)).sum()),'limitation':'Only strict same-animal technical remeasurements have response-specific full-space IG under one implementation. This benchmark evaluates measurement robustness, not independent-cohort biological generalization.'}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(OUT/'provenance.json').write_text(json.dumps({'embeddings_recomputed':False,'attribution_recomputed':False,'attribution_source':'task4_gene_attribution_diagnostic signed full-response IG','pathway_method':'same standardized signed-attribution gene-set profile across all responses','composite':'unweighted mean percentile rank; no fitted classifier','positive_definition':'three exact same-animal technical remeasurements','null_definition':'all cross-response pairs among same-species same-tissue profiles','claim_scope':'technical measurement reproducibility only'},indent=2)+'\n');print(d.sort_values('latent_cosine',ascending=False).to_string(index=False));print('\nScores\n',score.to_string(index=False));print('\n',json.dumps(summary,indent=2));print('[complete]',OUT)
if __name__=='__main__':main()
