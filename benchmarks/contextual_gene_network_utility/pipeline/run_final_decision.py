#!/usr/bin/env python3
"""Final contextual-network decision analyses, excluding new model inference."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import hypergeom, spearmanr

HERE=Path(__file__).resolve().parents[1];REPO=HERE.parents[1];OUT=HERE/'results/final_decision'
EX=REPO/'benchmarks/cross_species_exercise_response';T3=REPO/'benchmarks/osdr_batch_effect_representation';T4=REPO/'benchmarks/library_prep_disentanglement'
sys.path.insert(0,str(HERE/'pipeline'))
from run_benchmark import G,canonical_genes,build_effects,pathway_sets
AX={'Axis A':['human_GSE108643','human_GSE86931','mouse_GSE126962','mouse_GSE132520'],'Axis B':['human_GSE71972','human_GSE87748','mouse_GSE97718']}

def say(s):print(f'[{time.strftime("%F %T")}] {s}',flush=True)
def node_scores(effect):
 c,v=effect; turn=np.zeros(G);signed=np.zeros(G);np.add.at(turn,c//G,np.abs(v));np.add.at(turn,c%G,np.abs(v));np.add.at(signed,c//G,v);np.add.at(signed,c%G,v);return turn,signed
def ranks_desc(x):
 order=np.argsort(-np.nan_to_num(x,nan=-np.inf));r=np.empty(len(x),int);r[order]=np.arange(1,len(x)+1);return r
def load_de_ig(contrast,genes):
 out=pd.DataFrame({'gene':genes})
 if contrast.startswith(('human_','mouse_')):
  d=pd.read_parquet(EX/'results/full_transcriptome_de/within_vocabulary_gene_comparisons.parquet').query('contrast_id == @contrast')
  keep=[c for c in ['gene','log2_fold_change','de_abs','de_rank','ig_abs','ig_rank','contextual_change','contextual_rank'] if c in d]
  out=out.merge(d[keep],on='gene',how='left')
  # FDR is retained in the full DE table.
  species,gse=contrast.split('_',1);p=EX/'results/full_transcriptome_de'/f'{species}_{gse}_full_de.parquet'
  if p.exists():
   q=pd.read_parquet(p);gc=next((c for c in q if c.lower() in ['gene_symbol','symbol','gene']),None);fc=next((c for c in q if c.lower() in ['fdr','padj','adj.p.val']),None)
   if gc and fc:out=out.merge(q[[gc,fc]].rename(columns={gc:'gene',fc:'DE_FDR'}),on='gene',how='left')
 else:
  ed=pd.read_csv(T4/'results/task4_confounding_profiler/independent_biological_replication/edger_results.csv.gz')
  q=ed[ed.contrast_id.eq(contrast)] if 'contrast_id' in ed else ed.iloc[0:0]
  gc=next((c for c in q if c.lower() in ['gene_symbol','gene','symbol']),None)
  if gc:
   cols={gc:'gene'}
   for c in q:
    if c.lower()=='logfc':cols[c]='log2_fold_change'
    if c.lower()=='fdr':cols[c]='DE_FDR'
   out=out.merge(q[list(cols)].rename(columns=cols),on='gene',how='left')
   out['de_abs']=out.log2_fold_change.abs();out['de_rank']=ranks_desc(out.de_abs.to_numpy())
  cl=pd.read_csv(T3/'results/task3c_cluster_assignments.csv');mode=cl.set_index('contrast_id').geometry_cluster.get(contrast,np.nan)
  if pd.notna(mode):
   ig=pd.read_parquet(T3/'results/task3d_mode_ig/mode_ig_gene_rankings.parquet').query('mode == @mode')
   out=out.merge(ig[['gene_symbol_human','mean_absolute_contrast_ig','rank']].rename(columns={'gene_symbol_human':'gene','mean_absolute_contrast_ig':'ig_abs','rank':'ig_rank'}),on='gene',how='left')
 return out
def ora(gene_set,collections,universe):
 rows=[];chosen=set(gene_set)&universe;M=len(universe)
 for source,sets in collections.items():
  for i,s in enumerate(sets):
   bg=s&universe;k=len(chosen&bg)
   if k:rows.append({'source':source,'set_index':i,'overlap':k,'set_size':len(bg),'p_value':hypergeom.sf(k-1,M,len(bg),len(chosen))})
 d=pd.DataFrame(rows)
 if len(d):
  order=np.argsort(d.p_value.to_numpy());raw=d.p_value.to_numpy()[order]*len(d)/np.arange(1,len(d)+1);adj=np.minimum.accumulate(raw[::-1])[::-1];fdr=np.empty(len(d));fdr[order]=np.minimum(1,adj);d['FDR']=fdr
 return d
def main():
 OUT.mkdir(parents=True,exist_ok=True);genes=canonical_genes().symbol.to_numpy();effects={}
 for name in ['exercise','task3']:
  e,_=build_effects(name,10);effects.update(e)
 rows=[];scores={}
 for cid,effect in effects.items():
  turn,strength=node_scores(effect);scores[cid]=turn;d=load_de_ig(cid,genes);d['contrast_id']=cid;d['graph_neighborhood_turnover']=turn;d['graph_strength_change']=strength;d['graph_rewiring_rank']=ranks_desc(turn)
  de=d.de_abs.fillna(0).to_numpy() if 'de_abs' in d else np.zeros(G); low=de<=np.nanmedian(de);high=turn>=np.nanpercentile(turn,95);d['context_excess_candidate']=low&high
  rows.append(d)
 candidates=pd.concat(rows,ignore_index=True);candidates.to_parquet(OUT/'per_gene_metrics.parquet',index=False)
 cand=candidates[candidates.context_excess_candidate].sort_values(['contrast_id','graph_rewiring_rank']);cand.to_csv(OUT/'low_de_high_rewiring_candidates.csv',index=False)
 # Reproducibility of gene rewiring ranks.
 rr=[];ids=list(scores)
 for i,a in enumerate(ids):
  for b in ids[i+1:]:
   ra=ranks_desc(scores[a]);rb=ranks_desc(scores[b]);ta=set(np.argsort(-scores[a])[:100]);tb=set(np.argsort(-scores[b])[:100]);u500=set(np.argsort(-scores[a])[:500]);v500=set(np.argsort(-scores[b])[:500])
   rr.append({'contrast_A':a,'contrast_B':b,'rank_spearman':spearmanr(ra,rb).statistic,'top100_overlap':len(ta&tb),'top500_overlap':len(u500&v500),'expected_top100_overlap':100*100/G,'expected_top500_overlap':500*500/G})
 rep=pd.DataFrame(rr);rep.to_csv(OUT/'rewired_gene_reproducibility.csv',index=False)
 # Prospective recurrent exercise ranking and pathway coherence.
 exids=[x for x in scores if x.startswith(('human_','mouse_'))];mean_pct=np.mean([ranks_desc(scores[x])/G for x in exids],axis=0);rec=np.argsort(mean_pct)[:500]
 recurrent=pd.DataFrame({'gene':genes[rec],'mean_rank_percentile':mean_pct[rec],'rank':np.arange(1,501)});recurrent.to_csv(OUT/'recurrent_exercise_rewired_genes.csv',index=False)
 enrich=ora(recurrent.head(100).gene,pathway_sets(),set(genes));enrich.to_csv(OUT/'recurrent_graph_gene_enrichment.csv',index=False)
 # Scientist-facing top candidates with available evidence.
 final=cand.groupby('contrast_id').head(25).copy();final['replicated']=final.gene.isin(recurrent.head(100).gene);final['pathway_function']='See recurrent_graph_gene_enrichment.csv';final['deletion_effect']=np.nan
 keep=['gene','contrast_id','log2_fold_change','DE_FDR','ig_rank','graph_rewiring_rank','replicated','pathway_function','deletion_effect']
 for c in keep:
  if c not in final:final[c]=np.nan
 final[keep].rename(columns={'contrast_id':'perturbation'}).to_csv(OUT/'scientist_facing_gene_table_pre_deletion.csv',index=False)
 # Literature-style prespecified Top-20 graph stress result (separate from k10).
 stress=pd.read_csv(REPO/'benchmarks/frozen_sample_embedding_readout/results/graph_fingerprint/rr1_rr3_stress/graph_response_stress_metrics.csv')
 stress.query("k == 20 and graph == 'union'").to_csv(OUT/'literature_style_top20_stress.csv',index=False)
 pd.DataFrame([{'analysis':'TF-target recovery','status':'unavailable','reason':'No authoritative verified local TF-target ground truth'}]).to_csv(OUT/'tf_target_status.csv',index=False)
 (OUT/'provenance.json').write_text(json.dumps({'thresholds':{'low_moderate_DE':'at or below within-contrast median absolute DE','high_rewiring':'within-contrast top 5% graph turnover'},'primary_graph':'union k10','literature_style_graph':'prespecified union Top-20 contextual similarity network','genes':G,'encoder_retrained':False,'TF_target':'unavailable','note':'contextual token displacement reused where present for exercise; not recomputed'},indent=2)+'\n');say('non-inference final-decision analyses complete')
if __name__=='__main__':main()
