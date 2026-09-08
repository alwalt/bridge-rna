#!/usr/bin/env python3
"""Combine new graph deletion with existing comparators and finalize decision."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np,pandas as pd

HERE=Path(__file__).resolve().parents[1];REPO=HERE.parents[1];OUT=HERE/'results/final_decision';FIG=HERE/'results/figures';FIG.mkdir(parents=True,exist_ok=True)
graph=pd.read_csv(OUT/'graph_ranked_deletion.csv');rows=[]
for r in graph.itertuples():rows.append({'target':r.target,'ranking':'graph_turnover','genes_masked':r.genes_masked,'fraction_signal_remaining':r.fraction_signal_remaining,'sd':np.nan,'replicates':1})
ex=REPO/'benchmarks/cross_species_exercise_response/results/latent_axis_attribution/extended_deletion_sweep_summary.csv'
if ex.exists():
 d=pd.read_csv(ex)
 for r in d.itertuples():rows.append({'target':{'Axis A':'exercise_axis_a','Axis B':'exercise_axis_b'}[r.axis],'ranking':r.panel_type,'genes_masked':r.genes_masked,'fraction_signal_remaining':r.fraction_remaining_mean,'sd':r.fraction_remaining_sd,'replicates':r.replicates})
t3=REPO/'benchmarks/osdr_batch_effect_representation/results/task3d_mode_ig/deletion_validation/deletion_summary.csv'
if t3.exists():
 d=pd.read_csv(t3)
 for r in d.itertuples():rows.append({'target':f'task3_mode_{r.mode}','ranking':r.panel_type,'genes_masked':r.genes_masked,'fraction_signal_remaining':r.fraction_remaining_mean,'sd':r.fraction_remaining_sd,'replicates':r.replicates})
combined=pd.DataFrame(rows);combined.to_csv(OUT/'deletion_all_rankings.csv',index=False)
fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
colors={'graph_turnover':'#7B2CBF','ig_ranked':'#0072B2','de_ranked':'#D55E00','random':'#777777'}
for ax,(target,q) in zip(axes.ravel(),combined.groupby('target',sort=False)):
 for name,z in q.groupby('ranking'):
  z=z.sort_values('genes_masked');ax.plot(z.genes_masked,z.fraction_signal_remaining,marker='o',label=name,color=colors.get(name))
  if z.sd.notna().any():ax.fill_between(z.genes_masked,z.fraction_signal_remaining-z.sd.fillna(0),z.fraction_signal_remaining+z.sd.fillna(0),alpha=.15,color=colors.get(name))
 ax.axhline(1,color='black',ls='--',lw=.8);ax.set_title(target);ax.set_ylabel('Fraction response remaining');ax.legend(frameon=False,fontsize=8)
for ax in axes[-1]:ax.set_xlabel('Genes masked')
fig.suptitle('Functional leverage of graph-, IG-, DE-, and random-ranked genes');fig.tight_layout();fig.savefig(FIG/'final_graph_deletion_comparison.png',dpi=350);fig.savefig(FIG/'final_graph_deletion_comparison.pdf');plt.close(fig)
# Add panel-level deletion result to scientist table.
table=pd.read_csv(OUT/'scientist_facing_gene_table_pre_deletion.csv');pan=pd.read_parquet(OUT/'graph_ranked_deletion_panels.parquet');effects=[]
for r in table.itertuples():
 targets=[]
 if str(r.perturbation).startswith(('human_','mouse_')):
  targets=['exercise_axis_a'] if r.perturbation in ['human_GSE108643','human_GSE86931','mouse_GSE126962','mouse_GSE132520'] else ['exercise_axis_b']
 else:
  cl=pd.read_csv(REPO/'benchmarks/osdr_batch_effect_representation/results/task3c_cluster_assignments.csv');m=cl.set_index('contrast_id').geometry_cluster.get(r.perturbation,np.nan);targets=[f'task3_mode_{int(m)}'] if pd.notna(m) else []
 p=pan[(pan.target.isin(targets))&pan.gene.eq(r.gene)].sort_values('genes_masked')
 if len(p):
  size=int(p.genes_masked.iloc[0]);q=graph[(graph.target.isin(targets))&graph.genes_masked.eq(size)];effects.append(float(1-q.fraction_signal_remaining.iloc[0]))
 else:effects.append(np.nan)
table['deletion_effect']=effects;table.to_csv(OUT/'scientist_facing_gene_table.csv',index=False)
# Response-level fair reference table.
vector=pd.read_csv(REPO/'benchmarks/frozen_sample_embedding_readout/results/graph_fingerprint/rr1_rr3_stress/cross_representation_comparison.csv').pivot(index='representation',columns='comparison',values='similarity').reset_index()
stress=pd.read_csv(REPO/'benchmarks/frozen_sample_embedding_readout/results/graph_fingerprint/rr1_rr3_stress/graph_response_stress_metrics.csv')
for k,label in [(10,'contextual kNN graph'),(20,'literature-style Top20 graph')]:
 q=stress.query("k == @k and graph == 'union'").set_index('comparison').signed_edge_cosine
 vector=pd.concat([vector,pd.DataFrame([{'representation':label,'RR1':q.get('RR1'),'RR3-39':q.get('RR3-39'),'RR3-40':q.get('RR3-40'),'RR1↔RR3-39 false friend':q.get('RR1↔RR3-39 false friend')}])],ignore_index=True)
vector.to_csv(OUT/'response_level_comparison.csv',index=False)
# Prospective decision: reproducibility and deletion both required for A.
rep=pd.read_csv(OUT/'rewired_gene_reproducibility.csv');rr3=rep[(rep.contrast_A.str.contains('OSD-137'))&(rep.contrast_B.str.contains('OSD-168'))]
g=combined[combined.ranking.eq('graph_turnover')];rnd=combined[combined.ranking.eq('random')][['target','genes_masked','fraction_signal_remaining']].rename(columns={'fraction_signal_remaining':'random_remaining'})
cmp=g.merge(rnd,on=['target','genes_masked'],how='left');cmp['graph_more_leverage_than_random']=cmp.fraction_signal_remaining<cmp.random_remaining
above=float(cmp.graph_more_leverage_than_random.mean()) if cmp.random_remaining.notna().any() else np.nan
medrep=float(rr3.rank_spearman.median()) if len(rr3) else np.nan
if above>=.75 and medrep>=.5:decision='A. GRAPH REWIRING HAS DISTINCT UTILITY'
elif above>=.5 or medrep>=.3:decision='B. GRAPH REWIRING IS COMPLEMENTARY'
elif above<.5 and medrep<.3:decision='C. GRAPH SIGNAL MOSTLY RECAPITULATES EXPRESSION'
else:decision='D. MIXED'
summary={'decision':decision,'graph_deletion_conditions_more_leverage_than_random_fraction':above,'RR3_rewiring_rank_spearman_median':medrep,'TF_target_recovery':'unavailable','thresholds_frozen':True,'graph_parameters_selected_on_RR1_RR3':False}
(OUT/'final_decision.json').write_text(json.dumps(summary,indent=2)+'\n');pd.DataFrame([summary]).to_csv(OUT/'final_decision.csv',index=False);print(json.dumps(summary,indent=2))
