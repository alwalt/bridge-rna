#!/usr/bin/env python3
"""Render compact cross-study validation figures from saved test outputs."""
from pathlib import Path
import textwrap,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/intersection_validation'
ORDER=['Muscle atrophy','Bone loss','Radiation']

def main():
 for p in font_manager.findSystemFonts():
  if Path(p).stem.lower().startswith('arial'):font_manager.fontManager.addfont(p)
 font_manager.findfont('Arial',fallback_to_default=False)
 plt.rcParams.update({'font.family':'Arial','font.size':7.5,'axes.titlesize':8.5,'axes.titleweight':'normal','axes.labelsize':7.5,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'})
 def save(fig,stem):
  for ext in ['png','pdf','svg']:fig.savefig(OUT/f'{stem}.{ext}',dpi=600,facecolor='white')
  plt.close(fig)
 summary=pd.read_csv(OUT/'recurrence_summary.csv');hits=pd.read_csv(OUT/'significant_pathways.csv')
 fig,ax=plt.subplots(figsize=(183/25.4,75/25.4))
 for method,offset,color in [('BRIDGE',-.18,'#0072B2'),('DE',.18,'#E69F00')]:
  q=summary[summary.method.eq(method)].set_index('condition').loc[ORDER]
  bars=ax.bar(np.arange(3)+offset,q.all_three_genes,width=.34,color=color,label=method)
  ax.bar_label(bars,padding=3,fontsize=7)
 ax.set(xticks=np.arange(3),xticklabels=ORDER,ylabel='Genes selected in all three studies',ylim=(0,120),title='Cross-study recurrence from matched top-500 gene sets')
 ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False)
 fig.subplots_adjust(left=.10,right=.98,bottom=.17,top=.83);save(fig,'three_study_gene_recurrence')
 selected=[]
 fig,axes=plt.subplots(3,1,figsize=(183/25.4,160/25.4))
 for ax,condition in zip(axes,ORDER):
  q=hits[hits.scope.eq('all_three')&hits.method.eq('BRIDGE')&hits.condition.eq(condition)].sort_values(['fdr','pvalue','library','term']).head(5)
  selected.append(q)
  if q.empty:
   n=int(summary[summary.condition.eq(condition)&summary.method.eq('BRIDGE')].all_three_genes.iloc[0])
   ax.text(0,.7,f'{condition}: {n} shared genes',ha='left',transform=ax.transAxes,fontsize=8.5)
   ax.text(0,.4,'No intersection pathways pass q < 0.05',ha='left',transform=ax.transAxes,fontsize=7.5)
   ax.set_axis_off()
  else:
   labels=['\n'.join(textwrap.wrap(t,52)) for t in q.term]
   ax.barh(np.arange(len(q)),-np.log10(q.fdr.clip(lower=1e-300)),color='#0072B2',height=.65)
   ax.set(yticks=np.arange(len(q)),yticklabels=labels);ax.invert_yaxis()
  ax.set_title(condition,loc='left')
  if not q.empty:ax.set_xlabel('−log10(BH-adjusted p-value)')
  ax.spines[['top','right']].set_visible(False)
 fig.subplots_adjust(left=.48,right=.98,bottom=.08,top=.94,hspace=.85)
 save(fig,'intersection_pathway_enrichment')
 shown=pd.concat(selected,ignore_index=True);shown.to_csv(OUT/'plotted_pathways.csv',index=False)
 rec=pd.read_csv(OUT/'pathway_recurrence.csv')
 # Distinct endpoint: terms enriched separately in all three full 500-gene lists.
 fig,axes=plt.subplots(3,1,figsize=(183/25.4,160/25.4))
 recurrent_shown=[]
 for ax,condition in zip(axes,ORDER):
  q=rec[rec.condition.eq(condition)&rec.method.eq('BRIDGE')&rec.studies.eq(3)].sort_values(['max_fdr','library','term']).head(5)
  recurrent_shown.append(q)
  if q.empty:
   ax.text(0,.7,condition,ha='left',transform=ax.transAxes,fontsize=8.5)
   ax.text(0,.4,'No pathways pass in all three top-500 lists',ha='left',transform=ax.transAxes,fontsize=7.5)
   ax.set_axis_off()
  else:
   ax.barh(np.arange(len(q)),-np.log10(q.max_fdr),color='#0072B2',height=.65)
   ax.set(yticks=np.arange(len(q)),yticklabels=['\n'.join(textwrap.wrap(t,52)) for t in q.term])
   ax.invert_yaxis();ax.set_title(condition,loc='left');ax.set_xlabel('−log10(largest study-specific BH-adjusted p-value)')
  ax.spines[['top','right']].set_visible(False)
 fig.subplots_adjust(left=.48,right=.98,bottom=.08,top=.94,hspace=.85)
 save(fig,'pathways_recurrent_across_studies')
 pd.concat(recurrent_shown,ignore_index=True).to_csv(OUT/'plotted_recurrent_pathways.csv',index=False)

 counts=[]
 for condition in ORDER:
  for method in ['BRIDGE','DE']:
   subset=hits[hits.condition.eq(condition)&hits.method.eq(method)&hits.scope.eq('all_three')]
   shared=rec[rec.condition.eq(condition)&rec.method.eq(method)&rec.studies.eq(3)]
   counts.append({'condition':condition,'method':method,'significant_intersection_terms':len(subset),'pathways_significant_in_all_three_top500_sets':len(shared)})
 pd.DataFrame(counts).to_csv(OUT/'pathway_summary.csv',index=False)
 lines=['# Three-study gene recurrence and pathway validation','','## Protocol','',
 '- Reuse the frozen 500-gene BRIDGE and DE lists independently selected in each of three studies per condition. No model or DE refitting.',
 '- Use the original positive-expression canonical-gene universe for each study; use its three-study intersection for recurrent-gene enrichment. Restrict to library-annotated genes for each library.',
 '- Test GO Biological Process 2025 and Reactome Pathways 2024 annotations downloaded from Enrichr. One-sided hypergeometric tests; BH correction across both libraries per query; terms containing 10–500 background genes. Display significant rows with at least three overlapping genes.',
 '- Use 1,000 uniform random top-500 selections from each study’s eligible genes as a simple intersection-size control. This does not match expression propensity or establish a BRIDGE–DE significance difference.',
 '', '## Recurrence','']
 for row in summary.itertuples():lines.append(f'- {row.condition}, {row.method}: {row.all_three_genes} shared genes; background {row.common_eligible_genes}; random mean {row.random_mean_intersection:.3f}, maximum {row.random_max_intersection}; empirical p={row.random_empirical_p:.6g}.')
 lines+=['','## Biological annotations: strongest results without manual term selection','']
 for condition in ORDER:
  lines.append('### '+condition)
  if shown[shown.condition.eq(condition)].empty:lines.append('- No intersection pathways pass the specified BH FDR and overlap criteria.')
  for r in shown[shown.condition.eq(condition)].itertuples():lines.append(f'- {r.term}: q={r.fdr:.3g}; {r.overlap}/{r.query_genes} annotated query genes; {r.fold_enrichment:.2f}-fold; {r.overlap_genes}.')
 lines+=['','## Equal-size per-study pathway comparison','']
 for row in pd.DataFrame(counts).itertuples():
  lines.append(f'- {row.condition}, {row.method}: {row.pathways_significant_in_all_three_top500_sets} pathway rows pass the same criterion in all three independently selected 500-gene sets.')
 lines.append('- Bone BRIDGE lists repeatedly implicate extracellular-matrix organization, ECM proteoglycans and collagen-fibril organization, despite no significant enrichment of the smaller exact three-study intersection. Radiation has no pathway passing in all three lists under this protocol.')
 lines+=['','## Limits','',
 '- Enrichment describes annotated biological coherence, not specificity to a stressor, causal mechanisms, directional consistency, or therapeutic benefit.',
 '- Small DE intersections (including an empty set) have little enrichment power; significance for BRIDGE but not DE is not a direct method-difference test.',
 '- Equal-size per-study pathway results and pathway recurrence are supplied separately; term counts are descriptive and affected by overlapping annotations.',
 '- Heterogeneous tissues and exposures, and partial terrestrial pretraining exposure, remain relevant.',
 '- This is a new exploratory analysis in a separate results directory. Prior source results are preserved.']
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
 print(pd.DataFrame(counts).to_string(index=False))
if __name__=='__main__':main()
