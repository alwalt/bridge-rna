#!/usr/bin/env python3
"""Three library columns, three condition rows; preserve distinct endpoints."""
from pathlib import Path
import json,textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/intersection_validation_three_libraries'
CONDITIONS=['Muscle atrophy','Bone loss','Radiation']
LIBRARIES=[('GO_Biological_Process_2025','GO Biological Process'),('KEGG_2021_Human','KEGG'),('Reactome_Pathways_2024','Reactome')]

def main():
 for p in font_manager.findSystemFonts():
  if Path(p).stem.lower().startswith('arial'):font_manager.fontManager.addfont(p)
 font_manager.findfont('Arial',fallback_to_default=False)
 plt.rcParams.update({'font.family':'Arial','font.size':7.5,'axes.titlesize':8.5,'axes.titleweight':'normal','axes.labelsize':7.5,'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'})
 hits=pd.read_csv(OUT/'significant_pathways.csv');rec=pd.read_csv(OUT/'pathway_recurrence.csv');summary=pd.read_csv(OUT/'recurrence_summary.csv')
 def save(fig,stem):
  for ext in ['png','pdf','svg']:fig.savefig(OUT/f'{stem}.{ext}',dpi=600,facecolor='white')
  plt.close(fig)
 for endpoint,stem in [('intersection','intersection_pathway_enrichment'),('study_recurrence','pathways_recurrent_across_studies')]:
  data=hits[hits.scope.eq('all_three')&hits.method.eq('BRIDGE')].copy() if endpoint=='intersection' else rec[rec.method.eq('BRIDGE')&rec.studies.eq(3)].copy()
  qcol='fdr' if endpoint=='intersection' else 'max_fdr'
  largest=max(2,float(np.ceil((-np.log10(data[qcol].clip(lower=1e-300))).max()))) if len(data) else 2
  fig,axes=plt.subplots(3,3,figsize=(183/25.4,195/25.4))
  plotted=[]
  for i,condition in enumerate(CONDITIONS):
   for j,(library,label) in enumerate(LIBRARIES):
    ax=axes[i,j];q=data[data.condition.eq(condition)&data.library.eq(library)].sort_values([qcol,'term']).head(3)
    plotted.append(q)
    ax.set(xlim=(0,largest*1.06),ylim=(11.8,-.3),yticks=[])
    if i==0:ax.set_title(label,pad=14)
    ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='x',length=2.5,width=.6)
    if q.empty:
     ax.text(.04,.60,'No terms pass\nthe stated criteria',transform=ax.transAxes,ha='left',va='center',fontsize=7,color='#555555')
    else:
     for k,row in enumerate(q.itertuples()):
      text='\n'.join(textwrap.wrap(textwrap.shorten(row.term,width=76,placeholder='…'),width=26))
      y=k*4
      ax.text(.01,y,text,transform=ax.get_yaxis_transform(),ha='left',va='top',fontsize=7)
      ax.barh(y+2.5,-np.log10(getattr(row,qcol)),height=.65,color='#0072B2')
    ax.axvline(-np.log10(.05),color='#AAAAAA',lw=.6,ls=':',zorder=0)
  fig.subplots_adjust(left=.13,right=.98,bottom=.10,top=.90,wspace=.30,hspace=.33)
  for i,condition in enumerate(CONDITIONS):
   pos=axes[i,0].get_position()
   fig.text(.025,(pos.y0+pos.y1)/2,condition,rotation=90,va='center',ha='center',fontsize=8.5)
  xlabel='−log10(BH-adjusted p-value)' if endpoint=='intersection' else '−log10(largest study-specific BH-adjusted p-value)'
  fig.text(.56,.045,xlabel,ha='center',fontsize=7.5)
  save(fig,stem)
  shown=pd.concat(plotted,ignore_index=True);shown.to_csv(OUT/f'{stem}_plotted_values.csv',index=False)
 # Keep the recurrence count figure identical in definition and appearance.
 fig,ax=plt.subplots(figsize=(183/25.4,75/25.4))
 for method,offset,color in [('BRIDGE',-.18,'#0072B2'),('DE',.18,'#E69F00')]:
  q=summary[summary.method.eq(method)].set_index('condition').loc[CONDITIONS]
  bars=ax.bar(np.arange(3)+offset,q.all_three_genes,width=.34,color=color,label=method);ax.bar_label(bars,padding=3,fontsize=7)
 ax.set(xticks=np.arange(3),xticklabels=CONDITIONS,ylabel='Genes selected in all three studies',ylim=(0,120),title='Cross-study recurrence from matched top-500 gene sets');ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False)
 fig.subplots_adjust(left=.10,right=.98,bottom=.17,top=.83);save(fig,'three_study_gene_recurrence')
 counts=[]
 for condition in CONDITIONS:
  for method in ['BRIDGE','DE']:
   for lib,label in LIBRARIES:
    counts.append({'condition':condition,'method':method,'library':label,'significant_intersection_terms':int((hits.condition.eq(condition)&hits.method.eq(method)&hits.scope.eq('all_three')&hits.library.eq(lib)).sum()),'pathways_significant_in_all_three_top500_sets':int((rec.condition.eq(condition)&rec.method.eq(method)&rec.studies.eq(3)&rec.library.eq(lib)).sum())})
 counts=pd.DataFrame(counts);counts.to_csv(OUT/'pathway_summary.csv',index=False)
 lines=['# GO, KEGG and Reactome: three-study validation','','## Gene selection','',
 '- Select 500 genes independently per study and method, then compute three-study intersections: BRIDGE 102/66/50 and DE 7/0/3 (muscle/bone/radiation). No list of 500 shared genes is used.',
 '- Intersection enrichment and pathways recurring across three separately enriched top-500 lists are distinct endpoints, shown in separate three-column figures.',
 '- Libraries: GO Biological Process 2025; KEGG Human 2021; Reactome Pathways 2024. Hypergeometric tests use eligible library-annotated backgrounds; BH correction jointly across all three libraries for each query. q<0.05 and at least three overlapping genes; term size 10–500.',
 '- Columns show the three libraries, rows the three conditions. Up to three strongest passing terms per cell, chosen by adjusted p-value without manually selecting stress-related terms. Display labels may be shortened; full names are in CSV.',
 '- Recurrent-pathway bars use the largest of the three study-specific q-values, not a combined p-value. Pathways can overlap and need not have the same contributing genes in each study.',
 '- Prior two-library analysis is preserved under results/intersection_validation/. Adding KEGG expands the correction family; use the new q-values for these figures.','', '## Passing term counts','']
 for row in counts.itertuples():lines.append(f'- {row.condition}, {row.method}, {row.library}: intersection {row.significant_intersection_terms}; recurring across study lists {row.pathways_significant_in_all_three_top500_sets}.')
 lines+=['','## Limitations','','- Small DE intersections have limited power. Different significance outcomes are not a direct method-difference test.','- Annotation enrichment supports coherence, not condition specificity, causal mechanisms, or treatment benefit.','- Random size controls are uniform eligibility-matched selections, not expression-propensity matched controls.','- Heterogeneous contexts and partial terrestrial pretraining exposure remain relevant.']
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
 (OUT/'plot_provenance.json').write_text(json.dumps({'layout':'3 condition rows x 3 library columns','selection':'Up to 3 lowest q passing terms per panel; no manual condition filter','font':'Arial','width_mm':183,'height_mm':195,'q_family':'All three libraries jointly within each query','endpoints':['Enrichment of three-way gene intersection','Terms enriched separately in all three top500 study lists']},indent=2)+'\n')
 print(counts[counts.method.eq('BRIDGE')].to_string(index=False))
if __name__=='__main__':main()
