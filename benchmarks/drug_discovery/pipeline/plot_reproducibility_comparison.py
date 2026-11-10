#!/usr/bin/env python3
"""Compare gene and pathway recurrence from independently selected top-500 lists."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/intersection_validation_three_libraries'
CONDITIONS=['Muscle atrophy','Bone loss','Radiation']
METHODS=[('BRIDGE',-.18,'#0072B2'),('DE',.18,'#E69F00')]

def main():
 for p in font_manager.findSystemFonts():
  if Path(p).stem.lower().startswith('arial'):font_manager.fontManager.addfont(p)
 font_manager.findfont('Arial',fallback_to_default=False)
 plt.rcParams.update({'font.family':'Arial','font.size':7.5,'axes.titlesize':8.5,'axes.titleweight':'normal','axes.labelsize':7.5,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'})
 genes=pd.read_csv(OUT/'recurrence_summary.csv');terms=pd.read_csv(OUT/'pathway_summary.csv')
 assert genes.top_genes_per_study.eq(500).all() and genes.studies.eq(3).all()
 assert len(genes)==6 and len(terms)==18
 fig=plt.figure(figsize=(183/25.4,132/25.4))
 grid=fig.add_gridspec(2,3,height_ratios=[1,1],hspace=.75,wspace=.38)
 top=fig.add_subplot(grid[0,:]);bottom=[fig.add_subplot(grid[1,i]) for i in range(3)]
 rows=[]
 for method,offset,color in METHODS:
  q=genes[genes.method.eq(method)].set_index('condition').loc[CONDITIONS]
  bars=top.bar(np.arange(3)+offset,q.all_three_genes,width=.34,color=color,label=method)
  top.bar_label(bars,padding=3,fontsize=7)
  for condition,v in q.all_three_genes.items():rows.append(dict(endpoint='gene_recurrence',condition=condition,method=method,library='',count=int(v)))
 top.set(xticks=range(3),xticklabels=CONDITIONS,ylabel='Genes selected in all three studies',ylim=(0,120),title='Gene recurrence across studies')
 top.legend(frameon=False,loc='upper right',ncol=2)
 top.text(-.09,1.09,'a',transform=top.transAxes,fontweight='bold',fontsize=10)
 for i,(ax,library) in enumerate(zip(bottom,['GO Biological Process','KEGG','Reactome'])):
  for method,offset,color in METHODS:
   q=terms[terms.method.eq(method)&terms.library.eq(library)].set_index('condition').loc[CONDITIONS]
   vals=q.pathways_significant_in_all_three_top500_sets
   bars=ax.bar(np.arange(3)+offset,vals,width=.34,color=color)
   ax.bar_label(bars,padding=3,fontsize=7)
   for condition,v in vals.items():rows.append(dict(endpoint='pathway_recurrence',condition=condition,method=method,library=library,count=int(v)))
  ax.set(title=library,xticks=range(3),xticklabels=['Muscle\natrophy','Bone\nloss','Radiation'],ylim=(0,7.5),yticks=[0,2,4,6])
  if i==0:ax.set_ylabel('Terms enriched in all three studies')
  else:ax.tick_params(labelleft=False)
 bottom[0].text(-.34,1.23,'b',transform=bottom[0].transAxes,fontweight='bold',fontsize=10)
 for ax in [top,*bottom]:ax.spines[['top','right']].set_visible(False);ax.tick_params(width=.6,length=3)
 fig.subplots_adjust(left=.12,right=.98,bottom=.12,top=.90)
 for ext in ['png','pdf','svg']:fig.savefig(OUT/f'reproducibility_comparison.{ext}',dpi=600,facecolor='white')
 plt.close(fig)
 pd.DataFrame(rows).to_csv(OUT/'reproducibility_comparison_values.csv',index=False)
 caption='''Cross-study reproducibility from matched top-500 gene lists. (a) Number of genes selected in all three studies for each condition. BRIDGE and DE each select 500 genes independently per study from the same eligible universe. (b) Number of terms enriched independently in all three study-specific top-500 lists, separated by annotation library. Each enrichment test uses the study-specific eligible, library-annotated background, one-sided hypergeometric tests, and BH correction jointly across GO Biological Process 2025, KEGG Human 2021, and Reactome Pathways 2024 per query. Passing terms require q<0.05, at least three overlapping genes, and background term size 10–500. Panel b does not enrich the gene intersections in panel a; contributing genes may differ across studies. Annotation terms overlap and counts are not independent biological discoveries. Zero labels indicate no passing terms. Counts are descriptive, without uncertainty intervals or a direct statistical test of method differences. DE has more significant study–term results overall (605 versus 272 for BRIDGE); this figure addresses recurrence rather than total within-study enrichment.'''
 (OUT/'reproducibility_comparison_caption.md').write_text(caption+'\n')
 sources=['recurrence_summary.csv','pathway_summary.csv']
 (OUT/'reproducibility_comparison_provenance.json').write_text(json.dumps({'sources':{s:hashlib.sha256((OUT/s).read_bytes()).hexdigest() for s in sources},'analysis_rerun':False,'top_genes_per_study':500,'conditions':CONDITIONS,'libraries':['GO Biological Process 2025','KEGG Human 2021','Reactome Pathways 2024'],'figure_width_mm':183,'font':'Arial regular','no_direct_method_difference_test':True},indent=2)+'\n')
 print('Saved reproducibility_comparison PNG/PDF/SVG, values, caption and provenance.')
if __name__=='__main__':main()
