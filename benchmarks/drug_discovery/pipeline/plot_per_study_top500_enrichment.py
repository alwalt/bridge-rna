#!/usr/bin/env python3
"""Per-study ORA of 500 BRIDGE versus 500 DE genes; no recurrence filter."""
from pathlib import Path
import json,hashlib,textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/intersection_validation_three_libraries'
LIBS=[('GO_Biological_Process_2025','GO Biological Process'),('KEGG_2021_Human','KEGG'),('Reactome_Pathways_2024','Reactome')]
STUDIES={'Muscle atrophy':['GSE211204','GSE113165','GSE234465'],'Bone loss':['GSE189524','GSE276529','GSE273868'],'Radiation':['GSE297090','GSE184119','GSE297560']}
COLORS={'BRIDGE':'#0072B2','DE':'#E69F00'}
def main():
 for p in font_manager.findSystemFonts():
  if Path(p).stem.lower().startswith('arial'):font_manager.fontManager.addfont(p)
 font_manager.findfont('Arial',fallback_to_default=False)
 plt.rcParams.update({'font.family':'Arial','font.size':7.5,'axes.titlesize':8.5,'axes.titleweight':'normal','axes.labelsize':7.5,'xtick.labelsize':7,'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'})
 source=OUT/'significant_pathways.csv';d=pd.read_csv(source);d=d[d.scope.eq('study_top500')].copy()
 assert d.fdr.lt(.05).all() and d.overlap.ge(3).all()
 selected=[];counts=[]
 for condition,studies in STUDIES.items():
  fig,axes=plt.subplots(3,3,figsize=(183/25.4,260/25.4))
  cohort=d[d.condition.eq(condition)]
  xmax=max(2,np.ceil(-np.log10(cohort.fdr.min())))
  for i,study in enumerate(studies):
   for j,(library,label) in enumerate(LIBS):
    ax=axes[i,j];q=cohort[cohort.dataset.eq(study)&cohort.library.eq(library)]
    names=[]
    for method in COLORS:
     m=q[q.method.eq(method)].sort_values(['fdr','term']);counts.append(dict(condition=condition,dataset=study,library=label,method=method,significant_terms=len(m)))
     names.extend(m.head(3).term.tolist())
    names=list(dict.fromkeys(names))
    names=sorted(names,key=lambda name:q[q.term.eq(name)].fdr.min())
    for k,name in enumerate(names):
     y=k*5
     labeltext='\n'.join(textwrap.wrap(name,width=30))
     ax.text(.01,y,labeltext,transform=ax.get_yaxis_transform(),ha='left',va='top',fontsize=6.5)
     for method,offset in [('BRIDGE',3.6),('DE',4.25)]:
      hit=q[q.term.eq(name)&q.method.eq(method)]
      if len(hit):
       row=hit.iloc[0];ax.barh(y+offset,-np.log10(row.fdr),height=.5,color=COLORS[method]);selected.append(row.to_dict())
    ax.set(xlim=(0,xmax*1.05),ylim=(30,-.3),yticks=[])
    ax.axvline(-np.log10(.05),color='.7',ls=':',lw=.6,zorder=0)
    ax.spines[['top','right','left']].set_visible(False)
    if i==0:ax.set_title(label,pad=16)
    if j==0:ax.text(-.28,.5,study,transform=ax.transAxes,rotation=90,va='center',ha='center',fontsize=8)
    if not names:ax.text(.05,.5,'No significant terms',transform=ax.transAxes,color='.4')
  fig.suptitle(condition+' — individual top-500 gene lists',fontsize=9,y=.985)
  fig.legend(handles=[Patch(color=color,label=method) for method,color in COLORS.items()],loc='upper center',bbox_to_anchor=(.5,.965),ncol=2,frameon=False,fontsize=7)
  fig.supxlabel('−log10(BH-adjusted p-value)',fontsize=7.5,y=.015)
  fig.subplots_adjust(left=.12,right=.98,top=.90,bottom=.055,wspace=.30,hspace=.12)
  stem='per_study_top500_'+condition.lower().replace(' ','_')
  for ext in ['png','pdf','svg']:fig.savefig(OUT/f'{stem}.{ext}',dpi=600,facecolor='white')
  plt.close(fig)
 pd.DataFrame(selected).drop_duplicates().to_csv(OUT/'per_study_top500_plotted_terms.csv',index=False)
 c=pd.DataFrame(counts);assert len(c)==54;assert c.groupby('method').significant_terms.sum().to_dict()=={'BRIDGE':272,'DE':605}
 c.to_csv(OUT/'per_study_top500_enrichment_counts.csv',index=False)
 caption='''Each row is an individual study; columns are GO Biological Process 2025, KEGG Human 2021 and Reactome Pathways 2024. BRIDGE and DE each contribute 500 genes selected independently from the same study-specific eligible universe. No gene intersection or cross-study pathway recurrence filter is applied. Displayed terms are the union of the three lowest-q passing terms for each method within each study/library, ordered by the smaller q. For each displayed term, blue and orange bars show passing BRIDGE and DE results respectively; a missing bar means that method did not meet the reporting criteria, not zero enrichment. Terms may occur for both methods. All significant terms and exact overlap counts are available in significant_pathways.csv. One-sided hypergeometric tests use study-specific eligible, library-annotated backgrounds; BH correction is joint across all three libraries per query. Reporting requires q<0.05, overlap>=3, and background term size 10–500. Bar length measures statistical evidence, not fold enrichment; annotations overlap, and a difference in significance is not a direct statistical test between methods.'''
 (OUT/'per_study_top500_caption.md').write_text(caption+'\n')
 (OUT/'per_study_top500_plot_provenance.json').write_text(json.dumps({'input':'significant_pathways.csv','sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'scope':'study_top500','top_terms_per_method_per_study_library':3,'cross_study_filter':False,'analysis_rerun':False,'studies':STUDIES,'font':'Arial'},indent=2)+'\n')
 print(c.groupby(['condition','method']).significant_terms.sum().to_string())
if __name__=='__main__':main()
