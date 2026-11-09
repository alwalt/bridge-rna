#!/usr/bin/env python3
"""Manuscript panels and PCA sensitivity from saved evidence from validated saved results."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures/manuscript';OUT.mkdir(exist_ok=True)
point_source=ROOT/'figures/bridge_oof_ic50_regression_values.csv.gz'
score_source=ROOT/'data/tables/per_drug_metrics.csv'
points=pd.read_csv(point_source);scores=pd.read_csv(score_source);scores=scores[scores.representation.eq('bridge_45.6m')]
assert len(points)==212299 and len(scores)==355 and scores.screen_id.is_unique
assert not points.duplicated(['screen_id','cell_line_id']).any()
for p in font_manager.findSystemFonts():
 if Path(p).stem.lower().startswith('arial'):font_manager.fontManager.addfont(p)
font=font_manager.findfont('Arial',fallback_to_default=False)
plt.rcParams.update({'font.family':'Arial','font.size':7.5,'axes.titlesize':8.5,'axes.titleweight':'normal','axes.labelsize':7.5,'xtick.labelsize':7,'ytick.labelsize':7,'text.color':'#222222','axes.labelcolor':'#222222','xtick.color':'#222222','ytick.color':'#222222','axes.edgecolor':'#333333','axes.linewidth':.6,'svg.fonttype':'none','pdf.fonttype':42})
fig=plt.figure(figsize=(183/25.4,90/25.4),facecolor='white')
a=fig.add_axes([.09,.19,.46,.68],facecolor='white')
b=fig.add_axes([.70,.19,.28,.68],facecolor='white')
lo=float(np.floor(points[['observed_ic50','predicted_ic50']].min().min()));hi=float(np.ceil(points[['observed_ic50','predicted_ic50']].max().max()));pad=(hi-lo)*.025
# Identical points and native target scale; alpha adjusted only for dark placement.
a.scatter(points.observed_ic50,points.predicted_ic50,s=1,alpha=.025,color='#0072B2',edgecolors='none',rasterized=True)
a.plot([lo-pad,hi+pad],[lo-pad,hi+pad],color='#777777',ls='--',lw=.7)
a.set(xlim=(lo-pad,hi+pad),ylim=(lo-pad,hi+pad),aspect='equal',xlabel='Observed IC50 (released scale)',ylabel='Predicted IC50 (released scale)',title='BRIDGE held-out predictions')
r=float(pearsonr(points.observed_ic50,points.predicted_ic50).statistic)
a.text(.04,.96,f'All-point r = {r:.3f}\nn = {len(points):,}',transform=a.transAxes,va='top',fontsize=7)
mean=float(scores.pcc.mean());edges=np.linspace(-1,1,41)
counts,_,_=b.hist(scores.pcc,bins=edges,color='#0072B2',edgecolor='white',linewidth=.35)
assert int(counts.sum())==355
b.axvline(mean,color='#222222',lw=.9,ls='--',label=f'Mean r = {mean:.3f}')
b.set(xlim=(-1,1),xlabel='Within-screen Pearson r',ylabel='Number of drug screens',title='Within-screen correlations',xticks=[-1,-.5,0,.5,1])
b.legend(frameon=False,fontsize=7,loc='upper left',handlelength=1.4,handletextpad=.4)
for ax in [a,b]:
 ax.spines[['top','right']].set_visible(False);ax.tick_params(width=.6,length=3)
fig.canvas.draw()
# Place panel labels above the actual axes bounds, accounting for equal aspect.
for label,ax in [('a',a),('b',b)]:
 pos=ax.get_position();fig.text(pos.x0-.035,.94,label,fontsize=10,fontweight='bold',ha='left')
stem='drug_response_prediction_panels'
for ext in ['png','svg','pdf']:fig.savefig(OUT/f'{stem}.{ext}',dpi=600,transparent=False)
plt.close(fig)
(OUT/f'{stem}.json').write_text(json.dumps({'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [point_source,score_source]},'panels':{'a':'All 212299 measured responses against held-out predictions; native released scale; no standardization or subsampling.','b':'Histogram of all 355 within-screen Pearson correlations; 40 bins of width 0.05 over [-1,1].'},'all_point_pearson':r,'mean_within_screen_pearson':mean,'background':'white; dark text','font':font,'font_pt':7.5,'title_font_pt':8.5,'title_weight':'normal','width_mm':183,'height_mm':90,'png_dpi':600,'point_layer':'Rasterized at 600 dpi in PDF/SVG; axes/text remain vector.','layout':'Side-by-side manuscript panels; regular-weight Arial headings.'},indent=2)+'\n')
print(OUT/f'{stem}.png')

# Compact dimensionality sensitivity, with all evaluated component counts retained.
pca_source=ROOT/'data/tables/pca_component_sensitivity.csv'
pca=pd.read_csv(pca_source).sort_values('components')
summary=pd.read_csv(ROOT/'data/tables/primary_results.csv')
bridge=float(summary.loc[summary.representation.eq('bridge_45.6m'),'mean_pcc'].iloc[0])
fig,ax=plt.subplots(figsize=(183/25.4,75/25.4),facecolor='white')
ax.plot(pca.components,pca.mean_pcc,'o-',color='#E69F00',ms=3,lw=1,label='PCA')
ax.axhline(bridge,color='#0072B2',ls='--',lw=1,label=f'BRIDGE ({bridge:.3f})')
ax.set(xscale='log',xticks=pca.components.to_list(),xlabel='PCA components (log scale)',ylabel='Mean within-screen Pearson r',title='PCA dimensionality sensitivity',ylim=(0,.5))
ax.xaxis.set_major_formatter(plt.ScalarFormatter())
ax.legend(frameon=False,fontsize=7,loc='upper right',bbox_to_anchor=(1,.82));ax.spines[['top','right']].set_visible(False)
fig.subplots_adjust(left=.10,right=.98,bottom=.19,top=.83)
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'pca_component_sensitivity.{ext}',dpi=600,facecolor='white')
plt.close(fig)
pca.to_csv(OUT/'pca_component_sensitivity_values.csv',index=False)
(OUT/'pca_component_sensitivity_provenance.json').write_text(json.dumps({'source':str(pca_source.relative_to(ROOT)),'sha256':hashlib.sha256(pca_source.read_bytes()).hexdigest(),'selection':'All evaluated PCA dimensions; dimensionality sweep exploratory.','baseline':bridge,'baseline_source':'data/tables/primary_results.csv','font':'Arial','width_mm':183,'height_mm':75,'png_dpi':600},indent=2)+'\n')
