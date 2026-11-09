#!/usr/bin/env python3
"""Style classification figures from saved results; no training or inference."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'; FIGURES=ROOT/'figures'; WIDTH=183/25.4
REPS=['bridge','pca','raw']
COLORS={'bridge':'#0072B2','pca':'#E69F00','raw':'#009E73'}
LABELS={'bridge':'BRIDGE','pca':'PCA','raw':'Raw expression'}
TITLES={'gtex_detailed':'GTEx: 51 tissues','tcga_cancer':'TCGA: 33 cancer types','gtex_broad':'GTEx: 30 broad tissues'}

def style():
    for p in font_manager.findSystemFonts():
        if Path(p).stem.lower().startswith('arial'):font_manager.fontManager.addfont(p)
    font=font_manager.findfont('Arial',fallback_to_default=False)
    plt.rcParams.update({'font.family':'Arial','font.size':7.5,'axes.titlesize':8.5,'axes.labelsize':7.5,'axes.titleweight':'normal','figure.titleweight':'normal',
        'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'axes.linewidth':.6,
        'pdf.fonttype':42,'svg.fonttype':'none','figure.facecolor':'white','axes.facecolor':'white'})
    return font

def save(fig,stem,values):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    # Verify all visible text is contained by the fixed-size export canvas.
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            b=text.get_window_extent(renderer)
            assert b.x0>=-1 and b.y0>=-1 and b.x1<=fig.bbox.width+1 and b.y1<=fig.bbox.height+1, (stem,text.get_text())
    for ext in ['png','pdf','svg']:
        fig.savefig(FIGURES/f'{stem}.{ext}',dpi=600,facecolor='white')
    values.to_csv(FIGURES/f'{stem}_values.csv',index=False)
    plt.close(fig)

def dots(ax,folds,task,metric,label,limits):
    rng=np.random.default_rng(20260923)
    for i,rep in enumerate(REPS):
        v=folds.loc[folds.task.eq(task)&folds.representation.eq(rep),metric].to_numpy()
        assert len(v)==5
        ax.scatter(i+rng.uniform(-.055,.055,5),v,s=13,color=COLORS[rep],edgecolor='white',linewidth=.35,zorder=3)
        ax.errorbar(i,v.mean(),yerr=v.std(ddof=1),fmt='D',ms=3.5,color='#222222',capsize=2,lw=.8,zorder=4)
    ax.set(xticks=range(3),xticklabels=[LABELS[r] for r in REPS],ylim=limits,ylabel=label,xlim=(-.45,2.55))
    ax.grid(axis='y',color='#EEEEEE',lw=.5);ax.spines[['top','right']].set_visible(False)

def main():
    font=style();sources=list(DATA.glob('*.csv'))
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    folds=pd.read_csv(DATA/'annotation_figure_data.csv'); tasks=['gtex_detailed','tcga_cancer']
    fig,axes=plt.subplots(1,2,figsize=(WIDTH,78/25.4),sharey=True)
    for ax,task in zip(axes,tasks):
        dots(ax,folds,task,'macro_f1','Macro F1' if ax is axes[0] else '',(.84,.98))
        ax.set_title(TITLES[task])
    fig.subplots_adjust(left=.09,right=.98,bottom=.17,top=.86,wspace=.17)
    save(fig,'main_annotation_macro_f1',folds[folds.task.isin(tasks)])
    fig,ax=plt.subplots(figsize=(90/25.4,78/25.4))
    dots(ax,folds,'gtex_broad','macro_f1','Macro F1',(.89,1.005));ax.set_title(TITLES['gtex_broad'])
    fig.subplots_adjust(left=.19,right=.98,bottom=.17,top=.86)
    save(fig,'supplementary_gtex_broad_macro_f1',folds[folds.task.eq('gtex_broad')])
    fig,axes=plt.subplots(2,2,figsize=(WIDTH,135/25.4))
    for row,(metric,label) in enumerate([('accuracy','Accuracy'),('weighted_f1','Weighted F1')]):
        for col,task in enumerate(tasks):
            dots(axes[row,col],folds,task,metric,label if col==0 else '',(.90,.985))
            if row==0:axes[row,col].set_title(TITLES[task])
    fig.subplots_adjust(left=.09,right=.98,bottom=.10,top=.92,hspace=.40,wspace=.20)
    save(fig,'supplementary_accuracy_weighted_f1',folds[folds.task.isin(tasks)])
    per=pd.read_csv(DATA/'per_class_summary.csv')
    for task in tasks:
        matrix=pd.read_csv(DATA/f'{task}_bridge_confusion_normalized.csv',index_col=0)
        fig,ax=plt.subplots(figsize=(WIDTH,WIDTH))
        image=ax.imshow(matrix.to_numpy(),cmap='Blues',vmin=0,vmax=1,aspect='auto',interpolation='nearest')
        ax.set_xticks(range(len(matrix.columns)),matrix.columns,rotation=90,fontsize=7)
        ax.set_yticks(range(len(matrix)),matrix.index,fontsize=7)
        ax.tick_params(length=1.5,pad=2)
        ax.set(xlabel='Predicted label',ylabel='True label',title='BRIDGE — '+TITLES[task])
        cb=fig.colorbar(image,ax=ax,fraction=.03,pad=.025);cb.set_label('Row-normalized fraction')
        # Long tissue names require generous fixed margins at readable font size.
        fig.subplots_adjust(left=.34 if task=='gtex_detailed' else .13,right=.92,bottom=.34 if task=='gtex_detailed' else .14,top=.93)
        save(fig,f'{task}_bridge_confusion_normalized',matrix.rename_axis('true_label').reset_index())
        q=per[per.task.eq(task)&per.representation.eq('bridge')].sort_values(['f1_mean','class'])
        fig,ax=plt.subplots(figsize=(WIDTH,max(105,len(q)*3.9)/25.4))
        ax.barh(q['class'],q.f1_mean,xerr=q.f1_sd,color=COLORS['bridge'],capsize=1.5,error_kw={'elinewidth':.7,'capthick':.7})
        ax.set(xticks=[0,.25,.5,.75,1],xlim=(0,1.08),xlabel='Per-class F1',title='BRIDGE — '+TITLES[task])
        ax.grid(axis='x',color='#EEEEEE',lw=.5);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
        fig.subplots_adjust(left=.34 if task=='gtex_detailed' else .13,right=.98,bottom=.08,top=.95)
        save(fig,f'{task}_bridge_per_class_f1',q)
    assert hashes=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (FIGURES/'classification_plot_provenance.json').write_text(json.dumps({'script':'pipeline/plot_classification.py','source_sha256':hashes,'font':font,'width_mm':183,'broad_width_mm':90,'font_pt':7.5,'ticks_pt':7,'heading_pt':8.5,'colors':COLORS,'png_dpi':600,'intervals':'Mean ± sample SD across five fixed folds; not confidence intervals.','removed_annotations':'Uniform-class reference text removed; 1/K is not a general macro-F1 chance baseline.','source_values_unchanged':True},indent=2)+'\n')
    print('Seven figures regenerated from unchanged saved data.')

if __name__=='__main__':main()
