#!/usr/bin/env python3
"""Regenerate survival figures from validated saved metrics without model fitting."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
COLORS={'Bridge 45.6M':'#0072B2','PCA-128':'#E69F00','Full raw expression':'#009E73'}
LABELS={'Bridge 45.6M':'BRIDGE','PCA-128':'PCA-128','Full raw expression':'Raw expression'}
LIMITS=(-.015,1.02)
WIDTH=183/25.4


def manuscript_style():
    for path in font_manager.findSystemFonts():
        if Path(path).stem.lower().startswith('arial'):
            font_manager.fontManager.addfont(path)
    font=font_manager.findfont('Arial',fallback_to_default=False)
    plt.rcParams.update({'font.family':'Arial','font.size':7.5,'axes.titlesize':8.5,
        'axes.labelsize':7.5,'xtick.labelsize':7,'ytick.labelsize':7,
        'legend.fontsize':7,'axes.linewidth':.6,'xtick.major.width':.6,
        'ytick.major.width':.6,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
        'figure.facecolor':'white','axes.facecolor':'white','text.color':'#222222',
        'axes.labelcolor':'#222222','xtick.color':'#333333','ytick.color':'#333333'})
    return font


def save(fig,stem):
    for ext in ['png','pdf','svg']:
        options={'metadata':{'CreationDate':None,'ModDate':None}} if ext=='pdf' else {}
        fig.savefig(ROOT/'figures'/f'{stem}.{ext}',dpi=600,facecolor='white',**options)
    plt.close(fig)


def decorate(ax,order,support):
    y=np.arange(len(order));ax.set_yticks(y,order);ax.set_ylim(len(order)-.4,-.8)
    ax.set_xlim(*LIMITS);ax.set_xticks([0,.25,.5,.75,1],['0','0.25','0.50','0.75','1.00'])
    ax.axvline(.5,color='#777777',ls=(0,(3,3)),lw=.7,zorder=0)
    ax.set_xlabel('Held-out within-cancer C-index')
    for idx,row in enumerate(support.itertuples()):
        ax.text(1.025,idx,f'{int(row.n_events)} / {int(row.n_patients)}',
                transform=ax.get_yaxis_transform(),va='center',fontsize=7,clip_on=False)
    ax.text(1.025,1.015,'Events / N',transform=ax.transAxes,fontsize=7,fontweight='bold',clip_on=False)
    ax.grid(axis='x',color='#EEEEEE',lw=.5);ax.spines[['top','right']].set_visible(False)
    ax.tick_params(length=3)


def intervals(ax,q,y,rep,label=None):
    assert q[['c_index','ci_low','ci_high']].notna().all().all()
    assert q.ci_low.le(q.c_index).all() and q.ci_high.ge(q.c_index).all()
    ax.errorbar(q.c_index,y,xerr=np.vstack([q.c_index-q.ci_low,q.ci_high-q.c_index]),
                fmt='o',ms=3,capsize=2,elinewidth=.8,capthick=.8,
                color=COLORS[rep],label=label)


def draw_bridge_two_columns(data,order):
    bridge=data[data.representation.eq('Bridge 45.6M')].set_index('cancer').loc[order]
    assert len(bridge)==32
    fig,axes=plt.subplots(1,2,figsize=(WIDTH,135/25.4),sharex=True)
    for panel,ax in enumerate(axes):
        q=bridge.iloc[panel*16:(panel+1)*16]
        intervals(ax,q,np.arange(16),'Bridge 45.6M')
        decorate(ax,q.index.tolist(),q)
    fig.suptitle('BRIDGE within-cancer survival discrimination',fontsize=8.5,fontweight='normal',y=.98)
    fig.subplots_adjust(left=.07,right=.90,bottom=.10,top=.91,wspace=.60)
    stem='oof_survival_forest_bridge_only';save(fig,stem)
    plotted=bridge.reset_index();plotted['plot_order']=np.arange(32);plotted['plot_panel']=np.repeat([1,2],16)
    plotted.to_csv(ROOT/'figures'/f'{stem}_values.csv',index=False)
    return stem



def draw_comparison_two_columns(data,order):
    """All three representations, globally ordered by BRIDGE, 16 cancers per panel."""
    fig,axes=plt.subplots(1,2,figsize=(WIDTH,155/25.4),sharex=True)
    offsets={'Bridge 45.6M':-.24,'PCA-128':0.,'Full raw expression':.24}
    for panel,ax in enumerate(axes):
        cancers=order[panel*16:(panel+1)*16]
        for rep,offset in offsets.items():
            q=data[data.representation.eq(rep)].set_index('cancer').loc[cancers]
            intervals(ax,q,np.arange(16)+offset,rep,LABELS[rep])
        support=data[data.representation.eq('Bridge 45.6M')].set_index('cancer').loc[cancers]
        decorate(ax,cancers,support)
    fig.suptitle('Within-cancer survival discrimination',fontsize=8.5,fontweight='normal',y=.98)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.96),ncol=3,frameon=False)
    fig.subplots_adjust(left=.07,right=.90,bottom=.09,top=.89,wspace=.60)
    stem='oof_survival_forest_comparison_two_columns';save(fig,stem)
    plotted=data.copy()
    plotted['plot_order']=plotted.cancer.map({c:i for i,c in enumerate(order)})
    plotted['plot_panel']=plotted.plot_order//16+1
    plotted.sort_values(['plot_order','representation'],kind='stable').to_csv(ROOT/'figures'/f'{stem}_values.csv',index=False)
    return stem


def draw(data,order,bridge_only=False):
    if bridge_only:return draw_bridge_two_columns(data,order)
    fig,ax=plt.subplots(figsize=(WIDTH,230/25.4))
    offsets={'Bridge 45.6M':-.22,'PCA-128':0.,'Full raw expression':.22}
    for rep,offset in offsets.items():
        q=data[data.representation.eq(rep)].set_index('cancer').loc[order]
        intervals(ax,q,np.arange(32)+offset,rep,LABELS[rep])
    support=data[data.representation.eq('Bridge 45.6M')].set_index('cancer').loc[order]
    decorate(ax,order,support)
    fig.suptitle('Within-cancer survival discrimination',fontsize=8.5,fontweight='normal',y=.985)
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.965),ncol=3,frameon=False)
    fig.subplots_adjust(left=.09,right=.86,bottom=.065,top=.93)
    stem='oof_survival_forest';save(fig,stem)
    plotted=data.copy();plotted['plot_order']=plotted.cancer.map({c:i for i,c in enumerate(order)})
    plotted.sort_values(['plot_order','representation'],kind='stable').to_csv(ROOT/'figures'/f'{stem}_values.csv',index=False)
    return stem


def plot_overall():
    data=pd.read_csv(ROOT/'data/survival_head_factorial_metrics.csv')
    heads=['Linear Cox','Cox MLP [512,256]'];reps=list(COLORS);x=np.arange(3);width=.34
    fig,ax=plt.subplots(figsize=(WIDTH,75/25.4))
    for i,(head,color) in enumerate(zip(heads,['#0072B2','#D55E00'])):
        values=[float(data[data.representation.eq(rep)&data['head'].eq(head)].oof_c_index.iloc[0]) for rep in reps]
        bars=ax.bar(x+(i-.5)*width,values,width,label=head,color=color,edgecolor='#333333',linewidth=.35)
        ax.bar_label(bars,labels=[f'{v:.3f}' for v in values],padding=3,fontsize=7)
    ax.set_xticks(x,[LABELS[r] for r in reps]);ax.set_ylabel('Fold-aware OOF C-index');ax.set_ylim(.72,.78)
    ax.set_title('Overall survival discrimination',fontweight='normal',pad=12)
    ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,1.02),ncol=2)
    ax.spines[['top','right']].set_visible(False)
    fig.subplots_adjust(left=.09,right=.98,bottom=.15,top=.80)
    stem='oof_survival_overall';save(fig,stem)
    data.to_csv(ROOT/'figures'/f'{stem}_values.csv',index=False)
    return stem


def main():
    (ROOT/'figures').mkdir(exist_ok=True);font=manuscript_style()
    source=ROOT/'data/survival_5fold_oof_per_cancer.csv';data=pd.read_csv(source)
    assert len(data)==96 and not data.duplicated(['representation','cancer']).any()
    bridge=data[data.representation.eq('Bridge 45.6M')].sort_values(['c_index','cancer'],ascending=[False,True],kind='stable')
    order=bridge.cancer.tolist();assert len(order)==32 and bridge.c_index.is_monotonic_decreasing
    stems=[draw(data,order),draw(data,order,True),draw_comparison_two_columns(data,order),plot_overall()]
    (ROOT/'figures/survival_forest_plot_provenance.json').write_text(json.dumps({
        'source':'data/'+source.name,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'overall_source':'data/survival_head_factorial_metrics.csv',
        'order':order,'sort':'decreasing Bridge 45.6M c_index; alphabetical cancer tie-break',
        'model_display_name':'BRIDGE','source_representation_keys_unchanged':True,
        'x_limits':list(LIMITS),'intervals':'Existing saved ci_low/ci_high; no bootstrapping or refitting.',
        'bridge_only_layout':'Two columns of 16; first 16 left, next 16 right',
        'style':{'font':'Arial','font_file':font,'width_mm':183,'font_pt':7.5,'tick_font_pt':7,
                 'heading_font_pt':8.5,'png_dpi':600,'pdf_fonttype':42,'svg_text':'editable'},
        'removed_annotations':['Sorting subtitle','Rank annotations','Redundant single-model legend'],
        'figures':stems},indent=2)+'\n')
    print('Updated BRIDGE plot labels and manuscript typography; numerical values unchanged.')

if __name__=='__main__':main()
