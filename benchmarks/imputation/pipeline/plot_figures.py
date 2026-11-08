"""Journal-style imputation figures drawn exclusively from saved summaries."""
from pathlib import Path
import hashlib
import json
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter

STYLES = {
    ('ours_45.6m','ours'): ('BRIDGE', '#0072B2', '-', 'o'),
    ('bulkformer_50m','bulkformer'): ('BulkFormer 50M', '#009E73', '-', 's'),
    ('bulkformer_147m','bulkformer'): ('BulkFormer 147M', '#D55E00', '-', '^'),
    ('train_gene_mean','ours'): ('Training-gene mean', '#666666', '--', None),
}


def configure_style():
    for path in font_manager.findSystemFonts():
        if Path(path).stem.lower().startswith('arial'):
            font_manager.fontManager.addfont(path)
    font=font_manager.findfont('Arial',fallback_to_default=False)
    plt.rcParams.update({'font.family':'Arial','font.size':7.5,'axes.titlesize':8,
        'axes.labelsize':7.5,'xtick.labelsize':7,'ytick.labelsize':7.5,
        'axes.linewidth':.6,'xtick.major.width':.6,'ytick.major.width':.6,
        'pdf.fonttype':42,'svg.fonttype':'none','figure.facecolor':'white',
        'axes.facecolor':'white','savefig.facecolor':'white'})
    return font


def render(summary, metrics, results, stem):
    configure_style()
    fig,axes=plt.subplots(1,len(metrics),figsize=(183/25.4,3.05),squeeze=False)
    data=summary.query("benchmark == 'native_vocab' and method != 'train_gene_median'")
    titles={'pearson':'Pearson correlation','spearman':'Spearman correlation','mse':'Mean squared error'}
    for index,(ax,metric) in enumerate(zip(axes.flat,metrics)):
        for (method,space),(label,color,linestyle,marker) in STYLES.items():
            frame=data[data.method.eq(method)&data.target_space.eq(space)].sort_values('mask_ratio')
            assert len(frame)==6 and frame.seeds.eq(3).all()
            if method == 'train_gene_mean':
                ax.plot(frame.mask_ratio,frame[f'{metric}_mean'],label=label,color=color,
                        linestyle=(0,(5,3)),linewidth=1.0,dash_capstyle='butt',zorder=2)
                ax.fill_between(frame.mask_ratio,
                    frame[f'{metric}_mean']-frame[f'{metric}_std'],
                    frame[f'{metric}_mean']+frame[f'{metric}_std'],
                    color=color,alpha=.12,linewidth=0,zorder=1)
                continue
            ax.errorbar(frame.mask_ratio,frame[f'{metric}_mean'],yerr=frame[f'{metric}_std'],
                label=label,color=color,linestyle=linestyle,marker=marker,markersize=3.5,
                linewidth=1.15,elinewidth=.7,capsize=2,capthick=.7,zorder=3 if not method.startswith('train_') else 2)
        ax.set(xlabel='Masked genes (%)',ylabel='Native log1p(TPM) squared error' if metric=='mse' else 'Mean per-sample correlation',
               title=titles[metric],xlim=(.11,1.04),xticks=[.15,.30,.50,.70,.90,1.0])
        ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(direction='out',length=3)
        if metric!='mse':
            lower=(data[f'{metric}_mean']-data[f'{metric}_std']).min()
            ax.set_ylim(min(-.045,lower-.025),1.025)
        else:ax.set_ylim(bottom=0)
        if len(metrics)>1:
            ax.text(-.16,1.08,chr(ord('a')+index),transform=ax.transAxes,fontsize=10,fontweight='bold')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.52,.015),ncol=4,
               frameon=False,fontsize=7,handlelength=2.3,columnspacing=1.4,labelspacing=.7)
    fig.subplots_adjust(left=.10,right=.98,top=.87,bottom=.32,wspace=.36)
    results=Path(results)
    for ext in ['png','pdf','svg']:
        fig.savefig(results/f'{stem}.{ext}',dpi=600)
    return fig


def save_style_provenance(results):
    results=Path(results)
    provenance={'font':'Arial','font_file':configure_style(),'width_mm':183,
        'label_size_pt':7.5,'title_size_pt':8,'png_dpi':600,'svg_editable_text':True,
        'style_reference':'benchmarks/gene_function_prediction/pipeline/build_manuscript_pair_figures.py',
        'series':[{'method':m,'target_space':s,'label':v[0],'color':v[1],'linestyle':v[2]} for (m,s),v in STYLES.items()],
        'source_sha256':hashlib.sha256((results/'summary_results.csv').read_bytes()).hexdigest(),
        'plot_generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'error_bars':'Model error bars and baseline shaded band: SD across three mask seeds, not patient confidence intervals',
        'undefined_correlations':'NaN preserved as gaps; never replaced with zero',
        'baseline_display':'Only the BRIDGE-target training mean is drawn; both target-space baselines remain in saved tables.',
        'mse_limitation':'Model-specific gene panels and preprocessing: MSE is in native target units.'}
    (results/'figure_style_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
