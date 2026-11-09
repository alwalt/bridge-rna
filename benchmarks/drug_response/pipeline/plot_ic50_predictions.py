#!/usr/bin/env python3
"""Plot saved held-out IC50 predictions; no fitting or inference.

First build: --research-root ORIGINAL_RESEARCH_DIRECTORY.
Subsequent builds reuse the validated compressed plotted-value CSV.
"""
from pathlib import Path
import argparse,hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
ROOT=Path(__file__).resolve().parents[1];FIG=ROOT/'figures';STEM='bridge_oof_ic50_regression'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--research-root',type=Path);args=parser.parse_args()
    FIG.mkdir(exist_ok=True);value_path=FIG/f'{STEM}_values.csv.gz';provenance_path=FIG/f'{STEM}_provenance.json'
    if args.research_root:
        research=args.research_root.resolve();obs_path=research/'results/observation_manifest.parquet';sample_path=research/'results/cell_line_splits.csv';pred_path=research/'work/bridge_45.6m_oof_predictions.npy'
        obs=pd.read_parquet(obs_path);samples=pd.read_csv(sample_path);pred=np.load(pred_path,mmap_mode='r')
        screens=sorted(obs.screen_id.unique());assert pred.shape==(700,355) and samples.depmap_id.is_unique
        sample_index=dict(zip(samples.depmap_id,range(len(samples))));screen_index={s:i for i,s in enumerate(screens)}
        saved=pd.read_csv(ROOT/'data/tables/per_drug_metrics.csv');saved=saved[saved.representation.eq('bridge_45.6m')].set_index('screen_id')
        # Verify array ordering and saved metrics on all screens before extracting panels.
        maximum_error=0.;parts=[]
        for screen,frame in obs.groupby('screen_id'):
            assert frame.ModelID.is_unique
            indices=frame.ModelID.map(sample_index);assert indices.notna().all()
            truth=frame.IC50.to_numpy(np.float32);estimate=np.asarray(pred[indices.to_numpy(int),screen_index[screen]])
            assert np.isfinite(estimate).all() and len(truth)==saved.loc[screen,'n']
            error=abs(float(pearsonr(truth,estimate).statistic)-saved.loc[screen,'pcc']);maximum_error=max(maximum_error,error)
            assert error<1e-6,(screen,error)
            parts.append(pd.DataFrame({'screen_id':screen,'drug_name':frame['Drug Name'].to_numpy(),'dataset_version':frame['Dataset Version'].to_numpy(),'cell_line_id':frame.ModelID.to_numpy(),'fold':samples.iloc[indices.to_numpy(int)].fold.to_numpy(),'observed_ic50':truth,'predicted_ic50':estimate}))
        points=pd.concat(parts,ignore_index=True)
        provenance={'source_files':{str(p):sha(p) for p in [obs_path,sample_path,pred_path]},'all_355_screens_metric_max_abs_error':maximum_error,'prediction_type':'Saved out-of-fold predictions; whole cell lines held out from downstream training.','source_precision':'Observed targets and predictions are float32, as used by saved evaluation.'}
    else:
        points=pd.read_csv(value_path);provenance=json.loads(provenance_path.read_text())
    assert not points.duplicated(['screen_id','cell_line_id']).any()
    for p in font_manager.findSystemFonts():
        if Path(p).stem.lower().startswith('arial'):font_manager.fontManager.addfont(p)
    font=font_manager.findfont('Arial',fallback_to_default=False)
    plt.rcParams.update({'font.family':'Arial','font.size':7.5,'axes.titlesize':8.5,'axes.titleweight':'normal','axes.labelsize':7.5,'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'})
    assert len(points)==212299 and points.screen_id.nunique()==355 and points.cell_line_id.nunique()==700
    fig,ax=plt.subplots(figsize=(183/25.4,160/25.4))
    lo=float(np.floor(points[['observed_ic50','predicted_ic50']].min().min()))
    hi=float(np.ceil(points[['observed_ic50','predicted_ic50']].max().max()));padding=(hi-lo)*.025
    ax.scatter(points.observed_ic50,points.predicted_ic50,s=1,alpha=.025,color='#0072B2',edgecolors='none',rasterized=True)
    ax.plot([lo-padding,hi+padding],[lo-padding,hi+padding],color='#777777',lw=.7,ls='--',zorder=2)
    ax.set(xlim=(lo-padding,hi+padding),ylim=(lo-padding,hi+padding),aspect='equal',
           xlabel='Observed IC50 (released scale)',ylabel='Predicted IC50 (released scale)')
    ax.set_title('BRIDGE held-out drug-response predictions',pad=10)
    r=float(pearsonr(points.observed_ic50,points.predicted_ic50).statistic)
    ax.text(.04,.96,f'All-point r = {r:.3f}\nn = {len(points):,}',transform=ax.transAxes,ha='left',va='top',fontsize=7)
    ax.spines[['top','right']].set_visible(False);ax.tick_params(width=.6,length=3)
    fig.subplots_adjust(left=.12,right=.98,bottom=.13,top=.91)
    fig.canvas.draw()
    for ext in ['png','pdf','svg']:fig.savefig(FIG/f'{STEM}.{ext}',dpi=600,facecolor='white')
    plt.close(fig)
    points.to_csv(value_path,index=False,compression='gzip')
    provenance.update({'selection_rule':'All 212299 measured cell-line–screen responses; no screen selection or subsampling.',
        'evaluation_axis':'One measured cell-line–drug-screen response per point; all 355 screens on shared raw axes.',
        'transform':'None; no standardization, averaging, or recalibration of observed/predicted IC50.',
        'all_point_pearson':r,'all_point_pearson_interpretation':'Includes between-screen differences; not mean within-screen PCC.',
        'equality_line':'y=x; no fitted regression line or prediction recalibration.',
        'input_boundary':'Released IC50 target scale retained; no physical concentration units or additional log transform asserted.',
        'point_rendering':'Every point drawn; point layer rasterized at 600 dpi in PDF/SVG to avoid excessive vector size; axes/text remain vector.',
        'font':font,'width_mm':183,'height_mm':160,'font_pt':7.5,'tick_font_pt':7,'heading_font_pt':8.5,'png_dpi':600,'script':'pipeline/plot_ic50_predictions.py'})
    provenance.pop('selection',None)
    provenance_path.write_text(json.dumps(provenance,indent=2)+'\n')
    print(f'All {len(points):,} points plotted; all-point Pearson r={r:.6f}')

if __name__=='__main__':main()
