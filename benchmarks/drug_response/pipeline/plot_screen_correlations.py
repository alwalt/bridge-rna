#!/usr/bin/env python3
"""Display saved BRIDGE correlations across screens; no model fitting."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def main():
    source=ROOT/'data/tables/per_drug_metrics.csv'
    data=pd.read_csv(source);data=data[data.representation.eq('bridge_45.6m')].sort_values('screen_id').copy()
    assert len(data)==355 and data.screen_id.is_unique and data.pcc.notna().all()
    for path in font_manager.findSystemFonts():
        if Path(path).stem.lower().startswith('arial'):font_manager.fontManager.addfont(path)
    font=font_manager.findfont('Arial',fallback_to_default=False)
    plt.rcParams.update({'font.family':'Arial','font.size':7.5,'axes.titlesize':8.5,'axes.titleweight':'normal','axes.labelsize':7.5,'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'})
    mean=float(data.pcc.mean());output=ROOT/'figures'
    # Retain every screen and the full correlation scale.
    fig,ax=plt.subplots(figsize=(85/25.4,90/25.4))
    edges=np.linspace(-1,1,41)
    counts,_,_=ax.hist(data.pcc,bins=edges,color='#0072B2',edgecolor='white',linewidth=.4)
    assert int(counts.sum())==355
    ax.axvline(mean,color='#222222',lw=1,ls='--',label=f'Mean r = {mean:.3f}')
    ax.set(xlim=(-1,1),xlabel='Within-screen Pearson r',ylabel='Number of drug screens',title='Within-screen correlation distribution')
    ax.legend(frameon=False,fontsize=7);ax.spines[['top','right']].set_visible(False)
    ax.tick_params(width=.6,length=3)
    fig.subplots_adjust(left=.20,right=.97,bottom=.17,top=.86)
    for ext in ['png','pdf','svg']:fig.savefig(output/f'bridge_per_screen_pcc_histogram.{ext}',dpi=600,facecolor='white')
    plt.close(fig)
    pd.DataFrame({'bin_left':edges[:-1],'bin_right':edges[1:],'screens':counts.astype(int)}).to_csv(output/'bridge_per_screen_pcc_histogram_bins.csv',index=False)
    data.to_csv(output/'bridge_per_screen_pcc_histogram_values.csv',index=False)
    (output/'bridge_per_screen_pcc_histogram_provenance.json').write_text(json.dumps({'source':str(source.relative_to(ROOT)),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'screens':355,'mean_pcc':mean,'histogram':'40 equal-width bins over [-1,1], width 0.05; all 355 scores counted.','point_unit':'Within-screen Pearson r between observed and held-out predicted IC50 across cell lines.','width_mm':85,'height_mm':90,'png_dpi':600,'font':font},indent=2)+'\n')
    print(f'Saved all {len(data)} screen scores; mean={mean:.6f}')
if __name__=='__main__':main()
