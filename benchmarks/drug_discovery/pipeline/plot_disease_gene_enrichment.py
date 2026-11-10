#!/usr/bin/env python3
"""Figure-3B-style enrichment from saved top-500 overlaps; no model inference."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact,false_discovery_control
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data/disease_gene_recovery.csv';OUT=ROOT/'figures'

def main():
    data=pd.read_csv(SOURCE)
    q=data.query("module_size == 500 and reference_definition == 'all_v7_associations'").copy()
    assert len(q)==6 and not q.duplicated(['disease','method']).any()
    # Recover each exact 2x2 table from the saved overlap and universe counts.
    recalculated=[]
    for row in q.itertuples():
        a=int(row.known_genes_recovered);b=int(row.module_genes_in_universe)-a
        c=int(row.disgenet_genes_in_universe)-a;d=int(row.universe_genes)-a-b-c
        assert min(a,b,c,d)>=0
        result=fisher_exact([[a,b],[c,d]],alternative='greater')
        assert np.isclose(result.statistic,row.odds_ratio,rtol=1e-12)
        assert np.isclose(result.pvalue,row.pvalue,rtol=1e-8,atol=0)
        recalculated.append(result.pvalue)
    assert np.allclose(false_discovery_control(recalculated,method='bh'),q.fdr,rtol=1e-8,atol=0)
    for p in font_manager.findSystemFonts():
        if Path(p).stem.lower().startswith('arial'):font_manager.fontManager.addfont(p)
    font=font_manager.findfont('Arial',fallback_to_default=False)
    plt.rcParams.update({'font.family':'Arial','font.size':7.5,'axes.titlesize':8.5,'axes.titleweight':'normal','axes.labelsize':7.5,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'})
    order=['Multiple sclerosis — active lesion',"Crohn's disease — ileum",'Systemic lupus erythematosus — blood']
    labels=['Multiple sclerosis · active lesion',"Crohn’s disease · ileum",'Systemic lupus erythematosus · blood']
    fig,ax=plt.subplots(figsize=(183/25.4,80/25.4))
    for method,offset,color,label in [('Bridge',-.18,'#0072B2','BRIDGE'),('DE',.18,'#E69F00','Differential expression')]:
        v=q[q.method.eq(method)].set_index('disease').loc[order]
        bars=ax.barh(np.arange(3)+offset,v.odds_ratio,height=.32,color=color,label=label)
        ax.bar_label(bars,labels=[f'{x:.2f}' for x in v.odds_ratio],padding=3,fontsize=7)
    ax.axvline(1,color='#777777',ls='--',lw=.7,zorder=0)
    ax.set(yticks=np.arange(3),yticklabels=labels,xlim=(0,3.2),xlabel='Disease-gene enrichment (odds ratio)',title='Documented disease-gene enrichment in top-500 gene sets')
    ax.invert_yaxis();ax.spines[['top','right']].set_visible(False)
    ax.legend(frameon=False,loc='lower right');ax.tick_params(width=.6,length=3)
    fig.subplots_adjust(left=.40,right=.98,bottom=.19,top=.84)
    OUT.mkdir(exist_ok=True);stem='task1_disease_gene_enrichment'
    for ext in ['png','pdf','svg']:fig.savefig(OUT/f'{stem}.{ext}',dpi=600,facecolor='white')
    plt.close(fig);q.to_csv(OUT/f'{stem}_values.csv',index=False)
    (OUT/f'{stem}_provenance.json').write_text(json.dumps({'source':str(SOURCE.relative_to(ROOT)),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'reference':'DisGeNET v7 all associations, exact disease CUI; common eligible gene universe within each cohort.','selection':'Prespecified top 500 BRIDGE and matched DE genes; three existing disease cohorts only.','validation':'All six odds ratios and right-sided Fisher p-values reconstructed from saved contingency counts; BH correction over six primary tests reproduced.','comparison_scope':'Per-list enrichment versus background, not a direct BRIDGE–DE significance test. Not an exact replication of the original paper’s 25 comparisons or reference filtering.','font':font,'width_mm':183,'height_mm':80,'png_dpi':600},indent=2)+'\n')
    print(q[['disease','method','known_genes_recovered','odds_ratio','fdr']].to_string(index=False))
if __name__=='__main__':main()
