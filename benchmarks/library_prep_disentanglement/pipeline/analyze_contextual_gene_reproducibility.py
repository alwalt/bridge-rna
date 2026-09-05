#!/usr/bin/env python3
"""Exploratory contextual-gene reproducibility for Task 3 technical replications."""
from __future__ import annotations
import argparse, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import gseapy as gp

ROOT=Path(__file__).resolve().parents[1]; REPO=ROOT.parents[1]
OUT=ROOT/'results/task4_confounding_profiler'; FIG=OUT/'figures'; WORK=ROOT/'work/task4_contextual_gene_reproducibility'
T3=REPO/'benchmarks/osdr_batch_effect_representation'; R3=T3/'results'; W3=T3/'work'
sys.path.insert(0,str(REPO/'benchmarks/tcga_downstream/pipeline')); sys.path.insert(0,str(REPO))
from run_attention_pooling import load_frozen_encoder
from src.fm_embed.vocab import load_canonical_genes
GENES=np.array(load_canonical_genes(REPO/'data/ensembl/canonical_genes.csv'))
PAIRS={'RR1':('RR1_OSD48_original_matched','RR1_OSD168_no-ERCC'),
       'RR3-39':('C01_OSD137_original_matched','C01_OSD168_all_ERCC'),
       'RR3-40':('C02_OSD137_original_matched','C02_OSD168_all_ERCC')}

def condition(s): return 'FLT' if '_FLT_' in s else 'GC' if '_GC_' in s else 'other'

def response(model,x,indices,conditions,device,batch):
    sums={c:np.zeros((len(GENES),512),np.float32) for c in ['FLT','GC']}; counts={c:0 for c in sums}
    for start in range(0,len(indices),batch):
        ids=indices[start:start+batch]; cs=conditions[start:start+batch]
        values=torch.as_tensor(np.asarray(x[ids]),dtype=torch.float32,device=device)
        with torch.no_grad(),torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
            h=model._encode_hidden(values).float().cpu().numpy()
        for j,c in enumerate(cs): sums[c]+=h[j];counts[c]+=1
    assert min(counts.values())>0
    return sums['FLT']/counts['FLT']-sums['GC']/counts['GC']

def figures(metrics,summary):
    colors=['#CC3311','#0077BB','#009988']; order=list(PAIRS)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    s=summary.set_index('comparison').loc[order]
    for ax,col,title in [(axes[0],'median_context_reproducibility','Median gene-context reproducibility'),(axes[1],'fraction_context_reproducibility_lt_0','Reversed contextual responses')]:
        bars=ax.bar(order,s[col],color=colors);ax.axhline(0,color='black',lw=.8);ax.set(title=title,ylabel='Cosine' if 'median' in col else 'Fraction of genes')
        for b,v in zip(bars,s[col]):ax.text(b.get_x()+b.get_width()/2,v+(.02 if v>=0 else -.05),f'{v:.3f}',ha='center')
    fig.savefig(FIG/'gene_context_reproducibility_bars.png',dpi=300);fig.savefig(FIG/'gene_context_reproducibility_bars.pdf');plt.close(fig)
    for col,title,path,ylabel in [
        ('median_context_reproducibility','Median Gene Context Reproducibility','gene_context_reproducibility_bars','Cosine'),
        ('fraction_context_reproducibility_lt_0','Fraction of genes with reversed contextual response','gene_context_reversal_fraction_bars','Fraction of genes')]:
        fig,ax=plt.subplots(figsize=(7,4.5),layout='constrained');bars=ax.bar(order,s[col],color=colors);ax.axhline(0,color='black',lw=.8)
        ax.set(ylabel=ylabel,title=title,ylim=(-.2,1) if col.startswith('median') else (0,1))
        for b,v in zip(bars,s[col]):ax.text(b.get_x()+b.get_width()/2,v+(.025 if v>=0 else -.055),f'{v:.3f}',ha='center',fontweight='bold')
        fig.savefig(OUT/f'{path}.png',dpi=300);fig.savefig(OUT/f'{path}.pdf');plt.close(fig)
    rr=metrics.query("comparison=='RR1'").nsmallest(20,'context_reproducibility').sort_values('context_reproducibility')
    fig,ax=plt.subplots(figsize=(8,6),layout='constrained');ax.barh(rr.gene_symbol,rr.context_reproducibility,color='#CC3311');ax.axvline(0,color='black',lw=.8)
    ax.set(xlabel='Gene Context Reproducibility (cosine)',title='RR1 contextually least reproducible genes');fig.savefig(FIG/'RR1_contextually_unstable_genes.png',dpi=300);fig.savefig(FIG/'RR1_contextually_unstable_genes.pdf');plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');p.add_argument('--batch-size',type=int,default=1);a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(exist_ok=True);WORK.mkdir(parents=True,exist_ok=True)
    device=torch.device(a.device if torch.cuda.is_available() else 'cpu');model=load_frozen_encoder(device)
    manifest=pd.read_csv(R3/'sample_manifest.csv');index=dict(zip(manifest.sample_id,range(len(manifest))))
    design=pd.read_csv(R3/'task3_osd168_technical_replication/technical_response_design.csv').set_index('representation')
    x=np.load(W3/'bridgerna_log1p_tpm_inputs.npy',mmap_mode='r');assert x.shape[1]==len(GENES)
    needed=sum(([u,v] for u,v in PAIRS.values()),[]); responses={};started=time.time()
    print(f'[start] contextual inference representations={len(needed)} unique_samples=34 device={device}',flush=True)
    for n,name in enumerate(needed,1):
        ids=str(design.loc[name,'samples']).split(' | ');idx=[index[s] for s in ids];cs=[condition(s) for s in ids]
        responses[name]=response(model,x,idx,cs,device,a.batch_size)
        print(f'[heartbeat] response={n}/{len(needed)} {name} samples={len(ids)} elapsed={(time.time()-started)/60:.1f}m',flush=True)
    rows=[]
    for pair,(u,v) in PAIRS.items():
        A,B=responses[u],responses[v];dot=np.sum(A*B,axis=1);na=np.linalg.norm(A,axis=1);nb=np.linalg.norm(B,axis=1)
        cos=dot/np.maximum(na*nb,1e-12);dist=np.linalg.norm(A-B,axis=1);relative=dist/np.maximum(na+nb,1e-12)
        rows.append(pd.DataFrame({'comparison':pair,'gene_symbol':GENES,'context_reproducibility':cos,
            'context_discrepancy_norm':dist,'normalized_context_discrepancy':relative,'original_response_norm':na,'remeasurement_response_norm':nb}))
    metrics=pd.concat(rows,ignore_index=True);metrics.to_parquet(OUT/'gene_context_reproducibility.parquet',index=False)
    summary=metrics.groupby('comparison').agg(genes=('gene_symbol','size'),median_context_reproducibility=('context_reproducibility','median'),
        q25=('context_reproducibility',lambda z:z.quantile(.25)),q75=('context_reproducibility',lambda z:z.quantile(.75)),
        fraction_context_reproducibility_lt_0=('context_reproducibility',lambda z:(z<0).mean()),
        fraction_context_reproducibility_lt_0_5=('context_reproducibility',lambda z:(z<.5).mean()),
        median_normalized_context_discrepancy=('normalized_context_discrepancy','median')).reset_index()
    summary.to_csv(OUT/'gene_context_reproducibility_summary.csv',index=False)
    metrics.sort_values(['comparison','context_reproducibility']).groupby('comparison').head(100).to_csv(OUT/'contextually_unstable_genes.csv',index=False)
    # Overlap with cached IG and controlled-signature Top-100 sets.
    ranks=pd.read_parquet(RESULTS:=ROOT/'results/task4_gene_attribution_diagnostic/per_response_gene_rankings.parquet')
    sets=pd.read_csv(ROOT/'results/task4_gene_attribution_diagnostic/interpretation_gene_sets.csv')
    overlap=[]
    for pair,(u,v) in PAIRS.items():
        unstable=set(metrics.query('comparison==@pair').nsmallest(100,'context_reproducibility').gene_symbol)
        ig=set(ranks[(ranks.response.isin([u,v]))&(ranks.rank_absolute<=100)].gene_symbol_human)
        controlled=set(ranks[(ranks.response=='controlled_tcell_ribo_minus_polyA')&(ranks.rank_absolute<=100)].gene_symbol_human)
        pathway=set(sets[sets.gene_set.str.startswith(pair.replace('-','_'),na=False)].gene_symbol)
        for label,target in [('IG_Top100_union',ig),('controlled_Top100',controlled),('existing_interpretation_sets',pathway)]:
            overlap.append({'comparison':pair,'unstable_top_n':100,'reference':label,'reference_genes':len(target),'overlap':len(unstable&target),'genes':';'.join(sorted(unstable&target))})
    pd.DataFrame(overlap).to_csv(OUT/'contextual_gene_overlap.csv',index=False)
    # Ranked GSEA uses normalized discrepancy, bounded by the triangle inequality,
    # to avoid prioritizing negligible vectors solely because their cosine is noisy.
    gmt_root=REPO/'benchmarks/cross_species_exercise_response/results/per_study_ranked_gsea';enr=[]
    for pair in PAIRS:
        rank=metrics.query('comparison==@pair')[['gene_symbol','normalized_context_discrepancy']].sort_values('normalized_context_discrepancy',ascending=False)
        for source,file in [('GO:BP','GO_Biological_Process_2026.gmt'),('KEGG','KEGG_2026.gmt'),('REAC','Reactome_Pathways_2024.gmt')]:
            pre=gp.prerank(rnk=rank,gene_sets=str(gmt_root/file),min_size=10,max_size=500,permutation_num=250,threads=8,seed=42631,outdir=None,verbose=False)
            z=pre.res2d.rename(columns={'Term':'pathway','NES':'nes','FDR q-val':'fdr','NOM p-val':'nominal_p'});z['comparison']=pair;z['source']=source;enr.append(z[['comparison','source','pathway','nes','fdr','nominal_p']])
    enrichment=pd.concat(enr,ignore_index=True);enrichment.to_csv(OUT/'contextual_gene_pathway_enrichment.csv',index=False)
    sig=enrichment[enrichment.fdr<.05].sort_values(['comparison','fdr']);unstable_sig=sig[sig.nes>0].copy()
    unstable_sig.groupby('comparison').head(10).to_csv(OUT/'contextual_gene_pathway_top.csv',index=False)
    if len(unstable_sig):
        q=unstable_sig.groupby('pathway').fdr.min().nsmallest(15).index;plot=unstable_sig[unstable_sig.pathway.isin(q)].copy();plot['value']=-np.log10(plot.fdr.clip(lower=1e-300));piv=plot.pivot_table(index='pathway',columns='comparison',values='value',aggfunc='max').fillna(0).reindex(columns=list(PAIRS))
        fig,ax=plt.subplots(figsize=(9,6),layout='constrained');im=ax.imshow(piv,aspect='auto',cmap='viridis');ax.set(xticks=range(3),xticklabels=piv.columns,yticks=range(len(piv)),yticklabels=piv.index,title='Pathways enriched among contextually unstable genes');fig.colorbar(im,ax=ax,label='-log10(FDR)');fig.savefig(FIG/'contextual_gene_pathway_enrichment.png',dpi=300);fig.savefig(FIG/'contextual_gene_pathway_enrichment.pdf');plt.close(fig)
    figures(metrics,summary)
    prov={'created_utc':datetime.now(timezone.utc).isoformat(),'frozen_bridge':True,'sample_inputs':str(W3/'bridgerna_log1p_tpm_inputs.npy'),'samples':34,
      'genes':len(GENES),'context_response':'mean contextual embedding FLT minus mean contextual embedding GC','context_reproducibility':'cosine between original and remeasurement gene-context responses',
      'normalized_discrepancy':'norm(A-B)/(norm(A)+norm(B))','gsea_permutations':250,'embeddings_retrained':False,'full_context_tensor_cached':False}
    (OUT/'gene_context_provenance.json').write_text(json.dumps(prov,indent=2));print(summary.to_string(index=False));print(f'[complete] elapsed={(time.time()-started)/60:.1f}m',flush=True)
if __name__=='__main__':main()
