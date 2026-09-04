#!/usr/bin/env python3
"""Compare edgeR logFC response geometry with frozen BridgeRNA response geometry."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from scipy.stats import spearmanr

HERE=Path(__file__).resolve().parents[1]; OUT=HERE/'results/de_response_geometry'; OUT.mkdir(parents=True,exist_ok=True)
ORDER=['GSE108643','GSE86931','GSE126962','GSE132520','GSE151066','GSE71972','GSE87748','GSE97718']
PATTERN={**{g:'A' for g in ORDER[:4]},'GSE151066':'Intermediate',**{g:'B' for g in ORDER[5:]}}
SPECIES={g:('human' if g in {'GSE108643','GSE86931','GSE151066','GSE71972','GSE87748'} else 'mouse') for g in ORDER}

def load_vector(gse,mode):
    species=SPECIES[gse]; cid=f'{species}_{gse}'; x=pd.read_parquet(HERE/f'results/full_transcriptome_de/{cid}_full_de.parquet').query('tested').copy()
    if mode=='ortholog': x=x[x.in_bridgerna_vocabulary].dropna(subset=['bridgerna_gene_symbol']); key='bridgerna_gene_symbol'
    else: key='gene_id'
    return x.sort_values('absolute_effect_rank').drop_duplicates(key).set_index(key).log2_fold_change

def matrix(gses,mode):
    vectors={g:load_vector(g,mode) for g in gses}; common=sorted(set.intersection(*(set(v.index) for v in vectors.values())))
    x=np.stack([vectors[g].loc[common].to_numpy(float) for g in gses]); norms=np.linalg.norm(x,axis=1); cosine=(x@x.T)/np.outer(norms,norms)
    rank=np.eye(len(gses))
    for i in range(len(gses)):
      for j in range(i): rank[i,j]=rank[j,i]=spearmanr(x[i],x[j]).statistic
    return x,common,cosine,rank

def pair_summary(name,gses,metric,values):
    rows=[]
    for i in range(len(gses)):
      for j in range(i):
        a,b=PATTERN[gses[i]],PATTERN[gses[j]]
        category='within_A' if a==b=='A' else ('within_B' if a==b=='B' else ('between_A_B' if {a,b}=={'A','B'} else 'involving_intermediate'))
        rows.append({'analysis':name,'metric':metric,'GSE_1':gses[j],'GSE_2':gses[i],'pattern_1':b,'pattern_2':a,'pair_category':category,'similarity':values[i,j]})
    return rows

def heatmap(ax,values,gses,title,vmin=-1,vmax=1):
    im=ax.imshow(values,cmap='RdBu_r',vmin=vmin,vmax=vmax); labels=[f'{g}\n{PATTERN[g]}' for g in gses]; ax.set_xticks(range(len(gses)),labels,rotation=45,ha='right',fontsize=8); ax.set_yticks(range(len(gses)),labels,fontsize=8); ax.set_title(title)
    for i in range(len(gses)):
      for j in range(len(gses)): ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',fontsize=7,color='white' if abs(values[i,j])>.55 else 'black')
    return im

def main():
    all_rows=[]; matrices={}; gene_counts=[]
    analyses={'all_8_orthologs':(ORDER,'ortholog'),'human_only':([g for g in ORDER if SPECIES[g]=='human'],'native'),'mouse_only':([g for g in ORDER if SPECIES[g]=='mouse'],'native')}
    for name,(gses,mode) in analyses.items():
      _,genes,cosine,spearman=matrix(gses,mode); gene_counts.append({'analysis':name,'studies':len(gses),'common_genes':len(genes),'gene_space':'human–mouse one-to-one orthologs' if mode=='ortholog' else f'{SPECIES[gses[0]]} native edgeR genes'})
      matrices[(name,'cosine')]=(gses,cosine); matrices[(name,'spearman')]=(gses,spearman)
      all_rows += pair_summary(name,gses,'cosine',cosine)+pair_summary(name,gses,'spearman',spearman)
      pd.DataFrame(cosine,index=gses,columns=gses).to_csv(OUT/f'{name}_cosine.csv'); pd.DataFrame(spearman,index=gses,columns=gses).to_csv(OUT/f'{name}_spearman.csv')
      fig,axes=plt.subplots(1,2,figsize=(max(9,1.45*len(gses)),max(4.5,.82*len(gses))),layout='constrained')
      im=heatmap(axes[0],cosine,gses,'Cosine similarity'); heatmap(axes[1],spearman,gses,'Spearman correlation'); fig.colorbar(im,ax=axes,shrink=.78,pad=.025,label='Similarity'); fig.suptitle(f'edgeR logFC geometry: {name.replace("_"," ")}',fontweight='bold'); fig.savefig(OUT/f'{name}_heatmaps.png',dpi=350,bbox_inches='tight'); fig.savefig(OUT/f'{name}_heatmaps.pdf',bbox_inches='tight'); plt.close(fig)
    pairs=pd.DataFrame(all_rows); pairs.to_csv(OUT/'de_pairwise_similarities.csv',index=False)
    summary=pairs[pairs.pair_category.ne('involving_intermediate')].groupby(['analysis','metric','pair_category'],as_index=False).agg(mean_similarity=('similarity','mean'),sd_similarity=('similarity','std'),pairs=('similarity','size')); summary.to_csv(OUT/'de_geometry_summary.csv',index=False); pd.DataFrame(gene_counts).to_csv(OUT/'gene_space_summary.csv',index=False)
    # Frozen BridgeRNA response vectors in exactly the same study order.
    meta=pd.read_csv(HERE/'results/response_contrasts.csv'); emb=np.load(HERE/'work/response_effects_bridgerna.npy'); lookup={g:i for i,g in enumerate(meta.GSE)}; b=np.stack([emb[lookup[g]] for g in ORDER]); cosine=(b@b.T)/np.outer(np.linalg.norm(b,axis=1),np.linalg.norm(b,axis=1)); spear=np.eye(len(ORDER))
    for i in range(len(ORDER)):
      for j in range(i): spear[i,j]=spear[j,i]=spearmanr(b[i],b[j]).statistic
    bridge_rows=pair_summary('bridgerna',ORDER,'cosine',cosine)+pair_summary('bridgerna',ORDER,'spearman',spear); bridge=pd.DataFrame(bridge_rows); bridge.to_csv(OUT/'bridgerna_pairwise_similarities.csv',index=False); bridge_summary=bridge[bridge.pair_category.ne('involving_intermediate')].groupby(['analysis','metric','pair_category'],as_index=False).agg(mean_similarity=('similarity','mean'),sd_similarity=('similarity','std'),pairs=('similarity','size')); bridge_summary.to_csv(OUT/'bridgerna_geometry_summary.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(13,6.6),layout='constrained'); im=heatmap(axes[0],cosine,ORDER,'Cosine similarity'); heatmap(axes[1],spear,ORDER,'Spearman correlation'); fig.colorbar(im,ax=axes,shrink=.78,pad=.025,label='Similarity'); fig.suptitle('Frozen BridgeRNA study-level response geometry',fontweight='bold'); fig.savefig(OUT/'bridgerna_response_heatmaps.png',dpi=350,bbox_inches='tight'); fig.savefig(OUT/'bridgerna_response_heatmaps.pdf',bbox_inches='tight'); plt.close(fig)
    combined=pd.concat([summary.query("analysis == 'all_8_orthologs'").assign(representation='edgeR logFC'),bridge_summary.assign(representation='BridgeRNA')]); wide=combined.pivot_table(index=['metric','pair_category'],columns='representation',values='mean_similarity').reset_index(); wide['BridgeRNA_minus_edgeR']=wide.BridgeRNA-wide['edgeR logFC']; wide.to_csv(OUT/'de_vs_bridgerna_geometry_comparison.csv',index=False)
    provenance={'DE_source':'saved edgeR post/exercise-minus-pre/control log2 fold changes','all_8_gene_space':'one-to-one ortholog genes present in BridgeRNA vocabulary and tested in every study','within_species_gene_space':'native Ensembl genes tested in every study of that species','order':ORDER,'pattern_labels':PATTERN,'intermediate_excluded_from_group_means':True,'BridgeRNA':'existing frozen study-level response_effects_bridgerna.npy','statistical_testing':False}
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n'); print(pd.DataFrame(gene_counts).to_string(index=False)); print('\nDE summary\n',summary.to_string(index=False)); print('\nComparison\n',wide.to_string(index=False))
if __name__=='__main__': main()
