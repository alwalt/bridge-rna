#!/usr/bin/env python3
"""Gene-level RNA-processing analysis across controlled T cells, RR1, and RR3."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import hypergeom, spearmanr

HERE=Path(__file__).resolve().parents[1];REPO=HERE.parents[1]
OUT=HERE/'results/task4_confounding_profiler/rna_processing_gene_analysis';FIG=OUT/'figures';WORK=HERE/'work/task4_rna_processing_gene_analysis'
PROF=HERE/'results/task4_confounding_profiler';T3=REPO/'benchmarks/osdr_batch_effect_representation';R3=T3/'results';W3=T3/'work'
CTRL=PROF/'controlled_gene_context';EXCESS=PROF/'expression_adjusted_context';FULL=HERE/'results/task4_full_vs_bridge_vocab_expression'
GMT_ROOT=REPO/'benchmarks/cross_species_exercise_response/results/per_study_ranked_gsea'
GMT={'GO:BP':'GO_Biological_Process_2026.gmt','KEGG':'KEGG_2026.gmt','REAC':'Reactome_Pathways_2024.gmt'};SEED=48131
sys.path[:0]=[str(REPO/'benchmarks/tcga_downstream/pipeline'),str(REPO)]
from run_attention_pooling import load_frozen_encoder
from src.fm_embed.vocab import load_canonical_genes
GENES=np.asarray(load_canonical_genes(REPO/'data/ensembl/canonical_genes.csv'))
PAIRS={'RR1':('RR1_OSD48_original_matched','RR1_OSD168_no-ERCC'),
       'RR3-39':('C01_OSD137_original_matched','C01_OSD168_all_ERCC'),
       'RR3-40':('C02_OSD137_original_matched','C02_OSD168_all_ERCC')}

def cosine_rows(a,b):
    return np.sum(a*b,axis=1)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-12)

def rna_universe():
    universe=set(GENES);membership=[]
    for source,file in GMT.items():
        for term,members in gp.parser.read_gmt(path=str(GMT_ROOT/file)).items():
            t=term.upper()
            include=(('RNA PROCESS' in t) or ('RNA SPLIC' in t) or ('MRNA PROCESS' in t) or ('RRNA PROCESS' in t) or
                     ('RNA METABOL' in t) or ('RIBOSOME' in t and any(q in t for q in ['BIOGEN','MATUR','ASSEMB','RRNA'])))
            if include:
                for gene in sorted(set(members)&universe):membership.append({'gene':gene,'source':source,'pathway':term})
    long=pd.DataFrame(membership).drop_duplicates();long.to_csv(OUT/'rna_processing_pathway_membership.csv',index=False)
    grouped=long.groupby('gene').pathway.agg(lambda x:'; '.join(sorted(set(x)))).reset_index(name='pathway_membership')
    grouped.to_csv(OUT/'rna_processing_gene_universe.csv',index=False);return grouped

def controlled_vectors():
    path=HERE/'work/task4_controlled_gene_context/controlled_context_displacements.float16.dat';shape=(40,15165,512)
    if not path.exists() or path.stat().st_size != np.prod(shape)*2:raise FileNotFoundError('Controlled contextual displacement cache is missing')
    mm=np.memmap(path,dtype='float16',mode='r',shape=shape)
    # Chunking avoids copying the 607-MB cache in one allocation.
    mean=np.empty((15165,512),np.float32)
    for s in range(0,15165,256):mean[s:s+256]=np.asarray(mm[:,s:s+256],np.float32).mean(0)
    return mean

def condition(sample):return 'FLT' if '_FLT_' in sample else 'GC' if '_GC_' in sample else 'other'

def osdr_vectors(device_name,batch_size):
    cache=WORK/'matched_contextual_responses.float16.dat';names=[n for p in PAIRS.values() for n in p];shape=(len(names),15165,512)
    WORK.mkdir(parents=True,exist_ok=True)
    if cache.exists() and cache.stat().st_size==np.prod(shape)*2:
        return names,np.memmap(cache,dtype='float16',mode='r',shape=shape)
    manifest=pd.read_csv(R3/'sample_manifest.csv');index=dict(zip(manifest.sample_id,range(len(manifest))))
    design=pd.read_csv(R3/'task3_osd168_technical_replication/technical_response_design.csv').set_index('representation')
    x=np.load(W3/'bridgerna_log1p_tpm_inputs.npy',mmap_mode='r');device=torch.device(device_name if torch.cuda.is_available() else 'cpu');model=load_frozen_encoder(device)
    mm=np.memmap(cache,dtype='float16',mode='w+',shape=shape);started=time.time()
    for ni,name in enumerate(names):
        samples=str(design.loc[name,'samples']).split(' | ');ids=[index[s] for s in samples];conditions=[condition(s) for s in samples]
        sums={c:np.zeros((15165,512),np.float32) for c in ['FLT','GC']};counts={'FLT':0,'GC':0}
        for start in range(0,len(ids),batch_size):
            q=ids[start:start+batch_size];cs=conditions[start:start+batch_size]
            values=torch.as_tensor(np.asarray(x[q]),dtype=torch.float32,device=device)
            with torch.no_grad(),torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                h=model._encode_hidden(values).float().cpu().numpy()
            for j,c in enumerate(cs):sums[c]+=h[j];counts[c]+=1
        mm[ni]=((sums['FLT']/counts['FLT'])-(sums['GC']/counts['GC'])).astype(np.float16);mm.flush()
        print(f'[heartbeat] contextual response {ni+1}/{len(names)} {name} elapsed={(time.time()-started)/60:.1f}m',flush=True)
    return names,np.memmap(cache,dtype='float16',mode='r',shape=shape)

def controlled_basis():
    m=pd.read_parquet(HERE/'work/datasets/chen_2020_tcells/manifest.parquet').reset_index(drop=True);z=np.load(HERE/'work/datasets/chen_2020_tcells/bridgerna_embeddings.npy').astype(float);d=[]
    for _,g in m.groupby('pair_id',sort=True):d.append(z[g.index[g.library_prep.eq('ribo')]].mean(0)-z[g.index[g.library_prep.eq('polyA')]].mean(0))
    return np.linalg.svd(np.stack(d),full_matrices=False)[2][:2]

def project_fraction_rows(a,basis):
    coords=a@basis.T;return np.sum(coords**2,axis=1)/np.maximum(np.sum(a**2,axis=1),1e-12)

def build_master(rna,names,D,tcell):
    idx={n:i for i,n in enumerate(names)};genes=pd.DataFrame({'gene':GENES});master=rna.merge(genes,on='gene',validate='one_to_one')
    ctrl=pd.read_parquet(CTRL/'controlled_gene_sensitivity.parquet').rename(columns={'gene_symbol':'gene','mean_displacement_magnitude':'tcell_context_magnitude','sensitivity_rank':'tcell_context_rank','sensitivity_score':'tcell_sensitivity_score'})
    master=master.merge(ctrl[['gene','tcell_context_magnitude','tcell_context_rank','tcell_sensitivity_score','loo_directional_consistency','mean_consensus_displacement']],on='gene')
    tcde=pd.read_csv(HERE/'results/task4_confounding_profiler/conventional_expression_baseline/tcell_edger.csv').rename(columns={'gene_symbol':'gene','logFC':'tcell_expression_logFC','FDR':'tcell_expression_FDR'})
    tcex=pd.read_csv(EXCESS/'controlled_tcell_context_excess_ranking.csv').rename(columns={'gene_symbol':'gene'})
    excol='standardized_residual' if 'standardized_residual' in tcex else 'residual_contextual_sensitivity'
    master=master.merge(tcde[['gene','tcell_expression_logFC','tcell_expression_FDR']],on='gene',how='left').merge(tcex[['gene',excol]].rename(columns={excol:'tcell_context_excess'}),on='gene',how='left')
    rrmetrics=pd.read_parquet(PROF/'gene_context_reproducibility.parquet')
    edger=pd.read_csv(FULL/'bridge_vocab_edger.csv.gz');ids={'RR3-39':'C01__OSD-137__RR3__39-day','RR3-40':'C02__OSD-137__RR3__40-day'}
    rr1=pd.read_parquet(EXCESS/'RR1_matched_gene_table.parquet').rename(columns={'gene_symbol':'gene','logFC':'rr1_logFC','FDR':'rr1_FDR','standardized_residual':'rr1_context_excess'})
    master=master.merge(rr1[['gene','rr1_logFC','rr1_FDR','rr1_context_excess']],on='gene',how='left')
    nonde=set(pd.read_csv(EXCESS/'RR1_nonDE_top5pct_context_excess_genes.csv').gene_symbol);master['rr1_context_excess_nonde_top5pct']=master.gene.isin(nonde)
    basis=controlled_basis();arrays={n:np.asarray(D[idx[n]],np.float32) for n in names}
    for label,(original,remeasure) in PAIRS.items():
        prefix={'RR1':'rr1','RR3-39':'rr3_39','RR3-40':'rr3_40'}[label];a,b=arrays[original],arrays[remeasure]
        q=rrmetrics[rrmetrics.comparison.eq(label)].rename(columns={'gene_symbol':'gene','context_reproducibility':f'{prefix}_context_reproducibility'})
        master=master.merge(q[['gene',f'{prefix}_context_reproducibility']],on='gene',how='left');master[f'{prefix}_reversed']=master[f'{prefix}_context_reproducibility']<0
        master[f'{prefix}_original_context_magnitude']=np.linalg.norm(a,axis=1)[np.searchsorted(GENES,master.gene)] if np.all(GENES[:-1]<=GENES[1:]) else master.gene.map(dict(zip(GENES,np.linalg.norm(a,axis=1))))
        master[f'{prefix}_remeasure_context_magnitude']=master.gene.map(dict(zip(GENES,np.linalg.norm(b,axis=1))))
        master[f'{prefix}_pc12_association']=master.gene.map(dict(zip(GENES,project_fraction_rows(a,basis))))
        master[f'tcell_vs_{prefix}_original_cosine']=master.gene.map(dict(zip(GENES,cosine_rows(tcell,a))))
        master[f'tcell_vs_{prefix}_remeasure_cosine']=master.gene.map(dict(zip(GENES,cosine_rows(tcell,b))))
        if label!='RR1':
            q=edger[(edger.contrast_id.eq(ids[label]))&edger.tested].rename(columns={'gene_symbol':'gene','logFC':f'{prefix}_logFC','FDR':f'{prefix}_FDR'})
            master=master.merge(q[['gene',f'{prefix}_logFC',f'{prefix}_FDR']],on='gene',how='left')
    # Scores and ranks within the predefined RNA universe.
    master['rr1_instability_score']=1-master.rr1_context_reproducibility
    for prefix in ['rr3_39','rr3_40']:
        mag=(master[f'{prefix}_original_context_magnitude']+master[f'{prefix}_remeasure_context_magnitude'])/2
        master[f'{prefix}_reproducible_score']=mag*master[f'{prefix}_context_reproducibility'].clip(lower=0)
    for score in ['tcell_sensitivity_score','rr1_instability_score','rr3_39_reproducible_score','rr3_40_reproducible_score']:
        master[score.replace('_score','_rna_rank')]=master[score].rank(ascending=False,method='min').astype(int)
    master.to_csv(OUT/'rna_processing_gene_profiles.csv',index=False);return master,arrays

def overlap_analysis(master):
    score={'T-cell sensitive':'tcell_sensitivity_score','RR1 unstable':'rr1_instability_score','RR3-39 reproducible':'rr3_39_reproducible_score','RR3-40 reproducible':'rr3_40_reproducible_score'};M=len(master);sets={};rows=[]
    for frac in [.05,.10,.20]:
        n=max(1,int(np.ceil(M*frac)))
        for label,column in score.items():sets[(label,frac)]=set(master.nlargest(n,column).gene)
        for i,a in enumerate(score):
            for b in list(score)[i+1:]:
                k=len(sets[(a,frac)]&sets[(b,frac)]);rows.append({'fraction':frac,'set_A':a,'set_B':b,'set_size':n,'overlap':k,'expected':n*n/M,'fold_enrichment':k/(n*n/M),'hypergeom_p':hypergeom.sf(k-1,M,n,n)})
    overlap=pd.DataFrame(rows);overlap.to_csv(OUT/'sensitivity_set_overlaps.csv',index=False)
    corr=[]
    for a,ca in score.items():
        for b,cb in score.items():
            if list(score).index(b)<=list(score).index(a):continue
            corr.append({'ranking_A':a,'ranking_B':b,'genes':M,'spearman':spearmanr(master[ca],master[cb]).statistic})
    pd.DataFrame(corr).to_csv(OUT/'rna_processing_rank_correlations.csv',index=False)
    # Evidence categories use the primary 10% sets; RR3 requires either reproducible timepoint.
    f=.10;t=sets[('T-cell sensitive',f)];r=sets[('RR1 unstable',f)];a=sets[('RR3-39 reproducible',f)];b=sets[('RR3-40 reproducible',f)];rr=a|b
    categories={'T-cell + RR1':t&r,'RR1 + reproducible RR3, weak T-cell':(r&rr)-t,'T-cell + RR1 + reproducible RR3':t&r&rr,
                'Reproducible RR3 only':rr-t-r,'T-cell only':t-r-rr}
    out=[]
    for category,genes in categories.items():
        for gene in sorted(genes):out.append({'category':category,'gene':gene})
    cats=pd.DataFrame(out);cats.to_csv(OUT/'evidence_categories.csv',index=False)
    return overlap,pd.DataFrame(corr),cats,sets

def permutations(master,sets,nperm=10000):
    rng=np.random.default_rng(SEED);M=len(master);rows=[]
    tests=[('T-cell + RR1','T-cell sensitive','RR1 unstable'),('T-cell + RR3-39','T-cell sensitive','RR3-39 reproducible'),('T-cell + RR3-40','T-cell sensitive','RR3-40 reproducible'),('RR3-39 + RR3-40','RR3-39 reproducible','RR3-40 reproducible')]
    for name,a,b in tests:
        for frac in [.05,.10,.20]:
            A=sets[(a,frac)];B=sets[(b,frac)];n=len(A);obs=len(A&B);null=np.empty(nperm,int)
            for i in range(nperm):null[i]=len(set(rng.choice(M,n,False))&set(rng.choice(M,n,False)))
            rows.append({'comparison':name,'fraction':frac,'observed':obs,'null_mean':null.mean(),'empirical_p':(1+(null>=obs).sum())/(nperm+1)})
    out=pd.DataFrame(rows);out.to_csv(OUT/'matched_rna_gene_permutations.csv',index=False);return out

def rr3_state(master):
    a=master.rr3_39_reproducible_score;b=master.rr3_40_reproducible_score;n=max(1,int(np.ceil(len(master)*.10)));A=set(master.nlargest(n,'rr3_39_reproducible_score').gene);B=set(master.nlargest(n,'rr3_40_reproducible_score').gene)
    shared=master[master.gene.isin(A&B)];cross=(master.tcell_vs_rr3_39_original_cosine*0) # initialized with correct length
    # Direct RR3-39 versus RR3-40 context direction is calculated from saved/recomputed vectors in main and added later.
    result={'rna_genes':int(len(master)),'top10pct_each':int(n),'top10pct_overlap':int(len(A&B)),'top10pct_jaccard':float(len(A&B)/len(A|B)),'score_spearman':float(spearmanr(a,b).statistic),
            'shared_gene_median_within_pair_RR3_39':float(shared.rr3_39_context_reproducibility.median()),'shared_gene_median_within_pair_RR3_40':float(shared.rr3_40_context_reproducibility.median())}
    return result

def figures(master,categories,sets,arrays,tcell):
    # Panel A: top union by best rank, signed relative-to-T-cell direction x column-normalized magnitude.
    rankcols=['tcell_sensitivity_rna_rank','rr1_instability_rna_rank','rr3_39_reproducible_rna_rank','rr3_40_reproducible_rna_rank'];genes=master.assign(best=master[rankcols].min(axis=1)).nsmallest(50,'best').gene
    names=['T-cell','RR1 orig','RR1 remeasure','RR3-39 orig','RR3-39 remeasure','RR3-40 orig','RR3-40 remeasure'];vec=[tcell,arrays[PAIRS['RR1'][0]],arrays[PAIRS['RR1'][1]],arrays[PAIRS['RR3-39'][0]],arrays[PAIRS['RR3-39'][1]],arrays[PAIRS['RR3-40'][0]],arrays[PAIRS['RR3-40'][1]]]
    gi=np.array([np.where(GENES==g)[0][0] for g in genes]);mat=np.zeros((len(gi),len(vec)))
    for j,v in enumerate(vec):
        mag=np.linalg.norm(v[gi],axis=1);direction=np.ones(len(gi)) if j==0 else cosine_rows(tcell[gi],v[gi]);mat[:,j]=direction*(mag/np.maximum(np.median(mag),1e-12))
    mat=np.clip(mat,-3,3);fig,ax=plt.subplots(figsize=(10,13),layout='constrained');im=ax.imshow(mat,aspect='auto',cmap='coolwarm',vmin=-3,vmax=3);ax.set(xticks=range(len(names)),xticklabels=names,yticks=range(len(genes)),yticklabels=genes,title='RNA-processing contextual responses relative to controlled T-cell direction');ax.tick_params(axis='x',rotation=35);fig.colorbar(im,ax=ax,label='Direction × magnitude / column median (clipped)');fig.savefig(FIG/'rna_contextual_response_heatmap.png',dpi=350);fig.savefig(FIG/'rna_contextual_response_heatmap.pdf');plt.close(fig)
    # B rank comparison.
    fig,ax=plt.subplots(figsize=(7,6),layout='constrained');repro=(master.rr3_39_reproducible_rna_rank<=np.ceil(len(master)*.1))|(master.rr3_40_reproducible_rna_rank<=np.ceil(len(master)*.1));ax.scatter(master.tcell_sensitivity_rna_rank,master.rr1_instability_rna_rank,c=np.where(repro,'#E45756','#BBBBBB'),s=12,alpha=.7);ax.set(xlabel='T-cell sensitivity rank',ylabel='RR1 instability rank',title='Controlled sensitivity versus RR1 instability\nred: reproducible RR3 Top 10%');ax.invert_xaxis();ax.invert_yaxis();fig.savefig(FIG/'tcell_vs_rr1_gene_ranks.png',dpi=350);fig.savefig(FIG/'tcell_vs_rr1_gene_ranks.pdf');plt.close(fig)
    # C RR3 ranks.
    fig,ax=plt.subplots(figsize=(7,6),layout='constrained');ax.scatter(master.rr3_39_reproducible_rna_rank,master.rr3_40_reproducible_rna_rank,s=13,alpha=.65,color='#4C78A8');ax.set(xlabel='RR3-39 reproducible rank',ylabel='RR3-40 reproducible rank',title='RR3 RNA-processing gene rankings');ax.invert_xaxis();ax.invert_yaxis();fig.savefig(FIG/'rr3_39_vs_40_gene_ranks.png',dpi=350);fig.savefig(FIG/'rr3_39_vs_40_gene_ranks.pdf');plt.close(fig)
    # D equivalent to UpSet: exact category counts.
    counts=categories.groupby('category').size().sort_values();fig,ax=plt.subplots(figsize=(9,5),layout='constrained');ax.barh(counts.index,counts,color='#72B7B2');ax.set(xlabel='RNA-processing genes',title='Top-10% evidence categories');[ax.text(v+1,i,str(v),va='center') for i,v in enumerate(counts)];fig.savefig(FIG/'rna_gene_evidence_categories.png',dpi=350);fig.savefig(FIG/'rna_gene_evidence_categories.pdf');plt.close(fig)
    # Top 20 evidence table as CSV and rendered figure.
    top=master.assign(best_rank=master[rankcols].min(axis=1)).nsmallest(20,'best_rank');top.to_csv(OUT/'top20_informative_rna_processing_genes.csv',index=False)
    cols=['gene','tcell_sensitivity_rna_rank','rr1_instability_rna_rank','rr3_39_reproducible_rna_rank','rr3_40_reproducible_rna_rank'];fig,ax=plt.subplots(figsize=(10,7),layout='constrained');ax.axis('off');tbl=ax.table(cellText=top[cols].values,colLabels=['Gene','T-cell rank','RR1 rank','RR3-39 rank','RR3-40 rank'],loc='center',cellLoc='center');tbl.auto_set_font_size(False);tbl.set_fontsize(9);tbl.scale(1,1.35);ax.set_title('Top informative RNA-processing genes across analyses',pad=16);fig.savefig(FIG/'top20_rna_processing_genes.png',dpi=350);fig.savefig(FIG/'top20_rna_processing_genes.pdf');plt.close(fig)

def summary(master,overlap,corr,categories,state):
    top=lambda col:', '.join(master.nlargest(10,col).gene)
    f10=overlap[overlap.fraction.eq(.10)];tr=f10[(f10.set_A.eq('T-cell sensitive'))&(f10.set_B.eq('RR1 unstable'))].iloc[0];t39=f10[(f10.set_A.eq('T-cell sensitive'))&(f10.set_B.eq('RR3-39 reproducible'))].iloc[0];t40=f10[(f10.set_A.eq('T-cell sensitive'))&(f10.set_B.eq('RR3-40 reproducible'))].iloc[0]
    text=f'''# RNA-processing gene analysis

- Exact Bridge RNA-processing universe: **{len(master):,} genes** from the pre-existing GO BP/KEGG/Reactome definitions.
- Most controlled-sensitive genes: {top('tcell_sensitivity_score')}.
- Most RR1-unstable genes: {top('rr1_instability_score')}.
- Most reproducible RR3-39 genes: {top('rr3_39_reproducible_score')}.
- Most reproducible RR3-40 genes: {top('rr3_40_reproducible_score')}.
- At Top 10%, T-cell/RR1 overlap is {int(tr.overlap)} genes (expected {tr.expected:.1f}; {tr.fold_enrichment:.2f}x); T-cell/RR3-39 is {int(t39.overlap)} and T-cell/RR3-40 is {int(t40.overlap)}.
- RR3-39/RR3-40 reproducible-score Spearman is {state['score_spearman']:.3f}; their Top-10% overlap is {state['top10pct_overlap']} (Jaccard {state['top10pct_jaccard']:.3f}).

The analysis supports a **mixed model**: statistically enriched subsets of the same RNA-processing machinery recur, but overlap is incomplete and rankings differ substantially. BridgeRNA also maps different gene-level contextual transformations onto a shared higher-order RNA-processing-associated organization. RR1 genes that are simultaneously controlled-sensitive, unstable, and reproducible in at least one RR3 state are the most difficult to interpret; they are not labeled artifacts. RR3 sample sizes are only 2 FLT/2 GC in each matched technical comparison, and all gene rankings are descriptive even where set overlap exceeds a matched-gene null.
'''
    (OUT/'rna_processing_gene_summary.md').write_text(text);return text

def main(args):
    OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True);rna=rna_universe();tcell=controlled_vectors();names,D=osdr_vectors(args.device,args.batch_size);master,arrays=build_master(rna,names,D,tcell)
    # Exact direction between the two reproducible RR3 configurations.
    m={g:i for i,g in enumerate(GENES)};idx=np.array([m[g] for g in master.gene]);master['rr3_39_vs_40_original_context_cosine']=cosine_rows(arrays[PAIRS['RR3-39'][0]][idx],arrays[PAIRS['RR3-40'][0]][idx]);master.to_csv(OUT/'rna_processing_gene_profiles.csv',index=False)
    overlap,corr,categories,sets=overlap_analysis(master);perm=permutations(master,sets);state=rr3_state(master);state['median_RR3_39_vs_40_original_gene_cosine']=float(master.rr3_39_vs_40_original_context_cosine.median());state['fraction_RR3_39_vs_40_opposite']=float((master.rr3_39_vs_40_original_context_cosine<0).mean());(OUT/'rr3_39_vs_40_state_test.json').write_text(json.dumps(state,indent=2)+'\n')
    figures(master,categories,sets,arrays,tcell);text=summary(master,overlap,corr,categories,state)
    (OUT/'provenance.json').write_text(json.dumps({'frozen_bridge':True,'contextual_inference_rerun_only_because_cross_experiment_512D_gene_vectors_were_not_saved':True,'osdr_profiles':34,'controlled_context_cache_reused':True,'genes':len(master),'top_set_fractions':[.05,.10,.20],'permutations':10000,'seed':SEED,'caveat':'Technical-reference overlap is not artifact or causality.'},indent=2)+'\n');print(text);print('[complete]',OUT)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');p.add_argument('--batch-size',type=int,default=2);main(p.parse_args())
