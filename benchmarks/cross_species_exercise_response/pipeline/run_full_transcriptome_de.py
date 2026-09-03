#!/usr/bin/env python3
"""Run full-transcriptome edgeR and compare DE with frozen IG/context rankings."""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path
import h5py, matplotlib.pyplot as plt, numpy as np, pandas as pd
from scipy.stats import spearmanr

HERE=Path(__file__).resolve().parents[1]; ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from src.fm_embed.vocab import load_canonical_genes
from src.fm_embed.species import load_human_ensembl_to_symbol_map, load_mouse_to_human_symbol_map
OUT=HERE/'results/full_transcriptome_de'; WORK=HERE/'work/full_transcriptome_de'; FIG=OUT/'figures'
H5={'human':ROOT/'data/archs4/human_gene_v2.5.h5','mouse':ROOT/'data/archs4/mouse_gene_v2.5.h5'}
AXES={'Axis A':['GSE108643','GSE86931','GSE126962','GSE132520'],'Axis B':['GSE71972','GSE87748','GSE97718']}
def log(x): print(f"[{time.strftime('%H:%M:%S')}] {x}",flush=True)
def decode(a): return np.array([x.decode() if isinstance(x,bytes) else str(x) for x in a])
def overlap(a,b,n): return len(set(np.argsort(-a)[:n]) & set(np.argsort(-b)[:n]))

def prepare(members, manifest, force):
    design=members.merge(manifest[['GSM','subject_id']],on='GSM',how='left',validate='one_to_one')
    for species, sg in design.groupby('species'):
      with h5py.File(H5[species]) as h:
        access=decode(h['meta/samples/geo_accession'][:]); lookup={x:i for i,x in enumerate(access)}
        ids=decode(h['meta/genes/ensembl_gene'][:]); symbols=decode(h['meta/genes/symbol'][:]); biotypes=decode(h['meta/genes/biotype'][:])
        for cid,g in sg.groupby('contrast_id',sort=True):
          d=WORK/cid; d.mkdir(parents=True,exist_ok=True)
          if force or not (d/'counts.tsv.gz').exists():
            missing=set(g.GSM)-set(lookup)
            if missing: raise KeyError(f'{cid}: missing GSMs: {sorted(missing)}')
            counts=np.stack([np.asarray(h['data/expression'][:,lookup[x]],dtype=np.uint32) for x in g.GSM],axis=1)
            frame=pd.DataFrame(counts,columns=g.GSM.tolist()); frame.insert(0,'gene_id',ids)
            frame.to_csv(d/'counts.tsv.gz',sep='\t',index=False,compression='gzip')
            pd.DataFrame({'gene_id':ids,'gene_symbol':symbols,'biotype':biotypes}).to_parquet(d/'gene_metadata.parquet',index=False)
          g[['GSM','role','subject_id']].fillna('').to_csv(d/'design.csv',index=False)
          log(f'prepared {cid}: {len(ids):,} genes x {len(g):,} samples')

def differential_expression(members,force):
    canonical=set(load_canonical_genes(ROOT/'data/ensembl/canonical_genes.csv'))
    env=os.environ.copy(); env['R_LIBS_USER']=str(HERE/'work/R_library')
    for cid in sorted(members.contrast_id.unique()):
      d=WORK/cid; raw=d/'edgeR.csv'; final=OUT/f'{cid}_full_de.parquet'
      if force or not raw.exists():
        subprocess.run(['Rscript',str(HERE/'pipeline/edger_full_transcriptome.R'),str(d/'counts.tsv.gz'),str(d/'design.csv'),str(raw)],check=True,env=env)
      meta=pd.read_parquet(d/'gene_metadata.parquet'); de=pd.read_csv(raw)
      full=meta.merge(de,on='gene_id',how='left',validate='one_to_one'); full['tested']=full.p_value.notna()
      full['absolute_effect']=full.log2_fold_change.abs(); full['absolute_effect_rank']=full.loc[full.tested,'absolute_effect'].rank(ascending=False,method='min')
      species=members.loc[members.contrast_id.eq(cid),'species'].iloc[0]
      mapper=(load_human_ensembl_to_symbol_map if species=='human' else load_mouse_to_human_symbol_map)(ROOT/'data/ensembl/orthologs_one2one.txt')
      full['bridgerna_gene_symbol']=full.gene_id.str.split('.').str[0].map(mapper)
      full['in_bridgerna_vocabulary']=full.bridgerna_gene_symbol.isin(canonical)
      full.insert(0,'contrast_id',cid); full.insert(1,'species',species); full.to_parquet(final,index=False)
      log(f'edgeR {cid}: {full.tested.sum():,}/{len(full):,} genes tested')

def analyze(members):
    genes=load_canonical_genes(ROOT/'data/ensembl/canonical_genes.csv'); gi={g:i for i,g in enumerate(genes)}
    order=pd.read_csv(HERE/'results/response_contrasts.csv').sort_values('contrast_id').reset_index(drop=True)
    ig=np.load(HERE/'work/latent_axis_attribution/study_integrated_gradient_changes.npy')
    context=np.load(HERE/'work/gene_context_hallmark/exercise_gene_hallmark_deltas.npy',mmap_mode='r')
    context=np.sqrt(np.mean(np.asarray(context,dtype=np.float32)**2,axis=2)); assert len(order)==len(ig)==len(context)
    coverage=[]; comparisons=[]; rows=[]; outside=[]; scores={}
    for i,r in order.iterrows():
      df=pd.read_parquet(OUT/f'{r.contrast_id}_full_de.parquet'); tested=df[df.tested].copy(); inside=int(tested.in_bridgerna_vocabulary.sum())
      rec={'contrast_id':r.contrast_id,'GSE':r.GSE,'species':r.species,'samples':len(members[members.contrast_id.eq(r.contrast_id)]),'genes_available':len(df),'genes_tested':len(tested),'inside_vocabulary':inside,'inside_pct':inside/len(tested),'outside_vocabulary':len(tested)-inside,'outside_pct':1-inside/len(tested)}
      for n in [50,100,500]:
        top=tested.nsmallest(n,'absolute_effect_rank'); count=int((~top.in_bridgerna_vocabulary).sum()); rec[f'top{n}_outside']=count; rec[f'top{n}_outside_pct']=count/len(top)
      coverage.append(rec); outside.append(tested[~tested.in_bridgerna_vocabulary].nsmallest(100,'absolute_effect_rank').assign(GSE=r.GSE))
      x=tested[tested.in_bridgerna_vocabulary].sort_values('absolute_effect_rank').drop_duplicates('bridgerna_gene_symbol').set_index('bridgerna_gene_symbol')
      de=np.zeros(len(genes)); signed=np.zeros(len(genes)); available=np.zeros(len(genes),bool)
      common=[g for g in genes if g in x.index]; idx=np.array([gi[g] for g in common]); de[idx]=x.loc[common,'absolute_effect']; signed[idx]=x.loc[common,'log2_fold_change']; available[idx]=True
      measures={'DE':de,'IG':np.abs(ig[i]),'contextual':context[i]}; scores[r.contrast_id]=measures
      for a,b in [('DE','IG'),('DE','contextual'),('IG','contextual')]:
        q={'contrast_id':r.contrast_id,'GSE':r.GSE,'species':r.species,'comparison':f'{a}_vs_{b}','genes_compared':int(available.sum()),'spearman':spearmanr(measures[a][available],measures[b][available]).statistic}
        for n in [50,100,500]: q[f'top{n}_overlap']=overlap(np.where(available,measures[a],-np.inf),np.where(available,measures[b],-np.inf),n)
        comparisons.append(q)
      rank={k:pd.Series(-v).rank(method='average').to_numpy() for k,v in measures.items()}
      for j in np.where(available)[0]:
        cats=[]
        if rank['DE'][j]<=100 and rank['IG'][j]<=100: cats.append('high_de_high_ig')
        if rank['IG'][j]<=100 and rank['DE'][j]>500: cats.append('modest_de_high_ig')
        if rank['contextual'][j]<=100 and rank['IG'][j]<=100: cats.append('high_context_high_ig')
        rows.append({'contrast_id':r.contrast_id,'GSE':r.GSE,'species':r.species,'gene':genes[j],'log2_fold_change':signed[j],'de_abs':de[j],'de_rank':rank['DE'][j],'ig_change':ig[i,j],'ig_abs':measures['IG'][j],'ig_rank':rank['IG'][j],'contextual_change':context[i,j],'contextual_rank':rank['contextual'][j],'categories':';'.join(cats)})
    coverage=pd.DataFrame(coverage); coverage.to_csv(OUT/'vocabulary_coverage_summary.csv',index=False)
    pd.DataFrame(comparisons).to_csv(OUT/'ranking_comparisons.csv',index=False)
    allgenes=pd.DataFrame(rows); allgenes.to_parquet(OUT/'within_vocabulary_gene_comparisons.parquet',index=False)
    pd.concat(outside).to_parquet(OUT/'top_outside_vocabulary_genes.parquet',index=False)
    categorized=allgenes[allgenes.categories.ne('')].copy()
    categorized.to_csv(OUT/'interesting_gene_categories.csv',index=False)
    groups={'all_8':order.contrast_id.tolist()}
    species_by_gse=dict(zip(order.GSE,order.species))
    for axis,gses in AXES.items(): groups[axis]=[f'{species_by_gse[g]}_{g}' for g in gses]
    consensus=[]
    for label,cids in groups.items():
      q=allgenes[allgenes.contrast_id.isin(cids)].groupby('gene').agg(studies=('contrast_id','nunique'),mean_de_rank=('de_rank','mean'),mean_ig_rank=('ig_rank','mean'),mean_contextual_rank=('contextual_rank','mean'),mean_abs_log2fc=('de_abs','mean'),mean_abs_ig=('ig_abs','mean'),mean_contextual_change=('contextual_change','mean')).reset_index(); q['group']=label; consensus.append(q)
    pd.concat(consensus).to_parquet(OUT/'consensus_gene_table.parquet',index=False)
    # Conserved high-IG genes: intersect species-consensus Top 100 within each fixed axis.
    conserved=[]
    for axis,gses in AXES.items():
      axis_cids=[f'{species_by_gse[g]}_{g}' for g in gses]
      z=allgenes[allgenes.contrast_id.isin(axis_cids)]
      species_ranks={}
      for species in ['human','mouse']:
        q=z[z.species.eq(species)].groupby('gene',as_index=False).ig_rank.mean().sort_values('ig_rank')
        species_ranks[species]=q.set_index('gene').ig_rank
      shared=set(species_ranks['human'].nsmallest(100).index)&set(species_ranks['mouse'].nsmallest(100).index)
      for gene in sorted(shared): conserved.append({'axis':axis,'gene':gene,'human_mean_ig_rank':species_ranks['human'][gene],'mouse_mean_ig_rank':species_ranks['mouse'][gene]})
    pd.DataFrame(conserved).to_csv(OUT/'conserved_high_ig_genes.csv',index=False)
    fig,axs=plt.subplots(2,4,figsize=(16,8))
    for ax,(cid,m) in zip(axs.flat,scores.items()):
      keep=m['DE']>0; rho=spearmanr(m['DE'][keep],m['IG'][keep]).statistic; ax.scatter(m['DE'][keep],m['IG'][keep],s=3,alpha=.25); ax.set(title=f"{cid.split('_')[1]} ρ={rho:.2f}",xlabel='|edgeR log2FC|',ylabel='|IG change|')
    fig.tight_layout(); fig.savefig(FIG/'de_vs_ig.png',dpi=320); fig.savefig(FIG/'de_vs_ig.pdf'); plt.close(fig)
    p=coverage.set_index('GSE')[[f'top{n}_outside_pct' for n in [50,100,500]]]; ax=p.plot.bar(figsize=(11,5)); ax.set(ylabel='Fraction outside BridgeRNA vocabulary',xlabel='',ylim=(0,1)); ax.legend(['Top 50','Top 100','Top 500']); plt.tight_layout(); plt.savefig(FIG/'top_de_vocabulary_coverage.png',dpi=320); plt.savefig(FIG/'top_de_vocabulary_coverage.pdf'); plt.close()
    prior=pd.read_csv(HERE/'results/latent_axis_attribution/attribution_vs_de_summary.csv'); stability=pd.read_csv(HERE/'results/latent_axis_attribution/within_axis_ranking_stability.csv'); verify={'prior_median_de_ig_spearman':float(prior.absolute_attribution_vs_absolute_de_spearman.median()),'prior_median_top100_overlap':float(prior.top100_overlap.median())}
    for axis,z in stability.groupby('axis'):
      cross=z[[species_by_gse[a]!=species_by_gse[b] for a,b in zip(z.GSE_1,z.GSE_2)]]; verify[f'{axis}_cross_species_mean_rank_spearman']=float(cross.absolute_rank_spearman.mean()); verify[f'{axis}_cross_species_mean_top100_overlap']=float(cross.top100_overlap.mean())
    (OUT/'verified_prior_results.json').write_text(json.dumps(verify,indent=2)); log('analysis complete')

def main():
    p=argparse.ArgumentParser(); p.add_argument('--force',action='store_true'); a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True); WORK.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
    members=pd.read_parquet(HERE/'results/contrast_members.parquet'); manifest=pd.read_parquet(HERE/'results/matched_manifest.parquet')
    assert set(members.GSE)=={'GSE108643','GSE151066','GSE71972','GSE86931','GSE87748','GSE126962','GSE132520','GSE97718'}
    prepare(members,manifest,a.force); differential_expression(members,a.force); analyze(members)
    provenance={'source':'ARCHS4 v2.5 gene-level integer raw counts','method':'edgeR quasi-likelihood GLM, TMM, filterByExpr','direction':'post/exercise minus pre/control','paired_human_design':'subject + condition where complete pairing exists','mouse_design':'condition','manufactured_counts_from_TPM':False,'contrasts':'unchanged results/contrast_members.parquet'}
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2))
if __name__=='__main__': main()
