#!/usr/bin/env python3
"""Empirical biological-contrast null for the fixed T-cell PC1-2 reference."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
OUT = HERE / "results/task4_pc12_empirical_biological_null"
FIG = OUT / "figures"
T2 = REPO / "benchmarks/cross_species_exercise_response"
T3 = REPO / "benchmarks/osdr_batch_effect_representation"
SEED = 20260906
N_PERM = 1000
N_RANDOM = 2000
N_BOOT = 2000


def fixed_basis() -> np.ndarray:
    m = pd.read_parquet(HERE / "work/datasets/chen_2020_tcells/manifest.parquet").reset_index(drop=True)
    z = np.load(HERE / "work/datasets/chen_2020_tcells/bridgerna_embeddings.npy").astype(float)
    diffs = []
    for _, q in m.groupby("pair_id", sort=True):
        diffs.append(z[q.index[q.library_prep.eq("ribo")]].mean(0) - z[q.index[q.library_prep.eq("polyA")]].mean(0))
    return np.linalg.svd(np.stack(diffs), full_matrices=False)[2][:2]


def response_stats(v: np.ndarray, b: np.ndarray) -> dict:
    norm = float(np.linalg.norm(v)); pc = v @ b.T; proj = pc @ b
    return {"response_norm": norm, "PC1": pc[0], "PC2": pc[1],
            "PC1_2_projected_norm": float(np.linalg.norm(proj)),
            "E_PC12": float(np.dot(proj, proj) / np.dot(v, v)) if norm else np.nan,
            "PC12_angle_degrees": float(np.degrees(np.arctan2(pc[1], pc[0])))}


def coordination(x: np.ndarray) -> dict:
    a = np.abs(x); s = a.sum(); p = a / s if s else np.zeros_like(a)
    top = max(1, int(np.ceil(.01 * len(a))))
    return {"expression_response_rms": float(np.sqrt(np.mean(x*x))),
            "top1pct_abs_effect_fraction": float(np.sort(a)[-top:].sum()/s) if s else np.nan,
            "expression_effective_gene_count": float(1/np.sum(p*p)) if np.any(p) else np.nan}


def load_contrasts(b: np.ndarray):
    rows=[]; objects={}
    # Task 2 curated exercise responses.
    members=pd.read_parquet(T2/'results/contrast_members.parquet'); manifest=pd.read_parquet(T2/'results/matched_manifest.parquet').reset_index(drop=True)
    z=np.load(T2/'work/matched_embeddings.npy'); x=np.load(T2/'work/matched_log1p_tpm_corrected.npy',mmap_mode='r')
    ix=dict(zip(manifest.GSM,range(len(manifest))))
    meta=pd.read_csv(T2/'results/response_contrasts.csv').set_index('contrast_id')
    for cid,q in members.groupby('contrast_id'):
        ia=np.array([ix[g] for g in q[q.role.eq('post_exercise')].GSM]); ib=np.array([ix[g] for g in q[q.role.eq('pre_control')].GSM])
        v=z[ia].mean(0)-z[ib].mean(0); ex=np.asarray(x[ia].mean(0)-x[ib].mean(0)); r=meta.loc[cid]
        row={"contrast_id":cid,"source_benchmark":"Task2 exercise","study":r.GSE,"species":r.species,"tissue":"skeletal muscle","condition_A":"post/exercise","condition_B":"pre/control","n_A":len(ia),"n_B":len(ib),"perturbation_class":"exercise","library_preparation":"not harmonized"}
        row.update(response_stats(v,b));row.update(coordination(ex));rows.append(row);objects[cid]=(z[np.r_[ia,ib]],len(ia),v)
    # Task 3 curated FLT-GC responses.
    members=pd.read_csv(T3/'results/task3b_contrast_sample_membership.csv'); manifest=pd.read_csv(T3/'results/sample_manifest.csv').reset_index(drop=True)
    ann=pd.read_csv(T3/'results/task3c_contrast_annotations.csv').set_index('contrast_id');z=np.load(T3/'work/bridgerna_embeddings.npy');x=np.load(T3/'work/bridgerna_log1p_tpm_inputs.npy',mmap_mode='r');ix=dict(zip(manifest.sample_id,range(len(manifest))))
    for cid,q in members.groupby('contrast_id'):
        ia=np.array([ix[s] for s in q[q.condition.eq('FLT')].sample_id]);ib=np.array([ix[s] for s in q[q.condition.eq('GC')].sample_id]);v=z[ia].mean(0)-z[ib].mean(0);ex=np.asarray(x[ia].mean(0)-x[ib].mean(0));r=ann.loc[cid]
        row={"contrast_id":cid,"source_benchmark":"Task3 spaceflight","study":r.OSD,"species":"mouse","tissue":"liver","condition_A":"flight","condition_B":"ground control","n_A":len(ia),"n_B":len(ib),"perturbation_class":"spaceflight","library_preparation":r.library_preparation}
        row.update(response_stats(v,b));row.update(coordination(ex));rows.append(row);objects[cid]=(z[np.r_[ia,ib]],len(ia),v)
    d=pd.DataFrame(rows);d['primary_null']=d[['n_A','n_B']].min(axis=1).ge(2);return d,objects


def permutation_and_bootstrap(table, objects, b):
    rng=np.random.default_rng(SEED);pr=[];br=[]
    for _,r in table.iterrows():
        zz,na,_=objects[r.contrast_id];n=len(zz)
        for i in range(N_PERM):
            jj=rng.permutation(n);v=zz[jj[:na]].mean(0)-zz[jj[na:]].mean(0);pr.append((r.contrast_id,i,response_stats(v,b)['E_PC12']))
        a,bz=zz[:na],zz[na:]
        for i in range(N_BOOT):
            v=a[rng.integers(len(a),size=len(a))].mean(0)-bz[rng.integers(len(bz),size=len(bz))].mean(0);br.append((r.contrast_id,i,response_stats(v,b)['E_PC12']))
    p=pd.DataFrame(pr,columns=['contrast_id','iteration','E_PC12']);boot=pd.DataFrame(br,columns=['contrast_id','iteration','E_PC12'])
    sm=[]
    for _,r in table.iterrows():
        q=p[p.contrast_id.eq(r.contrast_id)].E_PC12;w=boot[boot.contrast_id.eq(r.contrast_id)].E_PC12
        sm.append({'contrast_id':r.contrast_id,'observed_E_PC12':r.E_PC12,'permutation_p_ge':(1+(q>=r.E_PC12).sum())/(len(q)+1),'permutation_percentile':100*(q<r.E_PC12).mean(),'bootstrap_median':w.median(),'bootstrap_low':w.quantile(.025),'bootstrap_high':w.quantile(.975)})
    return p,boot,pd.DataFrame(sm)


def global_geometry(b):
    z=np.load(T2/'work/hallmark_readout/archs4_bridgerna_embeddings.npy',mmap_mode='r').astype(np.float64);mu=z.mean(0);zc=z-mu
    cov=zc.T@zc/(len(zc)-1);evals,evecs=np.linalg.eigh(cov);o=np.argsort(evals)[::-1];evals=evals[o];v=evecs[:,o].T
    rows=[]
    for k in [2,5,10,20,50]:
        q=v[:k];overlap=float(np.trace((b@q.T)@(q@b.T))/2);ang=np.degrees(subspace_angles(b.T,q.T))
        rows.append({'global_PC_prefix':k,'mean_squared_subspace_overlap':overlap,'smallest_principal_angle_deg':ang.min(),'largest_principal_angle_deg':ang.max()})
    captured=float(np.trace(b@cov@b.T)/np.trace(cov));
    summary=pd.DataFrame(rows);summary['global_variance_fraction_captured_by_Tcell_PC12']=captured
    spectrum=pd.DataFrame({'global_PC':np.arange(1,513),'eigenvalue':evals,'variance_fraction':evals/evals.sum(),'cumulative_variance':np.cumsum(evals/evals.sum())})
    return v,evals,summary,spectrum


def random_subspaces(table, objects, b, global_v, evals):
    rng=np.random.default_rng(SEED+1);rows=[]
    # One common isotropic ensemble, and a variance-weighted ensemble of global-PC planes.
    iso=[]
    for i in range(N_RANDOM):
        q,_=np.linalg.qr(rng.normal(size=(512,2)));iso.append(q.T)
    weights=np.maximum(evals,0)/np.maximum(evals,0).sum();vp=[]
    for i in range(N_RANDOM):
        jj=rng.choice(512,2,replace=False,p=weights);vp.append(global_v[jj])
    for _,r in table.iterrows():
        v=objects[r.contrast_id][2];den=np.dot(v,v);obs=r.E_PC12
        for kind,planes in [('isotropic',iso),('global_variance_weighted',vp)]:
            vals=np.array([np.sum((v@q.T)**2)/den for q in planes]);rows.append({'contrast_id':r.contrast_id,'null_type':kind,'observed_E_PC12':obs,'null_mean':vals.mean(),'null_median':np.median(vals),'null_95_high':np.quantile(vals,.95),'empirical_p_ge':(1+(vals>=obs).sum())/(len(vals)+1),'empirical_percentile':100*(vals<obs).mean()})
    return pd.DataFrame(rows)


def reference_table(table):
    # References are loaded from the existing audit where possible and matched to empirical ranks.
    refs=[('RR3 GC39→GC40',.087),('RR3 FLT39→FLT40',.620),('OSD-48 RR1 carcass FLT−GC',.938),('OSD-168 RR1 FLT−GC',.886),('OSD-137 RR3-40 FLT−GC',.733),('OSD-168 RR3-40 FLT−GC',.689)]
    null=table[table.primary_null].E_PC12.to_numpy();rows=[]
    for name,x in refs:rows.append({'reference':name,'E_PC12':x,'empirical_percentile':100*np.mean(null<x),'empirical_p_ge':(1+np.sum(null>=x))/(len(null)+1),'rank_descending':1+np.sum(null>x),'primary_null_n':len(null)})
    return pd.DataFrame(rows)


def figures(table, refs, perm, rand, global_summary, spectrum):
    plt.style.use('seaborn-v0_8-whitegrid');FIG.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(12,7));q=table[table.primary_null];ax.hist(q.E_PC12,bins=10,color='#4C78A8',alpha=.55)
    colors=plt.colormaps['tab10'](np.linspace(0,1,len(refs)))
    for (_,r),c in zip(refs.iterrows(),colors):ax.axvline(r.E_PC12,color=c,lw=2,label=f"{r.reference} ({r.E_PC12:.1%})")
    ax.set(xlabel='Response energy in fixed T-cell PC1–2',ylabel='Biological contrasts',title='Empirical biological-contrast occupancy null');ax.legend(fontsize=8,bbox_to_anchor=(1.02,1),loc='upper left');fig.tight_layout()
    for e in ['png','pdf']:fig.savefig(FIG/f'empirical_occupancy_distribution.{e}',dpi=400,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,7));q=table.sort_values('E_PC12');ax.barh(q.contrast_id,q.E_PC12,color=np.where(q.primary_null,'#4C78A8','#BAB0AC'));ax.set(xlabel='E_PC12',title='Fixed PC1–2 occupancy across curated contrasts');fig.tight_layout()
    for e in ['png','pdf']:fig.savefig(FIG/f'contrast_occupancy.{e}',dpi=400,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,6));axes[0].plot(spectrum.global_PC[:50],spectrum.cumulative_variance[:50],marker='o');axes[0].set(xlabel='Global BridgeRNA PC',ylabel='Cumulative variance',title='Global latent variance spectrum')
    axes[1].plot(global_summary.global_PC_prefix,global_summary.mean_squared_subspace_overlap,marker='o');axes[1].set(xlabel='Top global-PC prefix',ylabel='Mean squared overlap (0–1)',title='T-cell PC1–2 vs global-PC spaces');fig.tight_layout()
    for e in ['png','pdf']:fig.savefig(FIG/f'global_variance_alignment.{e}',dpi=400);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,6));
    for label,g in table.groupby('source_benchmark'):
        axes[0].scatter(g.top1pct_abs_effect_fraction,g.E_PC12,s=70,label=label);axes[1].scatter(g.expression_effective_gene_count,g.E_PC12,s=70,label=label)
    axes[0].set(xlabel='Top 1% fraction of absolute expression effect',ylabel='E_PC12');axes[1].set(xlabel='Expression-response effective gene count',ylabel='E_PC12');axes[0].legend(fontsize=9);fig.suptitle('Expression-response coordination versus PC1–2 occupancy');fig.tight_layout()
    for e in ['png','pdf']:fig.savefig(FIG/f'coordination_hypothesis.{e}',dpi=400);plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(exist_ok=True);b=fixed_basis();table,objects=load_contrasts(b);perm,boot,robust=permutation_and_bootstrap(table,objects,b);gv,ev,gs,spectrum=global_geometry(b);random=random_subspaces(table,objects,b,gv,ev);refs=reference_table(table)
    table=table.merge(robust,on=['contrast_id','observed_E_PC12'],how='left') if 'observed_E_PC12' in table else table.merge(robust.drop(columns='observed_E_PC12'),on='contrast_id')
    table.to_csv(OUT/'biological_contrast_occupancy.csv',index=False);perm.to_parquet(OUT/'within_study_permutation_null.parquet',index=False);boot.to_parquet(OUT/'contrast_bootstrap.parquet',index=False);robust.to_csv(OUT/'contrast_robustness_summary.csv',index=False);random.to_csv(OUT/'random_subspace_null_summary.csv',index=False);refs.to_csv(OUT/'reference_contrast_percentiles.csv',index=False);gs.to_csv(OUT/'global_subspace_alignment.csv',index=False);spectrum.to_csv(OUT/'global_variance_spectrum.csv',index=False)
    primary=table[table.primary_null];corr=[]
    for metric in ['top1pct_abs_effect_fraction','expression_effective_gene_count','expression_response_rms']:
        s=spearmanr(primary.E_PC12,primary[metric]);corr.append({'coordination_measure':metric,'spearman_rho':s.statistic,'p_value':s.pvalue,'n':len(primary)})
    pd.DataFrame(corr).to_csv(OUT/'coordination_correlations.csv',index=False)
    bins=pd.qcut(primary.E_PC12.rank(method='first'),[0,.1,.5,.9,.95,1],labels=['bottom_10pct','lower_middle','upper_middle','top_10_to_5pct','top_5pct']);char=primary.assign(occupancy_group=bins.astype(str));char.to_csv(OUT/'occupancy_metadata_characterization.csv',index=False)
    figures(table,refs,perm,random,gs,spectrum)
    result={'primary_contrasts':int(table.primary_null.sum()),'secondary_underpowered':int((~table.primary_null).sum()),'median_E_PC12':float(primary.E_PC12.median()),'global_variance_captured':float(gs.global_variance_fraction_captured_by_Tcell_PC12.iloc[0]),'coordination_correlations':corr,'claim':'Fixed T-cell PC1-2 occupancy is a geometric diagnostic; neither technical specificity nor shared biological mechanism is assumed.'}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n');(OUT/'provenance.json').write_text(json.dumps({'basis':'unchanged uncentered SVD PC1-2 from 40 controlled same-RNA T-cell Ribo-minus-PolyA differences','biological_contrasts':['8 curated Task2 exercise','14 curated Task3 FLT-GC'],'primary_rule':'at least 2 samples in each arm','permutations':N_PERM,'bootstraps':N_BOOT,'random_subspaces':N_RANDOM,'seed':SEED,'embeddings_recomputed':False},indent=2)+'\n')
    print(table[['contrast_id','n_A','n_B','primary_null','E_PC12','permutation_p_ge','bootstrap_low','bootstrap_high']].sort_values('E_PC12',ascending=False).to_string(index=False));print('\nReferences\n',refs.to_string(index=False));print('\nGlobal\n',gs.to_string(index=False));print('\nCoordination\n',pd.DataFrame(corr).to_string(index=False));print('[complete]',OUT)


if __name__ == '__main__': main()
