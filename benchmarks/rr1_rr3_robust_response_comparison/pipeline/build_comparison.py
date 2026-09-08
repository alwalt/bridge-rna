#!/usr/bin/env python3
"""Compare RR1/RR3 robust cores using identical frozen definitions."""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

ROOT=Path(__file__).resolve().parents[3];BENCH=ROOT/'benchmarks/rr1_rr3_robust_response_comparison';OUT=BENCH/'results';FIG=OUT/'figures'
RR1=ROOT/'benchmarks/rr1_library_prep_interaction/results';T4=ROOT/'benchmarks/library_prep_disentanglement/results';FR=ROOT/'benchmarks/frozen_sample_embedding_readout/results'
sys.path.insert(0,str(ROOT/'benchmarks/rr1_rr3_contextual_graph_case_study/pipeline'))
from build_case_study import response_graphs  # noqa:E402
COHORTS=['RR1','RR3-39','RR3-40'];KEY={'RR1':'rr1','RR3-39':'rr3_39','RR3-40':'rr3_40'}
RESP={'RR1':('RR1 OSD48','RR1 OSD168'),'RR3-39':('RR3-39 OSD137','RR3-39 OSD168'),'RR3-40':('RR3-40 OSD137','RR3-40 OSD168')}
GRAPH={'RR1':('RR1_original','RR1_remeasurement'),'RR3-39':('RR3_39_original','RR3_39_remeasurement'),'RR3-40':('RR3_40_original','RR3_40_remeasurement')}

def sets():
    result={f'Hallmark: {k}':set(map(str.upper,v)) for k,v in json.loads((ROOT/'data/gsea/hallmark_gene_sets.json').read_text()).items()}
    base=ROOT/'benchmarks/cross_species_exercise_response/results/per_study_ranked_gsea'
    for source,fn in [('GO BP','GO_Biological_Process_2026.gmt'),('Reactome','Reactome_Pathways_2024.gmt')]:
        for line in (base/fn).read_text().splitlines():
            f=line.split('\t');result[f'{source}: {f[0]}']=set(map(str.upper,f[2:]))
    return result
def ora(label,selected,universe,resources):
    selected=set(selected);rows=[]
    for term,members in resources.items():
        members=members&universe;hit=members&selected
        if len(members)>=5 and hit:rows.append({'cohort':label,'term':term,'overlap':len(hit),'set_size':len(members),'p_value':hypergeom.sf(len(hit)-1,len(universe),len(members),len(selected))})
    d=pd.DataFrame(rows);d['fdr']=multipletests(d.p_value,method='fdr_bh')[1];return d.sort_values(['fdr','p_value'])
def graph_matrix(v,G=15165):
    codes,values=v.indices,v.data.astype(float);i,j=codes//G,codes%G
    return sparse.csr_matrix((np.r_[values,values],(np.r_[i,j],np.r_[j,i])),shape=(G,G))
def graph_metrics(a,b):
    num=np.asarray(a.multiply(b).sum(1)).ravel();den=np.sqrt(np.asarray(a.multiply(a).sum(1)).ravel()*np.asarray(b.multiply(b).sum(1)).ravel());local=np.full(a.shape[0],np.nan);ok=den>0;local[ok]=num[ok]/den[ok]
    A=a.copy();B=b.copy();A.data=np.ones_like(A.data);B.data=np.ones_like(B.data);inter=np.asarray(A.multiply(B).sum(1)).ravel();union=np.asarray(((A+B)>0).sum(1)).ravel();jac=np.divide(inter,union,out=np.full_like(inter,np.nan,dtype=float),where=union>0)
    return local,jac

def main():
    for sub in ['manifest','genes','pathways','figures','summary']: (OUT/sub).mkdir(parents=True,exist_ok=True)
    mapping=pd.read_csv(T4/'task4_rr1_rr3_paired_technical_replication/animal_mapping.csv');mapping.to_csv(OUT/'manifest/exact_matched_animals.csv',index=False)
    audit=pd.read_parquet(T4/'task4_rr1_rr3_sample_pc12_audit/master_sample_level_audit.parquet')
    audit=audit[audit.sample_id.isin(mapping.sample_id)].copy()
    audit['RIN_numeric']=pd.to_numeric(audit.RIN_numeric,errors='coerce')
    metadata_summary=(audit.groupby(['cohort','OSD','condition'],dropna=False)
        .agg(n=('sample_id','nunique'),animal_ids=('animal_id',lambda x:'; '.join(sorted(set(map(str,x.dropna()))))),
             library_preparation=('library_preparation',lambda x:'; '.join(sorted(set(map(str,x.dropna()))))),
             sequencing_configuration=('sequencing_parameters',lambda x:'; '.join(sorted(set(map(str,x.dropna()))))),
             sequencing_facility=('sequencing_facility',lambda x:'; '.join(sorted(set(map(str,x.dropna()))))),
             preservation=('preservation',lambda x:'; '.join(sorted(set(map(str,x.dropna()))))),
             strain=('strain',lambda x:'; '.join(sorted(set(map(str,x.dropna()))))),
             sex=('sex',lambda x:'; '.join(sorted(set(map(str,x.dropna()))))),
             RIN_median=('RIN_numeric','median'),RIN_min=('RIN_numeric','min'),RIN_max=('RIN_numeric','max'))
        .reset_index())
    metadata_summary.to_csv(OUT/'manifest/cohort_metadata_summary.csv',index=False)
    robust={c:pd.read_parquet(RR1/('robust_core/integrated/rr1_gene_evidence.parquet' if c=='RR1' else f'robust_core/rr3_controls/{KEY[c]}_gene_evidence.parquet')) for c in COHORTS}
    interactions=pd.read_parquet(T4/'task4_attribution_vs_expression/contrast_gene_tables.parquet');resources=sets();universe=set(robust['RR1'].gene_symbol.str.upper())
    graph=response_graphs();gsummary=[]
    for c,(x,y) in GRAPH.items():
        # response_graphs() already returns symmetric G x G signed response matrices.
        local,jac=graph_metrics(graph[x],graph[y]);gsummary.append({'cohort':c,'median_neighborhood_response_cosine':np.nanmedian(local),'mean_neighborhood_jaccard':np.nanmean(jac),'positive_neighborhood_fraction':np.nanmean(local>0)})
    pd.DataFrame(gsummary).to_csv(OUT/'summary/graph_robustness.csv',index=False)
    sizes=pd.read_csv(RR1/'robust_core/rr3_controls/robust_core_size.csv').set_index('cohort')
    multi=pd.read_csv(ROOT/'benchmarks/multiscale_response_similarity/results/summary/primary_table.csv').set_index('relationship_type')
    igm=pd.read_csv(T4/'task4_gene_attribution_diagnostic/technical_replication_ig_comparison.csv').set_index('comparison')
    rows=[]
    for c in COHORTS:
        r=multi.loc['technically_sensitive_remeasurement' if c=='RR1' else 'true_technical_remeasurement']
        if isinstance(r,pd.DataFrame):r=r[r.comparison.str.contains(c)].iloc[0]
        q=robust[c];gs=next(x for x in gsummary if x['cohort']==c)
        rows.append({'metric':'Global response cosine','cohort':c,'value':r.global_bridgerna})
        rows.append({'metric':'Expression cosine','cohort':c,'value':r.expression})
        rows.append({'metric':'Expression Spearman','cohort':c,'value':pd.read_csv(RR1/'robust_core/rr3_controls/multilevel_reproducibility.csv').query("cohort==@c and level=='expression'").spearman.iloc[0]})
        rows.append({'metric':'Hallmark cosine','cohort':c,'value':r.hallmark_program})
        rows.append({'metric':'IG cosine','cohort':c,'value':pd.read_csv(T4/'task4_attribution_vs_expression/reproducibility_pair_scores.csv').query("y==1").iloc[COHORTS.index(c)].attribution_similarity})
        rows.append({'metric':'IG Spearman','cohort':c,'value':igm.loc[c,'full_signed_spearman']})
        rows.append({'metric':'Graph response cosine','cohort':c,'value':r.contextual_graph})
        rows.append({'metric':'Median local graph cosine','cohort':c,'value':gs['median_neighborhood_response_cosine']})
        for label,col in [('Robust expression genes','robust_expression'),('Robust IG genes','robust_attribution'),('Robust graph genes','robust_graph'),('≥2-level genes','genes_2plus_evidence'),('3-level genes','genes_3of3_evidence')]:rows.append({'metric':label,'cohort':c,'value':sizes.loc[c,col]})
    metrics=pd.DataFrame(rows);metrics.pivot(index='metric',columns='cohort',values='value').reindex(columns=COHORTS).to_csv(OUT/'summary/primary_comparison.csv')
    # Interaction rankings and identical enrichment.
    enrich=[]
    for c in COHORTS:
        oa,ob=({'RR1':('RR1 OSD48','RR1 OSD168'),'RR3-39':('RR3-39 OSD137','RR3-39 OSD168'),'RR3-40':('RR3-40 OSD137','RR3-40 OSD168')})[c]
        q=interactions[interactions.contrast.isin([oa,ob])].pivot(index='gene_symbol',columns='contrast',values='expression_change').dropna();q['interaction']=q[ob]-q[oa];q['absolute_interaction_rank']=q.interaction.abs().rank(ascending=False,method='min').astype(int);q=q.reset_index();q.to_parquet(OUT/'genes'/f'{KEY[c]}_expression_interaction.parquet',index=False)
        enrich.append(ora(c,q.nsmallest(500,'absolute_interaction_rank').gene_symbol.str.upper(),universe,resources))
    sensitivity=pd.concat(enrich,ignore_index=True);sensitivity.to_csv(OUT/'pathways/measurement_sensitive_enrichment.csv',index=False)
    # Uniform pathway-profile classification.
    pp=pd.read_parquet(T4/'task4_rr1_rr3_paired_technical_replication/pathway_profiles.parquet');wide=pp.pivot(index=['source','pathway'],columns='response',values='score');pathrows=[]
    for c,(a,b) in RESP.items():
        ra=wide[a].abs().rank(ascending=False,method='min');rb=wide[b].abs().rank(ascending=False,method='min');same=np.sign(wide[a])==np.sign(wide[b])
        for idx in wide.index:
            cls='protocol_robust' if same.loc[idx] and ra.loc[idx]<=250 and rb.loc[idx]<=250 else 'direction_reversing' if (not same.loc[idx]) and (ra.loc[idx]<=250 or rb.loc[idx]<=250) else 'original_specific' if ra.loc[idx]<=250 and rb.loc[idx]>250 else 'remeasurement_specific' if rb.loc[idx]<=250 and ra.loc[idx]>250 else 'ambiguous'
            pathrows.append({'cohort':c,'source':idx[0],'pathway':idx[1],'original_score':wide.loc[idx,a],'remeasurement_score':wide.loc[idx,b],'rank_original':ra.loc[idx],'rank_remeasurement':rb.loc[idx],'classification':cls})
    pathways=pd.DataFrame(pathrows);pathways.to_csv(OUT/'pathways/pathway_classification.csv',index=False)
    pcounts=pathways.groupby(['cohort','classification']).size().unstack(fill_value=0).reset_index();pcounts.to_csv(OUT/'summary/pathway_counts.csv',index=False)
    # Shared/unique robust genes and pathways.
    cores={c:set(q.loc[q.evidence_count>=2,'gene_symbol']) for c,q in robust.items()};all3=set.intersection(*cores.values());rr3=cores['RR3-39']&cores['RR3-40']
    union=set.union(*cores.values());bio=[]
    for gene in sorted(union):
        flags={c:gene in cores[c] for c in COHORTS};label='shared_all_three' if all(flags.values()) else 'shared_RR3' if flags['RR3-39'] and flags['RR3-40'] and not flags['RR1'] else next((f'{c}_specific' for c in COHORTS if flags[c] and sum(flags.values())==1),'shared_subset')
        row={'feature_type':'gene','gene_or_pathway':gene,'interpretation':label}
        for c in COHORTS:
            q=robust[c].set_index('gene_symbol');row[f'{c}_robust']=flags[c];row[f'{c}_expression_support']=bool(q.loc[gene,'robust_expression']) if gene in q.index else False;row[f'{c}_attribution_support']=bool(q.loc[gene,'robust_attribution']) if gene in q.index else False;row[f'{c}_graph_support']=bool(q.loc[gene,'robust_graph']) if gene in q.index else False
        bio.append(row)
    pc={c:set(pathways.query("cohort==@c and classification=='protocol_robust'").pathway) for c in COHORTS}
    for pathway in sorted(set.union(*pc.values())):
        flags={c:pathway in pc[c] for c in COHORTS};label='shared_all_three' if all(flags.values()) else 'shared_RR3' if flags['RR3-39'] and flags['RR3-40'] and not flags['RR1'] else next((f'{c}_specific' for c in COHORTS if flags[c] and sum(flags.values())==1),'shared_subset')
        row={'feature_type':'pathway','gene_or_pathway':pathway,'interpretation':label}
        for c in COHORTS:row[f'{c}_robust']=flags[c]
        bio.append(row)
    biological=pd.DataFrame(bio);biological.to_csv(OUT/'summary/biological_summary.csv',index=False)
    contribution=[]
    for c,q in robust.items():
        e=q.robust_expression.astype(bool);a=q.robust_attribution.astype(bool);g=q.robust_graph.astype(bool)
        contribution += [
            {'cohort':c,'evidence':'Expression only','genes':int((e&~a&~g).sum())},
            {'cohort':c,'evidence':'Expression + attribution','genes':int((e&a).sum())},
            {'cohort':c,'evidence':'Expression + attribution + graph','genes':int((e&a&g).sum())},
            {'cohort':c,'evidence':'Attribution only','genes':int((a&~e&~g).sum())},
            {'cohort':c,'evidence':'Graph only','genes':int((g&~e&~a).sum())},
        ]
    pd.DataFrame(contribution).to_csv(OUT/'summary/model_contribution.csv',index=False)
    shared={'all_three_genes':len(all3),'shared_rr3_genes':len(rr3),'rr1_specific_genes':len(cores['RR1']-(cores['RR3-39']|cores['RR3-40'])),'rr3_39_specific_genes':len(cores['RR3-39']-(cores['RR1']|cores['RR3-40'])),'rr3_40_specific_genes':len(cores['RR3-40']-(cores['RR1']|cores['RR3-39'])),'all_three_pathways':len(set.intersection(*pc.values())),'shared_rr3_pathways':len(pc['RR3-39']&pc['RR3-40'])};(OUT/'summary/shared_unique_counts.json').write_text(json.dumps(shared,indent=2)+'\n')
    # Figures.
    plt.style.use('seaborn-v0_8-whitegrid');order=['Robust expression genes','Robust IG genes','Robust graph genes','≥2-level genes','3-level genes'];tab=metrics[metrics.metric.isin(order)].pivot(index='metric',columns='cohort',values='value').loc[order,COHORTS]
    fig,ax=plt.subplots(figsize=(10,5.5));tab.T.plot.bar(ax=ax);ax.set(ylabel='Genes',xlabel='',title='Protocol-robust core size under identical definitions');ax.tick_params(axis='x',rotation=0);ax.legend(fontsize=8);fig.tight_layout()
    for e in ['png','pdf']:fig.savefig(FIG/f'robust_core_size.{e}',dpi=300,bbox_inches='tight');plt.close(fig)
    mult=metrics[metrics.metric.isin(['Expression cosine','Global response cosine','Hallmark cosine','IG cosine','Graph response cosine'])].pivot(index='metric',columns='cohort',values='value').reindex(columns=COHORTS)
    fig,ax=plt.subplots(figsize=(9,5.5));mult.T.plot.bar(ax=ax);ax.axhline(0,color='black',lw=.8);ax.set(ylim=(-1,1),ylabel='Within-representation similarity',xlabel='',title='Multiscale technical remeasurement');ax.tick_params(axis='x',rotation=0);ax.legend(fontsize=8);fig.tight_layout()
    for e in ['png','pdf']:fig.savefig(FIG/f'multiscale_reproducibility.{e}',dpi=300,bbox_inches='tight');plt.close(fig)
    # Matched expression/IG scatters.
    fig,axes=plt.subplots(2,3,figsize=(14,9))
    for j,c in enumerate(COHORTS):
        q=robust[c];inter=pd.read_parquet(OUT/'genes'/f'{KEY[c]}_expression_interaction.parquet').set_index('gene_symbol');high=set(inter.nsmallest(500,'absolute_interaction_rank').index);allthree=q.evidence_count.eq(3)
        axes[0,j].scatter(q.expression_original,q.expression_remeasurement,s=3,alpha=.12,color='#777');axes[0,j].scatter(q.loc[q.gene_symbol.isin(high),'expression_original'],q.loc[q.gene_symbol.isin(high),'expression_remeasurement'],s=8,color='#E45756',alpha=.4);axes[0,j].scatter(q.loc[allthree,'expression_original'],q.loc[allthree,'expression_remeasurement'],s=18,color='#2A9D8F');axes[0,j].set(title=f'{c} expression',xlabel='Original',ylabel='Remeasurement')
        axes[1,j].scatter(q.attribution_original,q.attribution_remeasurement,s=3,alpha=.12,color='#777');axes[1,j].scatter(q.loc[allthree,'attribution_original'],q.loc[allthree,'attribution_remeasurement'],s=18,color='#2A9D8F');axes[1,j].set(title=f'{c} signed IG',xlabel='Original',ylabel='Remeasurement')
    fig.tight_layout();
    for e in ['png','pdf']:fig.savefig(FIG/f'original_vs_remeasurement.{e}',dpi=300,bbox_inches='tight');plt.close(fig)
    # Common robust/sensitive pathway display.
    selected=[]
    for c in COHORTS:
        selected+=pathways.query("cohort==@c and classification!='ambiguous'").assign(priority=lambda x:np.minimum(x.rank_original,x.rank_remeasurement)).nsmallest(8,'priority').pathway.tolist()
    selected=list(dict.fromkeys(selected))[:30];cols=[]
    for c,(a,b) in RESP.items():cols += [(c,a),(c,b)]
    M=np.full((len(selected),6),np.nan)
    for i,p in enumerate(selected):
        for j,(c,rname) in enumerate(cols):
            q=pathways[(pathways.cohort.eq(c))&pathways.pathway.eq(p)];M[i,j]=q.original_score.iloc[0] if rname==RESP[c][0] else q.remeasurement_score.iloc[0]
    lim=np.nanpercentile(np.abs(M),95);fig,ax=plt.subplots(figsize=(11,10));im=ax.imshow(M,aspect='auto',cmap='RdBu_r',vmin=-lim,vmax=lim);ax.set_xticks(range(6),['RR1 original','RR1 remeasure','RR3-39 original','RR3-39 remeasure','RR3-40 original','RR3-40 remeasure'],rotation=35,ha='right');ax.set_yticks(range(len(selected)),selected,fontsize=7);fig.colorbar(im,ax=ax,label='Existing signed pathway score');ax.set_title('Robust and measurement-sensitive pathway profiles');fig.tight_layout()
    for e in ['png','pdf']:fig.savefig(FIG/f'pathway_profiles.{e}',dpi=300,bbox_inches='tight');plt.close(fig)
    prov={'bridge_inference_rerun':False,'ig_rerun':False,'gene_criteria':'mutual Top-500 same-direction expression/IG; Top-500 positive local graph cosine','pathway_criteria':'mutual Top-250 absolute score and same direction','enrichment_background':15165,'claims':'technical remeasurement concordance, not independent biological replication'};(OUT/'summary/provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
    print(metrics.pivot(index='metric',columns='cohort',values='value').reindex(columns=COHORTS).to_string());print('\n',pcounts.to_string(index=False));print('\n',json.dumps(shared,indent=2))
if __name__=='__main__':main()
