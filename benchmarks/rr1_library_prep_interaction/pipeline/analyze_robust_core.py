#!/usr/bin/env python3
"""Identify gene- and program-level RR1 evidence robust across measurements."""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import hypergeom, pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests

ROOT=Path(__file__).resolve().parents[3];BENCH=ROOT/'benchmarks/rr1_library_prep_interaction';OUT=BENCH/'results/robust_core';FIG=OUT/'figures'
T3=ROOT/'benchmarks/osdr_batch_effect_representation';T4=ROOT/'benchmarks/library_prep_disentanglement';FR=ROOT/'benchmarks/frozen_sample_embedding_readout'
sys.path.insert(0,str(ROOT/'benchmarks/rr1_rr3_contextual_graph_case_study/pipeline'))
from build_case_study import response_graphs, row_cosine  # noqa:E402

PAIRS={'RR1':('RR1_OSD48_original_matched','RR1_OSD168_no-ERCC'),
       'RR3-39':('C01_OSD137_original_matched','C01_OSD168_all_ERCC'),
       'RR3-40':('C02_OSD137_original_matched','C02_OSD168_all_ERCC')}
GRAPH={'RR1':('RR1_original','RR1_remeasurement'),'RR3-39':('RR3_39_original','RR3_39_remeasurement'),'RR3-40':('RR3_40_original','RR3_40_remeasurement')}

def ranks(v):
    order=np.argsort(-np.abs(v));r=np.empty(len(v),int);r[order]=np.arange(1,len(v)+1);return r
def cosine(a,b):
    d=np.linalg.norm(a)*np.linalg.norm(b);return float(a@b/d) if d else np.nan

def vectors():
    expr=np.load(T4/'results/task4_gene_attribution_diagnostic/expression_response_vectors.npz',allow_pickle=True)
    expr={k:expr[k] for k in expr.files if k!='gene_symbol'}
    d=pd.read_parquet(T4/'results/task4_gene_attribution_diagnostic/per_response_gene_rankings.parquet')
    ig={name:q.set_index('gene_symbol_human').reindex(GENES).signed_attribution.to_numpy() for name,q in d.groupby('response')}
    return expr,ig

def metrics(label,a,b):
    ra,rb=ranks(a),ranks(b);rows=[]
    for n in [100,250,500]:
        shared=np.flatnonzero((ra<=n)&(rb<=n));rows.append({'cohort':label,'top_n':n,'overlap':len(shared),'direction_agreement':float(np.mean(np.sign(a[shared])==np.sign(b[shared]))) if len(shared) else np.nan})
    return {'cohort':label,'pearson':pearsonr(a,b).statistic,'spearman':spearmanr(a,b).statistic,'cosine':cosine(a,b),'genome_direction_agreement':np.mean(np.sign(a)==np.sign(b))},rows

def ora(name,selected,sets):
    universe=set(GENES);selected=set(selected);rows=[]
    for term,members in sets.items():
        members &= universe;hit=members&selected
        if len(members)>=5 and hit:rows.append({'gene_set':name,'term':term,'overlap':len(hit),'set_size':len(members),'p_value':hypergeom.sf(len(hit)-1,len(universe),len(members),len(selected))})
    d=pd.DataFrame(rows)
    if not d.empty:d['fdr']=multipletests(d.p_value,method='fdr_bh')[1];d=d.sort_values(['fdr','p_value'])
    return d

def gene_sets():
    sets={f'Hallmark: {k}':set(map(str.upper,v)) for k,v in json.loads((ROOT/'data/gsea/hallmark_gene_sets.json').read_text()).items()}
    base=ROOT/'benchmarks/cross_species_exercise_response/results/per_study_ranked_gsea'
    for source,fn in [('GO BP','GO_Biological_Process_2026.gmt'),('Reactome','Reactome_Pathways_2024.gmt')]:
        for line in (base/fn).read_text().splitlines():
            f=line.split('\t');sets[f'{source}: {f[0]}']=set(map(str.upper,f[2:]))
    return sets

def handling_flags(core):
    m=pd.read_csv(T3/'results/task3b_contrast_sample_membership.csv');sm=pd.read_csv(T3/'results/sample_manifest.csv');ix=dict(zip(sm.sample_id,range(len(sm))));x=np.load(T3/'work/bridgerna_log1p_tpm_inputs.npy',mmap_mode='r')
    def response(cid):
        q=m[m.contrast_id.eq(cid)];ids={c:q[q.condition.eq(c)].sample_id.map(ix).to_numpy(int) for c in ['FLT','GC']};return np.asarray(x[ids['FLT']],float).mean(0)-np.asarray(x[ids['GC']],float).mean(0)
    carc=response('C14__OSD-48__RR1-NASA__37-day');euth=response('C13__OSD-48__RR1-NASA__37-day');difference=np.abs(carc-euth);high=ranks(difference)<=500
    core['carcass_response']=carc;core['euthanasia_response']=euth;core['handling_difference']=difference;core['handling_sensitive']=high|(np.sign(carc)!=np.sign(euth));return core

def main():
    for sub in ['expression','attribution','programs','graph','integrated','rr3_controls','figures']: (OUT/sub).mkdir(parents=True,exist_ok=True)
    global GENES;GENES=pd.read_csv(ROOT/'data/ensembl/canonical_genes.csv').sort_values('token_id').gene_symbol.astype(str).str.upper().to_numpy();expr,ig=vectors();graphs=response_graphs()
    summary=[];overlaps=[];gene_tables={}
    for cohort,(oa,ob) in PAIRS.items():
        em,eo=metrics(cohort,expr[oa],expr[ob]);em['level']='expression';summary.append(em);overlaps.extend(dict(x,level='expression') for x in eo)
        am,ao=metrics(cohort,ig[oa],ig[ob]);am['level']='attribution';summary.append(am);overlaps.extend(dict(x,level='attribution') for x in ao)
        ga,gb=GRAPH[cohort];local=row_cosine(graphs[ga],graphs[gb]);summary.append({'cohort':cohort,'level':'graph_local','pearson':np.nan,'spearman':np.nan,'cosine':np.nan,'genome_direction_agreement':np.nan,'median_local_cosine':np.nanmedian(local),'positive_local_fraction':np.nanmean(local>0)})
        er1,er2=ranks(expr[oa]),ranks(expr[ob]);ar1,ar2=ranks(ig[oa]),ranks(ig[ob]);gr=ranks(np.nan_to_num(local,nan=-2))
        q=pd.DataFrame({'gene_symbol':GENES,'expression_original':expr[oa],'expression_remeasurement':expr[ob],'expression_rank_original':er1,'expression_rank_remeasurement':er2,'expression_sign_agreement':np.sign(expr[oa])==np.sign(expr[ob]),'attribution_original':ig[oa],'attribution_remeasurement':ig[ob],'attribution_rank_original':ar1,'attribution_rank_remeasurement':ar2,'attribution_sign_agreement':np.sign(ig[oa])==np.sign(ig[ob]),'graph_local_cosine':local,'graph_stability_rank':gr})
        q['robust_expression']=(er1<=500)&(er2<=500)&q.expression_sign_agreement
        q['robust_attribution']=(ar1<=500)&(ar2<=500)&q.attribution_sign_agreement
        q['robust_graph']=(gr<=500)&(q.graph_local_cosine>0)
        q['evidence_count']=q[['robust_expression','robust_attribution','robust_graph']].sum(axis=1)
        gene_tables[cohort]=q
        q.to_parquet(OUT/('integrated' if cohort=='RR1' else 'rr3_controls')/f'{cohort.lower().replace("-","_")}_gene_evidence.parquet',index=False)
    pd.DataFrame(summary).to_csv(OUT/'rr3_controls/multilevel_reproducibility.csv',index=False);pd.DataFrame(overlaps).to_csv(OUT/'rr3_controls/topn_overlap.csv',index=False)
    rr=gene_tables['RR1'];interaction=pd.read_parquet(BENCH/'results/interaction/per_gene_interaction.parquet');interaction['gene_symbol']=interaction.gene_symbol.str.upper();rr=rr.merge(interaction[['gene_symbol','expression_interaction','absolute_interaction_rank']],on='gene_symbol',how='left')
    tech=pd.read_parquet(T4/'results/task4_attribution_vs_expression/contrast_gene_tables.parquet');tech=tech[tech.contrast.eq('controlled_tcell')].copy();tech.index=tech.gene_symbol.str.upper()
    rr['tcell_expression_shift']=rr.gene_symbol.map(tech.expression_change);rr['tcell_attribution']=rr.gene_symbol.map(tech.attribution);rr['tcell_top500']=rr.gene_symbol.isin(set(tech.nsmallest(500,'expression_rank').gene_symbol.str.upper()))
    rr=handling_flags(rr);rr['protocol_flight_sensitive']=rr.absolute_interaction_rank<=500
    rr['category']=np.select([rr.robust_expression&rr.robust_attribution,rr.robust_attribution&~rr.robust_expression,rr.robust_expression&~rr.robust_attribution,rr.protocol_flight_sensitive],['robust_expression_and_attribution','robust_attribution_weak_expression','robust_expression_weak_attribution','technically_unstable'],default='ambiguous')
    rr['interpretation']=np.where(rr.evidence_count>=2,'protocol-robust multilevel candidate',np.where(rr.protocol_flight_sensitive,'protocol-sensitive candidate','ambiguous'))
    rr.to_parquet(OUT/'integrated/protocol_robust_evidence.parquet',index=False);rr[rr.evidence_count>=2].sort_values(['evidence_count','absolute_interaction_rank'],ascending=[False,True]).to_csv(OUT/'integrated/protocol_robust_candidates.csv',index=False)
    rr[rr.protocol_flight_sensitive].to_csv(OUT/'integrated/protocol_sensitive_candidates.csv',index=False);rr[rr.interpretation.eq('ambiguous')].to_csv(OUT/'integrated/ambiguous_candidates.csv',index=False)
    # Program profiles already use matched contextual response projections and fixed pathway sets.
    pp=pd.read_parquet(T4/'results/task4_rr1_rr3_paired_technical_replication/pathway_profiles.parquet');w=pp.pivot(index=['source','pathway'],columns='response',values='score')
    a,b=w['RR1 OSD48'],w['RR1 OSD168'];program=pd.DataFrame({'source':w.index.get_level_values(0),'pathway':w.index.get_level_values(1),'polyA_score':a,'ribo_score':b});program['direction_agreement']=np.sign(a)==np.sign(b);program['rank_polyA']=ranks(a.to_numpy());program['rank_ribo']=ranks(b.to_numpy());program['classification']=np.select([program.direction_agreement&(program.rank_polyA<=250)&(program.rank_ribo<=250),(~program.direction_agreement)&((program.rank_polyA<=250)|(program.rank_ribo<=250)),(program.rank_polyA<=250)&(program.rank_ribo>250),(program.rank_ribo<=250)&(program.rank_polyA>250)],['protocol_robust','direction_reversing','polyA_specific','ribo_specific'],default='ambiguous')
    program.to_csv(OUT/'programs/rr1_program_classification.csv',index=False)
    sets=gene_sets();enrich=pd.concat([ora('robust_multilevel',rr[rr.evidence_count>=2].gene_symbol,sets),ora('protocol_interaction_top500',rr[rr.protocol_flight_sensitive].gene_symbol,sets)],ignore_index=True);enrich.to_csv(OUT/'programs/robust_vs_interaction_enrichment.csv',index=False)
    counts=pd.concat([pd.DataFrame([{'cohort':c,'genes_2plus_evidence':int((q.evidence_count>=2).sum()),'genes_3of3_evidence':int((q.evidence_count==3).sum()),'robust_expression':int(q.robust_expression.sum()),'robust_attribution':int(q.robust_attribution.sum()),'robust_graph':int(q.robust_graph.sum())}]) for c,q in gene_tables.items()],ignore_index=True);counts.to_csv(OUT/'rr3_controls/robust_core_size.csv',index=False)
    # Scientist-facing table.
    final=rr.sort_values(['evidence_count','absolute_interaction_rank'],ascending=[False,True]).head(250).copy();final['program_annotation']='';final[['gene_symbol','expression_original','expression_remeasurement','expression_sign_agreement','attribution_original','attribution_remeasurement','attribution_sign_agreement','graph_local_cosine','protocol_flight_sensitive','handling_sensitive','evidence_count','interpretation']].to_csv(OUT/'integrated/scientist_facing_table.csv',index=False)
    # Figures.
    plt.style.use('seaborn-v0_8-whitegrid');FIG.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(8,7));ax.scatter(rr.expression_original,rr.expression_remeasurement,s=5,alpha=.12,color='#777');rob=rr.robust_expression&rr.robust_attribution;unst=rr.protocol_flight_sensitive
    ax.scatter(rr.loc[unst,'expression_original'],rr.loc[unst,'expression_remeasurement'],s=13,color='#E45756',alpha=.55,label='Top-500 interaction');ax.scatter(rr.loc[rob,'expression_original'],rr.loc[rob,'expression_remeasurement'],s=22,color='#2A9D8F',label='Robust expression + IG');lo=min(ax.get_xlim()[0],ax.get_ylim()[0]);hi=max(ax.get_xlim()[1],ax.get_ylim()[1]);ax.plot([lo,hi],[lo,hi],'k--',lw=.8);ax.set(xlabel='OSD-48 PolyA FLT−GC',ylabel='OSD-168 Ribo FLT−GC',title='RR1 expression: robust core versus protocol interaction');ax.legend();fig.tight_layout()
    for e in ['png','pdf']:fig.savefig(FIG/f'expression_robust_vs_unstable.{e}',dpi=300,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,7));ax.scatter(rr.attribution_original,rr.attribution_remeasurement,s=5,alpha=.12,color='#777');ax.scatter(rr.loc[rr.robust_attribution,'attribution_original'],rr.loc[rr.robust_attribution,'attribution_remeasurement'],s=20,color='#2A9D8F',label='Mutual Top-500, same direction');ax.axhline(0,color='black',lw=.5);ax.axvline(0,color='black',lw=.5);ax.set(xlabel='PolyA signed IG',ylabel='Ribo signed IG',title='RR1 attribution concordance');ax.legend();fig.tight_layout()
    for e in ['png','pdf']:fig.savefig(FIG/f'attribution_concordance.{e}',dpi=300,bbox_inches='tight');plt.close(fig)
    heat=final.head(35).set_index('gene_symbol')[['expression_original','expression_remeasurement','attribution_original','attribution_remeasurement','graph_local_cosine','expression_interaction']];heat=(heat-heat.mean())/heat.std(ddof=0).replace(0,1)
    fig,ax=plt.subplots(figsize=(9,11));im=ax.imshow(heat,aspect='auto',cmap='RdBu_r',vmin=-2.5,vmax=2.5);ax.set_xticks(range(6),['PolyA expr','Ribo expr','PolyA IG','Ribo IG','Graph concord.','Protocol×flight'],rotation=35,ha='right');ax.set_yticks(range(len(heat)),heat.index,fontsize=7);fig.colorbar(im,ax=ax,label='Column z-score');ax.set_title('Highest multilevel RR1 evidence');fig.tight_layout()
    for e in ['png','pdf']:fig.savefig(FIG/f'biological_core_heatmap.{e}',dpi=300,bbox_inches='tight');plt.close(fig)
    result={'definition':'mutual Top-500 same-direction expression/IG; Top-500 positive local graph cosine','rr1_counts':counts[counts.cohort.eq('RR1')].iloc[0].to_dict(),'program_robust_count':int(program.classification.eq('protocol_robust').sum()),'program_reversing_count':int(program.classification.eq('direction_reversing').sum()),'bridge_inference_rerun':False}
    (OUT/'integrated/summary.json').write_text(json.dumps(result,indent=2)+"\n");print(counts.to_string(index=False));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
