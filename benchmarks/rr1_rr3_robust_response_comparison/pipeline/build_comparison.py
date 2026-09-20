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
    for source,fn in [('GO BP','GO_Biological_Process_2026.gmt'),('Reactome','Reactome_Pathways_2024.gmt'),('KEGG','KEGG_2026.gmt')]:
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

def conserved_module(pathway):
    """Collapse only clear redundancies in the conserved pathway core."""
    p=pathway.lower()
    if any(x in p for x in ['aerobic respiration','cellular respiration','respiratory electron','oxidative phosphorylation','respiratory chain complex']): return 'Mitochondrial respiration / OXPHOS'
    if 'mitochondrial translation' in p or 'mitochondrial gene expression' in p: return 'Mitochondrial translation / gene expression'
    if any(x in p for x in ['translation','ribosome','ribosomal','40s','60s','protein biosynthetic','cap-binding complex']): return 'Cytosolic translation / ribosome'
    if 'amino acid' in p: return 'Amino-acid metabolism'
    if 'protein localization' in p or 'protein import' in p: return 'Protein localization / mitochondrial import'
    if 'macromolecule biosynthetic' in p: return 'Macromolecule biosynthesis'
    return pathway

def independent_module(pathway):
    """Themes for the T-cell-independent 22-pathway signed-IG core."""
    p=pathway.lower()
    if any(x in p for x in ['carnitine','lipid','triglyceride','bile secretion']): return 'Lipid, carnitine, and bile metabolism'
    if any(x in p for x in ['glycine','serine','threonine','glyoxylate','pyruvate']): return 'Amino-acid and central-carbon metabolism'
    if any(x in p for x in ['immune','biotic stimulus','complement']): return 'Innate immune and complement regulation'
    if any(x in p for x in ['atf4','perk','unfolded protein']): return 'ER stress / integrated stress response'
    if any(x in p for x in ['foxo','rora']): return 'Stress and metabolic transcription'
    if 'fluid transport' in p or 'water transport' in p: return 'Fluid and water transport'
    if 'insulin-like growth factor' in p: return 'IGF-receptor signaling'
    return pathway

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
    # Exact pathway overlap and conserved biological-module summary.
    membership=(pathways.assign(robust=pathways.classification.eq('protocol_robust'))
        .pivot_table(index=['source','pathway'],columns='cohort',values='robust',aggfunc='max',fill_value=False)
        .reindex(columns=COHORTS,fill_value=False).reset_index())
    membership.to_csv(OUT/'pathways/robust_pathway_membership.csv',index=False)
    patterns=[]
    for bits,q in membership.groupby(COHORTS,dropna=False):
        label=' & '.join(c for c,on in zip(COHORTS,bits) if on) or 'None'
        patterns.append({'intersection':label,**{c:bool(on) for c,on in zip(COHORTS,bits)},'pathways':len(q)})
    intersections=pd.DataFrame(patterns).sort_values('pathways',ascending=False)
    intersections.to_csv(OUT/'summary/pathway_upset_intersections.csv',index=False)
    common=membership[membership[COHORTS].all(axis=1)][['source','pathway']]
    full=pathways.merge(common,on=['source','pathway'],how='inner')
    full['signed_response']=(full.original_score+full.remeasurement_score)/2
    full['response_magnitude']=full.signed_response.abs()
    full['module']=full.pathway.map(conserved_module)
    full.to_csv(OUT/'pathways/conserved_core_full.csv',index=False)
    modules=(full.groupby(['module','cohort']).agg(signed_response=('signed_response','mean'),
        response_magnitude=('response_magnitude','mean'),pathways=('pathway','nunique')).reset_index())
    modules.to_csv(OUT/'pathways/conserved_core_modules.csv',index=False)
    # Primary T-cell-independent core: full signed input-gene IG pathway profiles.
    igp=pd.read_parquet(T4/'task4_multilayer_reproducibility/pathway_profiles.parquet')
    ig_names={'RR1':('RR1_original','RR1_remeasurement'),'RR3-39':('RR3_39_original','RR3_39_remeasurement'),'RR3-40':('RR3_40_original','RR3_40_remeasurement')}
    igw=igp.pivot(index=['source','pathway'],columns='response',values='score');ind_rows=[];ind_sets={}
    for c,(a,b) in ig_names.items():
        ra=igw[a].abs().rank(ascending=False,method='min');rb=igw[b].abs().rank(ascending=False,method='min');keep=ra.le(250)&rb.le(250)&(np.sign(igw[a])==np.sign(igw[b]));ind_sets[c]=set(igw.index[keep])
        for idx in igw.index:
            ind_rows.append({'cohort':c,'source':idx[0],'pathway':idx[1],'original_score':igw.loc[idx,a],'remeasurement_score':igw.loc[idx,b],'rank_original':ra.loc[idx],'rank_remeasurement':rb.loc[idx],'protocol_robust':bool(keep.loc[idx])})
    independent=pd.DataFrame(ind_rows);independent.to_csv(OUT/'pathways/independent_signed_ig_pathway_classification.csv',index=False)
    ind_membership=[]
    for idx in igw.index:ind_membership.append({'source':idx[0],'pathway':idx[1],**{c:idx in ind_sets[c] for c in COHORTS}})
    ind_membership=pd.DataFrame(ind_membership);ind_membership.to_csv(OUT/'pathways/independent_robust_pathway_membership.csv',index=False)
    ind_patterns=[]
    for bits,q in ind_membership.groupby(COHORTS):
        ind_patterns.append({'intersection':' & '.join(c for c,on in zip(COHORTS,bits) if on) or 'None',**{c:bool(on) for c,on in zip(COHORTS,bits)},'pathways':len(q)})
    ind_intersections=pd.DataFrame(ind_patterns).sort_values('pathways',ascending=False);ind_intersections.to_csv(OUT/'summary/independent_pathway_upset_intersections.csv',index=False)
    ind_common=set.intersection(*ind_sets.values());ind_full=independent[independent.set_index(['source','pathway']).index.isin(ind_common)].copy();ind_full['signed_response']=(ind_full.original_score+ind_full.remeasurement_score)/2;ind_full['response_magnitude']=ind_full.signed_response.abs();ind_full['module']=ind_full.pathway.map(independent_module);ind_full.to_csv(OUT/'pathways/independent_conserved_core_full.csv',index=False)
    ind_modules=(ind_full.groupby(['module','cohort']).agg(signed_response=('signed_response','mean'),response_magnitude=('response_magnitude','mean'),pathways=('pathway','nunique')).reset_index());ind_modules.to_csv(OUT/'pathways/independent_conserved_core_modules.csv',index=False)
    (OUT/'summary/independent_pathway_core_summary.json').write_text(json.dumps({'method':'full signed input-gene IG pathway profiles; no T-cell projection','criterion':'mutual Top-250 absolute pathway profile with concordant sign across original and remeasurement','robust_pathways':{c:len(ind_sets[c]) for c in COHORTS},'shared_all_three':len(ind_common),'collapsed_themes':sorted(ind_full.module.unique()),'former_pc12_core':'29-pathway/six-module result retained only as secondary T-cell PolyA/ribo technical control'},indent=2)+'\n')
    # Conventional-expression test of the six-module RR1/RR3 opposition.
    module_genes={m:set() for m in full.module.unique()}
    for m,q in full[['module','pathway']].drop_duplicates().groupby('module'):
        for pathway in q.pathway:
            matches=[genes for term,genes in resources.items() if term.split(': ',1)[-1]==pathway]
            for genes in matches: module_genes[m] |= (genes & universe)
    pd.DataFrame([{'module':m,'gene_symbol':g} for m,genes in module_genes.items() for g in sorted(genes)]).to_csv(OUT/'pathways/conserved_core_module_genes.csv',index=False)
    edger=pd.read_csv(T4/'task4_confounding_profiler/independent_biological_replication/edger_results.csv.gz')
    edge_ids={'RR1':'C14__OSD-48__RR1-NASA__37-day','RR3-39':'C01__OSD-137__RR3__39-day','RR3-40':'C02__OSD-137__RR3__40-day'}
    edge_rows=[];matched_rows=[];cross_rows=[];edge_gene_rows=[];consensus={};edge_vectors={}
    for c in COHORTS:
        eq=edger[(edger.contrast_id.eq(edge_ids[c]))&edger.tested.fillna(False)].set_index('gene_symbol');edge_vectors[c]=eq.logFC.dropna()
        rq=robust[c].set_index('gene_symbol');consensus[c]=(rq.expression_original+rq.expression_remeasurement)/2
        for m,genes in module_genes.items():
            eg=eq.loc[eq.index.intersection(genes)].dropna(subset=['logFC'])
            for gene,z in eg.iterrows():edge_gene_rows.append({'cohort':c,'module':m,'gene_symbol':gene,'logFC':z.logFC,'FDR':z.FDR})
            edge_rows.append({'cohort':c,'module':m,'genes':len(eg),'mean_logFC':eg.logFC.mean(),'median_logFC':eg.logFC.median(),'positive_fraction':(eg.logFC>0).mean(),'negative_fraction':(eg.logFC<0).mean(),'edgeR_contrast':edge_ids[c]})
            mg=rq.loc[rq.index.intersection(genes)];a=mg.expression_original.to_numpy();b=mg.expression_remeasurement.to_numpy()
            matched_rows.append({'cohort':c,'module':m,'genes':len(mg),'original_mean_effect':np.mean(a),'remeasurement_mean_effect':np.mean(b),'original_median_effect':np.median(a),'remeasurement_median_effect':np.median(b),'direction_agreement':np.mean(np.sign(a)==np.sign(b)),'original_remeasurement_cosine':np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)),'original_remeasurement_spearman':pd.Series(a).corr(pd.Series(b),method='spearman')})
    for m,genes in module_genes.items():
        for a,b in [('RR1','RR3-39'),('RR1','RR3-40'),('RR3-39','RR3-40')]:
            idx=consensus[a].index.intersection(consensus[b].index).intersection(pd.Index(genes));x=consensus[a].loc[idx].to_numpy();y=consensus[b].loc[idx].to_numpy()
            cross_rows.append({'data_type':'Exact matched mean Δlog1p(TPM)','module':m,'comparison':f'{a} ↔ {b}','genes':len(idx),'cosine':np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)),'spearman':pd.Series(x).corr(pd.Series(y),method='spearman'),'direction_agreement':np.mean(np.sign(x)==np.sign(y))})
            idx=edge_vectors[a].index.intersection(edge_vectors[b].index).intersection(pd.Index(genes));x=edge_vectors[a].loc[idx].to_numpy();y=edge_vectors[b].loc[idx].to_numpy()
            cross_rows.append({'data_type':'Raw-count edgeR log2FC (original cohorts)','module':m,'comparison':f'{a} ↔ {b}','genes':len(idx),'cosine':np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)),'spearman':pd.Series(x).corr(pd.Series(y),method='spearman'),'direction_agreement':np.mean(np.sign(x)==np.sign(y))})
    edge_gene=pd.DataFrame(edge_gene_rows);edge_gene.to_csv(OUT/'genes/conserved_modules_edger_logfc.csv',index=False)
    pd.DataFrame(edge_rows).to_csv(OUT/'summary/conventional_module_edger_summary.csv',index=False)
    pd.DataFrame(matched_rows).to_csv(OUT/'summary/conventional_module_matched_consistency.csv',index=False)
    pd.DataFrame(cross_rows).to_csv(OUT/'summary/conventional_module_cross_cohort_similarity.csv',index=False)
    # Three-method recovery of the six conserved modules.
    gsea=pd.read_parquet(T4/'task4_confounding_profiler/independent_biological_replication/pathway_enrichment.parquet')
    gsea['source_key']=gsea.source.replace({'REAC':'Reactome'});gsea['pathway_key']=gsea.pathway.str.casefold()
    gsea_ids={'RR1':'C14__OSD-48__RR1-NASA__37-day','RR3-39':'C01__OSD-137__RR3__39-day','RR3-40':'C02__OSD-137__RR3__40-day'}
    # Validate each member of the primary, T-cell-independent 22-pathway core
    # against conventional ranked GSEA and a conservative, module-level literature audit.
    literature={
        'Lipid, carnitine, and bile metabolism':(
            'previously reported in spaceflight',
            'Mouse-liver spaceflight studies directly report altered lipid/fatty-acid metabolism, carnitine-related metabolites, and bile-acid/bile-secretion biology.',
            'https://pmc.ncbi.nlm.nih.gov/articles/PMC4838331/; https://pmc.ncbi.nlm.nih.gov/articles/PMC6915713/; https://pmc.ncbi.nlm.nih.gov/articles/PMC11362537/'),
        'Amino-acid and central-carbon metabolism':(
            'previously reported in spaceflight',
            'Rodent liver spaceflight studies report amino-acid catabolism and changes in glycine/serine/threonine, gluconeogenic, and central metabolic readouts.',
            'https://pmc.ncbi.nlm.nih.gov/articles/PMC3635244/; https://pmc.ncbi.nlm.nih.gov/articles/PMC4838331/; https://pmc.ncbi.nlm.nih.gov/articles/PMC5443495/'),
        'ER stress / integrated stress response':(
            'related biology previously reported',
            'Spaceflight multi-omics and RR1 liver analyses report mitochondrial/proteostasis stress and ER/ribosome-related processes, but the exact PERK-ATF4 pathways are not uniformly established across these cohorts.',
            'https://pmc.ncbi.nlm.nih.gov/articles/PMC7870178/; https://pmc.ncbi.nlm.nih.gov/articles/PMC9576569/'),
        'Innate immune and complement regulation':(
            'related biology previously reported',
            'Rodent spaceflight studies report altered innate immune and oxidative-stress biology, including liver gene-expression changes; exact complement-cascade recurrence is less directly established.',
            'https://pmc.ncbi.nlm.nih.gov/articles/PMC2692779/; https://pmc.ncbi.nlm.nih.gov/articles/PMC5443495/; https://pmc.ncbi.nlm.nih.gov/articles/PMC7828077/'),
        'Stress and metabolic transcription':(
            'related biology previously reported',
            'Spaceflight mouse liver work directly supports PPAR/oxidative/metabolic transcriptional remodeling, while exact FOXO/RORA pathway recurrence is less directly documented.',
            'https://pmc.ncbi.nlm.nih.gov/articles/PMC4838331/; https://pmc.ncbi.nlm.nih.gov/articles/PMC8660801/'),
        'Fluid and water transport':(
            'potentially novel/hypothesis-generating',
            'Related hepatic ion/solute transport and bile-flow biology has been reported, but a reproducible spaceflight water/fluid-transport program was not directly identified in the focused literature audit.',
            'https://pmc.ncbi.nlm.nih.gov/articles/PMC5578152/'),
        'IGF-receptor signaling':(
            'potentially novel/hypothesis-generating',
            'Hormone/peptide-response pathways have been reported in spaceflight liver, but direct evidence for recurrent IGF-receptor regulation in the relevant rodent liver response was not found in the focused audit.',
            'https://pmc.ncbi.nlm.nih.gov/articles/PMC6321533/')}
    validation=[]
    core_terms=ind_full[['module','source','pathway']].drop_duplicates()
    for _,term in core_terms.iterrows():
        row={'module':term.module,'source':term.source,'pathway':term.pathway}
        sig_count=0;tested_count=0
        for c in COHORTS:
            q=gsea[(gsea.analysis.eq('edgeR_expression'))&gsea.contrast_id.eq(gsea_ids[c])&gsea.source_key.eq(term.source)&gsea.pathway_key.eq(term.pathway.casefold())]
            tested=bool(len(q));detected=bool(tested and q.fdr.iloc[0] < .05);tested_count+=tested;sig_count+=detected
            key=c.replace('-','_')
            row[f'{key}_IG']=True;row[f'{key}_GSEA_tested']=tested;row[f'{key}_GSEA_detected']=detected
            row[f'{key}_GSEA_NES']=q.nes.iloc[0] if tested else np.nan;row[f'{key}_GSEA_FDR']=q.fdr.iloc[0] if tested else np.nan
            row[f'{key}_GSEA_direction']='positive' if tested and q.nes.iloc[0]>0 else 'negative' if tested else 'not tested'
        row['GSEA_recurrence']=sig_count
        support=[]
        for c in COHORTS:
            key=c.replace('-','_');label='sig' if row[f'{key}_GSEA_detected'] else 'NS' if row[f'{key}_GSEA_tested'] else 'not tested'
            support.append(f'{c}: {label}')
        row['DE_GSEA_support']='; '.join(support)
        row['evidence_classification']='IG + conventional support' if sig_count else ('reproducible IG reprioritization with weak/absent GSEA support' if tested_count==len(COHORTS) else 'ambiguous')
        prior,basis,refs=literature[term.module];row['prior_spaceflight_evidence']=prior;row['literature_basis']=basis;row['literature_references']=refs
        row['interpretation']=f"{row['evidence_classification']}; {prior}."
        validation.append(row)
    validation=pd.DataFrame(validation).sort_values(['module','pathway'])
    validation.to_csv(OUT/'pathways/independent_core_pathway_validation.csv',index=False)
    validation[['module','prior_spaceflight_evidence','literature_basis','literature_references']].drop_duplicates().to_csv(OUT/'pathways/spaceflight_literature_evidence.csv',index=False)
    pd.DataFrame([{'pathways':len(validation),'GSEA_tested_all_three':int((validation[[f'{c.replace("-","_")}_GSEA_tested' for c in COHORTS]].sum(axis=1)==3).sum()),'GSEA_supported_any':int(validation.GSEA_recurrence.gt(0).sum()),'GSEA_supported_all_three':int(validation.GSEA_recurrence.eq(3).sum()),'IG_reprioritized_tested_all_three':int(((validation.GSEA_recurrence==0)&(validation[[f'{c.replace("-","_")}_GSEA_tested' for c in COHORTS]].sum(axis=1)==3)).sum()),'ambiguous_not_fully_tested':int((validation[[f'{c.replace("-","_")}_GSEA_tested' for c in COHORTS]].sum(axis=1)<3).sum())}]).to_csv(OUT/'summary/independent_core_pathway_validation_summary.csv',index=False)
    method_rows=[];ig_enrichments=[]
    module_terms=full[['module','source','pathway']].drop_duplicates()
    for c in COHORTS:
        # DE/GSEA is available for the full original biological stratum only.
        dg=gsea[(gsea.analysis.eq('edgeR_expression'))&gsea.contrast_id.eq(gsea_ids[c])]
        rq=robust[c].copy();ig_genes=set(rq.loc[rq.robust_attribution,'gene_symbol']);ie=ora(c,ig_genes,universe,resources);ie['pathway']=ie.term.str.split(': ',n=1).str[-1];ig_enrichments.append(ie)
        av=(rq.attribution_original+rq.attribution_remeasurement)/2;rq['signed_ig_z']=(av-av.mean())/(av.std(ddof=0) or 1)
        for m,t in module_terms.groupby('module'):
            tt=t.assign(source_key=t.source,pathway_key=t.pathway.str.casefold());q=dg.merge(tt[['source_key','pathway_key']],on=['source_key','pathway_key'],how='inner');sig=q[q.fdr.lt(.05)]
            method_rows.append({'module':m,'cohort':c,'method':'DE/GSEA','support':len(sig)>0,'signed_score':sig.nes.median() if len(sig) else np.nan,'supported_terms':len(sig),'module_terms':len(t),'direction':'positive' if len(sig) and sig.nes.median()>0 else 'negative' if len(sig) else 'not significant','provenance':'raw-count edgeR ranked GSEA; original full stratum'})
            # ORA of the already-defined technically reproducible IG genes.
            isig=ie[ie.pathway.isin(t.pathway)&ie.fdr.lt(.05)];member=module_genes[m]&ig_genes;vals=rq.loc[rq.gene_symbol.isin(member),'signed_ig_z']
            method_rows.append({'module':m,'cohort':c,'method':'IG enrichment','support':len(isig)>0,'signed_score':vals.median() if len(isig) and len(vals) else np.nan,'supported_terms':len(isig),'module_terms':len(t),'direction':'positive contribution' if len(isig) and vals.median()>0 else 'negative contribution' if len(isig) else 'not significant','provenance':'ORA of reproducible IG genes; FDR<0.05; direction from median standardized signed IG among robust member genes'})
            cq=pathways[(pathways.cohort.eq(c))].merge(t[['source','pathway']],on=['source','pathway'],how='inner');cv=cq[cq.classification.eq('protocol_robust')];scores=(cv.original_score+cv.remeasurement_score)/2
            method_rows.append({'module':m,'cohort':c,'method':'Contextual pathway profile','support':len(cv)>0,'signed_score':scores.median() if len(scores) else np.nan,'supported_terms':len(cv),'module_terms':len(t),'direction':'positive' if len(scores) and scores.median()>0 else 'negative' if len(scores) else 'not supported','provenance':'contextual gene response projected into controlled T-cell PC1-2; mutual Top-250 and concordant sign'})
    method_matrix=pd.DataFrame(method_rows)
    method_matrix.to_csv(OUT/'pathways/conserved_module_three_method_matrix.csv',index=False)
    pd.concat(ig_enrichments,ignore_index=True).to_csv(OUT/'pathways/robust_ig_gene_enrichment.csv',index=False)
    overlap=(method_matrix.pivot_table(index=['module','cohort'],columns='method',values='support',aggfunc='max',fill_value=False).reset_index())
    meth=['DE/GSEA','IG enrichment','Contextual pathway profile'];overlap['support_count']=overlap[meth].sum(axis=1)
    overlap['classification']=np.select([overlap.support_count.eq(3),(overlap['IG enrichment']&~overlap['DE/GSEA']&~overlap['Contextual pathway profile']),(overlap['Contextual pathway profile']&~overlap['DE/GSEA']&~overlap['IG enrichment'])],['all_three_methods','IG_only','contextual_profile_only'],default='partial_or_DE_only')
    overlap.to_csv(OUT/'summary/conserved_module_method_overlap.csv',index=False)
    # Primary three-method comparison around the independent 22-pathway core.
    ind_module_terms=ind_full[['module','source','pathway']].drop_duplicates();ind_module_genes={m:set() for m in ind_module_terms.module.unique()}
    for m,q in ind_module_terms.groupby('module'):
        for pathway in q.pathway:
            for genes in [v for term,v in resources.items() if term.split(': ',1)[-1]==pathway]:ind_module_genes[m] |= (genes&universe)
    ind_method=[];ind_ig_enrich=[]
    for c in COHORTS:
        dg=gsea[(gsea.analysis.eq('edgeR_expression'))&gsea.contrast_id.eq(gsea_ids[c])];rq=robust[c].copy();ig_genes=set(rq.loc[rq.robust_attribution,'gene_symbol']);ie=ora(c,ig_genes,universe,resources);ie['pathway']=ie.term.str.split(': ',n=1).str[-1];ind_ig_enrich.append(ie)
        av=(rq.attribution_original+rq.attribution_remeasurement)/2;rq['signed_ig_z']=(av-av.mean())/(av.std(ddof=0) or 1)
        for m,t in ind_module_terms.groupby('module'):
            tt=t.assign(source_key=t.source,pathway_key=t.pathway.str.casefold());sig=dg.merge(tt[['source_key','pathway_key']],on=['source_key','pathway_key'],how='inner').query('fdr < 0.05')
            ind_method.append({'module':m,'cohort':c,'method':'DE/GSEA','support':len(sig)>0,'signed_score':sig.nes.median() if len(sig) else np.nan,'supported_terms':len(sig),'module_terms':len(t),'direction':'positive NES' if len(sig) and sig.nes.median()>0 else 'negative NES' if len(sig) else 'not significant'})
            isig=ie[ie.pathway.isin(t.pathway)&ie.fdr.lt(.05)];member=ind_module_genes[m]&ig_genes;vals=rq.loc[rq.gene_symbol.isin(member),'signed_ig_z']
            ind_method.append({'module':m,'cohort':c,'method':'IG enrichment','support':len(isig)>0,'signed_score':vals.median() if len(isig) and len(vals) else np.nan,'supported_terms':len(isig),'module_terms':len(t),'direction':'positive contribution' if len(isig) and vals.median()>0 else 'negative contribution' if len(isig) else 'not significant'})
            z=ind_modules[(ind_modules.module.eq(m))&ind_modules.cohort.eq(c)].iloc[0]
            ind_method.append({'module':m,'cohort':c,'method':'Robust signed-IG pathway profile','support':True,'signed_score':z.signed_response,'supported_terms':int(z.pathways),'module_terms':len(t),'direction':'positive contribution' if z.signed_response>0 else 'negative contribution'})
    ind_method=pd.DataFrame(ind_method);ind_method.to_csv(OUT/'pathways/independent_core_three_method_matrix.csv',index=False);pd.concat(ind_ig_enrich,ignore_index=True).to_csv(OUT/'pathways/independent_core_robust_ig_enrichment.csv',index=False)
    ind_overlap=ind_method.pivot_table(index=['module','cohort'],columns='method',values='support',aggfunc='max',fill_value=False).reset_index();imeth=['DE/GSEA','IG enrichment','Robust signed-IG pathway profile'];ind_overlap['support_count']=ind_overlap[imeth].sum(axis=1);ind_overlap['classification']=np.select([ind_overlap.support_count.eq(3),(ind_overlap['IG enrichment']&~ind_overlap['DE/GSEA']),(~ind_overlap['IG enrichment']&~ind_overlap['DE/GSEA'])],['all_three_methods','Bridge_IG_plus_profile','pathway_profile_only'],default='partial');ind_overlap.to_csv(OUT/'summary/independent_core_method_overlap.csv',index=False)
    ind_edge=[];ind_matched=[];ind_cross=[]
    for c in COHORTS:
        eq=edge_vectors[c];rq=robust[c].set_index('gene_symbol')
        for m,genes in ind_module_genes.items():
            v=eq.loc[eq.index.intersection(genes)];ind_edge.append({'cohort':c,'module':m,'genes':len(v),'mean_logFC':v.mean(),'median_logFC':v.median(),'positive_fraction':(v>0).mean()})
            q=rq.loc[rq.index.intersection(genes)];a=q.expression_original.to_numpy();b=q.expression_remeasurement.to_numpy();ind_matched.append({'cohort':c,'module':m,'genes':len(q),'original_mean_effect':np.mean(a),'remeasurement_mean_effect':np.mean(b),'direction_agreement':np.mean(np.sign(a)==np.sign(b)),'original_remeasurement_cosine':np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))})
    for m,genes in ind_module_genes.items():
        for a,b in [('RR1','RR3-39'),('RR1','RR3-40'),('RR3-39','RR3-40')]:
            idx=consensus[a].index.intersection(consensus[b].index).intersection(pd.Index(genes));x=consensus[a].loc[idx].to_numpy();y=consensus[b].loc[idx].to_numpy();ind_cross.append({'module':m,'comparison':f'{a} ↔ {b}','genes':len(idx),'cosine':np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)),'spearman':pd.Series(x).corr(pd.Series(y),method='spearman'),'direction_agreement':np.mean(np.sign(x)==np.sign(y))})
    pd.DataFrame(ind_edge).to_csv(OUT/'summary/independent_core_conventional_edger_summary.csv',index=False);pd.DataFrame(ind_matched).to_csv(OUT/'summary/independent_core_matched_expression_consistency.csv',index=False);pd.DataFrame(ind_cross).to_csv(OUT/'summary/independent_core_cross_cohort_expression.csv',index=False)
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
    # Final analysis: reproducible attribution not prioritized by expression.
    attr_sets={};expr_sets={};attr_rows=[];compare_rows=[];graph_rows=[]
    for c,q in robust.items():
        q=q.copy();q['cohort']=c
        q['attribution_only']=q.robust_attribution & ~q.robust_expression
        q['expression_only']=q.robust_expression & ~q.robust_attribution
        q['mean_abs_expression_effect']=(q.expression_original.abs()+q.expression_remeasurement.abs())/2
        q['mean_expression_rank']=(q.expression_rank_original+q.expression_rank_remeasurement)/2
        q['mean_attribution_rank']=(q.attribution_rank_original+q.attribution_rank_remeasurement)/2
        a=q[q.attribution_only].copy();e=q[q.expression_only].copy();attr_sets[c]=set(a.gene_symbol);expr_sets[c]=set(e.gene_symbol)
        a.to_csv(OUT/'genes'/f'{KEY[c]}_attribution_only.csv',index=False)
        for label,z in [('Attribution only',a),('Expression only',e),('Robust expression (all)',q[q.robust_expression])]:
            compare_rows.append({'cohort':c,'gene_set':label,'genes':len(z),'median_abs_expression_effect':z.mean_abs_expression_effect.median(),'median_expression_rank':z.mean_expression_rank.median(),'median_attribution_rank':z.mean_attribution_rank.median()})
            graph_rows.append({'cohort':c,'gene_set':label,'genes':len(z),'graph_supported':int(z.robust_graph.sum()),'graph_support_fraction':z.robust_graph.mean(),'median_graph_local_cosine':z.graph_local_cosine.median()})
    all_attr=set.union(*attr_sets.values())
    for gene in sorted(all_attr):
        flags={c:gene in attr_sets[c] for c in COHORTS};n=sum(flags.values())
        if n>=2:
            row={'gene_symbol':gene,'cohorts':'; '.join(c for c in COHORTS if flags[c]),'cohort_count':n}
            for c in COHORTS:
                row[f'{c}_attribution_only']=flags[c]
                if flags[c]:
                    z=robust[c].set_index('gene_symbol').loc[gene]
                    row[f'{c}_expression_rank_mean']=(z.expression_rank_original+z.expression_rank_remeasurement)/2
                    row[f'{c}_attribution_rank_mean']=(z.attribution_rank_original+z.attribution_rank_remeasurement)/2
                    row[f'{c}_graph_support']=bool(z.robust_graph)
            attr_rows.append(row)
    shared_attr=pd.DataFrame(attr_rows).sort_values(['cohort_count','gene_symbol'],ascending=[False,True])
    shared_attr.to_csv(OUT/'genes/shared_attribution_only_genes.csv',index=False)
    comparison=pd.DataFrame(compare_rows);comparison.to_csv(OUT/'summary/attribution_vs_expression_summary.csv',index=False)
    graph_support=pd.DataFrame(graph_rows);graph_support.to_csv(OUT/'summary/attribution_only_graph_support.csv',index=False)
    attr_enrichment=[]
    for c in COHORTS: attr_enrichment.append(ora(c,attr_sets[c],universe,resources).assign(gene_set='attribution_only'))
    recurring={x for x in all_attr if sum(x in attr_sets[c] for c in COHORTS)>=2}
    if recurring: attr_enrichment.append(ora('Recurring ≥2 cohorts',recurring,universe,resources).assign(gene_set='recurring_attribution_only'))
    attr_enrichment=pd.concat(attr_enrichment,ignore_index=True);attr_enrichment.to_csv(OUT/'pathways/attribution_only_enrichment.csv',index=False)
    pd.DataFrame([{'analysis':'Existing IG deletion cross-reference','compatible':False,'reason':'Existing deletion sweeps validate mode-level IG-ranked panels, not these cohort-specific technically reproducible attribution-only sets; no gene-set-specific causal deletion claim is made.'}]).to_csv(OUT/'summary/deletion_compatibility.csv',index=False)
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
    lim=np.nanpercentile(np.abs(M),95);fig,ax=plt.subplots(figsize=(11,10));im=ax.imshow(M,aspect='auto',cmap='RdBu_r',vmin=-lim,vmax=lim,interpolation='nearest')
    # Seaborn's axes grid bisected every heatmap cell and made six columns look
    # like twelve. Disable it and retain only cohort-pair separators.
    ax.grid(False)
    ax.set_xticks(range(6),['Original','Remeasurement']*3,rotation=30,ha='right')
    ax.set_yticks(range(len(selected)),selected,fontsize=7)
    for boundary in (1.5,3.5): ax.axvline(boundary,color='#202020',lw=1.5)
    top=ax.secondary_xaxis('top');top.set_xticks([.5,2.5,4.5],['RR1','RR3-39','RR3-40']);top.tick_params(length=0,pad=8);top.grid(False)
    fig.colorbar(im,ax=ax,label='Existing signed pathway score');ax.set_title('Robust and measurement-sensitive pathway profiles',pad=12);fig.tight_layout()
    for e in ['png','pdf']:fig.savefig(FIG/f'pathway_profiles.{e}',dpi=300,bbox_inches='tight');plt.close(fig)
    # UpSet-style exact intersection plot without an additional dependency.
    shown=intersections[intersections.pathways.gt(0) & intersections[COHORTS].any(axis=1)].copy().sort_values('pathways',ascending=False).reset_index(drop=True)
    fig=plt.figure(figsize=(9,6.2),layout='constrained');gs=fig.add_gridspec(2,1,height_ratios=[3.6,1.1]);ax=fig.add_subplot(gs[0]);mat=fig.add_subplot(gs[1],sharex=ax)
    x=np.arange(len(shown));bars=ax.bar(x,shown.pathways,color='#355C7D');ax.bar_label(bars,padding=3,fontsize=9);ax.set(ylabel='Pathways',title='Protocol-robust pathway overlap');ax.set_xticks([]);ax.grid(axis='x',visible=False);ax.spines[['top','right']].set_visible(False)
    for i,row in shown.iterrows():
        active=[]
        for y,c in enumerate(COHORTS):
            on=bool(row[c]);mat.scatter(i,y,s=75,color='#202020' if on else '#D8D8D8',zorder=3)
            if on:active.append(y)
        if len(active)>1:mat.plot([i,i],[min(active),max(active)],color='#202020',lw=2,zorder=2)
    mat.set_yticks(range(3),COHORTS);mat.invert_yaxis();mat.set_xticks(x);mat.set_xticklabels([]);mat.tick_params(axis='x',length=0);mat.set_xlabel('Exact intersections indicated by connected membership dots');mat.grid(False);mat.spines[['top','right','bottom']].set_visible(False)
    for e in ['png','pdf']:fig.savefig(FIG/f'robust_pathway_upset.{e}',dpi=300,bbox_inches='tight');plt.close(fig)
    # Conserved-core module dot plot. Color retains direction; area is magnitude.
    order=(modules.groupby('module').response_magnitude.mean().sort_values().index.tolist());fig,ax=plt.subplots(figsize=(8.6,5.6));lim=modules.response_magnitude.max()
    for _,r in modules.iterrows():
        ax.scatter(COHORTS.index(r.cohort),order.index(r.module),s=45+260*r.response_magnitude/lim,
            c=r.signed_response,cmap='RdBu_r',vmin=-lim,vmax=lim,edgecolor='#333',linewidth=.4)
    sm=plt.cm.ScalarMappable(cmap='RdBu_r',norm=plt.Normalize(-lim,lim));fig.colorbar(sm,ax=ax,label='Mean signed FLT−GC pathway response')
    ax.set_xticks(range(3),COHORTS);ax.set_yticks(range(len(order)),order);ax.set_title('Biological modules robust in all three comparisons');ax.set_xlabel('Dot area = mean absolute response across member pathways');ax.grid(False);ax.spines[['top','right']].set_visible(False);fig.tight_layout()
    for e in ['png','pdf']:fig.savefig(FIG/f'conserved_core_dotplot.{e}',dpi=300,bbox_inches='tight');plt.close(fig)
    # Primary T-cell-independent UpSet and conserved signed-IG module plot.
    shown=ind_intersections[ind_intersections.pathways.gt(0)&ind_intersections[COHORTS].any(axis=1)].sort_values('pathways',ascending=False).reset_index(drop=True)
    fig=plt.figure(figsize=(9,6.2),layout='constrained');gs=fig.add_gridspec(2,1,height_ratios=[3.6,1.1]);ax=fig.add_subplot(gs[0]);mat=fig.add_subplot(gs[1],sharex=ax);x=np.arange(len(shown));bars=ax.bar(x,shown.pathways,color='#355C7D');ax.bar_label(bars,padding=3,fontsize=9);ax.set(ylabel='Pathways',title='Protocol-robust signed-IG pathway overlap');ax.set_xticks([]);ax.grid(axis='x',visible=False);ax.spines[['top','right']].set_visible(False)
    for i,row in shown.iterrows():
        active=[]
        for y,c in enumerate(COHORTS):on=bool(row[c]);mat.scatter(i,y,s=75,color='#202020' if on else '#D8D8D8',zorder=3);active.append(y) if on else None
        if len(active)>1:mat.plot([i,i],[min(active),max(active)],color='#202020',lw=2,zorder=2)
    mat.set_yticks(range(3),COHORTS);mat.invert_yaxis();mat.set_xticks(x);mat.set_xticklabels([]);mat.tick_params(axis='x',length=0);mat.set_xlabel('Exact intersections indicated by connected membership dots');mat.grid(False);mat.spines[['top','right','bottom']].set_visible(False)
    for e in ['png','pdf']:fig.savefig(FIG/f'independent_robust_pathway_upset.{e}',dpi=300,bbox_inches='tight');plt.close(fig)
    order=ind_modules.groupby('module').response_magnitude.mean().sort_values().index.tolist();lim=ind_modules.response_magnitude.max();fig,ax=plt.subplots(figsize=(9.4,6.2))
    for _,r in ind_modules.iterrows():ax.scatter(COHORTS.index(r.cohort),order.index(r.module),s=45+260*r.response_magnitude/lim,c=r.signed_response,cmap='RdBu_r',vmin=-lim,vmax=lim,edgecolor='#333',linewidth=.4)
    sm=plt.cm.ScalarMappable(cmap='RdBu_r',norm=plt.Normalize(-lim,lim));fig.colorbar(sm,ax=ax,label='Mean signed IG pathway-profile score');ax.set_xticks(range(3),COHORTS);ax.set_yticks(range(len(order)),order);ax.set_title('Pathway modules robust in all three comparisons');ax.set_xlabel('Dot area = mean absolute signed-IG pathway score');ax.grid(False);ax.spines[['top','right']].set_visible(False);fig.tight_layout()
    for e in ['png','pdf']:fig.savefig(FIG/f'independent_conserved_core_dotplot.{e}',dpi=300,bbox_inches='tight');plt.close(fig)
    # Attribution versus expression rank and graph-support comparison.
    fig,axes=plt.subplots(1,3,figsize=(14,4.6),sharex=True,sharey=True)
    for ax,c in zip(axes,COHORTS):
        q=robust[c].copy();x=(q.expression_rank_original+q.expression_rank_remeasurement)/2;y=(q.attribution_rank_original+q.attribution_rank_remeasurement)/2
        ax.scatter(x,y,s=3,c='#C8C8C8',alpha=.18,rasterized=True,label='Other genes')
        a=q.robust_attribution&~q.robust_expression;e=q.robust_expression&~q.robust_attribution
        ax.scatter(x[e],y[e],s=12,c='#4C78A8',alpha=.65,label='Expression only');ax.scatter(x[a],y[a],s=14,c='#E45756',alpha=.7,label='Attribution only')
        ax.axvline(500.5,color='#666',ls=':',lw=.8);ax.axhline(500.5,color='#666',ls=':',lw=.8);ax.set(title=c,xlabel='Mean expression rank (lower is stronger)');ax.invert_xaxis();ax.invert_yaxis();ax.grid(False)
    axes[0].set_ylabel('Mean IG rank (lower is stronger)');axes[-1].legend(frameon=False,fontsize=8);fig.suptitle('Reproducible IG priorities versus expression-response rank');fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(FIG/f'attribution_vs_expression_rank.{ext}',dpi=300,bbox_inches='tight');plt.close(fig)
    gp=graph_support[graph_support.gene_set.isin(['Attribution only','Expression only'])].pivot(index='cohort',columns='gene_set',values='graph_support_fraction').reindex(COHORTS)
    fig,ax=plt.subplots(figsize=(8,4.8));gp.plot.bar(ax=ax,color=['#E45756','#4C78A8']);ax.set(ylabel='Fraction with robust contextual-graph support',xlabel='',ylim=(0,1),title='Contextual-neighborhood support');ax.tick_params(axis='x',rotation=0);ax.legend(title='');ax.grid(axis='x',visible=False);fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(FIG/f'attribution_only_graph_support.{ext}',dpi=300,bbox_inches='tight');plt.close(fig)
    # True raw-count edgeR logFC distributions for the original cohorts.
    mods=list(modules.groupby('module').response_magnitude.mean().sort_values(ascending=False).index);fig,axes=plt.subplots(2,3,figsize=(14,8),sharex=True)
    rng=np.random.default_rng(17)
    for ax,m in zip(axes.flat,mods):
        q=edge_gene[edge_gene.module.eq(m)]
        vals=[]
        for j,c in enumerate(COHORTS):
            v=q.loc[q.cohort.eq(c),'logFC'].dropna().to_numpy();vals.append(v);ax.scatter(j+rng.uniform(-.13,.13,len(v)),v,s=7,alpha=.22,color=['#C44E52','#4C72B0','#55A868'][j],rasterized=True)
        ax.boxplot(vals,positions=range(3),widths=.48,showfliers=False,medianprops={'color':'black'},boxprops={'color':'#333'},whiskerprops={'color':'#555'},capprops={'color':'#555'});ax.axhline(0,color='#777',lw=.8);ax.set_xticks(range(3),COHORTS);ax.set_title(m,fontsize=10);ax.grid(axis='x',visible=False)
    axes[0,0].set_ylabel('edgeR FLT−GC log2 fold change');axes[1,0].set_ylabel('edgeR FLT−GC log2 fold change');fig.suptitle('Conventional expression confirms cohort-dependent direction of conserved modules');fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(FIG/f'conserved_modules_edger_logfc.{ext}',dpi=300,bbox_inches='tight');plt.close(fig)
    # Three-method support matrix: normalize color within method, annotate raw score.
    mm=method_matrix.copy();mm['scaled']=np.nan
    for method,q in mm.groupby('method'):
        den=q.signed_score.abs().max();mm.loc[q.index,'scaled']=q.signed_score/den if den else np.nan
    modules_order=mods[::-1];columns=[(c,m) for c in COHORTS for m in meth];A=np.full((len(modules_order),len(columns)),np.nan);labels=np.full(A.shape,'',dtype=object)
    for i,module in enumerate(modules_order):
        for j,(c,methname) in enumerate(columns):
            q=mm[(mm.module.eq(module))&mm.cohort.eq(c)&mm.method.eq(methname)].iloc[0]
            if q.support:A[i,j]=q.scaled;labels[i,j]=f'{q.signed_score:+.2f}'
            else:labels[i,j]='—'
    fig,ax=plt.subplots(figsize=(13,6.8));masked=np.ma.masked_invalid(A);im=ax.imshow(masked,cmap='RdBu_r',vmin=-1,vmax=1,aspect='auto');ax.set_facecolor('#F0F0F0');ax.grid(False)
    ax.set_xticks(range(len(columns)),[{'DE/GSEA':'DE','IG enrichment':'IG','Contextual pathway profile':'Context'}[m] for _,m in columns],rotation=30,ha='right');ax.set_yticks(range(len(modules_order)),modules_order)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):ax.text(j,i,labels[i,j],ha='center',va='center',fontsize=7,color='white' if np.isfinite(A[i,j]) and abs(A[i,j])>.55 else '#222')
    for x in (2.5,5.5):ax.axvline(x,color='#222',lw=1.5)
    top=ax.secondary_xaxis('top');top.set_xticks([1,4,7],COHORTS);top.tick_params(length=0,pad=7);top.grid(False)
    fig.colorbar(im,ax=ax,label='Signed score (scaled within method)');ax.set_title('Recovery of six protocol-robust modules across three analytical methods',pad=12);fig.text(.5,.01,'Cell text = raw method-specific score; gray = unsupported. Scores are not directly comparable across methods.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.04,1,1))
    for ext in ['png','pdf']:fig.savefig(FIG/f'conserved_modules_three_method_heatmap.{ext}',dpi=300,bbox_inches='tight');plt.close(fig)
    # Primary method matrix for the independent conserved core.
    mm=ind_method.copy();mm['scaled']=np.nan
    for method,q in mm.groupby('method'):
        den=q.signed_score.abs().max();mm.loc[q.index,'scaled']=q.signed_score/den if den else np.nan
    mods2=ind_modules.groupby('module').response_magnitude.mean().sort_values().index.tolist();columns=[(c,m) for c in COHORTS for m in imeth];A=np.full((len(mods2),len(columns)),np.nan);labels=np.full(A.shape,'',dtype=object)
    for i,module in enumerate(mods2):
        for j,(c,m) in enumerate(columns):
            q=mm[(mm.module.eq(module))&mm.cohort.eq(c)&mm.method.eq(m)].iloc[0];labels[i,j]=f'{q.signed_score:+.2f}' if q.support else '—';A[i,j]=q.scaled if q.support else np.nan
    fig,ax=plt.subplots(figsize=(13,7.2));im=ax.imshow(np.ma.masked_invalid(A),cmap='RdBu_r',vmin=-1,vmax=1,aspect='auto');ax.set_facecolor('#F0F0F0');ax.grid(False);ax.set_xticks(range(9),['DE','IG enrich.','IG profile']*3,rotation=30,ha='right');ax.set_yticks(range(len(mods2)),mods2)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):ax.text(j,i,labels[i,j],ha='center',va='center',fontsize=7,color='white' if np.isfinite(A[i,j]) and abs(A[i,j])>.55 else '#222')
    for x in (2.5,5.5):ax.axvline(x,color='#222',lw=1.5)
    top=ax.secondary_xaxis('top');top.set_xticks([1,4,7],COHORTS);top.tick_params(length=0,pad=7);top.grid(False);fig.colorbar(im,ax=ax,label='Signed score (scaled within method)');ax.set_title('Conventional and BridgeRNA evidence for the independent conserved core',pad=12);fig.text(.5,.01,'Gray = unsupported; cell text = raw method score. Directional meanings differ across methods.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.04,1,1))
    for ext in ['png','pdf']:fig.savefig(FIG/f'independent_core_three_method_heatmap.{ext}',dpi=300,bbox_inches='tight');plt.close(fig)
    prov={'bridge_inference_rerun':False,'ig_rerun':False,'gene_criteria':'mutual Top-500 same-direction expression/IG; Top-500 positive local graph cosine','pathway_criteria':'mutual Top-250 absolute score and same direction','enrichment_background':15165,'claims':'technical remeasurement concordance, not independent biological replication'};(OUT/'summary/provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
    print(metrics.pivot(index='metric',columns='cohort',values='value').reindex(columns=COHORTS).to_string());print('\n',pcounts.to_string(index=False));print('\n',json.dumps(shared,indent=2))
if __name__=='__main__':main()
