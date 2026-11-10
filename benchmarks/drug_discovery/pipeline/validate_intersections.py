#!/usr/bin/env python3
"""Small cross-study recurrence and pathway benchmark from frozen gene rankings.
No encoder inference, DE refitting, or drug ranking. Preserve all prior analyses.
"""
from pathlib import Path
import hashlib,json,time,urllib.request,argparse
import numpy as np
import pandas as pd
from scipy.stats import hypergeom,false_discovery_control
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[1]
RESEARCH=REPO/'research/drug_discovery'
OUT=ROOT/'results/intersection_validation';WORK=ROOT/'work/intersection_validation'
ORDER={'Muscle atrophy':['GSE211204','GSE113165','GSE234465'],'Bone loss':['GSE189524','GSE276529','GSE273868'],'Radiation':['GSE297090','GSE184119','GSE297560']}
LIBRARIES=['GO_Biological_Process_2025','Reactome_Pathways_2024']
SEED=20261109;REPLICATES=1000

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def log(s):print(s,flush=True)

def main():
    global OUT,WORK,LIBRARIES
    parser=argparse.ArgumentParser();parser.add_argument('--include-kegg',action='store_true');args=parser.parse_args()
    if args.include_kegg:
        LIBRARIES=['GO_Biological_Process_2025','KEGG_2021_Human','Reactome_Pathways_2024']
        OUT=ROOT/'results/intersection_validation_three_libraries'
        WORK=ROOT/'work/intersection_validation_three_libraries'
    start=time.monotonic();OUT.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True)
    log('Estimate: 1–3 minutes including small cached annotation downloads; cached-only gene analysis, 1000 random draws per condition.')
    paths=[RESEARCH/'results/expanded_chembl_sensitivity/gene_partitions.csv.gz',RESEARCH/'results/manifests/contrast_members.csv']
    hashes={str(p.relative_to(REPO)):sha(p) for p in paths}
    parts=pd.read_csv(paths[0]);contrasts=pd.read_csv(paths[1])[['dataset','contrast_id']].drop_duplicates()
    canonical_order=pd.read_csv(REPO/'data/ensembl/canonical_genes.csv').gene_symbol.tolist()
    canonical=set(canonical_order)
    eligible={};modules={};frozen=[];universe_rows=[]
    for condition,datasets in ORDER.items():
        for dataset in datasets:
            universe=set()
            for cid in contrasts.loc[contrasts.dataset.eq(dataset),'contrast_id']:
                source=RESEARCH/'results/differential_expression'/f'{cid}.csv.gz'
                de=pd.read_csv(source);hashes[str(source.relative_to(REPO))]=sha(source)
                assert set(de.eligible.dropna().unique())<=set([True,False])
                universe.update(de.loc[de.eligible.eq(True),'gene'])
            prepared=ROOT/'research_archive/drug_discovery/work/prepared'/f'{dataset}_log1p_tpm.npy'
            values=np.load(prepared,mmap_mode='r')
            assert values.shape[1]==len(canonical_order) and np.isfinite(values).all()
            eligible[dataset]={g for g,total in zip(canonical_order,np.asarray(values).sum(axis=0)) if total>0}
            hashes[str(prepared.relative_to(REPO))]=sha(prepared)
            d=parts[parts.condition.eq(condition)&parts.dataset.eq(dataset)]
            for method,categories,rank_column in [('BRIDGE',['Bridge-only','Shared'],'bridge_rank'),('DE',['DE-only','Shared'],'de_rank')]:
                q=d[d.partition.isin(categories)].copy();genes=set(q.gene)
                assert len(q)==500 and len(genes)==500
                assert genes<=eligible[dataset],(condition,dataset,method,sorted(genes-eligible[dataset])[:5])
                modules[(condition,dataset,method)]=genes
                for row in q.itertuples():frozen.append({'condition':condition,'dataset':dataset,'method':method,'gene':row.gene,'rank':getattr(row,rank_column)})
        common=set.intersection(*(eligible[d] for d in datasets))
        assert len(common)>500
        universe_rows.extend({'condition':condition,'gene':g} for g in sorted(common))
    pd.DataFrame(frozen).to_csv(OUT/'frozen_top500_lists.csv',index=False)
    pd.DataFrame(universe_rows).to_csv(OUT/'common_eligible_genes.csv',index=False)
    # Freeze selected sets/backgrounds before querying pathway annotations.
    queries=[];intersection_rows=[];summary=[];null_rows=[];rng=np.random.default_rng(SEED)
    for condition,datasets in ORDER.items():
        common=set.intersection(*(eligible[d] for d in datasets));null=[]
        arrays=[np.array(sorted(eligible[d])) for d in datasets]
        for repeat in range(REPLICATES):
            draws=[set(rng.choice(a,size=500,replace=False)) for a in arrays]
            count=len(set.intersection(*draws));null.append(count);null_rows.append({'condition':condition,'replicate':repeat,'intersection_genes':count})
        for method in ['BRIDGE','DE']:
            sets=[modules[(condition,d,method)] for d in datasets];intersection=set.intersection(*sets)
            for gene in sorted(intersection):intersection_rows.append({'condition':condition,'method':method,'gene':gene})
            summary.append({'condition':condition,'method':method,'top_genes_per_study':500,'studies':3,'common_eligible_genes':len(common),'all_three_genes':len(intersection),'at_least_two_genes':sum(sum(g in s for s in sets)>=2 for g in set.union(*sets)),'random_mean_intersection':float(np.mean(null)),'random_95th_percentile':float(np.quantile(null,.95)),'random_max_intersection':max(null),'random_empirical_p':(1+sum(k>=len(intersection) for k in null))/(REPLICATES+1),'null_draws':REPLICATES})
            queries.append((condition,method,'all_three','all_three',intersection,common))
            for dataset,genes in zip(datasets,sets):queries.append((condition,method,'study_top500',dataset,genes,eligible[dataset]))
        log(f'{condition}: shared BRIDGE={len(set.intersection(*(modules[(condition,d,"BRIDGE")] for d in datasets)))}; DE={len(set.intersection(*(modules[(condition,d,"DE")] for d in datasets)))}; eligibility checked.')
    pd.DataFrame(summary).to_csv(OUT/'recurrence_summary.csv',index=False)
    pd.DataFrame(intersection_rows).to_csv(OUT/'intersection_genes.csv',index=False)
    pd.DataFrame(null_rows).to_csv(OUT/'random_intersection_counts.csv',index=False)
    annotations={};annotation_meta={}
    for library in LIBRARIES:
        url='https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName='+library
        path=ROOT/'work/intersection_validation'/f'{library}.gmt'
        if not path.exists():
            with urllib.request.urlopen(url,timeout=45) as response:raw=response.read()
            path.write_bytes(raw)
        terms={}
        for line in path.read_text().splitlines():
            fields=line.split('\t')
            if len(fields)>2:terms[fields[0]]=set(g for g in fields[2:] if g)
        assert len(terms)>100
        annotations[library]=terms;annotation_meta[library]={'url':url,'sha256':sha(path),'terms':len(terms)}
        log(f'Annotations ready: {library}, {len(terms)} terms.')
    full=[];query_meta=[]
    for number,(condition,method,scope,dataset,genes,universe) in enumerate(queries,1):
        rows=[]
        for library,terms in annotations.items():
            annotated=set.union(*terms.values());bg=universe&annotated;query=genes&bg
            candidates=[(name,ts&bg) for name,ts in terms.items()]
            # Preset term size window avoids tiny sets and very broad ancestors.
            candidates=[(name,ts) for name,ts in candidates if 10<=len(ts)<=500]
            query_meta.append({'condition':condition,'method':method,'scope':scope,'dataset':dataset,'library':library,'selected_genes':len(genes),'eligible_background':len(universe),'annotated_background':len(bg),'annotated_query_genes':len(query),'unannotated_query_genes':len(genes-query),'tested_terms':len(candidates)})
            for name,ts in candidates:
                k=len(query&ts);n=len(query);K=len(ts);M=len(bg)
                p=float(hypergeom.sf(k-1,M,K,n)) if n else 1.
                rows.append({'condition':condition,'method':method,'scope':scope,'dataset':dataset,'library':library,'term':name,'query_genes':n,'background_genes':M,'term_genes':K,'overlap':k,'fold_enrichment':k/n/(K/M) if n else np.nan,'pvalue':p,'overlap_genes':';'.join(sorted(query&ts))})
        frame=pd.DataFrame(rows);frame['fdr']=false_discovery_control(frame.pvalue.to_numpy(),method='bh')
        full.append(frame)
        log(f'heartbeat {number}/{len(queries)} queries; {condition}/{method}/{scope}; elapsed={time.monotonic()-start:.1f}s')
    results=pd.concat(full,ignore_index=True)
    results.to_csv(WORK/'all_enrichment_tests.csv.gz',index=False,compression='gzip')
    hits=results[(results.fdr<.05)&(results.overlap>=3)].copy()
    hits.sort_values(['condition','scope','method','fdr','library','term']).to_csv(OUT/'significant_pathways.csv',index=False)
    pd.DataFrame(query_meta).to_csv(OUT/'annotation_coverage.csv',index=False)
    # A fair equal-size comparison: pathways significant in all three 500-gene sets.
    recurrence=[]
    for (condition,method,library,term),group in hits[hits.scope.eq('study_top500')].groupby(['condition','method','library','term']):
        recurrence.append({'condition':condition,'method':method,'library':library,'term':term,'studies':group.dataset.nunique(),'datasets':';'.join(sorted(group.dataset.unique())),'max_fdr':group.fdr.max(),'min_fold_enrichment':group.fold_enrichment.min()})
    pd.DataFrame(recurrence).to_csv(OUT/'pathway_recurrence.csv',index=False)
    protected={str(p.relative_to(REPO)):sha(p) for p in paths};assert all(hashes[k]==v for k,v in protected.items())
    prov={'protocol':'Frozen top500 per method per study; three-way intersections; uniform per-study-eligibility random size control; overrepresentation of intersections and equal-sized individual lists.','seed':SEED,'random_draws':REPLICATES,'null_limit':'Uniform eligibility-matched draws, not expression/detectability/rank-propensity matched; tests random selection, not BRIDGE vs DE superiority. Random intersections are not used to claim biological coherence.','annotations':annotation_meta,'libraries':LIBRARIES,'term_size_in_background':[10,500],'fdr':'BH jointly across the selected libraries per query; significant display requires q<0.05 and >=3 overlap genes. No preselection of stress-related terms.','background':'Original frozen selection universe: canonical genes with positive summed prepared log1p(TPM) within each study. Intersection across all three studies for three-way ORA; library-annotated subset used per library. The DE fitting filter is stricter and is not substituted for the original shared selection universe.','source_sha256':hashes,'elapsed_seconds':time.monotonic()-start,'limitations':['Exploratory annotation-based coherence, not causal or therapeutic validation.','Small DE intersections lack enrichment power. Different significance outcomes are not a direct method-difference test.','Studies differ in tissues and exposures; some terrestrial cohorts had pretraining exposure.','Pathways overlap and are not independent biological endpoints.']}
    (OUT/'provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
    log(f'Completed in {time.monotonic()-start:.1f}s; {len(hits)} significant query–term rows.')
if __name__=='__main__':main()
