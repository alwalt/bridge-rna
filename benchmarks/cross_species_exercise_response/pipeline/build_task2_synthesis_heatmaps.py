#!/usr/bin/env python3
"""Final Pattern A/B synthesis of full-transcriptome DE and IG enrichment."""
from __future__ import annotations
import json, re, sys, textwrap
from datetime import datetime, timezone
from pathlib import Path
import matplotlib.pyplot as plt, numpy as np, pandas as pd, requests

HERE=Path(__file__).resolve().parents[1]; ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from src.fm_embed.vocab import load_canonical_genes
OUT=HERE/'results/final_synthesis'; OUT.mkdir(parents=True,exist_ok=True)
URL='https://biit.cs.ut.ee/gprofiler/api/gost/profile/'
AXES={'Pattern A':['human_GSE108643','human_GSE86931','mouse_GSE126962','mouse_GSE132520'],
      'Pattern B':['human_GSE71972','human_GSE87748','mouse_GSE97718']}
COLUMNS=['Full-transcriptome DE','Human IG','Mouse IG','Conserved human–mouse IG']
THEMES=[
 ('Muscle contraction',r'muscle system process|muscle contraction|striated muscle contraction'),
 ('Muscle structure / cytoskeleton',r'cytoskeleton in muscle|muscle structure|muscle organ development|muscle tissue development|striated muscle tissue development|myofibril|actomyosin|sarcomere'),
 ('Cardiac contraction / cardiomyopathy',r'cardiac muscle contraction|cardiomyopathy'),
 ('Vascular / circulatory development',r'vascular|vasculature|blood vessel|circulatory'),
 ('Immune / cytokine signaling',r'immune|cytokine|interferon|interleukin|inflamm|leukocyte|complement'),
 ('Lipid / fatty-acid response',r'lipid|fatty acid'),
 ('Oxidative / mitochondrial respiration',r'respirat|electron transport|complex i|oxidative phosphorylation|reactive oxygen'),
 ('Glycogen / carbohydrate metabolism',r'glycogen|carbohydrate|glucose|glycol'),
 ('Metabolic regulation',r'metabolic|metabolism|biosynthetic'),
 ('Oxygen / cellular stress response',r'oxygen-containing|hypoxi|stress|unfolded protein|heat shock'),
 ('General tissue / organ development',r'tissue development|organ development|multicellular organismal'),
]
DESCRIPTIONS={
 'Pattern A':'Human GSE108643/GSE86931: vastus lateralis cycling (immediate at 50% VO₂max/650 kcal; 4 h after 70 min at 70% VO₂max).\nMouse GSE126962/GSE132520: gastrocnemius/quadriceps treadmill responses at 3–4 h (60 min ramped and submaximal ramped protocols).',
 'Pattern B':'Human GSE71972/GSE87748: vastus lateralis after 60 min moderate unilateral knee extension (immediate) or 15 min at 80% VO₂max (4 h).\nMouse GSE97718: quadriceps 3 h after one maximal treadmill-to-exhaustion session.',
}

def post(query,background):
    payload={'organism':'hsapiens','query':query,'sources':['GO:BP','KEGG','REAC'],'user_threshold':0.05,'domain_scope':'custom','background':sorted(background),'no_evidences':False}
    response=requests.post(URL,json=payload,timeout=300); response.raise_for_status(); return response.json()

def full_de_sets():
    ortho=pd.read_csv(ROOT/'data/ensembl/orthologs_one2one.txt',sep='\t')
    ortho['Gene stable ID']=ortho['Gene stable ID'].astype(str).str.split('.').str[0]
    # Drop ambiguous mouse-to-human mappings rather than selecting one silently.
    pairs=ortho.dropna(subset=['Gene stable ID','Human gene name'])[['Gene stable ID','Human gene name']].drop_duplicates()
    counts=pairs.groupby('Gene stable ID')['Human gene name'].nunique(); pairs=pairs[pairs['Gene stable ID'].isin(counts[counts.eq(1)].index)]
    mouse_map=dict(zip(pairs['Gene stable ID'],pairs['Human gene name']))
    sets={}; backgrounds={}; ranks=[]
    for pattern,cids in AXES.items():
      studies=[]; universe=set()
      for cid in cids:
        x=pd.read_parquet(HERE/f'results/full_transcriptome_de/{cid}_full_de.parquet').query('tested').copy()
        if cid.startswith('human_'):
          x['mapped_symbol']=x.gene_symbol.where(~x.gene_symbol.str.startswith('ENSG',na=False))
        else: x['mapped_symbol']=x.gene_id.str.split('.').str[0].map(mouse_map)
        x=x.dropna(subset=['mapped_symbol']).sort_values('absolute_effect_rank').drop_duplicates('mapped_symbol')
        x['percentile']=x.absolute_effect_rank/len(x); studies.append(x.set_index('mapped_symbol').percentile); universe.update(x.mapped_symbol)
      matrix=pd.DataFrame(index=sorted(universe))
      for i,s in enumerate(studies): matrix[f'study_{i}']=s.reindex(matrix.index).fillna(1.0)
      matrix['mean_rank_percentile']=matrix.mean(axis=1); matrix=matrix.sort_values('mean_rank_percentile'); sets[pattern]=matrix.head(100).index.tolist(); backgrounds[pattern]=set(matrix.index)
      ranks.extend({'pattern':pattern,'gene':g,'rank':i+1,'mean_rank_percentile':v} for i,(g,v) in enumerate(matrix.mean_rank_percentile.items()))
    pd.DataFrame(ranks).to_parquet(OUT/'full_transcriptome_de_consensus_rankings.parquet',index=False)
    return sets,backgrounds

def ig_sets(model_genes):
    overall=pd.read_csv(HERE/'results/latent_axis_attribution/axis_consensus_attributed_genes.csv')
    species=pd.read_parquet(HERE/'results/latent_axis_attribution/species_specific_enrichment/species_specific_ig_rankings.parquet')
    sets={}
    for pattern in AXES:
      axis=pattern.replace('Pattern','Axis')
      h=species.query("axis == @axis and species == 'human' and rank <= 100").gene.tolist(); m=species.query("axis == @axis and species == 'mouse' and rank <= 100").gene.tolist()
      sets[pattern]={'Overall IG':overall.query('axis == @axis and rank <= 100').gene.tolist(),'Human IG':h,'Mouse IG':m,'Conserved human–mouse IG':sorted(set(h)&set(m))}
    return sets

def parse(raw,label):
    r=pd.DataFrame(raw.get('result',[]))
    if r.empty: return pd.DataFrame(columns=['column','source','native','name','p_value','intersection_size'])
    r['column']=label; return r[['column','source','native','name','p_value','intersection_size']]

def theme_for(name):
    for theme,pattern in THEMES:
      if re.search(pattern,name,flags=re.I): return theme
    return None

def main():
    model_genes=load_canonical_genes(ROOT/'data/ensembl/canonical_genes.csv')
    if len(model_genes)!=15165 or len(set(model_genes))!=15165: raise AssertionError('Expected exact model universe')
    de_sets,de_backgrounds=full_de_sets(); ig=ig_sets(model_genes); raw_all={}; all_results=[]; set_rows=[]
    for pattern in AXES:
      raw=post({'Full-transcriptome DE':de_sets[pattern]},de_backgrounds[pattern]); raw_all[f'{pattern} DE']=raw; all_results.append(parse(raw,'Full-transcriptome DE').assign(pattern=pattern))
      raw=post(ig[pattern],set(model_genes)); raw_all[f'{pattern} IG']=raw
      for column in COLUMNS[1:]: all_results.append(parse({'result':[x for x in raw.get('result',[]) if x.get('query')==column]},column).assign(pattern=pattern))
      set_rows += [{'pattern':pattern,'column':'Full-transcriptome DE','gene':g,'background_size':len(de_backgrounds[pattern])} for g in de_sets[pattern]]
      for column,values in ig[pattern].items(): set_rows += [{'pattern':pattern,'column':column,'gene':g,'background_size':15165} for g in values]
    results=pd.concat(all_results,ignore_index=True); results['theme']=results.name.map(theme_for); results.to_csv(OUT/'synthesis_enrichment_terms.csv',index=False); pd.DataFrame(set_rows).to_csv(OUT/'synthesis_gene_sets.csv',index=False)
    (OUT/'synthesis_enrichment_raw.json').write_text(json.dumps(raw_all,indent=2)+'\n')
    matrices=[]; detailed_matrices=[]
    for pattern in AXES:
      matrix=pd.DataFrame(np.nan,index=[x[0] for x in THEMES],columns=COLUMNS)
      z=results[(results.pattern==pattern)&results.theme.notna()]
      for (theme,column),g in z.groupby(['theme','column']): matrix.loc[theme,column]=(-np.log10(g.p_value.clip(lower=np.finfo(float).tiny))).max()
      matrix.insert(0,'theme',matrix.index); matrix.insert(0,'pattern',pattern); matrices.append(matrix.reset_index(drop=True))
      values=matrix[COLUMNS].to_numpy(float); masked=np.ma.masked_invalid(values)
      cmap=plt.colormaps['cividis'].copy(); cmap.set_bad('#e3e6e8')
      vmax=max(1.0,float(np.nanmax(values))); fig,ax=plt.subplots(figsize=(9.4,5.8)); image=ax.imshow(masked,aspect='auto',cmap=cmap,vmin=0,vmax=vmax)
      conserved_n=len(ig[pattern]['Conserved human–mouse IG'])
      short_columns=['Full-transcriptome DE\n(n=100)','Human IG\n(n=100)','Mouse IG\n(n=100)',f'Conserved H–M IG\n(n={conserved_n})']
      ax.set_xticks(np.arange(len(COLUMNS)),short_columns,rotation=22,ha='right'); ax.set_yticks(np.arange(len(matrix)),matrix.theme)
      ax.set_xticks(np.arange(-.5,len(COLUMNS),1),minor=True); ax.set_yticks(np.arange(-.5,len(matrix),1),minor=True)
      ax.grid(which='minor',color='white',linewidth=1.4); ax.tick_params(which='minor',bottom=False,left=False)
      for y in range(len(matrix)):
        for x in range(len(COLUMNS)):
          if np.isfinite(values[y,x]): ax.text(x,y,f'{values[y,x]:.1f}',ha='center',va='center',fontsize=9,fontweight='semibold',color='white' if values[y,x] < .48*vmax else '#111111')
      fig.suptitle(f'{pattern}: DE and BridgeRNA IG biological synthesis',x=.31,y=.97,ha='left',fontweight='bold',fontsize=13)
      subtitle='\n'.join(textwrap.fill(line,105) for line in DESCRIPTIONS[pattern].splitlines())
      fig.text(.31,.915,subtitle,ha='left',va='top',fontsize=8.5,color='#333333',linespacing=1.35)
      colorbar=fig.colorbar(image,ax=ax,fraction=.035,pad=.025); colorbar.set_label('−log10 adjusted p-value',fontsize=9)
      fig.subplots_adjust(left=.31,right=.90,bottom=.22,top=.82); stem=pattern.lower().replace(' ','_')+'_synthesis_heatmap'; fig.savefig(OUT/f'{stem}.png',dpi=400,bbox_inches='tight',facecolor='white'); fig.savefig(OUT/f'{stem}.pdf',bbox_inches='tight',facecolor='white'); plt.close(fig)
      # Companion plot retains actual ontology/database term names. To remain
      # legible, use the union of the top three corrected terms per source and
      # analysis column; values for all selected terms/columns are retained.
      selected=(z.sort_values('p_value').groupby(['column','source'],group_keys=False).head(3)[['source','native','name']].drop_duplicates())
      selected['term']=selected.source+' | '+selected.name
      detail=pd.DataFrame(np.nan,index=selected.term,columns=COLUMNS)
      for _,r in z.iterrows():
        term=f'{r.source} | {r["name"]}'
        if term in detail.index: detail.loc[term,r.column]=max(-np.log10(max(r.p_value,np.finfo(float).tiny)),detail.loc[term,r.column] if np.isfinite(detail.loc[term,r.column]) else 0)
      source_order={'GO:BP':0,'KEGG':1,'REAC':2}; order=sorted(detail.index,key=lambda term:(source_order.get(term.split(' | ',1)[0],9),-np.nanmax(detail.loc[term].values)))
      detail=detail.loc[order]; export=detail.copy(); export.insert(0,'term',export.index); export.insert(0,'source',[x.split(' | ',1)[0] for x in export.index]); export.insert(0,'pattern',pattern); detailed_matrices.append(export.reset_index(drop=True))
      dvalues=detail.to_numpy(float); dmasked=np.ma.masked_invalid(dvalues); dvmax=max(1.0,float(np.nanmax(dvalues))); dcmap=plt.colormaps['cividis'].copy(); dcmap.set_bad('#e3e6e8')
      dfig,dax=plt.subplots(figsize=(10.5,max(7.2,.34*len(detail)))); dimage=dax.imshow(dmasked,aspect='auto',cmap=dcmap,vmin=0,vmax=dvmax)
      dax.set_xticks(np.arange(len(COLUMNS)),short_columns,rotation=22,ha='right'); dax.set_yticks(np.arange(len(detail)),detail.index,fontsize=8)
      dax.set_xticks(np.arange(-.5,len(COLUMNS),1),minor=True); dax.set_yticks(np.arange(-.5,len(detail),1),minor=True); dax.grid(which='minor',color='white',linewidth=1.2); dax.tick_params(which='minor',bottom=False,left=False)
      for yy in range(len(detail)):
        for xx in range(len(COLUMNS)):
          if np.isfinite(dvalues[yy,xx]): dax.text(xx,yy,f'{dvalues[yy,xx]:.1f}',ha='center',va='center',fontsize=7.5,fontweight='semibold',color='white' if dvalues[yy,xx] < .48*dvmax else '#111111')
      dfig.suptitle(f'{pattern}: significant GO BP, KEGG, and Reactome terms',x=.39,y=.98,ha='left',fontweight='bold',fontsize=13)
      dfig.text(.39,.94,subtitle,ha='left',va='top',fontsize=8.3,color='#333333',linespacing=1.35)
      dbar=dfig.colorbar(dimage,ax=dax,fraction=.03,pad=.025); dbar.set_label('−log10 adjusted p-value',fontsize=9)
      dfig.subplots_adjust(left=.39,right=.91,bottom=.17,top=.84); dstem=pattern.lower().replace(' ','_')+'_detailed_terms_heatmap'; dfig.savefig(OUT/f'{dstem}.png',dpi=400,bbox_inches='tight',facecolor='white'); dfig.savefig(OUT/f'{dstem}.pdf',bbox_inches='tight',facecolor='white'); plt.close(dfig)
    pd.concat(matrices).to_csv(OUT/'harmonized_theme_matrix.csv',index=False)
    pd.concat(detailed_matrices).to_csv(OUT/'detailed_term_matrix.csv',index=False)
    provenance={'retrieved_utc':datetime.now(timezone.utc).isoformat(),'service':'g:Profiler g:GOSt','sources':['GO:BP','KEGG','REAC'],'multiple_testing':'g:SCS','significance_threshold':0.05,'DE':{'gene_set':'axis-consensus full-transcriptome DE Top 100','background':'union of all edgeR-tested genes in constituent axis studies with an unambiguous human-symbol mapping','background_sizes':{k:len(v) for k,v in de_backgrounds.items()}},'IG':{'background':'exact 15,165 BridgeRNA genes','sets':'existing overall and species consensus Top 100; conserved is human/mouse Top-100 intersection'},'theme_rules':[{'theme':a,'regex':b} for a,b in THEMES],'blank_cells':'no significant corrected source term mapped to theme','study_descriptions':DESCRIPTIONS}
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n'); print(pd.concat(matrices).to_string(index=False)); print('DE backgrounds',provenance['DE']['background_sizes'])
if __name__=='__main__': main()
