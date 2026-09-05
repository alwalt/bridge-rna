#!/usr/bin/env python3
"""Gene-level IG diagnosis of RR1 instability versus RR3 reproducibility."""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
from scipy.stats import hypergeom, spearmanr

ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[1]
OUT=ROOT/'results/task4_gene_attribution_diagnostic';OUT.mkdir(parents=True,exist_ok=True)
WORK=ROOT/'work/task4_gene_attribution_diagnostic';WORK.mkdir(parents=True,exist_ok=True)
T3=REPO/'benchmarks/osdr_batch_effect_representation';R3=T3/'results';W3=T3/'work'
sys.path.insert(0,str(REPO/'benchmarks/tcga_downstream/pipeline'));sys.path.insert(0,str(REPO))
from run_attention_pooling import load_frozen_encoder
from src.fm_embed.vocab import load_canonical_genes

GENES=load_canonical_genes(REPO/'data/ensembl/canonical_genes.csv'); assert len(GENES)==15165 and len(set(GENES))==15165
GPROFILER='https://biit.cs.ut.ee/gprofiler/api/gost/profile/'
PAIR_SPECS={
 'RR1':('RR1_OSD48_original_matched','RR1_OSD168_no-ERCC'),
 'RR3-39':('C01_OSD137_original_matched','C01_OSD168_all_ERCC'),
 'RR3-40':('C02_OSD137_original_matched','C02_OSD168_all_ERCC')}

def unit(x):return x/max(np.linalg.norm(x),1e-12)
def cosine(a,b):return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
def condition(s):return 'FLT' if '_FLT_' in s else 'GC' if '_GC_' in s else 'other'
def score(model,x,direction):return (model._encode_hidden(x).mean(1)*direction).sum(1)
def ig(model,values,direction,device,steps=16,path_batch=4):
 baseline=torch.zeros((1,len(values)),device=device); observed=torch.from_numpy(np.array(values,dtype=np.float32,copy=True)).to(device)[None];target=torch.from_numpy(direction.astype(np.float32)).to(device)[None];total=torch.zeros_like(baseline)
 alphas=(np.arange(steps,dtype=np.float32)+.5)/steps
 for start in range(0,steps,path_batch):
  a=torch.from_numpy(alphas[start:start+path_batch]).to(device)[:,None];x=(baseline+a*(observed-baseline)).requires_grad_(True);v=score(model,x,target.expand(len(x),-1));total+=torch.autograd.grad(v.sum(),x)[0].detach().sum(0,keepdim=True)
 attr=((observed-baseline)*total/steps)[0]
 with torch.no_grad():delta=score(model,observed,target)[0]-score(model,baseline,target)[0]
 return attr.cpu().numpy(),float(delta),float(attr.sum().cpu())

def design():
 m=pd.read_csv(R3/'sample_manifest.csv');idx=dict(zip(m.sample_id,range(len(m))));td=pd.read_csv(R3/'task3_osd168_technical_replication/technical_response_design.csv');z=np.load(W3/'bridgerna_embeddings.npy');x=np.load(W3/'bridgerna_log1p_tpm_inputs.npy',mmap_mode='r')
 specs={}
 wanted=set(sum(([a,b] for a,b in PAIR_SPECS.values()),[]))
 for row in td.itertuples():
  if row.representation in wanted:
   ids=str(row.samples).split(' | ');specs[row.representation]={'ids':ids,'indices':[idx[s] for s in ids],'conditions':[condition(s) for s in ids]}
 for label,(a,b) in PAIR_SPECS.items():
  va=np.load(R3/'task3_osd168_technical_replication/technical_response_vectors.npz',allow_pickle=True);vectors=dict(zip(va['names'],va['delta_z']))
  specs[a]['direction']=unit(vectors[a]);specs[b]['direction']=unit(vectors[a]);specs[a]['pair']=label;specs[b]['pair']=label
 return specs,x,z

def worker(names,device_name,steps,path_batch,tag):
 specs,x,_=design();device=torch.device(device_name if torch.cuda.is_available() else 'cpu');model=load_frozen_encoder(device);attrs=[];complete=[];started=time.monotonic()
 for n,name in enumerate(names,1):
  s=specs[name];roles={}
  for cond in ['FLT','GC']:
   ids=[i for i,c in zip(s['indices'],s['conditions']) if c==cond];profile=np.asarray(x[ids]).mean(0).astype(np.float32);a,delta,total=ig(model,profile,s['direction'],device,steps,path_batch);roles[cond]=a;complete.append({'response':name,'condition':cond,'samples':len(ids),'endpoint_delta':delta,'attribution_sum':total,'completeness_error':total-delta})
  attrs.append(roles['FLT']-roles['GC']);elapsed=time.monotonic()-started;print(f'[heartbeat] {tag} {n}/{len(names)} {name} elapsed={elapsed/60:.1f}m',flush=True)
 np.savez_compressed(WORK/f'{tag}.npz',names=np.array(names,dtype=object),attributions=np.stack(attrs));pd.DataFrame(complete).to_csv(WORK/f'{tag}_completeness.csv',index=False)

def controlled_signature(device_name,steps,path_batch):
 p=ROOT/'work/datasets/chen_2020_tcells';m=pd.read_parquet(p/'manifest.parquet').reset_index(drop=True);x=np.load(p/'log1p_tpm.npy',mmap_mode='r');z=np.load(p/'bridgerna_embeddings.npy');D=[]
 for _,g in m.groupby('pair_id',sort=True):D.append(z[g.index[g.library_prep.eq('ribo')]].mean(0)-z[g.index[g.library_prep.eq('polyA')]].mean(0))
 direction=unit(np.mean(D,axis=0)).astype(np.float32);device=torch.device(device_name if torch.cuda.is_available() else 'cpu');model=load_frozen_encoder(device);roles={};rows=[]
 for prep in ['ribo','polyA']:
  profile=np.asarray(x[m.index[m.library_prep.eq(prep)]]).mean(0).astype(np.float32);a,d,total=ig(model,profile,direction,device,steps,path_batch);roles[prep]=a;rows.append({'response':'controlled_tcell_ribo_minus_polyA','condition':prep,'samples':sum(m.library_prep.eq(prep)),'endpoint_delta':d,'attribution_sum':total,'completeness_error':total-d})
 np.savez_compressed(WORK/'controlled_signature.npz',attribution=roles['ribo']-roles['polyA'],direction=direction);pd.DataFrame(rows).to_csv(WORK/'controlled_completeness.csv',index=False)

def overlap_stats(a,b,n=100):
 A=set(np.argsort(-np.abs(a))[:n]);B=set(np.argsort(-np.abs(b))[:n]);shared=A&B;expected=n*n/len(a);p=hypergeom.sf(len(shared)-1,len(a),n,n)
 same=sum(np.sign(a[list(shared)])==np.sign(b[list(shared)])) if shared else 0
 return {'top_n':n,'shared':len(shared),'jaccard':len(shared)/len(A|B),'expected_random_overlap':expected,'overlap_enrichment':len(shared)/expected,'hypergeom_p':p,'same_sign_fraction':same/len(shared) if shared else np.nan,'opposite_sign_fraction':1-same/len(shared) if shared else np.nan}

def summarize(attrs,controlled):
 rows=[];rankrows=[]
 for pair,(oa,ob) in PAIR_SPECS.items():
  a,b=attrs[oa],attrs[ob];union=set(np.argsort(-np.abs(a))[:100])|set(np.argsort(-np.abs(b))[:100]);r=overlap_stats(a,b,100);r.update({'comparison':pair,'full_signed_spearman':spearmanr(a,b).statistic,'top100_union_signed_spearman':spearmanr(a[list(union)],b[list(union)]).statistic});rows.append(r)
  discrepancy=b-a
  for n in [100,250,500]:
   q=overlap_stats(controlled,discrepancy,n);q.update({'comparison':pair,'signature':'controlled_vs_remeasurement_minus_original'});rankrows.append(q)
 return pd.DataFrame(rows),pd.DataFrame(rankrows)

def expression_control():
 specs,x,_=design();vectors={}
 for name,s in specs.items():
  f=[i for i,c in zip(s['indices'],s['conditions']) if c=='FLT'];g=[i for i,c in zip(s['indices'],s['conditions']) if c=='GC'];vectors[name]=np.asarray(x[f]).mean(0)-np.asarray(x[g]).mean(0)
 rows=[]
 for pair,(a,b) in PAIR_SPECS.items():
  q=overlap_stats(vectors[a],vectors[b]);q.update({'comparison':pair,'full_gene_cosine':cosine(vectors[a],vectors[b]),'full_gene_spearman':spearmanr(vectors[a],vectors[b]).statistic});rows.append(q)
 return pd.DataFrame(rows),vectors

def gene_sets(attrs,controlled):
 sets={};top={k:set(np.argsort(-np.abs(v))[:100]) for k,v in attrs.items()}
 a,b=[attrs[x] for x in PAIR_SPECS['RR1']];shared=top[PAIR_SPECS['RR1'][0]]&top[PAIR_SPECS['RR1'][1]]
 sets['RR1_shared_same_sign']=[GENES[i] for i in shared if np.sign(a[i])==np.sign(b[i])];sets['RR1_shared_opposite_sign']=[GENES[i] for i in shared if np.sign(a[i])!=np.sign(b[i])]
 sets['RR1_original_specific']=[GENES[i] for i in top[PAIR_SPECS['RR1'][0]]-top[PAIR_SPECS['RR1'][1]]];sets['RR1_remeasurement_specific']=[GENES[i] for i in top[PAIR_SPECS['RR1'][1]]-top[PAIR_SPECS['RR1'][0]]]
 rr3=set()
 for pair in ['RR3-39','RR3-40']:
  x,y=PAIR_SPECS[pair];rr3|={i for i in top[x]&top[y] if np.sign(attrs[x][i])==np.sign(attrs[y][i])}
 sets['RR3_reproducible']= [GENES[i] for i in rr3];ct=set(np.argsort(-np.abs(controlled))[:100]);sets['controlled_polyA_ribo_top100']=[GENES[i] for i in ct]
 disc=attrs[PAIR_SPECS['RR1'][1]]-attrs[PAIR_SPECS['RR1'][0]];rd=set(np.argsort(-np.abs(disc))[:100]);sets['controlled_and_RR1_discrepancy']=[GENES[i] for i in ct&rd]
 return sets

def mouse_symbols():
 table=pd.read_csv(REPO/'data/ensembl/orthologs_one2one.txt',sep='\t',low_memory=False);table['human']=table['Human gene name'].astype(str).str.upper();table['mouse']=table['Gene name'].astype(str);q=table[['human','mouse']].dropna().drop_duplicates();q=q.groupby('human').filter(lambda x:x.mouse.nunique()==1).drop_duplicates('human');mapping=q.set_index('human').mouse.to_dict();return [mapping.get(g,'') for g in GENES]

def save_rankings(attrs,controlled):
 mouse=mouse_symbols();rows=[]
 for name,values in {'controlled_tcell_ribo_minus_polyA':controlled,**attrs}.items():
  order=np.argsort(-np.abs(values));rank=np.empty(len(values),int);rank[order]=np.arange(1,len(values)+1)
  rows.append(pd.DataFrame({'response':name,'gene_symbol_human':GENES,'gene_symbol_mouse':mouse,'signed_attribution':values,'absolute_attribution':np.abs(values),'rank_absolute':rank,'sign':np.sign(values).astype(int)}))
 out=pd.concat(rows,ignore_index=True).sort_values(['response','rank_absolute']);out.to_parquet(OUT/'per_response_gene_rankings.parquet',index=False);out.to_csv(OUT/'per_response_gene_rankings.csv.gz',index=False,compression='gzip')
 discrepancies=[]
 for pair,(a,b) in PAIR_SPECS.items():
  values=attrs[b]-attrs[a];order=np.argsort(-np.abs(values));rank=np.empty(len(values),int);rank[order]=np.arange(1,len(values)+1);discrepancies.append(pd.DataFrame({'comparison':pair,'gene_symbol_human':GENES,'gene_symbol_mouse':mouse,'remeasurement_minus_original_attribution':values,'absolute_discrepancy':np.abs(values),'rank_absolute_discrepancy':rank,'sign':np.sign(values).astype(int)}))
 pd.concat(discrepancies,ignore_index=True).sort_values(['comparison','rank_absolute_discrepancy']).to_parquet(OUT/'technical_discrepancy_gene_rankings.parquet',index=False)

def verify_latent_responses():
 specs,_,z=design();vectors={}
 for name,s in specs.items():
  f=[i for i,c in zip(s['indices'],s['conditions']) if c=='FLT'];g=[i for i,c in zip(s['indices'],s['conditions']) if c=='GC'];vectors[name]=z[f].mean(0)-z[g].mean(0)
 saved=pd.read_csv(R3/'task3_osd168_technical_replication/original_vs_osd168_response_similarity.csv');name_map={'RR1':'RR1_OSD168_no-ERCC','RR3-39':'C01_OSD168_all_ERCC','RR3-40':'C02_OSD168_all_ERCC'};rows=[]
 for pair,(a,b) in PAIR_SPECS.items():
  value=cosine(vectors[a],vectors[b]);expected=saved.loc[saved.OSD168_representation.eq(name_map[pair]),'cosine'].iloc[0];rows.append({'comparison':pair,'original_response':a,'remeasurement_response':b,'original_norm':np.linalg.norm(vectors[a]),'remeasurement_norm':np.linalg.norm(vectors[b]),'recomputed_cosine':value,'saved_cosine':expected,'absolute_difference':abs(value-expected),'verified':abs(value-expected)<1e-6})
 out=pd.DataFrame(rows);assert out.verified.all();out.to_csv(OUT/'latent_response_verification.csv',index=False)

def enrich(sets):
 queries={k:v for k,v in sets.items() if v};payload={'organism':'hsapiens','query':queries,'sources':['GO:BP','KEGG','REAC'],'user_threshold':.05,'domain_scope':'custom','background':GENES,'no_evidences':False};resp=requests.post(GPROFILER,json=payload,timeout=600);resp.raise_for_status();raw=resp.json();(OUT/'enrichment_raw.json').write_text(json.dumps(raw,indent=2));d=pd.DataFrame(raw.get('result',[]));d.to_csv(OUT/'enrichment.csv',index=False);return d

def figures(attrs,controlled,summary,overlap):
 fig,axes=plt.subplots(1,3,figsize=(14,4.3),layout='constrained')
 for ax,(pair,(a,b)) in zip(axes,PAIR_SPECS.items()):
  x,y=attrs[a],attrs[b];top=np.union1d(np.argsort(-np.abs(x))[:100],np.argsort(-np.abs(y))[:100]);ax.scatter(x,y,s=3,alpha=.15,color='#999999');ax.scatter(x[top],y[top],s=12,alpha=.7,color='#e15759' if pair=='RR1' else '#4e79a7');ax.axhline(0,color='black',lw=.5);ax.axvline(0,color='black',lw=.5);ax.set(title=pair,xlabel='Original signed IG',ylabel='Remeasurement signed IG')
 for ext in ['png','pdf']:fig.savefig(OUT/f'figure_a_signed_attribution_scatter.{ext}',dpi=400,bbox_inches='tight')
 plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');s=summary.set_index('comparison');axes[0].bar(s.index,s.shared,color=['#e15759','#4e79a7','#59a14f']);axes[0].set(ylabel='Shared Top-100 genes',title='Attribution overlap');axes[1].bar(s.index,s.same_sign_fraction,color=['#e15759','#4e79a7','#59a14f']);axes[1].set(ylim=(0,1),ylabel='Same-sign fraction among shared',title='Sign preservation')
 for ext in ['png','pdf']:fig.savefig(OUT/f'figure_b_top100_overlap_sign.{ext}',dpi=400,bbox_inches='tight')
 plt.close(fig)
 p=overlap.pivot(index='comparison',columns='top_n',values='overlap_enrichment').loc[list(PAIR_SPECS)];fig,ax=plt.subplots(figsize=(7,4),layout='constrained');p.plot.bar(ax=ax);ax.set(ylabel='Overlap / random expectation',title='Controlled signature vs technical discrepancy');ax.legend(title='Top N');plt.xticks(rotation=0)
 for ext in ['png','pdf']:fig.savefig(OUT/f'figure_c_controlled_signature_overlap.{ext}',dpi=400,bbox_inches='tight')
 plt.close(fig)
 names=['controlled']+sum(([a,b] for a,b in PAIR_SPECS.values()),[]);vectors={'controlled':controlled,**attrs};idx=set()
 for v in vectors.values():idx.update(np.argsort(-np.abs(v))[:25])
 idx=sorted(idx,key=lambda i:-max(abs(v[i]) for v in vectors.values()))[:100];M=np.stack([vectors[n][idx] for n in names]);scale=np.max(np.abs(M),axis=0);Z=M/np.maximum(scale,1e-12)
 fig,ax=plt.subplots(figsize=(15,5),layout='constrained');im=ax.imshow(Z,aspect='auto',cmap='RdBu_r',vmin=-1,vmax=1);ax.set(yticks=range(len(names)),yticklabels=names,xticks=range(len(idx)),xticklabels=[GENES[i] for i in idx],title='Signed IG attribution (gene-scaled)');ax.tick_params(axis='x',rotation=90,labelsize=5);fig.colorbar(im,ax=ax,label='Signed attribution / gene max |attribution|')
 for ext in ['png','pdf']:fig.savefig(OUT/f'figure_d_signed_gene_heatmap.{ext}',dpi=400,bbox_inches='tight')
 plt.close(fig)

def orchestrate(args):
 names=list(sum(([a,b] for a,b in PAIR_SPECS.values()),[]));chunks=[names[::2],names[1::2]];procs=[]
 if not args.reuse_attributions:
  for i,(chunk,device) in enumerate(zip(chunks,args.devices)):
   cmd=[sys.executable,__file__,'--worker','--names',*chunk,'--device',device,'--tag',f'worker_{i}','--ig-steps',str(args.ig_steps),'--path-batch',str(args.path_batch)];procs.append(subprocess.Popen(cmd));print('[start]',' '.join(cmd),flush=True)
  ctrl=subprocess.Popen([sys.executable,__file__,'--controlled-worker','--device',args.devices[0],'--ig-steps',str(args.ig_steps),'--path-batch',str(args.path_batch)])
  for p in procs+[ctrl]:
   if p.wait():raise RuntimeError(f'worker failed: {p.args}')
 attrs={}
 for i in range(2):
  q=np.load(WORK/f'worker_{i}.npz',allow_pickle=True);attrs.update(dict(zip(q['names'],q['attributions'])))
 controlled=np.load(WORK/'controlled_signature.npz')['attribution'];summary,overlap=summarize(attrs,controlled);expr,exprv=expression_control();sets=gene_sets(attrs,controlled);verify_latent_responses();save_rankings(attrs,controlled)
 summary.to_csv(OUT/'technical_replication_ig_comparison.csv',index=False);overlap.to_csv(OUT/'controlled_signature_discrepancy_overlap.csv',index=False);expr.to_csv(OUT/'expression_response_comparison.csv',index=False)
 pd.DataFrame([(k,g) for k,v in sets.items() for g in v],columns=['gene_set','gene_symbol']).to_csv(OUT/'interpretation_gene_sets.csv',index=False)
 np.savez_compressed(OUT/'signed_gene_attributions.npz',gene_symbol=np.array(GENES),controlled=controlled,**attrs);np.savez_compressed(OUT/'expression_response_vectors.npz',gene_symbol=np.array(GENES),**exprv)
 try:enr=enrich(sets);enrich_status=f'ok:{len(enr)} rows'
 except Exception as e:enrich_status=f'failed:{e}';print('[warning]',enrich_status,flush=True)
 figures(attrs,controlled,summary,overlap)
 prov={'created_utc':datetime.now(timezone.utc).isoformat(),'bridge_frozen':True,'models_retrained':False,'contrast_definitions_changed':False,'target_axis':'original technical-pair response direction, fixed for original and remeasurement','controlled_signature':'IG(mean T-cell Ribo)-IG(mean T-cell PolyA) onto independently defined mean embedding displacement','ig_baseline':'all-zero log1p(TPM)','ig_steps':args.ig_steps,'enrichment_background_genes':15165,'enrichment_status':enrich_status};(OUT/'provenance.json').write_text(json.dumps(prov,indent=2));print(summary.to_string(index=False));print('\nExpression\n',expr.to_string(index=False))

def parse():
 p=argparse.ArgumentParser();p.add_argument('--devices',nargs=2,default=['cuda:0','cuda:1']);p.add_argument('--worker',action='store_true');p.add_argument('--controlled-worker',action='store_true');p.add_argument('--reuse-attributions',action='store_true');p.add_argument('--names',nargs='*');p.add_argument('--device',default='cuda:0');p.add_argument('--tag',default='worker');p.add_argument('--ig-steps',type=int,default=16);p.add_argument('--path-batch',type=int,default=4);return p.parse_args()
if __name__=='__main__':
 a=parse()
 if a.worker:worker(a.names,a.device,a.ig_steps,a.path_batch,a.tag)
 elif a.controlled_worker:controlled_signature(a.device,a.ig_steps,a.path_batch)
 else:orchestrate(a)
