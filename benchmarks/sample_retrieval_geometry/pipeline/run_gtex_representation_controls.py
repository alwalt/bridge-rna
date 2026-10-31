#!/usr/bin/env python3
"""Frozen GTEx expression/PCA controls for the completed Bridge tissue benchmark."""
from __future__ import annotations
import hashlib, json, math, resource, time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, torch
from numba import cuda

from run_gtex_biological_retrieval import (ARCHS4_TISSUES, GTEX_TO_ARCHS4,
    MIN_GSES, MIN_REFS, KS, norm_text, metric_rows, wilson)

HERE=Path(__file__).resolve().parents[1]; OUT=HERE/'results/gtex_representation_controls'; FIG=OUT/'figures'; WORK=HERE/'work'
BASE=Path('/home/walt/bridge-rna/benchmarks/mouse_encode/results/task1a_balanced_geometry')
GTEX_MANIFEST=BASE/'gtex_human_sample_manifest.parquet'; GTEX_REP_MANIFEST=BASE/'cache/representation_manifest.parquet'; GTEX_EXPR=BASE/'cache/gtex_log1p_tpm.npy'; GENE_MAP=BASE/'human_gene_mapping.parquet'
CONTROL_WORK=WORK/'representation_metric_controls'; TOP_STEM=WORK/'gtex_representation_controls_top100'
CONV=['Expression Canberra','Expression Pearson','PCA-512 Canberra','PCA-white cosine','PCA-white Canberra']; BRIDGE=['Raw Bridge cosine','Bridge ABTT D=10','Bridge-white cosine']; METHODS=CONV+BRIDGE; KMAX=100
SEED=20260924

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def update_top(v,i,block,start,k=KMAX):
 idx=np.broadcast_to(np.arange(start,start+block.shape[1],dtype=np.int32),block.shape); vv=np.concatenate([v,block],1); ii=np.concatenate([i,idx],1); take=np.argpartition(vv,-k,axis=1)[:,-k:]
 return np.take_along_axis(vv,take,1),np.take_along_axis(ii,take,1)
def sort_top(v,i):
 o=np.argsort(-v,axis=1,kind='stable'); return np.take_along_axis(v,o,1),np.take_along_axis(i,o,1)
def unit(x): return torch.nn.functional.normalize(x,dim=1)
def _canberra(q,r):
 den=q[:,None,:].abs()+r[None,:,:].abs(); return -((q[:,None,:]-r[None,:,:]).abs()/den.clamp_min(1e-20)).sum(2)
def canberra(q,r):
 if not q.is_cuda: return _canberra(q,r)
 out=torch.empty((len(q),len(r)),device=q.device,dtype=torch.float32); block=(16,16); grid=(math.ceil(len(q)/block[0]),math.ceil(len(r)/block[1])); _canberra_cuda[grid,block](cuda.as_cuda_array(q),cuda.as_cuda_array(r),cuda.as_cuda_array(out)); cuda.synchronize(); return out
@cuda.jit
def _canberra_cuda(q,r,out):
 i,j=cuda.grid(2)
 if i<q.shape[0] and j<r.shape[0]:
  total=0.0
  for k in range(q.shape[1]):
   den=abs(q[i,k])+abs(r[j,k])
   if den>0: total+=abs(q[i,k]-r[j,k])/den
  out[i,j]=-total

def retrieve(name,q,loader,nref,device,log):
 nq=len(q); topv=np.full((nq,KMAX),-np.inf,np.float32); topi=np.full((nq,KMAX),-1,np.int32)
 if name=='Expression Canberra': rc,qb=512,512
 elif name=='Expression Pearson': rc,qb=8192,256
 elif 'Canberra' in name: rc,qb=4096,128
 else: rc,qb=16384,512
 qt=torch.from_numpy(np.asarray(q,np.float32)).to(device)
 if name=='Expression Pearson': qt=unit(qt-qt.mean(1,keepdim=True))
 elif name=='PCA-white cosine': qt=unit(qt)
 started=time.time()
 for rs in range(0,nref,rc):
  re=min(nref,rs+rc); rt=torch.from_numpy(np.asarray(loader(rs,re),np.float32)).to(device)
  if name=='Expression Pearson': rt=unit(rt-rt.mean(1,keepdim=True))
  elif name=='PCA-white cosine': rt=unit(rt)
  for qs in range(0,nq,qb):
   qe=min(nq,qs+qb)
   block=(qt[qs:qe]@rt.T if name in ['Expression Pearson','PCA-white cosine'] else canberra(qt[qs:qe],rt)).cpu().numpy()
   topv[qs:qe],topi[qs:qe]=update_top(topv[qs:qe],topi[qs:qe],block,rs)
  if re==nref or (rs//rc)%20==0:
   elapsed=time.time()-started; rate=re/elapsed; log(f'{name}: references {re:,}/{nref:,}; elapsed={elapsed:.0f}s; ETA={(nref-re)/rate:.0f}s')
 return (*sort_top(topv,topi),time.time()-started)

def aggregate(methods_top,ref_tissue,query_tissue,prevalence):
 micro_rows=[]; pq_rows=[]
 for m,inds in methods_top.items():
  a,p=metric_rows(m,inds,ref_tissue,query_tissue,np.ones(len(query_tissue),bool)); micro_rows.extend(a); pq_rows.extend(p)
 perq=pd.DataFrame(pq_rows); micro=pd.DataFrame(micro_rows)
 pt=(perq.groupby(['method','tissue','k'],observed=True).agg(queries=('query_index','size'),same_hits=('same_hits','sum'),labeled_hits=('labeled_hits','sum'),unknown_hits=('unknown_hits','sum')).reset_index())
 pt['annotated_precision']=pt.same_hits/pt.labeled_hits; pt['absolute_hit_rate']=pt.same_hits/(pt.queries*pt.k); pt['unknown_rate']=pt.unknown_hits/(pt.queries*pt.k); pt['background_prevalence']=pt.tissue.map(prevalence); pt['fold_enrichment']=pt.annotated_precision/pt.background_prevalence
 ci=np.array([wilson(int(s),int(n)) for s,n in zip(pt.same_hits,pt.labeled_hits)]); pt['precision_ci_low'],pt['precision_ci_high']=ci[:,0],ci[:,1]
 macro=(pt.groupby(['method','k'],observed=True).agg(tissues=('tissue','nunique'),annotated_precision=('annotated_precision','mean'),absolute_hit_rate=('absolute_hit_rate','mean'),unknown_rate=('unknown_rate','mean'),fold_enrichment=('fold_enrichment','mean')).reset_index())
 return micro,perq,pt,macro

def gtex_control(name,q,tissue,donor,device):
 n=len(q); qt=torch.from_numpy(np.asarray(q,np.float32)).to(device); higher=name in ['Expression Pearson','PCA-white cosine']
 if name=='Expression Pearson': qt=unit(qt-qt.mean(1,keepdim=True))
 elif name=='PCA-white cosine': qt=unit(qt)
 rows=[]
 for qs in range(0,n,32 if 'Expression' in name else 128):
  qe=min(n,qs+(32 if 'Expression' in name else 128)); vals=np.full((qe-qs,KMAX),-np.inf,np.float32); inds=np.full((qe-qs,KMAX),-1,np.int32)
  for rs in range(0,n,512 if 'Expression' in name else 2048):
   re=min(n,rs+(512 if 'Expression' in name else 2048)); rt=qt[rs:re]; block=(qt[qs:qe]@rt.T if higher else canberra(qt[qs:qe],rt)).cpu().numpy(); block[donor[qs:qe,None]==donor[None,rs:re]]=-np.inf; vals,inds=update_top(vals,inds,block,rs)
  _,inds=sort_top(vals,inds); rows.append(inds)
 inds=np.vstack(rows); out=[]
 for k in [1,5,10]:
  hit=tissue[inds[:,:k]]==tissue[:,None]; by=[hit[tissue==t].mean() for t in np.unique(tissue)]; out.append({'method':name,'k':k,'queries':n,'micro_precision':hit.mean(),'tissue_macro_precision':np.mean(by),'tissues':len(by)})
 return out

def main():
 started=time.time(); OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True); logf=(OUT/'run.log').open('w',buffering=1)
 def log(s): z=f'[{pd.Timestamp.now().isoformat()}] {s}'; print(z,flush=True); logf.write(z+'\n')
 log('Starting frozen GTEx representation controls; estimated runtime 50-90 minutes')
 cfg=json.loads((HERE/'config.json').read_text()); inp={k:Path(v) for k,v in cfg['inputs'].items()}; emb=inp['embedding_dir']; spec=json.loads((emb/'embedding_manifest.json').read_text())
 loc=pd.read_parquet(emb/'sample_locations.parquet'); sman=pd.read_parquet(inp['sample_manifest'],columns=['gsm','species']); human=loc.merge(sman,left_on='geo_accession',right_on='gsm',validate='one_to_one'); human=human[norm_text(human.species).eq('human')].reset_index(drop=True); assert len(human)==510709
 meta=pd.read_parquet(inp['sample_metadata']).drop_duplicates('gsm').set_index('gsm').reindex(human.geo_accession); source=norm_text(meta.source_name_ch1).fillna(''); ref_tissue_all=np.full(len(human),'',object)
 for label,pat in ARCHS4_TISSUES.items():
  mask=source.str.match(pat,na=False).to_numpy(); assert not np.any((ref_tissue_all!='')&mask); ref_tissue_all[mask]=label
 leak_text=meta.title.fillna('').astype(str)+' '+meta.characteristics_ch1.fillna('').astype(str); leak=leak_text.str.contains(r'GTEX[- _]|Genotype-Tissue Expression|\bGTEx\b',case=False,regex=True,na=False).to_numpy(); ref_indices=np.flatnonzero(~leak); assert len(ref_indices)==510376
 ref_tissue=ref_tissue_all[ref_indices]; ref_gse=meta.series_id.fillna('').astype(str).to_numpy()[ref_indices]; labeled=ref_tissue!=''; tissue_qc=pd.DataFrame({'tissue':ref_tissue[labeled],'gse':ref_gse[labeled]}).groupby('tissue').agg(references=('tissue','size'),gses=('gse','nunique')).reset_index(); tissue_qc['prevalence']=tissue_qc.references/labeled.sum(); prevalence=tissue_qc.set_index('tissue').prevalence.to_dict()
 gtex=pd.read_parquet(GTEX_MANIFEST).reset_index(drop=True); rep=pd.read_parquet(GTEX_REP_MANIFEST); gx=np.load(GTEX_EXPR,mmap_mode='r'); genes=pd.read_parquet(GENE_MAP).sort_values('gene_order'); canonical=pd.read_csv(inp['canonical_genes']).sort_values('token_id')
 assert gx.shape==(6314,15165) and rep.iloc[:6314].sample_id.reset_index(drop=True).equals(gtex.sample_id) and genes.gene_symbol.astype(str).str.upper().reset_index(drop=True).equals(canonical.gene_symbol.astype(str).str.upper().reset_index(drop=True))
 assert np.isfinite(gx).all() and gx.min()>=0; gtex['donor_id']=gtex.sample_id.str.extract(r'^(GTEX-[^-]+)',expand=False); gtex['archs4_tissue']=gtex.tissue.map(GTEX_TO_ARCHS4)
 frozen_harm=pd.read_csv(HERE/'results/gtex_biological_retrieval/tissue_harmonization.csv'); eligible=set(frozen_harm.loc[frozen_harm.eligible,'archs4_tissue']); eq=gtex.archs4_tissue.isin(eligible).to_numpy(); qmeta=gtex[eq].reset_index(drop=True); qexpr=np.asarray(gx[eq],np.float32); qtissue=qmeta.archs4_tissue.to_numpy(); assert len(qmeta)==6064 and len(eligible)==13
 expr=np.memmap(CONTROL_WORK/'archs4_human_log1p_tpm.float32.mmap',mode='r',dtype=np.float32,shape=(510709,15165)); pca=np.memmap(CONTROL_WORK/'archs4_pca512.float32.mmap',mode='r',dtype=np.float32,shape=(510709,512)); pp=np.load(CONTROL_WORK/'expression_pca512_transform.npz'); qpca=(qexpr-pp['mean'])@pp['components'].T; scale=1/np.sqrt(np.maximum(pp['eigenvalues'],float(pp['eigenvalue_floor']))); qwhite=qpca*scale
 audit={'gtex_expression':str(GTEX_EXPR),'gtex_expression_sha256':sha(GTEX_EXPR),'gtex_manifest':str(GTEX_MANIFEST),'gtex_manifest_sha256':sha(GTEX_MANIFEST),'gtex_representation_manifest':str(GTEX_REP_MANIFEST),'gene_mapping':str(GENE_MAP),'normalization':'natural log1p(TPM)','genes':15165,'gtex_samples':6314,'eligible_queries':6064,'eligible_tissues':13,'archs4_expression_cache':str(CONTROL_WORK/'archs4_human_log1p_tpm.float32.mmap'),'archs4_references_after_frozen_exclusion':len(ref_indices),'gtex_origin_exclusions':int(leak.sum()),'pca_transform':str(CONTROL_WORK/'expression_pca512_transform.npz'),'fit_on_gtex':False,'bridge_results_reused':True}; (OUT/'asset_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
 device=torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
 if device.type=='cuda':
  tq=torch.from_numpy(np.asarray(qexpr[:3])).to(device); tr=torch.from_numpy(np.asarray(expr[ref_indices[:5]])).to(device); delta=float((canberra(tq,tr)-_canberra(tq,tr)).abs().max().cpu()); assert delta<.005,delta; audit['canberra_cuda_vs_eager_max_abs_delta']=delta; (OUT/'asset_audit.json').write_text(json.dumps(audit,indent=2)+'\n'); log(f'Canberra fused-kernel validation max_abs_delta={delta:.6g}')
 runt=[]; tops={}; vals={}
 loaders={'Expression Canberra':lambda s,e:np.asarray(expr[ref_indices[s:e]]),'Expression Pearson':lambda s,e:np.asarray(expr[ref_indices[s:e]]),'PCA-512 Canberra':lambda s,e:np.asarray(pca[ref_indices[s:e]]),'PCA-white cosine':lambda s,e:np.asarray(pca[ref_indices[s:e]])*scale,'PCA-white Canberra':lambda s,e:np.asarray(pca[ref_indices[s:e]])*scale}; queries={'Expression Canberra':qexpr,'Expression Pearson':qexpr,'PCA-512 Canberra':qpca,'PCA-white cosine':qwhite,'PCA-white Canberra':qwhite}
 for j,m in enumerate(CONV):
  cache=Path(f'{TOP_STEM}_{j}.npz')
  if cache.exists():
   z=np.load(cache); assert int(z['eligible_queries'])==6064 and int(z['references'])==510376; vals[m],tops[m]=z['values'],z['indices']; sec=float(z['runtime_seconds']); log(f'Reused Top-100 cache: {m}')
  else:
   v,i,sec=retrieve(m,queries[m],loaders[m],len(ref_indices),device,log); vals[m],tops[m]=v,i; np.savez_compressed(cache,eligible_queries=6064,references=510376,values=v,indices=i,runtime_seconds=sec)
  runt.append({'method':m,'runtime_seconds':sec})
 # Existing Bridge Top-100 indices are source of truth.
 bz=np.load(WORK/'gtex_archs4_top100.npz'); source_names=['Raw cosine','ABTT D=10','Whitening'];
 for m,src in zip(BRIDGE,source_names): j=['Raw cosine','ABTT D=10','Whitening','NICDM','Local Scaling','CSLS'].index(src); tops[m]=bz[f'indices_{j}']; vals[m]=bz[f'values_{j}']
 # Calculate conventional metrics, then append unmodified established Bridge rows.
 cmicro,cperq,cpt,cmacro=aggregate({m:tops[m] for m in CONV},ref_tissue,qtissue,prevalence)
 old=HERE/'results/gtex_biological_retrieval'; om=pd.read_csv(old/'retrieval_metrics_micro.csv'); oma=pd.read_csv(old/'retrieval_metrics_tissue_macro.csv'); opt=pd.read_csv(old/'per_tissue_metrics.csv'); opq=pd.read_parquet(old/'per_query_metrics.parquet'); rename={'Raw cosine':'Raw Bridge cosine','ABTT D=10':'Bridge ABTT D=10','Whitening':'Bridge-white cosine'}
 om=om[om.method.isin(rename)].assign(method=lambda d:d.method.map(rename)); oma=oma[oma.method.isin(rename)].assign(method=lambda d:d.method.map(rename)); opt=opt[opt.method.isin(rename)].assign(method=lambda d:d.method.map(rename)); opq=opq[opq.method.isin(rename)].assign(method=lambda d:d.method.map(rename))
 micro=pd.concat([cmicro,om],ignore_index=True); macro=pd.concat([cmacro,oma],ignore_index=True); pt=pd.concat([cpt,opt],ignore_index=True); perq=pd.concat([cperq,opq],ignore_index=True); micro.to_csv(OUT/'retrieval_metrics_micro.csv',index=False); macro.to_csv(OUT/'retrieval_metrics_tissue_macro.csv',index=False); pt.to_csv(OUT/'per_tissue_metrics.csv',index=False); perq.to_parquet(OUT/'per_query_metrics.parquet',index=False)
 p10=pt[pt.k==10]; wide=[]
 for field in ['annotated_precision','absolute_hit_rate','unknown_rate','fold_enrichment']:
  x=p10.pivot(index='tissue',columns='method',values=field); x.columns=[f'{m} | {field}' for m in x.columns]; wide.append(x)
 pd.concat(wide,axis=1).reset_index().to_csv(OUT/'per_tissue_p10_comparison.csv',index=False)
 # Confusion matrices from the identical neighbor sets.
 tissues=sorted(eligible)
 for m in METHODS:
  lab=ref_tissue[tops[m][:,:10]]; pairs=[(q,x) for q,a in zip(qtissue,lab) for x in a if x]; cm=pd.crosstab(pd.Series([x[0] for x in pairs],name='query_tissue'),pd.Series([x[1] for x in pairs],name='retrieved_tissue')).reindex(index=tissues,columns=tissues,fill_value=0); cm.to_csv(OUT/f"confusion_{m.lower().replace(' ','_')}_top10.csv")
 # Different-donor GTEx control for conventional methods; append established Bridge summaries.
 control=[]
 control_tissue=qmeta.tissue.to_numpy()
 for m in CONV:
  log(f'GTEx→GTEx different-donor control: {m}'); control.extend(gtex_control(m,queries[m],control_tissue,qmeta.donor_id.to_numpy(),device))
 oc=pd.read_csv(old/'gtex_to_gtex_control.csv'); oc=oc[oc.method.isin(rename)].assign(method=lambda d:d.method.map(rename)); control=pd.concat([pd.DataFrame(control),oc[oc.k.isin([1,5,10])]],ignore_index=True); control.to_csv(OUT/'gtex_to_gtex_control.csv',index=False)
 # Top-10 neighborhood overlap against frozen Bridge whitening.
 overlaps=[]; target=tops['Bridge-white cosine'][:,:10]; target_labels=ref_tissue[target]; status=np.where(target_labels[:,0]==qtissue,'correct_top1',np.where(target_labels[:,0]=='','unknown_top1','incorrect_top1'))
 for m in ['Expression Canberra','Expression Pearson','PCA-512 Canberra']:
  for i,(a,b) in enumerate(zip(tops[m][:,:10],target)):
   shared=len(set(a).intersection(b)); overlaps.append({'method':m,'query_index':i,'tissue':qtissue[i],'bridge_white_status':status[i],'shared_neighbors':shared,'jaccard':shared/(20-shared)})
 overlap=pd.DataFrame(overlaps); overlap.to_csv(OUT/'neighborhood_overlap_per_query.csv',index=False); overlap.groupby(['method','tissue','bridge_white_status'],observed=True).agg(queries=('query_index','size'),mean_shared=('shared_neighbors','mean'),median_shared=('shared_neighbors','median'),mean_jaccard=('jaccard','mean'),median_jaccard=('jaccard','median')).reset_index().to_csv(OUT/'neighborhood_overlap_summary.csv',index=False)
 # Central identity-versus-biology comparison.
 rec=pd.read_csv(HERE/'results/representation_metric_controls/micro_metrics.csv').set_index(['representation','metric']); maprec={m:key for m,key in [('Expression Canberra',('log1p(TPM)','Canberra')),('Expression Pearson',('log1p(TPM)','Pearson')),('PCA-512 Canberra',('PCA-512','Canberra')),('PCA-white cosine',('PCA-512 whitened','Cosine')),('PCA-white Canberra',('PCA-512 whitened','Canberra')),('Raw Bridge cosine',('Bridge','Cosine')),('Bridge ABTT D=10',('ABTT D=10','Cosine')),('Bridge-white cosine',('Bridge whitened','Cosine'))]}; dims={'Expression Canberra':15165,'Expression Pearson':15165,'PCA-512 Canberra':512,'PCA-white cosine':512,'PCA-white Canberra':512,'Raw Bridge cosine':512,'Bridge ABTT D=10':502,'Bridge-white cosine':512}; central=[]
 for m in METHODS:
  central.append({'method':m,'dimensions':dims[m],'recount3_exact_r_at_10':rec.loc[maprec[m],'r_at_10'],'gtex_archs4_tissue_macro_p_at_10':macro[(macro.method==m)&(macro.k==10)].annotated_precision.iloc[0],'gtex_gtex_tissue_macro_p_at_10':control[(control.method==m)&(control.k==10)].tissue_macro_precision.iloc[0]})
 central=pd.DataFrame(central); central.to_csv(OUT/'identity_vs_biological_retrieval.csv',index=False)
 pd.DataFrame(runt).to_csv(OUT/'runtime.csv',index=False); storage=pd.DataFrame([{'representation':'Expression','dimensions':15165,'dtype':'float32','bytes_per_sample':60660,'reference_bytes':510376*60660},{'representation':'PCA-512','dimensions':512,'dtype':'float32','bytes_per_sample':2048,'reference_bytes':510376*2048},{'representation':'Bridge stored','dimensions':512,'dtype':'float16','bytes_per_sample':1024,'reference_bytes':510376*1024}]); storage.to_csv(OUT/'storage.csv',index=False)
 # Figures.
 def save(fig,name): fig.tight_layout(); fig.savefig(FIG/f'{name}.png',dpi=220); fig.savefig(FIG/f'{name}.pdf'); plt.close(fig)
 fig,ax=plt.subplots(figsize=(11,5)); x=np.arange(len(central)); ax.scatter(central.recount3_exact_r_at_10,central.gtex_archs4_tissue_macro_p_at_10,s=70); [ax.annotate(r.method,(r.recount3_exact_r_at_10,r.gtex_archs4_tissue_macro_p_at_10),fontsize=8,xytext=(3,3),textcoords='offset points') for _,r in central.iterrows()]; ax.set(xlabel='recount3 exact R@10',ylabel='GTEx→ARCHS4 tissue-macro P@10',title='Exact identity versus biological retrieval'); save(fig,'identity_vs_biology')
 for table,stem,title in [(macro,'gtex_archs4_macro_by_k','GTEx→ARCHS4 tissue-macro annotated precision'),(control,'gtex_gtex_macro_by_k','GTEx→GTEx different-donor tissue-macro precision')]:
  fig,ax=plt.subplots(figsize=(11,6));
  for m in METHODS:
   d=table[table.method==m]; y=d.annotated_precision if 'annotated_precision' in d else d.tissue_macro_precision; ax.plot(d.k,y,marker='o',label=m)
  ax.set(xscale='log',xticks=[1,5,10,100] if table is macro else [1,5,10],xticklabels=[1,5,10,100] if table is macro else [1,5,10],ylim=(0,1),xlabel='K',ylabel='Tissue-macro precision',title=title); ax.legend(fontsize=7,ncol=2); save(fig,stem)
 heat=pt[pt.k==10].pivot(index='tissue',columns='method',values='annotated_precision')[METHODS]; fig,ax=plt.subplots(figsize=(12,7)); im=ax.imshow(heat,cmap='viridis',vmin=0,vmax=1,aspect='auto'); ax.set_xticks(range(len(METHODS)),METHODS,rotation=35,ha='right'); ax.set_yticks(range(len(heat)),heat.index); fig.colorbar(im,ax=ax,label='Annotated P@10'); ax.set_title('Per-tissue GTEx→ARCHS4 retrieval'); save(fig,'per_tissue_p10')
 selected=['Expression Canberra','Expression Pearson','PCA-512 Canberra','Raw Bridge cosine','Bridge-white cosine']; fig,axes=plt.subplots(1,len(selected),figsize=(22,5));
 for ax,m in zip(axes,selected): cm=pd.read_csv(OUT/f"confusion_{m.lower().replace(' ','_')}_top10.csv",index_col=0); cm=cm.div(cm.sum(1).replace(0,np.nan),axis=0); ax.imshow(cm,cmap='Blues',vmin=0,vmax=1); ax.set_title(m,fontsize=9); ax.set_xticks([]); ax.set_yticks(range(len(cm)),cm.index if ax is axes[0] else [])
 save(fig,'tissue_confusion_comparison')
 ov=overlap.groupby(['method','tissue']).jaccard.mean().unstack(0); fig,ax=plt.subplots(figsize=(10,7)); im=ax.imshow(ov,cmap='magma',aspect='auto'); ax.set_xticks(range(len(ov.columns)),ov.columns,rotation=30,ha='right'); ax.set_yticks(range(len(ov)),ov.index); fig.colorbar(im,ax=ax,label='Mean Top-10 Jaccard vs Bridge-white'); ax.set_title('Neighborhood overlap by tissue'); save(fig,'neighborhood_overlap')
 fig,ax=plt.subplots(figsize=(9,6)); ax.scatter(central.dimensions,central.gtex_archs4_tissue_macro_p_at_10,s=80); [ax.annotate(r.method,(r.dimensions,r.gtex_archs4_tissue_macro_p_at_10),fontsize=8,xytext=(3,3),textcoords='offset points') for _,r in central.iterrows()]; ax.set(xscale='log',xlabel='Dimensions (log scale)',ylabel='GTEx→ARCHS4 tissue-macro P@10',title='Compression versus biological retrieval'); save(fig,'compression_vs_biology')
 fig,ax=plt.subplots(figsize=(12,7)); delta=heat.sub(heat['Bridge-white cosine'],axis=0); im=ax.imshow(delta,cmap='RdBu_r',vmin=-max(abs(delta.min().min()),abs(delta.max().max())),vmax=max(abs(delta.min().min()),abs(delta.max().max())),aspect='auto'); ax.set_xticks(range(len(delta.columns)),delta.columns,rotation=35,ha='right'); ax.set_yticks(range(len(delta)),delta.index); fig.colorbar(im,ax=ax,label='P@10 difference vs Bridge-white'); ax.set_title('Representation comparison by tissue'); save(fig,'representation_difference_by_tissue')
 prov={'created_at':pd.Timestamp.now(tz='UTC').isoformat(),'current_run_runtime_seconds':time.time()-started,'conventional_archs4_retrieval_runtime_seconds':sum(x['runtime_seconds'] for x in runt),'peak_rss_gb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,'seed':SEED,'methods_frozen_before_gtEx_extension':METHODS,'gtex_fit':False,'bridge_rankings_reused':True,'gtEx_bridge_rerun':False,'queries':6064,'references':510376,'limitations':['ARCHS4 tissue labels remain conservatively mapped and mostly unknown.','Raw expression retrieval is substantially larger and slower than 512-D representations.','GTEx had already been inspected; method list was frozen from recount3 controls before this extension.']}; (OUT/'provenance.json').write_text(json.dumps(prov,indent=2)+'\n'); log(f"COMPLETE runtime={prov['current_run_runtime_seconds']:.1f}s peak_RSS={prov['peak_rss_gb']:.2f}GB"); logf.close()
if __name__=='__main__': main()
