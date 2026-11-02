#!/usr/bin/env python3
"""Prepare full-corpus expression and ARCHS4-only PCA controls."""
from __future__ import annotations
import hashlib, json, re, resource, time
from pathlib import Path
import h5py, numpy as np, pandas as pd, torch
from scipy import sparse

HERE=Path(__file__).resolve().parents[1]; OUT=HERE/'results/representation_metric_controls'; WORK=HERE/'work/representation_metric_controls'
H5=Path('/home/walt/bridge-rna/data/archs4/human_gene_v2.5.h5'); LENGTHS=Path('/home/walt/bridge-rna/data/gencode/gencode_v49_gene_exon_lengths.csv')
SEED=20260924; BATCH=1024; PCA_BATCH=2048; K=512; OVER=32; POWER=1

def decode(x): return [v.decode() if isinstance(v,bytes) else str(v) for v in x]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def norm(s): return s.astype('string').str.strip().str.lower()

def main():
 t0=time.time(); OUT.mkdir(parents=True,exist_ok=True); WORK.mkdir(parents=True,exist_ok=True)
 old_log=(OUT/'prepare_run.log').read_text() if (OUT/'prepare_run.log').exists() else ''
 hits=re.findall(r'expression source scan ([\d,]+)/',old_log); resume_source=int(hits[-1].replace(',','')) if hits and not (OUT/'asset_audit.json').exists() else 0
 lf=open(OUT/'prepare_run.log','a' if resume_source else 'w',buffering=1)
 def log(x):
  z=f'[{pd.Timestamp.now().isoformat()}] {x}'; print(z,flush=True); lf.write(z+'\n')
 log('Preparing expression/PCA controls; estimated runtime 30-120 minutes')
 cfg=json.loads((HERE/'config.json').read_text()); inp={k:Path(v) for k,v in cfg['inputs'].items()}; emb=inp['embedding_dir']
 loc=pd.read_parquet(emb/'sample_locations.parquet'); man=pd.read_parquet(inp['sample_manifest'],columns=['gsm','species'])
 human=loc.merge(man,left_on='geo_accession',right_on='gsm',validate='one_to_one'); human=human[norm(human.species).eq('human')].reset_index(drop=True)
 genes=pd.read_csv(inp['canonical_genes']).sort_values('token_id').gene_symbol.astype(str).str.upper().tolist()
 lengths=pd.read_csv(LENGTHS).drop_duplicates('gene_symbol'); lm=lengths.set_index(lengths.gene_symbol.astype(str).str.upper()).exon_length
 lv=lm.reindex(genes).to_numpy(np.float64); assert len(human)==510709 and len(genes)==15165 and np.all(lv>0)
 expr_path=WORK/'archs4_human_log1p_tpm.float32.mmap'; shape=(len(human),len(genes))
 fallback_blocks=(json.loads((OUT/'asset_audit.json').read_text()).get('eligible_only_hdf5_fallback_blocks',[])
                  if (OUT/'asset_audit.json').exists() else [])
 with h5py.File(H5,'r') as h:
  hgsm=decode(h['meta/samples/geo_accession'][:]); idx={g:i for i,g in enumerate(hgsm)}
  order=np.array([idx[g] for g in human.geo_accession],np.int64)
  symbols=decode(h['meta/genes/symbol'][:]); target={g:i for i,g in enumerate(genes)}; rows=[]; cols=[]
  for j,g in enumerate(symbols):
   g=g.split('.')[0].upper()
   if g in target: rows.append(target[g]); cols.append(j)
  agg=sparse.csr_matrix((np.ones(len(rows),np.float32),(rows,cols)),shape=(len(genes),len(symbols)))
  if (not expr_path.exists() or expr_path.stat().st_size != np.prod(shape)*4
      or not (OUT/'asset_audit.json').exists()):
   mm=np.memmap(expr_path,mode='r+' if resume_source else 'w+',dtype=np.float32,shape=shape); ds=h['data/expression']; start_time=time.time(); done=int(np.sum(order < resume_source))
   # h5py arbitrary-column fancy indexing is pathologically slow for this
   # column-chunked file. Scan contiguous source columns once and scatter only
   # the fixed human-reference subset into benchmark order.
   target_for_source=np.full(len(hgsm),-1,np.int64); target_for_source[order]=np.arange(len(human))
   if resume_source: log(f'Resuming expression source scan from durable boundary {resume_source:,}')
   for hs in range(resume_source,len(hgsm),BATCH):
    he=min(len(hgsm),hs+BATCH); targets=target_for_source[hs:he]; selected=np.flatnonzero(targets>=0)
    if len(selected):
     try: raw=np.asarray(ds[:,hs:he])[:,selected]
     except OSError:
      # The shared HDF5 contains a few damaged non-reference columns. Read
      # only fixed eligible columns in an affected block; never impute a
      # requested reference sample.
      absolute=hs+selected; raw=np.asarray(ds[:,absolute.tolist()]); fallback_blocks.append([hs,he]); log(f'HDF5 filter failure in {hs:,}:{he:,}; eligible-only fallback succeeded')
     aligned=np.asarray(agg@raw,dtype=np.float64); rates=aligned/(lv[:,None]/1000); den=rates.sum(0)
     mm[targets[selected]]=np.log1p(np.divide(rates*1e6,den[None,:],out=np.zeros_like(rates),where=den[None,:]>0).T).astype(np.float32); done+=len(selected)
    if he==len(hgsm) or he//BATCH%20==0:
     mm.flush(); elapsed=time.time()-start_time; rate=(he-resume_source)/elapsed; log(f'expression source scan {he:,}/{len(hgsm):,}; human written={done:,}; ETA={(len(hgsm)-he)/rate:.0f}s')
   assert done==len(human)
   mm.flush(); del mm
  else: log('Reusing complete full-corpus expression cache')
 expr=np.memmap(expr_path,mode='r',dtype=np.float32,shape=shape)
 # Verify exact counterpart rows against upstream paired cache.
 up=inp['upstream_benchmark']; samples=pd.read_parquet(up/'work/paired_expression/samples.parquet'); keep=samples.study_exposure.eq('unseen_study').to_numpy(); samples=samples[keep].reset_index(drop=True)
 paired=np.load(up/'work/paired_expression/archs4_log1p_tpm.npy',mmap_mode='r')[keep]
 lookup=pd.Series(np.arange(len(human)),index=human.geo_accession); correct=samples.gsm.map(lookup).to_numpy(int)
 delta=float(np.max(np.abs(np.asarray(expr[correct])-np.asarray(paired))))
 assert delta<2e-5,delta; log(f'paired expression reproduction max_abs_delta={delta:.3g}')
 # Mean and total variance.
 mean=np.zeros(len(genes),np.float64); sumsq=0.;
 for s in range(0,len(human),PCA_BATCH):
  x=np.asarray(expr[s:s+PCA_BATCH],np.float64); mean+=x.sum(0); sumsq+=np.square(x).sum()
 mean/=len(human); total_ss=sumsq-len(human)*np.dot(mean,mean)
 rng=np.random.default_rng(SEED); omega=rng.standard_normal((len(genes),K+OVER),dtype=np.float32)/np.sqrt(K+OVER)
 ypath=WORK/'pca_range.float32.mmap'; Y=np.memmap(ypath,mode='w+',dtype=np.float32,shape=(len(human),K+OVER))
 dev=torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu'); om=torch.from_numpy(omega).to(dev); mt=torch.from_numpy(mean.astype(np.float32)).to(dev)
 for s in range(0,len(human),PCA_BATCH):
  x=torch.from_numpy(np.asarray(expr[s:s+PCA_BATCH])).to(dev)-mt; Y[s:s+len(x)]=(x@om).cpu().numpy()
  if (s//PCA_BATCH)%25==0: log(f'PCA range pass {min(s+PCA_BATCH,len(human)):,}/{len(human):,}')
 # One subspace power iteration: Y <- X (X'Y), with scaling for stability.
 for power in range(POWER):
  gram=torch.zeros((len(genes),K+OVER),dtype=torch.float32,device=dev)
  for s in range(0,len(human),PCA_BATCH):
   x=torch.from_numpy(np.asarray(expr[s:s+PCA_BATCH])).to(dev)-mt; y=torch.from_numpy(np.asarray(Y[s:s+len(x)])).to(dev); gram+=x.T@y
  gram/=torch.linalg.vector_norm(gram,dim=0,keepdim=True).clamp_min(1e-20)
  gt=gram
  for s in range(0,len(human),PCA_BATCH):
   x=torch.from_numpy(np.asarray(expr[s:s+PCA_BATCH])).to(dev)-mt; Y[s:s+len(x)]=(x@gt).cpu().numpy()
  log(f'PCA power iteration {power+1}/{POWER}')
 # Thin QR then B=Q'X and small SVD.
 yt=torch.from_numpy(np.asarray(Y)).to(dev); qt,_=torch.linalg.qr(yt,mode='reduced'); del yt
 B=torch.zeros((K+OVER,len(genes)),dtype=torch.float32,device=dev)
 for s in range(0,len(human),PCA_BATCH):
  x=torch.from_numpy(np.asarray(expr[s:s+PCA_BATCH])).to(dev)-mt; B+=qt[s:s+len(x)].T@x
  if (s//PCA_BATCH)%25==0: log(f'PCA B pass {min(s+PCA_BATCH,len(human)):,}/{len(human):,}')
 _,sing_t,vt_t=torch.linalg.svd(B,full_matrices=False); sing=sing_t[:K].cpu().numpy().astype(np.float64); components=vt_t[:K].cpu().numpy().astype(np.float32); eig=np.square(sing)/(len(human)-1)
 scores_path=WORK/'archs4_pca512.float32.mmap'; scores=np.memmap(scores_path,mode='w+',dtype=np.float32,shape=(len(human),K)); ct=torch.from_numpy(components).to(dev); score_sum=torch.zeros(K,dtype=torch.float64,device=dev); score_outer=torch.zeros((K,K),dtype=torch.float64,device=dev)
 for s in range(0,len(human),PCA_BATCH):
  x=torch.from_numpy(np.asarray(expr[s:s+PCA_BATCH])).to(dev)-mt; block=x@ct.T; scores[s:s+len(x)]=block.cpu().numpy(); bd=block.double(); score_sum+=bd.sum(0); score_outer+=bd.T@bd
 # Exact Rayleigh–Ritz diagonalization inside the fixed randomized subspace.
 cov=(score_outer-score_sum[:,None]*score_sum[None,:]/len(human))/(len(human)-1); eig_t,rot=torch.linalg.eigh(cov); eig_t=eig_t.flip(0); rot=rot.flip(1); eig=eig_t.cpu().numpy(); components=(rot.T.float()@ct).cpu().numpy().astype(np.float32); rt=rot.float()
 for s in range(0,len(human),PCA_BATCH):
  block=torch.from_numpy(np.asarray(scores[s:s+PCA_BATCH])).to(dev)@rt; scores[s:s+len(block)]=block.cpu().numpy()
 scores.flush(); del scores,qt,Y
 explained=eig*(len(human)-1)/total_ss
 floor=float(eig[0]*1e-12); affected=int((eig<floor).sum()); np.savez(WORK/'expression_pca512_transform.npz',mean=mean.astype(np.float32),components=components,eigenvalues=eig,eigenvalue_floor=floor)
 pd.DataFrame({'pc':np.arange(1,K+1),'eigenvalue':eig,'explained_variance_ratio':explained,'cumulative_explained_variance':np.cumsum(explained)}).to_csv(OUT/'pca_eigenvalues.csv',index=False)
 pd.DataFrame([{'components':K,'oversampling':OVER,'power_iterations':POWER,'solver':'randomized range finder + thin QR + small SVD','fit_samples':len(human),'genes':len(genes),'variance_explained_512':explained.sum(),'eigenvalue_floor':floor,'affected_dimensions':affected,'participation_ratio':eig.sum()**2/np.square(eig).sum()}]).to_csv(OUT/'pca_provenance.csv',index=False)
 white_cov=(rot.T@cov@rot)/(torch.sqrt(eig_t)[:,None]*torch.sqrt(eig_t)[None,:]); wc=white_cov.cpu().numpy(); white_var=np.diag(wc); off=wc-np.diag(white_var)
 pd.DataFrame([{'dimensions_retained':K,'eigenvalue_floor':floor,'affected_dimensions':affected,'post_whitening_variance_mean':white_var.mean(),'post_whitening_variance_sd':white_var.std(),'post_whitening_variance_min':white_var.min(),'post_whitening_variance_max':white_var.max(),'post_whitening_max_abs_offdiagonal':np.max(np.abs(off))}]).to_csv(OUT/'pca_whitening_provenance.csv',index=False)
 audit={'archs4_h5':str(H5),'h5_samples':888821,'human_reference_samples':len(human),'expression_dataset':'data/expression','source_dtype':'uint32 counts','source_genes':67186,'mapped_source_gene_rows':len(rows),'canonical_genes':str(inp['canonical_genes']),'canonical_genes_sha256':sha(inp['canonical_genes']),'exon_lengths':str(LENGTHS),'normalization':'counts / exon-union kb -> TPM -> natural log1p','genes':len(genes),'sample_order':str(emb/'sample_locations.parquet')+' filtered to human via sample manifest','recount3_queries':str(up/'work/paired_expression/recount3_log1p_tpm.npy'),'queries':562,'paired_reproduction_max_abs_delta':delta,'full_corpus_available':True,'large_cache':str(expr_path),'eligible_only_hdf5_fallback_blocks':fallback_blocks}
 (OUT/'asset_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
 prov={'runtime_seconds':time.time()-t0,'peak_rss_gb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,'device':str(dev),'seed':SEED,'reference_fit_only':True,'query_information_used_for_fit':False}
 (OUT/'prepare_provenance.json').write_text(json.dumps(prov,indent=2)+'\n'); log(f"COMPLETE runtime={prov['runtime_seconds']:.1f}s peak_RSS={prov['peak_rss_gb']:.2f}GB"); lf.close()
if __name__=='__main__': main()
