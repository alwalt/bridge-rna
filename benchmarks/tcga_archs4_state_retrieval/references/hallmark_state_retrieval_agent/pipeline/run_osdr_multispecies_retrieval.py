#!/usr/bin/env python3
"""OSDR retrieval against human, mouse, and exact merged ARCHS4 references."""
from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from common import RESULTS,WORK,l2_rows,save_json

LEG=RESULTS/'osdr_atlas'; OUT=RESULTS/'osdr_multispecies'; OW=WORK/'osdr_multispecies'; QW=WORK/'osdr_atlas'; MW=WORK/'archs4_multispecies'; TOP=100

def top_mask(x,k=10):
 idx=np.argpartition(np.abs(x),-k,axis=1)[:,-k:]; m=np.zeros(x.shape,bool);m[np.arange(len(x))[:,None],idx]=True;return m
def retrieve(q,path,overlap=False):
 r=np.load(path,mmap_mode='r'); dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
 if overlap:
  rm=top_mask(np.asarray(r),10); rr=np.concatenate([rm&(r>=0),rm&(r<0)],1).astype(np.float32); qm=top_mask(q,10); qq=np.concatenate([qm&(q>=0),qm&(q<0)],1).astype(np.float32); rt=torch.from_numpy(rr).to(dev); qx=qq
 else: rt=torch.from_numpy(np.asarray(r)).to(dev); qx=l2_rows(q)
 ii=np.empty((len(q),TOP),np.int32);vv=np.empty((len(q),TOP),np.float32);qqt=np.empty((len(q),5),np.float32); levels=(.5,.9,.95,.99,.999)
 for a in range(0,len(q),32):
  b=min(a+32,len(q)); score=torch.from_numpy(qx[a:b]).to(dev)@rt.T
  if overlap: score=score/10
  v,i=torch.topk(score,TOP,dim=1,sorted=True);ii[a:b]=i.cpu();vv[a:b]=v.cpu();qqt[a:b]=np.quantile(score.cpu().numpy(),levels,axis=1).T
  if b%256==0 or b==len(q): print(f'{path.name}: {b}/{len(q)}',flush=True)
 del rt
 if dev.type=='cuda': torch.cuda.empty_cache()
 return ii,vv,qqt
def merge_top(hi,hs,mi,ms):
 n=len(hi); oi=np.empty((n,TOP),np.int32);os=np.empty((n,TOP),np.float32);osp=np.empty((n,TOP),np.int8)
 for j in range(n):
  score=np.concatenate([hs[j],ms[j]]); idx=np.argpartition(score,-TOP)[-TOP:];idx=idx[np.argsort(score[idx])[::-1]]
  is_mouse=idx>=TOP; osp[j]=is_mouse; merged=np.empty(TOP,np.int32)
  merged[~is_mouse]=hi[j,idx[~is_mouse]]; merged[is_mouse]=mi[j,idx[is_mouse]-TOP]; oi[j]=merged; os[j]=score[idx]
 return oi,os,osp
def bh(p):
 o=np.argsort(p);q=np.empty(len(p));q[o]=np.minimum.accumulate((p[o]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1];return np.minimum(q,1)

def main():
 OUT.mkdir(parents=True,exist_ok=True);OW.mkdir(parents=True,exist_ok=True);started=time.time()
 man=pd.read_parquet(LEG/'eligible_sample_manifest.parquet').sort_values('matrix_row').reset_index(drop=True); b=np.load(QW/'bridge_embeddings.float32.npy');h=np.load(QW/'hallmark_profiles.float32.npy')
 cm=man.groupby(['group_id','condition'],sort=True).first().reset_index();cm['sample_id']=cm.group_id+'_'+cm.condition+'_CENTROID';cb=np.vstack([b[(man.group_id==r.group_id)&(man.condition==r.condition)].mean(0) for r in cm.itertuples()]);ch=np.vstack([h[(man.group_id==r.group_id)&(man.condition==r.condition)].mean(0) for r in cm.itertuples()])
 qm=pd.concat([man.assign(query_type='individual'),cm.assign(query_type='centroid')],ignore_index=True,sort=False);allb=np.vstack([b,cb]);allh=np.vstack([h,ch]);assert len(qm)==1647
 hc=np.load(QW/'retrieval_top100.npz'); hi={'Bridge-512 cosine':hc['bridge_indices'],'Hallmark-50 cosine':hc['hallmark_indices'],'Hallmark signed top10 overlap':hc['overlap_indices']};hs={'Bridge-512 cosine':hc['bridge_scores'],'Hallmark-50 cosine':hc['hallmark_scores'],'Hallmark signed top10 overlap':hc['overlap_scores']}
 cache=OW/'mouse_top100.npz'
 if cache.exists():
  z=np.load(cache);mbi,mbs,mhi,mhs,moi,mos=(z[k] for k in ['bi','bs','hi','hs','oi','os'])
 else:
  mbi,mbs,_=retrieve(allb,MW/'mouse_bridge_cosine_index.float32.npy');mhi,mhs,_=retrieve(allh,MW/'mouse_hallmark_cosine_index.float32.npy');moi,mos,_=retrieve(allh,MW/'mouse_hallmark_signatures.float32.npy',True);np.savez_compressed(cache,bi=mbi,bs=mbs,hi=mhi,hs=mhs,oi=moi,os=mos)
 mi={'Bridge-512 cosine':mbi,'Hallmark-50 cosine':mhi,'Hallmark signed top10 overlap':moi};ms={'Bridge-512 cosine':mbs,'Hallmark-50 cosine':mhs,'Hallmark signed top10 overlap':mos}
 human=pd.read_parquet(RESULTS/'archs4/sample_metadata.parquet');human['species']='human';mouse=pd.read_parquet(RESULTS/'archs4_multispecies/mouse_sample_metadata.parquet');mouse['species']='mouse';names=np.load(RESULTS/'frozen/frozen_gtex_hallmark_probe.npz')['hallmarks'].astype(str)
 href=np.load(WORK/'archs4_hallmark_signatures.float32.npy',mmap_mode='r');mref=np.load(MW/'mouse_hallmark_signatures.float32.npy',mmap_mode='r')
 rows=[]
 for method in hi:
  ci,cs,csp=merge_top(hi[method],hs[method],mi[method],ms[method])
  for refset,inds,scores,species in [('human',hi[method],hs[method],np.zeros_like(hi[method],np.int8)),('mouse',mi[method],ms[method],np.ones_like(mi[method],np.int8)),('combined',ci,cs,csp)]:
   nref={'human':len(human),'mouse':len(mouse),'combined':len(human)+len(mouse)}[refset]
   for qi in range(len(qm)):
    q=qm.iloc[qi]; dom=np.argsort(np.abs(allh[qi]))[::-1][:10]
    for rank,(ri,score,sp) in enumerate(zip(inds[qi],scores[qi],species[qi]),1):
     md=(mouse if sp else human).iloc[int(ri)]; rv=np.asarray(mref[int(ri)] if sp else href[int(ri)],np.float32);con=l2_rows(allh[qi:qi+1])[0]*l2_rows(rv[None])[0];tc=np.argsort(con)[::-1][:6]
     rows.append({'query_index':qi,'query_type':q.query_type,'group_id':q.group_id,'condition':q.condition,'osdr_sample_id':q.sample_id,'study':q.study,'tissue':q.tissue,'method':method,'reference_set':refset,'rank':rank,'score':float(score),'rank_percentile':1-(rank-.5)/nref,'reference_species':'mouse' if sp else 'human','reference_row':int(ri),'combined_row':int(ri+(len(human) if sp else 0)),'gsm':md.sample_id,'gse':md.series_id,'title':md.title,'source_name':md.source_name_ch1,'characteristics':md.characteristics_ch1,'dominant_query_hallmarks':'; '.join(f'{names[j]}={"+" if allh[qi,j]>=0 else "-"}' for j in dom),'dominant_shared_hallmarks':'; '.join(f'{names[j]}={con[j]:.4f}' for j in tc)})
 hits=pd.DataFrame(rows);hits.to_parquet(OUT/'individual_and_centroid_top100_all_references.parquet',index=False)
 # Deterministic representative: lexicographically first valid group and sample per condition within each study.
 reps=[]
 for study,g in man.groupby('study',sort=True):
  valid=[x for x,z in g.groupby('group_id') if {'FLT','GC'}<=set(z.condition)] ; gid=sorted(valid)[0];z=g[g.group_id==gid]
  for cond in ['FLT','GC']:
   r=z[z.condition==cond].sort_values(['sample_id','matrix_row']).iloc[0];reps.append({'study':study,'group_id':gid,'condition':cond,'sample_id':r.sample_id,'matrix_row':r.matrix_row,'selection_rule':'lexicographically first eligible group_id, then sample_id'})
 reps=pd.DataFrame(reps);reps.to_csv(OUT/'deterministic_study_representatives.csv',index=False)
 rh=hits[(hits.query_type=='individual')&hits.osdr_sample_id.isin(reps.sample_id)];rh.to_parquet(OUT/'representative_sample_top100.parquet',index=False)
 summary=(rh[rh.reference_set=='combined'].groupby(['study','osdr_sample_id','condition','method'])
          .agg(mouse_fraction_top100=('reference_species',lambda x:(x=='mouse').mean()),human_fraction_top100=('reference_species',lambda x:(x=='human').mean()),unique_gse_top100=('gse','nunique')).reset_index());summary.to_csv(OUT/'representative_combined_species_composition.csv',index=False)
 # Combined-reference FLT-GC differential retrieval; exact prior design/test unchanged.
 ind=hits[(hits.query_type=='individual')&(hits.reference_set=='combined')];rng=np.random.default_rng(20260925);out=[]
 for (gid,method),block in ind.groupby(['group_id','method']):
  labs=block[['osdr_sample_id','condition']].drop_duplicates().set_index('osdr_sample_id').condition;samps=labs.index.tolist();la=labs.to_numpy();nf=(la=='FLT').sum();cand=block.combined_row.value_counts().head(300).index.to_numpy();mat=np.zeros((len(samps),len(cand)),np.float32);sm={s:i for i,s in enumerate(samps)};rm={r:i for i,r in enumerate(cand)}
  for r in block.itertuples():
   if r.combined_row in rm: mat[sm[r.osdr_sample_id],rm[r.combined_row]]=1/np.log2(r.rank+1)
  obs=mat[la=='FLT'].mean(0)-mat[la=='GC'].mean(0);perm=np.empty((1000,len(cand)),np.float32)
  for pi in range(1000):
   take=rng.permutation(len(samps));perm[pi]=mat[take[:nf]].mean(0)-mat[take[nf:]].mean(0)
  p=(1+(np.abs(perm)>=np.abs(obs)).sum(0))/1001;qv=bh(p)
  for rr,de,pv,q in zip(cand,obs,p,qv):
   sub=block[block.combined_row==rr];r=sub.iloc[0];out.append({'group_id':gid,'method':method,'combined_row':rr,'reference_species':r.reference_species,'gsm':r.gsm,'gse':r.gse,'title':r.title,'source_name':r.source_name,'characteristics':r.characteristics,'rrf_delta_flt_minus_gc':de,'permutation_p':pv,'bh_q':q,'flt_replicate_prevalence':sub[sub.condition=='FLT'].osdr_sample_id.nunique()/nf,'gc_replicate_prevalence':sub[sub.condition=='GC'].osdr_sample_id.nunique()/(len(samps)-nf)})
 pd.DataFrame(out).to_parquet(OUT/'combined_flt_vs_gc_differential_retrieval.parquet',index=False)
 save_json(OUT/'provenance.json',{'human_reference_samples':len(human),'mouse_reference_samples':len(mouse),'combined_reference_samples':len(human)+len(mouse),'individual_queries':len(man),'centroid_queries':len(cm),'reference_sets':['human','mouse','combined'],'methods':list(hi),'top_k':100,'representative_rule':'lexicographically first eligible group_id, then sample_id; independent of retrieval','label_permutations':1000,'bridge_inference_rerun':False,'hallmark_head_retrained':False,'runtime_seconds':time.time()-started})
 print('complete',time.time()-started,flush=True)
if __name__=='__main__':main()
