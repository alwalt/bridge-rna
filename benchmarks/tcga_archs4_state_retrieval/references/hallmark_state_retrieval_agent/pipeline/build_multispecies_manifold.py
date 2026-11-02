#!/usr/bin/env python3
"""Deterministic Hallmark manifold for all compatible human+mouse ARCHS4 samples."""
from __future__ import annotations
import re,time
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE,trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from common import RESULTS,WORK,save_json

OUT=RESULTS/'archs4_multispecies/manifold';OW=WORK/'archs4_multispecies';FIG=OUT/'figures';SEED=20260925
LABELS=[('brain',r'brain|cortex|cerebell|hippocamp|neural|neuron'),('liver',r'liver|hepatic|hepatocyte'),('skeletal muscle',r'skeletal muscle|quadriceps|gastrocnemius|soleus|tibialis|digitorum|vastus'),('adipose',r'adipose|adipocyte|white fat|brown fat'),('lung',r'lung|pulmonary|bronch'),('kidney',r'kidney|renal'),('blood/immune',r'blood|pbmc|lymphocyte|leukocyte|monocyte|macrophage|bone marrow|spleen|thymus'),('heart',r'heart|cardiac|ventric'),('skin',r'skin|dermal|cutaneous|melanocyte'),('breast/mammary',r'breast|mammary'),('colon/intestine',r'colon|intestin|colorectal'),('pancreas',r'pancrea'),('reproductive',r'ovary|ovarian|uterus|uterine|testis|testicular|prostate|placenta')]
def label(row):
 s=' | '.join(str(row.get(c,'')) for c in ['title','source_name_ch1','characteristics_ch1']).lower()
 for n,p in LABELS:
  if re.search(p,s):return n
 return 'unknown'
def metrics(x,tissue,species):
 nn=NearestNeighbors(n_neighbors=16,metric='cosine',n_jobs=-1).fit(x);idx=nn.kneighbors(return_distance=False)[:,1:];tp=np.mean(tissue[idx]==tissue[:,None],axis=1);sp=np.mean(species[idx]==species[:,None],axis=1);mix=[]
 for t in sorted(set(tissue)-{'unknown'}):
  q=tissue==t;mix.append({'tissue':t,'n':int(q.sum()),'cross_species_neighbor_fraction':float(np.mean(species[idx[q]]!=species[q,None]))})
 take=np.random.default_rng(SEED).choice(len(x),min(20000,len(x)),replace=False)
 return {'tissue_silhouette_cosine':float(silhouette_score(x[take],tissue[take],metric='cosine')),'species_silhouette_cosine':float(silhouette_score(x[take],species[take],metric='cosine')),'tissue_nn_purity':float(tp.mean()),'species_nn_purity':float(sp.mean())},mix
def plot(coords,frame,prefix):
 for col,pal in [('tissue',None),('species',{'human':'#377eb8','mouse':'#e41a1c'})]:
  fig,ax=plt.subplots(figsize=(11,9)); values=frame[col].astype(str).to_numpy();cats=sorted(set(values));colors=plt.cm.tab20(np.linspace(0,1,len(cats))) if pal is None else [pal[c] for c in cats]
  for c,color in zip(cats,colors):
   q=values==c;ax.scatter(coords[q,0],coords[q,1],c=[color],label=c,s=1,alpha=.35,linewidths=0,rasterized=True)
  ax.set(title=f'{prefix.upper()} colored by {col}',xlabel=f'{prefix.upper()}1',ylabel=f'{prefix.upper()}2');ax.legend(markerscale=5,bbox_to_anchor=(1.02,1),loc='upper left',fontsize=7);fig.tight_layout();fig.savefig(FIG/f'{prefix}_{col}.png',dpi=300,bbox_inches='tight');fig.savefig(FIG/f'{prefix}_{col}.pdf',bbox_inches='tight');plt.close(fig)
def main():
 OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(exist_ok=True);start=time.time();human=pd.read_parquet(RESULTS/'archs4/sample_metadata.parquet');human['species']='human';mouse=pd.read_parquet(RESULTS/'archs4_multispecies/mouse_sample_metadata.parquet');mouse['species']='mouse';meta=pd.concat([human,mouse],ignore_index=True);meta['tissue']=meta.apply(label,axis=1);meta[['sample_id','series_id','species','tissue','global_index']].to_parquet(OUT/'manifold_sample_manifest.parquet',index=False)
 hh=np.load(WORK/'archs4_hallmark_signatures.float32.npy',mmap_mode='r');mh=np.load(OW/'mouse_hallmark_signatures.float32.npy',mmap_mode='r');hall=np.lib.format.open_memmap(OW/'combined_hallmark_signatures.float32.npy',mode='w+',dtype=np.float32,shape=(len(meta),50));hall[:len(hh)]=hh;hall[len(hh):]=mh;hall.flush()
 # deterministic stratified subset shared by UMAP, t-SNE, and metrics
 rng=np.random.default_rng(SEED);sel=[]
 for _,g in meta.groupby(['species','tissue'],sort=True): sel.extend(rng.choice(g.index,min(len(g),max(50,int(30000*len(g)/len(meta)))),replace=False))
 sel=np.array(sorted(set(sel))); # fill deterministically to 30k
 remaining=np.setdiff1d(np.arange(len(meta)),sel,assume_unique=True)
 if len(sel)<30000: sel=np.sort(np.concatenate([sel,rng.choice(remaining,30000-len(sel),replace=False)]))
 if len(sel)>30000: sel=np.sort(rng.choice(sel,30000,replace=False))
 sub=np.asarray(hall[sel]);sub=(sub-sub.mean(0))/np.maximum(sub.std(0),1e-8);sf=meta.iloc[sel].reset_index(drop=True)
 reducer=umap.UMAP(n_neighbors=30,min_dist=.2,n_components=2,metric='cosine',random_state=SEED,transform_seed=SEED,n_jobs=1).fit(sub);coords=np.empty((len(meta),2),np.float32)
 for a in range(0,len(meta),100000):
  b=min(a+100000,len(meta));x=np.asarray(hall[a:b]);x=(x-np.asarray(hall[sel]).mean(0))/np.maximum(np.asarray(hall[sel]).std(0),1e-8);coords[a:b]=reducer.transform(x);print('UMAP transform',b,len(meta),flush=True)
 np.save(OW/'combined_hallmark_umap_coordinates.npy',coords);pd.DataFrame({'sample_id':meta.sample_id,'umap1':coords[:,0],'umap2':coords[:,1]}).to_parquet(OUT/'umap_coordinates.parquet',index=False);plot(coords,meta,'umap')
 ts=TSNE(n_components=2,perplexity=30,learning_rate='auto',init='pca',max_iter=1000,random_state=SEED,method='barnes_hut').fit_transform(sub);np.save(OW/'hallmark_tsne_subset_coordinates.npy',ts);pd.DataFrame({'sample_id':sf.sample_id,'species':sf.species,'tissue':sf.tissue,'tsne1':ts[:,0],'tsne2':ts[:,1]}).to_parquet(OUT/'tsne_subset_coordinates.parquet',index=False);plot(ts,sf,'tsne')
 hm,hmix=metrics(sub,sf.tissue.to_numpy(),sf.species.to_numpy());pca=PCA().fit(np.asarray(hall));ev=pca.explained_variance_ratio_;er=float(np.exp(-np.sum(ev[ev>0]*np.log(ev[ev>0]))));pd.DataFrame({'component':np.arange(1,51),'explained_variance_ratio':ev,'cumulative':np.cumsum(ev)}).to_csv(OUT/'hallmark_pca_spectrum.csv',index=False);load=pd.DataFrame(pca.components_[:10].T,index=np.load(RESULTS/'frozen/frozen_gtex_hallmark_probe.npz')['hallmarks'].astype(str),columns=[f'PC{i}' for i in range(1,11)]);load.to_csv(OUT/'hallmark_manifold_axis_loadings.csv')
 # Bridge comparison on identical subset from authoritative float16 cache.
 import json
 spec=json.loads((Path('/home/walt/bridge-rna/embeddings/archs4/embedding_manifest.json')).read_text());loc=pd.read_parquet('/home/walt/bridge-rna/embeddings/archs4/sample_locations.parquet');raw=np.memmap('/home/walt/bridge-rna/embeddings/archs4/sample_embeddings.float16.mmap',mode='r',dtype=np.float16,shape=(spec['total_samples'],512));gidx=meta.global_index.to_numpy(int);bridge=np.asarray(raw[gidx[sel]],np.float32);bm,bmix=metrics(bridge,sf.tissue.to_numpy(),sf.species.to_numpy());bp=PCA().fit(bridge);bev=bp.explained_variance_ratio_;pd.DataFrame({'component':np.arange(1,513),'explained_variance_ratio':bev,'cumulative':np.cumsum(bev)}).to_csv(OUT/'bridge512_subset_pca_spectrum.csv',index=False)
 pd.concat([pd.DataFrame(hmix).assign(space='Hallmark-50'),pd.DataFrame(bmix).assign(space='Bridge-512')]).to_csv(OUT/'cross_species_mixing_by_tissue.csv',index=False)
 pd.DataFrame([{'space':'Hallmark-50',**hm},{'space':'Bridge-512',**bm}]).to_csv(OUT/'manifold_metrics.csv',index=False)
 save_json(OUT/'provenance.json',{'samples':len(meta),'human':int((meta.species=='human').sum()),'mouse':int((meta.species=='mouse').sum()),'tissue_label_coverage':float((meta.tissue!='unknown').mean()),'subset_samples':len(sel),'subset_selection':'seeded proportional stratification by species x conservative tissue label; minimum 50 per stratum; deterministic fill/truncate to 30,000','umap':{'fit_subset':30000,'transformed_all':len(meta),'n_neighbors':30,'min_dist':.2,'metric':'cosine','seed':SEED},'tsne':{'subset':30000,'perplexity':30,'iterations':1000,'seed':SEED},'hallmark_effective_rank_entropy':er,'hallmark_metrics':hm,'bridge_metrics':bm,'runtime_seconds':time.time()-start})
if __name__=='__main__':main()
