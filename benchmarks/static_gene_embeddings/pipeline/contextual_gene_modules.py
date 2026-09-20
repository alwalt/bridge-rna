#!/usr/bin/env python3
"""Cohort-mean 512-D contextual gene modules and TPM-dependence tests."""
from __future__ import annotations
import argparse,json,time
from itertools import combinations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np,pandas as pd,torch,umap
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score,silhouette_score
from matched_gtex_tcga import ROOT,HERE,OUT as MATCH_OUT,GTEX_X,TCGA_X,LAYERS,load_model
from run_static_embeddings import enrich,GMTS

OUT=HERE/'results/contextual_gene_modules';WORK=HERE/'work/contextual_gene_modules';VOCAB=ROOT/'data/ensembl/canonical_genes.csv';SEEDS=[42,43,44,45,46]

class Log:
 def __init__(self,p):self.p=p;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('')
 def __call__(self,s):
  z=f'[{pd.Timestamp.utcnow().isoformat()}] {s}';print(z,flush=True)
  with self.p.open('a') as f:f.write(z+'\n')

def extract(manifest,device,log):
 genes=pd.read_csv(VOCAB).sort_values('token_id');g=len(genes);ids=torch.arange(g,device=device);model=load_model(device);arrays={'GTEx':np.load(GTEX_X,mmap_mode='r'),'TCGA':np.load(TCGA_X,mmap_mode='r')};started=time.time()
 for dataset in ['GTEx','TCGA']:
  sub=manifest[manifest.dataset.eq(dataset)];sums={l:np.zeros((g,512),np.float64) for l in LAYERS};expr=np.zeros(g,np.float64)
  for i,row in enumerate(sub.itertuples(index=False),1):
   value=np.array(arrays[dataset][int(row.matrix_row)],copy=True);expr+=value;v=torch.as_tensor(value[None],device=device);states={'L0_static':model.gene_embedding(ids)}
   with torch.no_grad(),torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
    h=model.gene_embedding(ids).unsqueeze(0)+model.ree(v)
    for li,layer in enumerate(model.layers,1):
     h=layer(h)
     for key,num in LAYERS.items():
      if li==num:states[key]=h[0]
   for l,z in states.items():sums[l]+=z.float().cpu().numpy()
   if i%50==0:log(f'extract {dataset} {i}/{len(sub)} elapsed={(time.time()-started)/60:.1f}m')
  np.save(WORK/f'{dataset}_mean_log1p_tpm.npy',(expr/len(sub)).astype('float32'))
  for l,z in sums.items():np.save(WORK/f'{dataset}_{l}_mean_tokens.float32.npy',(z/len(sub)).astype('float32'))

def eta_squared(values,labels):
 total=np.square(values-values.mean()).sum();between=sum((values[labels==c].size)*np.square(values[labels==c].mean()-values.mean()) for c in np.unique(labels));return float(between/total)

def tpm_association(x,tpm,labels,nbr,rng):
 norm=np.linalg.norm(x,axis=1);rho_norm=spearmanr(norm,tpm).statistic;n=200000;a=rng.integers(0,len(x),n);b=rng.integers(0,len(x),n);u=x/np.maximum(norm[:,None],1e-12);sim=(u[a]*u[b]).sum(1);expr_sim=-np.abs(tpm[a]-tpm[b]);rho_pair=spearmanr(sim,expr_sim).statistic
 nn_diff=np.mean(np.abs(tpm[:,None]-tpm[nbr]));random_diff=np.mean(np.abs(tpm[a]-tpm[b]));bins=pd.qcut(pd.Series(tpm),10,labels=False,duplicates='drop').to_numpy();return {'norm_tpm_spearman':rho_norm,'pair_cosine_expression_similarity_spearman':rho_pair,'cluster_tpm_eta_squared':eta_squared(tpm,labels),'cluster_expression_bin_nmi':normalized_mutual_info_score(labels,bins),'neighbor_tpm_abs_difference':nn_diff,'random_pair_tpm_abs_difference':random_diff,'neighbor_expression_difference_ratio':nn_diff/random_diff}

def exact_neighbors(x,k=25,batch=256):
 dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');z=torch.nn.functional.normalize(torch.as_tensor(x,device=dev),dim=1);out=np.empty((len(x),k),np.int32)
 for a in range(0,len(x),batch):
  s=z[a:a+batch]@z.T;r=torch.arange(len(s),device=dev);s[r,torch.arange(a,a+len(s),device=dev)]=-torch.inf;out[a:a+len(s)]=torch.topk(s,k).indices.cpu().numpy()
 return out

def visualize(means,tpms,assignments,genes,out):
 rows=[]
 for layer in LAYERS:
  pair=np.vstack([means[('GTEx',layer)],means[('TCGA',layer)]]);p=PCA(50,random_state=42).fit_transform(pair);ts=TSNE(2,perplexity=30,init='pca',learning_rate='auto',max_iter=1000,random_state=42).fit_transform(p);um=umap.UMAP(n_neighbors=30,min_dist=.1,metric='cosine',random_state=42).fit_transform(p)
  for di,d in enumerate(['GTEx','TCGA']):
   sl=slice(di*len(genes),(di+1)*len(genes));rows.append(pd.DataFrame({'dataset':d,'layer':layer,'gene':genes,'cluster':assignments[(d,layer)],'mean_log1p_tpm':tpms[d],'tsne_1':ts[sl,0],'tsne_2':ts[sl,1],'umap_1':um[sl,0],'umap_2':um[sl,1]}))
 q=pd.concat(rows,ignore_index=True);q.to_csv(out/'matched_gene_module_projections.csv',index=False)
 for color,field,cmap in [('clusters','cluster','tab10'),('expression','mean_log1p_tpm','viridis')]:
  fig,axes=plt.subplots(4,4,figsize=(15,14),constrained_layout=True)
  for i,l in enumerate(LAYERS):
   for j,(d,x,y,title) in enumerate([('GTEx','tsne_1','tsne_2','t-SNE'),('TCGA','tsne_1','tsne_2','t-SNE'),('GTEx','umap_1','umap_2','UMAP'),('TCGA','umap_1','umap_2','UMAP')]):
    z=q[(q.dataset==d)&(q.layer==l)];axes[i,j].scatter(z[x],z[y],c=z[field],s=2,cmap=cmap,rasterized=True);axes[i,j].set(title=f'{d} {l} {title}',xticks=[],yticks=[])
  for ext in ['png','pdf']:fig.savefig(out/f'matched_projections_by_{color}.{ext}',dpi=180,bbox_inches='tight')
  plt.close(fig)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda:0');a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True);log=Log(OUT/'run.log');start=time.time();manifest=pd.read_csv(MATCH_OUT/'cohort_manifest.csv');genes=pd.read_csv(VOCAB).sort_values('token_id');names=genes.gene_symbol.astype(str).str.upper().tolist();dev=torch.device(a.device if torch.cuda.is_available() else 'cpu');extract(manifest,dev,log);means={};tpms={d:np.load(WORK/f'{d}_mean_log1p_tpm.npy') for d in ['GTEx','TCGA']};assign={};metrics=[];rng=np.random.default_rng(42)
 for d in ['GTEx','TCGA']:
  for layer in LAYERS:
   x=np.load(WORK/f'{d}_{layer}_mean_tokens.float32.npy');means[(d,layer)]=x;labs=[]
   for seed in SEEDS:labs.append(KMeans(10,random_state=seed,n_init=10,algorithm='lloyd').fit_predict(x))
   aris=[adjusted_rand_score(a,b) for a,b in combinations(labs,2)];labels=labs[0]+1;assign[(d,layer)]=labels;nbr=exact_neighbors(x);row={'dataset':d,'layer':layer,'silhouette':silhouette_score(x,labels,metric='euclidean',sample_size=3000,random_state=42),'stability_ari':np.mean(aris),'min_cluster_size':np.bincount(labels)[1:].min(),'max_cluster_size':np.bincount(labels)[1:].max(),**tpm_association(x,tpms[d],labels,nbr,rng)}
   table=pd.DataFrame({'token_id':genes.token_id,'gene':names,'cluster':labels,'mean_log1p_tpm':tpms[d]});table.to_csv(OUT/f'cluster_assignments_{d}_{layer}.csv',index=False)
   for lib,path in GMTS.items():
    e,_=enrich(table.rename(columns={'gene':'gene'}),lib.upper(),path);e.to_csv(OUT/f'{lib}_enrichment_{d}_{layer}.csv',index=False);row[f'significant_{lib}_terms']=int(e.significant.sum());top=e[e.significant].head(1);row[f'top_{lib}_term']=top.term.iloc[0] if len(top) else '' ;row[f'top_{lib}_fdr']=top.fdr.iloc[0] if len(top) else np.nan
   metrics.append(row);log(f'clustered {d} {layer}')
 pd.DataFrame(metrics).to_csv(OUT/'contextual_gene_module_summary.csv',index=False);visualize(means,tpms,assign,names,OUT)
 prov={'status':'complete','unit':'one cohort-mean 512-D contextual vector per gene','samples':{'GTEx':200,'TCGA':200},'layers':LAYERS,'clustering':{'algorithm':'ordinary Euclidean KMeans on original unnormalized 512-D cohort-mean tokens','k':10,'seeds':SEEDS,'n_init':10},'silhouette':'Euclidean, deterministic 3000-gene sample','enrichment_background':'library-annotated intersection of canonical 15165 model genes; BH within cluster/library','matched_projection':'joint PCA50 then joint t-SNE/UMAP per layer over all 15165 genes in both cohorts; identical parameters','tpm_association':['Spearman embedding norm vs mean log1p TPM','Spearman pair cosine vs negative absolute TPM difference, 200k pairs','cluster TPM eta squared and expression-decile NMI','top25 neighbor TPM difference versus random pairs'],'limitations':['cohort averaging targets shared contextual modules and does not replace existing within-sample clustering','normal tissues and tumors are not paired conditions','visual projections are exploratory'],'elapsed_seconds':time.time()-start};(OUT/'provenance.json').write_text(json.dumps(prov,indent=2)+'\n');log(f'Complete elapsed={(time.time()-start)/60:.1f}m')
if __name__=='__main__':main()
