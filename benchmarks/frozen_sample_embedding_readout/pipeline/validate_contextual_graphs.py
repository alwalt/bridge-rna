#!/usr/bin/env python3
"""General validation of frozen contextual-gene graphs; no RR1/RR3 use."""
from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np,pandas as pd
from numba import njit
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score,f1_score,balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import networkx as nx

HERE=Path(__file__).resolve().parents[1];REPO=HERE.parents[1];OUT=HERE/'results/graph_fingerprint/general_validation';WORK=HERE/'work/graph_fingerprint';PREV=REPO/'benchmarks/library_prep_disentanglement/results/task4_pca_vs_bridgerna_generalization';G=15165;SEED=20260907
def say(x):print(f'[{time.strftime("%F %T")}] {x}',flush=True)
def load(name):
 d=WORK/name;m=pd.read_parquet(d/'manifest.parquet');n=np.load(d/'neighbors_top20.uint16.npy',mmap_mode='r');w=np.load(d/'weights_top20.float16.npy',mmap_mode='r');return m,n,w
@njit
def union_profile(nbr,wei,k,mutual=False):
 codes=np.empty(G*k,np.int64);vals=np.empty(G*k,np.float32);z=0
 for i in range(G):
  for a in range(k):
   j=int(nbr[i,a]);ok=True
   if mutual:
    ok=False
    for b in range(k):
     if int(nbr[j,b])==i:ok=True;break
   if ok:
    codes[z]=min(i,j)*G+max(i,j);vals[z]=float(wei[i,a]);z+=1
 order=np.argsort(codes[:z]);c=codes[:z][order];v=vals[:z][order];oc=np.empty(z,np.int64);ov=np.empty(z,np.float32);q=0;i=0
 while i<z:
  code=c[i];best=v[i];i+=1
  while i<z and c[i]==code:
   if v[i]>best:best=v[i]
   i+=1
  oc[q]=code;ov[q]=best;q+=1
 return oc[:q],ov[:q]
@njit
def pair(c1,w1,c2,w2):
 i=j=inter=0;dot=s1=s2=sq1=sq2=mn=mx=0.
 while i<len(c1) or j<len(c2):
  if j==len(c2) or (i<len(c1) and c1[i]<c2[j]):a=w1[i];b=0.;i+=1
  elif i==len(c1) or c2[j]<c1[i]:a=0.;b=w2[j];j+=1
  else:a=w1[i];b=w2[j];i+=1;j+=1;inter+=1
  dot+=a*b;s1+=a;s2+=b;sq1+=a*a;sq2+=b*b;mn+=min(a,b);mx+=max(a,b)
 n=len(c1)+len(c2)-inter;cov=dot-s1*s2/n;den=((sq1-s1*s1/n)*(sq2-s2*s2/n))**.5;return inter/n,cov/den if den>0 else np.nan,mn/mx if mx else np.nan
@njit
def neighborhood_jaccard(a,b,k):
 total=0.
 for g in range(G):
  inter=0
  for i in range(k):
   for j in range(k):
    if a[g,i]==b[g,j]:inter+=1
  total+=inter/(2*k-inter)
 return total/G
def graph(n,w,k,mutual=False):return union_profile(np.asarray(n),np.asarray(w,dtype=np.float32),k,mutual)
def degree_features(n,w,k):
 c,v=graph(n,w,k);a=c//G;b=c%G;deg=np.bincount(np.r_[a,b],minlength=G);strength=np.bincount(np.r_[a,b],weights=np.r_[v,v],minlength=G);return np.r_[deg,strength].astype('float32'),c,v
def tech_retrieval():
 m,n,w=load('tcell');rows=[];summary=[]
 for k in [5,10,20]:
  for kind in ['union','mutual']:
   gs=[graph(n[i],w[i],k,kind=='mutual') for i in range(len(m))];metrics={q:np.full((len(m),len(m)),np.nan) for q in ['edge_jaccard','weight_correlation','weighted_overlap','neighborhood_jaccard']}
   for i in range(len(m)):
    for j in range(i+1,len(m)):
     z=pair(*gs[i],*gs[j]);metrics['edge_jaccard'][i,j]=metrics['edge_jaccard'][j,i]=z[0];metrics['weight_correlation'][i,j]=metrics['weight_correlation'][j,i]=z[1];metrics['weighted_overlap'][i,j]=metrics['weighted_overlap'][j,i]=z[2]
     if kind=='union':metrics['neighborhood_jaccard'][i,j]=metrics['neighborhood_jaccard'][j,i]=neighborhood_jaccard(n[i],n[j],k)
   for metric,S in metrics.items():
    if kind=='mutual' and metric=='neighborhood_jaccard':continue
    ranks=[]
    for i in range(len(m)):
     cand=np.flatnonzero(m.library_prep.to_numpy()!=m.library_prep.iloc[i]);target=cand[m.pair_id.iloc[cand].to_numpy()==m.pair_id.iloc[i]][0];rank=int(np.flatnonzero(cand[np.argsort(-S[i,cand])]==target)[0])+1;ranks.append(rank);rows.append({'k':k,'graph':kind,'metric':metric,'query':m.sample_id.iloc[i],'target':m.sample_id.iloc[target],'rank':rank,'similarity':S[i,target]})
    summary.append({'k':k,'graph':kind,'metric':metric,'queries':len(ranks),'R@1':np.mean(np.array(ranks)<=1),'R@5':np.mean(np.array(ranks)<=5),'median_rank':np.median(ranks)})
 pd.DataFrame(rows).to_parquet(OUT/'technical_retrieval_queries.parquet',index=False);pd.DataFrame(summary).to_csv(OUT/'technical_retrieval_summary.csv',index=False)
def communities_spectra():
 m,n,w=load('tcell');cp=WORK/'tcell_community_labels_k10.int16.npy';sp=WORK/'tcell_spectrum_k10.npy'
 if cp.exists() and sp.exists():labels=np.load(cp);spectra=np.load(sp)
 else:
  labels=np.empty((len(m),G),dtype='int16');spectra=np.empty((len(m),20),dtype='float32')
  for i in range(len(m)):
   c,v=graph(n[i],w[i],10);a=(c//G).astype(int);b=(c%G).astype(int);gr=nx.Graph();gr.add_nodes_from(range(G));gr.add_weighted_edges_from(zip(a,b,v.astype(float)));parts=nx.community.louvain_communities(gr,weight='weight',resolution=1,seed=SEED);lab=np.empty(G,dtype='int16')
   for j,p in enumerate(parts):lab[list(p)]=j
   labels[i]=lab;A=csr_matrix((np.r_[v,v],(np.r_[a,b],np.r_[b,a])),shape=(G,G));deg=np.asarray(A.sum(1)).ravel();inv=np.zeros(G);inv[deg>0]=1/np.sqrt(deg[deg>0]);N=A.multiply(inv[:,None]).multiply(inv[None,:]);spectra[i]=np.sort(eigsh(N,k=20,which='LM',return_eigenvectors=False))
   if (i+1)%5==0:say(f'community/spectrum {i+1}/{len(m)}')
  np.save(cp,labels);np.save(sp,spectra)
 rows=[]
 for i in range(len(m)):
  for j in range(i+1,len(m)):
   rows.append({'sample_A':m.sample_id.iloc[i],'sample_B':m.sample_id.iloc[j],'same_pair':m.pair_id.iloc[i]==m.pair_id.iloc[j],'community_ARI':adjusted_rand_score(labels[i],labels[j]),'community_NMI':normalized_mutual_info_score(labels[i],labels[j]),'spectral_distance':np.linalg.norm(spectra[i]-spectra[j])})
 pd.DataFrame(rows).to_parquet(OUT/'technical_community_spectral_similarity.parquet',index=False);pd.DataFrame({'sample_id':m.sample_id,'communities':[len(np.unique(x)) for x in labels]}).to_csv(OUT/'technical_graph_communities.csv',index=False)
def tissue():
 m,n,w=load('tissue');path=WORK/'tissue_degree_strength_k10.float16.npy'
 if not path.exists():
  X=np.lib.format.open_memmap(path,mode='w+',dtype='float16',shape=(len(m),2*G))
  for i in range(len(m)):X[i]=degree_features(n[i],w[i],10)[0].astype('float16');
  X.flush()
 X=np.load(path,mmap_mode='r');y=m.tissue.astype(str).to_numpy();groups=m.gse.astype(str).to_numpy();rows=[]
 for fold,(tr,te) in enumerate(GroupKFold(5).split(np.arange(len(m)),y,groups)):
  sc=StandardScaler().fit(X[tr]);mod=RidgeClassifier(alpha=1,solver='lsqr',class_weight='balanced').fit(sc.transform(X[tr]),y[tr]);p=mod.predict(sc.transform(X[te]));rows.append({'fold':fold,'macro_f1':f1_score(y[te],p,average='macro',zero_division=0),'balanced_accuracy':balanced_accuracy_score(y[te],p)})
 pd.DataFrame(rows).to_csv(OUT/'tissue_study_disjoint_graph_classifier.csv',index=False)
 # Prespecified, study-separated same/different pairs for exact graph metrics.
 rng=np.random.default_rng(SEED);pairs=[]
 for same in [True,False]:
  while sum(x[2]==same for x in pairs)<500:
   i,j=rng.choice(len(m),2,replace=False)
   if groups[i]!=groups[j] and (y[i]==y[j])==same:pairs.append((i,j,same))
 out=[]
 for k in [5,10,20]:
  for i,j,same in pairs:
   a=graph(n[i],w[i],k);b=graph(n[j],w[j],k);z=pair(*a,*b);out.append({'k':k,'sample_A':i,'sample_B':j,'same_tissue':same,'edge_jaccard':z[0],'weight_correlation':z[1],'weighted_overlap':z[2],'neighborhood_jaccard':neighborhood_jaccard(n[i],n[j],k)})
 pd.DataFrame(out).to_parquet(OUT/'tissue_pair_similarity.parquet',index=False)
def perturbations():
 m,n,w=load('exercise');members=pd.read_parquet(REPO/'benchmarks/cross_species_exercise_response/results/contrast_members.parquet');loc={x:i for i,x in enumerate(m.GSM.astype(str))};F=[]
 for i in range(len(m)):F.append(degree_features(n[i],w[i],10)[0])
 F=np.asarray(F);rows=[];effects={}
 for cid,q in members.groupby('contrast_id'):
  q=q[q.GSM.astype(str).isin(loc)];a=np.array([loc[x] for x in q[q.role.str.contains('post|exercise',case=False,regex=True)].GSM.astype(str)]);b=np.array([loc[x] for x in q[~q.index.isin(q[q.role.str.contains('post|exercise',case=False,regex=True)].index)].GSM.astype(str)])
  if len(a) and len(b):effects[cid]=F[a].mean(0)-F[b].mean(0);rows.append({'contrast_id':cid,'n_treatment':len(a),'n_control':len(b)})
 ids=list(effects);sim=np.array([[np.corrcoef(effects[a],effects[b])[0,1] for b in ids] for a in ids]);pd.DataFrame(sim,index=ids,columns=ids).to_csv(OUT/'perturbation_graph_response_similarity.csv');pd.DataFrame(rows).to_csv(OUT/'perturbation_contrasts.csv',index=False)
def null_and_function():
 m,n,w=load('tissue');genes=pd.read_csv(REPO/'data/ensembl/canonical_genes.csv').sort_values('token_id').gene_symbol.astype(str).str.upper().tolist();sources={'Hallmark':[set(str(x).upper() for x in v) for v in json.loads((REPO/'data/gsea/hallmark_gene_sets.json').read_text()).values()]};groot=REPO/'benchmarks/cross_species_exercise_response/results/per_study_ranked_gsea'
 for label,file in [('GO_BP','GO_Biological_Process_2026.gmt'),('Reactome','Reactome_Pathways_2024.gmt')]:
  sets=[]
  for line in open(groot/file):
   z=line.rstrip().split('\t');sets.append(set(x.upper() for x in z[2:]))
  sources[label]=sets
 memberships={source:{g:{i for i,s in enumerate(sets) if g in s} for g in genes} for source,sets in sources.items()};rng=np.random.default_rng(SEED);rows=[]
 for si in rng.choice(len(m),50,replace=False):
  c,v=graph(n[si],w[si],10);pick=rng.choice(len(c),min(10000,len(c)),replace=False);a=c[pick]//G;b=c[pick]%G;perm=rng.permutation(G)
  for source,membership in memberships.items():
   observed=np.mean([bool(membership[genes[i]]&membership[genes[j]]) for i,j in zip(a,b)]);shuf=np.mean([bool(membership[genes[perm[i]]]&membership[genes[perm[j]]]) for i,j in zip(a,b)]);rows.append({'sample':si,'source':source,'observed_shared_membership':observed,'gene_label_shuffle':shuf})
 pd.DataFrame(rows).to_csv(OUT/'functional_edge_null.csv',index=False)
def main():
 OUT.mkdir(parents=True,exist_ok=True);say('general validation begins; RR1/RR3 not loaded');tech_retrieval();say('technical retrieval complete');communities_spectra();say('communities/spectra complete');tissue();say('tissue complete');perturbations();say('perturbations complete');null_and_function();say('null/function complete');(OUT/'provenance.json').write_text(json.dumps({'primary_graph':'layer12 contextual cosine directed kNN k=10; union symmetrization','sensitivities_k':[5,20],'mutual_knn':True,'parameter_selection':'prespecified without RR1/RR3','community_algorithm':'NetworkX Louvain, resolution=1, fixed seed','spectrum':'20 eigenvalues of symmetric normalized weighted adjacency, largest magnitude','PPI_regulatory':'not run; no local authoritative resource available','limitations':['degree/strength classifier is a derived graph profile','exercise role parser is saved for audit before interpretation']},indent=2)+'\n');say('GENERAL VALIDATION COMPLETE')
if __name__=='__main__':main()
