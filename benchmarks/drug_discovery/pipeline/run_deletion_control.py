#!/usr/bin/env python3
"""Native mask-token deletion control for primary Bridge modules."""
import argparse,json
import numpy as np,pandas as pd,torch
import run_bridge as rb

def scores(model,x,direction,device,mask=None):
 out=[]; target=torch.as_tensor(direction,device=device).view(1,-1)
 with torch.no_grad():
  for start in range(0,len(x),2):
   v=torch.as_tensor(np.array(x[start:start+2],copy=True),device=device)
   if mask is not None:v[:,mask]=-10.0
   out.extend(rb.score(model,v,target.expand(len(v),-1)).cpu().numpy())
 return np.array(out)

def main():
 p=argparse.ArgumentParser();p.add_argument("--device",default="cuda:0");p.add_argument("--random-panels",type=int,default=5);a=p.parse_args();model,device,genes=rb.load_model(a.device);lookup={g:i for i,g in enumerate(genes)}
 members=pd.read_parquet(rb.HERE/"results/manifests/contrast_members.parquet");vec=pd.read_parquet(rb.OUT/"condition_vectors.parquet");ranks=pd.read_parquet(rb.OUT/"study_gene_rankings.parquet");rng=np.random.default_rng(rb.CONFIG["seed"]);rows=[]
 for gse,gm in members.groupby("dataset",sort=True):
  x=np.load(rb.PREP/f"{gse}_log1p_tpm.npy",mmap_mode="r"); mapping=pd.read_csv(rb.PREP/f"{gse}_matrix_rows.csv").set_index("sample_id").matrix_row.to_dict();top=ranks[(ranks.dataset.eq(gse))&(ranks["rank"]<=500)].gene.tolist(); topidx=np.array([lookup[g] for g in top]);mean=np.asarray(x).mean(0);bins=pd.qcut(pd.Series(mean,index=np.arange(len(genes))).rank(method="first"),10,labels=False);want=bins.loc[topidx].value_counts();panels=[("top",0,topidx)]
  for rep in range(a.random_panels):
   panel=[]
   for b,n in want.items():panel.extend(rng.choice(bins.index[bins.eq(b)],int(n),replace=False))
   panels.append(("random",rep,np.array(panel)))
  for panel_type,rep,panel in panels:
   effects=[]
   for cid,cm in gm.groupby("contrast_id",sort=True):
    direction=vec[vec.contrast_id.eq(cid)].sort_values("dimension").value.to_numpy(np.float32);cm=cm.reset_index(drop=True);values=np.stack([x[mapping[s]] for s in cm.sample_id]); original=scores(model,values,direction,device);masked=scores(model,values,direction,device,panel)
    oe,_=rb.contrast_contributions(cid,original[:,None],cm);me,_=rb.contrast_contributions(cid,masked[:,None],cm);effects.append((float(oe.mean()),float(me.mean())))
   original=float(np.mean([q[0] for q in effects]));masked=float(np.mean([q[1] for q in effects]));rows.append({"dataset":gse,"panel_type":panel_type,"replicate":rep,"genes_masked":len(panel),"original_axis_effect":original,"masked_axis_effect":masked,"absolute_score_change":abs(masked-original),"fraction_signal_remaining":masked/original if abs(original)>1e-12 else np.nan})
  rb.say(f"deletion control {gse}")
 out=pd.DataFrame(rows);out.to_csv(rb.OUT/"deletion_control.csv",index=False);summary=out.groupby(["dataset","panel_type"],as_index=False).agg(mean_absolute_change=("absolute_score_change","mean"),sd_absolute_change=("absolute_score_change","std"),mean_fraction_remaining=("fraction_signal_remaining","mean"));summary.to_csv(rb.OUT/"deletion_control_summary.csv",index=False);(rb.OUT/"deletion_control_provenance.json").write_text(json.dumps({"mask_token":-10,"module_size":500,"random_panels":a.random_panels,"random_matching":"dataset mean-expression decile"},indent=2)+"\n")
if __name__=="__main__":main()
