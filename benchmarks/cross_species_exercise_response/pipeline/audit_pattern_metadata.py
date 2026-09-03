#!/usr/bin/env python3
"""Descriptive metadata audit for fixed exercise Patterns A/intermediate/B."""
from __future__ import annotations
import json, textwrap
from pathlib import Path
import matplotlib.pyplot as plt, pandas as pd

HERE=Path(__file__).resolve().parents[1]; ROOT=HERE.parents[1]
OUT=HERE/'results/pattern_metadata_audit'; OUT.mkdir(parents=True,exist_ok=True)
ORDER=['GSE108643','GSE86931','GSE126962','GSE132520','GSE151066','GSE71972','GSE87748','GSE97718']
PATTERN={**{g:'A' for g in ORDER[:4]},'GSE151066':'Intermediate',**{g:'B' for g in ORDER[5:]}}

def values(z,column):
    vals=[]
    for x in z[column].dropna().astype(str):
      x=x.strip()
      if x and x.lower() not in {'nan','none','na','n/a'} and x not in vals: vals.append(x)
    return '; '.join(vals) if vals else 'Not reported'

def main():
    members=pd.read_parquet(HERE/'results/contrast_members.parquet')
    raw=pd.concat([pd.read_csv(p,sep='\t',dtype=str) for p in sorted((ROOT/'data/geprep').glob('*.txt'))],ignore_index=True)
    raw=raw.rename(columns={'datasets':'GSE'})
    selected=raw[raw.GSM.isin(members.GSM)].merge(members[['GSM','GSE','role','rule','stratum']],on=['GSM','GSE'],validate='one_to_one')
    rows=[]
    for gse in ORDER:
      z=selected[selected.GSE.eq(gse)]; species='human' if z.organism.str.contains('Homo',na=False).any() else 'mouse'
      paired='Paired pre/post subjects' if species=='human' and z['subject id(or sample id)'].nunique()==len(z)//2 else 'Independent exercise/control groups'
      rows.append({'order':ORDER.index(gse)+1,'pattern':PATTERN[gse],'GSE':gse,'species':species,'samples':len(z),'exercise_modality':values(z,'exercise type'),'protocol':values(z,'protocol'),'duration':values(z,'duration'),'intensity':values(z,'intensity'),'post_exercise_time':values(z[z.role.eq('post_exercise')],'biopsy timepoint'),'muscle_site':values(z,'sampling site'),'design':paired,'library_preparation':values(z,'library prepared method'),'extracted_molecule':values(z,'extracted molecule'),'read_layout':values(z,'layout'),'platform':values(z,'platform ID'),'instrument':values(z,'instrument model'),'sex':values(z,'gender'),'training_status':values(z,'fitness')})
    audit=pd.DataFrame(rows); audit.to_csv(OUT/'study_metadata_audit.csv',index=False)
    flags=pd.DataFrame([
      {'variable':'species','assessment':'Does not track','evidence':'Both A and B contain human and mouse studies.'},
      {'variable':'exercise modality','assessment':'Does not track','evidence':'All selected contrasts are aerobic exercise.'},
      {'variable':'protocol','assessment':'Strong partial tracking','evidence':'Both A human studies are cycling; B human studies use knee extension or an incompletely reported baseline protocol. Mouse studies are treadmill in both patterns.'},
      {'variable':'duration/intensity','assessment':'Possible biological confounding','evidence':'Protocols differ by study; B includes 80% VO2max and maximal exhaustion, but moderate exercise also occurs in B and intensive exercise in A.'},
      {'variable':'post-exercise time','assessment':'Does not track','evidence':'Both patterns contain immediate and 3–4 h sampling.'},
      {'variable':'muscle/site','assessment':'Does not track','evidence':'Vastus lateralis and quadriceps occur across patterns; gastrocnemius occurs only in one A study.'},
      {'variable':'paired vs control design','assessment':'Does not independently track','evidence':'Human studies are paired and mouse studies use independent groups in both patterns; this is confounded with species.'},
      {'variable':'sequencing instrument','assessment':'Strong partial tracking','evidence':'All B studies use Illumina HiSeq 2500; A is heterogeneous (HiSeq 2500, HiSeq 2000, NovaSeq 6000). HiSeq 2500 is not exclusive to B.'},
      {'variable':'library preparation/molecule','assessment':'Does not track','evidence':'PolyA and ribo-zero occur across patterns; preparation and layout largely follow study/species.'},
      {'variable':'sex','assessment':'Does not track / limited','evidence':'Reported studies are predominantly male; sex is missing for GSE71972.'},
      {'variable':'training status','assessment':'Possible / incomplete','evidence':'Available A human metadata include trained; B includes active or sedentary. Most A samples and all mouse studies lack this field.'},
    ]); flags.to_csv(OUT/'pattern_tracking_flags.csv',index=False)
    # Compact publication-style study-by-metadata table.
    display=audit.copy()
    compact_protocol={
      'GSE108643':'Cycling; ~92–100 min to expend 650 kcal; 50% VO2max',
      'GSE86931':'Cycling; 70 min; 70% VO2max',
      'GSE126962':'Treadmill; 60 min; ramp 6→16 m/min, 5% incline',
      'GSE132520':'Treadmill; submaximal ramp 10→23 m/min, 10° incline',
      'GSE151066':'Cycling; 40 min; 60–70% HRR',
      'GSE71972':'Unilateral knee extension; 60 min; 60% peak power',
      'GSE87748':'Exercise protocol not reported; 15 min; 80% VO2max',
      'GSE97718':'Graded treadmill to exhaustion; one session',
    }
    display['protocol / dose']=display.GSE.map(compact_protocol)
    display['library / platform']=display.extracted_molecule+'; '+display.library_preparation+'; '+display.read_layout+'; '+display.instrument
    display['sex / training']=display.sex+'; '+display.training_status
    cols=['pattern','GSE','species','protocol / dose','post_exercise_time','muscle_site','design','library / platform','sex / training']
    labels=['Pattern','Study','Species','Protocol, duration & intensity','Post time','Muscle/site','Design','Library & instrument','Sex/training']
    cells=[[textwrap.fill(str(v),32) for v in row] for row in display[cols].values]
    fig,ax=plt.subplots(figsize=(18,9)); ax.axis('off'); table=ax.table(cellText=cells,colLabels=labels,cellLoc='left',colLoc='left',loc='center',colWidths=[.06,.08,.06,.24,.08,.09,.13,.18,.10])
    table.auto_set_font_size(False); table.set_fontsize(7.7); table.scale(1,3.0)
    for j in range(len(cols)): table[(0,j)].set_facecolor('#263746'); table[(0,j)].set_text_props(color='white',weight='bold')
    colors={'A':'#dbeafe','Intermediate':'#eeeeee','B':'#ffedd5'}
    for i,pattern in enumerate(display.pattern,1):
      for j in range(len(cols)): table[(i,j)].set_facecolor(colors[pattern]); table[(i,j)].set_edgecolor('white')
    ax.set_title('Task 2 study metadata audit: Pattern A → intermediate → Pattern B',fontsize=15,fontweight='bold',pad=18)
    fig.tight_layout(); fig.savefig(OUT/'study_metadata_audit.png',dpi=350,bbox_inches='tight',facecolor='white'); fig.savefig(OUT/'study_metadata_audit.pdf',bbox_inches='tight',facecolor='white'); plt.close(fig)
    provenance={'source':'curated GEPREP human/mouse metadata restricted to exact saved contrast GSMs','order':ORDER,'patterns':PATTERN,'missing_values':'reported explicitly; not inferred','statistical_testing':False,'flagging':'descriptive manual audit of obvious perfect or strong tracking'}
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n'); print(audit.to_string(index=False)); print('\nFlags\n',flags.to_string(index=False))
if __name__=='__main__': main()
