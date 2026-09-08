#!/usr/bin/env python3
"""Small RR3-only technical-reference overlap time-course summary."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

HERE=Path(__file__).resolve().parents[1]
BASE=HERE/'results/task4_confounding_profiler/biological_technical_overlap'
OUT=BASE/'rr3_timecourse';OUT.mkdir(parents=True,exist_ok=True)

d=pd.read_csv(BASE/'per_contrast_overlap_metrics.csv')
q=d[(d.OSD=='OSD-137')&(d.mission=='RR3')].sort_values('flight_duration_days').copy()
assert q.flight_duration_days.tolist()==[39.0,40.0,41.0]
q['technical_replication_counterpart']=q.flight_duration_days.isin([39,40])
q['technical_replication_note']=q.flight_duration_days.map({39:'OSD-168 has matched F1/F2 and G1/G2 material',40:'OSD-168 has matched F3/F4 and G3/G5 material',41:'No OSD-168 counterpart for F6/G6/G7'})
cols=['contrast_id','OSD','mission','flight_duration_days','n_FLT','n_GC','technical_replication_counterpart','technical_replication_note','bridge_total_magnitude','aligned_magnitude_PC1_2','aligned_fraction_PC1_2','orthogonal_magnitude_PC1_2','rms_log2FC','pythagorean_relative_error_PC1_2']
q[cols].to_csv(OUT/'rr3_timepoint_metrics.csv',index=False)
metrics=['bridge_total_magnitude','aligned_magnitude_PC1_2','aligned_fraction_PC1_2','orthogonal_magnitude_PC1_2','rms_log2FC'];rows=[]
for metric in metrics:
 z=spearmanr(q.flight_duration_days,q[metric]);v=q[metric].to_numpy();direction='increasing' if v[0]<v[1]<v[2] else 'decreasing' if v[0]>v[1]>v[2] else 'non-monotonic'
 rows.append({'metric':metric,'day39':v[0],'day40':v[1],'day41':v[2],'strict_monotonic_pattern':direction,'spearman_rho_with_day':z.statistic,'p_value_descriptive_only':z.pvalue})
trend=pd.DataFrame(rows);trend.to_csv(OUT/'rr3_monotonic_trend_summary.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
axes[0].plot(q.flight_duration_days,q.bridge_total_magnitude,'o-',label='Total',lw=2);axes[0].plot(q.flight_duration_days,q.aligned_magnitude_PC1_2,'o-',label='PC1–2 aligned',lw=2);axes[0].plot(q.flight_duration_days,q.orthogonal_magnitude_PC1_2,'o-',label='Orthogonal',lw=2);axes[0].set(xlabel='Flight duration (days)',ylabel='Bridge response magnitude',xticks=[39,40,41],title='RR3 response decomposition');axes[0].legend(fontsize=8)
axes[1].plot(q.flight_duration_days,q.aligned_fraction_PC1_2,'o-',color='#CC6677',lw=2,label='Aligned fraction');ax2=axes[1].twinx();ax2.plot(q.flight_duration_days,q.rms_log2FC,'s--',color='#4477AA',lw=2,label='RMS log2FC');axes[1].set(xlabel='Flight duration (days)',ylabel='PC1–2 aligned fraction',xticks=[39,40,41],title='Aligned fraction and conventional response');ax2.set_ylabel('RMS log2FC');lines=axes[1].lines+ax2.lines;axes[1].legend(lines,[x.get_label() for x in lines],fontsize=8)
fig.savefig(OUT/'rr3_timecourse_overlap.png',dpi=350);fig.savefig(OUT/'rr3_timecourse_overlap.pdf');plt.close(fig)
summary={'technical_replication_omission':'OSD-168 lacks the RR3 41-day F6, G6, and G7 biological material; only 39- and 40-day RR3 animals were technically remeasured.','aligned_fraction_pattern':'strictly increasing numerically, but effectively plateaus from day 40 to 41','aligned_fraction_values':dict(zip(q.flight_duration_days.astype(int).astype(str),q.aligned_fraction_PC1_2)),'magnitude_pattern':'non-monotonic: lower at day 40 and highest at day 41','conventional_pattern':'non-monotonic: lower at day 40 and highest at day 41','caveat':'Only three time points; day 41 has 1 FLT and 2 GC samples. No inferential time-trend claim is supported.'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(q[cols].to_string(index=False));print('\n',trend.to_string(index=False))
