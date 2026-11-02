# Focused LUAD ARCHS4 retrieval audit

This analysis uses frozen outputs only. It does not rerun Bridge inference or
retrain the Hallmark head. The representative patient is the observed LUAD
sample closest by Hallmark cosine to the 516-patient LUAD Hallmark centroid;
selection is independent of ARCHS4 retrieval performance.

## Main result

| Query | Method | Conservative top-5 composition | Bridge/Hallmark top-5 overlap |
|---|---|---|---:|
| Representative TCGA-86-A4JF | Bridge-512 | 2 lung adenocarcinoma; 3 unspecified lung/NSCLC | 2 samples |
| Representative TCGA-86-A4JF | Hallmark-50 | 1 lung adenocarcinoma; 1 lung squamous carcinoma; 3 unspecified lung/NSCLC | 2 samples |
| All 516 LUAD patients, centroid | Bridge-512 | 5 lung adenocarcinoma | 0 samples |
| All 516 LUAD patients, centroid | Hallmark-50 | 1 lung adenocarcinoma; 1 lung squamous carcinoma; 3 unspecified lung/NSCLC | 0 samples |

For the centroid, Bridge-512 retrieves five histologically annotated lung
adenocarcinomas: GSM3405990, GSM8085628, GSM3405997, GSM3405985, and
GSM3405982. Four come from GSE120622. This is strong tissue and disease-identity
retrieval, with some study concentration.

Hallmark-50 retrieves LUAD GSM5081817, broad NSCLC/lung tumors GSM3995408,
GSM6444157 and GSM2142550, and lung squamous carcinoma GSM5482393. Their main
positive similarity contributions are E2F targets, DNA repair, MYC targets V1,
G2-M checkpoint, late estrogen response, interferon-alpha response, glycolysis,
and mTORC1 signaling. The LUSC result is therefore a shared proliferative/DNA
repair/interferon/metabolic state, not evidence that LUAD and LUSC are the same
disease.

The representative patient shows the same directional contrast, although two
samples (GSM4159540 and GSM3405985) occur in both method-specific top fives.
Bridge scores are near 0.9972–0.9978 and Hallmark scores near 0.9560–0.9600;
scores should not be compared numerically across dimensions.

## Subgroups

No LUAD subgroup retrieval was run. The authoritative local TCGA manifest has
sample/patient identity, cancer label, sample type, survival outcome and time,
but no reliable mutation, smoking, stage, histologic subtype, or molecular
subtype fields. Survival-defined groups would not answer the proposed
heterogeneity question and were not substituted. A subgroup analysis requires
a separately frozen, sample-matched clinical/molecular manifest.

## Files

- `luad_top5_retrieval_audit.csv`: exact hits, metadata, scores, and Hallmark contributions.
- `luad_top5_composition.csv`: conservative top-five composition.
- `luad_query_dominant_hallmarks.csv`: dominant signed query programs.
- `luad_cross_method_heatmap_values.csv`: full-background standardized heatmap values and ranks.
- `luad_bridge512_vs_hallmark50_heatmap.png`/`.pdf`: matched representative/centroid comparison.
- `luad_hallmark_terms_bridge512_vs_hallmark50.png`/`.pdf`: all 50 Hallmark terms across the query and both methods' selected hits.
- `luad_hallmark_term_heatmap_values.csv`: source values for the Hallmark-term heatmap.
- `subgroup_availability.csv`: label audit.
- `provenance.json`: frozen inputs and deterministic selection rules.

## Reproduce

```bash
cd benchmarks/hallmark_state_retrieval
/home/walt/bridge-rna/.venv/bin/python pipeline/audit_luad_retrieval.py
```
