# BridgeRNA drug-discovery benchmark: Phase 2 results

At the prespecified primary endpoint, Bridge **does not establish a statistically conclusive improvement** in cross-study drug convergence over matched DE (overall Bridge-minus-DE mean pairwise top-25 Jaccard = 0.2025; exact label-swap p = 0.05859).

This benchmark measures convergence of ChEMBL direct mechanism-target enrichment rankings. It does not test perturbational reversal, treatment direction, clinical benefit, or therapeutic efficacy.

## Primary results

| condition      |   bridge_mean_pairwise_top25_jaccard |   de_mean_pairwise_top25_jaccard |   bridge_minus_de |   exact_p |   permutations |
|:---------------|-------------------------------------:|---------------------------------:|------------------:|----------:|---------------:|
| Radiation      |                             0.155354 |                        0.0416667 |         0.113688  | 0.25      |              8 |
| Bone loss      |                             0.142308 |                        0.180556  |        -0.0382479 | 0.75      |              8 |
| Muscle atrophy |                             0.714286 |                        0.182299  |         0.531987  | 0.25      |              8 |
| Overall        |                             0.337316 |                        0.13484   |         0.202475  | 0.0585938 |            512 |

## Expression-matched random-module controls

| condition      | method   |   observed |   random_mean |   random_sd |   empirical_p |
|:---------------|:---------|-----------:|--------------:|------------:|--------------:|
| Radiation      | Bridge   |  0.155354  |     0.070885  |   0.081584  |     0.154845  |
| Radiation      | DE       |  0.0416667 |     0.0685686 |   0.0838717 |     0.521479  |
| Bone loss      | Bridge   |  0.142308  |     0.0807689 |   0.0802547 |     0.190809  |
| Bone loss      | DE       |  0.180556  |     0.0634882 |   0.0703827 |     0.0819181 |
| Muscle atrophy | Bridge   |  0.714286  |     0.0791101 |   0.10565   |     0.002997  |
| Muscle atrophy | DE       |  0.182299  |     0.0660388 |   0.0753082 |     0.0799201 |

## ARCHS4 pretraining-status analysis

| pretraining_pair   |   n_pairs |   bridge_mean |   de_mean |   bridge_minus_de |
|:-------------------|----------:|--------------:|----------:|------------------:|
| exposure_involved  |         5 |      0.474725 |  0.13438  |         0.340346  |
| strict_unseen_pair |         4 |      0.165554 |  0.135417 |         0.0301376 |

Comparisons involving a train/validation-exposed cohort are descriptive reproducibility results, not out-of-pretraining generalization. The strict-unseen row is the relevant held-out comparison.

## GSE211204 leakage sensitivity

| method   |   mean_pairwise_top25_jaccard | pairwise_jaccards                           |
|:---------|------------------------------:|:--------------------------------------------|
| Bridge   |                      0.714286 | 1.0;0.5714285714285714;0.5714285714285714   |
| DE       |                      0.171717 | 0.2;0.13333333333333333;0.18181818181818182 |

The sensitivity replaces the muscle discovery ranking with the seven subjects whose baseline and ULLS endpoints are both absent from the reconstructed ARCHS4 train/validation split.

## Module-size and target-threshold sensitivity

| condition      |   module_size |   min_targets |   Bridge |        DE |   bridge_minus_de |
|:---------------|--------------:|--------------:|---------:|----------:|------------------:|
| Bone loss      |           100 |             2 | 0.25     | 0         |        0.25       |
| Bone loss      |           100 |             3 | 0.333333 | 0         |        0.333333   |
| Bone loss      |           100 |             5 | 0        | 0         |        0          |
| Bone loss      |           250 |             2 | 0.2      | 0         |        0.2        |
| Bone loss      |           250 |             3 | 0.333333 | 0         |        0.333333   |
| Bone loss      |           250 |             5 | 0        | 0         |        0          |
| Bone loss      |           500 |             2 | 0.122006 | 0.12963   |       -0.00762386 |
| Bone loss      |           500 |             3 | 0.142308 | 0.180556  |       -0.0382479  |
| Bone loss      |           500 |             5 | 0        | 0         |        0          |
| Bone loss      |           750 |             2 | 0.113636 | 0.122357  |       -0.00872081 |
| Bone loss      |           750 |             3 | 0.175926 | 0.072652  |        0.103274   |
| Bone loss      |           750 |             5 | 0        | 0.166667  |       -0.166667   |
| Bone loss      |          1000 |             2 | 0.143684 | 0.12654   |        0.0171442  |
| Bone loss      |          1000 |             3 | 0.116842 | 0.104762  |        0.0120802  |
| Bone loss      |          1000 |             5 | 0        | 0.333333  |       -0.333333   |
| Muscle atrophy |           100 |             2 | 0        | 0         |        0          |
| Muscle atrophy |           100 |             3 | 0        | 0         |        0          |
| Muscle atrophy |           100 |             5 | 0        | 0         |        0          |
| Muscle atrophy |           250 |             2 | 0.611111 | 0.0909091 |        0.520202   |
| Muscle atrophy |           250 |             3 | 0.733333 | 0.0740741 |        0.659259   |
| Muscle atrophy |           250 |             5 | 0        | 0         |        0          |
| Muscle atrophy |           500 |             2 | 0.379277 | 0.137106  |        0.242171   |
| Muscle atrophy |           500 |             3 | 0.714286 | 0.182299  |        0.531987   |
| Muscle atrophy |           500 |             5 | 0        | 0.111111  |       -0.111111   |
| Muscle atrophy |           750 |             2 | 0.298008 | 0.0999443 |        0.198064   |
| Muscle atrophy |           750 |             3 | 0.410053 | 0.248148  |        0.161905   |
| Muscle atrophy |           750 |             5 | 0        | 0.111111  |       -0.111111   |
| Muscle atrophy |          1000 |             2 | 0.203283 | 0.107508  |        0.0957757  |
| Muscle atrophy |          1000 |             3 | 0.295815 | 0.316667  |       -0.0208514  |
| Muscle atrophy |          1000 |             5 | 0        | 0.111111  |       -0.111111   |
| Radiation      |           100 |             2 | 0        | 0         |        0          |
| Radiation      |           100 |             3 | 0        | 0         |        0          |
| Radiation      |           100 |             5 | 0        | 0         |        0          |
| Radiation      |           250 |             2 | 0.25     | 0.030303  |        0.219697   |
| Radiation      |           250 |             3 | 0.25     | 0.047619  |        0.202381   |
| Radiation      |           250 |             5 | 0        | 0         |        0          |
| Radiation      |           500 |             2 | 0.172161 | 0.0222222 |        0.149939   |
| Radiation      |           500 |             3 | 0.155354 | 0.0416667 |        0.113688   |
| Radiation      |           500 |             5 | 0        | 0         |        0          |
| Radiation      |           750 |             2 | 0.187302 | 0.0502646 |        0.137037   |
| Radiation      |           750 |             3 | 0.147334 | 0.0703704 |        0.0769638  |
| Radiation      |           750 |             5 | 0        | 0.111111  |       -0.111111   |
| Radiation      |          1000 |             2 | 0.317269 | 0.0392496 |        0.27802    |
| Radiation      |          1000 |             3 | 0.151608 | 0.0769231 |        0.0746851  |
| Radiation      |          1000 |             5 | 0        | 0.166667  |       -0.166667   |

## Primary ranking coverage

| method   | dataset   |   tested_drugs |   nonzero_overlap_drugs |   fdr_005 |
|:---------|:----------|---------------:|------------------------:|----------:|
| Bridge   | GSE113165 |             57 |                       5 |         0 |
| Bridge   | GSE184119 |             30 |                       4 |         0 |
| Bridge   | GSE189524 |             63 |                       4 |         0 |
| Bridge   | GSE211204 |             61 |                       5 |         0 |
| Bridge   | GSE234465 |             59 |                       6 |         0 |
| Bridge   | GSE273868 |             50 |                       1 |         0 |
| Bridge   | GSE276529 |             63 |                      10 |         0 |
| Bridge   | GSE297090 |             40 |                      11 |         0 |
| Bridge   | GSE297560 |             50 |                      10 |         0 |
| DE       | GSE113165 |             57 |                       4 |         0 |
| DE       | GSE184119 |             30 |                       2 |         0 |
| DE       | GSE189524 |             63 |                       4 |         0 |
| DE       | GSE211204 |             61 |                       7 |         0 |
| DE       | GSE234465 |             59 |                       9 |         0 |
| DE       | GSE273868 |             50 |                       3 |         0 |
| DE       | GSE276529 |             63 |                       7 |         0 |
| DE       | GSE297090 |             40 |                       0 |         0 |
| DE       | GSE297560 |             50 |                       7 |         0 |

## Attribution QA

Across 220 sample attributions, the median absolute IG endpoint-completeness error was 0.002496 (95th percentile 0.005672).

| dataset   |     random |       top |   top_to_random_change_ratio |
|:----------|-----------:|----------:|-----------------------------:|
| GSE113165 | 0.0224193  | 0.105169  |                      4.691   |
| GSE184119 | 0.00609989 | 0.0783773 |                     12.849   |
| GSE189524 | 0.0205714  | 0.106373  |                      5.17093 |
| GSE211204 | 0.0178678  | 0.142545  |                      7.97773 |
| GSE234465 | 0.0165855  | 0.211758  |                     12.7676  |
| GSE273868 | 0.00208298 | 0.0557057 |                     26.7433  |
| GSE276529 | 0.0070104  | 0.147159  |                     20.9916  |
| GSE297090 | 0.00573654 | 0.0368443 |                      6.42274 |
| GSE297560 | 0.00946112 | 0.133095  |                     14.0676  |

For every dataset, masking the Bridge top-500 changed the condition-axis effect more than the mean of five expression-decile-matched random panels.

## Scope and limitations

- Modules use absolute signed Bridge attribution or absolute DE statistic, with identical sizes and dataset-specific expressed-gene universes.
- Zero-target-overlap p=1 ties are retained in enrichment output but excluded from convergence rankings because identifier ordering is not biological signal.
- The direct-mechanism graph is sparse: primary nonzero-overlap lists can contain fewer than 25 drugs (including an empty DE list for GSE297090). Thus the nominal top-25 Jaccard uses all available nonzero-overlap drugs when fewer than 25 exist and must be interpreted as a sparse-candidate convergence endpoint.
- Bone convergence spans primary osteoporosis MSCs, glucocorticoid-associated cortical bone, and spaceflight bone-marrow MSCs; it is cross-etiology convergence, not literal disease replication.
- OSDR culture/chip replicates are not independent human donors.
- No direction-aware drug perturbation or reversal analysis was performed.
