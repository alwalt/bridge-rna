# Three-study gene recurrence and pathway validation

## Protocol

- Reuse the frozen 500-gene BRIDGE and DE lists independently selected in each of three studies per condition. No model or DE refitting.
- Use the original positive-expression canonical-gene universe for each study; use its three-study intersection for recurrent-gene enrichment. Restrict to library-annotated genes for each library.
- Test GO Biological Process 2025 and Reactome Pathways 2024 annotations downloaded from Enrichr. One-sided hypergeometric tests; BH correction across both libraries per query; terms containing 10–500 background genes. Display significant rows with at least three overlapping genes.
- Use 1,000 uniform random top-500 selections from each study’s eligible genes as a simple intersection-size control. This does not match expression propensity or establish a BRIDGE–DE significance difference.

## Recurrence

- Muscle atrophy, BRIDGE: 102 shared genes; background 14146; random mean 0.544, maximum 5; empirical p=0.000999001.
- Muscle atrophy, DE: 7 shared genes; background 14146; random mean 0.544, maximum 5; empirical p=0.000999001.
- Bone loss, BRIDGE: 66 shared genes; background 13765; random mean 0.613, maximum 5; empirical p=0.000999001.
- Bone loss, DE: 0 shared genes; background 13765; random mean 0.613, maximum 5; empirical p=1.
- Radiation, BRIDGE: 50 shared genes; background 12239; random mean 0.564, maximum 4; empirical p=0.000999001.
- Radiation, DE: 3 shared genes; background 12239; random mean 0.564, maximum 4; empirical p=0.01998.

## Biological annotations: strongest results without manual term selection

### Muscle atrophy
- Striated Muscle Contraction: q=2.75e-08; 9/70 annotated query genes; 31.42-fold; ACTA1;DES;MYBPC2;MYL1;NEB;TNNC1;TNNI1;TPM1;TPM2.
- Muscle Contraction: q=4.78e-07; 14/70 annotated query genes; 9.25-fold; ACTA1;ATP1A2;CACNB1;CASQ1;DES;MYBPC2;MYL1;NEB;PDE5A;SORBS1;TNNC1;TNNI1;TPM1;TPM2.
- Sarcomere Organization (GO:0045214): q=0.000265; 6/89 annotated query genes; 23.55-fold; ANKRD1;CASQ1;MYBPC2;MYLK3;MYOM1;TPM1.
- Myofibril Assembly (GO:0030239): q=0.002; 6/89 annotated query genes; 16.19-fold; ANKRD1;CASQ1;MYBPC2;MYLK3;MYOM1;TPM1.
- Actomyosin Structure Organization (GO:0031032): q=0.00315; 7/89 annotated query genes; 10.93-fold; ANKRD1;CASQ1;MYBPC2;MYLK3;MYOM1;SORBS1;TPM1.
### Bone loss
- No intersection pathways pass the specified BH FDR and overlap criteria.
### Radiation
- No intersection pathways pass the specified BH FDR and overlap criteria.

## Equal-size per-study pathway comparison

- Muscle atrophy, BRIDGE: 8 pathway rows pass the same criterion in all three independently selected 500-gene sets.
- Muscle atrophy, DE: 0 pathway rows pass the same criterion in all three independently selected 500-gene sets.
- Bone loss, BRIDGE: 4 pathway rows pass the same criterion in all three independently selected 500-gene sets.
- Bone loss, DE: 0 pathway rows pass the same criterion in all three independently selected 500-gene sets.
- Radiation, BRIDGE: 0 pathway rows pass the same criterion in all three independently selected 500-gene sets.
- Radiation, DE: 0 pathway rows pass the same criterion in all three independently selected 500-gene sets.
- Bone BRIDGE lists repeatedly implicate extracellular-matrix organization, ECM proteoglycans and collagen-fibril organization, despite no significant enrichment of the smaller exact three-study intersection. Radiation has no pathway passing in all three lists under this protocol.

## Limits

- Enrichment describes annotated biological coherence, not specificity to a stressor, causal mechanisms, directional consistency, or therapeutic benefit.
- Small DE intersections (including an empty set) have little enrichment power; significance for BRIDGE but not DE is not a direct method-difference test.
- Equal-size per-study pathway results and pathway recurrence are supplied separately; term counts are descriptive and affected by overlapping annotations.
- Heterogeneous tissues and exposures, and partial terrestrial pretraining exposure, remain relevant.
- This is a new exploratory analysis in a separate results directory. Prior source results are preserved.
