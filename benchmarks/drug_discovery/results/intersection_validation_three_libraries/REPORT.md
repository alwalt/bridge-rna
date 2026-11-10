# GO, KEGG and Reactome: three-study validation

## Gene selection

- Select 500 genes independently per study and method, then compute three-study intersections: BRIDGE 102/66/50 and DE 7/0/3 (muscle/bone/radiation). No list of 500 shared genes is used.
- Intersection enrichment and pathways recurring across three separately enriched top-500 lists are distinct endpoints, shown in separate three-column figures.
- Libraries: GO Biological Process 2025; KEGG Human 2021; Reactome Pathways 2024. Hypergeometric tests use eligible library-annotated backgrounds; BH correction jointly across all three libraries for each query. q<0.05 and at least three overlapping genes; term size 10–500.
- Columns show the three libraries, rows the three conditions. Up to three strongest passing terms per cell, chosen by adjusted p-value without manually selecting stress-related terms. Display labels may be shortened; full names are in CSV.
- Recurrent-pathway bars use the largest of the three study-specific q-values, not a combined p-value. Pathways can overlap and need not have the same contributing genes in each study.
- Prior two-library analysis is preserved under results/intersection_validation/. Adding KEGG expands the correction family; use the new q-values for these figures.

## Passing term counts

- Muscle atrophy, BRIDGE, GO Biological Process: intersection 5; recurring across study lists 6.
- Muscle atrophy, BRIDGE, KEGG: intersection 3; recurring across study lists 2.
- Muscle atrophy, BRIDGE, Reactome: intersection 2; recurring across study lists 2.
- Muscle atrophy, DE, GO Biological Process: intersection 0; recurring across study lists 0.
- Muscle atrophy, DE, KEGG: intersection 0; recurring across study lists 0.
- Muscle atrophy, DE, Reactome: intersection 0; recurring across study lists 0.
- Bone loss, BRIDGE, GO Biological Process: intersection 0; recurring across study lists 2.
- Bone loss, BRIDGE, KEGG: intersection 0; recurring across study lists 0.
- Bone loss, BRIDGE, Reactome: intersection 0; recurring across study lists 2.
- Bone loss, DE, GO Biological Process: intersection 0; recurring across study lists 0.
- Bone loss, DE, KEGG: intersection 0; recurring across study lists 0.
- Bone loss, DE, Reactome: intersection 0; recurring across study lists 0.
- Radiation, BRIDGE, GO Biological Process: intersection 0; recurring across study lists 0.
- Radiation, BRIDGE, KEGG: intersection 0; recurring across study lists 0.
- Radiation, BRIDGE, Reactome: intersection 0; recurring across study lists 0.
- Radiation, DE, GO Biological Process: intersection 0; recurring across study lists 0.
- Radiation, DE, KEGG: intersection 0; recurring across study lists 0.
- Radiation, DE, Reactome: intersection 0; recurring across study lists 0.

## Limitations

- Small DE intersections have limited power. Different significance outcomes are not a direct method-difference test.
- Annotation enrichment supports coherence, not condition specificity, causal mechanisms, or treatment benefit.
- Random size controls are uniform eligibility-matched selections, not expression-propensity matched controls.
- Heterogeneous contexts and partial terrestrial pretraining exposure remain relevant.
