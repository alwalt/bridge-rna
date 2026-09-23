# Bridge pretraining-exposure audit

Bridge was pretrained on ARCHS4. The released benchmark expression table contains
DepMap model IDs and gene values but no GEO/SRA run, sample, or study accessions.
An exact row-level join to the Bridge pretraining manifest is therefore impossible.

All 700 profiles are classified **unresolved** for exact sample exposure and
**unresolved** for same-study exposure. Cell-line names alone are insufficient:
the same named line may appear in many independent experiments, and a name match
would not establish that this exact baseline profile was present during pretraining.

No samples are removed. No “unseen” subset is reported because the released
provenance cannot support one. Local held-out-cell-line folds measure downstream
readout generalization, not guaranteed pretraining novelty.
