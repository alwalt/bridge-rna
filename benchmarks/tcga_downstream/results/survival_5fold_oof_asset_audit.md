# Complete five-fold OOF survival asset audit

The existing `cohort_manifest.parquet` is authoritative. It contains 9,816
unique primary-tumor patients from 32 TCGA cancer projects. The unchanged
benchmark eligibility rule (`event` in `{0,1}` and `time_days > 0`) retains
9,668 patients with 2,774 observed deaths.

The 148 exclusions comprise 21 patients without a valid binary OS event, 34
patients with a valid event but missing OS time, and 93 patients with a valid
event but nonpositive OS time. These categories are mutually exclusive. No
additional patients were excluded by the OOF extension.

Every eligible row has a unique patient ID, cancer label, survival time, event
indicator, and valid matrix row. The cached Bridge embeddings (9,816 × 512),
Bridge-vocabulary expression used by PCA (9,816 × 15,165), and full expression
(9,816 × 25,150) are complete and finite for every eligible matrix row.

The analysis therefore reuses all frozen representations and performs no
Bridge encoder inference. It fits only the established downstream survival
heads and training-fold-only PCA/scaling.
