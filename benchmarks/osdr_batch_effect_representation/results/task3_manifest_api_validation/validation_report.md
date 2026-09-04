# Task 3 manifest validation against OSDR API

- Local samples audited: 112
- Local contrasts audited: 14
- Samples found by exact API sample name: 112/112

The audit is non-mutating: discrepancies below were reported but not applied to the Task 3 manifest.

## OSD-245 root cause

The API identifies OSD-245 as RR-6. `ISS-T` means ISS-terminal and `LAR` means live-animal return; both are strata within RR-6. The original local `mission()` parser incorrectly treated the substring `ISS-T` as evidence for RR-3. This was a local mapping-logic error, not a source-metadata error. It changed annotation and mission-boundary summaries, but not sample membership, FLT/GC assignment, or the two OSD-245 contrast memberships.

## Important comparability limitation

Sequencing facility is recovered from the API dataset protocol descriptions. Layout/read length/instrument are structured per sample; nominal requested depth is not consistently structured, so configuration rows are marked partial matches when layout and read length agree.
