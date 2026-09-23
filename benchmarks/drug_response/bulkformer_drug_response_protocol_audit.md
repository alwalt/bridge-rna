# Final BulkFormer drug-response protocol audit

Audit date: 2026-09-22. This audit distinguishes the final 2026 *Cell Systems*
release from the 2025 bioRxiv protocol; they are materially different.

## Final result verified

The final paper is Kang et al., “BulkFormer: A large-scale foundation model for
bulk transcriptomes,” *Cell Systems* 17(7), 101657 (2026), DOI
[10.1016/j.cels.2026.101657](https://doi.org/10.1016/j.cels.2026.101657).
The official repository README and final Table S1 (`mmc2.xlsx`) both report
BulkFormer **mean PCC = 0.373** for drug-response prediction. This is a verified
final-release value, not a locally reproduced value.

## What is explicit in final public materials

- Task: drug-response prediction from baseline transcriptomic state.
- Metric label: **mean PCC**.
- BulkFormer result: 0.373.
- The released final data contain GDSC1 and GDSC2 measurements, 700 cell lines,
  295 GDSC drug IDs, 255 unique PubChem CIDs, and 212,299 IC50 rows.
- BulkFormer sample representations are frozen final-layer contextual gene
  embeddings aggregated to sample level. The current official extraction
  notebook supports mean/max/median pooling and demonstrates mean pooling.
- The final release supplies processed expression and response files on Zenodo
  record [15744294](https://doi.org/10.5281/zenodo.15744294).

## Recoverable from final released data/code

- Every `(cell line, dataset version, drug ID)` is unique.
- `(cell line, drug ID)` is not unique because 60 drug IDs occur in both GDSC1
  and GDSC2. Treating version-specific drug screens as targets gives exactly 355
  screens with 134–697 observations each. This is the most plausible recoverable
  unit over which “mean PCC” is averaged, but the final text/code does not state
  it explicitly.
- Response columns are IC50, AUC, maximum concentration, curve-fit RMSE, and
  Z-score. The task is labeled IC50 by the release filename and the prior paper;
  the local common benchmark therefore uses the released `IC50` column.
- Baseline expression has 700 DepMap IDs × 19,098 genes. It is one expression
  profile per cell line and is therefore baseline state, not a post-treatment
  transcriptome.
- Current BulkFormer checkpoints are frozen for local feature extraction. The
  current public repository does not include final downstream training code.

## Unresolved final-protocol details

The final paper landing page and figure supplement were checked; the public
supplement contains figures but no STAR Methods. The official repository history
contains no downstream drug-response implementation. Consequently, these details
remain unresolved and are not invented here:

- exact GDSC release date/version beyond the `GDSC1`/`GDSC2` labels;
- provenance and exact preprocessing of the expression matrix;
- whether IC50 was transformed again or standardized before fitting;
- exact missing-value/filtering rules before the released tables;
- exact pooling choice used in the final drug experiment;
- final drug representation (or whether separate predictors were fit per screen);
- downstream architecture, loss, optimizer, learning rate, epochs, and stopping;
- exact train/validation/test or cross-validation scheme and split axis;
- fold/seed count;
- whether PCC is per version-specific screen, PubChem compound, cell line, fold,
  or global, and the precise definition of “mean PCC”;
- final SCC/RMSE values.

The published 0.373 is therefore displayed as a literature reference and is not
called directly comparable to local results.

## Historical 2025 preprint (not the final protocol)

The bioRxiv v1 paper (DOI
[10.1101/2025.06.11.659222](https://doi.org/10.1101/2025.06.11.659222)) explicitly
used 255 compounds across 700 cell lines and 32 cancer types; KPGT compound
features concatenated with max-pooled transcriptomic representations; an MLP;
10-fold cross-validation; and global PCC/SCC. It reported PCC 0.910 and SCC
0.879. Because the final result is 0.373 and is labeled *mean* PCC, this historical
protocol cannot be assumed to describe the final benchmark.

## Sources

1. Final paper DOI and PubMed PMID 42385705.
2. Official repository: <https://github.com/KangBoming/BulkFormer>, commit
   `5bcf5b9` inspected on 2026-09-22.
3. Final supplementary Table S1:
   <https://ars.els-cdn.com/content/image/1-s2.0-S2405471226001390-mmc2.xlsx>.
4. Final data release: <https://doi.org/10.5281/zenodo.15744294>.
5. Historical bioRxiv v1: <https://doi.org/10.1101/2025.06.11.659222>.
