# Independent methodological review

## Verdict

The analysis is technically reproducible and appropriately conservative about latent compatibility and therapeutic claims. Its main valid contribution is directional stratification of frozen Bridge and DE programs. It does not support a broad claim that Bridge improves drug reversal over DE.

## Strengths

- Upstream signatures and modules were reused with recorded hashes; no outcome-driven gene redefinition occurred.
- The resource, gene subset, scoring rule, context threshold, candidate rule, and nulls were frozen before reading the expression matrix.
- Identical LINCS signatures were used for Bridge and DE.
- Cell, dose, time, perturbagen identity, and replicate-quality metadata remain recoverable.
- Cross-study space-condition claims require agreement in all three datasets, including OSDR.
- Invalid Bridge latent encoding of L1000 z-scores was rejected rather than forced.
- Global FDR and post-ranking auxiliary nulls are clearly separated.

## Material limitations

1. **No globally significant drug enrichment.** The competitive rank p-values are approximately uniform by construction and all BH q-values are near one. Top-1% sets are screens, not significant hits.
2. **Selection-conditioned nulls.** Prioritized sign/expression nulls test structure after compounds were ranked; they are useful robustness checks but cannot establish discovery-wide significance.
3. **Cell-context mismatch.** Most conditions lack two relevant LINCS cell types. Cancer-cell signatures should receive lower evidentiary weight than matched normal or primary contexts.
4. **L1000 inference.** Only 978 genes are directly measured; even the BING analysis uses many inferred values.
5. **Level-5 controls.** Signatures are standardized to plate controls rather than reconstructed here from raw paired vehicle wells.
6. **Module sensitivity.** Bridge top-500/full-ranking agreement is lower than DE agreement, so some Bridge drug rankings depend on the module definition.
7. **Cross-study screen calibration.** Requiring a top-decile result in three correlated condition signatures can still yield many compounds; it is a reproducibility filter, not a family-wise significance test.
8. **Sparse target integration.** Direct expanded-ChEMBL support among reversal screens is rare and nonsignificant.
9. **No latent test.** This is a justified omission, but means the benchmark tests Bridge-derived gene programs rather than drug movement in Bridge latent space.

## Claim boundary

The results justify saying that Bridge contributes complementary, sometimes reproducible directional pharmacology hypotheses. They do not justify saying that Bridge is generally superior to DE, that a negative connectivity score predicts clinical benefit, or that any candidate is therapeutic.

## Recommended next validation

Before translational prioritization, test the frozen candidates in matched primary human cell or organoid perturbation data with explicit vehicle pairs, then prospectively validate only a small prespecified set. Do not expand the benchmark or tune thresholds using the current outcomes.
