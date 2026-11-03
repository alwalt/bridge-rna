# Independent methodological review

## Verdict

This is a valid primary observed-RNA-seq reversal analysis with unusually strong provenance controls. It supports a claim of Bridge/DE complementarity, not Bridge superiority or therapeutic validation.

## Strengths

- Uses only experimentally observed GSE264130 expression and directly encoded observed samples.
- Does not read benchmark #6 predictions or molecular descriptors.
- Preserves drug, dose, time, plate, DMSO, and cell identity.
- Reproduces archived control embeddings exactly before latent scoring.
- Keeps expression, latent, target, LINCS, and literature evidence separate.
- Tests both cells individually and uses strict independent-study replication for space conditions.
- Reports BH failure rather than promoting unadjusted screens as discoveries.

## Material limitations

1. All drug-level BH q-values are approximately 0.997; no discovery is multiplicity-corrected.
2. The `p≤0.05` both-cell rule is an exploratory empirical-rank screen, not a confirmation threshold.
3. DIPG6 and SF8628 are related cancer contexts and are biologically mismatched to most tested conditions.
4. One dose and one time point prevent dose/time robustness analysis.
5. PLATE-Seq is shallow 3′ RNA-seq, although it is directly observed and preferable to inferred L1000 values for this purpose.
6. Expression and latent readouts disagree substantially; neither should be privileged post hoc.
7. No candidate passes strict terrestrial–OSDR replication.
8. ChEMBL support is sparse and nonsignificant; same-direction LINCS replication is weak.
9. Broad latent radiation opposition without selective empirical support suggests geometry shared across many drug responses, not specific radiomitigators.

## Claim boundary

It is reasonable to say that Bridge identifies distinct, experimentally observed reversal hypotheses beyond DE. It is not reasonable to say Bridge is more accurate, that negative latent cosine implies benefit, or that any drug is validated for MS, Crohn's disease, SLE, radiation injury, bone loss, or muscle atrophy.
