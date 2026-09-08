# Frozen sample-embedding readout benchmark

This standalone benchmark asks whether a better sample-level representation can
be extracted from the frozen BridgeRNA backbone. It examines only Transformer
layers 11 and 12 and never updates encoder weights.

## Cohort and evaluation

The benchmark reuses the exact 3,272-sample, 14-tissue cohort and five
GSE-disjoint folds from Task 4. Fixed readouts reuse cached layer summaries.
Trainable attention/readout parameters are initialized, trained, selected, and
evaluated independently inside each outer training fold.

```bash
bash benchmarks/frozen_sample_embedding_readout/pipeline/run_all.sh
```

Monitor the persistent run with:

```bash
tail -f benchmarks/frozen_sample_embedding_readout/results/run.log
```

Expected wall time is approximately 3–5 hours on two RTX 3090 GPUs. The script
does not cache full contextual token tensors; these would require roughly 100 GB
for both layers. Compact per-fold metrics and pooled representations are saved.

## Interpretation

Selection uses both study-disjoint tissue utility and the unchanged OSDR
response-geometry safety checks. A readout is not considered better merely
because it increases tissue classification if it conceals RR1 sensitivity,
damages the reproducible RR3 controls, or increases the RR1↔RR3-39 false friend.

## Results

Layer-11 mean+SD produced the highest tissue score (macro F1 0.8355 ±
0.0338), only 0.0023 above layer-12 mean+SD (0.8332 ± 0.0254). Neither
reached raw expression (0.8716) or PCA (0.8883). Learned attention pooling was
substantially worse (best: layer-12 multi-head, 0.7325), and the cross-layer
PCA projection reached 0.8246.

The response-geometry safety check favors layer-12 mean+SD as the practical
selection. It preserves RR1/RR3 behavior and slightly reduces the known
RR1↔RR3-39 false-friend cosine (0.7953 versus 0.8108 for current mean).
Layer-11 mean+SD slightly improves tissue F1 but raises that false-friend
cosine to 0.8597. Attention lowers false similarity only by degrading the real
technical-replication structure as well. Thus mean+SD recovers some information
lost by mean pooling, but the shared-direction problem is not primarily caused
by pooling.

## Task-agnostic extension

`pipeline/run_task_agnostic.py` evaluates frozen internal-attention weighting,
50 Hallmark module representations, and fixed layer-11/layer-12 combinations.
It reuses the same cached inputs and exact GSE-disjoint folds. Full contextual
states are streamed once and only derived sample representations are cached
under `work/task_agnostic/`.

The strongest result is layer-12 Hallmark module mean+SD (51,200D): macro F1
0.8810 ± 0.0268, tissue 10-NN purity 0.7881, and participation ratio 3.73.
This is above raw expression (0.8716) and close to PCA (0.8883), although the
differences are small relative to fold variability. Its RR1↔RR3-39 false-friend
cosine is 0.5099 versus 0.8108 for global mean, while RR3-39 and RR3-40 remain
positive (0.7751 and 0.9224). RR1 remains reversed but is attenuated to −0.4237.

Frozen Ditto-style attention does not materially improve tissue classification
and can increase false similarity. The evidence therefore favors retaining
separate biological-module summaries over replacing global mean pooling with
internal attention weights. The winning representation is large, contains
overlapping Hallmark membership, and remains low-dimensional in effective-rank
terms; it is an analytical readout rather than a new checkpoint.

## Contextual-gene graph fingerprints

The graph extension uses exact layer-12 contextual cosine kNN graphs over all
15,165 fixed gene nodes. Union-symmetrized k=10 is primary; k=5/20 and mutual
kNN are prespecified sensitivities. `pipeline/build_contextual_graphs.py`
caches Top-20 neighbors/weights, `pipeline/validate_contextual_graphs.py` runs
general validation without accessing RR1/RR3 outcomes, and only afterward does
`pipeline/stress_test_contextual_graphs.py` run the NASA stress cases.

General validation supports edge/neighborhood fingerprints: controlled
same-RNA PolyA↔Ribo R@1 is 90.0–96.3% across primary metrics, same-tissue graphs
have 2.29-fold higher study-separated edge Jaccard, and graph degree+strength
reaches 0.8616 ± 0.0287 study-disjoint tissue macro F1. External Hallmark, GO
BP, and Reactome relationships are enriched among edges relative to shuffled
gene labels. Independent gene-identity shuffling collapses k=10 retrieval to
0% R@1. Louvain communities do not discriminate technical pairs and are
reported as a negative result.

With parameters frozen, signed graph-response profiles rank RR3-40 (0.399),
RR3-39 (0.319), and RR1 (0.183) above the unrelated RR1↔RR3-39 false friend
(0.012). Graph similarity has a different scale from vector cosine; the result
is interpreted through ranking rather than raw cross-method magnitude. Graphs
therefore recover contextual organization obscured by global mean pooling, but
do not outperform PCA/Hallmark module pooling on every endpoint.
