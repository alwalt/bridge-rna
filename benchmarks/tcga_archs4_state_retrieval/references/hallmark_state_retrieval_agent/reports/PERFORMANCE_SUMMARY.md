# Performance summary

| Evaluation | Metric | Queries | Hallmark-50 | Bridge-512 | Hallmark − Bridge | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| TCGA | Mean metadata keyword precision@10 | 22 | 0.777 | 0.809 | −0.032 | Bridge-512 better overall |
| TCGA | Queries won | 22 | 6 | 11 | — | Five queries tied |
| OSDR liver | Liver metadata precision@10 | 28 | 0.061 | 0.986 | −0.925 | Bridge-512 substantially better |
| OSDR liver | Mean top-10 cosine similarity | 28 | 0.632 | 0.972 | −0.341 | Raw cosine magnitudes are not calibrated across spaces |
| OSDR liver | Unique retrieved ARCHS4 studies | 28 | 3 | 22 | −19 | Hallmark retrieval was strongly concentrated |

TCGA Hallmark-50 improved 6 of 22 queries, tied 5, and lost 11. Its clearest gains were KIRC (+0.30 precision@10), BRCA (+0.10), and the LUAD and LUSC cohort centroids (+0.10 each). The aggregate result remained negative.

The ARCHS4 database contains 510,709 human samples. The Hallmark database has 50 dimensions and was produced from the frozen 512-D Bridge embeddings without retraining or new Bridge inference.
