# SciFact test benchmark

BEIR test · 300 judged claims · 5183 abstracts

Cross-encoder candidate depth **10**.

| Strategy | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| Dense | 0.7379 | 0.7833 | 0.6047 | 0.6451 |
| Sparse (BM25) | 0.7459 | 0.8088 | 0.6470 | 0.6830 |
| Hybrid (RRF) | 0.7871 | 0.8517 | 0.6763 | 0.7163 |
| Hybrid + cross-encoder | 0.7446 | 0.8517 | 0.6529 | 0.6941 |

Sequential retrieval latency excludes model loading and metric calculation. Hybrid with reranking includes the hybrid search and cross-encoder scoring of up to 10 candidates.

| Strategy | p50 (ms) | p95 (ms) |
|---|---:|---:|
| Dense | 29.4 | 44.2 |
| Sparse (BM25) | 405.5 | 490.5 |
| Hybrid (RRF) | 436.1 | 547.0 |
| Hybrid + cross-encoder | 481.8 | 606.7 |
