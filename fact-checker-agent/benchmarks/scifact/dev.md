# SciFact dev benchmark

BEIR train · 809 judged claims · 5183 abstracts

Cross-encoder candidate depth **10**.

| Strategy | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| Dense | 0.7347 | 0.7856 | 0.6261 | 0.6602 |
| Sparse (BM25) | 0.7458 | 0.8100 | 0.6704 | 0.6992 |
| Hybrid (RRF) | 0.7943 | 0.8463 | 0.6893 | 0.7219 |
| Hybrid + cross-encoder | 0.7757 | 0.8463 | 0.6654 | 0.7041 |

Sequential retrieval latency excludes model loading and metric calculation. Hybrid with reranking includes the hybrid search and cross-encoder scoring of up to 16 candidates.

| Strategy | p50 (ms) | p95 (ms) |
|---|---:|---:|
| Dense | 26.6 | 40.1 |
| Sparse (BM25) | 391.1 | 471.8 |
| Hybrid (RRF) | 424.5 | 504.8 |
| Hybrid + cross-encoder | 490.9 | 594.6 |

Candidate depth was selected by highest development nDCG@10. Ties favor the smaller depth.

| Candidates | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|---:|---:|---:|---:|---:|
| 10 | 0.7757 | 0.8463 | 0.6654 | 0.7041 |
| 16 | 0.7633 | 0.8327 | 0.6561 | 0.6935 |
