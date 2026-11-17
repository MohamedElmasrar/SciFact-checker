# SciFact dev benchmark

BEIR train · 809 judged claims · 5183 abstracts

Cross-encoder candidate depth **10**.

| Strategy | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| Dense | 0.7347 | 0.7856 | 0.6261 | 0.6602 |
| Sparse (BM25) | 0.7458 | 0.8100 | 0.6704 | 0.6992 |
| Hybrid (RRF) | 0.7943 | 0.8439 | 0.6988 | 0.7285 |
| Hybrid + cross-encoder | 0.7744 | 0.8439 | 0.6648 | 0.7031 |

Candidate depth was selected by highest development nDCG@10. Ties favor the smaller depth.

| Candidates | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|---:|---:|---:|---:|---:|
| 10 | 0.7744 | 0.8439 | 0.6648 | 0.7031 |
| 16 | 0.7633 | 0.8340 | 0.6563 | 0.6939 |
