# SciFact test benchmark

BEIR test · 300 judged claims · 5183 abstracts

Cross-encoder candidate depth **10**.

| Strategy | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| Dense | 0.7379 | 0.7833 | 0.6047 | 0.6451 |
| Sparse (BM25) | 0.7459 | 0.8088 | 0.6470 | 0.6830 |
| Hybrid (RRF) | 0.7871 | 0.8517 | 0.6829 | 0.7214 |
| Hybrid + cross-encoder | 0.7446 | 0.8517 | 0.6528 | 0.6940 |
