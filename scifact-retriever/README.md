# SciFact Hybrid Retriever

Retrieve scientific abstracts for a claim with dense Sentence Transformers embeddings, FastEmbed BM25, or Qdrant reciprocal rank fusion (RRF). An optional cross-encoder reorders the highest-ranked candidates. The library exposes a LangChain `BaseRetriever` for use in an existing LangGraph application.

## Setup

You need Python 3.12 or 3.13, [uv](https://docs.astral.sh/uv/), and Docker with Compose. The default models run on CPU. The first ingestion downloads the pinned BEIR SciFact corpus, model files, and BM25 resources.

```bash
cp .env.example .env
uv sync --locked
docker compose up -d
curl --fail http://127.0.0.1:6333/readyz
uv run scifact-ingest
```

Wait for `/readyz` to succeed before ingesting. Compose binds Qdrant to localhost and persists data in a named volume. Indexing writes 5,183 abstracts on a fresh collection. Repeating the command skips existing papers, and rerunning after an interrupted import resumes the missing batches. The collection becomes available for retrieval when ingestion completes.

## Retrieve evidence

```bash
uv run python examples/retrieve.py "Aspirin inhibits platelet aggregation."
SCIFACT_SEARCH__RERANK=true uv run python examples/retrieve.py "Aspirin inhibits platelet aggregation."
```

```python
from scifact_retriever import SearchOptions, Settings, build_retriever

retriever = build_retriever(Settings(search=SearchOptions(strategy="hybrid", top_k=5, rerank=True)))
claim = "Aspirin inhibits platelet aggregation."
evidence = retriever.invoke(claim)

# Inside an async LangGraph node:
# evidence = await retriever.ainvoke(claim)
```

Build the retriever once at application startup and reuse it. `ainvoke` uses LangChain's worker-thread fallback. Calls on one retriever share a lock around model inference to protect tokenizer state. To manage the connection explicitly, pass your existing `QdrantClient` to `build_retriever(settings, client=client)` and close it at shutdown. If the factory created the client, it is available at `retriever.store.client`.

Each result is a LangChain `Document` with the original abstract in `page_content`, the BEIR paper ID in `id` and `metadata["doc_id"]`, and the paper title in `metadata["title"]`. Available source metadata is preserved. `retrieval_score` holds raw Qdrant cosine, BM25, or RRF scores and `retrieval_strategy` identifies the scale. Reranked results also include `rerank_score` from the cross-encoder. Scores from different strategies use different scales.

## Configuration

See [.env.example](.env.example) for the full set of settings. Environment variables take precedence over `.env` entries.

| Setting | Default | Effect |
|---|---|---|
| `SCIFACT_SEARCH__STRATEGY` | `hybrid` | Use `dense`, `sparse`, or `hybrid` |
| `SCIFACT_SEARCH__TOP_K` | `5` | Maximum returned evidence |
| `SCIFACT_SEARCH__HYBRID_DEPTH` | `50` | Qdrant RRF prefetch depth per leg |
| `SCIFACT_SEARCH__RERANK` | `false` | Enable cross-encoder inference |
| `SCIFACT_SEARCH__RERANK_CANDIDATES` | `10` | Candidate count before reranking |
| `SCIFACT_SEARCH__{DENSE,SPARSE,HYBRID}_MIN_SCORE` | Unset | Filter on the active strategy's raw score before reranking |
| `SCIFACT_SEARCH__EXACT_DENSE_SEARCH` | `true` | Use exact search for the dense query |
| `SCIFACT_DENSE_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Sentence Transformers embedding model |
| `SCIFACT_RERANK_MODEL` | `cross-encoder/ms-marco-TinyBERT-L2-v2` | Single-score Sentence Transformers cross-encoder |

Use a new `SCIFACT_QDRANT_COLLECTION` and reingest when changing the dense model or its revision. Index compatibility checks include the corpus revision, model identity, and sparse IDF configuration. Default model revisions are pinned in code. For a custom Hub model, set its corresponding revision to a commit SHA for reproducibility. Dense and reranking inputs are truncated to their configured model limits. The Linux x86_64 lock uses the CPU PyTorch wheel.

## Architecture

| Layer | Module | Responsibility |
|---|---|---|
| Dataset | `dataset.py` | Load pinned Hugging Face datasets, validate rows, and convert BEIR abstracts to documents |
| Indexing | `indexing.py` | Batch title and abstract embeddings into named Qdrant vectors, use corpus-derived point UUIDs, and mark completed ingestion |
| Retrieval | `retrieval.py` | Query via `langchain-qdrant`, delegate RRF to Qdrant, and return LangChain documents |
| Reranking | `reranking.py` | Score claim and paper pairs with Sentence Transformers |
| Evaluation | `evaluation.py` | Compute macro-averaged retrieval metrics and select candidate depth on development data |
| Configuration | `config.py` | Validate runtime settings and search options with Pydantic |

Qdrant computes the IDF term over the full indexed corpus. The pinned SciFact corpus is the common search space for all strategies and both benchmark splits. Ingestion reads the papers only. CLI events are JSON on stderr, and the evaluation table is printed to stdout.

## Benchmark

[BEIR SciFact](https://huggingface.co/datasets/BeIR/scifact) publishes [train and test relevance judgments](https://huggingface.co/datasets/BeIR/scifact-qrels). The CLI maps BEIR train (809 judged claims) to `dev` and reserves BEIR test (300 judged claims) for final evaluation. Choose a candidate depth on development nDCG@10, with ties favoring the smaller depth. The test command requires the development JSON and checks that the benchmark settings match before loading test judgments.

```bash
uv run scifact-evaluate --split dev --candidate-depths 10 16 --output benchmarks/dev.json
uv run scifact-evaluate --split test --from-dev benchmarks/dev.json --output benchmarks/test.json
```

Each command writes a JSON record with the configuration and a Markdown score table. All runs compare dense, sparse, hybrid, and hybrid with cross-encoder reranking using Recall@5, Recall@10, MRR@10, and nDCG@10. Hybrid and reranked hybrid share the same RRF candidate pool. Test scores reflect the frozen development selection. The benchmark always scores the four variants at their standard cutoffs, independently of the serving strategy and `top_k`.

The measured [development results](benchmarks/dev.md) and [held-out test results](benchmarks/test.md) include all four strategies. The corresponding [development JSON](benchmarks/dev.json) and [test JSON](benchmarks/test.json) record the settings, revisions, and full-precision scores.

## Tests

```bash
uv run pytest -m "not integration"
SCIFACT_TEST_QDRANT_URL=http://127.0.0.1:6333 uv run pytest -m integration
uv run ruff check .
uv run ruff format --check .
```

The integration tests use temporary collections on the running Qdrant server and the real FastEmbed BM25 encoder. Deterministic dense and cross-encoder test doubles keep those tests small. The full benchmark uses the configured models.
