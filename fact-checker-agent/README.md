# Scientific Fact Checker

The existing Agent Chat UI streams conversations from the `fact_checker` LangGraph assistant. A Gemini chat model selects research tools as needed. The `fact_check` tool runs a separate graph that decomposes claims, looks up evidence, and returns a structured report.

## Run locally

Follow the [repository setup guide](../README.md#run-locally) to start Qdrant, the research MCP server, this agent, and the chat UI. The agent's `.env.example` lists the Gemini model, service addresses, timeouts, and retrieval settings. Use `GEMINI_MODEL` to choose a model and `SCIFACT_SEARCH__RERANK` to enable the retriever's optional reranking.

## Try a conversation

- General chat — “What's the difference between a hypothesis and a theory?”
- Literature search — “Find three papers about the transformer attention mechanism.”
- Corpus search — “Search SciFact for passages on barometric pressure and boiling point.”
- Verification — “Fact-check whether positional encodings are required for the original Transformer to use word order.”
- Follow-up — “Which source in that report addresses position most directly?”

## How verification works

The graph handles up to four claims per request. It retrieves SciFact passages and research MCP abstracts in parallel for each claim. When full text could resolve a material uncertainty, it asks the existing MCP passage tool for up to two passages from one paper. Supported and contradicted verdicts require a direct quote that matches the cited passage. The UI links PDF citations to the retrieved page.

When one source fails, the verifier can use the remaining evidence and show a notice. Claims that time out or cannot be assessed have no verdict and can be retried independently. `FACT_CHECK_TIMEOUT_SECONDS` sets the deadline for claim extraction, retrieval, and verification. The default is 90 seconds.

## Tests

```bash
cd fact-checker-agent
uv run --locked pytest
uv run --locked ruff check src tests evals
```

Unit and graph integration tests use deterministic model and tool doubles. They run without a Gemini key, Qdrant, or network access.

See the [verification evaluation](evals/README.md) for fixture results and the opt-in live Gemini command.
