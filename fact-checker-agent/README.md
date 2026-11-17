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

The graph handles up to four claims per request and uses both SciFact abstract passages and research MCP search results. It merges duplicate passages, retains contributing sources, and ranks candidates using their within-source positions. Gemini assesses the ranked passages and assigns one of three verdicts per claim. The graph checks citation IDs against actual retrieved passages and requires directionally relevant cited evidence for a supported or contradicted verdict. The interface shows the linked references, evidence passages and their role in the verdict.

When a source is unavailable, the report includes a notice and can continue with the remaining evidence. Literature verification uses abstracts and scope-limited retrieval. The chat agent can invoke the MCP PDF and citation tools for deeper research.

## Tests

```bash
cd fact-checker-agent
uv run --locked pytest
uv run --locked ruff check src tests
```

Unit and graph integration tests use deterministic model and tool doubles. They run without a Gemini key, Qdrant, or network access.
