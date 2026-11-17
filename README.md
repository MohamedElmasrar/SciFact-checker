# Scientific Fact Checker

A research chatbot that searches scientific literature and verifies claims against retrieved evidence. It uses Gemini for conversation and assessment, LangGraph for streaming and thread history, and Qdrant for hybrid SciFact retrieval. Verification reports show what the evidence supports, contradicts, or leaves unresolved.

| Component | Role |
| --- | --- |
| [Agent Chat UI](agent-chat-ui/README.md) | Fork of LangChain's chat frontend with streamed research progress, paper references, and fact-check reports |
| [Fact Checker Agent](fact-checker-agent/README.md) | Gemini conversational agent and claim-verification graph |
| [Research MCP Server](research-mcp-server/README.md) | Paper search, citation exploration, and PDF passages from scientific literature |
| [SciFact Retriever](scifact-retriever/README.md) | Dense and BM25 retrieval over BEIR SciFact, with optional reranking |

## Run locally

You need Python 3.12, [uv](https://docs.astral.sh/uv/), Node.js 20.19 or later, pnpm, Docker with Compose, and a Gemini API key. Start each step from the repository root in a separate terminal. Keep any existing local environment files when configuring a component.

1. Start Qdrant. Once the readiness check succeeds, index SciFact. If you already have an indexed collection, point the agent at it and skip this step.

   ```bash
   cd scifact-retriever
   docker compose up -d
   curl --fail http://127.0.0.1:6333/readyz
   uv run --locked scifact-ingest
   ```

   The first import downloads the pinned SciFact corpus and embedding models. If you customize the collection or model, use the same settings in the retriever and agent.

2. Start the research MCP server at `http://127.0.0.1:8000/mcp`.

   ```bash
   cd research-mcp-server
   uv run --locked research-mcp --transport http
   ```

   OpenAlex and Semantic Scholar credentials are optional. Set them in the research server's own `.env` using its `.env.example`. To use stdio instead, set `RESEARCH_MCP_TRANSPORT=stdio` in the agent's `.env` and skip the HTTP server command.

3. In `fact-checker-agent`, copy `.env.example` to `.env` if necessary and set `GOOGLE_API_KEY` or `GEMINI_API_KEY`. Start the graph server.

   ```bash
   cd fact-checker-agent
   uv sync --locked
   uv run langgraph dev
   ```

   LangGraph serves the `fact_checker` assistant at `http://localhost:2024`.

4. Start the frontend. Copy the example configuration only if `.env.local` does not already exist.

   ```bash
   cd agent-chat-ui
   test -f .env.local || cp .env.example .env.local
   pnpm install --frozen-lockfile
   pnpm dev
   ```

   Open [the local chat](http://localhost:3000) and try “Find three papers about the transformer attention mechanism” or “Fact-check whether the original Transformer uses positional encodings.”

## Checks

Run `uv run --locked pytest` from each Python component directory. To include the SciFact integration tests, set `SCIFACT_TEST_QDRANT_URL=http://127.0.0.1:6333` with Qdrant running. From `agent-chat-ui`, run `pnpm lint` and `pnpm build` after installing dependencies.

Commit the `.env.example` templates and dependency lockfiles. Put keys in local environment files, which Git ignores.
