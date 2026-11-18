# Scientific Fact Checker

A research chatbot that searches scientific literature and verifies claims against retrieved evidence. It uses Gemini for conversation and assessment, LangGraph for streaming and thread history, and Qdrant for hybrid SciFact retrieval. Verification reports show what the evidence supports, contradicts, or leaves unresolved.

| Component | Role |
| --- | --- |
| [Agent Chat UI](agent-chat-ui/README.md) | Fork of LangChain's chat frontend with streamed research progress, paper references, and fact-check reports |
| [Fact Checker Agent](fact-checker-agent/README.md) | Gemini agent and LangGraph claim-verification workflow |
| [Research MCP](fact-checker-agent/src/research_mcp) | Paper search, citation exploration, and PDF passages from OpenAlex, arXiv, and Semantic Scholar |
| [SciFact retriever](fact-checker-agent/src/scifact_retriever) | Dense and BM25 search over the BEIR SciFact corpus, stored in embedded Qdrant |

The three Python modules share one package and lockfile in `fact-checker-agent`.
LangGraph starts the research server on demand over stdio through the LangChain
MCP adapter. The agent opens the file-backed Qdrant index in its own process.

## Run locally

Use Python 3.12, [uv](https://docs.astral.sh/uv/), and a Gemini API key. From
the repository root, set up the agent once.

```bash
cd fact-checker-agent
test -f .env || cp .env.example .env
uv sync --locked
```

Set `GOOGLE_API_KEY` or `GEMINI_API_KEY` in `.env`. You can add optional
`RESEARCH_OPENALEX_API_KEY` and `RESEARCH_SEMANTICSCHOLAR_API_KEY` settings
in the same file.

Build the local SciFact index once if you do not already have it. The first
import downloads the pinned corpus and embedding models. Qdrant stores the
index under `~/.local/share/scifact-retriever/qdrant` by default. Stop the
agent before opening the same file-backed index from another process.

```bash
uv run --locked scifact-ingest
```

Start the agent from `fact-checker-agent`.

```bash
uv run --locked langgraph dev
```

LangGraph serves the `fact_checker` assistant at `http://localhost:2024`.

To use the chat UI, run these commands from the repository root in another
terminal. The UI requires Node.js 20.19 or later and pnpm.

```bash
cd agent-chat-ui
test -f .env.local || cp .env.example .env.local
pnpm install --frozen-lockfile
pnpm dev
```

Open the chat and try one of these prompts. [Local chat](http://localhost:3000)

- **Paper search** — Find three papers about the Transformer attention mechanism.
- **Local SciFact retrieval** — Use `retrieve_scifact` to find evidence for the
  claim that aspirin inhibits platelet aggregation. Quote a relevant passage and
  cite its source.
- **Fact-check** — Verify whether the original Transformer used recurrence or
  convolution. Cite the passages that support the verdict.

## Checks

Run `uv run --locked pytest` from `fact-checker-agent` to test all three
Python modules. The retriever integration tests use a temporary on-disk index.
From `agent-chat-ui`, run `pnpm lint` and `pnpm build` after installing dependencies.

The [CI workflow](.github/workflows/ci.yml) also runs the [frozen verification evaluation](fact-checker-agent/evals/README.md) and a Playwright test against LangGraph with local Gemini and research fixtures. To run the browser test locally, install Chromium with `pnpm exec playwright install chromium` and run `pnpm exec playwright test` from `agent-chat-ui`.

Commit the `.env.example` templates and dependency lockfiles. Put keys in local environment files, which Git ignores.
