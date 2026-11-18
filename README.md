# Scientific Fact Checker

A research chatbot that searches scientific literature and verifies claims against retrieved evidence. It uses Gemini for conversation and assessment, LangGraph for streaming and thread history, and Qdrant for hybrid SciFact retrieval. Verification reports show what the evidence supports, contradicts, or leaves unresolved.

| Component | Role |
| --- | --- |
| [Agent Chat UI](agent-chat-ui/README.md) | Fork of LangChain's chat frontend with streamed research progress, paper references, and fact-check reports |
| [Fact Checker Agent](fact-checker-agent/README.md) | Gemini conversational agent and claim-verification graph |
| Research MCP | Scientific paper search, citation exploration, and PDF passages |
| SciFact retriever | Dense and BM25 retrieval over the indexed BEIR SciFact corpus |

The agent, research server module, and retriever are installed together from
`fact-checker-agent`.

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
The LangChain MCP adapter launches the packaged research module over stdio
when the agent needs it. SciFact retrieval opens the local Qdrant index in
the agent process.

To use the chat UI, run these commands from the repository root in another
terminal. The UI requires Node.js 20.19 or later and pnpm.

```bash
cd agent-chat-ui
test -f .env.local || cp .env.example .env.local
pnpm install --frozen-lockfile
pnpm dev
```

Open [the local chat](http://localhost:3000) and try “Find three papers about
the transformer attention mechanism.”

## Checks

Run `uv run --locked pytest` from `fact-checker-agent` to test all three
Python modules. The retriever integration tests use a temporary on-disk index.
From `agent-chat-ui`, run `pnpm lint` and `pnpm build` after installing dependencies.

The [CI workflow](.github/workflows/ci.yml) also runs the [frozen verification evaluation](fact-checker-agent/evals/README.md) and a Playwright test against LangGraph with local Gemini and research fixtures. To run the browser test locally, install Chromium with `pnpm exec playwright install chromium` and run `pnpm exec playwright test` from `agent-chat-ui`.

Commit the `.env.example` templates and dependency lockfiles. Put keys in local environment files, which Git ignores.
