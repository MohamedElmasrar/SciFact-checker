# Scientific Research MCP Server

A FastMCP server for searching OpenAlex, arXiv, and Semantic Scholar, exploring citation graphs, and retrieving page-numbered evidence from open PDFs.

## Install and run

Install Python 3.11 or later and [uv](https://docs.astral.sh/uv/getting-started/installation/). From the project directory, run the following commands.

```bash
uv sync --locked
cp .env.example .env
uv run --locked research-mcp
```

The default transport is stdio. An MCP client normally starts the process for you. To run a local Streamable HTTP server instead, use the following command and connect the client to `http://127.0.0.1:8000/mcp`.

```bash
uv run --locked research-mcp --transport http
```

### Connect an MCP client

For a client that supports stdio and an `mcpServers` JSON configuration, substitute the absolute paths to `uv` and the project directory in this example. `--directory` also lets the server load the project's `.env` when the client starts in a different directory.

```json
{
  "mcpServers": {
    "scientific-research": {
      "command": "/absolute/path/to/uv",
      "args": ["--directory", "/absolute/path/to/research-mcp-server", "run", "--locked", "research-mcp"]
    }
  }
}
```

For an HTTP client, select the Streamable HTTP transport and use the `/mcp` endpoint. This server's HTTP endpoint has no built-in authentication. Bind to loopback for local use, or put it behind an authenticated gateway for shared deployments. [FastMCP transports](https://gofastmcp.com/deployment/running-server)

### Docker

```bash
docker build -t scientific-research-mcp .
docker run --rm --env-file .env -p 127.0.0.1:8000:8000 scientific-research-mcp
```

Connect the client to `http://127.0.0.1:8000/mcp`. To let an MCP client launch the container over stdio, use `docker run --rm -i --env-file /absolute/path/to/.env scientific-research-mcp --transport stdio` as its server command.

## MCP tools

| Tool | Arguments and defaults |
| --- | --- |
| `search_papers` | `query`, optional `providers`, inclusive `year_from` and `year_to`, `sort` (`relevance`, `newest`, or `most_cited`), and `limit` (10). Searches all three providers by default. |
| `get_paper` | `paper_id`. Resolves DOI, arXiv ID, OpenAlex work ID, Semantic Scholar paper ID, or `CorpusId:` followed by its numeric ID. Common DOI and paper URLs are accepted. |
| `get_paper_content` | `paper_id`, optional `pages` or `start_page` (1), and optional `max_chars`. Returns Markdown grouped by physical PDF page and the source URL. |
| `get_citations` | `paper_id`, optional `providers`, and `limit` (10). Searches OpenAlex and Semantic Scholar by default. |
| `get_references` | `paper_id`, optional `providers`, and `limit` (10). Searches OpenAlex and Semantic Scholar by default. |
| `find_related_papers` | `paper_id` and `limit` (10). Prioritizes Semantic Scholar recommendations and includes available references and citing papers with relationship labels. |
| `search_paper_content` | `paper_id`, `query`, optional `pages` or `start_page` (1), `limit` (5), and `passage_chars` (600). Returns ranked keyword excerpts with their PDF page numbers. |

Provider values are `openalex`, `arxiv`, and `semanticscholar`. Most result limits accept 1–100. Passage search accepts 1–30 results and 80–2,000 characters per excerpt.

### Example calls

These are MCP `tools/call` parameters. Use the same argument objects in any MCP client.

```json
{"name":"search_papers","arguments":{"query":"transformer attention mechanism","year_from":2017,"sort":"most_cited","limit":5}}
```
```json
{"name":"get_paper","arguments":{"paper_id":"https://arxiv.org/abs/1706.03762"}}
```
```json
{"name":"get_citations","arguments":{"paper_id":"10.48550/arXiv.1706.03762","providers":["openalex"],"limit":5}}
```
```json
{"name":"get_paper_content","arguments":{"paper_id":"1706.03762","pages":[2,4],"max_chars":6000}}
```
```json
{"name":"search_paper_content","arguments":{"paper_id":"1706.03762","query":"scaled dot product attention","start_page":1,"limit":4}}
```

### Interpret results

- Search combines provider rankings and merges exact identifier or normalized title matches. Merged records preserve each contributing provider's identifiers, venue, date, categories, URL, and citation count. The top-level citation count is the largest available count. arXiv has no citation count and contributes relevance-ranked candidates to citation-sorted searches. Sorts and limits apply to the candidates returned by each provider.
- A response can contain both results and `issues`. An unavailable provider leaves results from the others usable. For graph lookups, provider coverage and the presence of shared identifiers determine which relationships can be resolved. Metadata results may have a null abstract or other missing fields.
- PDF tools try available open PDF links in order. Pages are one-based and refer to the returned PDF URL. Explicit selections are deduplicated, sorted, and override `start_page`. For a sequential request, follow `next_page` to continue. If a page exceeds the entire Markdown budget, the server returns a clipped excerpt of that page and marks the result `truncated`. To read more of that page, request it with a higher `max_chars` or use passage search. Large PDFs are subject to download and per-call page limits. Image-only pages may contain no extractable text because OCR is disabled.
- Passage results include the source identifiers, title, PDF URL, and the physical page number. Cite the source PDF and page because pagination may differ between preprints and published versions.

## Configuration

Copy `.env.example` to `.env` or set the same variables in the process environment. Process environment values take precedence. Timeouts, rate intervals, cache lifetimes, and retry delays use seconds.

| Variable | Default or effect |
| --- | --- |
| `RESEARCH_OPENALEX_API_KEY`, `RESEARCH_OPENALEX_EMAIL` | Optional key and contact email. An API key raises the available budget. [OpenAlex API authentication](https://help.openalex.org/api/authentication/) |
| `RESEARCH_SEMANTICSCHOLAR_API_KEY` | Optional key. Shared unauthenticated access can return HTTP 429. [Semantic Scholar API](https://www.semanticscholar.org/product/api) |
| `RESEARCH_OPENALEX_TIMEOUT`, `RESEARCH_ARXIV_TIMEOUT`, `RESEARCH_SEMANTICSCHOLAR_TIMEOUT`, `RESEARCH_PDF_TIMEOUT` | `15`, `20`, `20`, and `30` seconds respectively. |
| `RESEARCH_PDF_TOTAL_TIMEOUT` | `60` seconds across PDF retrieval and extraction, after metadata lookup. `RESEARCH_PDF_TIMEOUT` caps downloads and retries for each candidate URL. |
| `RESEARCH_OPENALEX_INTERVAL`, `RESEARCH_ARXIV_INTERVAL`, `RESEARCH_SEMANTICSCHOLAR_INTERVAL` | Minimum spacing between requests to each provider in one server process. Defaults are `0.2`, `3`, and `1` seconds respectively. [arXiv API user manual](https://info.arxiv.org/help/api/user-manual.html) |
| `RESEARCH_RETRY_ATTEMPTS`, `RESEARCH_RETRY_MAX_DELAY` | `3` total attempts and at most `30` seconds of retry delay. Transient failures use exponential backoff with jitter. The server honors `Retry-After` when the SDK exposes it. |
| `RESEARCH_METADATA_CACHE_SIZE`, `RESEARCH_METADATA_CACHE_TTL` | Up to `512` successful provider metadata records for `3600` seconds. |
| `RESEARCH_SEARCH_CACHE_SIZE`, `RESEARCH_SEARCH_CACHE_TTL` | Up to `128` complete searches for `900` seconds. |
| `RESEARCH_DOCUMENT_CACHE_BYTES`, `RESEARCH_DOCUMENT_CACHE_TTL` | `33554432` bytes shared between downloaded PDFs and extracted Markdown for `3600` seconds. |
| `RESEARCH_PDF_MAX_BYTES`, `RESEARCH_PDF_MAX_PAGES` | `26214400` bytes per PDF and `40` pages per extraction or search. |
| `RESEARCH_CONTENT_DEFAULT_CHARS`, `RESEARCH_CONTENT_MAX_CHARS` | Default of `20000` Markdown characters and a server cap of `100000`. |
| `RESEARCH_LOG_LEVEL` | `INFO`. Logs go to stderr. |

Set a cache size to zero to disable storage for that cache. Caches and rate limits are per process. PDF downloads use direct connections and require public HTTP or HTTPS addresses, including after redirects.

## Development

```bash
uv sync --locked
uv run --locked pytest
```

Tests mock upstream APIs and generate small local PDFs. They also exercise all seven tools through a FastMCP client and verify tool discovery in a stdio subprocess.

The `server.py` module registers tools and validates input. `service.py` coordinates providers and graph traversal, using `aggregation.py` for ranking and deduplication. Provider adapters wrap their respective SDKs. `documents.py` handles retrieval, Markdown conversion, and passage ranking. `config.py` and `infrastructure.py` provide settings, cache storage, throttling, and retry behavior.
