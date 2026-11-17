# Scientific Research MCP Server

Search OpenAlex, arXiv, and Semantic Scholar, resolve paper identifiers, and
explore citation and reference relationships.

Use `uv run --locked research-mcp` for stdio or add `--transport http` to
serve Streamable HTTP on `http://127.0.0.1:8000/mcp`. See `.env.example` for
provider credentials, timeouts, and cache settings.
