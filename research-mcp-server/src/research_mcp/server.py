import argparse
import logging
from typing import Annotated

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from .config import Settings
from .models import (
    GraphProvider,
    PaperID,
    PaperList,
    PaperResult,
    Provider,
    Query,
    RelatedResult,
    Relationship,
    ResultLimit,
    SearchOptions,
    SortOrder,
    Year,
)
from .service import ResearchService

GraphProviders = Annotated[list[GraphProvider], Field(min_length=1, max_length=2)]
SearchProviders = Annotated[list[Provider], Field(min_length=1, max_length=3)]
READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=True)


def create_server(settings: Settings) -> FastMCP:
    research = ResearchService(settings)
    server = FastMCP(
        "Scientific Research",
        instructions="Search scientific literature, resolve paper identifiers, and explore citations.",
    )

    @server.tool(annotations=READ_ONLY)
    async def search_papers(
        query: Query,
        providers: SearchProviders | None = None,
        year_from: Year | None = None,
        year_to: Year | None = None,
        sort: SortOrder = SortOrder.RELEVANCE,
        limit: ResultLimit = 10,
    ) -> PaperList:
        """Search selected providers concurrently and return ranked, deduplicated publication metadata."""
        if year_from and year_to and year_from > year_to:
            raise ToolError("year_from must be less than or equal to year_to")
        options = SearchOptions(
            query=query,
            providers=providers or list(Provider),
            year_from=year_from,
            year_to=year_to,
            sort=sort,
            limit=limit,
        )
        return await research.search(options)

    @server.tool(annotations=READ_ONLY)
    async def get_paper(paper_id: PaperID) -> PaperResult:
        """Resolve a DOI, arXiv ID, OpenAlex work ID, or Semantic Scholar paper ID."""
        return await research.get(paper_id)

    @server.tool(annotations=READ_ONLY)
    async def get_citations(
        paper_id: PaperID,
        providers: GraphProviders | None = None,
        limit: ResultLimit = 10,
    ) -> PaperList:
        """Return papers that cite this paper. Search both graph providers when providers is omitted."""
        return await research.graph(
            paper_id, Relationship.CITING, providers or list(GraphProvider), limit
        )

    @server.tool(annotations=READ_ONLY)
    async def get_references(
        paper_id: PaperID,
        providers: GraphProviders | None = None,
        limit: ResultLimit = 10,
    ) -> PaperList:
        """Return works referenced by this paper. Search both graph providers when providers is omitted."""
        return await research.graph(
            paper_id, Relationship.REFERENCE, providers or list(GraphProvider), limit
        )

    @server.tool(annotations=READ_ONLY)
    async def find_related_papers(
        paper_id: PaperID, limit: ResultLimit = 10
    ) -> RelatedResult:
        """Return Semantic Scholar recommendations and citation neighbors with their relationship labels."""
        return await research.related(paper_id, limit)

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Scientific Research MCP server")
    parser.add_argument("--transport", choices=("stdio", "http"), default="stdio")
    parser.add_argument(
        "--host", default="127.0.0.1", help="Bind address for HTTP transport"
    )
    parser.add_argument(
        "--port", default=8000, type=int, help="Port for HTTP transport"
    )
    args = parser.parse_args()
    settings = Settings()
    logging.basicConfig(
        level=settings.log_level, format="%(levelname)s %(name)s %(message)s"
    )
    # These SDKs can include upstream request data in DEBUG logs.
    for name in ("arxiv", "pyalex", "semanticscholar", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)

    server = create_server(settings)
    if args.transport == "http":
        server.run(transport="http", host=args.host, port=args.port, show_banner=False)
    else:
        server.run(transport="stdio", show_banner=False)


if __name__ == "__main__":
    main()
