import sys
from unittest.mock import AsyncMock

import pytest
from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from fastmcp.exceptions import ToolError

from research_mcp.documents import DocumentService
from research_mcp.models import PaperIdentifiers, Provider
from research_mcp.providers.arxiv import ArxivProvider
from research_mcp.providers.base import make_paper
from research_mcp.providers.openalex import OpenAlexProvider
from research_mcp.providers.semantic import SemanticScholarProvider
from research_mcp.server import create_server

TOOLS = {
    "search_papers",
    "get_paper",
    "get_paper_content",
    "get_citations",
    "get_references",
    "find_related_papers",
    "search_paper_content",
}


async def test_mcp_registered_tools_and_structured_responses(settings, monkeypatch):
    import pymupdf

    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((72, 72), "Membrane transport measurements.")
        pdf_bytes = pdf.tobytes()
    target = make_paper(
        Provider.ARXIV,
        title="Transport measurements",
        year=2023,
        identifiers=PaperIdentifiers(arxiv="2301.00001", openalex="W123"),
        pdf_urls=["https://arxiv.org/pdf/2301.00001"],
    )
    neighbor = make_paper(Provider.OPENALEX, title="Reference publication", year=2022)
    monkeypatch.setattr(ArxivProvider, "search", AsyncMock(return_value=[target]))
    monkeypatch.setattr(ArxivProvider, "get", AsyncMock(return_value=target))
    monkeypatch.setattr(OpenAlexProvider, "get", AsyncMock(return_value=None))
    monkeypatch.setattr(SemanticScholarProvider, "get", AsyncMock(return_value=None))
    monkeypatch.setattr(OpenAlexProvider, "graph", AsyncMock(return_value=[neighbor]))
    monkeypatch.setattr(
        SemanticScholarProvider, "graph", AsyncMock(return_value=[neighbor])
    )
    monkeypatch.setattr(DocumentService, "_download", AsyncMock(return_value=pdf_bytes))

    async with Client(create_server(settings)) as client:
        assert {tool.name for tool in await client.list_tools()} == TOOLS
        responses = {
            "search_papers": ({"query": "membrane", "providers": ["arxiv"]}, "papers"),
            "get_paper": ({"paper_id": "2301.00001"}, "paper"),
            "get_citations": (
                {"paper_id": "2301.00001", "providers": ["openalex"]},
                "papers",
            ),
            "get_references": ({"paper_id": "2301.00001"}, "papers"),
            "find_related_papers": ({"paper_id": "2301.00001"}, "papers"),
            "get_paper_content": ({"paper_id": "2301.00001", "pages": [1]}, "pages"),
            "search_paper_content": (
                {"paper_id": "2301.00001", "query": "membrane"},
                "passages",
            ),
        }
        for name, (arguments, key) in responses.items():
            result = await client.call_tool(name, arguments)
            assert result.structured_content[key], name
            assert not result.structured_content["issues"], name

        with pytest.raises(ToolError, match="year_from"):
            await client.call_tool(
                "search_papers", {"query": "x", "year_from": 2024, "year_to": 2023}
            )
        with pytest.raises(ToolError):
            await client.call_tool(
                "get_paper_content", {"paper_id": "2301.00001", "pages": [0]}
            )


async def test_console_stdio_transport():
    transport = StdioTransport(
        command=sys.executable, args=["-m", "research_mcp.server"]
    )
    async with Client(transport, timeout=20) as client:
        assert {tool.name for tool in await client.list_tools()} == TOOLS
