from unittest.mock import AsyncMock

import httpx
import pymupdf
import pytest

from research_mcp.documents import (
    DocumentError,
    DocumentService,
    ExtractedDocument,
    find_passages,
    public_addresses,
)
from research_mcp.models import MarkdownPage, Paper, PaperIdentifiers, PaperResult


@pytest.fixture
def pdf_bytes():
    with pymupdf.open() as document:
        for number, name in enumerate(
            ("introductory findings", "membrane capacitance", "statistical appendix"), 1
        ):
            page = document.new_page()
            page.insert_text((72, 72), f"Page {number}: {name} and study results.")
        return document.tobytes()


def repository(*urls):
    paper = Paper(
        title="Experimental Study",
        identifiers=PaperIdentifiers(arxiv="2301.00001"),
        pdf_urls=list(urls),
    )
    return AsyncMock(get=AsyncMock(return_value=PaperResult(paper=paper)))


async def test_pdf_extraction_pagination_budgets_search_and_cache(
    settings, pdf_bytes, monkeypatch
):
    service = DocumentService(
        settings.model_copy(update={"pdf_max_pages": 2}),
        repository("https://example.org/main"),
    )
    download = AsyncMock(return_value=pdf_bytes)
    monkeypatch.setattr(service, "_download", download)

    contents = await service.content("2301.00001", None, 1, None)
    assert [page.page for page in contents.pages] == [1, 2]
    assert contents.next_page == 3 and contents.total_pages == 3 and contents.truncated
    assert "capacitance" in contents.pages[1].markdown

    passages = await service.search("2301.00001", "membrane", None, 1, 4, 90)
    assert passages.source.identifiers.arxiv == "2301.00001"
    assert passages.source.pdf_url == "https://example.org/main"
    assert passages.searched_pages == [1, 2] and passages.next_page == 3
    assert passages.passages[0].page == 2

    next_contents = await service.content("2301.00001", None, contents.next_page, None)
    assert [page.page for page in next_contents.pages] == [3]
    assert not next_contents.truncated

    clipped = await service.content("2301.00001", [3, 1, 3], 2, 9)
    assert [page.page for page in clipped.pages] == [1]
    assert len(clipped.pages[0].markdown) == 9 and clipped.truncated
    assert clipped.next_page is None
    assert download.await_count == 1


async def test_pdf_fallback_and_invalid_pages(settings, pdf_bytes, monkeypatch):
    urls = ("https://example.org/broken", "https://example.org/working")
    service = DocumentService(settings, repository(*urls))

    async def download(url):
        return pdf_bytes if url == urls[1] else b"broken"

    monkeypatch.setattr(service, "_download", download)
    content = await service.content("2301.00001", [1], 1, None)
    assert content.source.pdf_url == urls[1] and not content.issues
    empty = await service.content("2301.00001", [4], 1, None)
    assert empty.source is None and empty.issues[-1].code == "invalid_pages"
    missing = await DocumentService(settings, repository()).content(
        "2301.00001", None, 1, None
    )
    assert missing.issues[0].code == "pdf_unavailable"


async def test_download_stream_checks_size_type_and_redirect(
    settings, pdf_bytes, monkeypatch
):
    seen = []

    def handle(request):
        seen.append(
            (
                request.url.host,
                request.headers["host"],
                request.extensions["sni_hostname"],
            )
        )
        if request.url.path == "/redirect":
            return httpx.Response(
                302, headers={"location": "https://mirror.example.org/pdf"}
            )
        if request.url.path == "/notpdf":
            return httpx.Response(200, content=b"<html>Unavailable</html>")
        return httpx.Response(200, content=pdf_bytes)

    addresses = AsyncMock(return_value=["8.8.8.8"])
    monkeypatch.setattr("research_mcp.documents.public_addresses", addresses)
    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs),
    )
    documents = DocumentService(settings, repository())
    assert await documents._download("https://example.org/redirect") == pdf_bytes
    assert seen == [
        ("8.8.8.8", "example.org", "example.org"),
        ("8.8.8.8", "mirror.example.org", "mirror.example.org"),
    ]

    with pytest.raises(DocumentError, match="did not return a PDF"):
        await documents._download("https://example.org/notpdf")
    settings.pdf_max_bytes = 16
    with pytest.raises(DocumentError, match="size limit"):
        await documents._download("https://example.org/pdf")


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/pdf",
        "http://[::1]/pdf",
        "http://169.254.169.254/pdf",
        "ftp://example.org/pdf",
        "https://user:pass@example.org/pdf",
    ],
)
async def test_document_network_rejects_unsafe_targets(url):
    with pytest.raises(DocumentError, match="public HTTP"):
        await public_addresses(httpx.URL(url))


def test_passage_ranking_and_excerpt_around_late_hit():
    document = ExtractedDocument(
        2,
        (
            MarkdownPage(page=1, markdown="Cells have capacitance."),
            MarkdownPage(
                page=2,
                markdown=("Background information. " * 20)
                + "Membrane capacitance is discussed.",
            ),
        ),
    )
    passages = find_passages(document, "membrane capacitance", 2, 85)
    assert [passage.page for passage in passages] == [2, 1]
    assert "Membrane capacitance" in passages[0].text and len(passages[0].text) <= 85


async def test_pdf_timeout_applies_across_fallback_links(settings, monkeypatch):
    import asyncio

    settings.pdf_timeout = 0.1
    settings.pdf_total_timeout = 0.02
    service = DocumentService(
        settings, repository("https://example.org/slow", "https://example.org/next")
    )
    started = []

    async def slow_download(url):
        started.append(url)
        await asyncio.sleep(10)

    monkeypatch.setattr(service, "_download", slow_download)
    result = await asyncio.wait_for(service.content("2301.00001", None, 1, None), 1)
    assert not result.pages
    assert result.issues[-1].code == "pdf_timeout"
    assert started == ["https://example.org/slow"]


async def test_pdf_download_timeout_allows_next_link(settings, pdf_bytes, monkeypatch):
    import asyncio

    settings.pdf_timeout = 0.02
    settings.pdf_total_timeout = 2
    urls = ("https://example.org/slow", "https://example.org/next")
    service = DocumentService(settings, repository(*urls))

    async def download(url):
        if url == urls[0]:
            await asyncio.sleep(10)
        return pdf_bytes

    monkeypatch.setattr(service, "_download", download)
    result = await service.content("2301.00001", [2], 1, None)
    assert result.source.pdf_url == urls[1]
    assert result.pages[0].page == 2
