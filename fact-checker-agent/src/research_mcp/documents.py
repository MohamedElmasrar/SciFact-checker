import asyncio
import ipaddress
import logging
import re
import socket
from collections import Counter
from dataclasses import dataclass
from threading import Lock

import httpx
import pymupdf
import pymupdf4llm

from .config import Settings
from .infrastructure import AsyncCache, RequestGate
from .models import (
    ContentResult,
    DocumentSource,
    Issue,
    MarkdownPage,
    Paper,
    Passage,
    PassageResult,
)
from .service import ResearchService

logger = logging.getLogger(__name__)
_PDF_LOCK = Lock()
_REDIRECTS = {301, 302, 303, 307, 308}
_WORDS = re.compile(r"[^\W_]+", re.UNICODE)


class DocumentError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ExtractedDocument:
    total_pages: int
    pages: tuple[MarkdownPage, ...]

    @property
    def cache_size(self) -> int:
        return 128 + sum(
            len(page.markdown.encode("utf-8")) + 100 for page in self.pages
        )


class DocumentService:
    def __init__(self, settings: Settings, research: ResearchService):
        self.settings = settings
        self.research = research
        self.cache = AsyncCache[tuple, bytes | ExtractedDocument](
            settings.document_cache_bytes,
            settings.document_cache_ttl,
            getsizeof=lambda item: (
                len(item) if isinstance(item, bytes) else item.cache_size
            ),
        )
        self.download_slots = asyncio.Semaphore(2)
        self.extraction_gate = asyncio.Lock()

    async def _load(self, paper_id: str, pages: list[int] | None, start_page: int):
        page_key = tuple(sorted(set(pages))) if pages else ()
        page_limit = self.settings.pdf_max_pages
        if len(page_key) > page_limit:
            return (
                None,
                None,
                [
                    Issue(
                        source="pdf",
                        code="page_limit",
                        message=f"Select at most {page_limit} pages per request.",
                    )
                ],
            )
        lookup = await self.research.get(paper_id)
        if lookup.paper is None:
            return None, None, lookup.issues
        paper = lookup.paper
        if not paper.pdf_urls:
            return (
                None,
                None,
                [
                    *lookup.issues,
                    Issue(
                        source="pdf",
                        code="pdf_unavailable",
                        message="No open PDF link is available for this paper.",
                    ),
                ],
            )

        try:
            async with asyncio.timeout(self.settings.pdf_total_timeout):
                return await self._load_pdf(
                    paper, page_key, start_page, page_limit, lookup.issues
                )
        except TimeoutError:
            return (
                None,
                None,
                [
                    *lookup.issues,
                    Issue(
                        source="pdf",
                        code="pdf_timeout",
                        message="PDF retrieval exceeded the total time limit.",
                    ),
                ],
            )

    async def _load_pdf(
        self,
        paper: Paper,
        page_key: tuple[int, ...],
        start_page: int,
        page_limit: int,
        issues: list[Issue],
    ):
        last_error = None
        for url in paper.pdf_urls:
            # Closure defaults bind the URL even if a caller cancels while a load is shared.
            async def extract(pdf_url=url):
                data = await self.cache.get(
                    ("download", pdf_url), lambda: self._download_with_retries(pdf_url)
                )
                async with self.extraction_gate:
                    return await asyncio.to_thread(
                        extract_pages, data, page_key, start_page, page_limit
                    )

            try:
                document = await self.cache.get(
                    ("pages", url, page_key, start_page), extract
                )
                source = DocumentSource(
                    identifiers=paper.identifiers, title=paper.title, pdf_url=url
                )
                return source, document, issues
            except TimeoutError:
                last_error = Issue(
                    source="pdf",
                    code="pdf_timeout",
                    message="The PDF download timed out.",
                )
            except DocumentError as error:
                last_error = Issue(source="pdf", code=error.code, message=str(error))
            except (
                httpx.HTTPError,
                httpx.InvalidURL,
                OSError,
                ValueError,
                RuntimeError,
            ) as error:
                logger.warning(
                    "PDF download or extraction failed (%s)", type(error).__name__
                )
                last_error = Issue(
                    source="pdf",
                    code="pdf_unavailable",
                    message="Could not download or read the PDF.",
                )
        return None, None, [*issues, last_error]

    async def content(
        self,
        paper_id: str,
        pages: list[int] | None,
        start_page: int,
        max_chars: int | None,
    ) -> ContentResult:
        source, document, issues = await self._load(paper_id, pages, start_page)
        if document is None:
            return ContentResult(issues=issues)

        budget = min(
            max_chars or self.settings.content_default_chars,
            self.settings.content_max_chars,
        )
        result = []
        truncated = False
        for page in document.pages:
            if len(page.markdown) > budget:
                truncated = True
                if not result:
                    result.append(
                        page.model_copy(update={"markdown": page.markdown[:budget]})
                    )
                break
            result.append(page)
            budget -= len(page.markdown)

        upcoming = next_page(
            document, pages, result[-1].page if result else start_page - 1
        )
        return ContentResult(
            source=source,
            total_pages=document.total_pages,
            pages=result,
            next_page=upcoming,
            truncated=truncated or upcoming is not None,
            issues=issues,
        )

    async def search(
        self,
        paper_id: str,
        query: str,
        pages: list[int] | None,
        start_page: int,
        limit: int,
        passage_chars: int,
    ) -> PassageResult:
        if not _WORDS.search(query):
            return PassageResult(
                issues=[
                    Issue(
                        source="pdf",
                        code="invalid_query",
                        message="Provide at least one word to search.",
                    )
                ]
            )
        source, document, issues = await self._load(paper_id, pages, start_page)
        if document is None:
            return PassageResult(issues=issues)
        return PassageResult(
            source=source,
            total_pages=document.total_pages,
            searched_pages=[page.page for page in document.pages],
            next_page=next_page(document, pages, document.pages[-1].page),
            passages=find_passages(document, query, limit, passage_chars),
            issues=issues,
        )

    async def _download_with_retries(self, url: str) -> bytes:
        async with self.download_slots, asyncio.timeout(self.settings.pdf_timeout):
            return await RequestGate(0, self.settings).run(lambda: self._download(url))

    async def _download(self, url: str) -> bytes:
        async with asyncio.timeout(self.settings.pdf_timeout):
            for _ in range(6):
                target = httpx.URL(url)
                addresses = await public_addresses(target)
                # A fresh client per redirect avoids reusing a pinned connection for another host.
                async with httpx.AsyncClient(
                    trust_env=False, timeout=self.settings.pdf_timeout
                ) as client:
                    target_url = None
                    for ip in addresses:
                        try:
                            # Pin the connection to a validated address, retaining HTTP Host and TLS SNI.
                            async with client.stream(
                                "GET",
                                target.copy_with(host=ip),
                                headers={
                                    "Host": target.netloc.decode("ascii"),
                                    "Accept": "application/pdf, */*",
                                },
                                extensions={"sni_hostname": target.host},
                            ) as response:
                                if (
                                    response.status_code in _REDIRECTS
                                    and response.headers.get("Location")
                                ):
                                    target_url = str(
                                        target.join(response.headers["Location"])
                                    )
                                    break
                                response.raise_for_status()
                                return await self._read_pdf(response)
                        except (httpx.ConnectError, httpx.ConnectTimeout):
                            if ip == addresses[-1]:
                                raise
                    if target_url is None:
                        break
                    url = target_url
        raise DocumentError(
            "invalid_pdf_url", "The PDF link redirected too many times."
        )

    async def _read_pdf(self, response: httpx.Response) -> bytes:
        limit = self.settings.pdf_max_bytes
        length = response.headers.get("Content-Length", "")
        if length.isdecimal() and len(length) < 20 and int(length) > limit:
            raise DocumentError(
                "pdf_too_large", "The PDF exceeds the configured download size limit."
            )
        data = bytearray()
        async for chunk in response.aiter_bytes(chunk_size=65536):
            data.extend(chunk)
            if len(data) > limit:
                raise DocumentError(
                    "pdf_too_large",
                    "The PDF exceeds the configured download size limit.",
                )
            if len(data) >= 1024 and b"%PDF-" not in data[:1024]:
                raise DocumentError("invalid_pdf", "The link did not return a PDF.")
        if b"%PDF-" not in data[:1024]:
            raise DocumentError("invalid_pdf", "The link did not return a PDF.")
        return bytes(data)


async def public_addresses(url: httpx.URL) -> list[str]:
    port = {"http": 80, "https": 443}.get(url.scheme)
    if (
        not port
        or not url.host
        or url.userinfo
        or (url.port is not None and url.port != port)
    ):
        raise DocumentError(
            "invalid_pdf_url", "The PDF must use a public HTTP or HTTPS address."
        )
    results = await asyncio.get_running_loop().getaddrinfo(
        url.raw_host.decode("ascii"), port, type=socket.SOCK_STREAM
    )
    addresses = list(dict.fromkeys(result[4][0] for result in results))
    if not addresses or any(
        not is_public(ipaddress.ip_address(ip)) for ip in addresses
    ):
        raise DocumentError(
            "invalid_pdf_url", "The PDF must use a public HTTP or HTTPS address."
        )
    return addresses


def is_public(address) -> bool:
    mapped = getattr(address, "ipv4_mapped", None)
    if mapped:
        address = mapped
    return address.is_global and not address.is_multicast


def extract_pages(
    data: bytes, selected: tuple[int, ...], start: int, cap: int
) -> ExtractedDocument:
    # PyMuPDF does not support concurrent use from threads.
    with _PDF_LOCK, pymupdf.open(stream=data, filetype="pdf") as pdf:
        total = pdf.page_count
        if pdf.is_encrypted:
            raise DocumentError("encrypted_pdf", "The PDF requires a password.")
        pages = selected or tuple(range(start, min(start + cap, total + 1)))
        if not pages or pages[-1] > total:
            raise DocumentError("invalid_pages", f"The PDF contains {total} pages.")
        chunks = pymupdf4llm.to_markdown(
            pdf,
            pages=[page - 1 for page in pages],
            page_chunks=True,
            show_progress=False,
            use_ocr=False,
        )
        if len(chunks) != len(pages):
            raise DocumentError(
                "extraction_failed",
                "The PDF converter did not return the requested pages.",
            )
        markdown = tuple(
            MarkdownPage(page=number, markdown=chunk["text"].strip())
            for number, chunk in zip(pages, chunks, strict=True)
        )
        return ExtractedDocument(total, markdown)


def next_page(
    document: ExtractedDocument, selected: list[int] | None, last: int
) -> int | None:
    return last + 1 if not selected and last < document.total_pages else None


def find_passages(
    document: ExtractedDocument, query: str, limit: int, size: int
) -> list[Passage]:
    query_tokens = _WORDS.findall(query.casefold())
    phrase = " ".join(query_tokens)
    terms = set(query_tokens)
    matches = []
    for page in document.pages:
        for paragraph in re.split(r"\n\s*\n", page.markdown):
            text = " ".join(paragraph.split())
            words = list(_WORDS.finditer(text))
            matching = [word for word in words if word.group().casefold() in terms]
            if not matching:
                continue

            # Score bounded excerpts around matches, including matches late in long paragraphs.
            seen = set()
            for word in matching:
                start = max(0, min(word.start() - size // 4, len(text) - size))
                if start in seen:
                    continue
                seen.add(start)
                passage = text[start : start + size]
                tokens = _WORDS.findall(passage.casefold())
                counts = Counter(tokens)
                score = sum(term in counts for term in terms) / len(terms)
                score += 0.15 * sum(min(3, counts[term]) for term in terms) / len(terms)
                if f" {phrase} " in f" {' '.join(tokens)} ":
                    score += 1
                matches.append((score, page.page, text, start, passage))

    passages = []
    chosen = {}
    for score, page, paragraph, start, passage in sorted(
        matches, key=lambda match: -match[0]
    ):
        positions = chosen.setdefault((page, paragraph), [])
        if any(abs(start - position) < size // 2 for position in positions):
            continue
        positions.append(start)
        passages.append(Passage(page=page, text=passage, score=round(score, 3)))
        if len(passages) == limit:
            break
    return passages
