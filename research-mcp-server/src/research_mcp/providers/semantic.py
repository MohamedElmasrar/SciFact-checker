from urllib.parse import quote, quote_plus

from semanticscholar import AsyncSemanticScholar
from semanticscholar.ApiRequester import ApiRequester
from semanticscholar.SemanticScholarException import (
    ObjectNotFoundException,
    ServerErrorException,
)
from tenacity import RetryError

from ..config import Settings
from ..identifiers import Identifier, arxiv_base, normalized_id
from ..infrastructure import RequestGate, UpstreamRateLimit, is_transient
from ..models import (
    Author,
    Paper,
    PaperIdentifiers,
    Provider,
    Relationship,
    SearchOptions,
    SortOrder,
)
from .base import distinct, make_paper

FIELDS = [
    "paperId",
    "corpusId",
    "externalIds",
    "url",
    "title",
    "abstract",
    "authors",
    "year",
    "citationCount",
    "venue",
    "publicationVenue",
    "fieldsOfStudy",
    "openAccessPdf",
    "publicationDate",
]


def sdk_identifier(identifier: Identifier) -> str | None:
    if identifier.kind == "semanticscholar":
        return identifier.value
    prefixes = {"doi": "DOI", "arxiv": "ARXIV", "corpus_id": "CorpusId"}
    if identifier.kind in prefixes:
        value = (
            arxiv_base(identifier.value)
            if identifier.kind == "arxiv"
            else identifier.value
        )
        return f"{prefixes[identifier.kind]}:{quote(value, safe='/')}"
    return None


class AuthenticatedRequester(ApiRequester):
    """Supply auth on graph calls and unwrap the SDK's single-attempt retry."""

    def __init__(self, timeout: float, headers: dict):
        super().__init__(timeout=timeout, retry=False)
        self.headers = headers

    async def get_data_async(self, url, parameters, headers, payload=None):
        try:
            data = await super().get_data_async(
                url, parameters, headers or self.headers, payload
            )
        except RetryError as error:
            # The SDK uses ConnectionRefusedError to report HTTP 429 and drops the headers.
            if isinstance(error.last_attempt.exception(), ConnectionRefusedError):
                raise UpstreamRateLimit() from error
            error.reraise()
        if not data:
            raise ServerErrorException("Semantic Scholar returned an empty response")
        # The SDK instantiates a Paper from each nested value, which can be null.
        field = {"citations": "citingPaper", "references": "citedPaper"}.get(
            url.rsplit("/", 1)[-1]
        )
        if field and isinstance(data, dict) and "data" in data:
            data["data"] = [row for row in data["data"] if row.get(field)]
        return data


class SemanticScholarProvider:
    name = Provider.SEMANTICSCHOLAR

    def __init__(self, settings: Settings):
        api_key = settings.semanticscholar_api_key
        self.client = AsyncSemanticScholar(
            api_key=api_key.get_secret_value() if api_key else None,
            timeout=settings.semanticscholar_timeout,
            retry=False,
        )
        self.client._requester = AuthenticatedRequester(
            settings.semanticscholar_timeout, self.client.auth_header
        )
        self.gate = RequestGate(settings.semanticscholar_interval, settings)

    async def _call(self, operation):
        return await self.gate.run(
            operation,
            lambda error: (
                is_transient(error) or isinstance(error, ServerErrorException)
            ),
        )

    async def search(self, options: SearchOptions) -> list[Paper]:
        start, end = options.year_from, options.year_to
        year = f"{start or ''}-{end or ''}" if start or end else None
        sort = {
            SortOrder.NEWEST: "publicationDate:desc",
            SortOrder.MOST_CITED: "citationCount:desc",
        }.get(options.sort)
        # The SDK interpolates query into an already encoded query string.
        page = await self._call(
            lambda: self.client.search_paper(
                quote_plus(options.query),
                year=year,
                fields=FIELDS,
                limit=options.limit,
                bulk=sort is not None,
                sort=sort,
            )
        )
        return self._normalize_many(page.raw_data[: options.limit])

    async def get(self, identifier: Identifier) -> Paper | None:
        value = sdk_identifier(identifier)
        if value is None:
            return None
        try:
            item = await self._call(lambda: self.client.get_paper(value, fields=FIELDS))
        except ObjectNotFoundException:
            return None
        return normalize(item.raw_data)

    async def graph(
        self, identifier: Identifier, relationship: Relationship, limit: int
    ) -> list[Paper]:
        value = sdk_identifier(identifier)
        if value is None:
            return []
        try:
            if relationship == Relationship.RECOMMENDED:
                papers = await self._call(
                    lambda: self.client.get_recommended_papers(
                        value, fields=FIELDS, limit=limit
                    )
                )
                return self._normalize_many(item.raw_data for item in papers)

            method, field = (
                (self.client.get_paper_citations, "citingPaper")
                if relationship == Relationship.CITING
                else (self.client.get_paper_references, "citedPaper")
            )
            page = await self._call(lambda: method(value, fields=FIELDS, limit=limit))
            return self._normalize_many(row.get(field) for row in page.raw_data)
        except ObjectNotFoundException:
            return []

    @staticmethod
    def _normalize_many(items) -> list[Paper]:
        return [
            paper for item in items if item and (paper := normalize(item)) is not None
        ]


def normalize(item: dict) -> Paper | None:
    title = item.get("title")
    if not title or not title.strip():
        return None
    ids = item.get("externalIds") or {}
    pdf = (item.get("openAccessPdf") or {}).get("url")
    return make_paper(
        Provider.SEMANTICSCHOLAR,
        identifiers=PaperIdentifiers(
            doi=normalized_id(ids.get("DOI"), "doi"),
            arxiv=normalized_id(ids.get("ArXiv"), "arxiv"),
            semanticscholar=normalized_id(item.get("paperId"), "semanticscholar"),
            corpus_id=normalized_id(
                item.get("corpusId") or ids.get("CorpusId"), "corpus_id"
            ),
        ),
        title=title.strip(),
        abstract=item.get("abstract") or None,
        authors=[
            Author(name=author["name"])
            for author in item.get("authors") or []
            if author.get("name")
        ],
        year=item.get("year"),
        citation_count=item.get("citationCount"),
        venue=(item.get("publicationVenue") or {}).get("name")
        or item.get("venue")
        or None,
        categories=distinct(item.get("fieldsOfStudy") or []),
        pdf_urls=[pdf] if pdf else [],
        url=item.get("url"),
        publication_date=item.get("publicationDate"),
    )
