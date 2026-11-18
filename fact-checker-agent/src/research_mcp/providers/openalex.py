import asyncio
from collections.abc import Callable

import pyalex
import requests
from pyalex.api import OpenAlexAuth

from ..config import Settings
from ..identifiers import Identifier, arxiv_base, normalized_id
from ..infrastructure import RequestGate, TimeoutSession, status_code
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


class Works(pyalex.Works):
    """Keep the SDK's query builder and supply a timeout-aware session."""

    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings

    def _get_from_url(self, url, session=None):
        config = type(pyalex.config)(pyalex.config)
        key = self.settings.openalex_api_key
        config.update(
            api_key=key.get_secret_value() if key else None,
            email=self.settings.openalex_email,
        )
        with _OpenAlexSession(self.settings.openalex_timeout) as client:
            client.auth = OpenAlexAuth(config)
            return super()._get_from_url(url, session=client)


class _OpenAlexSession(TimeoutSession):
    def get(self, url, *, auth):
        # PyAlex supplies global auth per request. Use this instance's settings.
        return super().get(url)


class OpenAlexProvider:
    name = Provider.OPENALEX

    def __init__(self, settings: Settings):
        self.settings = settings
        self.gate = RequestGate(settings.openalex_interval, settings)

    async def _call(self, operation: Callable):
        return await self.gate.run(lambda: asyncio.to_thread(operation))

    async def search(self, options: SearchOptions) -> list[Paper]:
        def request():
            works = Works(self.settings).search(options.query)
            dates = {}
            if options.year_from:
                dates["from_publication_date"] = f"{options.year_from}-01-01"
            if options.year_to:
                dates["to_publication_date"] = f"{options.year_to}-12-31"
            if dates:
                works.filter(**dates)
            if options.sort == SortOrder.NEWEST:
                works.sort(publication_date="desc")
            elif options.sort == SortOrder.MOST_CITED:
                works.sort(cited_by_count="desc")
            return works.get(per_page=options.limit)

        return self._normalize_many(await self._call(request))

    async def _get_work(self, identifier: Identifier):
        if identifier.kind == "openalex":
            value = identifier.value
        elif identifier.kind == "doi":
            value = f"https://doi.org/{identifier.value}"
        elif identifier.kind == "arxiv":
            value = f"https://doi.org/10.48550/arxiv.{arxiv_base(identifier.value)}"
        else:
            return None
        try:
            return await self._call(lambda: Works(self.settings)[value])
        except requests.HTTPError as error:
            if status_code(error) == 404:
                return None
            raise

    async def get(self, identifier: Identifier) -> Paper | None:
        item = await self._get_work(identifier)
        return normalize(item) if item else None

    async def graph(
        self, identifier: Identifier, relationship: Relationship, limit: int
    ) -> list[Paper]:
        work_id = identifier.value if identifier.kind == "openalex" else None
        if relationship == Relationship.REFERENCE or not work_id:
            item = await self._get_work(identifier)
            if item is None:
                return []
            work_id = normalized_id(item.get("id"), "openalex")
        if relationship == Relationship.CITING:
            if not work_id:
                return []
            items = await self._call(
                lambda: Works(self.settings).filter(cites=work_id).get(per_page=limit)
            )
            return self._normalize_many(items)

        ids = [
            identifier
            for raw in item.get("referenced_works") or []
            if (identifier := normalized_id(raw, "openalex"))
        ][:limit]
        if not ids:
            return []
        papers = self._normalize_many(
            await self._call(lambda: Works(self.settings)[ids])
        )
        by_id = {paper.identifiers.openalex: paper for paper in papers}
        return [by_id[work_id] for work_id in ids if work_id in by_id]

    @staticmethod
    def _normalize_many(items) -> list[Paper]:
        return [paper for item in items if (paper := normalize(item)) is not None]


def normalize(item) -> Paper | None:
    title = item.get("title") or item.get("display_name")
    if not title or not title.strip():
        return None
    ids = item.get("ids") or {}
    doi = normalized_id(ids.get("doi") or item.get("doi"), "doi")
    primary = item.get("primary_location") or {}
    locations = [
        item.get("best_oa_location") or {},
        primary,
        *(item.get("locations") or []),
    ]
    pdf_urls = distinct(
        loc.get("pdf_url") for loc in locations if loc and loc.get("is_oa")
    )
    arxiv_id = normalized_id(ids.get("arxiv"), "arxiv")
    if not arxiv_id and doi and doi.startswith("10.48550/arxiv."):
        arxiv_id = normalized_id(doi.removeprefix("10.48550/arxiv."), "arxiv")
    if not arxiv_id:
        for location in filter(None, locations):
            arxiv_id = normalized_id(location.get("landing_page_url"), "arxiv")
            if arxiv_id:
                break
    authors = [entry.get("author") or {} for entry in item.get("authorships") or []]
    return make_paper(
        Provider.OPENALEX,
        identifiers=PaperIdentifiers(
            doi=doi,
            arxiv=arxiv_id,
            openalex=normalized_id(ids.get("openalex") or item.get("id"), "openalex"),
        ),
        title=title.strip(),
        abstract=item.get("abstract")
        or pyalex.api.invert_abstract(item.get("abstract_inverted_index")),
        authors=[
            Author(name=author["display_name"], orcid=author.get("orcid"))
            for author in authors
            if author.get("display_name")
        ],
        year=item.get("publication_year"),
        citation_count=item.get("cited_by_count"),
        venue=(primary.get("source") or {}).get("display_name"),
        categories=distinct(
            topic.get("display_name") for topic in item.get("topics") or []
        ),
        pdf_urls=pdf_urls,
        url=item.get("id"),
        publication_date=item.get("publication_date"),
    )
