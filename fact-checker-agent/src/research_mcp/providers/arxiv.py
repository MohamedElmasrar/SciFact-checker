import asyncio

import arxiv

from ..config import Settings
from ..identifiers import Identifier, normalized_id
from ..infrastructure import RequestGate, TimeoutSession, is_transient
from ..models import Author, Paper, PaperIdentifiers, Provider, SearchOptions, SortOrder
from .base import make_paper


class ArxivProvider:
    name = Provider.ARXIV

    def __init__(self, settings: Settings):
        self.settings = settings
        self.gate = RequestGate(settings.arxiv_interval, settings)

    async def _results(self, search: arxiv.Search) -> list[arxiv.Result]:
        def request():
            client = arxiv.Client(
                page_size=search.max_results,
                num_retries=0,
                delay_seconds=self.gate.interval,
            )
            # The SDK has no public session or timeout parameter.
            client._session.close()
            with TimeoutSession(self.settings.arxiv_timeout) as session:
                client._session = session
                return list(client.results(search))

        return await self.gate.run(
            lambda: asyncio.to_thread(request),
            lambda error: (
                is_transient(error) or isinstance(error, arxiv.UnexpectedEmptyPageError)
            ),
        )

    async def search(self, options: SearchOptions) -> list[Paper]:
        query = options.query
        if options.year_from or options.year_to:
            start = options.year_from or 1000
            end = options.year_to or 9999
            query = f"({query}) AND submittedDate:[{start}01010000 TO {end}12312359]"
        criterion = (
            arxiv.SortCriterion.SubmittedDate
            if options.sort == SortOrder.NEWEST
            else arxiv.SortCriterion.Relevance
        )
        return [
            normalize(item)
            for item in await self._results(
                arxiv.Search(
                    query=query,
                    max_results=options.limit,
                    sort_by=criterion,
                    sort_order=arxiv.SortOrder.Descending,
                )
            )
        ]

    async def get(self, identifier: Identifier) -> Paper | None:
        if identifier.kind == "arxiv":
            search = arxiv.Search(id_list=[identifier.value], max_results=1)
        elif identifier.kind == "doi":
            prefix = "10.48550/arxiv."
            aid = normalized_id(identifier.value.removeprefix(prefix), "arxiv")
            if not identifier.value.startswith(prefix) or not aid:
                return None
            search = arxiv.Search(id_list=[aid], max_results=1)
        else:
            return None
        results = await self._results(search)
        return normalize(results[0]) if results else None


def normalize(item: arxiv.Result) -> Paper:
    return make_paper(
        Provider.ARXIV,
        identifiers=PaperIdentifiers(
            arxiv=normalized_id(item.get_short_id(), "arxiv"),
            doi=normalized_id(item.doi, "doi"),
        ),
        title=" ".join(item.title.split()),
        abstract=" ".join(item.summary.split()) or None,
        authors=[Author(name=author.name) for author in item.authors],
        year=item.published.year,
        venue=item.journal_ref,
        categories=item.categories,
        pdf_urls=[item.pdf_url] if item.pdf_url else [],
        url=item.entry_id,
        publication_date=item.published.date().isoformat(),
    )
