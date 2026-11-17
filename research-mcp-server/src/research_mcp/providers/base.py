from typing import Protocol

from ..identifiers import Identifier
from ..models import Paper, Provider, ProviderMetadata, SearchOptions


class LiteratureProvider(Protocol):
    name: Provider

    async def search(self, options: SearchOptions) -> list[Paper]: ...

    async def get(self, identifier: Identifier) -> Paper | None: ...


def make_paper(
    provider: Provider,
    *,
    url: str | None = None,
    publication_date: str | None = None,
    **data,
) -> Paper:
    paper = Paper(**data)
    paper.sources[provider] = ProviderMetadata(
        identifiers=paper.identifiers,
        url=url,
        publication_date=publication_date,
        venue=paper.venue,
        citation_count=paper.citation_count,
        categories=paper.categories,
    )
    return paper


def distinct(values):
    return list(dict.fromkeys(value for value in values if value))
