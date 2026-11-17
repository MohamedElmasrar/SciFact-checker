import re
import unicodedata
from collections.abc import Iterable

from .identifiers import arxiv_base
from .models import Paper, PaperIdentifiers, SortOrder
from .providers.base import distinct


def identity_keys(paper: Paper) -> set[tuple[str, str]]:
    keys = {
        (kind, arxiv_base(value) if kind == "arxiv" else value.casefold())
        for kind, value in paper.identifiers
        if value
    }
    title = unicodedata.normalize("NFKC", paper.title).casefold()
    title = " ".join(re.findall(r"[^\W_]+", title))
    if title:
        keys.add(("title", title))
    return keys


def groups(papers: list[Paper]) -> list[list[int]]:
    """Group connected matches, including records joined by a later alias."""
    parents = list(range(len(papers)))
    index: dict[tuple[str, str], int] = {}

    def root(i: int) -> int:
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for i, paper in enumerate(papers):
        for key in identity_keys(paper):
            other = index.setdefault(key, i)
            parents[root(i)] = root(other)

    result: dict[int, list[int]] = {}
    for i in range(len(papers)):
        result.setdefault(root(i), []).append(i)
    return list(result.values())


def merge(papers: list[Paper]) -> Paper:
    def first(field):
        return next(
            (value for paper in papers if (value := getattr(paper, field)) is not None),
            None,
        )

    ids = {}
    for paper in papers:
        for kind, value in paper.identifiers:
            if value:
                ids.setdefault(kind, value)

    sources = {}
    for paper in papers:
        for provider, metadata in paper.sources.items():
            sources.setdefault(provider, metadata)

    counts = [
        paper.citation_count for paper in papers if paper.citation_count is not None
    ]
    author_lists = [paper.authors for paper in papers]
    authors = max(
        author_lists,
        key=lambda items: (len(items), sum(bool(author.orcid) for author in items)),
    )
    return Paper(
        identifiers=PaperIdentifiers(**ids),
        title=papers[0].title,
        abstract=max(
            (paper.abstract for paper in papers if paper.abstract),
            key=len,
            default=None,
        ),
        authors=authors,
        year=first("year"),
        citation_count=max(counts, default=None),
        venue=first("venue"),
        categories=distinct(
            category for paper in papers for category in paper.categories
        ),
        pdf_urls=distinct(url for paper in papers for url in paper.pdf_urls),
        sources=sources,
    )


def aggregate(
    batches: Iterable[list[Paper]], sort: SortOrder = SortOrder.RELEVANCE
) -> list[Paper]:
    papers = []
    ranking = []
    for batch_number, batch in enumerate(batches):
        for rank, paper in enumerate(batch, start=1):
            papers.append(paper)
            ranking.append((batch_number, rank))

    results = []
    for group in groups(papers):
        best_ranks = {}
        for i in group:
            batch, rank = ranking[i]
            best_ranks.setdefault(batch, rank)
        relevance = sum(1 / (60 + rank) for rank in best_ranks.values())
        paper = merge([papers[i] for i in group])
        primary = relevance
        if sort == SortOrder.NEWEST:
            primary = max(
                (
                    meta.publication_date or str(paper.year or "")
                    for meta in paper.sources.values()
                ),
                default=str(paper.year or ""),
            )
        elif sort == SortOrder.MOST_CITED:
            primary = paper.citation_count if paper.citation_count is not None else -1
        results.append((primary, relevance, paper))

    results.sort(key=lambda item: item[:2], reverse=True)
    return [paper for _, _, paper in results]
