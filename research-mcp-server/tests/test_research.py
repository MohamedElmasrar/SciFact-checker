import asyncio
from unittest.mock import AsyncMock

import pytest

from research_mcp.aggregation import aggregate
from research_mcp.identifiers import parse_identifier
from research_mcp.models import (
    GraphProvider,
    PaperIdentifiers,
    Provider,
    Relationship,
    SearchOptions,
    SortOrder,
)
from research_mcp.providers.base import make_paper
from research_mcp.service import ResearchService


def paper(provider, title, ids=None, **kwargs):
    return make_paper(
        provider, title=title, identifiers=PaperIdentifiers(**(ids or {})), **kwargs
    )


@pytest.mark.parametrize(
    ("raw", "kind", "normalized"),
    [
        ("https://doi.org/10.1234/UPPER%28A%29", "doi", "10.1234/upper(a)"),
        ("https://arxiv.org/pdf/hep-th/0202020v3.pdf", "arxiv", "hep-th/0202020v3"),
        ("https://api.openalex.org/works/W123", "openalex", "W123"),
        (
            "https://www.semanticscholar.org/paper/a-title/" + "A" * 40,
            "semanticscholar",
            "a" * 40,
        ),
        ("CorpusId:012", "corpus_id", "12"),
    ],
)
def test_identifiers(raw, kind, normalized):
    parsed = parse_identifier(raw)
    assert (parsed.kind, parsed.value) == (kind, normalized)


def test_transitive_dedup_preserves_provider_metadata_and_sorting():
    alex = paper(
        Provider.OPENALEX, "A study", {"doi": "10.1234/a"}, citation_count=8, year=2022
    )
    s2 = paper(
        Provider.SEMANTICSCHOLAR,
        "Published Study",
        {"arxiv": "2301.00001"},
        citation_count=21,
    )
    arxiv = paper(
        Provider.ARXIV,
        "a STUDY!",
        {"arxiv": "2301.00001v3"},
        abstract="An abstract",
        categories=["cs.AI"],
        pdf_urls=["https://arxiv.org/pdf/2301.00001"],
    )
    newest = paper(Provider.OPENALEX, "New record", year=2025, citation_count=0)

    match, other = aggregate([[alex, newest], [arxiv], [s2]], SortOrder.MOST_CITED)
    assert other.title == "New record"
    assert match.identifiers.doi == "10.1234/a"
    assert match.citation_count == 21
    assert match.abstract == "An abstract"
    assert match.categories == ["cs.AI"]
    assert len(match.pdf_urls) == 1
    assert match.sources[Provider.OPENALEX].citation_count == 8
    assert match.sources[Provider.ARXIV].identifiers.arxiv.endswith("v3")
    assert aggregate([[alex, newest]], SortOrder.NEWEST)[0] == newest


async def test_concurrent_search_partial_failure_and_cache(settings, monkeypatch):
    service = ResearchService(settings)
    entered = set()
    all_entered = asyncio.Event()

    async def search(provider, _, fail=False):
        entered.add(provider)
        if len(entered) == 3:
            all_entered.set()
        await asyncio.wait_for(all_entered.wait(), timeout=1)
        if fail:
            raise TimeoutError()
        return [paper(provider, "Shared study!", year=2023)]

    async def alex(options):
        return await search(Provider.OPENALEX, options)

    async def arxiv(options):
        return await search(Provider.ARXIV, options, fail=True)

    async def s2(options):
        return await search(Provider.SEMANTICSCHOLAR, options)

    monkeypatch.setattr(
        service.providers[Provider.OPENALEX], "search", AsyncMock(side_effect=alex)
    )
    monkeypatch.setattr(
        service.providers[Provider.ARXIV], "search", AsyncMock(side_effect=arxiv)
    )
    monkeypatch.setattr(
        service.providers[Provider.SEMANTICSCHOLAR], "search", AsyncMock(side_effect=s2)
    )
    options = SearchOptions(query="study", year_from=2020, limit=2)
    result = await service.search(options)
    assert len(result.papers) == 1 and len(result.papers[0].sources) == 2
    assert result.issues[0].source == Provider.ARXIV

    arxiv_search = AsyncMock(return_value=[])
    monkeypatch.setattr(service.providers[Provider.ARXIV], "search", arxiv_search)
    assert not (await service.search(options)).issues
    assert not (await service.search(options)).issues
    assert arxiv_search.await_count == 1


async def test_metadata_alias_resolution_graph_and_relationship_labels(
    settings, monkeypatch
):
    service = ResearchService(settings)
    alex, s2 = (
        service.graph_providers[provider]
        for provider in (Provider.OPENALEX, Provider.SEMANTICSCHOLAR)
    )
    target = paper(
        Provider.OPENALEX, "Target", {"doi": "10.1234/target", "openalex": "W123"}
    )
    match = paper(Provider.SEMANTICSCHOLAR, "Target", {"semanticscholar": "a" * 40})
    neighbor = paper(
        Provider.SEMANTICSCHOLAR, "Neighbor", {"semanticscholar": "b" * 40}
    )
    extra = paper(Provider.OPENALEX, "Extra", {"openalex": "W456"})
    monkeypatch.setattr(alex, "get", AsyncMock(return_value=target))
    monkeypatch.setattr(s2, "get", AsyncMock(return_value=match))
    monkeypatch.setattr(alex, "graph", AsyncMock(return_value=[extra]))
    monkeypatch.setattr(s2, "graph", AsyncMock(return_value=[neighbor, match]))

    first = await service.get("https://openalex.org/W123")
    second = await service.get("w123")
    assert first == second and not first.issues
    assert first.paper.identifiers.semanticscholar == "a" * 40
    assert alex.get.await_count == 1
    assert s2.get.await_args.args[0] == parse_identifier("10.1234/target")

    graph = await service.graph(
        "W123", Relationship.REFERENCE, [GraphProvider.OPENALEX], 1
    )
    assert graph.papers == [extra]
    related = await service.related("W123", 5)
    assert [result.paper.title for result in related.papers] == ["Neighbor", "Extra"]
    assert set(related.papers[0].relationships) == set(Relationship)
    assert (await service.get("https://example.com/file.pdf")).issues[
        0
    ].code == "invalid_identifier"
