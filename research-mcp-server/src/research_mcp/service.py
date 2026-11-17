import asyncio
import logging
from collections.abc import Awaitable

from .aggregation import aggregate, groups, identity_keys, merge
from .config import Settings
from .identifiers import Identifier, parse_identifier
from .infrastructure import AsyncCache, CooldownActive, status_code
from .models import (
    GraphProvider,
    Issue,
    Paper,
    PaperList,
    PaperResult,
    Provider,
    RelatedPaper,
    RelatedResult,
    Relationship,
    SearchOptions,
)
from .providers.arxiv import ArxivProvider
from .providers.base import LiteratureProvider
from .providers.openalex import OpenAlexProvider
from .providers.semantic import SemanticScholarProvider

logger = logging.getLogger(__name__)

SUPPORTED = {
    Provider.OPENALEX: ("openalex", "doi", "arxiv"),
    Provider.ARXIV: ("arxiv", "doi"),
    Provider.SEMANTICSCHOLAR: ("semanticscholar", "corpus_id", "doi", "arxiv"),
}


def supports(provider: Provider, identifier: Identifier) -> bool:
    if provider == Provider.ARXIV and identifier.kind == "doi":
        return identifier.value.startswith("10.48550/arxiv.")
    return identifier.kind in SUPPORTED[provider]


class ResearchService:
    def __init__(self, settings: Settings):
        openalex = OpenAlexProvider(settings)
        semantic = SemanticScholarProvider(settings)
        self.providers: dict[Provider, LiteratureProvider] = {
            Provider.OPENALEX: openalex,
            Provider.ARXIV: ArxivProvider(settings),
            Provider.SEMANTICSCHOLAR: semantic,
        }
        self.graph_providers = {
            Provider.OPENALEX: openalex,
            Provider.SEMANTICSCHOLAR: semantic,
        }
        self.search_cache = AsyncCache[str, PaperList](
            settings.search_cache_size, settings.search_cache_ttl
        )
        self.metadata_cache = AsyncCache[tuple[Provider, Identifier], Paper | None](
            settings.metadata_cache_size, settings.metadata_cache_ttl
        )

    async def search(self, options: SearchOptions) -> PaperList:
        providers = list(dict.fromkeys(options.providers))
        options = options.model_copy(update={"providers": providers})

        async def load() -> PaperList:
            batches, issues = await collect(
                [
                    (provider, self.providers[provider].search(options))
                    for provider in providers
                ]
            )
            papers = aggregate(batches, options.sort)
            if options.year_from or options.year_to:
                lo, hi = options.year_from or 1000, options.year_to or 9999
                papers = [
                    paper
                    for paper in papers
                    if paper.year is not None and lo <= paper.year <= hi
                ]
            return PaperList(papers=papers[: options.limit], issues=issues)

        return await self.search_cache.get(
            options.model_dump_json(), load, cache_if=lambda result: not result.issues
        )

    async def get(self, paper_id: str) -> PaperResult:
        try:
            identifier = parse_identifier(paper_id)
        except ValueError as error:
            return PaperResult(
                paper=None,
                issues=[
                    Issue(
                        source="metadata", code="invalid_identifier", message=str(error)
                    )
                ],
            )
        candidates = sorted(
            self.providers, key=lambda provider: provider.value != identifier.kind
        )
        attempted = [
            provider for provider in candidates if supports(provider, identifier)
        ]
        records, issues = await collect(
            [(provider, self._lookup(provider, identifier)) for provider in attempted]
        )
        papers = [paper for paper in records if paper is not None]
        if not papers:
            issues.append(
                Issue(
                    source="metadata",
                    code="lookup_incomplete" if issues else "not_found",
                    message="No provider returned metadata for this identifier.",
                )
            )
            return PaperResult(paper=None, issues=issues)

        seed = merge(papers)
        followups = []
        for provider in candidates:
            if provider in attempted:
                continue
            alias = self._pick_identifier(provider, identifier, seed)
            if alias:
                followups.append((provider, self._lookup(provider, alias)))
        more, more_issues = await collect(followups)
        papers.extend(paper for paper in more if paper is not None)
        return PaperResult(paper=merge(papers), issues=issues + more_issues)

    async def _lookup(self, provider: Provider, identifier: Identifier) -> Paper | None:
        return await self.metadata_cache.get(
            (provider, identifier), lambda: self.providers[provider].get(identifier)
        )

    @staticmethod
    def _pick_identifier(
        provider: Provider, requested: Identifier, metadata: Paper | None
    ):
        if supports(provider, requested):
            return requested
        if metadata:
            for kind in SUPPORTED[provider]:
                if value := getattr(metadata.identifiers, kind):
                    alias = Identifier(kind, value)
                    if supports(provider, alias):
                        return alias
        return None

    async def graph(
        self,
        paper_id: str,
        relationship: Relationship,
        providers: list[GraphProvider],
        limit: int,
    ) -> PaperList:
        operations = [
            (Provider(provider), relationship) for provider in dict.fromkeys(providers)
        ]
        batches, issues, _ = await self._graph_batches(paper_id, operations, limit)
        return PaperList(
            papers=aggregate(batch for _, batch in batches)[:limit], issues=issues
        )

    async def related(self, paper_id: str, limit: int) -> RelatedResult:
        operations = [(Provider.SEMANTICSCHOLAR, Relationship.RECOMMENDED)]
        operations.extend(
            (provider, direction)
            for direction in (Relationship.REFERENCE, Relationship.CITING)
            for provider in self.graph_providers
        )
        batches, issues, original = await self._graph_batches(
            paper_id, operations, limit
        )
        tagged = [
            (paper, relationship) for relationship, batch in batches for paper in batch
        ]
        ranked = []
        for group in groups([paper for paper, _ in tagged]):
            paper = merge([tagged[i][0] for i in group])
            if identity_keys(paper) & original:
                continue
            relationships = list(dict.fromkeys(tagged[i][1] for i in group))
            ranked.append(RelatedPaper(paper=paper, relationships=relationships))
        # Prefer API recommendations, then interleave the two citation directions.
        recommendations = [
            entry for entry in ranked if Relationship.RECOMMENDED in entry.relationships
        ]
        fallback = [
            entry
            for entry in ranked
            if Relationship.RECOMMENDED not in entry.relationships
        ]
        references = [
            entry for entry in fallback if Relationship.REFERENCE in entry.relationships
        ]
        citations = [
            entry
            for entry in fallback
            if Relationship.REFERENCE not in entry.relationships
        ]
        for i in range(max(len(references), len(citations))):
            recommendations.extend(references[i : i + 1])
            recommendations.extend(citations[i : i + 1])
        return RelatedResult(papers=recommendations[:limit], issues=issues)

    async def _graph_batches(self, paper_id: str, operations, limit: int):
        try:
            requested = parse_identifier(paper_id)
        except ValueError as error:
            return (
                [],
                [
                    Issue(
                        source="metadata", code="invalid_identifier", message=str(error)
                    )
                ],
                set(),
            )

        metadata = None
        issues = []
        if any(requested.kind not in SUPPORTED[provider] for provider, _ in operations):
            lookup = await self.get(paper_id)
            metadata = lookup.paper
            # An unused lookup provider need not have succeeded.
            if metadata is None:
                issues.extend(lookup.issues)

        calls = []
        for provider, relationship in operations:
            identifier = self._pick_identifier(provider, requested, metadata)
            if identifier:
                calls.append(
                    (
                        provider,
                        self._graph_call(provider, identifier, relationship, limit),
                    )
                )
            elif not any(issue.source == provider for issue in issues):
                issues.append(
                    Issue(
                        source=provider,
                        code="unsupported_identifier",
                        message="No compatible paper identifier was found for this provider.",
                    )
                )

        batches, failures = await collect(calls)
        keys = identity_keys(metadata) if metadata else set()
        keys |= identity_keys(
            Paper(title="", identifiers={requested.kind: requested.value})
        )
        return batches, issues + failures, keys

    async def _graph_call(self, provider, identifier, relationship, limit):
        papers = await self.graph_providers[provider].graph(
            identifier, relationship, limit
        )
        return relationship, papers


async def collect(calls: list[tuple[Provider, Awaitable]]):
    values = await asyncio.gather(
        *(operation for _, operation in calls), return_exceptions=True
    )
    results = []
    issues = []
    for (provider, _), value in zip(calls, values, strict=True):
        if isinstance(value, asyncio.CancelledError):
            raise value
        if isinstance(value, Exception):
            status = status_code(value)
            logger.warning(
                "%s request failed (%s, status=%s)",
                provider,
                type(value).__name__,
                status,
            )
            suffix = f" (HTTP {status})" if status is not None else ""
            if status == 429 or isinstance(value, CooldownActive):
                issue = Issue(
                    source=provider,
                    code="rate_limited",
                    message="Provider rate limit reached.",
                )
            else:
                issue = Issue(
                    source=provider,
                    code="provider_error",
                    message=f"Provider request failed{suffix}.",
                )
            if issue not in issues:
                issues.append(issue)
        else:
            results.append(value)
    return results, issues
