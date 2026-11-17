from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints, model_validator


class Provider(StrEnum):
    OPENALEX = "openalex"
    ARXIV = "arxiv"
    SEMANTICSCHOLAR = "semanticscholar"


class GraphProvider(StrEnum):
    OPENALEX = "openalex"
    SEMANTICSCHOLAR = "semanticscholar"


class SortOrder(StrEnum):
    RELEVANCE = "relevance"
    NEWEST = "newest"
    MOST_CITED = "most_cited"


class Relationship(StrEnum):
    CITING = "citing"
    REFERENCE = "reference"
    RECOMMENDED = "recommended"


Query = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)
]
PaperID = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=512)
]
ResultLimit = Annotated[int, Field(ge=1, le=100)]
Year = Annotated[int, Field(ge=1000, le=9999)]
PageNumbers = Annotated[
    list[Annotated[int, Field(ge=1)]], Field(min_length=1, max_length=500)
]


class SearchOptions(BaseModel):
    query: Query
    providers: Annotated[list[Provider], Field(min_length=1, max_length=3)] = Field(
        default_factory=lambda: list(Provider)
    )
    year_from: Year | None = None
    year_to: Year | None = None
    sort: SortOrder = SortOrder.RELEVANCE
    limit: ResultLimit = 10

    @model_validator(mode="after")
    def validate_year_range(self):
        if self.year_from and self.year_to and self.year_from > self.year_to:
            raise ValueError("year_from must be less than or equal to year_to")
        return self


class PaperIdentifiers(BaseModel):
    doi: str | None = None
    arxiv: str | None = None
    openalex: str | None = None
    semanticscholar: str | None = None
    corpus_id: str | None = None


class Author(BaseModel):
    name: str
    orcid: str | None = None


class ProviderMetadata(BaseModel):
    identifiers: PaperIdentifiers
    url: str | None = None
    publication_date: str | None = None
    venue: str | None = None
    citation_count: int | None = None
    categories: list[str] = Field(default_factory=list)


class Paper(BaseModel):
    identifiers: PaperIdentifiers = Field(default_factory=PaperIdentifiers)
    title: str
    abstract: str | None = None
    authors: list[Author] = Field(default_factory=list)
    year: int | None = None
    citation_count: int | None = None
    venue: str | None = None
    categories: list[str] = Field(default_factory=list)
    pdf_urls: list[str] = Field(default_factory=list)
    sources: dict[Provider, ProviderMetadata] = Field(default_factory=dict)


class Issue(BaseModel):
    source: str
    code: str
    message: str


class PaperList(BaseModel):
    papers: list[Paper]
    issues: list[Issue] = Field(default_factory=list)


class PaperResult(BaseModel):
    paper: Paper | None
    issues: list[Issue] = Field(default_factory=list)


class RelatedPaper(BaseModel):
    paper: Paper
    relationships: list[Relationship]


class RelatedResult(BaseModel):
    papers: list[RelatedPaper]
    issues: list[Issue] = Field(default_factory=list)


class DocumentSource(BaseModel):
    identifiers: PaperIdentifiers
    title: str
    pdf_url: str


class MarkdownPage(BaseModel):
    page: int
    markdown: str


class ContentResult(BaseModel):
    source: DocumentSource | None = None
    total_pages: int | None = None
    pages: list[MarkdownPage] = Field(default_factory=list)
    next_page: int | None = None
    truncated: bool = False
    issues: list[Issue] = Field(default_factory=list)


class Passage(BaseModel):
    page: int
    text: str
    score: float


class PassageResult(BaseModel):
    source: DocumentSource | None = None
    total_pages: int | None = None
    searched_pages: list[int] = Field(default_factory=list)
    next_page: int | None = None
    passages: list[Passage] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)
