import re
import unicodedata
from typing import Literal
from urllib.parse import quote

from langchain_core.documents import Document
from pydantic import BaseModel, Field, HttpUrl, JsonValue, TypeAdapter, ValidationError


class PaperReference(BaseModel):
    title: str
    identifiers: dict[str, str] = Field(default_factory=dict)
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    url: str | None = None


class Provenance(BaseModel):
    source: Literal["scifact", "research_mcp"]
    rank: int = Field(ge=1)
    providers: list[str] = Field(default_factory=list)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class Evidence(BaseModel):
    id: str = ""
    paper: PaperReference
    passage: str
    provenance: list[Provenance]
    page: int | None = Field(default=None, ge=1)
    pdf_url: str | None = None


class EvidenceBatch(BaseModel):
    evidence: list[Evidence] = Field(default_factory=list)
    notices: list[str] = Field(default_factory=list)
    status: Literal["completed", "timeout", "unavailable"] = "completed"
    papers: list[PaperReference] = Field(default_factory=list)


URL = TypeAdapter(HttpUrl)
PASSAGE_LIMIT = 3000


def safe_url(value: str | None) -> str | None:
    if value:
        try:
            parsed = URL.validate_python(value)
            if not (parsed.username or parsed.password):
                return str(parsed)
        except ValidationError:
            pass
    return None


def excerpt(text: str) -> str:
    text = text.strip()
    if len(text) > PASSAGE_LIMIT:
        return text[:PASSAGE_LIMIT].rstrip() + "…"
    return text


def from_scifact(document: Document, rank: int) -> Evidence:
    metadata = document.metadata
    doc_id = document.id or str(metadata["doc_id"])
    identifiers = {"scifact": doc_id}
    if isinstance(metadata.get("doi"), str) and metadata["doi"].strip():
        identifiers["doi"] = normalize_doi(metadata["doi"])
    return Evidence(
        paper=PaperReference(
            title=str(metadata.get("title") or f"SciFact document {doc_id}"),
            identifiers=identifiers,
            url=paper_url(identifiers),
        ),
        passage=excerpt(document.page_content),
        provenance=[Provenance(source="scifact", rank=rank, metadata=metadata)],
    )


def normalize_doi(value: str) -> str:
    return re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", value.strip(), flags=re.I)


def paper_url(identifiers: dict[str, str]) -> str | None:
    for kind, prefix in (
        ("doi", "https://doi.org/"),
        ("arxiv", "https://arxiv.org/abs/"),
        ("openalex", "https://openalex.org/"),
        ("semanticscholar", "https://www.semanticscholar.org/paper/"),
    ):
        if identifier := identifiers.get(kind):
            return safe_url(prefix + quote(identifier, safe="/"))
    return None


def research_reference(paper: dict) -> PaperReference:
    identifiers = {key: value for key, value in paper.get("identifiers", {}).items() if value}
    if "doi" in identifiers:
        identifiers["doi"] = normalize_doi(identifiers["doi"])
    providers = paper.get("sources", {})
    url = paper_url(identifiers) or next(
        (url for source in providers.values() if (url := safe_url(source.get("url")))),
        None,
    )
    return PaperReference(
        title=paper["title"],
        identifiers=identifiers,
        authors=[author["name"] for author in paper.get("authors", [])],
        year=paper.get("year"),
        url=url,
    )


def from_research(
    paper: dict, rank: int, *, reference: PaperReference | None = None
) -> Evidence | None:
    abstract = paper.get("abstract")
    if not isinstance(abstract, str) or not abstract.strip():
        return None
    providers = paper.get("sources", {})
    return Evidence(
        paper=reference or research_reference(paper),
        passage=excerpt(abstract),
        provenance=[
            Provenance(
                source="research_mcp",
                rank=rank,
                providers=list(providers),
                metadata={"sources": providers, "venue": paper.get("venue")},
            )
        ],
    )


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def same_paper(left: PaperReference, right: PaperReference) -> bool:
    if normalize_text(left.title) == normalize_text(right.title):
        return True
    return any(
        kind in right.identifiers and value.casefold() == right.identifiers[kind].casefold()
        for kind, value in left.identifiers.items()
    )


def rank_evidence(batches: list[EvidenceBatch], *, limit: int = 8) -> list[Evidence]:
    # Interleave by within-source rank. Dense, RRF, and provider scores are incomparable.
    ordered = sorted(
        (item for batch in batches for item in batch.evidence if item.passage.strip()),
        key=lambda item: item.provenance[0].rank,
    )
    unique: list[Evidence] = []
    for item in ordered:
        duplicate = next(
            (
                existing
                for existing in unique
                if same_paper(existing.paper, item.paper)
                and (existing.pdf_url, existing.page) == (item.pdf_url, item.page)
                and normalize_text(existing.passage) == normalize_text(item.passage)
            ),
            None,
        )
        if duplicate is None:
            unique.append(item.model_copy(deep=True))
            continue
        duplicate.provenance.extend(item.provenance)
        paper = duplicate.paper
        paper.identifiers = {**item.paper.identifiers, **paper.identifiers}
        paper.url = paper.url or item.paper.url
        paper.authors = paper.authors or item.paper.authors
        paper.year = paper.year or item.paper.year
    return [
        item.model_copy(update={"id": f"E{index}"}) for index, item in enumerate(unique[:limit], 1)
    ]
