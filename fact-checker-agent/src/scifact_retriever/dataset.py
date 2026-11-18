"""Load the pinned BEIR corpus, queries, and relevance judgments from Hugging Face."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field, field_validator

CORPUS_REPO = "BeIR/scifact"
CORPUS_REVISION = "b3b5335604bf5ee3c4447671af975ea25143d4f5"
QRELS_REPO = "BeIR/scifact-qrels"
QRELS_REVISION = "2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029"
Split = Literal["dev", "test"]
SOURCE_SPLITS: dict[Split, str] = {"dev": "train", "test": "test"}


class TextRow(BaseModel):
    id: str = Field(alias="_id")
    text: str

    @field_validator("id", "text")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must contain a non-whitespace character")
        return value


class CorpusRow(TextRow):
    model_config = ConfigDict(extra="allow")

    title: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Judgment(BaseModel):
    query_id: int = Field(alias="query-id", ge=0)
    corpus_id: int = Field(alias="corpus-id", ge=0)
    score: int = Field(ge=0)


@dataclass(frozen=True)
class EvaluationData:
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]


def paper_prefix(title: str) -> str:
    return f"{title}\n\n" if title else ""


def corpus_documents(rows: Iterable[dict[str, Any]]) -> Iterator[Document]:
    seen = set()
    for row in rows:
        paper = CorpusRow.model_validate(row)
        if paper.id in seen:
            raise ValueError(f"Duplicate corpus ID {paper.id}")
        seen.add(paper.id)
        metadata = {
            **(paper.model_extra or {}),
            **paper.metadata,
            "doc_id": paper.id,
            "title": paper.title,
        }
        yield Document(id=paper.id, page_content=paper.text, metadata=metadata)


def load_corpus() -> Iterator[Document]:
    from datasets import load_dataset

    rows = load_dataset(CORPUS_REPO, "corpus", split="corpus", revision=CORPUS_REVISION)
    yield from corpus_documents(rows)


def benchmark_data(
    queries: Iterable[dict[str, Any]], judgments: Iterable[dict[str, Any]]
) -> EvaluationData:
    qrels: dict[str, dict[str, int]] = {}
    for row in judgments:
        judgment = Judgment.model_validate(row)
        relevant = qrels.setdefault(str(judgment.query_id), {})
        doc_id = str(judgment.corpus_id)
        previous = relevant.setdefault(doc_id, judgment.score)
        if previous != judgment.score:
            raise ValueError(f"Conflicting judgments for claim {judgment.query_id}, paper {doc_id}")

    if not qrels or any(not any(scores.values()) for scores in qrels.values()):
        raise ValueError("Every evaluated claim must have a positive relevance judgment")

    claims = {}
    for row in queries:
        claim = TextRow.model_validate(row)
        if claim.id in qrels:
            if claim.id in claims:
                raise ValueError(f"Duplicate claim ID {claim.id}")
            claims[claim.id] = claim.text
    if missing := qrels.keys() - claims.keys():
        raise ValueError(f"Qrels reference missing claims: {sorted(missing)[:5]}")
    return EvaluationData(queries=claims, qrels=qrels)


def load_evaluation_data(split: Split) -> EvaluationData:
    from datasets import load_dataset

    queries = load_dataset(CORPUS_REPO, "queries", split="queries", revision=CORPUS_REVISION)
    source_split = SOURCE_SPLITS[split]
    qrels_path = f"hf://datasets/{QRELS_REPO}@{QRELS_REVISION}/{source_split}.tsv"
    judgments = load_dataset(
        "csv", data_files={source_split: qrels_path}, delimiter="\t", split=source_split
    )
    return benchmark_data(queries, judgments)
