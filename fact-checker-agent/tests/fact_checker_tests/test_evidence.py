from langchain_core.documents import Document

from fact_checker.evidence import (
    EvidenceBatch,
    from_research,
    from_scifact,
    rank_evidence,
    safe_url,
)


def test_duplicate_passages_keep_identifiers_and_provenance():
    local = from_scifact(
        Document(
            id="19",
            page_content="Observed a difference.\n",
            metadata={"title": "A study", "retrieval_score": 0.8, "rerank_score": 3.5},
        ),
        rank=1,
    )
    remote = from_research(
        {
            "title": "A STUDY",
            "abstract": "Observed a  difference.",
            "identifiers": {"doi": "https://doi.org/10.1234/example"},
            "authors": [{"name": "Example Author"}],
            "sources": {"openalex": {}},
        },
        rank=1,
    )
    assert remote is not None

    result = rank_evidence([EvidenceBatch(evidence=[local]), EvidenceBatch(evidence=[remote])])

    assert len(result) == 1
    merged = result[0]
    assert merged.id == "E1"
    assert merged.paper.identifiers == {"scifact": "19", "doi": "10.1234/example"}
    assert merged.paper.url == "https://doi.org/10.1234/example"
    assert merged.paper.authors == ["Example Author"]
    assert [source.source for source in merged.provenance] == ["scifact", "research_mcp"]
    assert merged.provenance[0].metadata["rerank_score"] == 3.5
    assert local.paper.url is None


def test_ranking_interleaves_sources_and_retains_distinct_passages():
    def local(text: str, rank: int):
        return from_scifact(
            Document(
                id="19",
                page_content=text,
                metadata={"title": "A study", "retrieval_score": 1000},
            ),
            rank=rank,
        )

    remote = from_research({"title": "Other study", "abstract": "A result."}, rank=1)
    assert remote is not None
    result = rank_evidence(
        [
            EvidenceBatch(evidence=[local("One", 1), local("Two", 2)]),
            EvidenceBatch(evidence=[remote]),
        ]
    )
    assert [item.passage for item in result] == ["One", "A result.", "Two"]


def test_metadata_without_passage_and_non_web_links_are_excluded():
    assert from_research({"title": "Metadata only", "abstract": None}, rank=1) is None
    assert safe_url("javascript:alert(1)") is None
    assert safe_url("https://user:pass@example.com") is None
