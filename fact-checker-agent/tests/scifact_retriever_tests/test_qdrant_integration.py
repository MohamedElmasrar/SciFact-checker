import asyncio
import os
from contextlib import closing
from unittest.mock import Mock
from uuid import uuid4

import pytest
from langchain_core.embeddings import Embeddings
from qdrant_client import models

from scifact_retriever import SearchOptions, Settings, Strategy, build_retriever
from scifact_retriever.dataset import benchmark_data, corpus_documents
from scifact_retriever.evaluation import METRICS, evaluate, markdown_report
from scifact_retriever.indexing import (
    IndexResult,
    index_corpus,
    make_client,
    make_sparse_embeddings,
)
from scifact_retriever.reranking import CrossEncoderReranker

pytestmark = pytest.mark.integration


class KeywordTestEmbeddings(Embeddings):
    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0] if "ibuprofen" in text.lower() else [0.0, 1.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


@pytest.fixture
def qdrant():
    url = os.environ.get("SCIFACT_TEST_QDRANT_URL")
    if not url:
        pytest.skip("Set SCIFACT_TEST_QDRANT_URL to run against a real Qdrant server")
    settings = Settings(
        _env_file=None,
        qdrant_url=url,
        qdrant_collection=f"scifact-test-{uuid4().hex}",
        qdrant_api_key=None,
        dense_model="test-keyword-embedding",
        index_batch_size=1,
        search=SearchOptions(),
    )
    with closing(make_client(settings)) as client:
        try:
            yield settings, client
        finally:
            if client.collection_exists(settings.qdrant_collection):
                client.delete_collection(settings.qdrant_collection)


@pytest.fixture
def papers():
    return list(
        corpus_documents(
            [
                {
                    "_id": "1",
                    "title": "Ibuprofen",
                    "text": "Pain intensity was reduced.",
                    "metadata": {"journal": "Example Journal"},
                },
                {"_id": "2", "title": "Memory", "text": "The experiment measured memory."},
                {"_id": "3", "title": "Exercise", "text": "Walking improved fitness."},
            ]
        )
    )


def test_ingestion_resumes_and_skips_embeddings(qdrant, papers):
    settings, client = qdrant
    sparse = make_sparse_embeddings(settings)
    dense = KeywordTestEmbeddings()

    def interrupted():
        yield papers[0]
        raise RuntimeError("interrupted ingestion")

    with pytest.raises(RuntimeError, match="interrupted"):
        index_corpus(interrupted(), client, settings, dense, sparse)
    with pytest.raises(ValueError, match="incomplete"):
        build_retriever(settings, client=client)
    assert index_corpus(papers, client, settings, dense, sparse) == IndexResult(2, 1)
    assert index_corpus(papers, client, settings, Mock(spec=Embeddings), sparse) == IndexResult(
        0, 3
    )

    info = client.get_collection(settings.qdrant_collection)
    assert client.count(settings.qdrant_collection, exact=True).count == 3
    assert info.config.params.sparse_vectors["bm25"].modifier == models.Modifier.IDF

    incompatible = settings.model_copy(update={"dense_model": "different-model"})
    with pytest.raises(ValueError, match="incompatible"):
        build_retriever(incompatible, client=client)


def test_retrieval_and_benchmark_against_qdrant(qdrant, papers, monkeypatch):
    settings, client = qdrant
    sparse = make_sparse_embeddings(settings)
    dense = KeywordTestEmbeddings()
    index_corpus(papers, client, settings, dense, sparse)
    for module in ("retrieval", "evaluation"):
        monkeypatch.setattr(f"scifact_retriever.{module}.make_dense_embeddings", lambda _: dense)
        monkeypatch.setattr(f"scifact_retriever.{module}.make_sparse_embeddings", lambda _: sparse)

    model = Mock()
    model.predict.side_effect = lambda pairs, **kwargs: [
        5.0 if "Ibuprofen" in paper else 0.0 for _, paper in pairs
    ]
    reranker = CrossEncoderReranker(model, 4)
    monkeypatch.setattr(CrossEncoderReranker, "from_settings", lambda _: reranker)

    for strategy in Strategy:
        options = SearchOptions(strategy=strategy, top_k=1)
        retriever = build_retriever(settings.model_copy(update={"search": options}), client=client)
        [doc] = retriever.invoke("  ibuprofen  ")
        assert doc.id == doc.metadata["doc_id"] == "1"
        assert doc.page_content == papers[0].page_content
        assert doc.metadata["title"] == "Ibuprofen"
        assert doc.metadata["journal"] == "Example Journal"
        assert doc.metadata["retrieval_strategy"] == strategy
        assert doc.metadata["retrieval_score"] > 0
        with pytest.raises(ValueError, match="nonblank"):
            retriever.invoke(" ")

    options = SearchOptions(rerank=True, rerank_candidates=3, top_k=1)
    retriever = build_retriever(settings.model_copy(update={"search": options}), client=client)

    async def search_twice():
        return await asyncio.gather(retriever.ainvoke("ibuprofen"), retriever.ainvoke("ibuprofen"))

    for [doc] in asyncio.run(search_twice()):
        assert doc.id == "1"
        assert doc.metadata["rerank_score"] == 5.0

    for strategy, threshold in [
        (Strategy.DENSE, {"dense_min_score": 0.99}),
        (Strategy.SPARSE, {"sparse_min_score": 100}),
        (Strategy.HYBRID, {"hybrid_min_score": 100}),
    ]:
        options = SearchOptions(strategy=strategy, top_k=3, **threshold)
        retriever = build_retriever(settings.model_copy(update={"search": options}), client=client)
        expected = ["1"] if strategy == Strategy.DENSE else []
        assert [doc.id for doc in retriever.invoke("ibuprofen")] == expected

    data = benchmark_data(
        [{"_id": "10", "text": "ibuprofen"}], [{"query-id": 10, "corpus-id": 1, "score": 1}]
    )
    report = evaluate(data, "dev", settings, client, [10, 16])
    assert report["metrics"] == dict.fromkeys(
        ("dense", "sparse", "hybrid", "hybrid_reranked"), dict.fromkeys(METRICS, 1.0)
    )
    assert report["selected_rerank_candidates"] == 10
    assert "Hybrid + cross-encoder" in markdown_report(report)
