from unittest.mock import Mock

import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from scifact_retriever import SearchOptions
from scifact_retriever.reranking import CrossEncoderReranker


def test_search_options_validate_candidate_limits_and_thresholds():
    with pytest.raises(ValidationError, match="rerank_candidates"):
        SearchOptions(rerank=True, rerank_candidates=3, top_k=4)
    with pytest.raises(ValidationError, match="hybrid_depth"):
        SearchOptions(rerank=True, rerank_candidates=12, hybrid_depth=6)
    with pytest.raises(ValidationError):
        SearchOptions(dense_min_score=1.2)
    assert SearchOptions(strategy="sparse", sparse_min_score=2).score_threshold == 2


def test_reranker_scores_claim_paper_pairs_and_preserves_original_scores():
    model = Mock()
    model.predict.return_value = [-3.0, 4.0, 4.0]
    candidates = [
        Document(
            id=key,
            page_content=f"Abstract for {key}",
            metadata={"title": key, "retrieval_score": score},
        )
        for key, score in zip(("a", "b", "c"), (1.0, 0.5, 0.2), strict=True)
    ]

    ranker = CrossEncoderReranker(model, batch_size=3)
    ranked = ranker.rank("claim", candidates)

    assert [doc.id for doc in ranked] == ["b", "c", "a"]
    assert ranked[0].metadata == {"title": "b", "retrieval_score": 0.5, "rerank_score": 4.0}
    assert "rerank_score" not in candidates[0].metadata
    model.predict.assert_called_once_with(
        [("claim", f"{key}\n\nAbstract for {key}") for key in ("a", "b", "c")],
        batch_size=3,
        show_progress_bar=False,
    )
