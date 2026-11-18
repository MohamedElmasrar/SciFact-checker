from math import log2

import pytest

from scifact_retriever import SearchOptions, Settings
from scifact_retriever.evaluation import (
    METRICS,
    aggregate,
    benchmark_signature,
    evaluate,
    frozen_depth,
    ranking_metrics,
)


def test_metrics_with_graded_relevance_duplicates_and_cutoffs():
    judgments = {"a": 1, "b": 2, "c": 1}
    ranking = ["irrelevant", "a", "a", "x", "y", "z", "b", "u", "v", "w", "p", "c"]
    result = ranking_metrics(ranking, judgments)
    assert result == pytest.approx(
        {
            "Recall@5": 1 / 3,
            "Recall@10": 2 / 3,
            "MRR@10": 1 / 2,
            "nDCG@10": (1 / log2(3) + 3 / log2(7)) / (3 + 1 / log2(3) + 1 / log2(4)),
        }
    )
    empty = ranking_metrics([], judgments)
    assert empty == dict.fromkeys(METRICS, 0)
    assert aggregate([result, empty])["Recall@10"] == pytest.approx(1 / 3)


def test_final_evaluation_rejects_multiple_depths_before_connecting():
    with pytest.raises(ValueError, match="frozen"):
        evaluate(None, "test", Settings(_env_file=None), None, [10, 16])


def test_frozen_depth_rejects_changed_configuration():
    settings = Settings(_env_file=None)
    report = {
        "split": "dev",
        "source_split": "train",
        "configuration": benchmark_signature(settings),
        "selected_rerank_candidates": 16,
    }
    assert frozen_depth(report, settings) == 16
    with pytest.raises(ValueError, match="does not match"):
        frozen_depth(report, Settings(_env_file=None, search=SearchOptions(hybrid_depth=20)))
