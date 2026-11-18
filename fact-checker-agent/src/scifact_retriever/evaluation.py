"""SciFact ranking metrics and development-only candidate-depth selection."""

import logging
from collections.abc import Iterable, Mapping
from importlib.metadata import version
from itertools import batched
from math import ceil, log2
from statistics import mean, median
from time import perf_counter

from langchain_qdrant import RetrievalMode
from qdrant_client import QdrantClient

from .config import SearchOptions, Settings, Strategy
from .dataset import QRELS_REPO, QRELS_REVISION, SOURCE_SPLITS, EvaluationData, Split
from .indexing import (
    index_signature,
    make_dense_embeddings,
    make_sparse_embeddings,
    make_store,
    point_id,
    validate_collection,
)
from .reranking import CrossEncoderReranker
from .retrieval import SciFactRetriever

logger = logging.getLogger(__name__)
METRIC_K = 10
METRICS = ("Recall@5", "Recall@10", "MRR@10", "nDCG@10")


def ranking_metrics(ranking: Iterable[str], judgments: Mapping[str, int]) -> dict[str, float]:
    ordered = list(dict.fromkeys(ranking))[:METRIC_K]
    relevant = sum(grade > 0 for grade in judgments.values())
    if not relevant:
        raise ValueError("Ranking metrics require at least one positive relevance judgment")
    grades = [judgments.get(doc_id, 0) for doc_id in ordered]
    ideal = sorted(judgments.values(), reverse=True)[:METRIC_K]
    dcg = sum((2**grade - 1) / log2(rank + 1) for rank, grade in enumerate(grades, 1))
    idcg = sum((2**grade - 1) / log2(rank + 1) for rank, grade in enumerate(ideal, 1))
    return {
        "Recall@5": sum(grade > 0 for grade in grades[:5]) / relevant,
        "Recall@10": sum(grade > 0 for grade in grades) / relevant,
        "MRR@10": next((1 / rank for rank, grade in enumerate(grades, 1) if grade > 0), 0.0),
        "nDCG@10": dcg / idcg,
    }


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    return {metric: mean(row[metric] for row in rows) for metric in METRICS}


def benchmark_signature(settings: Settings) -> dict:
    excluded = {"strategy", "top_k", "rerank", "rerank_candidates"}
    return {
        "index": index_signature(settings),
        "qrels": {"repo": QRELS_REPO, "revision": QRELS_REVISION},
        "search": settings.search.model_dump(mode="json", exclude=excluded),
        "reranker": {
            "model": settings.rerank_model,
            "revision": settings.resolved_rerank_revision,
            "max_length": settings.rerank_max_length,
            "batch_size": settings.rerank_batch_size,
            "device": settings.rerank_device,
        },
        "dense_device": settings.dense_device,
        "embedding_batch_size": settings.embedding_batch_size,
        "torch_threads": settings.torch_threads,
        "packages": {
            name: version(name)
            for name in (
                "fact-checker-agent",
                "datasets",
                "fastembed",
                "langchain-qdrant",
                "qdrant-client",
                "sentence-transformers",
                "torch",
            )
        },
    }


def evaluate(
    data: EvaluationData,
    split: Split,
    settings: Settings,
    client: QdrantClient,
    candidate_depths: Iterable[int],
) -> dict:
    depths = sorted(set(candidate_depths))
    if not depths or depths[0] < METRIC_K or depths[-1] > settings.search.hybrid_depth:
        raise ValueError("All reranking depths must be between 10 and hybrid_depth")
    if split == "test" and len(depths) != 1:
        raise ValueError("Test evaluation requires one frozen reranking depth")

    indexed = validate_collection(client, settings)
    relevant_ids = {
        point_id(doc_id)
        for scores in data.qrels.values()
        for doc_id, grade in scores.items()
        if grade > 0
    }
    for ids in batched(sorted(relevant_ids), settings.index_batch_size):
        present = client.retrieve(settings.qdrant_collection, ids, with_payload=False)
        if len(present) != len(ids):
            raise ValueError("Some judged papers are missing from the index. Run scifact-ingest.")

    dense = make_dense_embeddings(settings)
    sparse = make_sparse_embeddings(settings)
    reranker = CrossEncoderReranker.from_settings(settings)
    search = settings.search.model_dump()
    retrievers = {}
    for strategy in Strategy:
        options = SearchOptions.model_validate(
            {
                **search,
                "strategy": strategy,
                "top_k": depths[-1] if strategy == Strategy.HYBRID else METRIC_K,
                "rerank": False,
            }
        )
        store = make_store(
            client,
            settings,
            RetrievalMode(strategy),
            dense if strategy != Strategy.SPARSE else None,
            sparse if strategy != Strategy.DENSE else None,
        )
        retrievers[strategy] = SciFactRetriever(store=store, options=options)

    baseline = {strategy.value: [] for strategy in Strategy}
    timings = {strategy.value: [] for strategy in Strategy}
    timings["hybrid_reranked"] = []
    reranked = {depth: [] for depth in depths}
    for number, (query_id, claim) in enumerate(sorted(data.queries.items()), 1):
        judgments = data.qrels[query_id]
        for strategy in (Strategy.DENSE, Strategy.SPARSE):
            started = perf_counter()
            documents = retrievers[strategy].invoke(claim)
            timings[strategy].append(perf_counter() - started)
            baseline[strategy].append(
                ranking_metrics((doc.metadata["doc_id"] for doc in documents), judgments)
            )
        started = perf_counter()
        hybrid = retrievers[Strategy.HYBRID].invoke(claim)
        hybrid_time = perf_counter() - started
        timings[Strategy.HYBRID].append(hybrid_time)
        hybrid_ids = [doc.metadata["doc_id"] for doc in hybrid]
        baseline[Strategy.HYBRID].append(ranking_metrics(hybrid_ids, judgments))

        started = perf_counter()
        ranked_ids = [doc.metadata["doc_id"] for doc in reranker.rank(claim.strip(), hybrid)]
        timings["hybrid_reranked"].append(hybrid_time + perf_counter() - started)
        for depth in depths:
            allowed = set(hybrid_ids[:depth])
            reranked[depth].append(
                ranking_metrics((doc_id for doc_id in ranked_ids if doc_id in allowed), judgments)
            )
        if number % 50 == 0:
            logger.info(
                "evaluation_progress",
                extra={
                    "fields": {
                        "split": split,
                        "claims_scored": number,
                        "total_claims": len(data.queries),
                    }
                },
            )

    rerank_metrics = {depth: aggregate(rows) for depth, rows in reranked.items()}
    selected = max(depths, key=lambda depth: (rerank_metrics[depth]["nDCG@10"], -depth))
    return {
        "split": split,
        "source_split": SOURCE_SPLITS[split],
        "claims": len(data.queries),
        "indexed_documents": indexed,
        "qdrant_version": client.info().version,
        "configuration": benchmark_signature(settings),
        "selected_rerank_candidates": selected,
        "metrics": {
            **{name: aggregate(rows) for name, rows in baseline.items()},
            "hybrid_reranked": rerank_metrics[selected],
        },
        "rerank_depth_metrics": {str(depth): metrics for depth, metrics in rerank_metrics.items()},
        "rerank_timed_candidates": depths[-1],
        "latency_ms": {
            name: {
                "p50": 1000 * median(samples),
                "p95": 1000 * sorted(samples)[ceil(len(samples) * 0.95) - 1],
            }
            for name, samples in timings.items()
        },
    }


def frozen_depth(dev_report: dict, settings: Settings) -> int:
    if (
        not isinstance(dev_report, dict)
        or dev_report.get("split") != "dev"
        or dev_report.get("source_split") != SOURCE_SPLITS["dev"]
        or dev_report.get("configuration") != benchmark_signature(settings)
    ):
        raise ValueError("Development report does not match the current benchmark configuration")
    depth = dev_report.get("selected_rerank_candidates")
    if type(depth) is not int or not METRIC_K <= depth <= settings.search.hybrid_depth:
        raise ValueError("Development report contains an invalid reranking depth")
    return depth


def markdown_report(report: dict) -> str:
    labels = {
        "dense": "Dense",
        "sparse": "Sparse (BM25)",
        "hybrid": "Hybrid (RRF)",
        "hybrid_reranked": "Hybrid + cross-encoder",
    }
    lines = [
        f"# SciFact {report['split']} benchmark",
        "",
        f"BEIR {report['source_split']} · {report['claims']} judged claims · "
        f"{report['indexed_documents']} abstracts",
        "",
        f"Cross-encoder candidate depth **{report['selected_rerank_candidates']}**.",
        "",
        "| Strategy | " + " | ".join(METRICS) + " |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, scores in report["metrics"].items():
        lines.append(
            f"| {labels[name]} | " + " | ".join(f"{scores[m]:.4f}" for m in METRICS) + " |"
        )
    if timings := report.get("latency_ms"):
        lines.extend(
            [
                "",
                "Sequential retrieval latency excludes model loading and metric calculation. "
                "Hybrid with reranking includes the hybrid search and cross-encoder scoring "
                f"of up to {report['rerank_timed_candidates']} candidates.",
                "",
                "| Strategy | p50 (ms) | p95 (ms) |",
                "|---|---:|---:|",
            ]
        )
        for name, latency in timings.items():
            lines.append(f"| {labels[name]} | {latency['p50']:.1f} | {latency['p95']:.1f} |")
    if len(report["rerank_depth_metrics"]) > 1:
        lines.extend(
            [
                "",
                "Candidate depth was selected by highest development nDCG@10. "
                "Ties favor the smaller depth.",
                "",
                "| Candidates | " + " | ".join(METRICS) + " |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for depth, scores in report["rerank_depth_metrics"].items():
            lines.append(f"| {depth} | " + " | ".join(f"{scores[m]:.4f}" for m in METRICS) + " |")
    return "\n".join(lines) + "\n"
