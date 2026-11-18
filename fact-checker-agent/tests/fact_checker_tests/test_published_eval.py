import asyncio
import json
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import HumanMessage

from fact_checker.evidence import Evidence, EvidenceBatch, PaperReference, Provenance
from fact_checker.workflow import (
    Assessment,
    CitedEvidence,
    Decomposition,
    FactCheckReport,
    VerificationResult,
)

runner = run_path(str(Path(__file__).resolve().parents[2] / "evals" / "run.py"))


def test_published_cases_are_split_by_paper_and_cover_all_verdicts():
    cases = runner["load_cases"]("published")
    assert 15 <= len(cases) <= 20
    assert len({case["id"] for case in cases}) == len(cases)
    by_split = {
        split: {case["paper"] for case in cases if case["split"] == split}
        for split in ("dev", "holdout")
    }
    assert all(by_split.values())
    assert not by_split["dev"] & by_split["holdout"]
    assert {case["expected"] for case in cases} == {
        "supported",
        "contradicted",
        "insufficient_evidence",
    }


@pytest.mark.parametrize("version", ["1706.03762v7", "1706.03762"])
def test_published_reference_matching_accepts_arxiv_versions(version):
    case = runner["load_cases"]("published")[0]
    reference = case["reference"]
    candidate = PaperReference(title="Alternate capitalization", identifiers={"arxiv": version})
    assert runner["matches_paper"](candidate, reference)
    assert not runner["matches_paper"](
        PaperReference(title=reference["title"], identifiers={"arxiv": "1801.12345"}), reference
    )


def test_alternate_paper_requires_semantic_review():
    case = runner["load_cases"]("published")[0]
    evidence = Evidence(
        id="E1",
        paper=PaperReference(title="Another valid paper"),
        passage="A direct quote.",
        provenance=[Provenance(source="research_mcp", rank=1)],
    )
    result = VerificationResult(
        claim={"statement": case["claim"], "search_query": "some research"},
        verdict=case["expected"],
        explanation="The evidence supports the claim.",
        citations=[CitedEvidence(evidence=evidence, relation="supports", quote=evidence.passage)],
    )
    score = runner["score"](case, FactCheckReport(results=[result], notices=[]))
    assert score["verdict_matches_label"]
    assert not score["correct"]
    assert score["score_status"] == "review_required"
    assert score["decisive_quotes_match_text"]
    assert not score["decisive_citations_match_key"]


async def test_live_model_request_cap_counts_attempts():
    structured = Mock()
    structured.ainvoke = AsyncMock(
        return_value={
            "raw": Mock(
                usage_metadata={"input_tokens": 10, "output_tokens": 6, "total_tokens": 16}
            ),
            "parsed": Assessment(
                verdict="insufficient_evidence", explanation="Unresolved.", evidence=[]
            ),
            "parsing_error": None,
        }
    )
    model = Mock()
    model.with_structured_output.return_value = structured
    measured = runner["MeasuredModel"](model)
    runnable = measured.with_structured_output(Assessment)
    for _ in range(3):
        await runnable.ainvoke([HumanMessage(content='{"claim":"claim","passages":[]}')])
    with pytest.raises(RuntimeError, match="three-request"):
        await runnable.ainvoke([HumanMessage(content='{"claim":"claim","passages":[]}')])
    assert structured.ainvoke.await_count == 3
    assert measured.calls[0]["tokens"]["total_tokens"] == 16


async def test_request_pacer_spaces_concurrent_starts():
    pacer = runner["RequestPacer"](6000)
    loop = asyncio.get_running_loop()

    async def request_start():
        await pacer.wait()
        return loop.time()

    starts = await asyncio.gather(*(request_start() for _ in range(3)))
    assert starts[1] - starts[0] >= 0.009
    assert starts[2] - starts[1] >= 0.009


async def test_wrapped_quota_error_records_status_code():
    class QuotaError(Exception):
        code = 429

    async def invoke(*args, **kwargs):
        try:
            raise QuotaError()
        except QuotaError as exc:
            raise RuntimeError("Provider adapter error") from exc

    model = Mock()
    model.with_structured_output.return_value.ainvoke = invoke
    measured = runner["MeasuredModel"](model)
    with pytest.raises(RuntimeError):
        await measured.with_structured_output(Decomposition).ainvoke([])
    assert measured.calls[0]["status_code"] == 429


@pytest.mark.parametrize("changed_claim", [True, False])
def test_matching_label_does_not_pass_without_claim_and_evidence(changed_claim):
    case = runner["load_cases"]("published")[0]
    result = VerificationResult(
        claim={
            "statement": "An unrelated result." if changed_claim else case["claim"],
            "search_query": "test",
        },
        verdict=case["expected"],
        explanation="The model returned the expected label.",
        citations=[],
    )
    row = runner["score"](case, FactCheckReport(results=[result], notices=[]))
    assert row["verdict_matches_label"]
    assert not row["correct"]
    assert row["claim_matches_input"] is not changed_claim


@pytest.mark.parametrize("parsed_error", [None, ValueError("invalid structured output")])
async def test_extraction_mismatch_is_distinct_from_schema_failure(parsed_error):
    case = runner["load_cases"]("published")[0]
    decomposition = Decomposition(
        claims=[
            {"statement": case["claim"], "search_query": "first"},
            {"statement": "Another claim.", "search_query": "second"},
        ],
        has_more_claims=False,
    )
    model = Mock()
    model.with_structured_output.return_value.ainvoke = AsyncMock(
        return_value={
            "parsed": None if parsed_error else decomposition,
            "raw": Mock(usage_metadata={"total_tokens": 10}),
            "parsing_error": parsed_error,
        }
    )
    scifact = SimpleNamespace(retrieve=AsyncMock())
    research = SimpleNamespace(search=AsyncMock(), search_pdf=AsyncMock())
    row = await runner["evaluate_published"](
        case,
        model,
        scifact,
        research,
        SimpleNamespace(fact_check_timeout_seconds=3),
        abstracts_only=False,
    )
    assert row["score_status"] == "unscored"
    assert row["status"] == ("unavailable" if parsed_error else "completed")
    assert row["extraction_outcome"] == (
        "schema_error" if parsed_error else "decomposition_mismatch"
    )
    if not parsed_error:
        assert len(row["model_calls"][0]["output"]["claims"]) == 2
    assert row["retrieval"] == []
    research.search.assert_not_awaited()


async def test_controlled_assessment_uses_the_original_claim_and_pinned_passage():
    case = runner["load_cases"]("published")[0]
    model = Mock()
    assessment = model.with_structured_output.return_value
    assessment.ainvoke = AsyncMock(
        return_value={
            "parsed": Assessment(
                verdict="supported",
                explanation="The architecture is specified in [E1].",
                evidence=[{"evidence_id": "E1", "relation": "supports", "quote": case["passage"]}],
            ),
            "raw": Mock(usage_metadata=None),
            "parsing_error": None,
        }
    )
    row = await runner["evaluate_controlled"](
        case, model, SimpleNamespace(fact_check_timeout_seconds=3)
    )
    model.with_structured_output.assert_called_once_with(
        Assessment, include_raw=True, method="json_schema"
    )
    assessment.ainvoke.assert_awaited_once()
    sent_input = json.loads(assessment.ainvoke.await_args.args[0][-1].content)
    assert set(sent_input) == {"claim", "passages"}
    assert sent_input["claim"] == case["claim"]
    assert sent_input["passages"][0]["passage"] == case["passage"]
    assert row["model_calls"][0]["input"] == sent_input
    assert row["correct"]


async def test_deadline_while_waiting_is_recorded_without_sending_a_request():
    pacer = Mock(wait=AsyncMock(side_effect=asyncio.CancelledError))
    model = Mock()
    model.with_structured_output.return_value.ainvoke = AsyncMock()
    measured = runner["MeasuredModel"](model, pacer)

    with pytest.raises(asyncio.CancelledError):
        await measured.with_structured_output(Decomposition).ainvoke([])

    model.with_structured_output.return_value.ainvoke.assert_not_awaited()
    assert measured.calls[0]["outcome"] == "cancelled_before_send"
    assert measured.calls[0]["sent"] is False
    assert "queue_seconds" in measured.calls[0]
    assert "seconds" not in measured.calls[0]


async def test_live_trace_distinguishes_reference_paper_from_excerpt():
    case = runner["load_cases"]("published")[0]
    reference = PaperReference(
        title=case["reference"]["title"], identifiers=case["reference"]["identifiers"]
    )
    context = Evidence(
        paper=reference,
        passage="We propose a new network architecture.",
        provenance=[Provenance(source="research_mcp", rank=1)],
    )
    scifact = SimpleNamespace(retrieve=AsyncMock(return_value=EvidenceBatch()))
    research = SimpleNamespace(
        search=AsyncMock(return_value=EvidenceBatch(evidence=[context], papers=[reference]))
    )
    model = Mock()
    model.with_structured_output.return_value.ainvoke = AsyncMock(
        side_effect=[
            {
                "parsed": Decomposition(
                    claims=[{"statement": case["claim"], "search_query": "head count"}],
                    has_more_claims=False,
                ),
                "raw": Mock(usage_metadata=None),
                "parsing_error": None,
            },
            {
                "parsed": Assessment(
                    verdict="insufficient_evidence", explanation="No head count.", evidence=[]
                ),
                "raw": Mock(usage_metadata=None),
                "parsing_error": None,
            },
        ]
    )

    row = await runner["evaluate_published"](
        case,
        model,
        scifact,
        research,
        SimpleNamespace(fact_check_timeout_seconds=3),
        abstracts_only=True,
    )

    assert row["reference_retrieved"]
    assert row["reference_passage_retrieved"]
    assert not row["reference_excerpt_retrieved"]
    assert (
        next(call for call in row["retrieval"] if call["source"] == "research_mcp")["query"]
        == "head count"
    )
    assert row["model_calls"][1]["input"]["passages"] == [
        {
            "id": "E1",
            "title": reference.title,
            "year": None,
            "page": None,
            "passage": context.passage,
        }
    ]
