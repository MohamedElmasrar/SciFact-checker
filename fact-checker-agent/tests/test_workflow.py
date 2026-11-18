import asyncio
from unittest.mock import AsyncMock

import pytest
from langchain_core.runnables import RunnableLambda

from fact_checker.evidence import EvidenceBatch, from_research
from fact_checker.workflow import (
    Assessment,
    Claim,
    Decomposition,
    EvidenceAssessment,
    build_workflow,
    ground_assessment,
)

CLAIM = Claim(
    statement="The intervention reduced the outcome.", search_query="intervention outcome"
)
PAPER = {"title": "Trial results", "abstract": "The intervention reduced the outcome."}


class WorkflowModel:
    def __init__(self, verdict, relation, evidence_id="E1"):
        self.assessment = Assessment(
            verdict=verdict,
            explanation=f"The comparison is described in [{evidence_id}].",
            evidence=[
                EvidenceAssessment(
                    evidence_id=evidence_id, relation=relation, quote=PAPER["abstract"]
                )
            ],
        )
        self.assess = AsyncMock(return_value=self.assessment)

    def with_structured_output(self, schema, **kwargs):
        if schema is Decomposition:
            return RunnableLambda(lambda _: Decomposition(claims=[CLAIM], has_more_claims=False))

        async def assess(messages):
            return await self.assess(messages)

        return RunnableLambda(assess)


@pytest.mark.parametrize(
    ("verdict", "relation"),
    [
        ("supported", "supports"),
        ("contradicted", "contradicts"),
        ("insufficient_evidence", "context"),
    ],
)
async def test_workflow_verifies_and_cites_retrieved_material(verdict, relation):
    evidence = from_research(PAPER, rank=1)
    scifact = AsyncMock(return_value=EvidenceBatch(notices=["SciFact unavailable."]))
    research = AsyncMock(return_value=EvidenceBatch(evidence=[evidence]))
    graph = build_workflow(WorkflowModel(verdict, relation), scifact, research)

    report = (await graph.ainvoke({"request": "Please verify the claim."}))["report"]

    scifact.assert_awaited_once_with(CLAIM.statement)
    research.assert_awaited_once_with(CLAIM.search_query)
    result = report.results[0]
    assert result.notices == ["SciFact unavailable."]
    assert result.verdict == verdict
    citation = result.citations[0]
    assert citation.relation == relation
    assert citation.evidence.paper.title == PAPER["title"]
    assert citation.evidence.passage == PAPER["abstract"]


async def test_no_passages_skips_verification_model():
    empty = AsyncMock(return_value=EvidenceBatch())
    model = WorkflowModel("supported", "supports")
    graph = build_workflow(model, empty, empty)
    result = (await graph.ainvoke({"request": CLAIM.statement}))["report"].results[0]
    assert result.verdict == "insufficient_evidence"
    assert result.citations == []
    model.assess.assert_not_awaited()


@pytest.mark.parametrize(("identifier", "relation"), [("E99", "supports"), ("E1", "context")])
def test_invented_ids_or_missing_direction_cannot_establish_a_verdict(identifier, relation):
    entry = from_research(PAPER, rank=1).model_copy(update={"id": "E1"})
    model = WorkflowModel("supported", relation, identifier)
    checked = ground_assessment(CLAIM, [entry], model.assessment)
    assert checked.verdict == "insufficient_evidence"
    assert all(citation.evidence.id == "E1" for citation in checked.citations)


@pytest.mark.parametrize("markers", ["[E99]", "[E1, E99]"])
def test_explanation_cannot_reference_an_uncited_passage(markers):
    entry = from_research(PAPER, rank=1).model_copy(update={"id": "E1"})
    assessment = WorkflowModel("supported", "supports").assessment
    assessment.explanation = f"A different study {markers} supports this claim."
    checked = ground_assessment(CLAIM, [entry], assessment)
    assert checked.verdict == "insufficient_evidence"
    assert "E99" not in checked.explanation


@pytest.mark.parametrize("quote", [None, "A sentence absent from the source."])
def test_decisive_citations_require_a_source_quote(quote):
    entry = from_research(PAPER, rank=1).model_copy(update={"id": "E1"})
    assessment = WorkflowModel("supported", "supports").assessment
    assessment.evidence[0].quote = quote
    checked = ground_assessment(CLAIM, [entry], assessment)
    assert checked.verdict == "insufficient_evidence"
    assert not checked.citations


async def test_total_retrieval_failure_has_no_verdict():
    unavailable = AsyncMock(return_value=EvidenceBatch(status="unavailable", notices=["Offline"]))
    model = WorkflowModel("supported", "supports")
    graph = build_workflow(model, unavailable, unavailable)
    result = (await graph.ainvoke({"request": CLAIM.statement}))["report"].results[0]
    assert result.status == "unavailable"
    assert result.verdict is None
    assert result.notices == ["Offline"]
    model.assess.assert_not_awaited()


async def test_slow_source_leaves_time_to_assess_other_evidence():
    async def slow(_):
        await asyncio.sleep(5)

    evidence = from_research(PAPER, rank=1)
    research = AsyncMock(return_value=EvidenceBatch(evidence=[evidence]))
    graph = build_workflow(WorkflowModel("supported", "supports"), slow, research, timeout=0.2)
    result = (await graph.ainvoke({"request": CLAIM.statement}))["report"].results[0]
    assert result.status == "completed"
    assert result.verdict == "supported"
    assert result.notices == ["SciFact retrieval timed out."]


async def test_failed_claim_preserves_earlier_verdict_at_request_deadline():
    other = Claim(statement="Another outcome increased.", search_query="another outcome")
    model = WorkflowModel("supported", "supports")
    model_decompose = model.with_structured_output

    def structured(schema, **kwargs):
        if schema is Decomposition:
            return RunnableLambda(
                lambda _: Decomposition(claims=[CLAIM, other], has_more_claims=False)
            )
        return model_decompose(schema, **kwargs)

    async def assess(messages):
        if other.statement in messages[-1].content:
            await asyncio.sleep(5)
        return model.assessment

    model.with_structured_output = structured
    model.assess.side_effect = assess
    evidence = from_research(PAPER, rank=1)
    retrieval = AsyncMock(return_value=EvidenceBatch(evidence=[evidence]))
    graph = build_workflow(model, retrieval, retrieval, timeout=0.2)
    results = (await graph.ainvoke({"request": "Check both claims"}))["report"].results
    assert [(item.status, item.verdict) for item in results] == [
        ("completed", "supported"),
        ("timeout", None),
    ]


async def test_inconclusive_abstract_uses_one_pdf_pass_and_cites_its_page():
    evidence = from_research({**PAPER, "identifiers": {"openalex": "W123"}}, rank=1)
    page = evidence.model_copy(update={"page": 4, "pdf_url": "https://example.org/paper.pdf"})
    abstract = AsyncMock(return_value=EvidenceBatch(evidence=[evidence], papers=[evidence.paper]))
    scifact = AsyncMock(return_value=EvidenceBatch())
    search_pdf = AsyncMock(return_value=EvidenceBatch(evidence=[page]))
    model = WorkflowModel("supported", "supports")
    model.assess.side_effect = [
        Assessment(
            verdict="insufficient_evidence",
            explanation="The abstract is incomplete.",
            evidence=[],
            needs_full_text=True,
        ),
        model.assessment,
    ]
    graph = build_workflow(model, scifact, abstract, search_pdf=search_pdf)
    result = (await graph.ainvoke({"request": CLAIM.statement}))["report"].results[0]
    search_pdf.assert_awaited_once_with(evidence.paper, CLAIM.search_query)
    assert result.verdict == "supported"
    assert result.citations[0].evidence.page == 4
    assert result.citations[0].quote == PAPER["abstract"]


async def test_decisive_abstract_skips_pdf_lookup():
    evidence = from_research({**PAPER, "identifiers": {"openalex": "W123"}}, rank=1)
    abstract = AsyncMock(return_value=EvidenceBatch(evidence=[evidence], papers=[evidence.paper]))
    search_pdf = AsyncMock()
    graph = build_workflow(
        WorkflowModel("supported", "supports"), abstract, abstract, search_pdf=search_pdf
    )
    result = (await graph.ainvoke({"request": CLAIM.statement}))["report"].results[0]
    assert result.verdict == "supported"
    search_pdf.assert_not_awaited()
