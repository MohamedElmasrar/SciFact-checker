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
            evidence=[EvidenceAssessment(evidence_id=evidence_id, relation=relation)],
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
    assert report.notices == ["SciFact unavailable."]
    result = report.results[0]
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


def test_explanation_cannot_reference_an_uncited_passage():
    entry = from_research(PAPER, rank=1).model_copy(update={"id": "E1"})
    assessment = WorkflowModel("supported", "supports").assessment
    assessment.explanation = "A different study [E99] supports this claim."
    checked = ground_assessment(CLAIM, [entry], assessment)
    assert checked.verdict == "insufficient_evidence"
    assert "E99" not in checked.explanation
