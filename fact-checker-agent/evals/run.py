"""Run the real verification graph against frozen search and PDF responses."""

import argparse
import asyncio
import json
import math
import statistics
import time
import unicodedata
from pathlib import Path
from unittest.mock import AsyncMock

from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from fact_checker.config import Settings
from fact_checker.evidence import EvidenceBatch
from fact_checker.sources import ResearchSource
from fact_checker.workflow import Decomposition, build_workflow

HERE = Path(__file__).resolve().parent
DECISIVE_RELATIONS = {"supported": "supports", "contradicted": "contradicts"}


class ScriptedModel:
    """Exercise fixed model outputs, including deliberately invalid assertions."""

    def __init__(self, case):
        self.case = {
            key: case[key]
            for key in (
                "id",
                "claim",
                "pdf",
                "fixture_verdict",
                "fixture_quote",
                "fixture_pdf_verdict",
            )
            if key in case
        }

    def with_structured_output(self, schema, **kwargs):
        if schema is Decomposition:
            return RunnableLambda(
                lambda _: schema(
                    claims=[{"statement": self.case["claim"], "search_query": self.case["id"]}],
                    has_more_claims=False,
                )
            )

        def propose(messages):
            passages = json.loads(messages[-1].content)["passages"]
            pdf = self.case.get("pdf", {})
            detailed = next((p for p in passages if p["passage"] == pdf.get("text")), None)
            verdict = self.case["fixture_pdf_verdict"] if detailed else self.case["fixture_verdict"]
            proposed = detailed or passages[-1]
            relation = DECISIVE_RELATIONS.get(verdict, "context")
            quote = self.case.get("fixture_quote", proposed["passage"])
            return schema(
                verdict=verdict,
                explanation=f"The supplied passage is [{proposed['id']}].",
                evidence=[{"evidence_id": proposed["id"], "relation": relation, "quote": quote}],
                needs_full_text=bool(pdf and not detailed),
            )

        return RunnableLambda(propose)


def frozen_research(case):
    @tool
    async def search_papers(query: str, limit: int) -> str:
        """Return the pinned paper metadata for this evaluation claim."""
        return json.dumps({"papers": case["papers"], "issues": []})

    @tool
    async def search_paper_content(paper_id: str, query: str, limit: int) -> str:
        """Return the pinned PDF passage when available for this evaluation claim."""
        pdf = case.get("pdf")
        if pdf:
            paper = case["papers"][pdf["paper_index"]]
            if paper_id == paper["identifiers"]["openalex"]:
                return json.dumps(
                    {
                        "passages": [{"page": pdf["page"], "text": pdf["text"]}],
                        "source": {"pdf_url": "https://example.org/fixture.pdf"},
                    }
                )
        return '{"passages": []}'

    client = AsyncMock()
    client.get_tools.return_value = [search_papers, search_paper_content]
    return ResearchSource(client, 1, 1)


async def empty_search(_: str) -> EvidenceBatch:
    return EvidenceBatch()


def normalized(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def score(case, report):
    result = report.results[0] if len(report.results) == 1 else None
    actual = result.verdict if result else None
    expected = case["expected"]
    citation_match = None
    quotes_match = None
    if result and actual in DECISIVE_RELATIONS:
        decisive = [c for c in result.citations if c.relation == DECISIVE_RELATIONS[actual]]
        supported_titles = {c.evidence.paper.title for c in decisive}
        citation_match = bool(
            actual == expected
            and supported_titles
            and supported_titles <= set(case["decisive_titles"])
        )
        quotes_match = bool(decisive) and all(
            (quote := normalized(getattr(c, "quote", None) or ""))
            and quote in normalized(c.evidence.passage)
            for c in decisive
        )
    return {
        "id": case["id"],
        "split": case["split"],
        "expected": expected,
        "actual": actual,
        "status": getattr(result, "status", getattr(report, "status", "completed")),
        "correct": actual == expected,
        "wrong_definitive": actual in DECISIVE_RELATIONS and actual != expected,
        "correct_abstention": actual == expected == "insufficient_evidence",
        "correct_definitive": actual == expected and expected in DECISIVE_RELATIONS,
        "gold_definitive": expected in DECISIVE_RELATIONS,
        "decisive_citations_match_key": citation_match,
        "decisive_quotes_match_text": quotes_match,
        "rationale": case["rationale"],
        "explanation": result.explanation if result else None,
        "citations": [
            {
                "title": citation.evidence.paper.title,
                "relation": citation.relation,
                "quote": getattr(citation, "quote", None),
                "page": getattr(citation.evidence, "page", None),
            }
            for citation in result.citations
        ]
        if result
        else [],
    }


async def run(args):
    cases = [json.loads(line) for line in (HERE / "claims.jsonl").read_text().splitlines()]
    live_model = None
    if args.mode == "gemini":
        settings = Settings()
        live_model = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            api_key=settings.google_api_key,
            vertexai=False,
            timeout=settings.gemini_timeout_seconds,
        )
    rows = []
    for case in cases:
        if args.split != "all" and case["split"] != args.split:
            continue
        model = live_model if live_model is not None else ScriptedModel(case)
        research = frozen_research(case)
        args_by_version = {} if args.abstracts_only else {"search_pdf": research.search_pdf}
        workflow = build_workflow(model, empty_search, research.search, **args_by_version)
        start = time.perf_counter()
        report = (await workflow.ainvoke({"request": case["claim"]}))["report"]
        rows.append({**score(case, report), "seconds": round(time.perf_counter() - start, 4)})
    durations = sorted(row["seconds"] for row in rows)
    return {
        "mode": args.mode,
        "model": live_model.model if live_model is not None else "scripted fixture",
        "split": args.split,
        "summary": {
            "cases": len(rows),
            **{
                key: sum(row[key] is True for row in rows)
                for key in (
                    "correct",
                    "wrong_definitive",
                    "correct_abstention",
                    "correct_definitive",
                    "gold_definitive",
                    "decisive_citations_match_key",
                    "decisive_quotes_match_text",
                )
            },
            "definitive_verdicts": sum(row["actual"] in DECISIVE_RELATIONS for row in rows),
            "execution_errors": sum(row["status"] != "completed" for row in rows),
            "median_seconds": statistics.median(durations),
            "p95_seconds": durations[math.ceil(len(durations) * 0.95) - 1],
        },
        "cases": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "holdout", "all"), default="all")
    parser.add_argument("--mode", choices=("fixture", "gemini"), default="fixture")
    parser.add_argument("--abstracts-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--check", action="store_true", help="Enforce deterministic regression checks."
    )
    options = parser.parse_args()
    if options.check and (options.mode != "fixture" or options.abstracts_only):
        parser.error("--check requires the full fixture workflow")
    report = asyncio.run(run(options))
    output = json.dumps(report, indent=2) + "\n"
    if options.output:
        options.output.write_text(output)
    else:
        print(output, end="")
    if options.check:
        summary = report["summary"]
        if (
            summary["wrong_definitive"]
            or summary["execution_errors"]
            or summary["correct_definitive"] < summary["gold_definitive"] - 1
            or summary["decisive_quotes_match_text"] != summary["definitive_verdicts"]
            or summary["decisive_citations_match_key"] != summary["definitive_verdicts"]
        ):
            raise SystemExit("Frozen fact-check regression detected.")
