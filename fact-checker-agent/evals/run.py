"""Evaluate verification with frozen fixtures or live retrieval for published papers."""

import argparse
import asyncio
import json
import math
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from unittest.mock import AsyncMock

from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from fact_checker.config import Settings
from fact_checker.evidence import Evidence, EvidenceBatch, PaperReference, Provenance, normalize_doi
from fact_checker.sources import ResearchSource, SciFactSource, research_client
from fact_checker.workflow import (
    Assessment,
    Claim,
    Decomposition,
    FactCheckReport,
    assess_claim,
    build_workflow,
    ground_assessment,
    normalize_quote,
)

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


def evidence_text(text):
    """Ignore PDF line breaks and spacing without dropping units or punctuation."""
    return "".join(normalize_quote(text).replace("’", "'").split())


def matches_paper(paper: PaperReference, reference: dict) -> bool:
    def identifier(kind, value):
        if kind == "arxiv":
            return re.sub(r"v\d+$", "", value.casefold())
        if kind == "doi":
            return normalize_doi(value).casefold()
        return value.casefold()

    shared = paper.identifiers.keys() & reference["identifiers"].keys()
    if shared:
        return any(
            identifier(kind, paper.identifiers[kind])
            == identifier(kind, reference["identifiers"][kind])
            for kind in shared
        )
    return re.findall(r"\w+", paper.title.casefold()) == re.findall(
        r"\w+", reference["title"].casefold()
    )


def score(case, report):
    result = report.results[0] if len(report.results) == 1 else None
    actual = result.verdict if result else None
    expected = case["expected"]
    citation_match = None
    quotes_match = None
    anchor_quote_match = None
    reference = case.get("reference")
    claim_match = (
        normalize_quote(result.claim.statement) == normalize_quote(case["claim"])
        if result
        else None
    )
    if result and actual in DECISIVE_RELATIONS:
        decisive = [c for c in result.citations if c.relation == DECISIVE_RELATIONS[actual]]
        if reference:
            # A known reference is an anchor for review. Other papers may be valid evidence.
            citation_match = any(matches_paper(c.evidence.paper, reference) for c in decisive)
            anchor_quote_match = any(
                matches_paper(c.evidence.paper, reference)
                and evidence_text(case["excerpt"]) in evidence_text(c.quote or "")
                and evidence_text(c.quote or "") in evidence_text(case["passage"])
                for c in decisive
            )
        else:
            citation_match = bool(
                actual == expected
                and decisive
                and all(c.evidence.paper.title in case["decisive_titles"] for c in decisive)
            )
        quotes_match = bool(decisive) and all(
            (quote := normalize_quote(getattr(c, "quote", None) or ""))
            and quote in normalize_quote(c.evidence.passage)
            for c in decisive
        )
    status = result.status if result else report.status
    partial = bool(result and result.full_text_status)
    grounding_rejected = bool(result and result.grounding_rejected)
    score_status = "scored"
    if status != "completed" or actual is None or partial:
        score_status = "unscored"
    elif not claim_match or (
        reference and actual in DECISIVE_RELATIONS and quotes_match and not anchor_quote_match
    ):
        score_status = "review_required"
    correct = (
        score_status == "scored"
        and actual == expected
        and not grounding_rejected
        and (actual not in DECISIVE_RELATIONS or (quotes_match and citation_match))
    )
    return {
        "id": case["id"],
        "split": case["split"],
        "expected": expected,
        "actual": actual,
        "status": status,
        "full_text_status": result.full_text_status if result else None,
        "score_status": score_status,
        "verdict_matches_label": actual == expected,
        "grounding_rejected": grounding_rejected,
        "claim_matches_input": claim_match,
        "correct": correct,
        "wrong_definitive": claim_match is True
        and actual in DECISIVE_RELATIONS
        and actual != expected,
        "correct_abstention": correct and expected == "insufficient_evidence",
        "correct_definitive": correct and expected in DECISIVE_RELATIONS,
        "gold_definitive": expected in DECISIVE_RELATIONS,
        "decisive_citations_match_key": citation_match,
        "decisive_quotes_match_text": quotes_match,
        "decisive_quote_matches_anchor": anchor_quote_match,
        "rationale": case["rationale"],
        "explanation": result.explanation if result else None,
        "interpreted_claim": result.claim.statement if result else None,
        "notices": [*report.notices, *getattr(result, "notices", [])],
        "citations": [
            {
                "title": citation.evidence.paper.title,
                "relation": citation.relation,
                "quote": getattr(citation, "quote", None),
                "page": getattr(citation.evidence, "page", None),
                "url": getattr(citation.evidence, "pdf_url", None) or citation.evidence.paper.url,
                "identifiers": citation.evidence.paper.identifiers,
            }
            for citation in result.citations
        ]
        if result
        else [],
    }


class RequestPacer:
    """Share the minimum interval between request starts across live cases."""

    def __init__(self, requests_per_minute):
        self.interval = 60 / requests_per_minute
        self.next_start = 0.0
        self.lock = asyncio.Lock()

    async def wait(self):
        async with self.lock:
            loop = asyncio.get_running_loop()
            delay = self.next_start - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            self.next_start = loop.time() + self.interval


class MeasuredModel:
    """Cap each published case at three model requests and collect available usage."""

    def __init__(self, model, pacer=None):
        self.model = model
        self.pacer = pacer
        self.calls = []

    def with_structured_output(self, schema, **kwargs):
        structured = self.model.with_structured_output(schema, include_raw=True, **kwargs)

        async def invoke(messages, config):
            if len(self.calls) >= 3:
                raise RuntimeError("Published case exceeded its three-request model budget")
            call = {"stage": "decompose" if schema is Decomposition else "assess", "sent": False}
            if schema is Assessment:
                call["input"] = json.loads(messages[-1].content)
            self.calls.append(call)
            queued = time.perf_counter()
            try:
                if self.pacer is not None:
                    await self.pacer.wait()
            except asyncio.CancelledError:
                call.update(
                    outcome="cancelled_before_send",
                    queue_seconds=round(time.perf_counter() - queued, 4),
                )
                raise
            start = time.perf_counter()
            call.update(sent=True, queue_seconds=round(start - queued, 4))
            try:
                result = await structured.ainvoke(messages, config=config)
                if usage := getattr(result["raw"], "usage_metadata", None):
                    call["tokens"] = dict(usage)
                parsed = result["parsed"]
                if not isinstance(parsed, schema) or result["parsing_error"]:
                    call["outcome"] = "schema_error"
                    raise result["parsing_error"] or ValueError("No valid structured output")
                call["output"] = parsed.model_dump(mode="json")
                if schema is Decomposition and (len(parsed.claims) != 1 or parsed.has_more_claims):
                    # Cancel verification of this benchmark case without discarding the extraction.
                    call["outcome"] = "decomposition_mismatch"
                    return Decomposition(claims=[], has_more_claims=False)
                call["outcome"] = "completed"
                return parsed
            except asyncio.CancelledError:
                call["outcome"] = "cancelled_in_flight"
                raise
            except Exception as exc:
                call["error_type"] = type(exc).__name__
                code = (
                    getattr(exc, "status_code", None)
                    or getattr(exc, "code", None)
                    or getattr(exc.__cause__, "code", None)
                )
                if isinstance(code, int):
                    call.update(outcome="http_error", status_code=code)
                elif "outcome" not in call:
                    call["outcome"] = (
                        "request_timeout" if isinstance(exc, TimeoutError) else "request_error"
                    )
                raise
            finally:
                call["seconds"] = round(time.perf_counter() - start, 4)

        return RunnableLambda(invoke)


def load_cases(suite):
    if suite == "fixture":
        return [json.loads(line) for line in (HERE / "claims.jsonl").read_text().splitlines()]

    dataset = json.loads((HERE / "published-claims.json").read_text())
    cases = []
    case_ids = set()
    for case in dataset["cases"]:
        reference = dataset["papers"][case["paper"]]
        if (
            case["id"] in case_ids
            or case["page"] < 1
            or not case["excerpt"].strip()
            or evidence_text(case["excerpt"]) not in evidence_text(case["passage"])
            or case["split"] not in {"dev", "holdout"}
            or case["expected"] not in {*DECISIVE_RELATIONS, "insufficient_evidence"}
        ):
            raise ValueError(f"Invalid published case {case['id']}")
        case_ids.add(case["id"])
        cases.append({**case, "reference": reference})
    return cases


async def evaluate_controlled(case, model, settings, *, pacer=None):
    reference = case["reference"]
    claim = Claim(statement=case["claim"], search_query=case["id"])
    passage = Evidence(
        id="E1",
        paper=PaperReference(title=reference["title"], identifiers=reference["identifiers"]),
        page=case["page"],
        pdf_url=reference["pdf_url"],
        passage=case["passage"],
        provenance=[Provenance(source="research_mcp", rank=1)],
    )
    measured_model = MeasuredModel(model, pacer)
    assess = measured_model.with_structured_output(Assessment, method="json_schema")
    start = time.perf_counter()
    try:
        async with asyncio.timeout(settings.fact_check_timeout_seconds):
            assessment = await assess_claim(assess, claim, [passage])
    except TimeoutError:
        report = FactCheckReport(results=[], status="timeout", notices=["Assessment timed out."])
    except Exception:
        if not measured_model.calls or not measured_model.calls[-1].get("error_type"):
            raise
        report = FactCheckReport(results=[], status="unavailable", notices=["Assessment failed."])
    else:
        report = FactCheckReport(
            results=[ground_assessment(claim, [passage], assessment)], notices=[]
        )
    return {
        **score(case, report),
        "claim": case["claim"],
        "model_calls": measured_model.calls,
        "seconds": round(time.perf_counter() - start, 4),
    }


async def evaluate_published(
    case, model, scifact, research, settings, *, abstracts_only, pacer=None
):
    calls = []
    reference = case["reference"]

    def measure(label, invoke):
        async def wrapped(*args):
            start = time.perf_counter()
            record = {"source": label, "query": args[-1]}
            if label == "pdf":
                record["target_paper"] = args[0].model_dump(mode="json")
            calls.append(record)
            try:
                batch = await invoke(*args)
                papers = [*batch.papers, *(entry.paper for entry in batch.evidence)]
                record.update(
                    status=batch.status,
                    paper_titles=list(dict.fromkeys(paper.title for paper in papers)),
                    reference_found=any(matches_paper(paper, reference) for paper in papers),
                    reference_passage_found=any(
                        matches_paper(entry.paper, reference) for entry in batch.evidence
                    ),
                    reference_excerpt_found=any(
                        matches_paper(entry.paper, reference)
                        and evidence_text(case["excerpt"]) in evidence_text(entry.passage)
                        for entry in batch.evidence
                    ),
                    notices=batch.notices,
                )
                return batch
            except asyncio.CancelledError:
                record["status"] = "cancelled"
                raise
            except Exception as exc:
                record.update(status="error", error_type=type(exc).__name__)
                raise
            finally:
                record["seconds"] = round(time.perf_counter() - start, 4)

        return wrapped

    measured_model = MeasuredModel(model, pacer)
    workflow = build_workflow(
        measured_model,
        measure("scifact", scifact.retrieve),
        measure("research_mcp", research.search),
        search_pdf=None if abstracts_only else measure("pdf", research.search_pdf),
        timeout=settings.fact_check_timeout_seconds,
    )
    start = time.perf_counter()
    result = (await workflow.ainvoke({"request": case["claim"]}))["report"]
    model_calls = measured_model.calls
    first = model_calls[0] if model_calls else {}
    return {
        **score(case, result),
        "extraction_outcome": first.get("outcome", "not_called"),
        "claim": case["claim"],
        "reference": {
            **reference,
            "page": case["page"],
            "location": case["location"],
            "excerpt": case["excerpt"],
        },
        "reference_retrieved": any(record.get("reference_found") for record in calls),
        "reference_passage_retrieved": any(
            record.get("reference_passage_found") for record in calls
        ),
        "reference_excerpt_retrieved": any(
            record.get("reference_excerpt_found") for record in calls
        ),
        "retrieval": calls,
        "model_calls": model_calls,
        "seconds": round(time.perf_counter() - start, 4),
    }


async def run(args):
    cases = [
        case
        for case in load_cases(args.suite)
        if (args.split == "all" or args.split == case["split"])
        and (args.case is None or args.case == case["id"])
    ][: args.limit]
    if not cases:
        raise ValueError("No cases match the selected split and case ID")
    live_model = None
    if args.mode == "gemini":
        settings = Settings()
        live_model = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            api_key=settings.google_api_key,
            vertexai=False,
            timeout=settings.gemini_timeout_seconds,
            **({"max_retries": 1, "max_output_tokens": 2048} if args.suite == "published" else {}),
        )
    rows = []
    evidence_mode = getattr(args, "evidence", "live")
    if args.suite == "published":
        if evidence_mode == "live":
            scifact = SciFactSource(settings.scifact_timeout_seconds)
            research = ResearchSource(
                research_client(settings),
                settings.research_mcp_discovery_seconds,
                settings.research_mcp_timeout_seconds,
            )
        slots = asyncio.Semaphore(2)
        pacer = RequestPacer(args.requests_per_minute) if args.requests_per_minute else None

        async def evaluate(case):
            async with slots:
                if evidence_mode == "controlled":
                    row = await evaluate_controlled(case, live_model, settings, pacer=pacer)
                else:
                    row = await evaluate_published(
                        case,
                        live_model,
                        scifact,
                        research,
                        settings,
                        abstracts_only=args.abstracts_only,
                        pacer=pacer,
                    )
                print(f"{row['id']}: {row['actual'] or row['status']}", file=sys.stderr, flush=True)
                return row

        rows = await asyncio.gather(*(evaluate(case) for case in cases))
    else:
        for case in cases:
            model = live_model if live_model is not None else ScriptedModel(case)
            research = frozen_research(case)
            args_by_version = {} if args.abstracts_only else {"search_pdf": research.search_pdf}
            workflow = build_workflow(model, empty_search, research.search, **args_by_version)
            start = time.perf_counter()
            report = (await workflow.ainvoke({"request": case["claim"]}))["report"]
            rows.append({**score(case, report), "seconds": round(time.perf_counter() - start, 4)})
    durations = sorted(row["seconds"] for row in rows)
    model_calls = [call for row in rows for call in row.get("model_calls", [])]
    return {
        "suite": args.suite,
        "evidence": evidence_mode if args.suite == "published" else "fixture",
        "mode": args.mode,
        "model": live_model.model if live_model is not None else "scripted fixture",
        "requests_per_minute": args.requests_per_minute,
        "split": args.split,
        "summary": {
            "cases": len(rows),
            **{
                key: sum(row[key] is True for row in rows)
                for key in (
                    "correct",
                    "verdict_matches_label",
                    "grounding_rejected",
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
            "partial_cases": sum(row["full_text_status"] is not None for row in rows),
            "score_statuses": dict(Counter(row["score_status"] for row in rows)),
            "median_seconds": statistics.median(durations),
            "p95_seconds": durations[math.ceil(len(durations) * 0.95) - 1],
            **(
                {
                    "reference_retrieved": sum(row["reference_retrieved"] for row in rows),
                    "reference_passage_retrieved": sum(
                        row["reference_passage_retrieved"] for row in rows
                    ),
                    "reference_excerpt_retrieved": sum(
                        row["reference_excerpt_retrieved"] for row in rows
                    ),
                    "extraction_outcomes": dict(Counter(row["extraction_outcome"] for row in rows)),
                }
                if args.suite == "published" and evidence_mode == "live"
                else {}
            ),
            **(
                {
                    "model_calls": sum(call["sent"] for call in model_calls),
                    "model_outcomes": dict(Counter(call["outcome"] for call in model_calls)),
                    "model_http_errors": dict(
                        Counter(
                            str(call["status_code"])
                            for call in model_calls
                            if "status_code" in call
                        )
                    ),
                }
                if args.suite == "published"
                else {}
            ),
        },
        "cases": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("fixture", "published"), default="fixture")
    parser.add_argument("--evidence", choices=("live", "controlled"), default="live")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case", help="Run a single case by ID.")
    parser.add_argument("--split", choices=("dev", "holdout", "all"), default="all")
    parser.add_argument("--mode", choices=("fixture", "gemini"), default="fixture")
    parser.add_argument(
        "--requests-per-minute", type=int, help="Pace model request starts in the published suite."
    )
    parser.add_argument("--abstracts-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--check", action="store_true", help="Enforce deterministic regression checks."
    )
    options = parser.parse_args()
    if options.limit is not None and options.limit < 1:
        parser.error("--limit must be positive")
    if options.suite == "published" and options.mode != "gemini":
        parser.error("the published suite requires --mode gemini")
    if options.evidence == "controlled" and (
        options.suite != "published" or options.abstracts_only
    ):
        parser.error("--evidence controlled requires the published suite and its fixed passages")
    if options.requests_per_minute is not None and (
        options.requests_per_minute < 1 or options.suite != "published"
    ):
        parser.error("--requests-per-minute requires a positive value and the published suite")
    if options.check and (
        options.mode != "fixture" or options.abstracts_only or options.limit or options.case
    ):
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
