import asyncio
import json
import logging
import time
from collections.abc import Callable
from threading import Lock
from typing import Any
from uuid import uuid4

from langchain_core.messages import ToolMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from .config import Settings
from .evidence import (
    Evidence,
    EvidenceBatch,
    PaperReference,
    Provenance,
    excerpt,
    from_research,
    from_scifact,
    research_reference,
    safe_url,
)

log = logging.getLogger(__name__)


def load_scifact() -> BaseRetriever:
    from scifact_retriever import Settings as SciFactSettings
    from scifact_retriever import build_retriever
    from torch import set_num_threads

    settings = SciFactSettings()
    set_num_threads(settings.torch_threads)
    return build_retriever(settings)


class SciFactSource:
    def __init__(self, timeout: float, factory: Callable[[], BaseRetriever] = load_scifact):
        self.timeout = timeout
        self.factory = factory
        self._retriever: BaseRetriever | None = None
        self._lock = Lock()
        self._slots = asyncio.Semaphore(1)

    def _get_retriever(self) -> BaseRetriever:
        with self._lock:
            if self._retriever is None:
                self._retriever = self.factory()
            return self._retriever

    async def retrieve(self, query: str) -> EvidenceBatch:
        if not query.strip():
            return EvidenceBatch(notices=["Provide a nonblank scientific query."])

        async def search() -> EvidenceBatch:
            retriever = await asyncio.to_thread(self._get_retriever)
            docs = await retriever.ainvoke(query.strip())
            return EvidenceBatch(
                evidence=[from_scifact(doc, rank) for rank, doc in enumerate(docs, 1)]
            )

        # An executor call cannot be interrupted. Keep the slot until it really finishes.
        task = None
        try:
            async with asyncio.timeout(self.timeout):
                await self._slots.acquire()
                task = asyncio.create_task(search())
                await asyncio.shield(task)
                return task.result()
        except TimeoutError:
            log.warning("retrieval_timeout source=scifact")
            return EvidenceBatch(status="timeout", notices=["SciFact retrieval timed out."])
        except Exception as exc:
            log.warning("retrieval_failed source=scifact error_type=%s", type(exc).__name__)
            return EvidenceBatch(
                status="unavailable", notices=["SciFact retrieval is unavailable."]
            )
        finally:
            if task is not None:

                def release(finished: asyncio.Task) -> None:
                    self._slots.release()
                    if not finished.cancelled():
                        finished.exception()

                task.add_done_callback(release)


def research_client(settings: Settings) -> MultiServerMCPClient:
    if settings.research_mcp_transport == "stdio":
        connection = {
            "transport": "stdio",
            "command": "uv",
            "args": [
                "--directory",
                str(settings.research_mcp_directory.resolve()),
                "run",
                "--locked",
                "research-mcp",
            ],
        }
    else:
        connection = {"transport": "streamable_http", "url": str(settings.research_mcp_url)}
    return MultiServerMCPClient({"research": connection})


def structured_payload(message: ToolMessage) -> dict[str, Any] | None:
    artifact = message.artifact
    if isinstance(artifact, dict) and isinstance(artifact.get("structured_content"), dict):
        return artifact["structured_content"]
    blocks = message.content if isinstance(message.content, list) else [message.content]
    for block in blocks:
        text = block.get("text") if isinstance(block, dict) else block
        if isinstance(text, str):
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    return None


def paper_lookup_id(paper: PaperReference) -> str | None:
    for key in ("doi", "arxiv", "openalex", "semanticscholar", "corpus_id"):
        if identifier := paper.identifiers.get(key):
            return f"CorpusId:{identifier}" if key == "corpus_id" else identifier
    return None


class ResearchSource:
    def __init__(self, client: MultiServerMCPClient, discovery_timeout: float, timeout: float):
        self.client = client
        self.discovery_timeout = discovery_timeout
        self.timeout = timeout
        self._tools: dict[str, BaseTool] = {}
        self._refresh_at = 0.0
        self._lock = asyncio.Lock()

    async def tools(self) -> dict[str, BaseTool]:
        if time.monotonic() < self._refresh_at:
            return self._tools
        async with self._lock:
            if time.monotonic() >= self._refresh_at:
                try:
                    tools = await asyncio.wait_for(
                        self.client.get_tools(server_name="research"), self.discovery_timeout
                    )
                    self._tools = {tool.name: tool for tool in tools}
                    self._refresh_at = time.monotonic() + 300
                    log.info("research_tools_discovered count=%d", len(tools))
                except Exception as exc:
                    # Cached schemas remain usable if the service comes back between turns.
                    self._refresh_at = time.monotonic() + 10
                    log.warning("research_discovery_failed error_type=%s", type(exc).__name__)
        return self._tools

    async def search(self, query: str) -> EvidenceBatch:
        try:
            payload = await self._call("search_papers", {"query": query[:500], "limit": 5})
            issues = payload.get("issues", [])
            notices = [
                f"Research provider {issue['source']} reported {issue['code']}." for issue in issues
            ]
            if {"openalex", "arxiv", "semanticscholar"} <= {issue["source"] for issue in issues}:
                return EvidenceBatch(status="unavailable", notices=notices)
            papers = [research_reference(paper) for paper in payload["papers"]]
            evidence = [
                entry
                for rank, (paper, reference) in enumerate(
                    zip(payload["papers"], papers, strict=True), 1
                )
                if (entry := from_research(paper, rank, reference=reference)) is not None
            ]
            if payload["papers"] and not evidence:
                notices.append("The research search returned papers without usable abstracts.")
            return EvidenceBatch(evidence=evidence, notices=notices, papers=papers)
        except TimeoutError:
            log.warning("retrieval_timeout source=research_mcp")
            return EvidenceBatch(status="timeout", notices=["Research MCP search timed out."])
        except Exception as exc:
            log.warning("retrieval_failed source=research_mcp error_type=%s", type(exc).__name__)
            return EvidenceBatch(
                status="unavailable", notices=["Research MCP search is unavailable."]
            )

    async def search_pdf(self, paper: PaperReference, query: str) -> EvidenceBatch:
        paper_id = paper_lookup_id(paper)
        if paper_id is None:
            return EvidenceBatch()
        try:
            payload = await self._call(
                "search_paper_content", {"paper_id": paper_id, "query": query[:500], "limit": 2}
            )
            issues = payload.get("issues", [])
            notices = [f"PDF retrieval reported {issue['code']}." for issue in issues]
            url = safe_url((payload.get("source") or {}).get("pdf_url"))
            if not url:
                codes = {issue["code"] for issue in issues}
                if "pdf_timeout" in codes:
                    return EvidenceBatch(status="timeout", notices=notices)
                if codes <= {"pdf_unavailable", "not_found"}:
                    return EvidenceBatch(notices=notices or ["No accessible PDF was found."])
                return EvidenceBatch(status="unavailable", notices=notices)
            evidence = [
                Evidence(
                    paper=paper,
                    passage=excerpt(passage["text"]),
                    page=passage["page"],
                    pdf_url=url,
                    provenance=[Provenance(source="research_mcp", rank=rank)],
                )
                for rank, passage in enumerate(payload["passages"], 1)
                if passage["text"].strip()
            ]
            return EvidenceBatch(evidence=evidence, notices=notices)
        except TimeoutError:
            return EvidenceBatch(status="timeout", notices=["PDF retrieval timed out."])
        except Exception as exc:
            log.warning("pdf_retrieval_failed error_type=%s", type(exc).__name__)
            return EvidenceBatch(status="unavailable", notices=["PDF retrieval is unavailable."])

    async def _call(self, name: str, args: dict) -> dict[str, Any]:
        selected = (await self.tools()).get(name)
        if selected is None:
            raise ValueError(f"Research tool {name} is unavailable")
        message = await asyncio.wait_for(
            selected.ainvoke({"type": "tool_call", "name": name, "id": uuid4().hex, "args": args}),
            self.timeout,
        )
        if message.status == "error" or (payload := structured_payload(message)) is None:
            raise ValueError(f"Research tool {name} returned no structured results")
        return payload
