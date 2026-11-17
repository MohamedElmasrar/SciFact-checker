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
from .evidence import EvidenceBatch, from_research, from_scifact

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
            return EvidenceBatch(notices=["SciFact retrieval timed out."])
        except Exception as exc:
            log.warning("retrieval_failed source=scifact error_type=%s", type(exc).__name__)
            return EvidenceBatch(notices=["SciFact retrieval is unavailable."])
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
        search = (await self.tools()).get("search_papers")
        if search is None:
            return EvidenceBatch(notices=["Research MCP search is unavailable."])
        try:
            message = await asyncio.wait_for(
                search.ainvoke(
                    {
                        "type": "tool_call",
                        "name": search.name,
                        "id": uuid4().hex,
                        "args": {"query": query[:500], "limit": 5},
                    }
                ),
                self.timeout,
            )
            if message.status == "error" or (payload := structured_payload(message)) is None:
                raise ValueError("Research search returned no structured results")
            notices = [
                f"Research provider {issue['source']} reported {issue['code']}."
                for issue in payload.get("issues", [])
            ]
            evidence = [
                entry
                for rank, paper in enumerate(payload["papers"], 1)
                if (entry := from_research(paper, rank)) is not None
            ]
            if payload["papers"] and not evidence:
                notices.append("The research search returned papers without usable abstracts.")
            return EvidenceBatch(evidence=evidence, notices=notices)
        except TimeoutError:
            log.warning("retrieval_timeout source=research_mcp")
            return EvidenceBatch(notices=["Research MCP search timed out."])
        except Exception as exc:
            log.warning("retrieval_failed source=research_mcp error_type=%s", type(exc).__name__)
            return EvidenceBatch(notices=["Research MCP search is unavailable."])
