import json
from unittest.mock import AsyncMock

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

from fact_checker.agent import build_agent
from fact_checker.config import Settings
from fact_checker.sources import ResearchSource, SciFactSource, research_client


class ScriptedChat(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        return RunnableLambda(lambda _: schema(claims=[], has_more_claims=False))


async def test_adapter_discovers_packaged_mcp_tools_over_stdio(monkeypatch):
    monkeypatch.delenv("RESEARCH_MCP_TRANSPORT", raising=False)
    settings = Settings(_env_file=None, google_api_key="test")
    assert settings.research_mcp_transport == "stdio"
    tools = await research_client(settings).get_tools(server_name="research")
    by_name = {tool.name: tool for tool in tools}
    assert {"search_papers", "search_paper_content"} <= by_name.keys()

    # The server rejects this range before contacting a research provider.
    result = await by_name["search_papers"].ainvoke(
        {
            "type": "tool_call",
            "name": "search_papers",
            "id": "stdio-test",
            "args": {"query": "test", "year_from": 2020, "year_to": 2019},
        }
    )
    assert result.status == "error"
    assert "year_from" in str(result.content)


def call(name, args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "test-call"}])


def build_sources(*tools):
    client = AsyncMock()
    client.get_tools.return_value = tools
    retriever = AsyncMock()
    retriever.ainvoke.return_value = [
        Document(
            id="12",
            page_content="An indexed abstract.",
            metadata={"title": "Indexed paper", "retrieval_strategy": "hybrid"},
        )
    ]
    return ResearchSource(client, 1, 1), SciFactSource(1, factory=lambda: retriever), retriever


async def test_general_chat_does_not_retrieve_evidence():
    research, scifact, retriever = build_sources()
    model = ScriptedChat(responses=[AIMessage(content="Hello")])
    result = await build_agent(model, scifact, research).ainvoke(
        {"messages": [{"role": "user", "content": "Hello"}]}
    )
    assert result["messages"][-1].content == "Hello"
    retriever.ainvoke.assert_not_awaited()


async def test_scifact_tool_returns_structured_evidence_and_progress():
    research, scifact, retriever = build_sources()
    model = ScriptedChat(
        responses=[call("retrieve_scifact", {"query": "  query  "}), AIMessage(content="Found it.")]
    )
    agent = build_agent(model, scifact, research)
    events = [
        event
        async for event in agent.astream(
            {"messages": [{"role": "user", "content": "Search SciFact"}]},
            stream_mode=["values", "custom"],
        )
    ]
    assert ("custom", {"type": "scientific_progress", "stage": "retrieve_scifact"}) in events
    payload = json.loads(events[-1][1]["messages"][-2].content)
    assert payload["evidence"][0]["paper"]["identifiers"] == {"scifact": "12"}
    assert payload["evidence"][0]["provenance"][0]["metadata"]["retrieval_strategy"] == "hybrid"
    retriever.ainvoke.assert_awaited_once_with("query")


async def test_discovered_research_tool_is_executable_and_failures_are_tool_messages():
    @tool
    async def search_papers(query: str) -> str:
        """Search for papers."""
        return json.dumps({"papers": [{"title": query, "abstract": "Example results."}]})

    @tool
    async def get_paper(paper_id: str) -> str:
        """Look up a paper."""
        raise ConnectionError

    research, scifact, _ = build_sources(search_papers, get_paper)
    for tool_call, status in [
        (call("search_papers", {"query": "A paper"}), "success"),
        (call("get_paper", {"paper_id": "arXiv:123"}), "error"),
    ]:
        model = ScriptedChat(responses=[tool_call, AIMessage(content="Done")])
        result = await build_agent(model, scifact, research).ainvoke(
            {"messages": [{"role": "user", "content": "Find a paper"}]}
        )
        message = result["messages"][-2]
        assert isinstance(message, ToolMessage)
        assert message.status == status
        if status == "success":
            assert json.loads(message.content)["papers"][0]["title"] == "A paper"
    research.client.get_tools.assert_awaited_once()


async def test_fact_check_tool_returns_report_for_empty_decomposition():
    research, scifact, retriever = build_sources()
    model = ScriptedChat(
        responses=[call("fact_check", {"request": "Verify this"}), AIMessage(content="No claim.")]
    )
    result = await build_agent(model, scifact, research).ainvoke(
        {"messages": [{"role": "user", "content": "Verify this"}]}
    )
    assert json.loads(result["messages"][-2].content) == {
        "results": [],
        "notices": ["No verifiable scientific claim was identified."],
        "status": "completed",
    }
    retriever.ainvoke.assert_not_awaited()


async def test_research_uses_structured_artifact_and_keeps_partial_provider_results():
    @tool(response_format="content_and_artifact")
    async def search_papers(query: str, limit: int):
        """Search for papers."""
        return "display text", {
            "structured_content": {
                "papers": [{"title": "A paper", "abstract": "Actual passage."}],
                "issues": [{"source": "arxiv", "code": "timeout"}],
            }
        }

    research, _, _ = build_sources(search_papers)
    batch = await research.search("test")
    assert [entry.passage for entry in batch.evidence] == ["Actual passage."]
    assert batch.notices == ["Research provider arxiv reported timeout."]


async def test_slow_sources_are_reported():
    import asyncio

    @tool
    async def search_papers(query: str, limit: int) -> str:
        """Search for papers."""
        await asyncio.sleep(1)
        return "{}"

    research, scifact, retriever = build_sources(search_papers)
    research.timeout = 0.01
    retriever.ainvoke.side_effect = ConnectionError
    assert (await research.search("test")).notices == ["Research MCP search timed out."]
    assert (await scifact.retrieve("test")).notices == ["SciFact retrieval is unavailable."]
