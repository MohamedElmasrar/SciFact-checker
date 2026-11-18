"""Run the production graph against isolated Gemini and MCP fixtures for Playwright."""

import argparse
import asyncio
import json
import os
import subprocess
import sys
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from time import sleep
from typing import TypedDict

import uvicorn
from mcp.server.fastmcp import FastMCP

CLAIM = "The coated panel ran cooler under the lamp."
FAILING_CLAIM = "The backup sensor maintained its calibration."
QUOTE = "The coated panel was 8 °C cooler than the control."
PAPER = {
    "title": "Panel Coating Study",
    "identifiers": {"openalex": "W7123"},
    "abstract": "The experiment compared temperatures but the abstract omitted the results.",
}


class PaperResults(TypedDict):
    papers: list[dict]
    issues: list[dict]


class PassageResults(TypedDict):
    source: dict
    passages: list[dict]
    issues: list[dict]


def research_app():
    mcp = FastMCP("Frozen science fixtures", stateless_http=True, json_response=True)

    @mcp.tool()
    async def search_papers(query: str, limit: int) -> PaperResults:
        """Search the frozen fixture papers."""
        return {"papers": [PAPER], "issues": []}

    @mcp.tool()
    async def search_paper_content(paper_id: str, query: str, limit: int) -> PassageResults:
        """Retrieve a frozen, page-numbered result from the fixture PDF."""
        assert paper_id == "W7123"
        await asyncio.sleep(1)
        return {
            "source": {"pdf_url": "https://example.org/coating-study.pdf"},
            "passages": [{"page": 4, "text": QUOTE}],
            "issues": [],
        }

    return mcp.streamable_http_app()


class GeminiFixture(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        parts = body["contents"][-1]["parts"]
        text = " ".join(part.get("text", "") for part in parts)
        configuration = body.get("generationConfig", {})
        if configuration.get("responseMimeType") == "application/json":
            sleep(0.5)
            if text.startswith("{"):
                request = json.loads(text)
                if request["claim"] == FAILING_CLAIM:
                    self.reply(
                        {
                            "error": {
                                "code": 400,
                                "status": "INVALID_ARGUMENT",
                                "message": "Simulated assessment failure",
                            }
                        },
                        status=400,
                    )
                    return
                passage = next((p for p in request["passages"] if p["passage"] == QUOTE), None)
                result = {
                    "verdict": "supported" if passage else "insufficient_evidence",
                    "explanation": (
                        f"The measured difference is reported in [{passage['id']}]."
                        if passage
                        else "The abstract leaves the temperature difference unresolved."
                    ),
                    "evidence": (
                        [{"evidence_id": passage["id"], "relation": "supports", "quote": QUOTE}]
                        if passage
                        else []
                    ),
                    "needs_full_text": passage is None,
                }
            else:
                claims = [CLAIM, FAILING_CLAIM] if "both" in text else [FAILING_CLAIM]
                result = {
                    "claims": [{"statement": claim, "search_query": claim} for claim in claims],
                    "has_more_claims": False,
                }
            answer = {"text": json.dumps(result)}
        elif any("functionResponse" in part for part in parts):
            answer = {"text": "The fact-check report is ready."}
        else:
            answer = {"functionCall": {"name": "fact_check", "args": {"request": text}}}
        self.reply(
            {
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [answer]},
                        "finishReason": "STOP",
                        "index": 0,
                    }
                ]
            },
            stream=":streamGenerateContent" in self.path,
        )

    def reply(self, payload, *, status=200, stream=False):
        data = json.dumps(payload)
        body = (f"data: {data}\n\n" if stream else data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/event-stream" if stream else "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=3201, type=int)
    options = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    with (
        TemporaryDirectory(prefix="fact-checker-smoke-") as directory,
        closing(ThreadingHTTPServer(("127.0.0.1", 0), GeminiFixture)) as gemini,
    ):
        gemini_thread = Thread(target=gemini.serve_forever, daemon=True)
        gemini_thread.start()
        # Let uvicorn bind a private ephemeral port and wait for MCP's lifespan to start.
        mcp = uvicorn.Server(
            uvicorn.Config(research_app(), host="127.0.0.1", port=0, log_level="warning")
        )
        mcp_thread = Thread(target=mcp.run, daemon=True)
        mcp_thread.start()
        try:
            for _ in range(250):
                if mcp.started:
                    break
                if not mcp_thread.is_alive():
                    raise RuntimeError("Research fixture failed to start")
                sleep(0.02)
            else:
                raise RuntimeError("Research fixture did not become ready")
            mcp_port = mcp.servers[0].sockets[0].getsockname()[1]
            config = json.loads((project / "langgraph.json").read_text())
            config.pop("env", None)
            config["dependencies"] = [str(project)]
            Path(directory, "langgraph.json").write_text(json.dumps(config))
            env = {
                **os.environ,
                "GOOGLE_API_KEY": "local-smoke-fixture",
                "GEMINI_MODEL": "gemini-3.5-flash-lite",
                "GOOGLE_GEMINI_BASE_URL": f"http://127.0.0.1:{gemini.server_port}",
                "RESEARCH_MCP_TRANSPORT": "http",
                "RESEARCH_MCP_URL": f"http://127.0.0.1:{mcp_port}/mcp",
                "SCIFACT_QDRANT_URL": f"http://127.0.0.1:{mcp_port}",
                "SCIFACT_QDRANT_TIMEOUT": "1",
                "LANGSMITH_TRACING": "false",
            }
            env.pop("LANGSMITH_API_KEY", None)
            env.pop("GEMINI_API_KEY", None)
            env.pop("SCIFACT_QDRANT_API_KEY", None)
            command = [
                sys.executable,
                str(Path(sys.executable).with_name("langgraph")),
                "dev",
                "--no-browser",
                "--no-reload",
                "--port",
                str(options.port),
            ]
            return subprocess.call(command, cwd=directory, env=env)
        finally:
            mcp.should_exit = True
            mcp_thread.join(timeout=3)
            gemini.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
