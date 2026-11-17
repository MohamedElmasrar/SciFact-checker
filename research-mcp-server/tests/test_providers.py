import json
from urllib.parse import parse_qs, urlsplit

import httpx
import requests
from pydantic import SecretStr

from research_mcp.identifiers import parse_identifier
from research_mcp.models import Provider, Relationship, SearchOptions
from research_mcp.providers.arxiv import ArxivProvider
from research_mcp.providers.openalex import OpenAlexProvider
from research_mcp.providers.semantic import SemanticScholarProvider

DOI = "10.1234/example"
S2_ID = "a" * 40
TITLE = "Example Study"


def reply(request, *, body, status=200):
    response = requests.Response()
    response.status_code = status
    response._content = body
    response.request = request
    response.url = request.url
    return response


async def test_openalex_queries_and_normalization(settings, monkeypatch):
    work = {
        "id": "https://openalex.org/W123",
        "doi": f"https://doi.org/{DOI.upper()}",
        "title": TITLE,
        "abstract_inverted_index": {"paper": [2], "A": [0], "test": [1]},
        "publication_year": 2023,
        "publication_date": "2023-04-01",
        "cited_by_count": 17,
        "authorships": [
            {
                "author": {
                    "display_name": "Ada Smith",
                    "orcid": "https://orcid.org/confirmed",
                }
            }
        ],
        "primary_location": {
            "source": {"display_name": "Study Journal"},
            "is_oa": False,
            "pdf_url": "https://example.com/paywalled.pdf",
        },
        "best_oa_location": {"is_oa": True, "pdf_url": "https://example.com/paper.pdf"},
        "topics": [{"display_name": "Biology"}],
        "referenced_works": ["https://openalex.org/W456"],
    }
    referenced = {"id": "https://openalex.org/W456", "title": "Prior study"}
    calls = []

    def send(adapter, request, **kwargs):
        assert kwargs["timeout"] == settings.openalex_timeout
        assert request.headers["Authorization"] == "Bearer test-key"
        url = urlsplit(request.url)
        params = parse_qs(url.query)
        calls.append(params)
        if url.path != "/works":
            only = work
            if "W999" in url.path:
                return reply(request, body=b'{"error":"not found"}', status=404)
        else:
            rows = (
                [referenced]
                if "openalex_id:W456" in params.get("filter", [""])[0]
                else [work]
            )
            only = {"results": rows, "meta": {"count": len(rows)}}
        return reply(request, body=json.dumps(only).encode())

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    provider = OpenAlexProvider(
        settings.model_copy(update={"openalex_api_key": SecretStr("test-key")})
    )
    papers = await provider.search(
        SearchOptions(
            query="cell & tissue",
            year_from=2020,
            year_to=2024,
            sort="most_cited",
            limit=2,
        )
    )
    assert calls[0]["search"] == ["cell & tissue"]
    assert calls[0]["sort"] == ["cited_by_count:desc"]
    assert "from_publication_date:2020-01-01" in calls[0]["filter"][0]
    assert papers[0].identifiers.doi == DOI
    assert papers[0].abstract == "A test paper"
    assert papers[0].authors[0].name == "Ada Smith"
    assert papers[0].pdf_urls == ["https://example.com/paper.pdf"]
    assert papers[0].sources[Provider.OPENALEX].citation_count == 17

    assert (await provider.get(parse_identifier(DOI))).identifiers.openalex == "W123"
    assert await provider.get(parse_identifier("W999")) is None
    citations = await provider.graph(parse_identifier("W123"), Relationship.CITING, 1)
    assert citations[0].title == TITLE
    assert calls[-1]["filter"] == ["cites:W123"]
    references = await provider.graph(
        parse_identifier("W123"), Relationship.REFERENCE, 2
    )
    assert references[0].identifiers.openalex == "W456"


ATOM = b"""<feed xmlns="http://www.w3.org/2005/Atom"
  xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"
  xmlns:arxiv="http://arxiv.org/schemas/atom">
  <opensearch:totalResults>1</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/abs/2301.00001v2</id>
    <updated>2023-04-14T00:00:00Z</updated><published>2023-01-01T00:00:00Z</published>
    <title>Example Study</title><summary>A study of cells.</summary>
    <author><name>Ada Smith</name></author>
    <link title="pdf" href="https://arxiv.org/pdf/2301.00001v2" type="application/pdf"/>
    <category term="q-bio.CB"/><arxiv:primary_category term="q-bio.CB"/>
    <arxiv:doi>10.1234/example</arxiv:doi><arxiv:journal_ref>Study Journal</arxiv:journal_ref>
  </entry>
</feed>"""


async def test_arxiv_queries_and_normalization(settings, monkeypatch):
    calls = []

    def send(adapter, request, **kwargs):
        assert kwargs["timeout"] == settings.arxiv_timeout
        calls.append(parse_qs(urlsplit(request.url).query))
        return reply(request, body=ATOM)

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    provider = ArxivProvider(settings)
    papers = await provider.search(
        SearchOptions(
            query="cell study", year_from=2021, year_to=2023, limit=3, sort="newest"
        )
    )
    assert calls[0]["sortBy"] == ["submittedDate"]
    assert calls[0]["search_query"] == [
        "(cell study) AND submittedDate:[202101010000 TO 202312312359]"
    ]
    assert papers[0].identifiers.arxiv == "2301.00001v2"
    assert papers[0].identifiers.doi == DOI
    assert papers[0].categories == ["q-bio.CB"]
    assert papers[0].citation_count is None
    assert papers[0].venue == "Study Journal"
    assert (
        await provider.get(parse_identifier("10.48550/arXiv.2301.00001v2"))
    ).title == TITLE
    assert calls[-1]["id_list"] == ["2301.00001v2"]
    assert await provider.get(parse_identifier(DOI)) is None


async def test_semantic_queries_auth_normalization_and_graph(settings, monkeypatch):
    calls = []
    row = {
        "paperId": S2_ID,
        "corpusId": 123,
        "externalIds": {"DOI": DOI, "ArXiv": "2301.00001"},
        "title": TITLE,
        "abstract": None,
        "citationCount": 29,
        "year": 2023,
        "publicationVenue": {"name": "Conference"},
        "fieldsOfStudy": ["Biology"],
        "openAccessPdf": {"url": "https://example.com/paper.pdf"},
    }

    def handle(request):
        calls.append(request)
        assert request.headers["x-api-key"] == "test-key"
        path = request.url.path
        if path.endswith("search/bulk"):
            return httpx.Response(200, json={"data": [row, {"title": None}]})
        if path.endswith("/references"):
            return httpx.Response(
                200, json={"data": [{"citedPaper": None}, {"citedPaper": row}]}
            )
        if path.endswith("/citations"):
            return httpx.Response(200, json={"data": [{"citingPaper": row}]})
        if "/recommendations/" in path:
            return httpx.Response(200, json={"recommendedPapers": [row]})
        if path.endswith("CorpusId:999"):
            return httpx.Response(404, json={"error": "Not found"})
        return httpx.Response(200, json=row)

    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda: original(transport=httpx.MockTransport(handle))
    )
    provider = SemanticScholarProvider(
        settings.model_copy(update={"semanticscholar_api_key": SecretStr("test-key")})
    )
    search = await provider.search(
        SearchOptions(
            query="C++ & learning", year_from=2022, year_to=2023, limit=2, sort="newest"
        )
    )
    assert calls[0].url.params["query"] == "C++ & learning"
    assert calls[0].url.params["year"] == "2022-2023"
    assert calls[0].url.params["sort"] == "publicationDate:desc"
    assert len(search) == 1
    assert search[0].identifiers.semanticscholar == S2_ID
    assert search[0].identifiers.corpus_id == "123"
    assert search[0].abstract is None
    assert search[0].venue == "Conference"
    assert search[0].citation_count == 29

    identifier = parse_identifier(DOI)
    assert (await provider.get(identifier)).title == TITLE
    assert await provider.get(parse_identifier("corpusid:999")) is None
    for relationship in Relationship:
        assert (await provider.graph(identifier, relationship, 2))[0].title == TITLE
