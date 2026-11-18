import re
from dataclasses import dataclass
from urllib.parse import unquote, urlsplit

DOI = re.compile(r"10\.\d{4,9}/[^\s\x00-\x1f\x7f]+", re.IGNORECASE)
ARXIV = re.compile(
    r"(?P<base>\d{4}\.\d{4,5}|[a-z][a-z0-9.\-]*/\d{7})(?:v\d+)?", re.IGNORECASE
)
OPENALEX = re.compile(r"w\d+", re.IGNORECASE)
S2 = re.compile(r"[a-f0-9]{40}", re.IGNORECASE)


@dataclass(frozen=True)
class Identifier:
    kind: str
    value: str


def parse_identifier(raw: str) -> Identifier:
    value = raw.strip()
    if value.lower().startswith(("http://", "https://")):
        url = urlsplit(value)
        host = url.hostname
        value = unquote(url.path.lstrip("/"))
        if host in {"doi.org", "dx.doi.org"}:
            value = f"doi:{value}"
        elif host in {"arxiv.org", "www.arxiv.org"} and value.startswith(
            ("abs/", "pdf/")
        ):
            value = "arxiv:" + value.split("/", 1)[1].removesuffix(".pdf")
        elif host in {"openalex.org", "api.openalex.org"}:
            value = value.removeprefix("works/")
        elif host in {
            "semanticscholar.org",
            "www.semanticscholar.org",
        } and value.startswith("paper/"):
            value = value.rstrip("/").rsplit("/", 1)[-1]
        else:
            raise ValueError(
                "Use a DOI, arXiv ID, OpenAlex work ID, or Semantic Scholar paper ID"
            )

    prefix, sep, body = value.partition(":")
    if sep and prefix.lower() in {
        "doi",
        "arxiv",
        "openalex",
        "semanticscholar",
        "s2",
        "corpusid",
    }:
        kind = {"s2": "semanticscholar", "corpusid": "corpus_id"}.get(
            prefix.lower(), prefix.lower()
        )
        value = body.strip()
    else:
        kind = None

    for candidate, pattern in (
        ("doi", DOI),
        ("arxiv", ARXIV),
        ("openalex", OPENALEX),
        ("semanticscholar", S2),
    ):
        if kind in (None, candidate) and pattern.fullmatch(value):
            normalized = value.upper() if candidate == "openalex" else value.lower()
            return Identifier(candidate, normalized)
    if kind == "corpus_id" and re.fullmatch(r"[0-9]{1,20}", value):
        return Identifier(kind, str(int(value)))
    raise ValueError(
        "Use a DOI, arXiv ID, OpenAlex work ID, or Semantic Scholar paper ID"
    )


def normalized_id(value: str | int | None, kind: str) -> str | None:
    if value is None:
        return None
    candidate = str(value)
    if kind == "corpus_id":
        candidate = f"CorpusId:{candidate}"
    try:
        identifier = parse_identifier(candidate)
    except ValueError:
        return None
    return identifier.value if identifier.kind == kind else None


def arxiv_base(value: str) -> str:
    match = ARXIV.fullmatch(value)
    return match["base"].lower() if match else value.lower()
