import pytest

from research_mcp.config import Settings


@pytest.fixture
def settings():
    return Settings(
        _env_file=None,
        openalex_api_key=None,
        semanticscholar_api_key=None,
        openalex_interval=0,
        arxiv_interval=0,
        semanticscholar_interval=0,
        retry_attempts=1,
    )
