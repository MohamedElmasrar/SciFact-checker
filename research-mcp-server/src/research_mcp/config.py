from typing import Annotated

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PositiveSeconds = Annotated[float, Field(gt=0, le=600)]
Interval = Annotated[float, Field(ge=0, le=600)]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RESEARCH_", env_file=".env", env_ignore_empty=True, extra="ignore"
    )

    openalex_api_key: SecretStr | None = None
    openalex_email: str | None = None
    semanticscholar_api_key: SecretStr | None = None
    openalex_timeout: PositiveSeconds = 15
    arxiv_timeout: PositiveSeconds = 20
    semanticscholar_timeout: PositiveSeconds = 20
    pdf_timeout: PositiveSeconds = 30
    pdf_total_timeout: PositiveSeconds = 60
    openalex_interval: Interval = 0.2
    arxiv_interval: Interval = 3
    semanticscholar_interval: Interval = 1
    retry_attempts: int = Field(3, ge=1, le=8)
    retry_max_delay: PositiveSeconds = 30

    metadata_cache_size: int = Field(512, ge=0)
    search_cache_size: int = Field(128, ge=0)
    document_cache_bytes: int = Field(32 * 1024 * 1024, ge=0)
    metadata_cache_ttl: Annotated[float, Field(gt=0)] = 3600
    search_cache_ttl: Annotated[float, Field(gt=0)] = 900
    document_cache_ttl: Annotated[float, Field(gt=0)] = 3600

    pdf_max_bytes: int = Field(25 * 1024 * 1024, gt=0)
    pdf_max_pages: int = Field(40, ge=1, le=500)
    content_default_chars: int = Field(20_000, ge=1)
    content_max_chars: int = Field(100_000, ge=1)
    log_level: str = "INFO"
