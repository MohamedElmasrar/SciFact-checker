from typing import Literal

from pydantic import AliasChoices, Field, HttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")

    google_api_key: SecretStr = Field(
        validation_alias=AliasChoices("GOOGLE_API_KEY", "GEMINI_API_KEY")
    )
    gemini_model: str = "gemini-3.5-flash-lite"
    fact_check_timeout_seconds: float = Field(default=90, gt=0, allow_inf_nan=False)
    gemini_timeout_seconds: int = Field(default=45, gt=0)
    research_mcp_transport: Literal["http", "stdio"] = "stdio"
    research_mcp_url: HttpUrl = HttpUrl("http://127.0.0.1:8000/mcp")
    research_mcp_discovery_seconds: float = Field(default=5, gt=0)
    research_mcp_timeout_seconds: float = Field(default=75, gt=0)
    scifact_timeout_seconds: float = Field(default=45, gt=0)
