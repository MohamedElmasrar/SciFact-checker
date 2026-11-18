"""Runtime settings and reproducible defaults."""

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DENSE_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-TinyBERT-L2-v2"
DEFAULT_RERANK_REVISION = "81d1926f67cb8eee2c2be17ca9f793c7c3bd20cc"
SPARSE_MODEL = "Qdrant/bm25"
DEFAULT_QDRANT_PATH = Path.home() / ".local" / "share" / "scifact-retriever" / "qdrant"


class Strategy(StrEnum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class SearchOptions(BaseModel):
    model_config = {"frozen": True}

    strategy: Strategy = Strategy.HYBRID
    top_k: int = Field(default=5, ge=1)
    hybrid_depth: int = Field(default=50, ge=1)
    rerank: bool = False
    rerank_candidates: int = Field(default=10, ge=1)
    dense_min_score: float | None = Field(default=None, ge=-1, le=1)
    sparse_min_score: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    hybrid_min_score: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    exact_dense_search: bool = True

    @model_validator(mode="after")
    def validate_depths(self) -> "SearchOptions":
        if self.rerank and self.rerank_candidates < self.top_k:
            raise ValueError("rerank_candidates must be at least top_k when reranking")
        required_depth = self.rerank_candidates if self.rerank else self.top_k
        if self.strategy == Strategy.HYBRID and self.hybrid_depth < required_depth:
            raise ValueError("hybrid_depth must cover top_k and the active reranking depth")
        return self

    @property
    def score_threshold(self) -> float | None:
        return {
            Strategy.DENSE: self.dense_min_score,
            Strategy.SPARSE: self.sparse_min_score,
            Strategy.HYBRID: self.hybrid_min_score,
        }[self.strategy]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SCIFACT_",
        env_file=".env",
        env_ignore_empty=True,
        env_nested_delimiter="__",
        extra="ignore",
    )

    qdrant_path: Path = DEFAULT_QDRANT_PATH
    qdrant_url: HttpUrl | None = None
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = Field(default="scifact", min_length=1)
    qdrant_timeout: int = Field(default=60, ge=1)

    dense_model: str = Field(default=DEFAULT_DENSE_MODEL, min_length=1)
    dense_revision: str | None = None
    dense_device: str = "cpu"
    embedding_batch_size: int = Field(default=32, ge=1)
    index_batch_size: int = Field(default=64, ge=1)

    rerank_model: str = Field(default=DEFAULT_RERANK_MODEL, min_length=1)
    rerank_revision: str | None = None
    rerank_device: str = "cpu"
    rerank_max_length: int = Field(default=256, ge=2)
    rerank_batch_size: int = Field(default=16, ge=1)
    torch_threads: int = Field(default=4, ge=1)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    search: SearchOptions = Field(default_factory=SearchOptions)

    @property
    def resolved_dense_revision(self) -> str | None:
        if self.dense_model == DEFAULT_DENSE_MODEL:
            return self.dense_revision or DEFAULT_DENSE_REVISION
        return self.dense_revision

    @property
    def resolved_rerank_revision(self) -> str | None:
        if self.rerank_model == DEFAULT_RERANK_MODEL:
            return self.rerank_revision or DEFAULT_RERANK_REVISION
        return self.rerank_revision
