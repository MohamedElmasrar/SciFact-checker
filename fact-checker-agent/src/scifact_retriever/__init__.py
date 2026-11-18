"""Scientific evidence retrieval over BEIR SciFact."""

from .config import SearchOptions, Settings, Strategy
from .retrieval import SciFactRetriever, build_retriever

__all__ = ["SciFactRetriever", "SearchOptions", "Settings", "Strategy", "build_retriever"]
