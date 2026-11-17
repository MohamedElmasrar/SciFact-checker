"""LangChain retriever for dense, BM25, and Qdrant RRF search."""

from threading import Lock

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from pydantic import ConfigDict, PrivateAttr, model_validator
from qdrant_client import QdrantClient, models

from .config import SearchOptions, Settings, Strategy
from .dataset import paper_prefix
from .indexing import (
    make_client,
    make_dense_embeddings,
    make_sparse_embeddings,
    make_store,
    validate_collection,
)
from .reranking import CrossEncoderReranker


class SciFactRetriever(BaseRetriever):
    """Supports invoke, ainvoke, and LangChain retriever callbacks."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    store: QdrantVectorStore
    options: SearchOptions
    reranker: CrossEncoderReranker | None = None
    _search_lock = PrivateAttr(default_factory=Lock)

    @model_validator(mode="after")
    def validate_pipeline(self) -> "SciFactRetriever":
        if self.options.rerank and self.reranker is None:
            raise ValueError("Reranking is enabled but no cross-encoder was provided")
        if self.store.retrieval_mode.value != self.options.strategy.value:
            raise ValueError("The Qdrant store mode must match the retrieval strategy")
        return self

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        claim = query.strip()
        if not claim:
            raise ValueError("Scientific claim must be nonblank")
        # Tokenizer settings can change during encoding. Serialize use of each model instance.
        with self._search_lock:
            return self._search(claim)

    def _search(self, claim: str) -> list[Document]:
        options = self.options
        limit = options.rerank_candidates if options.rerank else options.top_k
        depth = options.hybrid_depth if options.strategy == Strategy.HYBRID else limit
        hits = self.store.similarity_search_with_score(
            claim,
            k=depth,
            search_params=models.SearchParams(exact=options.exact_dense_search),
            score_threshold=options.score_threshold,
        )
        documents = []
        for hit, score in hits[:limit]:
            metadata = {
                key: value
                for key, value in hit.metadata.items()
                if key not in {"_id", "_collection_name"}
            }
            metadata.update(retrieval_score=float(score), retrieval_strategy=options.strategy.value)
            documents.append(
                Document(
                    id=metadata["doc_id"],
                    page_content=hit.page_content.removeprefix(paper_prefix(metadata["title"])),
                    metadata=metadata,
                )
            )
        if self.reranker is not None and options.rerank:
            documents = self.reranker.rank(claim, documents)
        return documents[: options.top_k]


def build_retriever(
    settings: Settings | None = None, *, client: QdrantClient | None = None
) -> SciFactRetriever:
    """Build once and reuse. The caller may supply and manage a Qdrant client."""
    settings = settings if settings is not None else Settings()
    owns_client = client is None
    client = client if client is not None else make_client(settings)
    try:
        validate_collection(client, settings)
        options = settings.search
        dense = None if options.strategy == Strategy.SPARSE else make_dense_embeddings(settings)
        sparse = None if options.strategy == Strategy.DENSE else make_sparse_embeddings(settings)
        store = make_store(client, settings, RetrievalMode(options.strategy), dense, sparse)
        reranker = CrossEncoderReranker.from_settings(settings) if options.rerank else None
        return SciFactRetriever(store=store, options=options, reranker=reranker)
    except Exception:
        if owns_client:
            client.close()
        raise
