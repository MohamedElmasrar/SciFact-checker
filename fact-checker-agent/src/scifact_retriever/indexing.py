"""Create, validate, and populate the Qdrant hybrid index."""

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import batched
from uuid import NAMESPACE_URL, uuid5

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode, SparseEmbeddings
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from .config import SPARSE_MODEL, Settings
from .dataset import CORPUS_REPO, CORPUS_REVISION, paper_prefix

logger = logging.getLogger(__name__)
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"
READY_COUNT = "scifact_indexed_count"


@dataclass(frozen=True)
class IndexResult:
    indexed: int
    skipped: int


def make_client(settings: Settings) -> QdrantClient:
    return QdrantClient(
        url=str(settings.qdrant_url),
        api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
        timeout=settings.qdrant_timeout,
    )


def make_dense_embeddings(settings: Settings) -> Embeddings:
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=settings.dense_model,
        model_kwargs={
            "device": settings.dense_device,
            "revision": settings.resolved_dense_revision,
        },
        encode_kwargs={"batch_size": settings.embedding_batch_size, "normalize_embeddings": True},
    )


def make_sparse_embeddings(settings: Settings) -> SparseEmbeddings:
    return FastEmbedSparse(model_name=SPARSE_MODEL, batch_size=settings.index_batch_size)


def index_signature(settings: Settings) -> dict[str, str | int | None]:
    return {
        "schema_version": 1,
        "corpus": CORPUS_REPO,
        "corpus_revision": CORPUS_REVISION,
        "dense_model": settings.dense_model,
        "dense_revision": settings.resolved_dense_revision,
        "sparse_model": SPARSE_MODEL,
    }


def point_id(doc_id: str) -> str:
    return str(
        uuid5(NAMESPACE_URL, f"https://huggingface.co/datasets/{CORPUS_REPO}/corpus/{doc_id}")
    )


def make_store(
    client: QdrantClient,
    settings: Settings,
    mode: RetrievalMode,
    dense: Embeddings | None,
    sparse: SparseEmbeddings | None,
) -> QdrantVectorStore:
    return QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        retrieval_mode=mode,
        embedding=dense,
        sparse_embedding=sparse,
        vector_name=DENSE_VECTOR,
        sparse_vector_name=SPARSE_VECTOR,
        validate_collection_config=False,
    )


def validate_collection(
    client: QdrantClient, settings: Settings, *, require_ready: bool = True
) -> int | None:
    name = settings.qdrant_collection
    try:
        info = client.get_collection(name)
    except UnexpectedResponse as exc:
        if exc.status_code == 404:
            raise ValueError(f"Collection {name!r} is missing. Run scifact-ingest.") from exc
        raise
    metadata = info.config.metadata or {}
    params = info.config.params
    dense = params.vectors.get(DENSE_VECTOR) if isinstance(params.vectors, dict) else None
    sparse = (params.sparse_vectors or {}).get(SPARSE_VECTOR)
    if (
        metadata.get("scifact_index") != index_signature(settings)
        or dense is None
        or dense.distance != models.Distance.COSINE
        or sparse is None
        or sparse.modifier != models.Modifier.IDF
    ):
        raise ValueError(
            f"Collection {name!r} is incompatible with these index settings. "
            "Choose a new SCIFACT_QDRANT_COLLECTION and run scifact-ingest."
        )
    count = metadata.get(READY_COUNT)
    if require_ready and (
        not isinstance(count, int) or count < 1 or client.count(name, exact=True).count != count
    ):
        raise ValueError(f"Collection {name!r} is incomplete or modified. Run scifact-ingest.")
    return count


def ensure_collection(client: QdrantClient, settings: Settings, dense: Embeddings) -> None:
    name = settings.qdrant_collection
    if not client.collection_exists(name):
        dimension = len(dense.embed_query("scientific evidence"))
        try:
            client.create_collection(
                name,
                vectors_config={
                    DENSE_VECTOR: models.VectorParams(
                        size=dimension, distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
                metadata={"scifact_index": index_signature(settings)},
            )
        except UnexpectedResponse as exc:
            if exc.status_code not in {400, 409} or not client.collection_exists(name):
                raise
    validate_collection(client, settings, require_ready=False)


def index_corpus(
    documents: Iterable[Document],
    client: QdrantClient,
    settings: Settings,
    dense: Embeddings,
    sparse: SparseEmbeddings,
) -> IndexResult:
    ensure_collection(client, settings, dense)
    name = settings.qdrant_collection
    store = make_store(client, settings, RetrievalMode.HYBRID, dense, sparse)
    indexed = skipped = 0
    seen: set[str] = set()

    for batch in batched(documents, settings.index_batch_size):
        ids = [point_id(doc.metadata["doc_id"]) for doc in batch]
        if seen.intersection(ids) or len(set(ids)) != len(ids):
            raise ValueError("Corpus contains duplicate document IDs")
        seen.update(ids)
        existing = {
            record.id
            for record in client.retrieve(name, ids, with_vectors=False, with_payload=False)
        }
        pending = [(pid, doc) for pid, doc in zip(ids, batch, strict=True) if pid not in existing]
        skipped += len(batch) - len(pending)
        if pending:
            store.add_texts(
                [paper_prefix(doc.metadata["title"]) + doc.page_content for _, doc in pending],
                metadatas=[doc.metadata for _, doc in pending],
                ids=[pid for pid, _ in pending],
                batch_size=settings.index_batch_size,
                wait=True,
            )
            indexed += len(pending)
            if indexed % (settings.index_batch_size * 10) == 0:
                logger.info("index_progress", extra={"fields": {"indexed": indexed}})
    if not seen or client.count(name, exact=True).count != len(seen):
        raise ValueError(
            f"Collection {name!r} does not match the corpus. Check the input and index."
        )
    client.update_collection(name, metadata={READY_COUNT: len(seen)})
    return IndexResult(indexed=indexed, skipped=skipped)
