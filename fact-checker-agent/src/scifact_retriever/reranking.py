"""Cross-encoder scoring of claim and paper text pairs."""

from math import isfinite
from typing import TYPE_CHECKING

from langchain_core.documents import Document

from .config import Settings
from .dataset import paper_prefix

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(self, model: "CrossEncoder", batch_size: int) -> None:
        self.model = model
        self.batch_size = batch_size

    @classmethod
    def from_settings(cls, settings: Settings) -> "CrossEncoderReranker":
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(
            settings.rerank_model,
            revision=settings.resolved_rerank_revision,
            device=settings.rerank_device,
            max_length=settings.rerank_max_length,
        )
        return cls(model, settings.rerank_batch_size)

    def rank(self, claim: str, documents: list[Document]) -> list[Document]:
        if not documents:
            return []
        scores = self.model.predict(
            [(claim, paper_prefix(doc.metadata["title"]) + doc.page_content) for doc in documents],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        ranked = []
        for doc, raw_score in zip(documents, scores, strict=True):
            score = float(raw_score)
            if not isfinite(score):
                raise ValueError("Cross-encoder produced a non-finite score")
            ranked.append(
                doc.model_copy(
                    update={
                        "metadata": {**doc.metadata, "rerank_score": score},
                    }
                )
            )
        return sorted(ranked, key=lambda doc: doc.metadata["rerank_score"], reverse=True)
