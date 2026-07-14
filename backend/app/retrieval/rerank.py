"""Cross-encoder reranking wrapper around any Retriever.

Retrieves a wide candidate pool from the inner retriever, then rescores
each (question, chunk) pair with a cross-encoder, which reads both texts
jointly and is far more precise than bi-encoder similarity — at the cost
of one forward pass per candidate, which is why it only sees the top-N.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from app.models import Chunk, RetrievedChunk
from app.retrieval.base import Retriever

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"

# (question, passages) -> one relevance score per passage
PairScorer = Callable[[str, Sequence[str]], Sequence[float]]


class CrossEncoderScorer:
    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name, device="cpu")

    def __call__(self, question: str, passages: Sequence[str]) -> Sequence[float]:
        scores = self._model.predict(
            [(question, passage) for passage in passages],
            show_progress_bar=False,
        )
        return [float(s) for s in scores]


class RerankingRetriever:
    def __init__(
        self,
        inner: Retriever,
        scorer: PairScorer | None = None,
        fetch_k: int = 20,
    ) -> None:
        self._inner = inner
        self._scorer = scorer  # lazily constructed: the model download is heavy
        self._fetch_k = fetch_k

    def add(self, chunks: list[Chunk]) -> None:
        self._inner.add(chunks)

    def query(self, question: str, k: int) -> list[RetrievedChunk]:
        candidates = self._inner.query(question, k=max(self._fetch_k, k))
        if not candidates:
            return []
        if self._scorer is None:
            self._scorer = CrossEncoderScorer()

        scores = self._scorer(question, [c.text for c in candidates])
        rescored = [
            RetrievedChunk(
                id=c.id,
                text=c.text,
                source_file=c.source_file,
                page_number=c.page_number,
                chunk_index=c.chunk_index,
                score=score,
            )
            for c, score in zip(candidates, scores)
        ]
        rescored.sort(key=lambda c: c.score, reverse=True)
        return rescored[:k]
