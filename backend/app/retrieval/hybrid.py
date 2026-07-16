"""Hybrid retrieval: dense vectors + BM25 keyword scoring, fused with RRF.

Reciprocal Rank Fusion (Cormack et al., 2009): each candidate's score is
the sum of 1/(rrf_k + rank) over the rankings it appears in. Rank-based
fusion sidesteps the incomparability of cosine similarities and BM25
scores, which live on entirely different scales.
"""

from __future__ import annotations

import re

from app.models import Chunk, RetrievedChunk
from app.retrieval.vector import VectorRetriever

_WORD = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class HybridRetriever:
    def __init__(
        self,
        vector: VectorRetriever,
        rrf_k: int = 60,
        candidate_pool: int = 20,
    ) -> None:
        self._vector = vector
        self._rrf_k = rrf_k
        self._pool = candidate_pool
        self._bm25 = None
        self._corpus: list[Chunk] = []

    def add(self, chunks: list[Chunk]) -> None:
        self._vector.add(chunks)
        self._bm25 = None  # corpus changed; rebuild lazily

    def _ensure_bm25(self) -> None:
        if self._bm25 is not None:
            return
        from rank_bm25 import BM25Okapi

        self._corpus = self._vector.all_chunks()
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in self._corpus])

    def _bm25_ranking(self, question: str) -> list[Chunk]:
        assert self._bm25 is not None
        scores = self._bm25.get_scores(_tokenize(question))
        # zero score means no term overlap at all — not a keyword match,
        # so it must not earn an RRF rank position
        order = sorted(
            (i for i in range(len(self._corpus)) if scores[i] > 0),
            key=lambda i: scores[i],
            reverse=True,
        )
        return [self._corpus[i] for i in order[: self._pool]]

    def query(self, question: str, k: int) -> list[RetrievedChunk]:
        self._ensure_bm25()
        if not self._corpus:
            return []

        dense = self._vector.query(question, k=self._pool)
        sparse = self._bm25_ranking(question)

        fused: dict[str, float] = {}
        by_id: dict[str, Chunk] = {}
        for ranking in (dense, sparse):
            for rank, chunk in enumerate(ranking, start=1):
                fused[chunk.id] = fused.get(chunk.id, 0.0) + 1.0 / (self._rrf_k + rank)
                by_id.setdefault(chunk.id, chunk)

        top = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:k]
        return [
            RetrievedChunk(
                id=chunk_id,
                text=by_id[chunk_id].text,
                source_file=by_id[chunk_id].source_file,
                page_number=by_id[chunk_id].page_number,
                chunk_index=by_id[chunk_id].chunk_index,
                score=score,
            )
            for chunk_id, score in top
        ]
