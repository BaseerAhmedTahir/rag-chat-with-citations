"""Hybrid RRF fusion and reranking tests with injected fakes — no models."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from app.models import Chunk, RetrievedChunk, make_chunk_id
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.rerank import RerankingRetriever
from app.retrieval.vector import VectorRetriever
from tests.test_vector_retriever import BagOfWordsEmbedder


def _chunk(text: str, index: int, page: int = 1) -> Chunk:
    return Chunk(
        id=make_chunk_id("doc.pdf", index, text),
        text=text,
        source_file="doc.pdf",
        page_number=page,
        chunk_index=index,
    )


class StaticRetriever:
    """Returns a fixed ranking regardless of the question."""

    def __init__(self, ranking: list[RetrievedChunk]) -> None:
        self._ranking = ranking

    def add(self, chunks: list[Chunk]) -> None:
        pass

    def query(self, question: str, k: int) -> list[RetrievedChunk]:
        return self._ranking[:k]


def _retrieved(text: str, index: int, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        id=make_chunk_id("doc.pdf", index, text),
        text=text,
        source_file="doc.pdf",
        page_number=1,
        chunk_index=index,
        score=score,
    )


def test_hybrid_finds_keyword_match_dense_misses(tmp_path: Path):
    """A term outside the fake embedder's vocab is invisible to dense
    search but trivially found by BM25 — hybrid must surface it."""
    vector = VectorRetriever(BagOfWordsEmbedder(), persist_dir=tmp_path)
    hybrid = HybridRetriever(vector, candidate_pool=10)
    hybrid.add(
        [
            _chunk("alpha beta gamma", 0, page=1),
            _chunk("the serial number is XJ-4400", 1, page=2),
            _chunk("delta epsilon zeta", 2, page=3),
        ]
    )
    results = hybrid.query("XJ-4400", k=2)
    assert results, "hybrid returned nothing"
    assert results[0].page_number == 2


def test_hybrid_rrf_prefers_items_ranked_by_both(tmp_path: Path):
    vector = VectorRetriever(BagOfWordsEmbedder(), persist_dir=tmp_path)
    hybrid = HybridRetriever(vector, candidate_pool=10)
    hybrid.add(
        [
            _chunk("alpha beta", 0, page=1),
            _chunk("alpha gamma", 1, page=2),
            _chunk("unrelated words here", 2, page=3),
        ]
    )
    results = hybrid.query("alpha", k=3)
    pages = [r.page_number for r in results]
    # both alpha chunks rank above the unrelated one
    assert set(pages[:2]) == {1, 2}


def test_reranker_reorders_by_scorer():
    candidates = [
        _retrieved("weak match", 0, score=0.9),
        _retrieved("strong match", 1, score=0.5),
    ]

    def scorer(question: str, passages: Sequence[str]) -> Sequence[float]:
        return [1.0 if "strong" in p else 0.0 for p in passages]

    reranker = RerankingRetriever(StaticRetriever(candidates), scorer=scorer)
    results = reranker.query("anything", k=2)
    assert [r.text for r in results] == ["strong match", "weak match"]
    assert results[0].score == 1.0


def test_reranker_truncates_to_k():
    candidates = [_retrieved(f"passage {i}", i, score=1.0 - i / 10) for i in range(8)]

    def scorer(question: str, passages: Sequence[str]) -> Sequence[float]:
        return list(range(len(passages)))  # reverse the order

    reranker = RerankingRetriever(StaticRetriever(candidates), scorer=scorer, fetch_k=8)
    results = reranker.query("q", k=3)
    assert len(results) == 3
    assert results[0].text == "passage 7"


def test_reranker_empty_inner_returns_empty():
    reranker = RerankingRetriever(StaticRetriever([]), scorer=lambda q, p: [])
    assert reranker.query("q", k=5) == []
