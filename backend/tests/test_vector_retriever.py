"""VectorRetriever round-trip tests with a deterministic fake embedder.

These verify storage, metadata fidelity, and persistence — not embedding
quality, which is the real model's job.
"""

from __future__ import annotations

import math
from pathlib import Path

from app.models import Chunk, make_chunk_id
from app.retrieval.vector import VectorRetriever

_VOCAB = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]


class BagOfWordsEmbedder:
    """Maps text to normalized counts over a tiny fixed vocabulary."""

    def _embed(self, text: str) -> list[float]:
        words = text.lower().split()
        vec = [float(words.count(w)) for w in _VOCAB]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def _chunk(text: str, index: int, page: int = 1) -> Chunk:
    return Chunk(
        id=make_chunk_id("doc.pdf", index, text),
        text=text,
        source_file="doc.pdf",
        page_number=page,
        chunk_index=index,
    )


def test_query_returns_best_match_with_metadata(tmp_path: Path):
    retriever = VectorRetriever(BagOfWordsEmbedder(), persist_dir=tmp_path)
    retriever.add(
        [
            _chunk("alpha alpha alpha", 0, page=1),
            _chunk("beta beta beta", 1, page=2),
            _chunk("gamma gamma gamma", 2, page=3),
        ]
    )
    results = retriever.query("beta", k=2)
    assert results[0].text == "beta beta beta"
    assert results[0].page_number == 2
    assert results[0].source_file == "doc.pdf"
    assert results[0].score > results[1].score


def test_reingesting_same_chunks_does_not_duplicate(tmp_path: Path):
    retriever = VectorRetriever(BagOfWordsEmbedder(), persist_dir=tmp_path)
    chunks = [_chunk("alpha beta", 0), _chunk("gamma delta", 1)]
    retriever.add(chunks)
    retriever.add(chunks)
    assert retriever.count() == 2


def test_store_persists_across_instances(tmp_path: Path):
    first = VectorRetriever(BagOfWordsEmbedder(), persist_dir=tmp_path)
    first.add([_chunk("epsilon zeta", 0)])
    del first

    second = VectorRetriever(BagOfWordsEmbedder(), persist_dir=tmp_path)
    assert second.count() == 1
    results = second.query("epsilon", k=1)
    assert results[0].text == "epsilon zeta"


def test_query_empty_store_returns_empty_list(tmp_path: Path):
    retriever = VectorRetriever(BagOfWordsEmbedder(), persist_dir=tmp_path)
    assert retriever.query("anything", k=3) == []
