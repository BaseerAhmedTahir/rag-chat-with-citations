"""Tests for the slim (fastembed/ONNX) deploy stack.

These skip when the optional deps aren't installed (CI installs only core
requirements), but run locally where the full dev environment has both
stacks — which is exactly where equivalence is worth checking.
"""

from __future__ import annotations

import pytest

from app.ingestion.chunker import build_codec
from app.ingestion.embedder import build_embedder

fastembed = pytest.importorskip("fastembed")


def test_build_embedder_selects_fastembed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EMBEDDER_KIND", "fastembed")
    embedder = build_embedder()
    assert type(embedder).__name__ == "FastembedEmbedder"


def test_build_embedder_rejects_unknown_kind(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EMBEDDER_KIND", "bogus")
    with pytest.raises(ValueError, match="Unknown EMBEDDER_KIND"):
        build_embedder()


def test_fastembed_vectors_are_normalized_and_ranked_sensibly(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("EMBEDDER_KIND", "fastembed")
    embedder = build_embedder()
    docs = embedder.embed_documents(
        ["The battery lasts 8 hours.", "The office is in Tallinn."]
    )
    query = embedder.embed_query("How long does the battery last?")

    assert len(query) == 384
    norm = sum(v * v for v in query) ** 0.5
    assert abs(norm - 1.0) < 1e-3

    sims = [sum(q * d for q, d in zip(query, doc)) for doc in docs]
    assert sims[0] > sims[1], "battery doc should outrank office doc"


def test_fast_codec_roundtrips_and_counts_like_hf(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("transformers")
    text = "The Atlas P-100 carries payloads of up to 15 kg."

    monkeypatch.setenv("EMBEDDER_KIND", "fastembed")
    fast = build_codec()
    monkeypatch.setenv("EMBEDDER_KIND", "sentence_transformers")
    hf = build_codec()

    fast_ids = fast.encode(text)
    assert fast_ids == hf.encode(text), "token budgets must match across stacks"
    assert "atlas p - 100" in fast.decode(fast_ids).lower()
