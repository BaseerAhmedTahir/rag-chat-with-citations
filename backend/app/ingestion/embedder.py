"""Local CPU embeddings behind a small interface (swappable seam #3).

Two implementations of the same model (``bge-small-en-v1.5``):

- ``SentenceTransformerEmbedder`` — full fp32 via torch. Default for local
  dev, Docker, and all eval results.
- ``FastembedEmbedder`` — int8 ONNX via fastembed (~35 MB, no torch).
  Used on memory-constrained free hosting (Render, 512 MB).

Select with ``EMBEDDER_KIND`` (``sentence_transformers`` | ``fastembed``).
An index must be built and queried by the same embedder kind.
"""

from __future__ import annotations

import os
from typing import Protocol

from app.config import settings

# bge v1.5 models recommend prefixing retrieval *queries* (not passages)
# with this instruction for better relevance.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str = settings.embedding_model,
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)
        self._batch_size = batch_size
        self._is_bge = "bge" in model_name.lower()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        if self._is_bge:
            text = BGE_QUERY_INSTRUCTION + text
        embedding = self._model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embedding[0].tolist()


class FastembedEmbedder:
    """Quantized ONNX embeddings — no torch, fits small free-tier hosts.

    fastembed applies the BGE query instruction itself in ``query_embed``
    and returns normalized vectors, matching the SentenceTransformer
    implementation's conventions.
    """

    def __init__(self, model_name: str = settings.embedding_model) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


def build_embedder() -> Embedder:
    kind = os.getenv("EMBEDDER_KIND", "sentence_transformers").lower()
    if kind == "fastembed":
        return FastembedEmbedder()
    if kind == "sentence_transformers":
        return SentenceTransformerEmbedder()
    raise ValueError(f"Unknown EMBEDDER_KIND: {kind!r}")
