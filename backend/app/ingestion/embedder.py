"""Local CPU embeddings behind a small interface (swappable seam #3)."""

from __future__ import annotations

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
