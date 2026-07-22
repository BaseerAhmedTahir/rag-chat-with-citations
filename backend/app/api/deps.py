"""Shared, lazily-built singletons for the API.

The embedder, retriever, and reranker models are expensive to construct, so
they are built once on first use and reused across requests. The retriever
kind is env-configurable so the deployed demo can trade quality for cold-start
speed without code changes.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from app.config import BACKEND_DIR
from app.ingestion.chunker import TokenCodec, build_codec, chunk_document
from app.ingestion.embedder import build_embedder
from app.ingestion.parsers import SUPPORTED_EXTENSIONS, parse_document
from app.retrieval.base import Retriever
from app.retrieval.vector import VectorRetriever

RETRIEVER_KIND = os.getenv("RETRIEVER_KIND", "hybrid_rerank")
# On startup, ingest any documents sitting here so the demo works immediately
# (HF Spaces has no durable disk). Anchored to the backend dir, not the data
# dir, so relocating RAG_DATA_DIR does not break seeding.
SEED_DOCS_DIR = Path(
    os.getenv("SEED_DOCS_DIR", BACKEND_DIR / "evals" / "dataset" / "documents")
)


class AppState:
    """Holds the process-wide retriever and codec, built once."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._retriever: Retriever | None = None
        self._codec: TokenCodec | None = None

    def _build_retriever(self, vector: VectorRetriever) -> Retriever:
        if RETRIEVER_KIND == "vector":
            return vector
        if RETRIEVER_KIND == "hybrid":
            from app.retrieval.hybrid import HybridRetriever

            return HybridRetriever(vector)
        if RETRIEVER_KIND == "hybrid_rerank":
            from app.retrieval.hybrid import HybridRetriever
            from app.retrieval.rerank import RerankingRetriever

            return RerankingRetriever(HybridRetriever(vector))
        raise ValueError(f"Unknown RETRIEVER_KIND: {RETRIEVER_KIND!r}")

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            with self._lock:
                if self._retriever is None:
                    vector = VectorRetriever(build_embedder())
                    self._retriever = self._build_retriever(vector)
                    self._codec = build_codec()
        return self._retriever

    @property
    def codec(self) -> TokenCodec:
        if self._codec is None:
            _ = self.retriever  # triggers construction
        assert self._codec is not None
        return self._codec

    def ingest_file(self, path: Path) -> tuple[int, int]:
        """Parse, chunk, embed and store one document. Returns (pages, chunks)."""
        doc = parse_document(path)
        chunks = chunk_document(doc, self.codec)
        self.retriever.add(chunks)
        return len(doc.pages), len(chunks)

    def seed_from_disk(self) -> int:
        """Ingest seed documents if the store is empty. Returns files ingested."""
        vector = self._vector()
        if vector.count() > 0:
            return 0
        if not SEED_DOCS_DIR.is_dir():
            return 0
        ingested = 0
        for path in sorted(SEED_DOCS_DIR.iterdir()):
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                self.ingest_file(path)
                ingested += 1
        return ingested

    def _vector(self) -> VectorRetriever:
        """Reach the underlying VectorRetriever regardless of wrapping."""
        r = self.retriever
        seen = set()
        while not isinstance(r, VectorRetriever):
            if id(r) in seen:
                raise TypeError("no VectorRetriever found in retriever chain")
            seen.add(id(r))
            r = getattr(r, "_inner", None) or getattr(r, "_vector", None)
            if r is None:
                raise TypeError("no VectorRetriever found in retriever chain")
        return r

    def chunk_count(self) -> int:
        return self._vector().count()


state = AppState()
