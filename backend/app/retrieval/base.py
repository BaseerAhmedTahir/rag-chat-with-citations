"""The Retriever interface — swappable seam #1.

Everything downstream (generation, evals, API) depends only on this
protocol, never on a concrete implementation.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.models import Chunk, RetrievedChunk


@runtime_checkable
class Retriever(Protocol):
    def add(self, chunks: list[Chunk]) -> None: ...
    def query(self, question: str, k: int) -> list[RetrievedChunk]: ...
