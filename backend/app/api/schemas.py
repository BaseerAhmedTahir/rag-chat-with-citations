"""Pydantic request/response models for the HTTP API.

These mirror the internal dataclasses but are the stable wire contract the
frontend depends on — keep them explicit rather than serializing internals.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=20)


class CitationOut(BaseModel):
    marker: int
    source_file: str
    page_number: int
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    found: bool
    citations: list[CitationOut]


class IngestResponse(BaseModel):
    source_file: str
    pages: int
    chunks: int
    total_chunks: int


class HealthResponse(BaseModel):
    status: str
    chunks_indexed: int
    embedding_model: str
    llm_provider: str
    retriever_kind: str
