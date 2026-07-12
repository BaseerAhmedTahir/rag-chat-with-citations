"""Core data contracts shared across ingestion, retrieval, and generation.

The ``Chunk`` metadata contract is the backbone of citations: every chunk
stored in the vector DB must carry its source file and page number, or
citations become impossible downstream.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source_file: str  # original filename (basename, not full path)
    page_number: int  # 1-indexed page (or section index for docx)
    chunk_index: int  # order within the document, 0-indexed


@dataclass(frozen=True)
class RetrievedChunk(Chunk):
    score: float


def make_chunk_id(source_file: str, chunk_index: int, text: str) -> str:
    """Deterministic id so re-ingesting the same document upserts in place."""
    digest = hashlib.sha1(
        f"{source_file}|{chunk_index}|{text}".encode("utf-8")
    ).hexdigest()
    return digest[:16]
