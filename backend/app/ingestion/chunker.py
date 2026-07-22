"""Token-aware recursive chunker.

Chunks never span page boundaries — a chunk's ``page_number`` must be exact
for citations, so pages are the outermost split unit. Within a page, text is
split recursively (paragraphs, then sentences, then hard token splits) and
packed greedily up to the token budget, with token-level overlap between
consecutive chunks.
"""

from __future__ import annotations

import re
from typing import Protocol

from app.config import settings
from app.models import Chunk, make_chunk_id

from .parsers import ParsedDocument

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_BOUNDARY = re.compile(r"\n\s*\n")


class TokenCodec(Protocol):
    """Minimal tokenizer interface so the chunker is testable without
    downloading a model."""

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


class HFTokenCodec:
    """Wraps the embedding model's own tokenizer so chunk budgets line up
    with what the embedder actually sees."""

    def __init__(self, model_name: str = settings.embedding_model) -> None:
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=True)


class FastTokenCodec:
    """Same tokenizer via the lightweight ``tokenizers`` library — for slim
    deploys where ``transformers``/torch is not installed."""

    def __init__(self, model_name: str = settings.embedding_model) -> None:
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_pretrained(model_name)

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=True)


def build_codec() -> TokenCodec:
    """Codec matching the active embedder stack (see EMBEDDER_KIND)."""
    import os

    if os.getenv("EMBEDDER_KIND", "sentence_transformers").lower() == "fastembed":
        return FastTokenCodec()
    return HFTokenCodec()


def _split_into_units(text: str, codec: TokenCodec, max_tokens: int) -> list[str]:
    """Recursively split text into units that each fit the token budget."""
    units: list[str] = []
    for paragraph in _PARAGRAPH_BOUNDARY.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(codec.encode(paragraph)) <= max_tokens:
            units.append(paragraph)
            continue
        for sentence in _SENTENCE_BOUNDARY.split(paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            ids = codec.encode(sentence)
            if len(ids) <= max_tokens:
                units.append(sentence)
            else:
                for start in range(0, len(ids), max_tokens):
                    units.append(codec.decode(ids[start : start + max_tokens]))
    return units


def chunk_page_text(
    text: str,
    codec: TokenCodec,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    units = _split_into_units(text, codec, max_tokens)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = len(codec.encode(unit))
        if current and current_tokens + unit_tokens > max_tokens:
            chunk_text = "\n".join(current)
            chunks.append(chunk_text)
            if overlap_tokens > 0:
                tail_ids = codec.encode(chunk_text)[-overlap_tokens:]
                tail = codec.decode(tail_ids).strip()
                current = [tail] if tail else []
                current_tokens = len(tail_ids) if tail else 0
            else:
                current = []
                current_tokens = 0
        current.append(unit)
        current_tokens += unit_tokens

    if current:
        chunks.append("\n".join(current))
    return chunks


def chunk_document(
    doc: ParsedDocument,
    codec: TokenCodec,
    max_tokens: int = settings.chunk_size_tokens,
    overlap_tokens: int = settings.chunk_overlap_tokens,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    chunk_index = 0
    for page in doc.pages:
        for text in chunk_page_text(page.text, codec, max_tokens, overlap_tokens):
            chunks.append(
                Chunk(
                    id=make_chunk_id(doc.source_file, chunk_index, text),
                    text=text,
                    source_file=doc.source_file,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1
    return chunks
