"""Question -> retrieve -> generate -> verifiable citations.

Retrieved chunks are numbered [1..k] in the prompt; the model must ground
every claim in those markers. Markers are parsed back out of the answer and
resolved to structured citations (file + page + snippet), so every citation
in the final result maps to a real retrieved passage — never to model memory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import settings
from app.models import RetrievedChunk
from app.retrieval.base import Retriever

from .providers import ChatProvider

NOT_FOUND_MESSAGE = "I could not find this in the provided documents."

SYSTEM_PROMPT = f"""\
You are a precise assistant that answers questions using ONLY the numbered
context passages provided. Follow these rules strictly:

1. Use only facts stated in the context passages. Never use outside knowledge.
2. Cite the passage number inline after every factual claim, like [1] or [2][3].
3. Quote numbers, dates, and names exactly as they appear in the passages.
4. If the passages do not contain the information needed to answer, reply with
   exactly: "{NOT_FOUND_MESSAGE}" and nothing else.
"""

_MARKER_PATTERN = re.compile(r"\[(\d+)\]")
_SNIPPET_CHARS = 240


@dataclass(frozen=True)
class Citation:
    marker: int  # the [n] used in the answer text
    chunk_id: str
    source_file: str
    page_number: int
    snippet: str


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    citations: list[Citation]
    retrieved: list[RetrievedChunk]  # full retrieval context, for debugging/evals

    @property
    def found(self) -> bool:
        return NOT_FOUND_MESSAGE not in self.answer


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    blocks = [
        f"[{i}] (source: {c.source_file}, page {c.page_number})\n{c.text}"
        for i, c in enumerate(chunks, start=1)
    ]
    context = "\n\n".join(blocks)
    return f"Context passages:\n\n{context}\n\nQuestion: {question}"


def extract_citations(answer: str, chunks: list[RetrievedChunk]) -> list[Citation]:
    """Resolve [n] markers to their chunks, in first-appearance order.

    Markers outside [1..k] are dropped — a citation must point at a passage
    that was actually in the prompt.
    """
    citations: list[Citation] = []
    seen: set[int] = set()
    for match in _MARKER_PATTERN.finditer(answer):
        marker = int(match.group(1))
        if marker in seen or not (1 <= marker <= len(chunks)):
            continue
        seen.add(marker)
        chunk = chunks[marker - 1]
        snippet = chunk.text[:_SNIPPET_CHARS] + (
            " …" if len(chunk.text) > _SNIPPET_CHARS else ""
        )
        citations.append(
            Citation(
                marker=marker,
                chunk_id=chunk.id,
                source_file=chunk.source_file,
                page_number=chunk.page_number,
                snippet=snippet,
            )
        )
    return citations


def answer_question(
    question: str,
    retriever: Retriever,
    provider: ChatProvider,
    k: int = settings.top_k,
) -> AnswerResult:
    chunks = retriever.query(question, k=k)
    if not chunks:
        return AnswerResult(answer=NOT_FOUND_MESSAGE, citations=[], retrieved=[])

    answer = provider.complete(SYSTEM_PROMPT, build_user_prompt(question, chunks))
    citations = extract_citations(answer, chunks)
    return AnswerResult(answer=answer, citations=citations, retrieved=chunks)
