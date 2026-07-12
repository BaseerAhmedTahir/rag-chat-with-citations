"""Citation assembly tests with fake retriever/provider — no network calls."""

from __future__ import annotations

import pytest

from app.generation.providers import ProviderError, get_provider
from app.generation.rag import (
    NOT_FOUND_MESSAGE,
    answer_question,
    build_user_prompt,
    extract_citations,
)
from app.models import Chunk, RetrievedChunk, make_chunk_id


def _retrieved(text: str, index: int, page: int) -> RetrievedChunk:
    return RetrievedChunk(
        id=make_chunk_id("handbook.pdf", index, text),
        text=text,
        source_file="handbook.pdf",
        page_number=page,
        chunk_index=index,
        score=0.9 - index * 0.1,
    )


CHUNKS = [
    _retrieved("Employees get 27 vacation days.", 0, page=2),
    _retrieved("Hardware keys are mandatory.", 1, page=3),
]


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks

    def add(self, chunks: list[Chunk]) -> None:
        raise NotImplementedError

    def query(self, question: str, k: int) -> list[RetrievedChunk]:
        return self._chunks[:k]


class FakeProvider:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.last_user_prompt: str | None = None

    def complete(self, system: str, user: str) -> str:
        self.last_user_prompt = user
        return self.reply


def test_user_prompt_tags_chunks_with_source_and_page():
    prompt = build_user_prompt("How many vacation days?", CHUNKS)
    assert "[1] (source: handbook.pdf, page 2)" in prompt
    assert "[2] (source: handbook.pdf, page 3)" in prompt
    assert prompt.rstrip().endswith("Question: How many vacation days?")


def test_citations_resolve_to_correct_file_and_page():
    result = answer_question(
        "How many vacation days?",
        FakeRetriever(CHUNKS),
        FakeProvider("Employees receive 27 days of vacation [1]."),
        k=2,
    )
    assert result.found
    assert len(result.citations) == 1
    citation = result.citations[0]
    assert citation.marker == 1
    assert citation.source_file == "handbook.pdf"
    assert citation.page_number == 2
    assert "27 vacation days" in citation.snippet


def test_duplicate_and_out_of_range_markers_are_dropped():
    answer = "Keys are required [2][2], see also [7]."
    citations = extract_citations(answer, CHUNKS)
    assert [c.marker for c in citations] == [2]


def test_not_found_reply_yields_no_citations():
    result = answer_question(
        "What is the CEO's shoe size?",
        FakeRetriever(CHUNKS),
        FakeProvider(NOT_FOUND_MESSAGE),
        k=2,
    )
    assert not result.found
    assert result.citations == []


def test_empty_store_short_circuits_without_llm_call():
    class ExplodingProvider:
        def complete(self, system: str, user: str) -> str:
            raise AssertionError("provider must not be called")

    result = answer_question("anything", FakeRetriever([]), ExplodingProvider(), k=3)
    assert not result.found


def test_provider_registry_selects_by_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GROQ_API_KEY", "dummy")
    provider = get_provider("groq")
    assert type(provider).__name__ == "GroqProvider"

    monkeypatch.setenv("LLM_PROVIDER", "nope")
    with pytest.raises(ProviderError, match="Unknown LLM_PROVIDER"):
        get_provider()
