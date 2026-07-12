"""Chunker tests using a whitespace tokenizer so no model download is needed."""

from __future__ import annotations

from app.ingestion.chunker import chunk_document, chunk_page_text
from app.ingestion.parsers import PageSpan, ParsedDocument


class WhitespaceCodec:
    """1 word == 1 token. Encodes to indices into a shared vocab list."""

    def __init__(self) -> None:
        self._vocab: list[str] = []
        self._ids: dict[str, int] = {}

    def encode(self, text: str) -> list[int]:
        ids = []
        for word in text.split():
            if word not in self._ids:
                self._ids[word] = len(self._vocab)
                self._vocab.append(word)
            ids.append(self._ids[word])
        return ids

    def decode(self, ids: list[int]) -> str:
        return " ".join(self._vocab[i] for i in ids)


def _make_doc(pages: list[str]) -> ParsedDocument:
    return ParsedDocument(
        source_file="test.pdf",
        pages=[PageSpan(text=t, page_number=i) for i, t in enumerate(pages, start=1)],
    )


def test_short_page_is_single_chunk():
    codec = WhitespaceCodec()
    chunks = chunk_page_text("one two three.", codec, max_tokens=10, overlap_tokens=2)
    assert chunks == ["one two three."]


def test_long_text_is_split_within_budget():
    codec = WhitespaceCodec()
    text = " ".join(f"word{i}" for i in range(50))
    chunks = chunk_page_text(text, codec, max_tokens=20, overlap_tokens=5)
    assert len(chunks) > 1
    for chunk in chunks:
        # hard-split units are capped at max_tokens; packed chunks stay near it
        assert len(codec.encode(chunk)) <= 25  # max_tokens + overlap slack


def test_consecutive_chunks_overlap():
    codec = WhitespaceCodec()
    text = " ".join(f"word{i}" for i in range(60))
    chunks = chunk_page_text(text, codec, max_tokens=20, overlap_tokens=5)
    first_words = chunks[0].split()
    second_words = chunks[1].replace("\n", " ").split()
    assert first_words[-5:] == second_words[:5]


def test_chunks_never_span_pages_and_keep_metadata():
    codec = WhitespaceCodec()
    doc = _make_doc(["alpha beta gamma.", "delta epsilon zeta."])
    chunks = chunk_document(doc, codec, max_tokens=100, overlap_tokens=10)
    assert [c.page_number for c in chunks] == [1, 2]
    assert all(c.source_file == "test.pdf" for c in chunks)
    assert [c.chunk_index for c in chunks] == [0, 1]


def test_chunk_ids_are_stable_across_runs():
    codec = WhitespaceCodec()
    doc = _make_doc(["alpha beta gamma."])
    first = chunk_document(doc, codec)
    second = chunk_document(doc, codec)
    assert [c.id for c in first] == [c.id for c in second]
