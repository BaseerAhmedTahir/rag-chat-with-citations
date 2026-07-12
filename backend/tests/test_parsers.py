"""Parser tests generate their own fixture documents — no binary fixtures in git."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion.parsers import parse_document, parse_pdf


@pytest.fixture()
def three_page_pdf(tmp_path: Path) -> Path:
    import fitz

    path = tmp_path / "fixture.pdf"
    doc = fitz.open()
    for i in range(1, 4):
        page = doc.new_page()
        page.insert_text((72, 72), f"This is unique content for page {i}.")
    doc.save(str(path))
    doc.close()
    return path


def test_pdf_pages_are_one_indexed_and_ordered(three_page_pdf: Path):
    parsed = parse_pdf(three_page_pdf)
    assert parsed.source_file == "fixture.pdf"
    assert [p.page_number for p in parsed.pages] == [1, 2, 3]


def test_pdf_page_text_matches_page_number(three_page_pdf: Path):
    parsed = parse_pdf(three_page_pdf)
    for page in parsed.pages:
        assert f"page {page.page_number}" in page.text


def test_unsupported_extension_raises(tmp_path: Path):
    bogus = tmp_path / "notes.txt"
    bogus.write_text("hello")
    with pytest.raises(ValueError, match="Unsupported file type"):
        parse_document(bogus)
