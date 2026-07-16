"""API tests with a fake retriever + provider injected — no models, no network."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from app.models import Chunk, RetrievedChunk, make_chunk_id


class FakeRetriever:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk]) -> None:
        self._chunks.extend(chunks)

    def query(self, question: str, k: int) -> list[RetrievedChunk]:
        text = "The Atlas P-100 carries payloads of up to 15 kg."
        return [
            RetrievedChunk(
                id=make_chunk_id("atlas_spec.pdf", 0, text),
                text=text,
                source_file="atlas_spec.pdf",
                page_number=1,
                chunk_index=0,
                score=0.91,
            )
        ]

    def count(self) -> int:
        return len(self._chunks) or 1


class FakeProvider:
    def complete(self, system: str, user: str) -> str:
        return "The Atlas P-100 carries up to 15 kg [1]."


@pytest.fixture()
def client(monkeypatch):
    # Patch the app state before importing main so no real models load.
    from app.api import deps

    fake = FakeRetriever()

    class FakeState:
        retriever = fake

        def ingest_file(self, path):
            self.retriever.add([_dummy_chunk(path.name)])
            return 1, 1

        def seed_from_disk(self):
            return 0

        def chunk_count(self):
            return self.retriever.count()

    monkeypatch.setattr(deps, "state", FakeState())

    import app.api.main as main

    monkeypatch.setattr(main, "state", deps.state)
    monkeypatch.setattr(main, "get_provider", lambda: FakeProvider())
    monkeypatch.setenv("SEED_ON_STARTUP", "0")
    return TestClient(main.app)


def _dummy_chunk(name: str) -> Chunk:
    return Chunk(
        id=make_chunk_id(name, 0, name),
        text=name,
        source_file=name,
        page_number=1,
        chunk_index=0,
    )


def test_health_ok(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["chunks_indexed"] >= 1


def test_query_returns_answer_with_citation(client: TestClient):
    resp = client.post("/query", json={"question": "payload?", "k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["found"] is True
    assert len(body["citations"]) == 1
    citation = body["citations"][0]
    assert citation["source_file"] == "atlas_spec.pdf"
    assert citation["page_number"] == 1
    assert citation["marker"] == 1


def test_query_validation_rejects_empty_question(client: TestClient):
    resp = client.post("/query", json={"question": "", "k": 3})
    assert resp.status_code == 422


def test_query_rejects_out_of_range_k(client: TestClient):
    resp = client.post("/query", json={"question": "x", "k": 99})
    assert resp.status_code == 422


def test_ingest_rejects_unsupported_type(client: TestClient):
    resp = client.post(
        "/ingest",
        files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert resp.status_code == 400


def test_ingest_rejects_empty_file(client: TestClient):
    resp = client.post(
        "/ingest",
        files={"file": ("doc.pdf", io.BytesIO(b""), "application/pdf")},
    )
    assert resp.status_code == 400
