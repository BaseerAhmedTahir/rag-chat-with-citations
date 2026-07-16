"""FastAPI application: ingest documents, ask questions, get cited answers."""

from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.api.deps import state
from app.api.schemas import (
    CitationOut,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from app.config import settings
from app.generation.providers import ProviderError, get_provider
from app.generation.rag import answer_question
from app.ingestion.parsers import SUPPORTED_EXTENSIONS

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the models and seed the demo corpus so the first request is fast
    # and the deployed app is never empty.
    if os.getenv("SEED_ON_STARTUP", "1") == "1":
        state.seed_from_disk()
    yield


app = FastAPI(title="RAG Document Chat", version="1.0.0", lifespan=lifespan)

_origins = os.getenv("CORS_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"]
    if _origins == "*"
    else [o.strip() for o in _origins.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        chunks_indexed=state.chunk_count(),
        embedding_model=settings.embedding_model,
        llm_provider=os.getenv("LLM_PROVIDER", "gemini"),
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)) -> IngestResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type {suffix!r}; "
            f"supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 20 MB limit.")
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")

    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / (file.filename or f"upload{suffix}")
    tmp_path.write_bytes(data)
    try:
        pages, chunks = state.ingest_file(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)
        tmp_dir.rmdir()

    return IngestResponse(
        source_file=tmp_path.name,
        pages=pages,
        chunks=chunks,
        total_chunks=state.chunk_count(),
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    try:
        provider = get_provider()
    except ProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    result = answer_question(req.question, state.retriever, provider, k=req.k)
    return QueryResponse(
        answer=result.answer,
        found=result.found,
        citations=[
            CitationOut(
                marker=c.marker,
                source_file=c.source_file,
                page_number=c.page_number,
                snippet=c.snippet,
            )
            for c in result.citations
        ],
    )
