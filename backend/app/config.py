"""Application settings, overridable via environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("RAG_DATA_DIR", BACKEND_DIR / "data"))
    )
    chunk_size_tokens: int = int(os.getenv("CHUNK_SIZE_TOKENS", "500"))
    chunk_overlap_tokens: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))
    top_k: int = int(os.getenv("TOP_K", "5"))

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"


settings = Settings()
