"""Dense vector retrieval over a persisted ChromaDB collection."""

from __future__ import annotations

from pathlib import Path

from app.config import settings
from app.ingestion.embedder import Embedder
from app.models import Chunk, RetrievedChunk

_ADD_BATCH_SIZE = 256


class VectorRetriever:
    def __init__(
        self,
        embedder: Embedder,
        persist_dir: str | Path = settings.chroma_dir,
        collection_name: str = "documents",
    ) -> None:
        import chromadb

        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = embedder

    def add(self, chunks: list[Chunk]) -> None:
        for start in range(0, len(chunks), _ADD_BATCH_SIZE):
            batch = chunks[start : start + _ADD_BATCH_SIZE]
            embeddings = self._embedder.embed_documents([c.text for c in batch])
            self._collection.upsert(
                ids=[c.id for c in batch],
                embeddings=embeddings,
                documents=[c.text for c in batch],
                metadatas=[
                    {
                        "source_file": c.source_file,
                        "page_number": c.page_number,
                        "chunk_index": c.chunk_index,
                    }
                    for c in batch
                ],
            )

    def query(self, question: str, k: int) -> list[RetrievedChunk]:
        if self.count() == 0:
            return []
        embedding = self._embedder.embed_query(question)
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        retrieved: list[RetrievedChunk] = []
        for chunk_id, text, meta, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            retrieved.append(
                RetrievedChunk(
                    id=chunk_id,
                    text=text,
                    source_file=str(meta["source_file"]),
                    page_number=int(meta["page_number"]),
                    chunk_index=int(meta["chunk_index"]),
                    score=1.0 - float(distance),  # cosine distance -> similarity
                )
            )
        return retrieved

    def count(self) -> int:
        return self._collection.count()
