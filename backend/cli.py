"""Terminal interface: ingest documents and query the vector store.

Usage:
    python cli.py ingest <path>          # a .pdf/.docx file or a directory
    python cli.py query "<question>" [--k 5]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from app.config import settings
from app.ingestion.chunker import HFTokenCodec, chunk_document
from app.ingestion.embedder import SentenceTransformerEmbedder
from app.ingestion.parsers import SUPPORTED_EXTENSIONS, parse_document
from app.retrieval.vector import VectorRetriever


def _build_retriever() -> VectorRetriever:
    return VectorRetriever(embedder=SentenceTransformerEmbedder())


def _collect_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(
            p
            for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    raise FileNotFoundError(f"No such file or directory: {path}")


def cmd_ingest(path_arg: str) -> None:
    files = _collect_files(Path(path_arg))
    if not files:
        print(f"No supported documents ({sorted(SUPPORTED_EXTENSIONS)}) found.")
        sys.exit(1)

    codec = HFTokenCodec()
    retriever = _build_retriever()

    for file in files:
        start = time.perf_counter()
        doc = parse_document(file)
        chunks = chunk_document(doc, codec)
        retriever.add(chunks)
        elapsed = time.perf_counter() - start
        print(
            f"Ingested {doc.source_file}: {len(doc.pages)} pages -> "
            f"{len(chunks)} chunks in {elapsed:.1f}s"
        )
    print(
        f"Vector store now holds {retriever.count()} chunks "
        f"(persisted at {settings.chroma_dir})."
    )


def cmd_query(question: str, k: int) -> None:
    retriever = _build_retriever()
    results = retriever.query(question, k=k)
    if not results:
        print("No results — is the vector store empty? Run `ingest` first.")
        sys.exit(1)

    for rank, chunk in enumerate(results, start=1):
        print(
            f"\n[{rank}] {chunk.source_file} — page {chunk.page_number} "
            f"(score {chunk.score:.4f})"
        )
        print("-" * 72)
        text = chunk.text if len(chunk.text) <= 500 else chunk.text[:500] + " …"
        print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG document chat CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Parse, chunk, embed and store documents")
    p_ingest.add_argument("path", help="A .pdf/.docx file or a directory of them")

    p_query = sub.add_parser("query", help="Retrieve the top-k chunks for a question")
    p_query.add_argument("question")
    p_query.add_argument("--k", type=int, default=settings.top_k)

    args = parser.parse_args()
    if args.command == "ingest":
        cmd_ingest(args.path)
    elif args.command == "query":
        cmd_query(args.question, args.k)


if __name__ == "__main__":
    main()
