# RAG Document Chat — with Verifiable Citations & an Evaluation Harness

Chat with your documents and get answers where **every claim cites its exact source**
(file + page number), backed by a **measurable evaluation harness** (recall@k, MRR,
Ragas faithfulness) that proves retrieval quality instead of assuming it.

## Why this project is different

1. **Verifiable citations** — answers cite file name + page number; each citation maps
   back to the exact passage that grounded the claim.
2. **A real eval harness** — labeled dataset, deterministic retrieval metrics, and
   LLM-as-judge answer metrics, plus a config-comparison experiment with measured results.
3. **Swappable architecture** — embeddings, vector store, and LLM provider all sit behind
   small interfaces; any one can be replaced without touching the rest.

## Architecture

Three swappable seams (details in `backend/app/`):

- `Retriever` — vector (ChromaDB), hybrid (BM25 + dense), reranking (cross-encoder)
- `ChatProvider` — hosted LLM APIs, selected via `LLM_PROVIDER` env var
- `Embedder` — local CPU sentence-transformers (`BAAI/bge-small-en-v1.5`)

<!-- Architecture diagram, screenshot, eval results table, and live link added as milestones complete. -->

## Status

- [ ] M1 — Ingestion, local embeddings, vector store, CLI query
- [ ] M2 — Generation with verifiable citations
- [ ] M3 — Evaluation harness (recall@k, MRR, Ragas)
- [ ] M4 — Hybrid retrieval + reranking, measured improvement
- [ ] M5 — FastAPI + Next.js frontend, Docker, CI, deploy

## Setup

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Unix: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your LLM API key (needed from M2 on)
```

### CLI usage (M1)

```bash
python cli.py ingest path/to/document.pdf
python cli.py query "What does the document say about X?"
```
