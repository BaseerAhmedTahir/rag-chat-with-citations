# RAG Document Chat — with Verifiable Citations & an Evaluation Harness

[![CI](https://github.com/BaseerAhmedTahir/rag-chat-with-citations/actions/workflows/ci.yml/badge.svg)](https://github.com/BaseerAhmedTahir/rag-chat-with-citations/actions/workflows/ci.yml)

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

Three swappable seams, each a small interface the rest of the system programs
against (details in `backend/app/`):

```
                        ┌──────────────────────────────┐
  PDF / DOCX  ─ingest─▶ │  parse (page nums) → chunk    │
                        │  → Embedder → Retriever store │
                        └──────────────┬───────────────┘
                                       │
  question ──────────────────────────────────────────────────────────▶ answer
                        ┌───────────────────────────────┐             + citations
                        │  Retriever.query(k)            │              (file+page)
                        │  → ChatProvider.complete()     │
                        │  → parse [n] markers → cite    │
                        └───────────────────────────────┘

  Seam 1  Retriever     VectorRetriever · HybridRetriever (BM25+dense, RRF)
                        · RerankingRetriever (cross-encoder)
  Seam 2  ChatProvider  Gemini · Groq · OpenAI   (select via LLM_PROVIDER)
  Seam 3  Embedder      sentence-transformers  BAAI/bge-small-en-v1.5  (CPU)
```

Every chunk carries `source_file` + `page_number` + `chunk_index` from
ingestion onward — the metadata contract that makes citations verifiable.
The API resolves each `[n]` marker back to the exact passage shown to the
model, so a citation can never point at model memory.

<!-- Live demo GIF/screenshot added after deploy. -->
<!-- LIVE_LINK -->

## Quick start (Docker)

```bash
GROQ_API_KEY=... docker compose up --build
# frontend → http://localhost:3000   ·   API → http://localhost:8000/docs
```

The backend auto-ingests four sample PDFs on startup, so the demo works
immediately; upload your own via the UI or `POST /ingest`.

## Evaluation

Measured on 26 hand-labeled examples over 4 documents. Retrieval metrics
(recall@k, MRR) are implemented from scratch and unit-tested; answer metrics
are Ragas (LLM-as-judge, all means scored 26/26). Full tables, judge models,
and interpretation: [backend/evals/results](backend/evals/results/README.md).
The write-up [_How I measured and improved a RAG system's faithfulness_](docs/how-i-measured-and-improved-rag-faithfulness.md)
walks through the methodology.

**M4 result — hybrid retrieval + cross-encoder reranking vs the dense baseline:**

| config | recall@1 | mrr@10 | faithfulness | context precision |
|:--|--:|--:|--:|--:|
| dense-only (baseline) | 0.808 | 0.884 | 1.000 | 0.865 |
| hybrid + reranker | **0.885** | **0.929** | 1.000 | **0.914** |

recall@1 improved **+7.7 points**, MRR **+4.6 points**, and context precision
**+4.8 points**, while faithfulness held at 1.00 (every answer statement
entailed by its retrieved context). Every judged mean is backed by per-sample
scores at full 26/26 coverage — partially-scored runs are discarded, not
reported.

## Status

- [x] M1 — Ingestion, local embeddings, vector store, CLI query
- [x] M2 — Generation with verifiable citations
- [x] M3 — Evaluation harness (recall@k, MRR, Ragas)
- [x] M4 — Hybrid retrieval + reranking, measured improvement
- [x] M5 — FastAPI + Next.js frontend, Docker, CI · _deploy pending live URL_

## Local setup (without Docker)

**Backend**

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Unix: source .venv/bin/activate
pip install -r requirements.txt          # add -r requirements-eval.txt for evals
cp .env.example .env                      # set your LLM API key (M2+)
uvicorn app.api.main:app --reload         # API at http://localhost:8000
```

**Frontend**

```bash
cd frontend
npm install
cp .env.example .env.local                # NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev                               # UI at http://localhost:3000
```

### CLI (no server)

```bash
python cli.py ingest path/to/document.pdf
python cli.py query "What does the document say about X?"   # raw retrieval
python cli.py ask "What does the document say about X?"     # cited answer
```

`ask` prints the answer followed by resolved citations — each `[n]` marker
maps to the source file, page number, and the exact passage that grounded it.
Questions the documents can't answer get an explicit "not found" response
instead of a guess.

### Evaluation harness

```bash
pip install -r requirements-eval.txt
python evals/run_grid.py                  # deterministic recall@k / MRR grid
python evals/run_evals.py --label baseline  # + Ragas judged metrics
```

## Deployment

- **Backend** → Hugging Face Spaces (Docker SDK, port 7860); models baked into
  the image. See [backend/README.md](backend/README.md) for the Space card.
- **Frontend** → Vercel; set `NEXT_PUBLIC_API_URL` to the Space URL.
- Vector store stays ChromaDB (embedded); sample docs auto-ingest on startup.

Step-by-step instructions: [docs/DEPLOY.md](docs/DEPLOY.md).

## Tests & CI

`pytest` covers the chunker, parsers, retriever, citation assembly, retrieval
metrics, and API routes (37 tests). GitHub Actions runs ruff + pytest
(backend) and eslint + build (frontend) on every push.
