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

<!-- Architecture diagram, screenshot, and live link added as milestones complete. -->

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
python cli.py query "What does the document say about X?"   # raw retrieval
python cli.py ask "What does the document say about X?"     # cited answer (M2)
```

`ask` prints the answer followed by resolved citations — each `[n]` marker
maps to the source file, page number, and the exact passage that grounded it.
Questions the documents can't answer get an explicit "not found" response
instead of a guess.
