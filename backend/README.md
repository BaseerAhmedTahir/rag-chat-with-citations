# RAG Document Chat — Backend API

FastAPI backend for the [RAG Document Chat](https://github.com/BaseerAhmedTahir/rag-chat-with-citations)
project: retrieval-augmented generation with verifiable citations (file + page)
and a measured evaluation harness. Runs entirely on CPU.

## Endpoints

| method | path | purpose |
|:--|:--|:--|
| `GET`  | `/health` | status, chunks indexed, active retriever/embedder |
| `POST` | `/ingest` | upload a PDF/DOCX (multipart `file`) |
| `POST` | `/query`  | `{question, k}` → answer + structured citations |

Interactive docs at `/docs`.

## Two runtime stacks, one codebase

| | full (default) | slim (free hosting) |
|:--|:--|:--|
| requirements | `requirements.txt` | `requirements-slim.txt` |
| embedder | `bge-small-en-v1.5` fp32 (torch) | same model, int8 ONNX (fastembed) |
| retriever | `hybrid_rerank` (M4 winner) | `hybrid` (no cross-encoder) |
| memory | ~1.5 GB | ~300 MB |
| select via | — | `EMBEDDER_KIND=fastembed`, `RETRIEVER_KIND=hybrid` |

All eval results in `evals/results/` were produced with the full stack.

## Configuration (env vars)

| var | default | notes |
|:--|:--|:--|
| `LLM_PROVIDER` | `gemini` | `gemini` \| `groq` \| `openai` |
| `GROQ_API_KEY` / `GEMINI_API_KEY` / `OPENAI_API_KEY` | — | key for the chosen provider |
| `RETRIEVER_KIND` | `hybrid_rerank` | `vector` \| `hybrid` \| `hybrid_rerank` |
| `EMBEDDER_KIND` | `sentence_transformers` | `fastembed` for the slim stack |
| `CORS_ORIGINS` | `*` | comma-separated allowed origins |
| `SEED_ON_STARTUP` | `1` | auto-ingest the 4 sample PDFs when the store is empty |

On startup the sample PDFs are auto-ingested so the demo works immediately on
hosts without durable disks; uploads live for the session.
