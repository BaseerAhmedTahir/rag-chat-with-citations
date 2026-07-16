---
title: RAG Document Chat API
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# RAG Document Chat — Backend API

FastAPI backend for the [RAG Document Chat](https://github.com/BaseerAhmedTahir/rag-chat-with-citations)
project: retrieval-augmented generation with verifiable citations (file + page)
and a measured evaluation harness. Runs entirely on CPU.

## Endpoints

| method | path | purpose |
|:--|:--|:--|
| `GET`  | `/health` | status + number of chunks indexed |
| `POST` | `/ingest` | upload a PDF/DOCX (multipart `file`) |
| `POST` | `/query`  | `{question, k}` → answer + structured citations |

Interactive docs at `/docs`.

## Configuration (env vars)

| var | default | notes |
|:--|:--|:--|
| `LLM_PROVIDER` | `gemini` | `gemini` \| `groq` \| `openai` |
| `GROQ_API_KEY` / `GEMINI_API_KEY` / `OPENAI_API_KEY` | — | key for the chosen provider |
| `RETRIEVER_KIND` | `hybrid_rerank` | `vector` \| `hybrid` \| `hybrid_rerank` |
| `CORS_ORIGINS` | `*` | comma-separated allowed origins |

On startup the four sample PDFs are auto-ingested so the demo works
immediately (Spaces has no durable disk). Set the provider key in the Space's
**Settings → Variables and secrets** before querying.
