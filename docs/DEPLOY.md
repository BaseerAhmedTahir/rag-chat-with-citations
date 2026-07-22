# Deployment guide

Backend → **Render** (free tier, no card). Frontend → **Vercel** (free).

> Why not HF Spaces / Railway? Railway's free tier became a 30-day trial, and
> Hugging Face moved Docker/Gradio Spaces behind PRO (only static Spaces are
> free). Render's free tier (750 instance-hours/month, no card) fits the slim
> ONNX build of this backend (~300 MB measured).

## 1. Backend on Render

The repo contains a [render.yaml](../render.yaml) Blueprint that configures
everything (slim stack: `EMBEDDER_KIND=fastembed`, `RETRIEVER_KIND=hybrid`).

1. Sign up at https://render.com (GitHub login, no card).
2. **New → Blueprint** → select the `rag-chat-with-citations` repo → Apply.
3. When prompted for env vars, set **`GROQ_API_KEY`** (the only one not in the
   Blueprint).
4. First deploy takes a few minutes. Verify:
   `https://<service-name>.onrender.com/health`
   → should return `"chunks_indexed": 20, "retriever_kind": "hybrid"`.

*(Alternative without Blueprint: New → Web Service → repo, Root Directory
`backend`, build `pip install -r requirements-slim.txt`, start
`uvicorn app.api.main:app --host 0.0.0.0 --port $PORT`, plus the env vars
from render.yaml.)*

## 2. Frontend on Vercel

1. Import the GitHub repo at https://vercel.com/new
2. **Root Directory:** `frontend` (Vercel auto-detects Next.js)
3. **Environment variable:** `NEXT_PUBLIC_API_URL = https://<service-name>.onrender.com`
   (no trailing slash)
4. Deploy.

## 3. Verify end to end

- Open the Vercel URL → header shows `20 chunks indexed · … · groq`, footer
  shows `hybrid (BM25 + dense, RRF)`.
- Ask a sample question → answer renders with clickable `[n]` citations.
- Click a citation → the source passage (file + page) highlights.

## Notes

- **Cold starts:** the free service spins down after 15 min idle; the next
  request takes ~1 min (restart + model download + re-seed). Later requests
  are fast.
- **Persistence:** the filesystem is ephemeral — sample PDFs re-ingest on
  every start; uploaded documents live only until the next spin-down.
- **Slim vs full:** the live demo runs `hybrid` retrieval (recall@1 0.846).
  The M4-winning `hybrid_rerank` (0.885) needs ~1.5 GB and runs locally or
  via `docker compose up` — the README states this tradeoff.
- **Cost:** $0. 750 free instance-hours/month is ~a full month for one
  spin-down-enabled service.
