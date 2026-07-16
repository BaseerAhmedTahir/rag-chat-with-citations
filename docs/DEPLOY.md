# Deployment guide

Backend → **Hugging Face Spaces** (free, Docker). Frontend → **Vercel** (free).
Both free tiers; no card required.

## 1. Backend on Hugging Face Spaces

The backend is deployed as a Docker Space. The Space repo holds the **contents
of `backend/`** (its `Dockerfile` and `README.md` frontmatter configure it).

1. Create the Space: https://huggingface.co/new-space
   - **SDK:** Docker · **Template:** Blank · **Hardware:** CPU basic (free)
2. Push the backend into the Space repo. From the project root:

   ```bash
   # one-time: add the Space as a second remote for a backend-only subtree
   git remote add space https://huggingface.co/spaces/<user>/<space-name>
   git subtree push --prefix backend space main
   ```

   (Or clone the Space repo, copy `backend/*` into it, commit, and push.)
3. In the Space → **Settings → Variables and secrets**, add:
   - `LLM_PROVIDER = groq`
   - `GROQ_API_KEY = <your key>` (mark as **secret**)
   - optional: `CORS_ORIGINS = https://<your-vercel-app>.vercel.app`
4. The Space builds the image (bakes the models, ~5–8 min first time). When it
   shows **Running**, check `https://<user>-<space-name>.hf.space/health`.

> First build is slow because it downloads torch + the two models into the
> image. Later restarts are fast — the models are baked in.

## 2. Frontend on Vercel

1. Import the GitHub repo at https://vercel.com/new
2. **Root Directory:** `frontend`
3. **Environment variable:** `NEXT_PUBLIC_API_URL = https://<user>-<space-name>.hf.space`
   (no trailing slash)
4. Deploy. Vercel auto-detects Next.js.

## 3. Verify end to end

- Open the Vercel URL → the header should read `N chunks indexed · … · groq`.
- Ask a sample question → answer renders with clickable `[n]` citations.
- Click a citation → the source passage (file + page) highlights.

## Notes

- **Cold starts:** a free Space sleeps after ~48h idle and takes ~30–60s to
  wake (it reloads the reranker). The first request after sleep is slow.
- **Persistence:** the Space filesystem is ephemeral; the four sample PDFs are
  auto-ingested on every start. Uploaded documents live only for that session.
- **Cost:** $0. To reduce cold-start latency you can set `RETRIEVER_KIND=hybrid`
  (skips the reranker) at the cost of the M4 quality gain.
