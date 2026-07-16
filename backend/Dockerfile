# Backend image for Hugging Face Spaces (Docker SDK).
# HF Spaces runs the container as UID 1000 and expects the app on port 7860.
FROM python:3.12-slim

# System deps: PyMuPDF and onnxruntime (chromadb) need libgomp.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user (HF Spaces requirement) with a writable home for model caches.
RUN useradd -m -u 1000 appuser
ENV HOME=/home/appuser \
    HF_HOME=/home/appuser/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    RAG_DATA_DIR=/home/appuser/data \
    PORT=7860

WORKDIR /app

COPY --chown=appuser:appuser requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Bake the CPU models into the image so cold starts don't re-download ~1GB
# (HF Spaces has no durable disk). Runs as the same user that serves.
COPY --chown=appuser:appuser app ./app
USER appuser
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
CrossEncoder('BAAI/bge-reranker-base')"

# Seed documents for the always-available demo corpus.
COPY --chown=appuser:appuser evals/dataset/documents ./evals/dataset/documents

EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
