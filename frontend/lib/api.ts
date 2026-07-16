// Typed client for the RAG backend. Mirrors app/api/schemas.py.

const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(
  /\/$/,
  "",
);

export interface Citation {
  marker: number;
  source_file: string;
  page_number: number;
  snippet: string;
}

export interface QueryResponse {
  answer: string;
  found: boolean;
  citations: Citation[];
}

export interface IngestResponse {
  source_file: string;
  pages: number;
  chunks: number;
  total_chunks: number;
}

export interface Health {
  status: string;
  chunks_indexed: number;
  embedding_model: string;
  llm_provider: string;
}

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export async function getHealth(): Promise<Health> {
  return handle<Health>(await fetch(`${API_URL}/health`, { cache: "no-store" }));
}

export async function askQuestion(question: string, k = 5): Promise<QueryResponse> {
  const res = await fetch(`${API_URL}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, k }),
  });
  return handle<QueryResponse>(res);
}

export async function ingestFile(file: File): Promise<IngestResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_URL}/ingest`, { method: "POST", body: form });
  return handle<IngestResponse>(res);
}
