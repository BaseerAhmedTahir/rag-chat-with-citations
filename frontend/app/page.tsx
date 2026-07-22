"use client";

import { useEffect, useRef, useState } from "react";
import { AnswerView } from "@/components/AnswerView";
import {
  askQuestion,
  getHealth,
  ingestFile,
  type Health,
  type QueryResponse,
} from "@/lib/api";

const SAMPLE_QUESTIONS = [
  "How many vacation days do employees get, and can they carry them over?",
  "What is the maximum payload and top speed of the Atlas P-100?",
  "What was Northwind's Q3 2025 revenue and how much did it grow?",
  "By how much did throughput improve in the Rotterdam pilot?",
];

export default function Home() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [asking, setAsking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [uploadMsg, setUploadMsg] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const refreshHealth = () => {
    getHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
  };
  useEffect(refreshHealth, []);

  async function onAsk(q: string) {
    const query = q.trim();
    if (!query || asking) return;
    setAsking(true);
    setError(null);
    setResult(null);
    try {
      setResult(await askQuestion(query));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed");
    } finally {
      setAsking(false);
    }
  }

  async function onUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setUploadMsg(null);
    setError(null);
    try {
      const r = await ingestFile(file);
      setUploadMsg(`Added ${r.source_file}: ${r.pages} pages → ${r.chunks} chunks`);
      refreshHealth();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  return (
    <main className="container">
      <header className="site-header">
        <h1>
          RAG Document Chat <span className="accent">with citations</span>
        </h1>
        <p className="subtitle">
          Every answer cites its exact source — file name and page. Click a
          number to see the passage that grounded the claim.
        </p>
        <div className="status">
          {health ? (
            <>
              <span className="dot ok" /> {health.chunks_indexed} chunks indexed ·{" "}
              {health.embedding_model} · {health.llm_provider}
            </>
          ) : (
            <>
              <span className="dot down" /> backend unreachable — set
              NEXT_PUBLIC_API_URL
            </>
          )}
        </div>
      </header>

      <section className="panel">
        <label className="upload">
          <input
            ref={fileRef}
            type="file"
            accept=".pdf,.docx"
            onChange={onUpload}
            disabled={uploading}
          />
          <span>{uploading ? "Uploading…" : "Upload a PDF or DOCX"}</span>
        </label>
        {uploadMsg && <p className="upload-msg">{uploadMsg}</p>}
      </section>

      <section className="panel">
        <div className="ask-row">
          <input
            className="ask-input"
            value={question}
            placeholder="Ask a question about the documents…"
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && onAsk(question)}
            disabled={asking}
          />
          <button className="ask-btn" onClick={() => onAsk(question)} disabled={asking}>
            {asking ? "Thinking…" : "Ask"}
          </button>
        </div>
        <div className="samples">
          {SAMPLE_QUESTIONS.map((q) => (
            <button
              key={q}
              className="sample"
              onClick={() => {
                setQuestion(q);
                onAsk(q);
              }}
              disabled={asking}
            >
              {q}
            </button>
          ))}
        </div>
      </section>

      {error && <div className="error">{error}</div>}

      {result &&
        (result.found ? (
          <AnswerView answer={result.answer} citations={result.citations} />
        ) : (
          <div className="notfound">{result.answer}</div>
        ))}

      <footer className="site-footer">
        Retrieval:{" "}
        {health?.retriever_kind === "hybrid_rerank"
          ? "hybrid (BM25 + dense) → cross-encoder rerank"
          : health?.retriever_kind === "hybrid"
            ? "hybrid (BM25 + dense, RRF)"
            : "dense vector"}{" "}
        · Answers grounded in retrieved context only.
      </footer>
    </main>
  );
}
