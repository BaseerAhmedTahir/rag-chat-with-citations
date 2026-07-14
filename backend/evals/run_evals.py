"""Run the full evaluation harness and write results tables.

Usage (from backend/):
    python evals/run_evals.py                 # everything
    python evals/run_evals.py --skip-ragas    # deterministic metrics only
    python evals/run_evals.py --label reranker_on

Outputs CSV + Markdown into evals/results/<label>_*.{csv,md}.

Deterministic retrieval metrics (recall@k, hit@k, MRR) run locally and free.
Ragas metrics (faithfulness, answer relevancy, context precision/recall) use
an LLM judge via the Gemini API — throttled for free-tier rate limits.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = EVALS_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings  # noqa: E402
from app.generation.providers import get_provider  # noqa: E402
from app.generation.rag import answer_question  # noqa: E402
from app.ingestion.chunker import HFTokenCodec, chunk_document  # noqa: E402
from app.ingestion.embedder import SentenceTransformerEmbedder  # noqa: E402
from app.ingestion.parsers import parse_document  # noqa: E402
from app.retrieval.base import Retriever  # noqa: E402
from app.retrieval.vector import VectorRetriever  # noqa: E402
from evals.metrics.retrieval import (  # noqa: E402
    hit_at_k,
    mean_metric,
    reciprocal_rank,
)

DOCS_DIR = EVALS_DIR / "dataset" / "documents"
EVAL_SET_PATH = EVALS_DIR / "dataset" / "eval_set.json"
RESULTS_DIR = EVALS_DIR / "results"
INDEX_DIR = EVALS_DIR / ".index"

HIT_KS = (1, 3, 5)
MRR_DEPTH = 10


@dataclass(frozen=True)
class Example:
    question: str
    ground_truth_answer: str
    relevant: set[tuple[str, int]]


def load_eval_set() -> list[Example]:
    data = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))
    examples = []
    for row in data["examples"]:
        src = row["relevant_source"]
        examples.append(
            Example(
                question=row["question"],
                ground_truth_answer=row["ground_truth_answer"],
                relevant={(src["file"], src["page"])},
            )
        )
    return examples


def build_index(chunk_size: int, overlap: int, collection: str) -> Retriever:
    codec = HFTokenCodec()
    embedder = SentenceTransformerEmbedder()
    retriever = VectorRetriever(
        embedder, persist_dir=INDEX_DIR, collection_name=collection
    )
    for pdf in sorted(DOCS_DIR.glob("*.pdf")):
        doc = parse_document(pdf)
        chunks = chunk_document(
            doc, codec, max_tokens=chunk_size, overlap_tokens=overlap
        )
        retriever.add(chunks)
    return retriever


def eval_retrieval(
    retriever: Retriever, examples: list[Example]
) -> tuple[dict[str, float], list[dict]]:
    per_question: list[dict] = []
    for ex in examples:
        results = retriever.query(ex.question, k=MRR_DEPTH)
        ranked = [(r.source_file, r.page_number) for r in results]
        row: dict = {"question": ex.question}
        for k in HIT_KS:
            row[f"hit@{k}"] = hit_at_k(ranked, ex.relevant, k)
        row["reciprocal_rank"] = reciprocal_rank(ranked, ex.relevant)
        per_question.append(row)

    summary = {
        f"recall@{k}": mean_metric([r[f"hit@{k}"] for r in per_question])
        for k in HIT_KS
    }
    summary[f"mrr@{MRR_DEPTH}"] = mean_metric(
        [r["reciprocal_rank"] for r in per_question]
    )
    return summary, per_question


def generate_answers(
    retriever: Retriever,
    examples: list[Example],
    k: int,
    sleep_s: float,
    cache_path: Path,
) -> list[dict]:
    """Generate answers, caching each one so an interrupted run (rate
    limits, daily quota) resumes without re-spending API quota."""
    provider = get_provider()
    cache: dict[str, dict] = {}
    if cache_path.exists():
        cache = {
            row["question"]: row
            for row in json.loads(cache_path.read_text(encoding="utf-8"))
        }
        print(f"  resuming: {len(cache)} answers already cached")

    rows: list[dict] = []
    for i, ex in enumerate(examples, start=1):
        if ex.question in cache:
            rows.append(cache[ex.question])
            continue
        result = answer_question(ex.question, retriever, provider, k=k)
        row = {
            "question": ex.question,
            "answer": result.answer,
            "reference": ex.ground_truth_answer,
            "contexts": [c.text for c in result.retrieved],
            "cited_pages": [
                f"{c.source_file}:{c.page_number}" for c in result.citations
            ],
        }
        rows.append(row)
        cache[ex.question] = row
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(list(cache.values()), ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        print(f"  [{i}/{len(examples)}] generated ({len(result.citations)} citations)")
        if i < len(examples):
            time.sleep(sleep_s)
    return rows


def eval_ragas(generation_rows: list[dict]) -> dict[str, float]:
    import os

    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )
    from ragas.run_config import RunConfig

    judge_model = os.getenv("RAGAS_JUDGE_MODEL", "gemini-2.5-flash-lite")
    judge = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model=judge_model,
            temperature=0.0,
            google_api_key=os.environ["GEMINI_API_KEY"],
            max_retries=6,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=settings.embedding_model)
    )

    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": row["question"],
                "response": row["answer"],
                "retrieved_contexts": row["contexts"],
                "reference": row["reference"],
            }
            for row in generation_rows
        ]
    )
    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ],
        llm=judge,
        embeddings=embeddings,
        run_config=RunConfig(max_workers=1, timeout=120),
    )
    df = result.to_pandas()
    metric_cols = [
        c
        for c in df.columns
        if c not in ("user_input", "response", "retrieved_contexts", "reference")
    ]
    return {col: float(df[col].mean()) for col in metric_cols}


def write_tables(
    label: str, summary: dict[str, float], per_question: list[dict]
) -> None:
    import pandas as pd

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame([summary])
    summary_df.insert(0, "config", label)
    csv_path = RESULTS_DIR / f"{label}_summary.csv"
    md_path = RESULTS_DIR / f"{label}_summary.md"
    summary_df.to_csv(csv_path, index=False)
    md_path.write_text(
        f"# Eval results — {label}\n\n{summary_df.to_markdown(index=False)}\n",
        encoding="utf-8",
    )
    pd.DataFrame(per_question).to_csv(
        RESULTS_DIR / f"{label}_per_question.csv", index=False
    )
    print(f"\nWrote {csv_path.name}, {md_path.name}, {label}_per_question.csv")
    print(summary_df.to_markdown(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG evaluation harness")
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size_tokens)
    parser.add_argument("--overlap", type=int, default=settings.chunk_overlap_tokens)
    parser.add_argument("--k", type=int, default=settings.top_k)
    parser.add_argument(
        "--sleep",
        type=float,
        default=13.0,
        help="seconds between generation calls (free-tier RPM)",
    )
    parser.add_argument("--skip-ragas", action="store_true")
    args = parser.parse_args()

    examples = load_eval_set()
    print(f"Loaded {len(examples)} eval examples")

    collection = f"eval_{args.label}_cs{args.chunk_size}"
    print(f"Building index (chunk_size={args.chunk_size}, overlap={args.overlap})…")
    retriever = build_index(args.chunk_size, args.overlap, collection)

    print("Running retrieval metrics…")
    summary, per_question = eval_retrieval(retriever, examples)

    if not args.skip_ragas:
        import os

        gen_model = os.getenv("GEMINI_MODEL", "default")
        cache_path = EVALS_DIR / ".cache" / f"gen_{args.label}_{gen_model}.json"
        print(
            f"Generating answers (k={args.k}, sleep={args.sleep}s, model={gen_model})…"
        )
        generation_rows = generate_answers(
            retriever, examples, args.k, args.sleep, cache_path
        )
        print("Running Ragas judge metrics…")
        ragas_summary = eval_ragas(generation_rows)
        summary.update(ragas_summary)
        for row, gen in zip(per_question, generation_rows):
            row["answer"] = gen["answer"]
            row["cited_pages"] = "; ".join(gen["cited_pages"])

    write_tables(args.label, summary, per_question)


if __name__ == "__main__":
    main()
