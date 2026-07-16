"""Deterministic retrieval-metric grid: chunk size x retriever kind.

Runs entirely on CPU with no API calls. The top-k axis of the M4 grid
only affects generation (which contexts reach the LLM), not these
metrics — recall@{1,3,5} and MRR@10 are all computed from each
retriever's ranked top-10, so k variants would produce identical rows.

Usage (from backend/):
    python evals/run_grid.py
"""

from __future__ import annotations

import sys
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS_DIR.parent))

from evals.run_evals import (  # noqa: E402
    RESULTS_DIR,
    build_index,
    eval_retrieval,
    load_eval_set,
)

CHUNK_SIZES = (300, 500, 800)
RETRIEVERS = ("vector", "hybrid", "hybrid_rerank")
OVERLAP = 50


def main() -> None:
    import pandas as pd

    examples = load_eval_set()
    print(f"Loaded {len(examples)} eval examples")

    rows = []
    for chunk_size in CHUNK_SIZES:
        for kind in RETRIEVERS:
            print(f"Config: chunk_size={chunk_size}, retriever={kind} …")
            retriever = build_index(
                chunk_size, OVERLAP, f"eval_cs{chunk_size}_ov{OVERLAP}", kind
            )
            summary, _ = eval_retrieval(retriever, examples)
            rows.append({"chunk_size": chunk_size, "retriever": kind, **summary})
            print(f"  {summary}")

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "m4_retrieval_grid.csv", index=False)
    (RESULTS_DIR / "m4_retrieval_grid.md").write_text(
        "# M4 retrieval grid — deterministic metrics\n\n"
        f"{df.to_markdown(index=False)}\n",
        encoding="utf-8",
    )
    print("\n" + df.to_markdown(index=False))
    print("\nWrote m4_retrieval_grid.csv / .md")


if __name__ == "__main__":
    main()
