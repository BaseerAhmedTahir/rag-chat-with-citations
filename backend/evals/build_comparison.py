"""Assemble M3 baseline + M4 comparison tables from detail-backed metrics.

Retrieval metrics come from the deterministic grid (exact). Judged metrics
are read from each config's committed ``*_ragas_detail.csv`` so every number
is backed by per-sample scores and verified at full 26/26 coverage — a
metric below full coverage raises rather than ships a misleading mean.
"""

from pathlib import Path

import pandas as pd

RES = Path("evals/results")
N_EXPECTED = 26

# judged metrics: label suffix -> (detail column substring, output column)
JUDGED = {
    "f": ("faithfulness", "faithfulness"),
    "ar": ("answer_relevancy", "answer_relevancy"),
    "cp": ("precision", "context_precision"),
    "cr": ("recall", "context_recall"),
}


def judged_values(prefix: str) -> dict[str, float]:
    """Mean of each judged metric, refusing anything under full coverage.

    Also merges the per-metric detail files into one audit table per config.
    """
    out: dict[str, float] = {}
    merged: pd.DataFrame | None = None
    for suffix, (needle, col) in JUDGED.items():
        detail = RES / f"{prefix}_{suffix}_ragas_detail.csv"
        if not detail.exists():
            raise SystemExit(f"missing detail file: {detail}")
        df = pd.read_csv(detail)
        score_col = next(c for c in df.columns if needle in c)
        scored = int(df[score_col].notna().sum())
        if scored < N_EXPECTED:
            raise SystemExit(
                f"{detail.name}: only {scored}/{N_EXPECTED} scored — not final"
            )
        out[col] = float(df[score_col].mean())
        if merged is None:
            merged = df[["user_input", "response", "reference"]].copy()
        elif not merged["user_input"].equals(df["user_input"]):
            # rows are aligned positionally; a different order would
            # silently attach scores to the wrong question
            raise SystemExit(f"{detail.name}: question order differs from {prefix}_f")
        merged[col] = df[score_col].values

    assert merged is not None
    merged.to_csv(RES / f"{prefix}_ragas_detail.csv", index=False)
    return out


def retrieval_values(retriever: str) -> dict[str, float]:
    grid = pd.read_csv(RES / "m4_retrieval_grid.csv")
    row = grid[grid.retriever == retriever].iloc[0]
    return {k: float(row[k]) for k in ["recall@1", "recall@3", "recall@5", "mrr@10"]}


def summary_row(config: str, retriever: str, judged_prefix: str) -> dict:
    return {
        "config": config,
        **retrieval_values(retriever),
        **judged_values(judged_prefix),
    }


baseline = summary_row("baseline", "vector", "baseline")
winner = summary_row("hybrid_rerank", "hybrid_rerank", "rerank_best")

# rewrite canonical per-config summaries (CSV + MD)
for row, stem, title in [
    (baseline, "baseline", "baseline (M3)"),
    (winner, "rerank_best", "hybrid_rerank (M4 winner)"),
]:
    df = pd.DataFrame([row])
    df.to_csv(RES / f"{stem}_summary.csv", index=False)
    (RES / f"{stem}_summary.md").write_text(
        f"# Eval results — {title}\n\n{df.to_markdown(index=False)}\n",
        encoding="utf-8",
    )

# combined comparison, incl. the intermediate hybrid row (retrieval only)
hybrid = {"config": "hybrid (BM25+RRF)", **retrieval_values("hybrid")}
cols = [
    "config",
    "recall@1",
    "recall@3",
    "recall@5",
    "mrr@10",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]
comp = (
    pd.DataFrame(
        [
            {"config": "dense-only (baseline)", **{k: baseline[k] for k in cols[1:]}},
            hybrid,
            {"config": "hybrid + reranker", **{k: winner[k] for k in cols[1:]}},
        ]
    )
    .reindex(columns=cols)
    .round(4)
)
comp.to_csv(RES / "m4_comparison.csv", index=False)
md = comp.to_markdown(index=False).replace("nan", "  —")
note = (
    "\n\n_Judged metrics run for baseline and the winning config; the "
    'intermediate hybrid row shows retrieval metrics only ("—" = not '
    "evaluated). Every judged mean scored 26/26 samples (see "
    "`*_ragas_detail.csv`)._\n"
)
(RES / "m4_comparison.md").write_text(
    "# M4 configuration comparison\n\n" + md + note, encoding="utf-8"
)
print(md)
