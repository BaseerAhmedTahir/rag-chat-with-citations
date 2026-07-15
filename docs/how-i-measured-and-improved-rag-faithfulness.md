# How I measured and improved a RAG system's faithfulness

Most RAG demos stop at "it answers questions." This project treats answer
quality as a measurement problem: every claim must cite its exact source
(file + page), and every design change must prove itself on a labeled eval
set before it ships. This post walks through the methodology and what the
numbers actually showed.

## The setup

The pipeline is deliberately boring: PyMuPDF parsing that preserves page
numbers, token-aware chunking that never crosses a page boundary (a chunk
that spans pages cannot be cited precisely), `bge-small-en-v1.5` embeddings
on CPU, ChromaDB, and a hosted LLM behind a provider interface. The prompt
numbers each retrieved passage `[1..k]`, restricts the model to the provided
context, and requires inline citation markers. Markers are parsed back out
and resolved against the passages that were actually in the prompt — a
citation can point at a retrieved passage or nothing; it can never point at
model memory.

## The eval harness

Two layers, deliberately separated:

**Deterministic retrieval metrics** (own code, unit-tested): recall@k and
MRR at page granularity over 26 hand-labeled questions across 4 documents.
Page granularity matters because it matches what a citation exposes to the
user — retrieving the right document but the wrong page is a failed
citation.

**LLM-judged answer metrics** (Ragas): faithfulness, answer relevancy,
context precision, context recall. Faithfulness is the headline: the share
of answer statements entailed by the retrieved context. A faithfulness
failure is a hallucination that a citation would falsely legitimize — the
worst failure mode a cited system can have.

### What free-tier reality taught me

The judged metrics needed ~130 LLM calls per configuration, and free-tier
quotas made that genuinely hard: Gemini's free tier allows ~20
requests/day/project, and Groq caps tokens/day *per model*. Two fixes made
the harness practical:

1. **Per-answer caching** — generated answers are cached to disk keyed by
   config + model, so interrupted runs resume without re-spending quota.
2. **One judge model per metric** — each Ragas metric runs on its own Groq
   model, sideways through independent daily token buckets. Comparisons stay
   valid because each metric column keeps a consistent judge across configs.
   Not all metrics cost the same: `context_precision` issues one judge call
   *per retrieved context* (~130 per run, 5× the others), so it needs the
   model with the largest request budget, not the smartest one.

### The lesson that mattered most: report coverage, or report nothing

A judge that silently fails on 18 of 26 samples still returns a
plausible-looking average. That is the most dangerous failure mode in this
whole project, because it produces a number that looks like evidence.

I hit it for real. An early `context_precision` run reported **1.000** — a
perfect score — computed from **8 of 26 samples**; the other 18 had died on
rate limits and been quietly dropped from the mean. The fix was structural,
not clerical:

- every judged mean prints `N/26 samples scored` and writes per-sample
  scores to a committed `*_ragas_detail.csv`;
- the table builder **refuses to emit** a metric below full coverage;
- runs are gated in a retry loop until they reach 26/26.

Re-judging everything under that rule was worth it. Faithfulness, answer
relevancy, and context recall reproduced their original values exactly —
but context precision, the metric that had been quietly under-covered,
changed. The numbers I almost shipped were the wrong ones.

## The experiment

Baseline (M1-era): dense-only retrieval, chunk size 500/overlap 50, top-5.
Upgrades tested: **hybrid retrieval** (BM25 + dense, fused with reciprocal
rank fusion) and **cross-encoder reranking** (`bge-reranker-base`,
top-20 → top-5), across chunk sizes {300, 500, 800}.

### Result 1: chunk size did nothing — and that's a real finding

Every page in the eval corpus is shorter than even the 300-token budget, so
all three chunk sizes produced identical one-chunk-per-page indexes. The
lesson: chunking parameters only matter relative to your documents' actual
text density. Sweeping them without checking that is cargo-cult tuning.

### Result 2: the retriever axis is decisive

| config | recall@1 | recall@3 | MRR@10 | faithfulness | answer relevancy | context precision | context recall |
|:--|--:|--:|--:|--:|--:|--:|--:|
| dense-only baseline | 0.808 | 0.962 | 0.884 | **1.000** | 0.881 | 0.865 | 0.724 |
| hybrid (BM25+RRF) | 0.846 | **1.000** | 0.923 | — | — | — | — |
| hybrid + reranker | **0.885** | 0.962 | **0.929** | **1.000** | 0.885 | **0.914** | 0.678 |

- **recall@1: 0.808 → 0.885** (+7.7 points). One in five baseline questions
  ranked the correct page below the top result; hybrid+rerank cut that
  down. For a citation UI this is the metric users feel — the first citation
  is the one they click.
- **MRR: 0.884 → 0.929.** The correct page moved up the ranking across the
  board, not just at rank 1.
- **Context precision: 0.865 → 0.914.** Reranked top-5 contexts carry less
  padding, which is also a token-cost win.
- **Faithfulness held at 1.000** through the retrieval change — the
  grounding prompt and citation verification carry that property, and the
  eval confirms the upgrade didn't disturb it.
- One honest nuance: the reranker demoted a correct page out of the top-3
  on one question (hybrid alone had recall@3 = 1.0). Rerankers are not a
  free lunch at every cutoff; pick the cutoff your UI actually uses.
- **Context recall went down (0.724 → 0.678) and I'm not claiming a
  regression.** That metric is judged by the smallest model and returned
  0.704 and 0.678 on two separate fully-covered runs over identical inputs.
  The delta is inside the noise band, so the honest reading is "no signal,"
  not "reranking hurt recall."

## What I'd tell someone building this

1. Put page numbers in chunk metadata on day one. Citations are a data
   contract, not a UI feature.
2. Build the eval harness before the retrieval upgrades. Without the
   baseline table, the reranker's +7.7 recall@1 points would have been a
   vibe, not a result.
3. Report judge coverage or don't report judged metrics. A mean over an
   unknown denominator isn't a measurement.
4. Treat perfect scores with suspicion — including your own. My one
   suspicious 1.000 turned out to be an artifact of 8/26 coverage.
   Faithfulness 1.0 is real here, but it also partly reflects a clean,
   well-matched eval corpus; on messier documents context recall is the
   number that will move first, which is exactly why it stays in the table.
5. Know which of your metrics are noise. Context recall swung 0.704 → 0.678
   across two fully-covered runs on identical inputs. Any "improvement"
   smaller than a metric's run-to-run variance is not an improvement, and
   saying so is more useful than a table of green arrows.
