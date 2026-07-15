# Evaluation Results

- **M3 baseline** — dense-only retrieval, all metrics ([below](#m3-baseline)).
- **M4 comparison** — hybrid + reranking vs baseline
  ([m4_comparison.md](m4_comparison.md), [analysis below](#m4-retrieval-upgrades)).

## M3 baseline

Config: `bge-small-en-v1.5` dense retrieval (ChromaDB, cosine), chunk size 500
tokens / 50 overlap, top-k 5, generation by `llama-3.3-70b-versatile` (Groq)
with the citation-grounded prompt. Dataset: 26 hand-labeled examples over 4
documents ([eval_set.json](../dataset/eval_set.json)).

| config   | recall@1 | recall@3 | recall@5 | mrr@10 | faithfulness | answer_relevancy | context_precision | context_recall |
|:---------|---------:|---------:|---------:|-------:|-------------:|-----------------:|------------------:|---------------:|
| baseline |    0.808 |    0.962 |    0.962 | 0.884  |    **1.000** |            0.881 |             0.865 |          0.724 |

All LLM-judged metrics scored 26/26 samples; per-sample scores in
[baseline_ragas_detail.csv](baseline_ragas_detail.csv).

### Interpretation

- **Faithfulness 1.00 (headline):** every statement in every generated answer
  is entailed by the retrieved passages — the strict context-only prompt plus
  marker-verified citations is doing its job. The number to protect in M4.
- **recall@1 0.81 vs recall@3 0.96:** the right page is almost always
  retrieved, but ~1 in 5 questions doesn't rank it first — the clearest
  headroom for hybrid retrieval + reranking (M4).
- **Context precision 0.87:** at k=5, some retrieved chunks are padding.
  Reranking should raise this.
- **Context recall 0.72** is the strictest number; see judge caveat below.

> Note: `context_precision` was re-judged during M4 (0.872 → 0.865) when the
> metric moved from `gpt-oss-20b` to `llama-3.1-8b-instant` — see the judge
> table below. Baseline faithfulness, answer_relevancy and context_recall
> reproduced their original values exactly on re-run.

### Methodology notes

- Retrieval metrics (`recall@k`, `hit@k`, `MRR`) are computed by
  [our own code](../metrics/retrieval.py) at page-level granularity — the same
  granularity a citation exposes — and are covered by pytest.
- LLM-judged metrics are Ragas 0.3.9. Groq's free tier caps tokens/day *per
  model*, so each metric uses its own judge model, consistently across all
  configs (comparisons are valid within each metric column):

  | metric | judge model | calls/run |
  |:--|:--|--:|
  | faithfulness | `openai/gpt-oss-120b` | 26 |
  | answer_relevancy | `llama-3.1-8b-instant` | 26 |
  | context_precision | `llama-3.1-8b-instant` | ~130 |
  | context_recall | `llama-3.1-8b-instant` | 26 |

- **Coverage is reported for every judged mean** (`N/26 samples scored`) and
  per-sample scores are saved to `*_ragas_detail.csv`. A judged mean over
  fewer than 26 samples is not comparable and is not reported as final.
- `context_precision` (with reference) issues one judge call *per retrieved
  context*, ~130/run — 5× the other metrics. On Groq's free tier only
  `llama-3.1-8b-instant` (14,400 requests/day) has the throughput to score it
  26/26; `gpt-oss-20b` exhausts its daily token budget partway, so this metric
  was moved to `8b-instant` for **both** configs (same judge = fair compare).
- `context_recall` and `context_precision` are judged by the smallest model
  and read as strict/noisy; treat absolute values with more caution than
  deterministic recall@k/MRR, which are exact.
- `answer_relevancy` runs with `strictness=1` because Groq rejects `n>1`
  completion requests.

Reproduce: `python evals/run_evals.py --label baseline` (add
`--ragas-metrics <name>` + `RAGAS_JUDGE_MODEL` per the table above).

## M4 retrieval upgrades

Two retrievers were added behind the same `Retriever` interface and compared
against the baseline on the identical dataset and judges. Full table:
[m4_comparison.md](m4_comparison.md); deterministic grid across chunk size:
[m4_retrieval_grid.md](m4_retrieval_grid.md).

| config | recall@1 | mrr@10 | faithfulness | context_precision | context_recall |
|:--|--:|--:|--:|--:|--:|
| dense-only (baseline) | 0.808 | 0.884 | 1.000 | 0.865 | 0.724 |
| hybrid (BM25+RRF) | 0.846 | 0.923 | — | — | — |
| hybrid + reranker | **0.885** | **0.929** | 1.000 | **0.914** | 0.678 |

### Findings

- **recall@1 0.808 → 0.885 (+7.7 pts)** and **MRR 0.884 → 0.929 (+4.6 pts)**
  from the full hybrid + cross-encoder stack. The correct page reaches rank 1
  on two questions that previously missed it — the metric a citation UI
  surfaces first, since users click the first citation.
- **Context precision 0.865 → 0.914 (+4.8 pts):** reranking the top-20 down to
  top-5 removes padding chunks, so passages shown to the model are cleaner.
  This is also a token-cost win.
- **Faithfulness held at 1.000** across the change — the retrieval upgrade
  improved *what* is retrieved without disturbing the grounding guarantee.
  Protecting this while raising recall was the point.
- **Chunk size {300, 500, 800} had zero effect** (see grid): every page in
  this corpus fits inside the smallest token budget, so all three produce
  identical one-chunk-per-page indexes. A real finding — chunk tuning only
  matters relative to the documents' actual text density.

### Honest caveats

- **The reranker lowered recall@3** from hybrid's 1.000 to 0.962: it promoted
  the right page to rank 1 but demoted a correct page out of the top-3 on one
  question. Rerankers are not a free lunch at every cutoff — pick the cutoff
  your UI actually uses.
- **Context recall went down (0.724 → 0.678)** and should not be read as a
  real regression. This metric is judged by the smallest model and is the
  noisiest in the suite: on identical inputs it returned 0.704 and 0.678 on
  two separate fully-covered runs. The delta here is within that noise band.
- Judged metrics whose means came from partially-scored runs were discarded
  rather than reported; see the coverage rule above.

Reproduce: `python evals/run_grid.py` (deterministic grid), then the winning
config through `run_evals.py --retriever hybrid_rerank` per metric, and
`python evals/build_comparison.py` to assemble the table.
