# Evaluation Results

## M3 baseline

Config: `bge-small-en-v1.5` dense retrieval (ChromaDB, cosine), chunk size 500
tokens / 50 overlap, top-k 5, generation by `llama-3.3-70b-versatile` (Groq)
with the citation-grounded prompt. Dataset: 26 hand-labeled examples over 4
documents ([eval_set.json](../dataset/eval_set.json)).

| config   | recall@1 | recall@3 | recall@5 | mrr@10 | faithfulness | answer_relevancy | context_precision | context_recall |
|:---------|---------:|---------:|---------:|-------:|-------------:|-----------------:|------------------:|---------------:|
| baseline |    0.808 |    0.962 |    0.962 | 0.884  |    **1.000** |            0.881 |             0.872 |          0.724 |

All LLM-judged metrics scored 26/26 samples.

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

### Methodology notes

- Retrieval metrics (`recall@k`, `hit@k`, `MRR`) are computed by
  [our own code](../metrics/retrieval.py) at page-level granularity — the same
  granularity a citation exposes — and are covered by pytest.
- LLM-judged metrics are Ragas 0.3.9. Groq's free tier caps tokens/day *per
  model*, so each metric uses its own judge model, consistently across all
  configs (comparisons are valid within each metric column):

  | metric | judge model |
  |:--|:--|
  | faithfulness | `openai/gpt-oss-120b` |
  | answer_relevancy | `llama-3.1-8b-instant` |
  | context_precision | `openai/gpt-oss-20b` |
  | context_recall | `llama-3.1-8b-instant` |

- `context_recall` is judged by the smallest model and reads as strict/noisy;
  treat its absolute value with more caution than its config-to-config deltas.
- `answer_relevancy` runs with `strictness=1` because Groq rejects `n>1`
  completion requests.

Reproduce: `python evals/run_evals.py --label baseline` (add
`--ragas-metrics <name>` + `RAGAS_JUDGE_MODEL` per the table above).
