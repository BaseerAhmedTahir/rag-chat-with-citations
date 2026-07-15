# M4 configuration comparison

| config                |   recall@1 |   recall@3 |   recall@5 |   mrr@10 |   faithfulness |   answer_relevancy |   context_precision |   context_recall |
|:----------------------|-----------:|-----------:|-----------:|---------:|---------------:|-------------------:|--------------------:|-----------------:|
| dense-only (baseline) |     0.8077 |     0.9615 |     0.9615 |   0.8837 |              1 |             0.8809 |              0.8654 |           0.7238 |
| hybrid (BM25+RRF)     |     0.8462 |     1      |     1      |   0.9231 |              — |             —      |              —      |           —      |
| hybrid + reranker     |     0.8846 |     0.9615 |     0.9615 |   0.9295 |              1 |             0.8854 |              0.9135 |           0.6777 |

_Judged metrics run for baseline and the winning config; the intermediate hybrid row shows retrieval metrics only ("—" = not evaluated). Every judged mean scored 26/26 samples (see `*_ragas_detail.csv`)._
