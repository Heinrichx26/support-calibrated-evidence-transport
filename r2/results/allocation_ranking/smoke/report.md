# Frozen allocation and ranking decomposition

No models, transforms, thresholds, or per-query counts were fitted or selected in this analysis. The four cells cross the saved sentence ranks and saved per-query admitted counts. The cross-encoder uses raw scores and every rank uses the original fixed tie keys.

This smoke uses the first 20 lexicographically sorted FEVER seed-13 evaluation IDs. Counts are inherited from the complete held-out batch. The two count policies can therefore have different sentence totals within this small subset. Full-run cells use an identical total sentence budget.

| Dataset | Seed | Rank | Count allocation | Queries | Sentences | Correct | Complete sets | Precision | Set recall |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| fever | 13 | ce | ce | 20 | 21 | 9 | 9 | 0.42857 | 0.81818 |
| fever | 13 | scet | ce | 20 | 21 | 8 | 8 | 0.38095 | 0.72727 |
| fever | 13 | ce | scet | 20 | 20 | 8 | 8 | 0.40000 | 0.72727 |
| fever | 13 | scet | scet | 20 | 20 | 8 | 8 | 0.40000 | 0.72727 |

All original diagonals reproduce admitted counts, correct counts, complete-set outcomes, coverage, and original evidence counts exactly, for every evaluated query. Sorting and label invariance are checked in the smoke. Every full-run cell retains the original query denominators.

