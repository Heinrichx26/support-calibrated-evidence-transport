# Bounded training-only ranking-objective investigation

The existing smoke passed the numerical derivative, score invariance, source-group separation, and exact-workload checks. Its best new candidate matched the histogram calibration control on the single inspected SciFact fold. A single complete four-fold development run evaluated the predeclared finite family on SciFact development and the original HealthVer and COVID-Fact fitting claims. No transformer was retrained, and no previous external test or reserved labels were used.

The score features are the frozen full-context and isolated-sentence probabilities plus the previously defined context summaries. Full-context-only and dual-context models receive the same objectives and tree capacities. The new objectives compare positive candidates with the highest-scoring 5% or 20% of training negatives through pairwise logistic loss. Binary logistic loss is the matching control. Every evaluated policy admits exactly one sentence per query on average, with at most five per query.

| Development collection | Best existing dual histogram | Matching dual binary | Dual tail 20% | Dual tail 5% |
|---|---:|---:|---:|---:|
| SciFact | 0.562867 | 0.555462 | 0.550908 | 0.493236 |
| HealthVer fitting | 0.522378 | 0.509896 | 0.440692 | 0.394161 |
| COVID-Fact fitting | 0.507209 | 0.497961 | 0.428941 | 0.418271 |

The new tail objectives do not provide a usable improvement. The complete SciFact direction gate failed, and both external fitting collections showed larger losses. The method remains unchanged. The complete per-fold results, method means, and out-of-fold scores are retained in `development/cv.csv`, `development/mean.csv`, and `development/oof_scores.parquet`. No confirmatory or old-test evaluation follows this failed development direction.

The dual-context binary calibrators exceed full-context-only controls on the development collections, which is consistent with the existing contextual calibration design. These development comparisons are exploratory and do not supply new independent evidence for general superiority.
