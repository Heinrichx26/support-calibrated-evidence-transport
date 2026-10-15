# Fixed-workload admission allocation: independent additional analysis

The frozen SCET scores, raw cross-encoder scores, and existing SciFact MultiVerS rationale scores were evaluated on the original four held-out query partitions. Each policy admits exactly 100 sentences per 100 queries. The per-query cap is fixed at 1, 2, 3, or 5. Cap one admits one sentence for every query. Higher caps allocate the same batch budget across queries and allow zero output. No scorer is retrained. Every fixed cap is retained.

The finite family and stopping rule are in `data/scet_allocation_protocol.json`. A 12-query-per-dataset smoke passed finite-score, workload, cap, label-perturbation, and row-order checks. The complete run is in `results/scet_allocation/full/`. Existing HealthVer and COVID-Fact evaluation predictions and labels are not inputs to this computation. The original evaluation outcomes were inspected before this sensitivity analysis, so these results are exploratory and cannot serve as independent confirmation of a new method.

## Main findings

Values are arithmetic means over the four original partitions. Complete-set recall uses only queries with at least one annotated evidence sentence. The no-evidence column is the fraction of queries without annotated evidence that receive any sentence. HoVer has no such query in this sample; that metric is undefined.

| Dataset | Policy | Precision | Query coverage | Complete-set recall | No-evidence admission |
|---|---|---:|---:|---:|---:|
| SciFact | SCET, cap 5 | 0.360714 | 0.371429 | 0.383568 | 0.127244 |
| SciFact | Cross-encoder, cap 1 | 0.392857 | 1.000000 | 0.591638 | 1.000000 |
| SciFact | Cross-encoder, cap 5 | 0.392857 | 0.369048 | 0.441865 | 0.084918 |
| SciFact | MultiVerS, cap 1 | 0.442857 | 1.000000 | 0.647622 | 1.000000 |
| SciFact | MultiVerS, cap 5 | 0.610714 | 0.465476 | 0.634917 | 0.061831 |
| FEVER | SCET, cap 5 | 0.584345 | 0.618258 | 0.690552 | 0.334184 |
| FEVER | Cross-encoder, cap 1 | 0.596185 | 1.000000 | 0.809019 | 1.000000 |
| FEVER | Cross-encoder, cap 5 | 0.540431 | 0.636492 | 0.652462 | 0.458513 |
| HoVer | SCET, cap 5 | 0.679375 | 0.555000 | 0.062946 | — |
| HoVer | Cross-encoder, cap 1 | 0.833571 | 1.000000 | 0.000000 | — |
| HoVer | Cross-encoder, cap 5 | 0.671964 | 0.584464 | 0.043393 | — |

The raw cross-encoder cap-one result differs by one selected correct sentence across 840 partition-query instances from the previously reported SciFact top-one result (0.391667). The previous top-one result ranks the fitting empirical-CDF scores, which can tie at the upper endpoint. This sensitivity analysis ranks the raw cross-encoder scores with the same fixed tie keys. Both results must be identified by their score definition.

## Paired uncertainty

The following are pointwise two-sided 95% query-cluster bootstrap intervals, with 1,000 resamples. One weight per query is shared wherever that query recurs across the overlapping partitions. Admission decisions remain fixed while resampling. These are descriptive conditional intervals, not simultaneous superiority tests.

| Comparison | Metric | Difference | 95% interval |
|---|---|---:|---:|
| FEVER SCET cap 5 − cross-encoder cap 5 | Precision | +0.043914 | [0.036475, 0.052010] |
| FEVER SCET cap 5 − cross-encoder cap 5 | Complete-set recall | +0.038091 | [0.029993, 0.046113] |
| FEVER SCET cap 5 − cross-encoder cap 5 | No-evidence admission | −0.124329 | [−0.138083, −0.111048] |
| FEVER SCET cap 5 − cross-encoder cap 1 | Precision | −0.011840 | [−0.020337, −0.003465] |
| FEVER SCET cap 5 − cross-encoder cap 1 | Complete-set recall | −0.118467 | [−0.126242, −0.110640] |
| FEVER SCET cap 5 − cross-encoder cap 1 | No-evidence admission | −0.665816 | [−0.677120, −0.654042] |
| HoVer SCET cap 5 − cross-encoder cap 5 | Precision | +0.007411 | [−0.005662, 0.020945] |
| HoVer SCET cap 5 − cross-encoder cap 5 | Complete-set recall | +0.019554 | [0.014318, 0.025383] |
| HoVer SCET cap 5 − cross-encoder cap 1 | Precision | −0.154196 | [−0.169278, −0.138580] |
| HoVer SCET cap 5 − cross-encoder cap 1 | Complete-set recall | +0.062946 | [0.056026, 0.070578] |
| SciFact SCET cap 5 − cross-encoder cap 5 | Precision | −0.032143 | [−0.075237, 0.007179] |
| SciFact SCET cap 5 − cross-encoder cap 5 | Complete-set recall | −0.058298 | [−0.117247, −0.003576] |
| SciFact SCET cap 5 − cross-encoder cap 5 | No-evidence admission | +0.042327 | [0.004175, 0.084118] |
| SciFact MultiVerS cap 5 − MultiVerS cap 1 | Precision | +0.167857 | [0.099182, 0.240627] |
| SciFact MultiVerS cap 5 − MultiVerS cap 1 | Complete-set recall | −0.012706 | [−0.090737, 0.061869] |
| SciFact MultiVerS cap 5 − MultiVerS cap 1 | No-evidence admission | −0.938169 | [−0.981874, −0.888560] |

All comparisons and all six metrics are available in `full/paired_intervals.csv`, including evidence recall, query coverage, and query error rate.

## Fitting-selected cap sensitivity

Cap selection uses fitting precision only. Ties prefer the smaller cap. The fitting precision is conditional on the original frozen scorer, whose own training used those fitting queries. This exercise is an allocation sensitivity, not nested validation of a newly selected full method.

| Dataset | Selected-cap SCET precision | Selected-cap cross-encoder precision | SCET − cross-encoder, 95% interval |
|---|---:|---:|---:|
| SciFact | 0.394048 | 0.419048 | −0.025000 [−0.064840, 0.016109] |
| FEVER | 0.584649 | 0.596185 | −0.011537 [−0.019515, −0.003889] |
| HoVer | 0.814375 | 0.833571 | −0.019196 [−0.029191, −0.008085] |

Chosen SCET caps for seeds 13/17/29/7 are SciFact 2/3/2/5, FEVER 5/3/5/3, and HoVer 1/1/1/1. Cross-encoder chooses 5/3/2/5 on SciFact and cap one throughout FEVER and HoVer. MultiVerS chooses 3/5/3/5 on SciFact and reaches precision 0.600000. This finite cap search provides no evidence of a consistent SCET precision advantage over a cross-encoder with the same selection opportunity. It should remain a sensitivity analysis, with the original method unchanged.

## Material suitable for the manuscript

With the same cap and budget on FEVER, SCET increases precision and complete-set recall and reduces admission on queries without annotated evidence. Cross-encoder top one obtains higher precision and complete-set recall on FEVER, while issuing a sentence for every query without annotated evidence. On HoVer, complete evidence requires multiple sentences. SCET cap five recovers 6.3% of complete sets, compared with 4.3% for cross-encoder cap five, with precision 0.679 and 0.672. On SciFact, we use full-abstract MultiVerS: batch admission with cap five raises precision from 0.443 under top one to 0.611 and reduces admission on queries without annotated evidence from 1.000 to 0.062. These results quantify the relationship between sentence precision, abstention, and complete evidence retention at a fixed workload.

## Decision

The new analysis is usable for the response about simple controls, workload, abstention, and full-evidence retention. The fitting-selected-cap candidate does not establish an additional general SCET precision increment. No replacement of the frozen primary method is justified by these results. The complete candidate family and failed precision comparisons are retained for reproducibility.
