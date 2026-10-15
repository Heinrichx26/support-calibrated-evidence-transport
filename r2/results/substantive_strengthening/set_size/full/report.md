# Evidence-set-size analysis

All methods admit exactly 100 sentences per 100 held-out claims. Groups use the smallest original nonempty annotated evidence set. Complete-set recall retains every originally annotated claim in its overall denominator. Counts below are per-partition means (sample standard deviations); partitions overlap.

| Dataset | Method | Precision | Overall complete-set recall | Singleton-set recall | Multi-sentence-set recall | Multi-sentence complete claims |
|---|---|---:|---:|---:|---:|---:|
| scifact | scet_batch | 0.361 | 0.384 | 0.399 | 0.098 | 0.5 (0.6) |
| scifact | ce_batch | 0.393 | 0.442 | 0.460 | 0.098 | 0.5 (0.6) |
| scifact | ce_top1 | 0.393 | 0.592 | 0.621 | 0.000 | 0.0 (0.0) |
| scifact | scet_top1 | 0.386 | 0.560 | 0.588 | 0.000 | 0.0 (0.0) |
| fever | scet_batch | 0.584 | 0.691 | 0.755 | 0.046 | 39.2 (3.3) |
| fever | ce_batch | 0.540 | 0.652 | 0.710 | 0.075 | 63.5 (2.6) |
| fever | ce_top1 | 0.596 | 0.809 | 0.890 | 0.000 | 0.0 (0.0) |
| fever | scet_top1 | 0.570 | 0.775 | 0.853 | 0.000 | 0.0 (0.0) |
| hover | scet_batch | 0.679 | 0.063 | nan | 0.063 | 176.2 (34.3) |
| hover | ce_batch | 0.672 | 0.043 | nan | 0.043 | 121.5 (5.7) |
| hover | ce_top1 | 0.834 | 0.000 | nan | 0.000 | 0.0 (0.0) |
| hover | scet_top1 | 0.814 | 0.000 | nan | 0.000 | 0.0 (0.0) |

## HoVer exact counts

| Seed | Method | All annotated claims | Multi-sentence claims | Complete multi-sentence claims | Overall denominator recall | Conditional multi-sentence recall |
|---|---|---:|---:|---:|---:|---:|
| 13 | scet_batch | 2800 | 2800 | 195 | 0.06964 | 0.06964 |
| 13 | ce_batch | 2800 | 2800 | 124 | 0.04429 | 0.04429 |
| 13 | ce_top1 | 2800 | 2800 | 0 | 0.00000 | 0.00000 |
| 13 | scet_top1 | 2800 | 2800 | 0 | 0.00000 | 0.00000 |
| 17 | scet_batch | 2800 | 2800 | 125 | 0.04464 | 0.04464 |
| 17 | ce_batch | 2800 | 2800 | 125 | 0.04464 | 0.04464 |
| 17 | ce_top1 | 2800 | 2800 | 0 | 0.00000 | 0.00000 |
| 17 | scet_top1 | 2800 | 2800 | 0 | 0.00000 | 0.00000 |
| 29 | scet_batch | 2800 | 2800 | 196 | 0.07000 | 0.07000 |
| 29 | ce_batch | 2800 | 2800 | 113 | 0.04036 | 0.04036 |
| 29 | ce_top1 | 2800 | 2800 | 0 | 0.00000 | 0.00000 |
| 29 | scet_top1 | 2800 | 2800 | 0 | 0.00000 | 0.00000 |
| 7 | scet_batch | 2800 | 2800 | 189 | 0.06750 | 0.06750 |
| 7 | ce_batch | 2800 | 2800 | 124 | 0.04429 | 0.04429 |
| 7 | ce_top1 | 2800 | 2800 | 0 | 0.00000 | 0.00000 |
| 7 | scet_top1 | 2800 | 2800 | 0 | 0.00000 | 0.00000 |
