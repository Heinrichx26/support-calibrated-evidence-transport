# Support-Calibrated Evidence Transport

Current experiments, source code and result tables are in [r2/](r2/README.md).

The [complete numerical reproduction release](https://github.com/Heinrichx26/support-calibrated-evidence-transport/releases/tag/r2-complete-2026-10-15) provides the current code and saved results for Tables 1–4 and S1–S30. It includes frozen HealthVer and COVID-Fact calibration experiments, all declared comparisons and ablations, admission-allocation sensitivity, complete evidence-set stratification, and unsuccessful development trials. Source documents are restored from their pinned original provider; pretrained neural weights are obtained from their provider with complete hashes.

Download `R2_external_calibration_reproducibility.zip` and its SHA-256 file from that release, extract to a new directory, and follow [the reproduction instructions](r2/EXTERNAL_CALIBRATION_REPRODUCTION.md). The asset reproduces all 34 tables from saved outcomes and runs the external frozen-model replay independently of neural inference. The original full scorer matrices for recomputing the original experiments remain in the [original R2 archive](https://github.com/Heinrichx26/support-calibrated-evidence-transport/releases/tag/r2-2026-10-15).

The root `src/`, `results/`, `tests/` and `figures/` retain historical material. Use the `r2/` procedure for the current experiments. Earlier releases and history remain available.
