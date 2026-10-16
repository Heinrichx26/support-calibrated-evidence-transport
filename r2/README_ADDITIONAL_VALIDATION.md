# Additional validation and reproducibility

This supplement contains the complete numerical records, fitted models, protocols and source programs for additional admission controls, context features, selective query risk, refitted conflict ablations, source-component sensitivity and an independent FEVER refit. FEVER denotes Fact Extraction and VERification; CE denotes the cross-encoder scoring view; SCET denotes support-calibrated evidence transport. The dataset names HoVer and SciFact retain their published names.

## Files and input dependencies

Extract `R2_additional_validation.zip` into a new directory. Its lightweight numerical verification uses the contents of this archive alone. Check its adjacent SHA-256 file before extraction. `ADDITIONAL_VALIDATION_MANIFEST.json` provides every member's size and SHA-256, excluding the manifest itself.

For recomputing original raw-score transformations, model fitting or candidate scoring, first extract the following existing releases into another new directory, in this order, and then overlay this archive:

1. `R2_public_reproducibility.zip` from <https://github.com/Heinrichx26/support-calibrated-evidence-transport/releases/tag/r2-2026-10-15>.
2. `R2_external_calibration_reproducibility.zip` from <https://github.com/Heinrichx26/support-calibrated-evidence-transport/releases/tag/r2-complete-2026-10-15>.
3. `R2_allocation_ranking.zip` from the same complete release.
4. `R2_table_S31.zip` from the same complete release.

The four dependency hashes are:

```text
7e50b281ba52c6d729f499514062c991d0fe1774885f67561eedcdfff3ad1762  R2_public_reproducibility.zip
3207a03ddca9f6e27961e27b112d6adc06066e2423e25d1db2455a1d6300f5c3  R2_external_calibration_reproducibility.zip
06738047f5cd1b590ffd1bca5a037de402bd69e6ba5e3c32b67bcbad8248b870  R2_allocation_ranking.zip
80ef0e9bdf5e6394fa5d774bc1699ce20ee91b1f94ac1e8b02bbc1ca8592e284  R2_table_S31.zip
```

Large raw candidate-score matrices remain in these existing releases. The supplement includes the small MultiVerS candidate-score input and FEVER source-component mapping at their existing relative locations. Its programs use `data/scored/{fever,hover,scifact}_full_k100.parquet`, the matching processed query metadata, and the original split/model records under `results/experiments` and `results/strengthening`.

## Lightweight verification

Python 3.11 and `requirements-additional-validation.txt` describe the refit environment. The recorded context and control runs also used Python 3.12, pandas 3.0.3 and NumPy 2.4.4 with the same scikit-learn, SciPy and PyArrow versions. The pinned Python 3.11 environment reproduces the original portfolio refit exactly. Set the numerical-library thread limit to two when executing experimental programs.

From the extracted directory, run the smoke verification first. Review its `passed` field before the complete numerical verification:

```sh
python src/r5_verify_release.py --smoke
python src/r5_verify_release.py --full
```

The verifier checks member hashes; reconstructs per-query summary metrics from saved query records; independently reconstructs context admission decisions and workload thresholds from cached calibration/evaluation scores; recomputes Clopper-Pearson bounds; and checks the independent-refit parameter comparison. Its records are written to `results/r5_strengthening/release_verification`. It performs no model training or pretrained neural inference.

Tables S32-S35 are saved under `results/tables/additional_validation`. Their export program is `src/r5_additional_tables.py`; `results/r5_strengthening/additional_table_manifest.json` records their member hashes. Table export reads saved numerical records. The verifier reconstructs all four tables and checks exact CSV bytes. To write the table files again after the smoke check, run:

```sh
python src/r5_additional_tables.py
```

After overlaying all four earlier archives and this supplement, the updated complete exporter recreates Tables 1-4 and S1-S35, comprising 39 CSV tables:

```sh
python src/export_r2_tables.py --smoke
python src/export_r2_tables.py
```

## Complete retained records

| Location | Contents |
| --- | --- |
| `results/r5_context_gain/development` | Query-fold results for every declared depth/group, frozen depth choices, split identifiers and eight fitted models. |
| `results/r5_context_gain/evaluation` | Prospective and fixed-batch query outcomes, thresholds, summaries and all paired comparisons. |
| `results/r5_context_gain/controls` | Seven workload targets, fixed-budget comparison, both empirical risk grids at all six targets, cached scores, selected rules and every query outcome. |
| `results/r5_context_gain/conditional_risk` | Simultaneously corrected conditional query-error and coverage checks for the two frozen context scorers. |
| `results/r5_context_gain/source_sensitivity` | Source assignments, descriptive results, all 36 source-component comparisons and audits. |
| `results/r5_simple_controls` | Five dataset/scorer configurations, four partitions, all logistic tuning candidates, frozen models, complete workload/risk grids and all query outcomes. |
| `results/r5_risk_conflict/risk` | Descriptive coverage/conditional risk, complete independent-calibration grid, selected rules and evaluation outcomes. |
| `results/r5_risk_conflict/conflict` | Both fully refitted ablations for all three datasets and four partitions, all scoring-form qualities, models, outcomes and intervals. |
| `results/r5_reproduction` | Archive-member hashes, frozen FEVER seed-13 reconstruction, independent refit, parameter comparisons and fresh bootstrap calculations. |

All specified operating points and negative outcomes are retained. Empty-output conditional risk is undefined. The independent calibration records distinguish empirical risk targets from simultaneously corrected conditional guarantees.

## Recomputing the experiments

Keep the released result directory as a reference. Create a fresh computation directory by extracting the four dependency releases, then copy this supplement's `src` directory, `data/r2_protocol.json`, small MultiVerS candidate-score input and FEVER source-component mapping into the same relative paths. Run experimental programs from this fresh directory so their output folders begin empty. Copy no `results/r5_*` folders into it. This preserves the saved reference outcomes and the one-evaluation marker used by the context evaluator.

Every complete computation below requires its smoke stage to pass first. Context feature development has a fitting-only direction check in `results/r5_context_gain/smoke/audit.json`; inspect it before full development.

```sh
# Single-scorer list and top-one controls
python src/r5_simple_controls.py --smoke
python src/r5_simple_controls.py --full
python src/r5_simple_report.py

# Finite context feature development, then one frozen evaluation
python src/r5_context_gain.py --smoke
python src/r5_context_gain.py
python src/r5_context_evaluate.py
python src/r5_context_controls.py --smoke
python src/r5_context_controls.py --full
python src/r5_context_conditional_risk.py --smoke
python src/r5_context_conditional_risk.py

# Risk and independently refitted conflict ablations
python src/r5_risk_query.py --stage descriptive
python src/r5_risk_query.py --stage smoke
python src/r5_risk_query.py --stage full
python src/r5_conflict_ablation.py --stage smoke
python src/r5_conflict_ablation.py --stage full

# Source-component resampling after the context and simple controls
python src/r5_reproduce_context_source.py --smoke
python src/r5_reproduce_context_source.py --full
python src/r5_reproduce_context_features.py
```

For archive-based FEVER seed-13 reconstruction, put the original `R2_public_reproducibility.zip` under `results/` in the fresh computation directory. The frozen reconstruction verifies its hash and restores its required members to `results/r5_reproduction/restored_public_archive`. The independent refit uses that verified restoration:

```sh
python src/r5_reproduce_frozen.py --smoke
python src/r5_reproduce_frozen.py --full
python src/r5_reproduce_refit.py --smoke
python src/r5_reproduce_refit.py --full
```

## Interpretation of the retained comparisons

The context experiment uses a common twelve-view, top-five-union candidate screen for both the all-view model and its CE-feature capacity control. Depth selection uses fitting queries. The all-view model improves admission precision at the two declared 100-item comparisons. Complete evidence-set recall and no-evidence admission quantify the associated allocation pattern. The pure CE-list control retains its own CE candidate order and supplies a distinct comparator. The full workload and risk tables show where these comparisons hold.

The two all-view-minus-capacity-control precision comparisons, prospective and fixed batch, use nominal 95% simultaneous intervals across those two outcomes. Other intervals are pointwise exploratory comparisons. Four outer partitions overlap. Resampling shares weights across every occurrence of a query. Source sensitivity shares weights across source components, conditions on frozen decisions, and leaves the observed fixed-batch allocation unchanged. These are exploratory results on previously studied official development queries.

The conditional-risk experiment jointly corrects query-error upper bounds and coverage lower bounds over its declared scorer/threshold family within each fixed partition. A scorer can pass only when both bounds meet the declared targets. The conflict ablations refit every portfolio and reselect the scoring form; their retained results quantify the contribution of explicit conflict inputs alongside implicit information retained in raw views.

The independent FEVER refit reproduced all model numeric parameters exactly in the recorded Python 3.11 environment. The frozen admission reconstruction reproduced all 25 saved method outcomes for one complete held-out partition. The associated four-partition bootstrap combines newly reconstructed seed-13 outcomes with the other three released partitions. The archive retains the exact executed scope and numerical comparisons.

`PUBLIC_METADATA_CHANGES.json` records the replacement of absolute runtime paths in public metadata. Numerical CSV/Parquet files, serialized fitted models and experiment source programs retain their original bytes.
