# Analyses for the accepted article

This directory contains the analyses added for the accepted article, "Workload-Constrained Evidence Admission for Claim Verification" (Information Processing & Management). They use the frozen models, thresholds and query outcomes of release [r2-complete-2026-10-15](https://github.com/Heinrichx26/support-calibrated-evidence-transport/releases/tag/r2-complete-2026-10-15). No model is refitted and no neural inference is repeated.

## Contents

| Path | Content |
| --- | --- |
| `src/r3_analysis.py` | Equal-budget comparisons, verdict strata, candidate-pool audit, workload frontiers, capacity ceilings and text-availability sensitivity. |
| `src/plot_r3.py` | Figures 2-4 of the article from the saved CSV files. |
| `results/analysis/stratified_*.csv` | Table 3, Figure 2 and Tables S38-S40: equal-budget rows, verdict strata and their decomposition. |
| `results/analysis/frontiers_*.csv` | Figures 3-4 and Table S42: precision, complete-set recall, query output and capacity ceilings across workload targets. |
| `results/analysis/missing_candidates/` | Table S41: complete-text and missing-candidate FEVER groups at equal workload, with paired intervals. |
| `results/analysis/candidate_pool_audit.json` | FEVER candidate-pool counts, ranking ceiling and same-page distractor share reported in Section 6.1. |
| `results/analysis/selected_forms.csv` | Scoring forms selected on fitting queries in each partition. |
| `results/analysis/nei_source_examples.csv` | Top-ranked sentences for sampled NOT ENOUGH INFO claims discussed in Section 8.1. |
| `results/analysis/query/` | Query-level outcomes for every compared rule, dataset and partition seed (13, 17, 29 and 7). |
| `results/manifest.csv` | Size and SHA-256 checksum of every result file. |
| `figures/` | Editable SVG and diagrams.net sources with vector PDF exports. |

## Reproduction

1. Extract the R2 release archives into `../r2` as described in [REPRODUCE_ALL_TABLES.md](../r2/REPRODUCE_ALL_TABLES.md). Another location can be given through the `R2_DIR` environment variable.
2. Install the dependencies in `requirements.txt` (tested with Python 3.12).
3. Run the smoke check, which reproduces cached outcomes for a small query sample and verifies exact-budget selection:

```bash
python r3/src/r3_analysis.py --smoke
```

4. Run the complete analysis and redraw the figures:

```bash
python r3/src/r3_analysis.py
python r3/src/plot_r3.py
```

The figures can be redrawn from the saved CSV files without running the analysis. Source texts and pretrained model weights remain subject to the terms of their original providers.
