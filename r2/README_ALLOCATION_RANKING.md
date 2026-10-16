# Allocation and sentence-ranking replay

This attachment supplies the frozen 2 x 2 decomposition of sentence rankings
and per-query admitted counts on FEVER and HoVer. It contains the analysis
program, all saved full and smoke outputs, a dependency inventory, and a
per-member SHA-256 manifest. The analysis uses saved models and scores.

## Restore

1. Obtain `R2_public_reproducibility.zip` from:
   https://github.com/Heinrichx26/support-calibrated-evidence-transport/releases/tag/r2-2026-10-15
   Its size is 1,323,845,430 bytes and its SHA-256 is
   `7e50b281ba52c6d729f499514062c991d0fe1774885f67561eedcdfff3ad1762`.
2. Extract that base archive into an empty working directory, for example
   `work/r2/`. Its `src/`, `data/`, and `results/` folders belong together.
3. Verify this attachment against its adjacent `.sha256` file. Extract
   `R2_allocation_ranking.zip` into the same `work/r2/` directory. This ZIP
   already starts with `src/`, `data/`, and `results/`; do not add another
   `r2/` directory while extracting it.

The base archive is the only required earlier package. The published
`R2_external_calibration_reproducibility.zip` and `R2_evidence_archive.zip`
are optional for this analysis. The latter has a `r2/` prefix and belongs
in `work/` if used; it is not named `R2_evidence_completion.zip`.
`data/allocation_ranking_dependencies.json` pins every required base member
by path, byte size, and SHA-256. All required members match the base archive.

## Run from the restored directory

Use Python 3.11:

```text
python -m pip install -r requirements-allocation-ranking.txt
python src/analyze_allocation_ranking.py --smoke
python src/analyze_allocation_ranking.py
```

Run the full command after the smoke audit reports `passed: true`. The smoke
uses 20 FEVER seed-13 queries and preserves their saved full-batch counts.
The full command writes FEVER and HoVer results for all four saved partitions.
No fitting or neural inference is performed. Outputs are written beneath
`results/allocation_ranking/smoke/` and `results/allocation_ranking/full/`.
Copy this restored directory before replay if the original result bytes
should remain available beside regenerated files. Audit elapsed times can vary.

## Read and verify

`full/by_partition.csv` gives the four cells, `full/cell_intervals.csv` gives
their intervals, and `full/decomposition.csv` gives the ranking and count
contributions. The eight query CSV files retain the decisions and denominators.
Intervals condition on the frozen models, ranks, counts, and observed
partitions. The hybrid cells are diagnostic configurations.

`results/allocation_ranking/package_manifest.json` hashes every ZIP member
except itself. The adjacent ZIP checksum verifies the entire attachment.
The dependency inventory identifies the exact earlier inputs required for
replay. Manuscripts, review correspondence, and source checkpoints are excluded.
