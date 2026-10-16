# Reproduce Tables 1-4 and S1-S31

Use Python 3.11. The four archives below contain the saved numerical inputs and programs used for all 35 tables. Extract them into one new directory in the listed order, allowing the later archives to replace earlier files.

1. `R2_public_reproducibility.zip` from the [original R2 release](https://github.com/Heinrichx26/support-calibrated-evidence-transport/releases/tag/r2-2026-10-15).
2. `R2_external_calibration_reproducibility.zip` from the [complete release](https://github.com/Heinrichx26/support-calibrated-evidence-transport/releases/tag/r2-complete-2026-10-15).
3. `R2_allocation_ranking.zip` from the same complete release.
4. `R2_table_S31.zip` from the same complete release.

Verify each archive's SHA-256 before extraction:

    7e50b281ba52c6d729f499514062c991d0fe1774885f67561eedcdfff3ad1762  R2_public_reproducibility.zip
    3207a03ddca9f6e27961e27b112d6adc06066e2423e25d1db2455a1d6300f5c3  R2_external_calibration_reproducibility.zip
    06738047f5cd1b590ffd1bca5a037de402bd69e6ba5e3c32b67bcbad8248b870  R2_allocation_ranking.zip
    80ef0e9bdf5e6394fa5d774bc1699ce20ee91b1f94ac1e8b02bbc1ca8592e284  R2_table_S31.zip

From the extracted directory, install the table-export dependencies and perform the smoke export before exporting every table:

```sh
python -m pip install -r requirements-external.txt
python src/export_r2_tables.py --smoke
python src/export_r2_tables.py
```

The complete export writes 35 CSV tables and `results/tables/index.json`. Table S31 is at `results/tables/allocation_ranking/S31.csv`. The index supplies a SHA-256 for each CSV. The repository's [verification record](results/table_replay_verification.json) documents an independent restoration of these four archives: all 35 reconstructed CSVs matched the retained table bytes exactly. Table export reads saved numerical outcomes, including saved uncertainty estimates.

To regenerate S31 alone from the saved decomposition:

```sh
python src/export_r2_tables.py --s31-only
```

[README_ALLOCATION_RANKING.md](README_ALLOCATION_RANKING.md) gives the smoke and complete commands for recomputing the frozen sentence-ordering and admission-count comparison, with 1,000 paired query-cluster resamples and its exact input dependency inventory. Its full query outcomes remain in `R2_allocation_ranking.zip`. [EXTERNAL_CALIBRATION_REPRODUCTION.md](EXTERNAL_CALIBRATION_REPRODUCTION.md) gives the external prediction replay, refitting and neural-scoring commands. [README.md](README.md) gives the original experiment commands. Each procedure starts with its smoke check.

Archive member manifests describe their complete released ZIPs. The repository includes the small summaries and programs; detailed query records remain in the attachments. The separate source and contribution audits are described in [EVIDENCE_ANALYSIS_REPRODUCTION.md](EVIDENCE_ANALYSIS_REPRODUCTION.md).
