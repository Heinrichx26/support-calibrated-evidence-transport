# Evidence-source and contribution analyses

The source-correspondence audit, FEVER text-availability sensitivity and frozen-decision contribution analysis are supplied by `R2_evidence_archive.zip` in the [complete release](https://github.com/Heinrichx26/support-calibrated-evidence-transport/releases/tag/r2-complete-2026-10-15).

Archive SHA-256: `d27bb46d40dd5b88b17f7aaf840addfef1e01fb19d779f51a8147f4c41e51d70`.

Create an empty `work/` directory and a `work/r2/` subdirectory. Extract the original `R2_public_reproducibility.zip` and then `R2_external_calibration_reproducibility.zip` into `work/r2/`. Their release links and full hashes are in [REPRODUCE_ALL_TABLES.md](REPRODUCE_ALL_TABLES.md). Extract `R2_evidence_archive.zip` into `work/`, retaining its project-level `data/` and `r2/` prefixes. The included `README_R2_EVIDENCE.md` describes the required source snapshot and all dependency hashes.

From `work/r2/`, install and run each smoke check before its complete analysis:

```sh
python -m pip install -r requirements-evidence.txt
python src/verify_evidence_correspondence.py --smoke
python src/verify_evidence_correspondence.py --full
python src/analyze_fever_missingness.py --mode smoke
python src/analyze_fever_missingness.py --mode full
python src/analyze_scet_contribution.py --smoke
python src/analyze_scet_contribution.py
```

The correspondence analysis reads the original FEVER Wikipedia snapshot at `data/raw/fever/wiki-pages.zip`. Obtain its 1,713,485,474 bytes from [the pinned source](https://zenodo.org/api/files/91ad9fca-c15e-4069-80f9-bac6ec5c2d9b/wiki-pages.zip) and verify SHA-256 `4b06d95da6adf7fe02d2796176c670dacccb21348da89cba4c50676ab99665f2`. The two other analyses use the supplied saved inputs.

The repository retains the three programs, analysis plans, audit records and small numerical summaries under `results/evidence_completion/`. Detailed candidate and query records remain in the ZIP. Source correspondence and missingness use 1,000 paired resamples; the frozen-decision contribution analysis uses 2,000 paired resamples. Each analysis record specifies its denominators and whether decisions or batch budgets are fixed. The later S31 analysis uses its own 1,000-draw procedure described in [README_ALLOCATION_RANKING.md](README_ALLOCATION_RANKING.md).

Member manifests describe the original released ZIP bytes and include files stored only in those attachments. The repository table map links Tables 1 and 2 to `results/dataset_facts.json` and query metadata. Displayed CSV hashes are in `results/tables/index.json`.
