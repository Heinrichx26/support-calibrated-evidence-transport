# External calibration and complete experiment record

This asset is self-contained for Tables S25-S27 and supplies the saved summaries needed to export Tables 1-4 and S1-S30 (34 tables). It also retains every saved exploratory calibration family, the previous confirmation outcomes, all external raw predictions, frozen model selections, fitted calibrators, prospective thresholds, per-query outcomes and all declared pairwise comparisons. Manuscripts and review correspondence are excluded. New allocation and evidence-set analyses are listed in results/external_asset_scope.json when included.

Table S31 is supplied by a later attachment. [REPRODUCE_ALL_TABLES.md](REPRODUCE_ALL_TABLES.md) gives the four-archive procedure for all 35 current tables. The instructions below describe the unchanged external-calibration archive.

## Restore and verify

Use Python 3.11 and install requirements-external.txt. The tested package versions are recorded in results/external_reproduction_environment.json. Restore the pinned provider data before query evaluation or refitting:

```
python src/fetch_multivers_external.py --smoke
python src/fetch_multivers_external.py
python src/reproduce_multivers_external.py --smoke
python src/reproduce_multivers_external.py
python src/export_r2_tables.py --smoke
python src/export_r2_tables.py
python src/plot_multivers_external.py
```

The first reproduction step compares predictions with their frozen values, verifies exact budgets and confirms score invariance to row ordering and outcome columns. The full replay reproduces 36 query-level outputs and all 16 paired intervals. Historical study files remain unchanged. New results are written to results/external_reproduction/. The table export reads saved outcomes and checks every external displayed metric against query-level counts. Figures read the CSV files and can be redrawn independently of inference, fitting and bootstrap computation. Editable SVG and selectable-text PDF diagnostics are included; their Python sources remain in src/.

To reproduce fitting and train-only selection, use the frozen fitting/calibration identifiers:

```
python src/reproduce_multivers_external.py --refit
python src/reproduce_multivers_external.py --selection
```

The first command refits all frozen models and reproduces thresholds. The second also runs all 44 declared candidate forms in every training source fold, comparing the resulting training-fold table with the saved table. Both preserve the original test start marker and do not tune on test outcomes. A passing smoke record is required before either command. Refitting on another numerical platform may require a numerical-tolerance review if fitted tree predictions differ; the supplied frozen predictions and replay avoid refitting variation.

## Data, scores and model source versions

data/multivers_external/provider_source.json pins the immutable upstream processed archive by S3 version identifier, byte size and SHA-256, with a hash for every HealthVer/COVID-Fact member. Source text is obtained from the original provider. The asset supplies numeric candidate-feature matrices without claim or source text, raw full-document and isolated-sentence MultiVerS probabilities, and all saved candidate admission scores. The source corpora retain their original providers' terms; this release does not assign them the MultiVerS code licence.

MultiVerS source is retained under its MIT licence at src/vendor/mv/LICENSE. Its exact upstream commit is a6ce033f0e17ae38c1f102eae1ee4ca213fbbe2e. The fever_sci checkpoint is pinned by S3 version GYe58tFsyvvn1n7rib5H4wrsiq32ZY5Y and SHA-256 6b48f1b83eaedbc2570bc50135518ef3cf6f3c6dcdd380a9a3a433918e7f982d. Its 5,253,768,950-byte weights are obtained from the original host and are omitted from the asset. Tokenizer and encoder revisions are in data/models/multivers/ and external_inference/*model_manifest.json.

Optional neural recomputation uses requirements-neural.txt and the immutable downloader. Use a fresh extracted directory for recomputation so that archived raw predictions remain available for comparison:

```
python src/download_multivers.py --smoke
python src/download_multivers.py
python src/infer_multivers_external.py --prepare
python src/infer_multivers_external.py --context full --smoke
python src/infer_multivers_external.py --context full
python src/infer_multivers_external.py --context single --smoke
python src/infer_multivers_external.py --context single
```

The inference entry point preserves complete existing scores. Move a copied score file aside before recomputing that complete stream. Numeric precision, attention-window handling and official parameter loading are recorded in the inference audits.

## Study interpretation and full comparison record

HealthVer uses official train/dev/test; COVID-Fact uses source-connected components of official train for fitting/calibration and official test for evaluation. The selected single-context comparator is selected over raw and calibrated full-document forms. The arithmetic comparator is selected over weights 0.25, 0.50 and 0.75. Source-connected folds, training exclusions, complete selected form names and prospective threshold rules are recorded in each freeze.json.

The primary metric admits exactly 100 items per 100 test queries, with at most five items per query. The budget is reallocated inside each of 2,000 paired source-component bootstrap samples. Four co-primary comparisons use Bonferroni simultaneous 98.75% intervals. The simultaneous columns for secondary contrasts use the same quantiles for exploration; the declared simultaneous primary family contains the four primary rows. The pointwise gains on both external collections have intervals that include zero against the selected single-context control. All secondary and unsuccessful exploratory comparisons are retained. No post-test candidate tuning is represented as independent confirmation.

Source histories, frozen inputs and output hashes are in results/external_asset_manifest.json. results/table_to_result_index.json maps every current table to its saved source. SHA-256 hashes refer to the exact packaged bytes. Parquet text removal is recorded separately so that original input audits remain identifiable.
