# Independent FEVER confirmation and evidence-set allocation

This archive contains the frozen experimental inputs, source programs, fitted models and complete numerical outcomes for a new-query FEVER confirmation. FEVER denotes Fact Extraction and VERification. CE denotes the cross-encoder scoring view. SCET denotes support-calibrated evidence transport.

The archive supports numerical reconstruction without additional downloads. Pretrained neural parameters and reproducible intermediate embeddings are omitted. The adjacent SHA-256 file checks the archive; `INDEPENDENT_CONFIRMATION_MANIFEST.json` records the size and SHA-256 of every payload file.

## Sample and scope

The sample contains 60 smoke, 600 calibration and 3,000 evaluation claims from the official FEVER training partition, with equal numbers of support, refutation and insufficient-information claims in each split. Claim identifiers and normalized text differ from the previously studied development claims. Every annotated evidence page is absent from all earlier development candidate pages. Connected annotated-page components are assigned to one split before deterministic sampling. Evidence-free claims form separate components.

Candidate construction retains the original lexical retrieval and annotated-evidence augmentation protocol, followed by restoration from the June 2017 Wikipedia snapshot and exclusion of unavailable candidate text. Recall denominators retain every original annotated evidence set. This evaluates admission within the constructed candidate lists. Candidate pages unrelated to the annotation may overlap the earlier candidate corpus. Availability subsets, verdict subsets and all four frozen fitting partitions are retained.

The 600 calibration claims determine score thresholds at 100 sentences per 100 queries without using calibration labels. Evaluation predictions, thresholds and decisions are saved before evaluation annotations are joined. Fixed-batch allocation uses prediction values and fixed hash keys at the same budget. Both protocols allow zero output for a query and at most five sentences per query.

The four primary contrasts are: all-view minus CE-feature list-model precision under prospective and fixed-batch allocation; all-view minus CE-feature evidence-set-model precision under fixed-batch allocation; and evidence-set-model minus list-model multi-sentence complete-set recall under fixed-batch allocation. The 2,000 paired annotated-page-component bootstrap draws share query weights across all models. Verdict-stratum normalization preserves equal stratum totals. Bonferroni intervals cover these four primary contrasts jointly at nominal 95%; other intervals are pointwise. Models and admission decisions remain fixed during resampling.

## Files

| Location | Contents |
|---|---|
| `data/independent_confirmation` | Frozen sample plan, original identifiers, source sentences, separate labels and complete evidence sets, text exclusions and frozen lexical state. |
| `data/independent_confirmation/scores` | Twelve raw scoring views, frozen model predictions and model/version metadata for all three splits. |
| `data/raw_open/evidence_retrieval/fever` | Official train/development annotations and the original candidate-pool sentence cache. |
| `results/r5_context_gain/development` | Eight original fitted list models, fitting splits and selection records. |
| `results/evidence_set_confirmation` | Both finite evidence-set experiments, all fitting folds and weights, the unsuccessful initial smoke, frozen selected models and hand-built allocation checks. |
| `results/independent_confirmation/evaluation_full` | All query outcomes, thresholds, candidate predictions, four primary intervals, every secondary comparison, verdict results and text-availability subsets. |
| `src` | Input preparation, frozen scoring, model fitting, evaluation and reconstruction programs. |

Tables S36 and S37 are in `results/tables/independent_confirmation`. Their source program reads the saved numerical results directly:

```sh
python src/export_independent_confirmation_tables.py
```

`results/tables/index.json` lists all 41 tables, including the 39 tables supplied by earlier releases. After overlaying the five archives documented in `README_ADDITIONAL_VALIDATION.md` and this archive, the updated `src/export_r2_tables.py` recreates Tables 1–4 and S1–S37. The two new table exports require only this archive.

Only the two data-preparation programs receive a portable root-directory change. Numerical tables, serialized models and scoring/evaluation programs retain their executed bytes. `PUBLIC_METADATA_CHANGES.json` records every public-source or metadata adjustment. The root directories `data`, `src` and `results` keep the same roles throughout.

## Numerical reconstruction

Use Python 3.12 and install the recorded numerical environment:

```sh
python -m pip install -r requirements-independent-confirmation.txt
python src/verify_independent_confirmation_release.py --smoke
python src/verify_independent_confirmation_release.py --full
```

Run the complete verifier after the smoke record reports `passed: true`. It checks all archive-member hashes, reconstructs every frozen-model candidate prediction, recomputes calibration and batch thresholds, joins candidate labels by query/candidate identifier, reconstructs all query statistics and summary values, and regenerates the paired intervals. It performs no neural inference or model fitting. Verification outputs are written to `results/independent_confirmation/release_verification`.

To rerun the evaluation in an empty computation directory while preserving the reference outcomes:

```sh
python src/confirmation_workspace.py --output ../confirmation-numeric --cached-scores
cd ../confirmation-numeric
python src/independent_confirmation_evaluate.py --smoke
python src/independent_confirmation_evaluate.py --full
```

The workspace command requires a new destination directory. The evaluator requires its technical smoke checks before the complete run and preserves its one-evaluation marker.

## Recomputing raw neural scores

The recorded neural environment used PyTorch 2.12.0, Transformers 5.11.0, Sentence Transformers 5.5.1 and Hugging Face Hub 1.18.0. Install the appropriate PyTorch backend for the computation environment, then install `requirements-independent-confirmation-neural.txt`. Exact checkpoint revisions, pair direction and truncation length appear in the scoring source and per-model metadata. The three embedding models, cross-encoder and natural language inference model are downloaded at their fixed revisions.

Create another empty workspace without cached scores:

```sh
python src/confirmation_workspace.py --output ../confirmation-neural
cd ../confirmation-neural
python -m pip install -r requirements-independent-confirmation-neural.txt
python src/independent_confirmation_score.py --stage smoke --input data/independent_confirmation/smoke_inputs.parquet
python src/independent_confirmation_score.py --stage calibration --input data/independent_confirmation/calibration_inputs.parquet
python src/independent_confirmation_evaluate.py --smoke
python src/independent_confirmation_score.py --stage evaluation --input data/independent_confirmation/evaluation_inputs.parquet
python src/independent_confirmation_evaluate.py --full
```

The frozen lexical state supplies the original count vocabulary, inverse document frequencies, text normalization and truncated singular value decomposition. New claims and candidates are transformed with this state. Rebuilding the state from the original candidate texts is supported by `reference_lexical` in the scoring program; it verifies the reconstructed scores against the original scored data before saving the state.

## Refitting evidence-set allocation

The new allocation preserves the original list-model sentence order and learns the smallest prefix of one to five sentences that contains a complete annotated evidence set. Queries with no evidence, a minimum singleton set and a minimum multi-sentence set receive equal total training weight. A finite set of complete-set utility weights, 0, 1, 4 and 16, is evaluated using only fitting-query source-component folds. The selected weights are 4 for all views and 0 for CE features. Sentence selection uses the marginal slopes of the least concave majorant of the predicted prefix utility. This provides monotone prefix scores; the original nonconcave utility need not have the same optimum.

For independent refitting, extract `R2_public_reproducibility.zip` into a separate directory, here called `original-r2`. This existing release supplies the original scored fitting data, query metadata and split records:

<https://github.com/Heinrichx26/support-calibrated-evidence-transport/releases/tag/r2-2026-10-15>

```text
7e50b281ba52c6d729f499514062c991d0fe1774885f67561eedcdfff3ad1762  R2_public_reproducibility.zip
```

From this supplement directory, create a fresh fitting workspace:

```sh
python src/confirmation_workspace.py --output ../confirmation-fit --training-base ../original-r2
cd ../confirmation-fit
python src/evidence_set_confirmation.py --smoke
python src/evidence_set_balanced_confirmation.py --smoke
python src/evidence_set_balanced_confirmation.py --development
python src/evidence_set_freeze.py
```

The initial unweighted smoke records its unsuccessful multi-sentence-recall direction and has no complete-run stage. The balanced program has its own finite protocol and smoke gate. Its complete run evaluates all four original fitting partitions and three source-component folds, retaining every declared weight and control. The freeze program trains the final query models using fitting queries only and checks label/order invariance. No new confirmation outcomes select these models or weights.

## Rebuilding candidate inputs

`prepare_independent_confirmation.py` records the complete sampling and candidate-construction algorithm. For reconstruction from the original annotations, provide the original candidate reference `data/processed/fever_full_k100.parquet` from `R2_public_reproducibility.zip`. The official annotation files and original sentence-pool cache are included. The provided source-sentence subset reproduces the available new-query texts. To verify missing texts independently against the entire snapshot, place the official June 2017 FEVER Wikipedia archive at `data/raw/fever/wiki-pages.zip`. It is available from the FEVER dataset downloads at <https://fever.ai/dataset/fever.html>.

From a fresh directory containing these inputs and the supplement source/data files, run:

```sh
python src/prepare_independent_confirmation.py --mode plan
python src/prepare_independent_confirmation.py --mode smoke
python src/prepare_independent_confirmation.py --mode full
```

Keep the released sample plan and reference outcomes in their original directory. A fresh reconstruction can regenerate the deterministic plan from the supplied annotations and original candidate reference. Dataset names, annotations and sentence texts retain their respective dataset provenance.
