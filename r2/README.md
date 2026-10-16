# Current numerical reproduction

Tables 1-4 and S1-S37 contain 41 exported tables. [REPRODUCE_ALL_TABLES.md](REPRODUCE_ALL_TABLES.md) gives the extraction order and hashes of the six published archives, followed by the smoke and complete export commands. All 41 CSVs were independently reconstructed with identical bytes. The repository retains programs and small numerical summaries; detailed query outcomes, numerical candidate matrices and frozen fitted models are in the release attachments.

The [complete release](https://github.com/Heinrichx26/support-calibrated-evidence-transport/releases/tag/r2-complete-2026-10-15) supplies the external calibration archive, the frozen sentence-ordering/admission-count analysis, and the S31 table-export update. The [original R2 release](https://github.com/Heinrichx26/support-calibrated-evidence-transport/releases/tag/r2-2026-10-15) supplies the original full scorer matrices. Each archive retains its original manifest and checksum.

Tables S25-S27 report the two external collections, score controls, ablations and four primary simultaneous precision intervals. Tables S28-S30 report exploratory allocation and complete evidence-set analyses. Table S31 reports the frozen comparison of sentence ordering and per-query admission counts; [README_ALLOCATION_RANKING.md](README_ALLOCATION_RANKING.md) gives its exact dependencies and recomputation commands. Saved unsuccessful trials and the full comparison record remain available.

[EVIDENCE_ANALYSIS_REPRODUCTION.md](EVIDENCE_ANALYSIS_REPRODUCTION.md) describes the separately released source-correspondence, candidate-text availability and frozen-decision contribution analyses, including their source snapshot, extraction layout and smoke checks.

[README_ADDITIONAL_VALIDATION.md](README_ADDITIONAL_VALIDATION.md) describes the additional fitted models, same-capacity cross-encoder controls, single-scorer list calibration, joint conditional-risk and coverage bounds, source-component intervals, fully refitted conflict ablations and independent FEVER seed-13 refit. The complete numerical grids and query outcomes are in `R2_additional_validation.zip`; its lightweight verifier reproduces the four new tables and checks saved decisions and risk bounds.

[README_INDEPENDENT_CONFIRMATION.md](README_INDEPENDENT_CONFIRMATION.md) documents 3,000 new FEVER evaluation claims, 600 independent calibration claims, source-page separation, four prespecified simultaneous comparisons and complete evidence-set allocation. The new archive includes frozen lexical transformations, all models, raw scorer outputs and the complete comparison record for Tables S36–S37. Its numerical verifier reconstructs every candidate decision, query outcome, paired interval and both tables.

The sections below document the original experiments. Use the six-archive extraction procedure for all current table exports.

# R2 evidence admission experiments

This release replaces the R1 experiment procedure for the second revision. All prospective methods use fitting queries to select scoring forms and independent calibration queries to freeze admission thresholds. Candidate labels represent annotated verification-evidence membership. Directional support is evaluated separately on SciFact. Within-query normalization is removed; queries may receive zero admitted items.

The release contains the seven original R2 four-partition experiments and six additional fixed-batch analyses, plus the MultiVerS document-context comparison. The fixed-batch analysis selects exactly 100 items per 100 held-out queries from rank-five eligible candidates. This separate ordering analysis reports equal observed workload; the prospective results retain calibration-fixed thresholds.

## Files and input data

The repository contains code, protocol settings, model manifests, summary tables, frozen fitted admission models and editable figure sources. The accompanying R2_public_reproducibility.zip asset supplies verified text matrices, raw scorer matrices, all query-level outcomes, shared stress subsets, original SciFact abstracts, MultiVerS document predictions and a SHA-256 manifest. Extract the asset in its own directory and run the commands below from that directory. Each table and figure is mapped in results/table_to_result_index.json.

Python 3.11 with numpy, pandas, pyarrow, scipy, scikit-learn and matplotlib reproduces the admission analyses from the supplied raw scores. torch, transformers, sentence-transformers and huggingface-hub are used for neural scoring. Use the exact provider revisions in data/scored/*.json. Source corpora and checkpoints remain available from their original providers: https://github.com/allenai/scifact ; https://fever.ai/dataset/fever.html ; https://github.com/hover-nlp/hover ; https://hotpotqa.github.io/ . Candidate identifiers, source text hashes, unresolved counts and provenance appear in the retained inputs and audits.

Install the tested admission dependencies with `python -m pip install -r requirements-admission.txt`. Optional raw neural scoring uses `requirements-neural.txt`; select the torch wheel appropriate to the host. The actual package versions, including the torch provider build, are recorded in results/reproduction_environment.json. The scoring program resolves the five immutable checkpoint revisions directly and downloads that exact revision when it is absent from the provider cache.

## Run and inspect

Generate every manuscript and supplementary table directly from the saved results:

    python src/export_r2_tables.py --smoke
    python src/export_r2_tables.py

The 41 UTF-8 CSV tables and their hashes are written to results/tables/. Every header and displayed cell has been checked against the corresponding Word table. This export requires no manuscript files and performs no experimental scoring.

Run a smoke check before a full experiment:

    python src/run_r2_experiments.py --mode smoke --datasets scifact --pool 100
    python src/run_r2_experiments.py --mode full --datasets scifact fever hover --pool 100
    python src/run_r2_experiments.py --mode full --datasets scifact fever hover --pool 500
    python src/run_r2_experiments.py --mode full --datasets scifact --directional
    python src/strengthen_r2_analysis.py --smoke
    python src/strengthen_r2_analysis.py

The existing full inputs and outputs support direct inspection and threshold evaluation. Exact-budget results use raw single-view scores. Fitting-selected individual scorers and SCET forms are frozen before held-out measurement. The bootstrap resamples query clusters with shared weights across overlapping partitions. Reported intervals are conditional on the frozen fitted rules and observed partitions.

MultiVerS uses official MIT-licensed commit a6ce033f0e17ae38c1f102eae1ee4ca213fbbe2e and the immutable fever_sci checkpoint version in data/models/multivers/checkpoint_manifest.json. It was trained on FEVER, PubMedQA and Evidence Inference. Input is the claim, title and full abstract. Sentence admission score is rationale probability times one minus the document neutral probability. Strictly loaded parameters, tokenizer files, full document probabilities and all four independent thresholds are recorded. The checkpoint is obtained from its provider with the download script; its 5 GB weights are excluded from the asset.

    python src/run_multivers_r2.py --evaluate-only
    python src/download_multivers.py --smoke
    python src/download_multivers.py
    python src/run_multivers_r2.py --smoke
    python src/run_multivers_r2.py

The supplied scores allow the MultiVerS comparison to run without loading the neural checkpoint. The original full abstract corpus is in data/raw/scifact/corpus.jsonl.

## Figures and risk diagnostics

    python src/make_r2_figures.py --plots-only

Figures read saved summaries. Editing or rendering the workflow uses diagrams.net and Inkscape; set DRAWIO_BINARY, INKSCAPE_BINARY and PDFLATEX_BINARY when those executables are outside PATH. Editable drawio, SVG, PDF_TEX and selectable-text PDF files accompany the release.

The risk event is any unannotated admitted item within a query. Learn-then-Test tests 20 fixed thresholds at query-risk target 0.20 and family-wise failure probability 0.05 under independent, identically distributed calibration/future queries. Empty-output feasibility and held-out coverage are reported separately. Simultaneous Clopper–Pearson limits accompany the selected calibration rule. Brier and ten-bin ECE evaluate a separate logistic probability map against calibration-prevalence constant probabilities.

The structured BTS check reports protocol-specific evaluation rows and queue entries with precision changes. Counts may overlap across protocols. The main empirical comparisons use the text-evidence collections.

The package contains public experiment material. Manuscripts, author response letters and peer-review correspondence remain in the author workspace.
