# Data Sources

This repository stores derived result summaries and figure files. Raw public datasets are excluded and should be placed under `data/raw_open/evidence_retrieval/` when full experiments are rerun.

## SciFact

- Source: public SciFact release.
- Script source URL: `https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz`
- Expected local folder: `data/raw_open/evidence_retrieval/scifact/`
- Adapter: `src/ipm_evidence_dataset_adapters.py`

The adapter downloads and extracts the public release when the files are missing.

## FEVER

- Source: public FEVER shared-task data.
- Script source URLs:
  - `https://s3-eu-west-1.amazonaws.com/fever.public/train.jsonl`
  - `https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_dev.jsonl`
  - fallback URLs under `https://fever.ai/download/fever/`
- Expected local folder: `data/raw_open/evidence_retrieval/fever/`
- Adapter: `src/ipm_evidence_dataset_adapters.py`

The adapter downloads the public JSONL files when available.

## HoVer

- Source: public HoVer data and HotpotQA-style Wikipedia context files.
- Optional script route: Hugging Face datasets through `datasets.load_dataset`.
- Expected local folder: `data/raw_open/evidence_retrieval/hover/`
- Adapter: `src/ipm_evidence_dataset_adapters.py`

Large context files are not stored in Git. Place public HoVer and context files in the expected folder before running full HoVer experiments.

## BTS Portability Evidence

- Source: U.S. Bureau of Transportation Statistics Airline On-Time Performance data.
- Expected local inputs: derived review-queue summaries generated from public BTS on-time performance records.
- Adapter: `src/ipm_evidence_dataset_adapters.py`
- Portability benchmark: `src/ipm_event_evidence_benchmark.py`

The manuscript uses BTS as portability evidence for structured review queues. The main text evidence experiments are based on SciFact, FEVER, and HoVer.

## Excluded Files

The following files are intentionally excluded from the repository:

- raw downloaded archives and JSONL files;
- large reconstructed candidate pools;
- full SCET detail files;
- neural score caches;
- local logs, environments, and temporary outputs.

These files can be rebuilt from the public sources and scripts above.
