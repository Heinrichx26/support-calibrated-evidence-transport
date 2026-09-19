from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ipm_evidence_dataset_adapters import RAW_DEFAULT, load_fever_sentence_cache, load_hover_sentence_cache
from ipm_hover_wikipedia_sentence_cache import build_sentence_cache


PROJECT = Path(__file__).resolve().parents[1]
CANDIDATE_DIR = PROJECT / "data" / "interim" / "evidence_retrieval"
DEFAULT_OUT = RAW_DEFAULT / "fever" / "wikipedia_sentence_cache.json"


def collect_fever_candidate_ids(paths: list[Path], limit_rows: int | None = None) -> set[str]:
    ids: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        usecols = ["candidate_id"]
        reader = pd.read_csv(path, usecols=usecols, nrows=limit_rows)
        ids.update(reader["candidate_id"].astype(str).tolist())
    return ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover FEVER Wikipedia sentence text.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--sleep-seconds", type=float, default=0.25)
    parser.add_argument("--limit-rows", type=int, default=None)
    return parser.parse_args()


def build_missing_full_pages(
    candidate_ids: set[str],
    existing: dict[str, str],
    out: Path,
    batch_size: int,
    sleep_seconds: float,
) -> dict[str, str]:
    from ipm_hover_wikipedia_sentence_cache import batched, fetch_extracts, parse_candidate_id, split_wikipedia_sentences

    updated = dict(existing)
    title_to_candidates: dict[str, list[tuple[str, int]]] = {}
    for candidate_id in candidate_ids:
        if candidate_id in updated:
            continue
        parsed = parse_candidate_id(candidate_id)
        if not parsed:
            continue
        title, sentence_id = parsed
        title_to_candidates.setdefault(title, []).append((candidate_id, sentence_id))
    titles = sorted(title_to_candidates)
    print(f"missing_titles={len(titles)} missing_ids={sum(len(v) for v in title_to_candidates.values())}", flush=True)
    for index, title_batch in enumerate(batched(titles, batch_size), start=1):
        extracts = fetch_extracts(title_batch, batch_size=batch_size, sleep_seconds=sleep_seconds, intro_only=False)
        for title, extract in extracts.items():
            sentences = split_wikipedia_sentences(extract)
            for candidate_id, sentence_id in title_to_candidates.get(title, []):
                if candidate_id not in updated and 0 <= sentence_id < len(sentences):
                    clean_title = title.replace("_", " ")
                    updated[candidate_id] = f"{clean_title}. {sentences[sentence_id]}"
        if index % 5 == 0 or index == (len(titles) + batch_size - 1) // max(batch_size, 1):
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(updated, ensure_ascii=False), encoding="utf-8")
            recovered = sum(1 for cid in candidate_ids if cid in updated)
            print(f"batch={index} cache={len(updated)} recovered={recovered}/{len(candidate_ids)}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(updated, ensure_ascii=False), encoding="utf-8")
    return updated


def main() -> None:
    args = parse_args()
    names = ["fever_smoke_candidates_k100.csv"]
    if args.mode == "full":
        names.append("fever_full_candidates_k100.csv")
    paths = [CANDIDATE_DIR / name for name in names]
    candidate_ids = collect_fever_candidate_ids(paths, limit_rows=args.limit_rows)
    existing = load_fever_sentence_cache(args.out.parent)
    hover_cache = load_hover_sentence_cache(RAW_DEFAULT / "hover")
    for key, value in hover_cache.items():
        existing.setdefault(key, value)
    if args.mode == "full":
        updated = build_missing_full_pages(
            candidate_ids,
            existing,
            out=args.out,
            batch_size=max(8, args.batch_size // 2),
            sleep_seconds=args.sleep_seconds,
        )
    else:
        updated = build_sentence_cache(
            candidate_ids,
            existing,
            out=args.out,
            batch_size=args.batch_size,
            sleep_seconds=args.sleep_seconds,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(updated, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    recovered = len(set(candidate_ids) & set(updated))
    placeholders = len(candidate_ids) - recovered
    print(
        json.dumps(
            {
                "requested": len(candidate_ids),
                "recovered": recovered,
                "still_placeholder": placeholders,
                "cache_size": len(updated),
                "out": str(args.out),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
